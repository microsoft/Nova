//! This library implements core components of Nova.
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![deny(missing_docs)]

mod commitments;

pub mod bellperson;
mod circuit;
pub mod errors;
mod gadgets;
pub mod pasta;
mod poseidon;
pub mod r1cs;
pub mod traits;

use std::marker::PhantomData;

use commitments::{AppendToTranscriptTrait, CompressedCommitment};
use errors::NovaError;
use merlin::Transcript;
use r1cs::{
  R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
};
use traits::{ChallengeTrait, Group};

/// A SNARK that holds the proof of a step of an incremental computation
pub struct StepSNARK<G: Group> {
  comm_T: CompressedCommitment<G::CompressedGroupElement>,
  _p: PhantomData<G>,
}

impl<G: Group> StepSNARK<G> {
  fn protocol_name() -> &'static [u8] {
    b"NovaStepSNARK"
  }

  /// Takes as input a Relaxed R1CS instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with the same structure `shape`
  /// and defined with respect to the same `gens`, and outputs
  /// a folded Relaxed R1CS instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  pub fn prove(
    gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    U1: &RelaxedR1CSInstance<G>,
    W1: &RelaxedR1CSWitness<G>,
    U2: &R1CSInstance<G>,
    W2: &R1CSWitness<G>,
    transcript: &mut Transcript,
  ) -> Result<
    (
      StepSNARK<G>,
      (RelaxedR1CSInstance<G>, RelaxedR1CSWitness<G>),
    ),
    NovaError,
  > {
    // append the protocol name to the transcript
    transcript.append_message(b"protocol-name", StepSNARK::<G>::protocol_name());

    // compute a commitment to the cross-term
    let (T, comm_T) = S.commit_T(gens, U1, W1, U2, W2)?;

    // append `comm_T` to the transcript and obtain a challenge
    comm_T.append_to_transcript(b"comm_T", transcript);

    // compute a challenge from the transcript
    let r = G::Scalar::challenge(b"r", transcript);

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &comm_T, &r)?;

    // fold the witness using `r` and `T`
    let W = W1.fold(W2, &T, &r)?;

    // return the folded instance and witness
    Ok((
      StepSNARK {
        comm_T,
        _p: Default::default(),
      },
      (U, W),
    ))
  }

  /// Takes as input a relaxed R1CS instance `U1` and and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  pub fn verify(
    &self,
    U1: &RelaxedR1CSInstance<G>,
    U2: &R1CSInstance<G>,
    transcript: &mut Transcript,
  ) -> Result<RelaxedR1CSInstance<G>, NovaError> {
    // append the protocol name to the transcript
    transcript.append_message(b"protocol-name", StepSNARK::<G>::protocol_name());

    // append `comm_T` to the transcript and obtain a challenge
    self.comm_T.append_to_transcript(b"comm_T", transcript);

    // compute a challenge from the transcript
    let r = G::Scalar::challenge(b"r", transcript);

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &self.comm_T, &r)?;

    // return the folded instance
    Ok(U)
  }
}

/// A SNARK that holds the proof of the final step of an incremental computation
pub struct FinalSNARK<G: Group> {
  W: RelaxedR1CSWitness<G>,
}

impl<G: Group> FinalSNARK<G> {
  /// Produces a proof of a instance given its satisfying witness `W`.
  pub fn prove(W: &RelaxedR1CSWitness<G>) -> Result<FinalSNARK<G>, NovaError> {
    Ok(FinalSNARK { W: W.clone() })
  }

  /// Verifies the proof of a folded instance `U` given its shape `S` public parameters `gens`
  pub fn verify(
    &self,
    gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    U: &RelaxedR1CSInstance<G>,
  ) -> Result<(), NovaError> {
    // check that the witness is a valid witness to the folded instance `U`
    S.is_sat_relaxed(gens, U, &self.W)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::{Field, PrimeField};
  use rand::rngs::OsRng;

  type S = pasta_curves::pallas::Scalar;
  type G = pasta_curves::pallas::Point;

  fn synthesize_tiny_r1cs_bellperson<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    x_val: Option<Scalar>,
  ) -> Result<(), SynthesisError> {
    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(x_val.unwrap()))?;
    let _ = x.inputize(cs.namespace(|| "x is input"));

    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + Scalar::from(5u64))
    })?;
    let _ = y.inputize(cs.namespace(|| "y is output"));

    cs.enforce(
      || "y = x^3 + x + 5",
      |lc| {
        lc + x_cu.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    Ok(())
  }

  #[test]
  fn test_tiny_r1cs_bellperson() {
    use super::bellperson::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
    };

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_tiny_r1cs_bellperson(&mut cs, None);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    // Now get the instance and assignment for one instance
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = synthesize_tiny_r1cs_bellperson(&mut cs, Some(S::from(5)));
    let (U1, W1) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that the first instance is satisfiable
    assert!(shape.is_sat(&gens, &U1, &W1).is_ok());

    // Now get the instance and assignment for second instance
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = synthesize_tiny_r1cs_bellperson(&mut cs, Some(S::from(135)));
    let (U2, W2) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that the second instance is satisfiable
    assert!(shape.is_sat(&gens, &U2, &W2).is_ok());

    // execute a sequence of folds
    execute_sequence(&gens, &shape, &U1, &W1, &U2, &W2);
  }

  fn execute_sequence(
    gens: &R1CSGens<G>,
    shape: &R1CSShape<G>,
    U1: &R1CSInstance<G>,
    W1: &R1CSWitness<G>,
    U2: &R1CSInstance<G>,
    W2: &R1CSWitness<G>,
  ) {
    // produce a default running instance
    let mut r_W = RelaxedR1CSWitness::default(shape);
    let mut r_U = RelaxedR1CSInstance::default(gens, shape);

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let mut prover_transcript = Transcript::new(b"StepSNARKExample");
    let res = StepSNARK::prove(gens, shape, &r_U, &r_W, U1, W1, &mut prover_transcript);
    assert!(res.is_ok());
    let (step_snark, (_U, W)) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let mut verifier_transcript = Transcript::new(b"StepSNARKExample");
    let res = step_snark.verify(&r_U, U1, &mut verifier_transcript);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // produce a step SNARK with (W2, U2) as the second incoming witness-instance pair
    let res = StepSNARK::prove(gens, shape, &r_U, &r_W, U2, W2, &mut prover_transcript);
    assert!(res.is_ok());
    let (step_snark, (_U, W)) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let res = step_snark.verify(&r_U, U2, &mut verifier_transcript);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // produce a final SNARK
    let res = FinalSNARK::prove(&r_W);
    assert!(res.is_ok());
    let final_snark = res.unwrap();
    // verify the final SNARK
    let res = final_snark.verify(gens, shape, &r_U);
    assert!(res.is_ok());
  }

  #[test]
  fn test_tiny_r1cs() {
    let one = S::one();
    let (num_cons, num_vars, num_io, A, B, C) = {
      let num_cons = 4;
      let num_vars = 4;
      let num_io = 2;

      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      // The R1CS for this problem consists of the following constraints:
      // `I0 * I0 - Z0 = 0`
      // `Z0 * I0 - Z1 = 0`
      // `(Z1 + I0) * 1 - Z2 = 0`
      // `(Z2 + 5) * 1 - I1 = 0`

      // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
      // constraint and a column for every entry in z = (vars, u, inputs)
      // An R1CS instance is satisfiable iff:
      // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
      let mut A: Vec<(usize, usize, S)> = Vec::new();
      let mut B: Vec<(usize, usize, S)> = Vec::new();
      let mut C: Vec<(usize, usize, S)> = Vec::new();

      // constraint 0 entries in (A,B,C)
      // `I0 * I0 - Z0 = 0`
      A.push((0, num_vars + 1, one));
      B.push((0, num_vars + 1, one));
      C.push((0, 0, one));

      // constraint 1 entries in (A,B,C)
      // `Z0 * I0 - Z1 = 0`
      A.push((1, 0, one));
      B.push((1, num_vars + 1, one));
      C.push((1, 1, one));

      // constraint 2 entries in (A,B,C)
      // `(Z1 + I0) * 1 - Z2 = 0`
      A.push((2, 1, one));
      A.push((2, num_vars + 1, one));
      B.push((2, num_vars, one));
      C.push((2, 2, one));

      // constraint 3 entries in (A,B,C)
      // `(Z2 + 5) * 1 - I1 = 0`
      A.push((3, 2, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 2, one));

      (num_cons, num_vars, num_io, A, B, C)
    };

    // create a shape object
    let S = {
      let res = R1CSShape::new(num_cons, num_vars, num_io, &A, &B, &C);
      assert!(res.is_ok());
      res.unwrap()
    };

    // generate generators
    let gens = R1CSGens::new(num_cons, num_vars);

    let rand_inst_witness_generator =
      |gens: &R1CSGens<G>, I: &S| -> (S, R1CSInstance<G>, R1CSWitness<G>) {
        let i0 = *I;

        // compute a satisfying (vars, X) tuple
        let (O, vars, X) = {
          let z0 = i0 * i0; // constraint 0
          let z1 = i0 * z0; // constraint 1
          let z2 = z1 + i0; // constraint 2
          let i1 = z2 + one + one + one + one + one; // constraint 3

          // store the witness and IO for the instance
          let W = vec![z0, z1, z2, S::zero()];
          let X = vec![i0, i1];
          (i1, W, X)
        };

        let W = {
          let res = R1CSWitness::new(&S, &vars);
          assert!(res.is_ok());
          res.unwrap()
        };
        let U = {
          let comm_W = W.commit(gens);
          let res = R1CSInstance::new(&S, &comm_W, &X);
          assert!(res.is_ok());
          res.unwrap()
        };

        // check that generated instance is satisfiable
        assert!(S.is_sat(gens, &U, &W).is_ok());

        (O, U, W)
      };

    let mut csprng: OsRng = OsRng;
    let I = S::random(&mut csprng); // the first input is picked randomly for the first instance
    let (O, U1, W1) = rand_inst_witness_generator(&gens, &I);
    let (_O, U2, W2) = rand_inst_witness_generator(&gens, &O);

    // execute a sequence of folds
    execute_sequence(&gens, &S, &U1, &W1, &U2, &W2);
  }
}
