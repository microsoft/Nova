//! This module implements a non-interactive folding scheme
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]

use crate::{
  constants::{NUM_CHALLENGE_BITS, NUM_FE_FOR_RO},
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  scalar_as_base,
  traits::{commitment::CommitmentTrait, AbsorbInROTrait, Group, ROTrait},
  Commitment, CommitmentKey, CompressedCommitment,
};
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

/// A SNARK that holds the proof of a step of an incremental computation
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<G: Group> {
  pub(crate) comm_T: CompressedCommitment<G>,
  _p: PhantomData<G>,
}

type ROConstants<G> =
  <<G as Group>::RO as ROTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants;

impl<G: Group> NIFS<G> {
  /// Takes as input a Relaxed R1CS instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with the same structure `shape`
  /// and defined with respect to the same `ck`, and outputs
  /// a folded Relaxed R1CS instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  #[allow(clippy::too_many_arguments)]
  pub fn prove(
    ck: &CommitmentKey<G>,
    ro_consts: &ROConstants<G>,
    pp_digest: &G::Scalar,
    S: &R1CSShape<G>,
    U1: &RelaxedR1CSInstance<G>,
    W1: &RelaxedR1CSWitness<G>,
    U2: &R1CSInstance<G>,
    W2: &R1CSWitness<G>,
  ) -> Result<(NIFS<G>, (RelaxedR1CSInstance<G>, RelaxedR1CSWitness<G>)), NovaError> {
    // initialize a new RO
    let mut ro = G::RO::new(ro_consts.clone(), NUM_FE_FOR_RO);

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<G>(*pp_digest));

    // append U1 and U2 to transcript
    U1.absorb_in_ro(&mut ro);
    U2.absorb_in_ro(&mut ro);

    // compute a commitment to the cross-term
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2)?;

    // append `comm_T` to the transcript and obtain a challenge
    comm_T.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &comm_T, &r)?;

    // fold the witness using `r` and `T`
    let W = W1.fold(W2, &T, &r)?;

    // return the folded instance and witness
    Ok((
      Self {
        comm_T: comm_T.compress(),
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
    ro_consts: &ROConstants<G>,
    pp_digest: &G::Scalar,
    U1: &RelaxedR1CSInstance<G>,
    U2: &R1CSInstance<G>,
  ) -> Result<RelaxedR1CSInstance<G>, NovaError> {
    // initialize a new RO
    let mut ro = G::RO::new(ro_consts.clone(), NUM_FE_FOR_RO);

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<G>(*pp_digest));

    // append U1 and U2 to transcript
    U1.absorb_in_ro(&mut ro);
    U2.absorb_in_ro(&mut ro);

    // append `comm_T` to the transcript and obtain a challenge
    let comm_T = Commitment::<G>::decompress(&self.comm_T)?;
    comm_T.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &comm_T, &r)?;

    // return the folded instance
    Ok(U)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    r1cs::R1CS,
    traits::{Group, ROConstantsTrait},
  };
  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::{Field, PrimeField};
  use rand::rngs::OsRng;

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

  fn test_tiny_r1cs_bellperson_with<G>()
  where
    G: Group,
  {
    use crate::bellperson::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
    };

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_tiny_r1cs_bellperson(&mut cs, None);
    let (shape, ck) = cs.r1cs_shape();
    let ro_consts =
      <<G as Group>::RO as ROTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants::new();

    // Now get the instance and assignment for one instance
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = synthesize_tiny_r1cs_bellperson(&mut cs, Some(G::Scalar::from(5)));
    let (U1, W1) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the first instance is satisfiable
    assert!(shape.is_sat(&ck, &U1, &W1).is_ok());

    // Now get the instance and assignment for second instance
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = synthesize_tiny_r1cs_bellperson(&mut cs, Some(G::Scalar::from(135)));
    let (U2, W2) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the second instance is satisfiable
    assert!(shape.is_sat(&ck, &U2, &W2).is_ok());

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<G as Group>::Scalar::ZERO,
      &shape,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs_bellperson() {
    test_tiny_r1cs_bellperson_with::<G>();
  }

  #[allow(clippy::too_many_arguments)]
  fn execute_sequence<G>(
    ck: &CommitmentKey<G>,
    ro_consts: &<<G as Group>::RO as ROTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants,
    pp_digest: &<G as Group>::Scalar,
    shape: &R1CSShape<G>,
    U1: &R1CSInstance<G>,
    W1: &R1CSWitness<G>,
    U2: &R1CSInstance<G>,
    W2: &R1CSWitness<G>,
  ) where
    G: Group,
  {
    // produce a default running instance
    let mut r_W = RelaxedR1CSWitness::default(shape);
    let mut r_U = RelaxedR1CSInstance::default(ck, shape);

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let res = NIFS::prove(ck, ro_consts, pp_digest, shape, &r_U, &r_W, U1, W1);
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &r_U, U1);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // produce a step SNARK with (W2, U2) as the second incoming witness-instance pair
    let res = NIFS::prove(ck, ro_consts, pp_digest, shape, &r_U, &r_W, U2, W2);
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &r_U, U2);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    r_W = W;
    r_U = U;

    // check if the running instance is satisfiable
    assert!(shape.is_sat_relaxed(ck, &r_U, &r_W).is_ok());
  }

  fn test_tiny_r1cs_with<G: Group>() {
    let one = <G::Scalar as Field>::ONE;
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
      let mut A: Vec<(usize, usize, G::Scalar)> = Vec::new();
      let mut B: Vec<(usize, usize, G::Scalar)> = Vec::new();
      let mut C: Vec<(usize, usize, G::Scalar)> = Vec::new();

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

    // generate generators and ro constants
    let ck = R1CS::<G>::commitment_key(&S);
    let ro_consts =
      <<G as Group>::RO as ROTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants::new();

    let rand_inst_witness_generator =
      |ck: &CommitmentKey<G>, I: &G::Scalar| -> (G::Scalar, R1CSInstance<G>, R1CSWitness<G>) {
        let i0 = *I;

        // compute a satisfying (vars, X) tuple
        let (O, vars, X) = {
          let z0 = i0 * i0; // constraint 0
          let z1 = i0 * z0; // constraint 1
          let z2 = z1 + i0; // constraint 2
          let i1 = z2 + one + one + one + one + one; // constraint 3

          // store the witness and IO for the instance
          let W = vec![z0, z1, z2, <G::Scalar as Field>::ZERO];
          let X = vec![i0, i1];
          (i1, W, X)
        };

        let W = {
          let res = R1CSWitness::new(&S, &vars);
          assert!(res.is_ok());
          res.unwrap()
        };
        let U = {
          let comm_W = W.commit(ck);
          let res = R1CSInstance::new(&S, &comm_W, &X);
          assert!(res.is_ok());
          res.unwrap()
        };

        // check that generated instance is satisfiable
        assert!(S.is_sat(ck, &U, &W).is_ok());

        (O, U, W)
      };

    let mut csprng: OsRng = OsRng;
    let I = G::Scalar::random(&mut csprng); // the first input is picked randomly for the first instance
    let (O, U1, W1) = rand_inst_witness_generator(&ck, &I);
    let (_O, U2, W2) = rand_inst_witness_generator(&ck, &O);

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<G as Group>::Scalar::ZERO,
      &S,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs() {
    test_tiny_r1cs_with::<pasta_curves::pallas::Point>();
  }
}
