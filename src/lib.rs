#![allow(non_snake_case)]
#![feature(test)]
#![deny(missing_docs)]
#![feature(external_doc)]
#![doc(include = "../README.md")]

extern crate core;
extern crate curve25519_dalek;
extern crate digest;
extern crate merlin;
extern crate rand;
extern crate rayon;
extern crate sha3;
extern crate test;

mod commitments;
mod errors;
mod r1cs;

use commitments::{AppendToTranscriptTrait, CompressedCommitment, ProofTranscriptTrait};
use errors::NovaError;
use merlin::Transcript;
use r1cs::{R1CSGens, R1CSInstance, R1CSShape, R1CSWitness};

/// A SNARK that holds the proof of a step of an incremental computation
pub struct StepSNARK {
  comm_T: CompressedCommitment,
}

impl StepSNARK {
  fn protocol_name() -> &'static [u8] {
    b"NovaStepSNARK"
  }

  /// Takes as input two relaxed R1CS instance-witness tuples `(U1, W1)` and `(U2, W2)`
  /// with the same structure `shape` and defined with respect to the same `gens`,
  /// and outputs a folded instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  pub fn prove(
    gens: &R1CSGens,
    S: &R1CSShape,
    U1: &R1CSInstance,
    W1: &R1CSWitness,
    U2: &R1CSInstance,
    W2: &R1CSWitness,
    transcript: &mut Transcript,
  ) -> Result<(StepSNARK, (R1CSInstance, R1CSWitness)), NovaError> {
    // append the protocol name to the transcript
    transcript.append_protocol_name(StepSNARK::protocol_name());

    // compute a commitment to the cross-term
    let (T, comm_T) = S.commit_T(gens, U1, W1, U2, W2)?;

    // append `comm_T` to the transcript and obtain a challenge
    comm_T.append_to_transcript(b"comm_T", transcript);

    // compute a challenge from the transcript
    let r = transcript.challenge_scalar(b"r");

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &comm_T, &r)?;

    // fold the witness using `r` and `T`
    let W = W1.fold(W2, &T, &r)?;

    // return the folded instance and witness
    Ok((StepSNARK { comm_T }, (U, W)))
  }

  /// Takes as input two relaxed R1CS instances `U1` and `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  pub fn verify(
    &self,
    U1: &R1CSInstance,
    U2: &R1CSInstance,
    transcript: &mut Transcript,
  ) -> Result<R1CSInstance, NovaError> {
    // append the protocol name to the transcript
    transcript.append_protocol_name(StepSNARK::protocol_name());

    // append `comm_T` to the transcript and obtain a challenge
    self.comm_T.append_to_transcript(b"comm_T", transcript);

    // compute a challenge from the transcript
    let r = transcript.challenge_scalar(b"r");

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &self.comm_T, &r)?;

    // return the folded instance and witness
    Ok(U)
  }
}

/// A SNARK that holds the proof of the final step of an incremental computation
pub struct FinalSNARK {
  W: R1CSWitness,
}

impl FinalSNARK {
  /// Produces a proof of a instance given its satisfying witness `W`.
  pub fn prove(W: &R1CSWitness) -> Result<FinalSNARK, NovaError> {
    Ok(FinalSNARK { W: W.clone() })
  }

  /// Verifies the proof of a folded instance `U` given its shape `S` public parameters `gens`
  pub fn verify(&self, gens: &R1CSGens, S: &R1CSShape, U: &R1CSInstance) -> Result<(), NovaError> {
    // check that the witness is a valid witness to the folded instance `U`
    S.is_sat(gens, U, &self.W)
  }
}

#[cfg(test)]
mod tests {
  use super::commitments::Scalar;
  use super::*;
  use rand::rngs::OsRng;

  #[test]
  fn test_tiny_r1cs() {
    let one = Scalar::one();
    let (num_cons, num_vars, num_inputs, A, B, C) = {
      let num_cons = 4;
      let num_vars = 4;
      let num_inputs = 1;

      // The R1CS for this problem consists of the following constraints:
      // `Z0 * Z0 - Z1 = 0`
      // `Z1 * Z0 - Z2 = 0`
      // `(Z2 + Z0) * 1 - Z3 = 0`
      // `(Z3 + 5) * 1 - I0 = 0`

      // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
      // constraint and a column for every entry in z = (vars, u, inputs)
      // An R1CS instance is satisfiable iff:
      // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
      let mut A: Vec<(usize, usize, Scalar)> = Vec::new();
      let mut B: Vec<(usize, usize, Scalar)> = Vec::new();
      let mut C: Vec<(usize, usize, Scalar)> = Vec::new();

      // constraint 0 entries in (A,B,C)
      A.push((0, 0, one));
      B.push((0, 0, one));
      C.push((0, 1, one));

      // constraint 1 entries in (A,B,C)
      A.push((1, 1, one));
      B.push((1, 0, one));
      C.push((1, 2, one));

      // constraint 2 entries in (A,B,C)
      A.push((2, 2, one));
      A.push((2, 0, one));
      B.push((2, num_vars, one));
      C.push((2, 3, one));

      // constraint 3 entries in (A,B,C)
      A.push((3, 3, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 1, one));

      (num_cons, num_vars, num_inputs, A, B, C)
    };

    // create a shape object
    let S = {
      let res = R1CSShape::new(num_cons, num_vars, num_inputs, &A, &B, &C);
      assert!(res.is_ok());
      res.unwrap()
    };

    // generate generators
    let gens = R1CSGens::new(num_cons, num_vars);

    let rand_inst_witness_generator = |gens: &R1CSGens| -> (R1CSInstance, R1CSWitness) {
      // compute a satisfying (vars, X) tuple
      let (vars, X) = {
        let mut csprng: OsRng = OsRng;
        let z0 = Scalar::random(&mut csprng);
        let z1 = z0 * z0; // constraint 0
        let z2 = z1 * z0; // constraint 1
        let z3 = z2 + z0; // constraint 2
        let i0 = z3 + one + one + one + one + one; // constraint 3

        let vars = vec![z0, z1, z2, z3];
        let X = vec![i0];
        (vars, X)
      };

      let W = {
        let E = vec![Scalar::zero(); num_cons]; // default E
        let res = R1CSWitness::new(&S, &vars, &E);
        assert!(res.is_ok());
        res.unwrap()
      };
      let U = {
        let (comm_W, comm_E) = W.commit(&gens);
        let u = Scalar::one(); //default u
        let res = R1CSInstance::new(&S, &comm_W, &comm_E, &X, &u);
        assert!(res.is_ok());
        res.unwrap()
      };

      // check that generated instance is satisfiable
      let is_sat = S.is_sat(&gens, &U, &W);
      assert!(is_sat.is_ok());
      (U, W)
    };

    let (U1, W1) = rand_inst_witness_generator(&gens);
    let (U2, W2) = rand_inst_witness_generator(&gens);

    // produce a step SNARK
    let mut prover_transcript = Transcript::new(b"StepSNARKExample");
    let res = StepSNARK::prove(&gens, &S, &U1, &W1, &U2, &W2, &mut prover_transcript);
    assert!(res.is_ok());
    let (step_snark, (_U, W)) = res.unwrap();

    // verify the step SNARK
    let mut verifier_transcript = Transcript::new(b"StepSNARKExample");
    let res = step_snark.verify(&U1, &U2, &mut verifier_transcript);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // produce a final SNARK
    let res = FinalSNARK::prove(&W);
    assert!(res.is_ok());
    let final_snark = res.unwrap();
    // verify the final SNARK
    let res = final_snark.verify(&gens, &S, &U);
    assert!(res.is_ok());
  }
}
