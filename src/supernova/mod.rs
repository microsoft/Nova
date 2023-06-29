//! This module implements SuperNova NIVC.
// https://eprint.iacr.org/2022/1758.pdf
// By Wyatt Benno 2023.
#![allow(unused_imports)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use crate::ccs;

use crate::{
  constants::{NUM_CHALLENGE_BITS, NUM_FE_FOR_RO},
  errors::NovaError,
  r1cs::R1CS,
  scalar_as_base,
  traits::{Group, ROConstantsTrait, circuit::TrivialTestCircuit, circuit::StepCircuit},
  Commitment, CommitmentKey, CompressedCommitment,
  RecursiveSNARK, PublicParams
};

use ff::Field;
use ff::PrimeField;
use core::marker::PhantomData;

/*
 NIFS (non-interactive folding scheme) is where the folding occurs after the MUX chooses a function.
 It is part of the 'prover'.

 Nova case:
 In a single instance it would take ui, Ui and output Ui+1.
 ui = claim about prior step.
 Ui = running instance.

 SuperNova case:
 In multi-instance it takes:
 - Claim for C'pci_ui
 - pci = the program counter.
 - RC = a array of running instance [Ui-1, Ui-2, ..]

 NIFS returns RESULT = Ui+1,pci.
 A WRITE step takes pci 'aka the current program counter' with RESULT
 and updates RC in the correct index.

 MUX and WRITE functions are needed in the prover.

*/
use crate::nifs::NIFS;

/*
 NIVC contains both prover and verifier.
 It returns an array of updated running claims from NIFS and
 (Zi+1, pci + 1) from the verifier.

 Nova case:
 In a single instance it would additionaly have zi -> C -> zi+1 (the verfier portion of folding)
 z0 = initial input.
 
 SuperNova case:
 In multi-instance it takes as input: 
 - wi = a witness for each step.
 - zi = (witness vector, public input x, the error scalar u)
 - It runs these through two functions Cpci+1 and Phi.

 Cpci+1 outputs Zi+1
 Phi outputs =  pci + 1.

*/

pub struct RunningClaims<G: Group, Ca: StepCircuit<<G as Group>::Scalar>> {
  w: String,
  z: String, // (witness vector, x, u)
  program_counter: i32,
  claim_u: String,
  RC: Vec<Ca>,
  _phantom: PhantomData<G>, 
}

impl<G: Group, Ca: StepCircuit<<G as Group>::Scalar>> RunningClaims<G, Ca> {
  pub fn new() -> Self {
      Self {
          w: String::new(),
          z: String::new(),
          program_counter: 0,
          claim_u: String::new(),
          RC: Vec::new(),
          _phantom: PhantomData, 
      }
  }

  pub fn push_circuit(&mut self, circuit: Ca) {
    self.RC.push(circuit);
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

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

      Ok(vec![y])
    }

    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }


  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;

  fn test_trivial_nivc_with<G1, G2, T: StepCircuit<<G1 as Group>::Scalar>>(pci: i32, t1: &T, t2: &T)
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
    T: StepCircuit<<G1 as Group>::Scalar> + std::fmt::Debug,
  {
    let test_circuit2 = CubicCircuit::<<G2 as Group>::Scalar>::default();

    println!("here: {:?}", t2);

    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      T,
      CubicCircuit<<G2 as Group>::Scalar>,
    >::setup(t1.clone(), test_circuit2.clone());

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark: Option<
    RecursiveSNARK<
      G1,
      G2,
      T,
      CubicCircuit<<G2 as Group>::Scalar>,
    >,
  > = None;

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove_step(
      &pp,
      recursive_snark,
      t1.clone(),
      test_circuit2,
      vec![<G1 as Group>::Scalar::ZERO],
      vec![<G2 as Group>::Scalar::ZERO],
    );
    assert!(res.is_ok());
    let recursive_snark_unwrapped = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark_unwrapped.verify(
      &pp,
      num_steps,
      vec![<G1 as Group>::Scalar::ZERO],
      vec![<G2 as Group>::Scalar::ZERO],
    );
    assert!(res.is_ok());
  }

  #[test]
  fn test_trivial_nivc() {
      // Structuring running claims
      let mut running_claim1 = RunningClaims::<G1, TrivialTestCircuit<<G1 as Group>::Scalar>>::new();
      let mut running_claim2 = RunningClaims::<G1, CubicCircuit<<G1 as Group>::Scalar>>::new();

      let test_circuit1 = TrivialTestCircuit::<<G1 as Group>::Scalar>::default();
      let test_circuit2 = CubicCircuit::<<G1 as Group>::Scalar>::default();
      running_claim1.push_circuit(test_circuit1);
      running_claim2.push_circuit(test_circuit2);

      let claims_tuple = (running_claim1, running_claim2);
    
      //Expirementing with selecting the running claims for nifs
      test_trivial_nivc_with::<G1, G2, TrivialTestCircuit<<G1 as Group>::Scalar>>(
        claims_tuple.0.program_counter,
        &claims_tuple.0.RC[0],
        &claims_tuple.0.RC[0]
      );

     /* test_trivial_nivc_with::<G1, G2, CubicCircuit<<G1 as Group>::Scalar>>(
        claims_tuple.1.program_counter,
        &claims_tuple.1.RC[0],
        &claims_tuple.1.RC[0]
      );*/

  }
}

