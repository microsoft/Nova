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
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  scalar_as_base,
  traits::{commitment::CommitmentTrait, AbsorbInROTrait, Group, ROTrait},
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

pub struct NIVC<C> {
  w: String,
  z: String, // (witness vector, x, u)
  program_counter: i32,
  claim_u: String,
  RC: Vec<C>
}

impl<C> NIVC <C> {
  pub fn new() -> Self {
      Self {
          w: String::new(),
          z: String::new(),
          program_counter: 0,
          claim_u: String::new(),
          RC: Vec::new(),
      }
  }

  //implement here..
}

#[cfg(test)]
mod tests {
  use super::*;

  use crate::{
    r1cs::R1CS,
    traits::{Group, ROConstantsTrait, circuit::TrivialTestCircuit, circuit::StepCircuit},
  };
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

  fn test_trivial_nivc_with<G1, G2>(pci: i32)
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
  {
    let test_circuit1 = TrivialTestCircuit::<<G1 as Group>::Scalar>::default();
    let test_circuit2 = CubicCircuit::<<G2 as Group>::Scalar>::default();

    println!("here: {:?}", test_circuit2);

    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      TrivialTestCircuit<<G1 as Group>::Scalar>,
      CubicCircuit<<G2 as Group>::Scalar>,
    >::setup(test_circuit1.clone(), test_circuit2.clone());

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark: Option<
    RecursiveSNARK<
      G1,
      G2,
      TrivialTestCircuit<<G1 as Group>::Scalar>,
      CubicCircuit<<G2 as Group>::Scalar>,
    >,
  > = None;

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove_step(
      &pp,
      recursive_snark,
      test_circuit1,
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
      // Create a new instance of NIVC
      let mut nivc = NIVC::new();

      //TODO: build list of circuits here to be added RC vector.
      //let test_circuit1 = TrivialTestCircuit::<<G1 as Group>::Scalar>::default();
      //let test_circuit2 = CubicCircuit::<<G2 as Group>::Scalar>::default();
      //nivc.RC.push(test_circuit1);
      //nivc.RC.push(test_circuit2);
    
      //Starting with the ivc test and transforming it into an nivc test.
      test_trivial_nivc_with::<G1, G2>(nivc.program_counter);

  }
}

