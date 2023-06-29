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
};

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

pub struct NIVC<G: Group> {
  w: String,
  z: String, // (witness vector, x, u)
  program_counter: i32,
  claim_u: String,
  RC: Vec<NIFS<G>>
}

// Function F0_j
fn F0_j(j: usize, vk: (), Ui: (), ui: (), pci: (), i_z0_zi: (usize, (), ()), omega_i: (), T: ()) -> () {
  // Implement the function F0_j here
}

// Function trace
fn trace(F0_pci_plus_1: (), vk: (), Ui: (), ui: (), pci: (), i_z0_zi: (usize, (), ()), omega_i: (), T: ()) -> () {
  // Implement the function trace here
}

// Function hash
fn hash(vk: (), i_plus_1: usize, pci_plus_1: (), z0: (), zi_plus_1: (), Ui_plus_1: ()) -> () {
  // Implement the function hash here
}

impl <G: Group> NIVC<G> {
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
    traits::{Group, ROConstantsTrait, circuit::TrivialTestCircuit},
  };


  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;

  fn test_trivial_nivc_with<G1, G2>()
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
  {
  }

  #[test]
  fn test_trivial_nivc() {
      // Create a new instance of NIVC
      let mut nivc = NIVC::<G1>::new();
      
      // Test the initial values
      assert_eq!(nivc.RC.len(), 0);

      // We likely need to start with RC instances.

      test_trivial_nivc_with::<G1, G2>();

  }
}

