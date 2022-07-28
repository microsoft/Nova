//! Poseidon Constants and Poseidon-based RO used in Nova
use super::traits::{ROCircuitTrait, ROConstantsTrait, ROTrait};
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
  },
  ConstraintSystem, SynthesisError,
};
use core::marker::PhantomData;
use ff::{PrimeField, PrimeFieldBits};
use generic_array::typenum::{U19, U24};
use neptune::{
  circuit::poseidon_hash,
  poseidon::{Poseidon, PoseidonConstants},
  Strength,
};

/// All Poseidon Constants that are used in Nova
#[derive(Clone)]
pub struct PoseidonConstantsCircuit<Scalar>
where
  Scalar: PrimeField,
{
  constants19: PoseidonConstants<Scalar, U19>,
  constants24: PoseidonConstants<Scalar, U24>,
}

impl<Scalar> ROConstantsTrait<Scalar> for PoseidonConstantsCircuit<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  /// Generate Poseidon constants for the arities that Nova uses
  #[allow(clippy::new_without_default)]
  fn new() -> Self {
    let constants19 = PoseidonConstants::<Scalar, U19>::new_with_strength(Strength::Standard);
    let constants24 = PoseidonConstants::<Scalar, U24>::new_with_strength(Strength::Standard);
    Self {
      constants19,
      constants24,
    }
  }
}

/// A Poseidon-based RO to use outside circuits
pub struct PoseidonRO<Base, Scalar>
where
  Base: PrimeField + PrimeFieldBits,
  Scalar: PrimeField + PrimeFieldBits,
{
  // Internal State
  state: Vec<Base>,
  // Constants for Poseidon
  constants: PoseidonConstantsCircuit<Base>,
  _p: PhantomData<Scalar>,
}

impl<Base, Scalar> ROTrait<Base, Scalar> for PoseidonRO<Base, Scalar>
where
  Base: PrimeField + PrimeFieldBits,
  Scalar: PrimeField + PrimeFieldBits,
{
  type Constants = PoseidonConstantsCircuit<Base>;

  fn new(constants: PoseidonConstantsCircuit<Base>) -> Self {
    Self {
      state: Vec::new(),
      constants,
      _p: PhantomData::default(),
    }
  }

  /// Absorb a new number into the state of the oracle
  fn absorb(&mut self, e: Base) {
    self.state.push(e);
  }

  /// Compute a challenge by hashing the current state
  fn squeeze(&self, num_bits: usize) -> Scalar {
    let hash = match self.state.len() {
      19 => {
        Poseidon::<Base, U19>::new_with_preimage(&self.state, &self.constants.constants19).hash()
      }
      24 => {
        Poseidon::<Base, U24>::new_with_preimage(&self.state, &self.constants.constants24).hash()
      }
      _ => {
        panic!(
          "Number of elements in the RO state does not match any of the arities used in Nova: {:?}",
          self.state.len()
        );
      }
    };

    // Only return `num_bits`
    let bits = hash.to_le_bits();
    let mut res = Scalar::zero();
    let mut coeff = Scalar::one();
    for bit in bits[0..num_bits].into_iter() {
      if *bit {
        res += coeff;
      }
      coeff += coeff;
    }
    res
  }
}

/// A Poseidon-based RO gadget to use inside the verifier circuit.
pub struct PoseidonROCircuit<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  // Internal state
  state: Vec<AllocatedNum<Scalar>>,
  constants: PoseidonConstantsCircuit<Scalar>,
}

impl<Scalar> ROCircuitTrait<Scalar> for PoseidonROCircuit<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  type Constants = PoseidonConstantsCircuit<Scalar>;

  /// Initialize the internal state and set the poseidon constants
  fn new(constants: PoseidonConstantsCircuit<Scalar>) -> Self {
    Self {
      state: Vec::new(),
      constants,
    }
  }

  /// Absorb a new number into the state of the oracle
  fn absorb(&mut self, e: AllocatedNum<Scalar>) {
    self.state.push(e);
  }

  /// Compute a challenge by hashing the current state
  fn squeeze<CS>(
    &mut self,
    mut cs: CS,
    num_bits: usize,
  ) -> Result<Vec<AllocatedBit>, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let hash = match self.state.len() {
      19 => poseidon_hash(
        cs.namespace(|| "Poseidon hash"),
        self.state.clone(),
        &self.constants.constants19,
      )?,
      24 => poseidon_hash(
        cs.namespace(|| "Posideon hash"),
        self.state.clone(),
        &self.constants.constants24,
      )?,
      _ => {
        panic!(
          "Number of elements in the RO state does not match any of the arities used in Nova: {}",
          self.state.len()
        )
      }
    };

    // return the hash as a vector of bits, truncated
    Ok(
      hash
        .to_bits_le_strict(cs.namespace(|| "poseidon hash to boolean"))?
        .iter()
        .map(|boolean| match boolean {
          Boolean::Is(ref x) => x.clone(),
          _ => panic!("Wrong type of input. We should have never reached there"),
        })
        .collect::<Vec<AllocatedBit>>()[..num_bits]
        .into(),
    )
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  type S = pasta_curves::pallas::Scalar;
  type B = pasta_curves::vesta::Scalar;
  type G = pasta_curves::pallas::Point;
  use crate::{
    bellperson::solver::SatisfyingAssignment, constants::NUM_CHALLENGE_BITS,
    gadgets::utils::le_bits_to_num,
  };
  use ff::Field;
  use rand::rngs::OsRng;

  #[test]
  fn test_poseidon_ro() {
    // Check that the number computed inside the circuit is equal to the number computed outside the circuit
    let mut csprng: OsRng = OsRng;
    let constants = PoseidonConstantsCircuit::new();
    let mut ro: PoseidonRO<S, B> = PoseidonRO::new(constants.clone());
    let mut ro_gadget: PoseidonROCircuit<S> = PoseidonROCircuit::new(constants);
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    for i in 0..19 {
      let num = S::random(&mut csprng);
      ro.absorb(num);
      let num_gadget =
        AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(num)).unwrap();
      num_gadget
        .inputize(&mut cs.namespace(|| format!("input {}", i)))
        .unwrap();
      ro_gadget.absorb(num_gadget);
    }
    let num = ro.squeeze(NUM_CHALLENGE_BITS);
    let num2_bits = ro_gadget.squeeze(&mut cs, NUM_CHALLENGE_BITS).unwrap();
    let num2 = le_bits_to_num(&mut cs, num2_bits).unwrap();
    assert_eq!(num.to_repr(), num2.get_value().unwrap().to_repr());
  }
}
