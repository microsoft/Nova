//! Poseidon Constants and Poseidon-based RO used in Nova
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
  },
  ConstraintSystem, SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use generic_array::typenum::{U25, U27, U31, U8};
use neptune::{
  circuit::poseidon_hash,
  poseidon::{Poseidon, PoseidonConstants},
};

#[cfg(test)]
use neptune::Strength;

/// All Poseidon Constants that are used in Nova
#[derive(Clone)]
pub struct NovaPoseidonConstants<F>
where
  F: PrimeField,
{
  constants8: PoseidonConstants<F, U8>,
  constants25: PoseidonConstants<F, U25>,
  constants27: PoseidonConstants<F, U27>,
  constants31: PoseidonConstants<F, U31>,
}

#[cfg(test)]
impl<F> NovaPoseidonConstants<F>
where
  F: PrimeField,
{
  /// Generate Poseidon constants for the arities that Nova uses
  pub fn new() -> Self {
    let constants8 = PoseidonConstants::<F, U8>::new_with_strength(Strength::Strengthened);
    let constants25 = PoseidonConstants::<F, U25>::new_with_strength(Strength::Strengthened);
    let constants27 = PoseidonConstants::<F, U27>::new_with_strength(Strength::Strengthened);
    let constants31 = PoseidonConstants::<F, U31>::new_with_strength(Strength::Strengthened);
    Self {
      constants8,
      constants25,
      constants27,
      constants31,
    }
  }
}

/// A Poseidon-based RO to use outside circuits
pub struct PoseidonRO<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  // Internal State
  state: Vec<Scalar>,
  // Constants for Poseidon
  constants: NovaPoseidonConstants<Scalar>,
}

impl<Scalar> PoseidonRO<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  #[allow(dead_code)]
  pub fn new(constants: NovaPoseidonConstants<Scalar>) -> Self {
    Self {
      state: Vec::new(),
      constants,
    }
  }

  /// Absorb a new number into the state of the oracle
  #[allow(dead_code)]
  pub fn absorb(&mut self, e: Scalar) {
    self.state.push(e);
  }

  fn hash_inner(&mut self) -> Scalar {
    match self.state.len() {
      8 => {
        Poseidon::<Scalar, U8>::new_with_preimage(&self.state, &self.constants.constants8).hash()
      }
      25 => {
        Poseidon::<Scalar, U25>::new_with_preimage(&self.state, &self.constants.constants25).hash()
      }
      27 => {
        Poseidon::<Scalar, U27>::new_with_preimage(&self.state, &self.constants.constants27).hash()
      }
      31 => {
        Poseidon::<Scalar, U31>::new_with_preimage(&self.state, &self.constants.constants31).hash()
      }
      _ => {
        panic!("Number of elements in the RO state does not match any of the arities used in Nova")
      }
    }
  }

  /// Compute a challenge by hashing the current state
  #[allow(dead_code)]
  pub fn get_challenge(&mut self) -> Scalar {
    let hash = self.hash_inner();
    // Only keep 128 bits
    let bits = hash.to_le_bits();
    let mut res = Scalar::zero();
    let mut coeff = Scalar::one();
    for bit in bits[0..128].into_iter() {
      if *bit {
        res += coeff;
      }
      coeff += coeff;
    }
    res
  }

  #[allow(dead_code)]
  pub fn get_hash(&mut self) -> Scalar {
    let hash = self.hash_inner();
    // Only keep 250 bits
    let bits = hash.to_le_bits();
    let mut res = Scalar::zero();
    let mut coeff = Scalar::one();
    for bit in bits[0..250].into_iter() {
      if *bit {
        res += coeff;
      }
      coeff += coeff;
    }
    res
  }
}

/// A Poseidon-based RO gadget to use inside the verifier circuit.
pub struct PoseidonROGadget<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  // Internal state
  state: Vec<AllocatedNum<Scalar>>,
  constants: NovaPoseidonConstants<Scalar>,
}

impl<Scalar> PoseidonROGadget<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  /// Initialize the internal state and set the poseidon constants
  #[allow(dead_code)]
  pub fn new(constants: NovaPoseidonConstants<Scalar>) -> Self {
    Self {
      state: Vec::new(),
      constants,
    }
  }

  /// Absorb a new number into the state of the oracle
  #[allow(dead_code)]
  pub fn absorb(&mut self, e: AllocatedNum<Scalar>) {
    self.state.push(e);
  }

  fn hash_inner<CS>(&mut self, mut cs: CS) -> Result<Vec<AllocatedBit>, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let out = match self.state.len() {
      8 => poseidon_hash(
        cs.namespace(|| "Posideon hash"),
        self.state.clone(),
        &self.constants.constants8,
      )?,
      25 => poseidon_hash(
        cs.namespace(|| "Poseidon hash"),
        self.state.clone(),
        &self.constants.constants25,
      )?,
      27 => poseidon_hash(
        cs.namespace(|| "Poseidon hash"),
        self.state.clone(),
        &self.constants.constants27,
      )?,
      31 => poseidon_hash(
        cs.namespace(|| "Poseidon hash"),
        self.state.clone(),
        &self.constants.constants31,
      )?,
      _ => {
        panic!(
          "Number of elements in the RO state does not match any of the arities used in Nova {}",
          self.state.len()
        )
      }
    };

    // return the hash as a vector of bits
    Ok(
      out
        .to_bits_le_strict(cs.namespace(|| "poseidon hash to boolean"))?
        .iter()
        .map(|boolean| match boolean {
          Boolean::Is(ref x) => x.clone(),
          _ => panic!("Wrong type of input. We should have never reached there"),
        })
        .collect(),
    )
  }

  /// Compute a challenge by hashing the current state
  #[allow(dead_code)]
  pub fn get_challenge<CS>(&mut self, mut cs: CS) -> Result<Vec<AllocatedBit>, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let bits = self.hash_inner(cs.namespace(|| "hash"))?;
    // Only keep 128 bits
    Ok(bits[..128].into())
  }

  #[allow(dead_code)]
  pub fn get_hash<CS>(&mut self, mut cs: CS) -> Result<Vec<AllocatedBit>, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let bits = self.hash_inner(cs.namespace(|| "hash"))?;
    // Only keep 250 bits
    Ok(bits[..250].into())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  type S = pasta_curves::pallas::Scalar;
  type G = pasta_curves::pallas::Point;
  use crate::{bellperson::solver::SatisfyingAssignment, gadgets::utils::le_bits_to_num};
  use ff::Field;
  use rand::rngs::OsRng;

  #[test]
  fn test_poseidon_ro() {
    // Check that the number computed inside the circuit is equal to the number computed outside the circuit
    let mut csprng: OsRng = OsRng;
    let constants = NovaPoseidonConstants::new();
    let mut ro: PoseidonRO<S> = PoseidonRO::new(constants.clone());
    let mut ro_gadget: PoseidonROGadget<S> = PoseidonROGadget::new(constants);
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    for i in 0..31 {
      let num = S::random(&mut csprng);
      ro.absorb(num);
      let num_gadget =
        AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(num)).unwrap();
      let _ = num_gadget
        .inputize(&mut cs.namespace(|| format!("input {}", i)))
        .unwrap();
      ro_gadget.absorb(num_gadget);
    }
    let num = ro.get_challenge();
    let num2_bits = ro_gadget.get_challenge(&mut cs).unwrap();
    let num2 = le_bits_to_num(&mut cs, num2_bits).unwrap();
    assert_eq!(num, num2.get_value().unwrap());
  }
}
