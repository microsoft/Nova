//! Poseidon Constants and Poseidon-based RO used in Nova
#[cfg(not(feature = "std"))]
use crate::prelude::*;
use crate::{
  frontend::{
    gadgets::poseidon::{
      Elt, IOPattern, PoseidonConstants, Simplex, Sponge, SpongeAPI, SpongeCircuit, SpongeOp,
      SpongeTrait, Strength,
    },
    num::AllocatedNum,
    AllocatedBit, Boolean, ConstraintSystem, SynthesisError,
  },
  traits::{ROCircuitTrait, ROTrait},
};
use core::marker::PhantomData;
use ff::{PrimeField, PrimeFieldBits};
use generic_array::typenum::U24;
use serde::{Deserialize, Serialize};

/// All Poseidon Constants that are used in Nova
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct PoseidonConstantsCircuit<Scalar: PrimeField>(PoseidonConstants<Scalar, U24>);

impl<Scalar: PrimeField> Default for PoseidonConstantsCircuit<Scalar> {
  /// Generate Poseidon constants
  fn default() -> Self {
    Self(Sponge::<Scalar, U24>::api_constants(Strength::Standard))
  }
}

/// A Poseidon-based RO to use outside circuits
#[derive(Serialize, Deserialize)]
pub struct PoseidonRO<Base, Scalar>
where
  Base: PrimeField,
  Scalar: PrimeField,
{
  // Internal State
  state: Vec<Base>,
  constants: PoseidonConstantsCircuit<Base>,
  squeezed: bool,
  _p: PhantomData<Scalar>,
}

impl<Base, Scalar> ROTrait<Base, Scalar> for PoseidonRO<Base, Scalar>
where
  Base: PrimeField + PrimeFieldBits + Serialize + for<'de> Deserialize<'de>,
  Scalar: PrimeField,
{
  type CircuitRO = PoseidonROCircuit<Base>;
  type Constants = PoseidonConstantsCircuit<Base>;

  fn new(constants: PoseidonConstantsCircuit<Base>) -> Self {
    Self {
      state: Vec::new(),
      constants,
      squeezed: false,
      _p: PhantomData,
    }
  }

  /// Absorb a new number into the state of the oracle
  fn absorb(&mut self, e: Base) {
    assert!(!self.squeezed, "Cannot absorb after squeezing");
    self.state.push(e);
  }

  /// Compute a challenge by hashing the current state
  fn squeeze(&mut self, num_bits: usize) -> Scalar {
    // check if we have squeezed already
    assert!(!self.squeezed, "Cannot squeeze again after squeezing");
    self.squeezed = true;

    let mut sponge = Sponge::new_with_constants(&self.constants.0, Simplex);
    let acc = &mut ();
    let parameter = IOPattern(vec![
      SpongeOp::Absorb(self.state.len() as u32),
      SpongeOp::Squeeze(1u32),
    ]);

    sponge.start(parameter, None, acc);
    SpongeAPI::absorb(&mut sponge, self.state.len() as u32, &self.state, acc);
    let hash = SpongeAPI::squeeze(&mut sponge, 1, acc);
    sponge.finish(acc).unwrap();

    // Only return `num_bits`
    let bits = hash[0].to_le_bits();
    let mut res = Scalar::ZERO;
    let mut coeff = Scalar::ONE;
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
#[derive(Serialize, Deserialize)]
pub struct PoseidonROCircuit<Scalar: PrimeField> {
  // Internal state
  state: Vec<AllocatedNum<Scalar>>,
  constants: PoseidonConstantsCircuit<Scalar>,
  squeezed: bool,
}

impl<Scalar> ROCircuitTrait<Scalar> for PoseidonROCircuit<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits + Serialize + for<'de> Deserialize<'de>,
{
  type NativeRO<T: PrimeField> = PoseidonRO<Scalar, T>;
  type Constants = PoseidonConstantsCircuit<Scalar>;

  /// Initialize the internal state and set the poseidon constants
  fn new(constants: PoseidonConstantsCircuit<Scalar>) -> Self {
    Self {
      state: Vec::new(),
      constants,
      squeezed: false,
    }
  }

  /// Absorb a new number into the state of the oracle
  fn absorb(&mut self, e: &AllocatedNum<Scalar>) {
    assert!(!self.squeezed, "Cannot absorb after squeezing");
    self.state.push(e.clone());
  }

  /// Compute a challenge by hashing the current state
  fn squeeze<CS: ConstraintSystem<Scalar>>(
    &mut self,
    mut cs: CS,
    num_bits: usize,
  ) -> Result<Vec<AllocatedBit>, SynthesisError> {
    // check if we have squeezed already
    assert!(!self.squeezed, "Cannot squeeze again after squeezing");
    self.squeezed = true;
    let parameter = IOPattern(vec![
      SpongeOp::Absorb(self.state.len() as u32),
      SpongeOp::Squeeze(1u32),
    ]);
    let mut ns = cs.namespace(|| "ns");

    let hash = {
      let mut sponge = SpongeCircuit::new_with_constants(&self.constants.0, Simplex);
      let acc = &mut ns;

      sponge.start(parameter, None, acc);
      SpongeAPI::absorb(
        &mut sponge,
        self.state.len() as u32,
        &(0..self.state.len())
          .map(|i| Elt::Allocated(self.state[i].clone()))
          .collect::<Vec<Elt<Scalar>>>(),
        acc,
      );

      let output = SpongeAPI::squeeze(&mut sponge, 1, acc);
      sponge.finish(acc).unwrap();
      output
    };

    let hash = Elt::ensure_allocated(&hash[0], &mut ns.namespace(|| "ensure allocated"), true)?;

    // return the hash as a vector of bits, truncated
    Ok(
      hash
        .to_bits_le_strict(ns.namespace(|| "poseidon hash to boolean"))?
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
  use crate::{
    constants::NUM_CHALLENGE_BITS,
    frontend::solver::SatisfyingAssignment,
    gadgets::utils::le_bits_to_num,
    gadgets::utils::scalar_as_base,
    provider::{
      Bn256EngineKZG, GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::Engine,
  };
  use ff::Field;
  use rand::rngs::OsRng;

  fn test_poseidon_ro_with<E: Engine>() {
    // Check that the number computed inside the circuit is equal to the number computed outside the circuit
    let mut csprng: OsRng = OsRng;
    let constants = PoseidonConstantsCircuit::<E::Scalar>::default();
    let num_absorbs = 32;
    let mut ro: PoseidonRO<E::Scalar, E::Base> = PoseidonRO::new(constants.clone());
    let mut ro_gadget: PoseidonROCircuit<E::Scalar> = PoseidonROCircuit::new(constants);
    let mut cs = SatisfyingAssignment::<E>::new();
    for i in 0..num_absorbs {
      let num = E::Scalar::random(&mut csprng);
      ro.absorb(num);
      let num_gadget = AllocatedNum::alloc_infallible(cs.namespace(|| format!("data {i}")), || num);
      num_gadget
        .inputize(&mut cs.namespace(|| format!("input {i}")))
        .unwrap();
      ro_gadget.absorb(&num_gadget);
    }
    let num = ro.squeeze(NUM_CHALLENGE_BITS);
    let num2_bits = ro_gadget.squeeze(&mut cs, NUM_CHALLENGE_BITS).unwrap();
    let num2 = le_bits_to_num(&mut cs, &num2_bits).unwrap();
    assert_eq!(num, scalar_as_base::<E>(num2.get_value().unwrap()));
  }

  #[test]
  fn test_poseidon_ro() {
    test_poseidon_ro_with::<PallasEngine>();
    test_poseidon_ro_with::<VestaEngine>();
    test_poseidon_ro_with::<Bn256EngineKZG>();
    test_poseidon_ro_with::<GrumpkinEngine>();
    test_poseidon_ro_with::<Secp256k1Engine>();
    test_poseidon_ro_with::<Secq256k1Engine>();
  }
}
