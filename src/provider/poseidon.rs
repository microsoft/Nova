//! Poseidon Constants and Poseidon-based RO used in Nova
use crate::traits::{ROCircuitTrait, ROTrait};
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, SynthesisError,
};
use core::marker::PhantomData;
use ff::{PrimeField, PrimeFieldBits};
use generic_array::typenum::U24;
use neptune::{
  circuit2::Elt,
  poseidon::PoseidonConstants,
  sponge::{
    api::{IOPattern, SpongeAPI, SpongeOp},
    circuit::SpongeCircuit,
    vanilla::{Mode::Simplex, Sponge, SpongeTrait},
  },
  Strength,
};
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
  num_absorbs: usize,
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

  fn new(constants: PoseidonConstantsCircuit<Base>, num_absorbs: usize) -> Self {
    Self {
      state: Vec::new(),
      constants,
      num_absorbs,
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
      SpongeOp::Absorb(self.num_absorbs as u32),
      SpongeOp::Squeeze(1u32),
    ]);

    sponge.start(parameter, None, acc);
    assert_eq!(self.num_absorbs, self.state.len());
    SpongeAPI::absorb(&mut sponge, self.num_absorbs as u32, &self.state, acc);
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
  num_absorbs: usize,
  squeezed: bool,
}

impl<Scalar> ROCircuitTrait<Scalar> for PoseidonROCircuit<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits + Serialize + for<'de> Deserialize<'de>,
{
  type NativeRO<T: PrimeField> = PoseidonRO<Scalar, T>;
  type Constants = PoseidonConstantsCircuit<Scalar>;

  /// Initialize the internal state and set the poseidon constants
  fn new(constants: PoseidonConstantsCircuit<Scalar>, num_absorbs: usize) -> Self {
    Self {
      state: Vec::new(),
      constants,
      num_absorbs,
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
      SpongeOp::Absorb(self.num_absorbs as u32),
      SpongeOp::Squeeze(1u32),
    ]);
    let mut ns = cs.namespace(|| "ns");

    let hash = {
      let mut sponge = SpongeCircuit::new_with_constants(&self.constants.0, Simplex);
      let acc = &mut ns;
      assert_eq!(self.num_absorbs, self.state.len());

      sponge.start(parameter, None, acc);
      neptune::sponge::api::SpongeAPI::absorb(
        &mut sponge,
        self.num_absorbs as u32,
        &(0..self.state.len())
          .map(|i| Elt::Allocated(self.state[i].clone()))
          .collect::<Vec<Elt<Scalar>>>(),
        acc,
      );

      let output = neptune::sponge::api::SpongeAPI::squeeze(&mut sponge, 1, acc);
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
  use crate::provider::{
    Bn256EngineKZG, GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
  };
  use crate::{
    bellpepper::solver::SatisfyingAssignment, constants::NUM_CHALLENGE_BITS,
    gadgets::utils::le_bits_to_num, traits::Engine,
  };
  use ff::Field;
  use rand::rngs::OsRng;

  fn test_poseidon_ro_with<E: Engine>()
  where
    // we can print the field elements we get from E's Base & Scalar fields,
    // and compare their byte representations
    <<E as Engine>::Base as PrimeField>::Repr: std::fmt::Debug,
    <<E as Engine>::Scalar as PrimeField>::Repr: std::fmt::Debug,
    <<E as Engine>::Base as PrimeField>::Repr:
      PartialEq<<<E as Engine>::Scalar as PrimeField>::Repr>,
  {
    // Check that the number computed inside the circuit is equal to the number computed outside the circuit
    let mut csprng: OsRng = OsRng;
    let constants = PoseidonConstantsCircuit::<E::Scalar>::default();
    let num_absorbs = 32;
    let mut ro: PoseidonRO<E::Scalar, E::Base> = PoseidonRO::new(constants.clone(), num_absorbs);
    let mut ro_gadget: PoseidonROCircuit<E::Scalar> =
      PoseidonROCircuit::new(constants, num_absorbs);
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
    assert_eq!(num.to_repr(), num2.get_value().unwrap().to_repr());
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
