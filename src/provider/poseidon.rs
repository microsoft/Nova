//! Poseidon Constants and Poseidon-based RO used in Nova
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
pub struct PoseidonRO<Base: PrimeField> {
  // internal State
  state: Vec<Base>,
  constants: PoseidonConstantsCircuit<Base>,
}

impl<Base> ROTrait<Base> for PoseidonRO<Base>
where
  Base: PrimeField + PrimeFieldBits + Serialize + for<'de> Deserialize<'de>,
{
  type CircuitRO = PoseidonROCircuit<Base>;
  type Constants = PoseidonConstantsCircuit<Base>;

  fn new(constants: PoseidonConstantsCircuit<Base>) -> Self {
    Self {
      state: Vec::new(),
      constants,
    }
  }

  /// Absorb a new number into the state of the oracle
  fn absorb(&mut self, e: Base) {
    self.state.push(e);
  }

  /// Compute a challenge by hashing the current state
  fn squeeze(&mut self, num_bits: usize, start_with_one: bool) -> Base {
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

    // reset the state to only contain the squeezed value
    self.state = vec![hash[0]];

    // Only return `num_bits`
    let bits = hash[0].to_le_bits();
    let mut res = Base::ZERO;
    let mut coeff = Base::ONE;
    let start_idx = if start_with_one {
      res = coeff;
      coeff += coeff;
      1
    } else {
      0
    };
    for bit in bits[start_idx..num_bits].into_iter() {
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
}

impl<Scalar> ROCircuitTrait<Scalar> for PoseidonROCircuit<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits + Serialize + for<'de> Deserialize<'de>,
{
  type NativeRO = PoseidonRO<Scalar>;
  type Constants = PoseidonConstantsCircuit<Scalar>;

  /// Initialize the internal state and set the poseidon constants
  fn new(constants: PoseidonConstantsCircuit<Scalar>) -> Self {
    Self {
      state: Vec::new(),
      constants,
    }
  }

  /// Absorb a new number into the state of the oracle
  fn absorb(&mut self, e: &AllocatedNum<Scalar>) {
    self.state.push(e.clone());
  }

  /// Compute a challenge by hashing the current state
  fn squeeze<CS: ConstraintSystem<Scalar>>(
    &mut self,
    mut cs: CS,
    num_bits: usize,
    start_with_one: bool,
  ) -> Result<Vec<AllocatedBit>, SynthesisError> {
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

    // reset the state to only contain the squeezed value
    self.state = vec![hash.clone()];

    // return the hash as a vector of bits, truncated
    let mut bits: Vec<AllocatedBit> = hash
      .to_bits_le_strict(ns.namespace(|| "poseidon hash to boolean"))?
      .iter()
      .map(|boolean| match boolean {
        Boolean::Is(ref x) => x.clone(),
        _ => panic!("Wrong type of input. We should have never reached there"),
      })
      .collect::<Vec<AllocatedBit>>()[..num_bits]
      .to_vec();

    if start_with_one {
      bits[0] = AllocatedBit::alloc(ns.namespace(|| "set first bit to 1"), Some(true))?;
      ns.enforce(
        || "check bits[0] = 1",
        |lc| lc + bits[0].get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
      );
    }

    Ok(bits)
  }

  fn squeeze_scalar<CS: ConstraintSystem<Scalar>>(
    &mut self,
    mut cs: CS,
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
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

    // reset the state to only contain the squeezed value
    self.state = vec![hash.clone()];

    Ok(hash)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    constants::NUM_CHALLENGE_BITS,
    frontend::solver::SatisfyingAssignment,
    gadgets::utils::le_bits_to_num,
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
    let mut ro: PoseidonRO<E::Scalar> = PoseidonRO::new(constants.clone());
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
    let num = ro.squeeze(NUM_CHALLENGE_BITS, false);
    let num2_bits = ro_gadget
      .squeeze(&mut cs, NUM_CHALLENGE_BITS, false)
      .unwrap();
    let num2 = le_bits_to_num(&mut cs, &num2_bits).unwrap();
    assert_eq!(num, num2.get_value().unwrap());
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
