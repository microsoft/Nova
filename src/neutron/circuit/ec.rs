//! This module implements the scalar multiplication circuit for offloading scalar muls
//! on E1 to scalar muls on E2 via the CycleFold approach
use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{
    gadgets::{boolean::AllocatedBit, num::AllocatedNum},
    ConstraintSystem, SynthesisError,
  },
  gadgets::{ecc::AllocatedPoint, utils::le_bits_to_num},
  traits::{commitment::CommitmentTrait, Engine},
  Commitment,
};
use ff::{PrimeField, PrimeFieldBits};

pub struct ECCircuit<E: Engine> {
  r: Option<E::Base>, // 128-bit scalar can be uniquely represented it in both base and scalar fields
  P: Option<Commitment<E>>, // running commitment
  Q: Option<Commitment<E>>, // fresh commitment
}

impl<E: Engine> ECCircuit<E> {
  pub fn new(r: Option<E::Base>, P: Option<Commitment<E>>, Q: Option<Commitment<E>>) -> Self {
    Self { r, P, Q }
  }

  fn alloc_witness<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedNum<E::Base>,
      Vec<AllocatedBit>,
      AllocatedPoint<E>,
      AllocatedPoint<E>,
    ),
    SynthesisError,
  > {
    // allocate r as bits and assemble to scalar
    let r_bits: Vec<AllocatedBit> = Self::small_field_into_allocated_bits_le(
      cs.namespace(|| "r bits"),
      self.r,
      NUM_CHALLENGE_BITS,
    )?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // allocate P
    let P = AllocatedPoint::alloc(
      cs.namespace(|| "allocate P"),
      self.P.map(|c| c.to_coordinates()),
    )?;

    // allocate Q
    let Q = AllocatedPoint::alloc(
      cs.namespace(|| "allocate Q"),
      self.Q.map(|c| c.to_coordinates()),
    )?;

    Ok((r, r_bits, P, Q))
  }

  fn small_field_into_allocated_bits_le<CS>(
    mut cs: CS,
    value: Option<E::Base>,
    num_bits: usize,
  ) -> Result<Vec<AllocatedBit>, SynthesisError>
  where
    CS: ConstraintSystem<E::Base>,
  {
    // Deconstruct in big-endian bit order
    let values = match value {
      Some(ref value) => {
        let field_char = <E::Base as PrimeFieldBits>::char_le_bits();
        let mut field_char = field_char.into_iter().rev();

        let mut tmp = Vec::with_capacity(E::Base::NUM_BITS as usize);

        let mut found_one = false;
        for b in value.to_le_bits().into_iter().rev() {
          // Skip leading bits
          found_one |= field_char.next().unwrap();
          if !found_one {
            continue;
          }

          tmp.push(Some(b));
        }

        assert_eq!(tmp.len(), E::Base::NUM_BITS as usize);

        tmp
      }
      None => vec![None; E::Base::NUM_BITS as usize],
    };

    // Allocate in little-endian order
    let bits = values.into_iter().rev().collect::<Vec<_>>()[..num_bits]
      .iter()
      .enumerate()
      .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("bit {}", i)), *b))
      .collect::<Result<Vec<_>, SynthesisError>>()?;

    Ok(bits)
  }

  pub fn synthesize<CS: ConstraintSystem<E::Base>>(self, mut cs: CS) -> Result<(), SynthesisError> {
    // allocate witness
    let (r, r_bits, P, Q) = self.alloc_witness(cs.namespace(|| "allocate witness"))?;

    // compute r_updated = P + r * Q
    let r_times_Q = Q.scalar_mul(cs.namespace(|| "r_times_Q"), &r_bits)?;
    let R = P.add(cs.namespace(|| "R"), &r_times_Q)?;

    // inputize scalar, P, Q, R
    r.inputize(cs.namespace(|| "inputize r"))?;
    P.inputize(cs.namespace(|| "inputize P"))?;
    Q.inputize(cs.namespace(|| "inputize Q"))?;
    R.inputize(cs.namespace(|| "inputize R"))?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
    },
    provider::{
      poseidon::{PoseidonConstantsCircuit, PoseidonROCircuit},
      Bn256EngineKZG, GrumpkinEngine,
    },
    traits::{snark::default_ck_hint, Engine, ROCircuitTrait},
  };
  use ff::Field;
  use rand::rngs::OsRng;

  fn test_ec_circuit_with<E1, E2>(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    nz_A: usize,
    nz_B: usize,
    nz_C: usize,
  ) where
    E1: Engine<Scalar = <E2 as Engine>::Base>,
    E2: Engine<Scalar = <E1 as Engine>::Base>,
  {
    // generate a random challenge with Poseidon
    let mut csprng: OsRng = OsRng;
    let constants = PoseidonConstantsCircuit::<E1::Base>::default();
    let mut ro_gadget: PoseidonROCircuit<E1::Base> = PoseidonROCircuit::new(constants);
    let mut cs = SatisfyingAssignment::<E2>::new();

    let num_absorbs = 2;
    for i in 0..num_absorbs {
      let num = E1::Base::random(&mut csprng);
      let num_gadget = AllocatedNum::alloc_infallible(cs.namespace(|| format!("data {i}")), || num);
      ro_gadget.absorb(&num_gadget);
    }
    let r_bits = ro_gadget.squeeze(&mut cs, 128).unwrap();
    let r = le_bits_to_num(&mut cs, &r_bits)
      .unwrap()
      .get_value()
      .unwrap();

    let P = Commitment::<E1>::default();
    let Q = Commitment::<E1>::default();

    let circuit: ECCircuit<E1> = ECCircuit::<E1>::new(Some(r), Some(P), Some(Q));
    let mut cs: TestShapeCS<E2> = TestShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());

    assert_eq!(cs.num_constraints(), num_cons);
    assert_eq!(shape.num_vars, num_vars);
    assert_eq!(shape.num_io, num_io);
    assert_eq!(shape.A.len(), nz_A);
    assert_eq!(shape.B.len(), nz_B);
    assert_eq!(shape.C.len(), nz_C);

    let circuit: ECCircuit<E1> = ECCircuit::<E1>::new(Some(r), Some(P), Some(Q));
    let mut cs = SatisfyingAssignment::<E2>::new();
    let _ = circuit.synthesize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_ec_circuit() {
    test_ec_circuit_with::<Bn256EngineKZG, GrumpkinEngine>(1360, 1350, 10, 1777, 1762, 2417);
  }
}
