//!This module contains the ro gadget
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
  },
  ConstraintSystem, SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use neptune::{circuit::poseidon_hash, poseidon::PoseidonConstants, Arity, Strength};
use std::marker::PhantomData;

///A random oracle instantiated with the poseidon hash function of a specified Arity
pub struct PoseidonRO<Scalar, A>
where
  Scalar: PrimeField + PrimeFieldBits,
  A: Arity<Scalar>,
{
  //Internal state
  state: Vec<AllocatedNum<Scalar>>,
  phantom: PhantomData<A>,
}

impl<Scalar, A> PoseidonRO<Scalar, A>
where
  Scalar: PrimeField + PrimeFieldBits,
  A: Arity<Scalar>,
{
  ///Initialize the internal state to 0
  #[allow(dead_code)]
  pub fn new() -> Self {
    Self {
      state: Vec::new(),
      phantom: PhantomData,
    }
  }

  ///Absorb a new number into the state of the oracle
  #[allow(dead_code)]
  pub fn absorb(&mut self, e: AllocatedNum<Scalar>) {
    self.state.push(e.clone());
  }

  ///Compute a challenge by hashing the current state
  #[allow(dead_code)]
  pub fn get_challenge<CS>(self, mut cs: CS) -> Result<Vec<AllocatedBit>, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    //Make sure that the size of the state is equal to the arity
    assert_eq!(A::to_usize(), self.state.len());
    //Compute the constants and hash
    let constants = PoseidonConstants::<Scalar, A>::new_with_strength(Strength::Strengthened);
    let out = poseidon_hash(
      cs.namespace(|| "Poseidon hash"),
      self.state.clone(),
      &constants,
    )?;
    //Only keep 128 bits
    let bits: Vec<AllocatedBit> = out
      .to_bits_le_strict(cs.namespace(|| "poseidon hash to boolean"))?
      .iter()
      .map(|boolean| match boolean {
        Boolean::Is(ref x) => x.clone(),
        _ => panic!("Wrong type of input. We should have never reached there"),
      })
      .collect();
    Ok(bits[..128].into())
  }
}

///Gets as input a field element of F and returns its representation to a field element of G
#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellperson::shape_cs::ShapeCS;
  use crate::bellperson::solver::SatisfyingAssignment;
  use generic_array::typenum;
  type S = pasta_curves::pallas::Scalar;
  type G = pasta_curves::pallas::Point;
  use crate::gadgets::utils::le_bits_to_num;
  use crate::traits::PrimeField;
  use rand::rngs::OsRng;

  #[test]
  fn test_le_bits_to_num() {
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let mut csprng: OsRng = OsRng;
    let fe = S::random(&mut csprng);
    let num = AllocatedNum::alloc(cs.namespace(|| "input number"), || Ok(fe)).unwrap();
    let _ = num.inputize(&mut cs).unwrap();
    let bits = num
      .to_bits_le_strict(&mut cs)
      .unwrap()
      .iter()
      .map(|boolean| match boolean {
        Boolean::Is(ref x) => x.clone(),
        _ => panic!("Wrong type of input. We should have never reached there"),
      })
      .collect();
    let num2 = le_bits_to_num(&mut cs, bits).unwrap();
    assert!(num2.get_value() == num.get_value());
  }

  #[test]
  fn test_poseidon_ro() {
    let mut ro: PoseidonRO<S, typenum::U8> = PoseidonRO::new();
    let mut cs: ShapeCS<G> = ShapeCS::new();
    for i in 0..8 {
      let num =
        AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(S::zero())).unwrap();
      ro.absorb(num)
    }
    assert!(ro.get_challenge(&mut cs).is_ok());
    println!("Number of constraints {}", cs.num_constraints());
  }
}
