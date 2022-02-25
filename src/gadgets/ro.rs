//!This module contains the ro gadget
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
  },
  ConstraintSystem, SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};
use neptune::circuit::poseidon_hash;
use crate::poseidon::params::NovaPoseidonConstants;

///A random oracle instantiated with the poseidon hash function of a specified Arity
pub struct PoseidonRO<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  //Internal state
  state: Vec<AllocatedNum<Scalar>>,
  constants: NovaPoseidonConstants<Scalar>,
}

impl<Scalar> PoseidonRO<Scalar>
where
  Scalar: PrimeField + PrimeFieldBits,
{
  ///Initialize the internal state to 0
  #[allow(dead_code)]
  pub fn new(constants: NovaPoseidonConstants<Scalar>) -> Self {
    Self {
      state: Vec::new(),
      constants
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
    let out = match self.state.len() {
      9 => poseidon_hash(
            cs.namespace(|| "Poseidon hash"),
            self.state.clone(),
            &self.constants.constants9,
      )?,
      10 => poseidon_hash(
            cs.namespace(|| "Poseidon hash"),
            self.state.clone(),
            &self.constants.constants10,
      )?,
      _ => panic!("Number of elements in the RO state does not match any of the arities used in Nova")
    };
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
    let constants = NovaPoseidonConstants::new();
    let mut ro: PoseidonRO<S> = PoseidonRO::new(constants);
    let mut cs: ShapeCS<G> = ShapeCS::new();
    for i in 0..9 {
      let num =
        AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(S::zero())).unwrap();
      ro.absorb(num)
    }
    assert!(ro.get_challenge(&mut cs).is_ok());
    println!("Number of constraints {}", cs.num_constraints());
  }
}
