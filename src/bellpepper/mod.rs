//! Support for generating R1CS from [Bellpepper].
//!
//! [Bellpepper]: https://github.com/lurk-lab/bellpepper

pub mod r1cs;
pub mod shape_cs;
pub mod solver;
pub mod test_shape_cs;

#[cfg(test)]
mod tests {
  use crate::{
    bellpepper::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
    },
    traits::GroupExt,
  };
  use bellpepper_core::{num::AllocatedNum, ConstraintSystem};
  use ff::PrimeField;

  fn synthesize_alloc_bit<Fr: PrimeField, CS: ConstraintSystem<Fr>>(cs: &mut CS) {
    // get two bits as input and check that they are indeed bits
    let a = AllocatedNum::alloc_infallible(cs.namespace(|| "a"), || Fr::ONE);
    let _ = a.inputize(cs.namespace(|| "a is input"));
    cs.enforce(
      || "check a is 0 or 1",
      |lc| lc + CS::one() - a.get_variable(),
      |lc| lc + a.get_variable(),
      |lc| lc,
    );
    let b = AllocatedNum::alloc_infallible(cs.namespace(|| "b"), || Fr::ONE);
    let _ = b.inputize(cs.namespace(|| "b is input"));
    cs.enforce(
      || "check b is 0 or 1",
      |lc| lc + CS::one() - b.get_variable(),
      |lc| lc + b.get_variable(),
      |lc| lc,
    );
  }

  fn test_alloc_bit_with<G>()
  where
    G: GroupExt,
  {
    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    synthesize_alloc_bit(&mut cs);
    let (shape, ck) = cs.r1cs_shape();

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_alloc_bit(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_alloc_bit() {
    test_alloc_bit_with::<pasta_curves::pallas::Point>();
    test_alloc_bit_with::<crate::provider::bn256_grumpkin::bn256::Point>();
    test_alloc_bit_with::<crate::provider::secp_secq::secp256k1::Point>();
  }
}
