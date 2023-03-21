//! Support for generating R1CS from [Bellperson].
//!
//! [Bellperson]: https://github.com/filecoin-project/bellperson

pub mod r1cs;
pub mod shape_cs;
pub mod solver;

#[cfg(test)]
mod tests {
  use crate::bellperson::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  };
  use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::PrimeField;

  fn synthesize_alloc_bit<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    // get two bits as input and check that they are indeed bits
    let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::one()))?;
    let _ = a.inputize(cs.namespace(|| "a is input"));
    cs.enforce(
      || "check a is 0 or 1",
      |lc| lc + CS::one() - a.get_variable(),
      |lc| lc + a.get_variable(),
      |lc| lc,
    );
    let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::one()))?;
    let _ = b.inputize(cs.namespace(|| "b is input"));
    cs.enforce(
      || "check b is 0 or 1",
      |lc| lc + CS::one() - b.get_variable(),
      |lc| lc + b.get_variable(),
      |lc| lc,
    );
    Ok(())
  }

  #[test]
  fn test_alloc_bit() {
    type G = pasta_curves::pallas::Point;

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let (shape, ck) = cs.r1cs_shape();

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }
}
