use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use nova_snark::bellperson::{
  r1cs::{NovaShape, NovaWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};

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
  let shape = cs.r1cs_shape();
  let gens = cs.r1cs_gens();
  println!("Mult mod constraint no: {}", cs.num_constraints());

  // Now get the assignment
  let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
  let _ = synthesize_alloc_bit(&mut cs);
  let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

  // Make sure that this is satisfiable
  assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}
