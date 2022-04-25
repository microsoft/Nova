use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use nova_snark::bellperson::{
  r1cs::{NovaShape, NovaWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};

fn synthesize_use_cs_one<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
  cs: &mut CS,
) -> Result<(), SynthesisError> {
  let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::one()))?;
  let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::one()))?;
  cs.enforce(
    || "check a = b",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );
  let _ = a.inputize(cs.namespace(|| "a is input"));
  let _ = b.inputize(cs.namespace(|| "b is input"));
  Ok(())
}

fn synthesize_use_cs_one_after_inputize<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
  cs: &mut CS,
) -> Result<(), SynthesisError> {
  let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::one()))?;
  let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::one()))?;
  let _ = a.inputize(cs.namespace(|| "a is input"));
  cs.enforce(
    || "check a = b",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc,
  );
  let _ = b.inputize(cs.namespace(|| "b is input"));
  Ok(())
}

#[test]
fn test_use_cs_one() {
  type G = pasta_curves::pallas::Point;

  // First create the shape
  let mut cs: ShapeCS<G> = ShapeCS::new();
  let _ = synthesize_use_cs_one(&mut cs);
  let shape = cs.r1cs_shape();
  let gens = cs.r1cs_gens();

  // Now get the assignment
  let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
  let _ = synthesize_use_cs_one(&mut cs);
  let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

  // Make sure that this is satisfiable
  assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}

#[test]
fn test_use_cs_one_after_inputize() {
  type G = pasta_curves::pallas::Point;

  // First create the shape
  let mut cs: ShapeCS<G> = ShapeCS::new();
  let _ = synthesize_use_cs_one_after_inputize(&mut cs);
  let shape = cs.r1cs_shape();
  let gens = cs.r1cs_gens();

  // Now get the assignment
  let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
  let _ = synthesize_use_cs_one_after_inputize(&mut cs);
  let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

  // Make sure that this is satisfiable
  assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}
