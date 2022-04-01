use ff::PrimeField;
use nova_snark::bellperson::{r1cs::{NovaWitness, NovaShape},solver::SatisfyingAssignment, shape_cs::ShapeCS};
use bellperson::{gadgets::num::AllocatedNum, SynthesisError, ConstraintSystem};

fn synthesize_alloc_bit<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
) -> Result<(), SynthesisError>{
    let a = AllocatedNum::alloc(
        cs.namespace(|| "a"),
        || Ok(Fr::one()),
    )?;
    cs.enforce(
        || "check 0 or 1",
        |lc| lc + CS::one() - a.get_variable(),
        |lc| lc + a.get_variable(),
        |lc| lc,
    );
    Ok(())
}


#[test]
fn test_alloc_bit(){
    type G = pasta_curves::pallas::Point;
    
    //First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();
    println!(
      "Mult mod constraint no: {}",
      cs.num_constraints()
    );
   
    //Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let _ = synthesize_alloc_bit(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    //Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}
