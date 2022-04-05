use bellperson::{ConstraintSystem, SynthesisError};
use bellperson_nonnative::{
  mp::bignat::BigNat,
  util::{convert::nat_to_f, num::Num},
};
use ff::PrimeField;
use nova_snark::bellperson::{
  r1cs::{NovaShape, NovaWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};
use rug::Integer;

fn synthesize_is_equal<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
  cs: &mut CS,
  a_val: &Integer,
  limb_width: usize,
  n_limbs: usize,
) -> Result<(), SynthesisError> {
  let a_num = Num::alloc(cs.namespace(|| "alloc a num"), || {
    Ok(nat_to_f(a_val).unwrap())
  })?;
  let a1 = BigNat::from_num(
    cs.namespace(|| "allocate a1_limbs"),
    a_num,
    limb_width,
    n_limbs,
  )?;
  let a2 = BigNat::alloc_from_nat(
    cs.namespace(|| "alloc a2"),
    || Ok(a_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let _ = a2.inputize(cs.namespace(|| "make a input"));
  let _ = a1.equal_when_carried(cs.namespace(|| "check equal"), &a2)?;
  Ok(())
}

#[allow(clippy::too_many_arguments)]
fn synthesize_mult_mod<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
  cs: &mut CS,
  a_val: &Integer,
  b_val: &Integer,
  m_val: &Integer,
  q_val: &Integer,
  r_val: &Integer,
  limb_width: usize,
  n_limbs: usize,
) -> Result<(), SynthesisError> {
  let a_num = Num::alloc(cs.namespace(|| "alloc a num"), || {
    Ok(nat_to_f(a_val).unwrap())
  })?;
  let a = BigNat::from_num(
    cs.namespace(|| "allocate a_limbs"),
    a_num,
    limb_width,
    n_limbs,
  )?;
  let b = BigNat::alloc_from_nat(
    cs.namespace(|| "b"),
    || Ok(b_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let m = BigNat::alloc_from_nat(
    cs.namespace(|| "m"),
    || Ok(m_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let q = BigNat::alloc_from_nat(
    cs.namespace(|| "q"),
    || Ok(q_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let r = BigNat::alloc_from_nat(
    cs.namespace(|| "r"),
    || Ok(r_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let _ = r.inputize(cs.namespace(|| "input r"))?;
  let (qa, ra) = a.mult_mod(cs.namespace(|| "prod"), &b, &m)?;
  qa.equal(cs.namespace(|| "qcheck"), &q)?;
  ra.equal(cs.namespace(|| "rcheck"), &r)?;
  Ok(())
}

fn synthesize_add<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
  cs: &mut CS,
  a_val: &Integer,
  b_val: &Integer,
  c_val: &Integer,
  limb_width: usize,
  n_limbs: usize,
) -> Result<(), SynthesisError> {
  let a = BigNat::alloc_from_nat(
    cs.namespace(|| "a"),
    || Ok(a_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let _ = a.inputize(cs.namespace(|| "input a"))?;
  let b = BigNat::alloc_from_nat(
    cs.namespace(|| "b"),
    || Ok(b_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let _ = b.inputize(cs.namespace(|| "input b"))?;
  let c = BigNat::alloc_from_nat(
    cs.namespace(|| "c"),
    || Ok(c_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let ca = a.add::<CS>(&b)?;
  ca.equal(cs.namespace(|| "ccheck"), &c)?;
  Ok(())
}

fn synthesize_add_mod<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
  cs: &mut CS,
  a_val: &Integer,
  b_val: &Integer,
  c_val: &Integer,
  m_val: &Integer,
  limb_width: usize,
  n_limbs: usize,
) -> Result<(), SynthesisError> {
  let a = BigNat::alloc_from_nat(
    cs.namespace(|| "a"),
    || Ok(a_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let _ = a.inputize(cs.namespace(|| "input a"))?;
  let b = BigNat::alloc_from_nat(
    cs.namespace(|| "b"),
    || Ok(b_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let _ = b.inputize(cs.namespace(|| "input b"))?;
  let c = BigNat::alloc_from_nat(
    cs.namespace(|| "c"),
    || Ok(c_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let m = BigNat::alloc_from_nat(
    cs.namespace(|| "m"),
    || Ok(m_val.clone()),
    limb_width,
    n_limbs,
  )?;
  let d = a.add::<CS>(&b)?;
  let ca = d.red_mod(cs.namespace(|| "reduce"), &m)?;
  ca.equal(cs.namespace(|| "ccheck"), &c)?;
  Ok(())
}

#[test]
fn test_mult_mod() {
  type G = pasta_curves::pallas::Point;

  //Set the inputs
  let a_val = Integer::from_str_radix(
    "11572336752428856981970994795408771577024165681374400871001196932361466228192",
    10,
  )
  .unwrap();
  let b_val = Integer::from_str_radix(
    "87673389408848523602668121701204553693362841135953267897017930941776218798802",
    10,
  )
  .unwrap();
  let m_val = Integer::from_str_radix(
    "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    16,
  )
  .unwrap();
  let q_val = Integer::from_str_radix(
    "35048542371029440058224000662033175648615707461806414787901284501179083518342",
    10,
  )
  .unwrap();
  let r_val = Integer::from_str_radix(
    "26362617993085418618858432307761590013874563896298265114483698919121453084730",
    10,
  )
  .unwrap();

  //First create the shape
  let mut cs: ShapeCS<G> = ShapeCS::new();
  let _ = synthesize_mult_mod(&mut cs, &a_val, &b_val, &m_val, &q_val, &r_val, 32, 8);
  let shape = cs.r1cs_shape();
  let gens = cs.r1cs_gens();
  println!("Mult mod constraint no: {}", cs.num_constraints());

  //Now get the assignment
  let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
  let _ = synthesize_mult_mod(&mut cs, &a_val, &b_val, &m_val, &q_val, &r_val, 32, 8);
  let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

  //Make sure that this is satisfiable
  assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}

#[test]
fn test_add() {
  type G = pasta_curves::pallas::Point;

  //Set the inputs
  let a_val = Integer::from_str_radix(
    "11572336752428856981970994795408771577024165681374400871001196932361466228192",
    10,
  )
  .unwrap();
  let b_val = Integer::from_str_radix("1", 10).unwrap();
  let c_val = Integer::from_str_radix(
    "11572336752428856981970994795408771577024165681374400871001196932361466228193",
    10,
  )
  .unwrap();

  //First create the shape
  let mut cs: ShapeCS<G> = ShapeCS::new();
  let _ = synthesize_add(&mut cs, &a_val, &b_val, &c_val, 32, 8);
  let shape = cs.r1cs_shape();
  let gens = cs.r1cs_gens();
  println!("Add mod constraint no: {}", cs.num_constraints());

  //Now get the assignment
  let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
  let _ = synthesize_add(&mut cs, &a_val, &b_val, &c_val, 32, 8);
  let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

  //Make sure that this is satisfiable
  assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}

#[test]
fn test_add_mod() {
  type G = pasta_curves::pallas::Point;

  //Set the inputs
  let a_val = Integer::from_str_radix(
    "11572336752428856981970994795408771577024165681374400871001196932361466228192",
    10,
  )
  .unwrap();
  let b_val = Integer::from_str_radix("1", 10).unwrap();
  let c_val = Integer::from_str_radix(
    "11572336752428856981970994795408771577024165681374400871001196932361466228193",
    10,
  )
  .unwrap();
  let m_val = Integer::from_str_radix(
    "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    16,
  )
  .unwrap();

  //First create the shape
  let mut cs: ShapeCS<G> = ShapeCS::new();
  let _ = synthesize_add_mod(&mut cs, &a_val, &b_val, &c_val, &m_val, 32, 8);
  let shape = cs.r1cs_shape();
  let gens = cs.r1cs_gens();
  println!("Add mod constraint no: {}", cs.num_constraints());

  //Now get the assignment
  let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
  let _ = synthesize_add_mod(&mut cs, &a_val, &b_val, &c_val, &m_val, 32, 8);
  let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

  //Make sure that this is satisfiable
  assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}

#[test]
fn test_equal() {
  type G = pasta_curves::pallas::Point;

  //Set the inputs
  let a_val = Integer::from_str_radix("1157233675242885698197099479540877", 10).unwrap();

  //First create the shape
  let mut cs: ShapeCS<G> = ShapeCS::new();
  let _ = synthesize_is_equal(&mut cs, &a_val, 32, 8);
  let shape = cs.r1cs_shape();
  let gens = cs.r1cs_gens();
  println!("Equal constraint no: {}", cs.num_constraints());

  //Now get the assignment
  let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
  let _ = synthesize_is_equal(&mut cs, &a_val, 32, 8);
  let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

  //Make sure that this is satisfiable
  assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
}
