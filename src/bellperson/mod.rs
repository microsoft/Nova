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
  use bellperson_nonnative::{
    mp::bignat::BigNat,
    util::{convert::nat_to_f, num::Num},
  };
  use ff::PrimeField;
  use num_bigint::BigInt;
  use num_traits::Num as OtherNum;

  fn synthesize_alloc_bit<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    // get two bits as input and check that they are indeed bits
    let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::one()))?;
    a.inputize(cs.namespace(|| "a is input"));
    cs.enforce(
      || "check a is 0 or 1",
      |lc| lc + CS::one() - a.get_variable(),
      |lc| lc + a.get_variable(),
      |lc| lc,
    );
    let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::one()))?;
    b.inputize(cs.namespace(|| "b is input"));
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
    synthesize_alloc_bit(&mut cs);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_alloc_bit(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

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
    a.inputize(cs.namespace(|| "a is input"));
    b.inputize(cs.namespace(|| "b is input"));
    Ok(())
  }

  fn synthesize_use_cs_one_after_inputize<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    let a = AllocatedNum::alloc(cs.namespace(|| "a"), || Ok(Fr::one()))?;
    let b = AllocatedNum::alloc(cs.namespace(|| "b"), || Ok(Fr::one()))?;
    a.inputize(cs.namespace(|| "a is input"));
    cs.enforce(
      || "check a = b",
      |lc| lc + a.get_variable() - b.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );
    b.inputize(cs.namespace(|| "b is input"));
    Ok(())
  }

  #[test]
  fn test_use_cs_one() {
    type G = pasta_curves::pallas::Point;

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    synthesize_use_cs_one(&mut cs);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_use_cs_one(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

  #[test]
  fn test_use_cs_one_after_inputize() {
    type G = pasta_curves::pallas::Point;

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    synthesize_use_cs_one_after_inputize(&mut cs);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_use_cs_one_after_inputize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

  fn synthesize_is_equal<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    a_val: &BigInt,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<(), SynthesisError> {
    let a1 = BigNat::alloc_from_nat(
      cs.namespace(|| "alloc a2"),
      || Ok(a_val.clone()),
      limb_width,
      n_limbs,
    )?;
    a1.inputize(cs.namespace(|| "make a input"));

    let a_num = Num::alloc(cs.namespace(|| "alloc a num"), || {
      Ok(nat_to_f(a_val).unwrap())
    })?;
    let a2 = BigNat::from_num(
      cs.namespace(|| "allocate a1_limbs"),
      a_num,
      limb_width,
      n_limbs,
    )?;

    a1.equal_when_carried(cs.namespace(|| "check equal"), &a2)?;
    Ok(())
  }

  #[allow(clippy::too_many_arguments)]
  fn synthesize_mult_mod<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    a_val: &BigInt,
    b_val: &BigInt,
    m_val: &BigInt,
    q_val: &BigInt,
    r_val: &BigInt,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<(), SynthesisError> {
    let a_num = Num::alloc(cs.namespace(|| "alloc a num"), || {
      Ok(nat_to_f(a_val).unwrap())
    })?;
    let m = BigNat::alloc_from_nat(
      cs.namespace(|| "m"),
      || Ok(m_val.clone()),
      limb_width,
      n_limbs,
    )?;
    m.inputize(cs.namespace(|| "modulus m"))?;

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
    let (qa, ra) = a.mult_mod(cs.namespace(|| "prod"), &b, &m)?;
    qa.equal(cs.namespace(|| "qcheck"), &q)?;
    ra.equal(cs.namespace(|| "rcheck"), &r)?;
    Ok(())
  }

  fn synthesize_add<Fr: PrimeField, CS: ConstraintSystem<Fr>>(
    cs: &mut CS,
    a_val: &BigInt,
    b_val: &BigInt,
    c_val: &BigInt,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<(), SynthesisError> {
    let a = BigNat::alloc_from_nat(
      cs.namespace(|| "a"),
      || Ok(a_val.clone()),
      limb_width,
      n_limbs,
    )?;
    a.inputize(cs.namespace(|| "input a"))?;
    let b = BigNat::alloc_from_nat(
      cs.namespace(|| "b"),
      || Ok(b_val.clone()),
      limb_width,
      n_limbs,
    )?;
    b.inputize(cs.namespace(|| "input b"))?;
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
    a_val: &BigInt,
    b_val: &BigInt,
    c_val: &BigInt,
    m_val: &BigInt,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<(), SynthesisError> {
    let a = BigNat::alloc_from_nat(
      cs.namespace(|| "a"),
      || Ok(a_val.clone()),
      limb_width,
      n_limbs,
    )?;
    a.inputize(cs.namespace(|| "input a"))?;
    let b = BigNat::alloc_from_nat(
      cs.namespace(|| "b"),
      || Ok(b_val.clone()),
      limb_width,
      n_limbs,
    )?;
    b.inputize(cs.namespace(|| "input b"))?;
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

    // Set the inputs
    let a_val = BigInt::from_str_radix(
      "11572336752428856981970994795408771577024165681374400871001196932361466228192",
      10,
    )
    .unwrap();
    let b_val = BigInt::from_str_radix(
      "87673389408848523602668121701204553693362841135953267897017930941776218798802",
      10,
    )
    .unwrap();
    let m_val = BigInt::from_str_radix(
      "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
      16,
    )
    .unwrap();
    let q_val = BigInt::from_str_radix(
      "35048542371029440058224000662033175648615707461806414787901284501179083518342",
      10,
    )
    .unwrap();
    let r_val = BigInt::from_str_radix(
      "26362617993085418618858432307761590013874563896298265114483698919121453084730",
      10,
    )
    .unwrap();

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    synthesize_mult_mod(&mut cs, &a_val, &b_val, &m_val, &q_val, &r_val, 32, 8);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();
    println!("Mult mod constraint no: {}", cs.num_constraints());

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_mult_mod(&mut cs, &a_val, &b_val, &m_val, &q_val, &r_val, 32, 8);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

  #[test]
  fn test_add() {
    type G = pasta_curves::pallas::Point;

    // Set the inputs
    let a_val = BigInt::from_str_radix(
      "11572336752428856981970994795408771577024165681374400871001196932361466228192",
      10,
    )
    .unwrap();
    let b_val = BigInt::from_str_radix("1", 10).unwrap();
    let c_val = BigInt::from_str_radix(
      "11572336752428856981970994795408771577024165681374400871001196932361466228193",
      10,
    )
    .unwrap();

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    synthesize_add(&mut cs, &a_val, &b_val, &c_val, 64, 4);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();
    println!("Add mod constraint no: {}", cs.num_constraints());

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_add(&mut cs, &a_val, &b_val, &c_val, 64, 4);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

  #[test]
  fn test_add_mod() {
    type G = pasta_curves::pallas::Point;

    // Set the inputs
    let a_val = BigInt::from_str_radix(
      "11572336752428856981970994795408771577024165681374400871001196932361466228192",
      10,
    )
    .unwrap();
    let b_val = BigInt::from_str_radix("1", 10).unwrap();
    let c_val = BigInt::from_str_radix(
      "11572336752428856981970994795408771577024165681374400871001196932361466228193",
      10,
    )
    .unwrap();
    let m_val = BigInt::from_str_radix(
      "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
      16,
    )
    .unwrap();

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    synthesize_add_mod(&mut cs, &a_val, &b_val, &c_val, &m_val, 32, 8);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();
    println!("Add mod constraint no: {}", cs.num_constraints());

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_add_mod(&mut cs, &a_val, &b_val, &c_val, &m_val, 32, 8);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

  #[test]
  fn test_equal() {
    type G = pasta_curves::pallas::Point;

    // Set the inputs
    let a_val = BigInt::from_str_radix("1157233675242885698197099479540877", 10).unwrap();

    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    synthesize_is_equal(&mut cs, &a_val, 32, 8);
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();
    println!("Equal constraint no: {}", cs.num_constraints());

    // Now get the assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    synthesize_is_equal(&mut cs, &a_val, 32, 8);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }
}
