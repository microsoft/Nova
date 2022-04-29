//! This module implements various low-level gadgets
use crate::traits::Group;
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    Assignment,
  },
  ConstraintSystem, LinearCombination, SynthesisError,
};
use bellperson_nonnative::mp::bignat::{nat_to_limbs, BigNat};
use ff::{Field, PrimeField, PrimeFieldBits};
use rug::Integer;

/// Gets as input the little indian representation of a number and spits out the number
#[allow(dead_code)]
pub fn le_bits_to_num<Scalar, CS>(
  mut cs: CS,
  bits: Vec<AllocatedBit>,
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  // We loop over the input bits and construct the constraint
  // and the field element that corresponds to the result
  let mut lc = LinearCombination::zero();
  let mut coeff = Scalar::one();
  let mut fe = Some(Scalar::zero());
  for bit in bits.iter() {
    lc = lc + (coeff, bit.get_variable());
    fe = bit.get_value().map(|val| {
      if val {
        fe.unwrap() + coeff
      } else {
        fe.unwrap()
      }
    });
    coeff = coeff.double();
  }
  let num = AllocatedNum::alloc(cs.namespace(|| "Field element"), || {
    fe.ok_or(SynthesisError::AssignmentMissing)
  })?;
  lc = lc - num.get_variable();
  cs.enforce(|| "compute number from bits", |lc| lc, |lc| lc, |_| lc);
  Ok(num)
}

/// Allocate a variable that is set to zero
pub fn alloc_zero<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let zero = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(F::zero()))?;
  cs.enforce(
    || "check zero is valid",
    |lc| lc,
    |lc| lc,
    |lc| lc + zero.get_variable(),
  );
  Ok(zero)
}

/// Allocate a variable that is set to one
pub fn alloc_one<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let one = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(F::one()))?;
  cs.enforce(
    || "check one is valid",
    |lc| lc + CS::one(),
    |lc| lc + CS::one(),
    |lc| lc + one.get_variable(),
  );

  Ok(one)
}

/// Allocate a scalar as a base. Only to be used is the scalar fits in base!
pub fn alloc_scalar_as_base<G, CS>(
  mut cs: CS,
  input: Option<G::Scalar>,
) -> Result<AllocatedNum<G::Base>, SynthesisError>
where
  G: Group,
  <G as Group>::Scalar: PrimeFieldBits,
  CS: ConstraintSystem<<G as Group>::Base>,
{
  AllocatedNum::alloc(cs.namespace(|| "allocate scalar as base"), || {
    let input_bits = input.get()?.clone().to_le_bits();
    let mut mult = G::Base::one();
    let mut val = G::Base::zero();
    for bit in input_bits {
      if bit {
        val += mult;
      }
      mult = mult + mult;
    }
    Ok(val)
  })
}

/// Allocate bignat a constant
pub fn alloc_bignat_constant<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  val: &Integer,
  limb_width: usize,
  n_limbs: usize,
) -> Result<BigNat<F>, SynthesisError> {
  let limbs = nat_to_limbs(val, limb_width, n_limbs).unwrap();
  let bignat = BigNat::alloc_from_limbs(
    cs.namespace(|| "alloc bignat"),
    || Ok(limbs.clone()),
    None,
    limb_width,
    n_limbs,
  )?;
  // Now enforce that the limbs are all equal to the constants
  #[allow(clippy::needless_range_loop)]
  for i in 0..n_limbs {
    cs.enforce(
      || format!("check limb {}", i),
      |lc| lc + &bignat.limbs[i],
      |lc| lc + CS::one(),
      |lc| lc + (limbs[i], CS::one()),
    );
  }
  Ok(bignat)
}

/// Check that two numbers are equal and return a bit
pub fn alloc_num_equals<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
) -> Result<AllocatedBit, SynthesisError> {
  // Allocate and constrain `r`: result boolean bit.
  // It equals `true` if `a` equals `b`, `false` otherwise
  let r_value = match (a.get_value(), b.get_value()) {
    (Some(a), Some(b)) => Some(a == b),
    _ => None,
  };

  let r = AllocatedBit::alloc(cs.namespace(|| "r"), r_value)?;

  let delta = AllocatedNum::alloc(cs.namespace(|| "delta"), || {
    let a_value = *a.get_value().get()?;
    let b_value = *b.get_value().get()?;

    let mut delta = a_value;
    delta.sub_assign(&b_value);

    Ok(delta)
  })?;

  cs.enforce(
    || "delta = (a - b)",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + delta.get_variable(),
  );

  let delta_inv = AllocatedNum::alloc(cs.namespace(|| "delta_inv"), || {
    let delta = *delta.get_value().get()?;

    if delta.is_zero().unwrap_u8() == 1 {
      Ok(F::one()) // we can return any number here, it doesn't matter
    } else {
      Ok(delta.invert().unwrap())
    }
  })?;

  // Allocate `t = delta * delta_inv`
  // If `delta` is non-zero (a != b), `t` will equal 1
  // If `delta` is zero (a == b), `t` cannot equal 1
  let t = AllocatedNum::alloc(cs.namespace(|| "t"), || {
    let mut tmp = *delta.get_value().get()?;
    tmp.mul_assign(&(*delta_inv.get_value().get()?));

    Ok(tmp)
  })?;

  // Constrain allocation:
  // t = (a - b) * delta_inv
  cs.enforce(
    || "t = (a - b) * delta_inv",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + delta_inv.get_variable(),
    |lc| lc + t.get_variable(),
  );

  // Constrain:
  // (a - b) * (t - 1) == 0
  // This enforces that correct `delta_inv` was provided,
  // and thus `t` is 1 if `(a - b)` is non zero (a != b )
  cs.enforce(
    || "(a - b) * (t - 1) == 0",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + t.get_variable() - CS::one(),
    |lc| lc,
  );

  // Constrain:
  // (a - b) * r == 0
  // This enforces that `r` is zero if `(a - b)` is non-zero (a != b)
  cs.enforce(
    || "(a - b) * r == 0",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + r.get_variable(),
    |lc| lc,
  );

  // Constrain:
  // (t - 1) * (r - 1) == 0
  // This enforces that `r` is one if `t` is not one (a == b)
  cs.enforce(
    || "(t - 1) * (r - 1) == 0",
    |lc| lc + t.get_variable() - CS::one(),
    |lc| lc + r.get_variable() - CS::one(),
    |lc| lc,
  );

  Ok(r)
}

/// If condition return a otherwise b
pub fn conditionally_select<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  condition: &Boolean,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? {
      Ok(*a.get_value().get()?)
    } else {
      Ok(*b.get_value().get()?)
    }
  })?;

  // a * condition + b*(1-condition) = c ->
  // a * condition - b*condition = c - b
  cs.enforce(
    || "conditional select constraint",
    |lc| lc + a.get_variable() - b.get_variable(),
    |_| condition.lc(CS::one(), F::one()),
    |lc| lc + c.get_variable() - b.get_variable(),
  );

  Ok(c)
}

/// If condition return a otherwise b where a and b are BigNats
pub fn conditionally_select_bignat<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &BigNat<F>,
  b: &BigNat<F>,
  condition: &Boolean,
) -> Result<BigNat<F>, SynthesisError> {
  assert!(a.limbs.len() == b.limbs.len());
  let c = BigNat::alloc_from_nat(
    cs.namespace(|| "conditional select result"),
    || {
      if *condition.get_value().get()? {
        Ok(a.value.get()?.clone())
      } else {
        Ok(b.value.get()?.clone())
      }
    },
    a.params.limb_width,
    a.params.n_limbs,
  )?;

  // a * condition + b*(1-condition) = c ->
  // a * condition - b*condition = c - b
  for i in 0..c.limbs.len() {
    cs.enforce(
      || format!("conditional select constraint {}", i),
      |lc| lc + &a.limbs[i] - &b.limbs[i],
      |_| condition.lc(CS::one(), F::one()),
      |lc| lc + &c.limbs[i] - &b.limbs[i],
    );
  }
  Ok(c)
}

/// Same as the above but Condition is an AllocatedNum that needs to be
/// 0 or 1. 1 => True, 0 => False
pub fn conditionally_select2<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::one() {
      Ok(*a.get_value().get()?)
    } else {
      Ok(*b.get_value().get()?)
    }
  })?;

  // a * condition + b*(1-condition) = c ->
  // a * condition - b*condition = c - b
  cs.enforce(
    || "conditional select constraint",
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + condition.get_variable(),
    |lc| lc + c.get_variable() - b.get_variable(),
  );

  Ok(c)
}

/// If condition set to 0 otherwise a
pub fn select_zero_or<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::one() {
      Ok(F::zero())
    } else {
      Ok(*a.get_value().get()?)
    }
  })?;

  // a * (1 - condition) = c
  cs.enforce(
    || "conditional select constraint",
    |lc| lc + a.get_variable(),
    |lc| lc + CS::one() - condition.get_variable(),
    |lc| lc + c.get_variable(),
  );

  Ok(c)
}

/// If condition set to 1 otherwise a
pub fn select_one_or<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::one() {
      Ok(F::one())
    } else {
      Ok(*a.get_value().get()?)
    }
  })?;

  cs.enforce(
    || "conditional select constraint",
    |lc| lc + CS::one() - a.get_variable(),
    |lc| lc + condition.get_variable(),
    |lc| lc + c.get_variable() - a.get_variable(),
  );
  Ok(c)
}
