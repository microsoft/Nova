use bellperson::{ConstraintSystem, gadgets::{boolean::{AllocatedBit, Boolean}, num::AllocatedNum, Assignment}, SynthesisError};
use ff::PrimeField;

//The next two functions are borrowed from sapling-crypto crate
pub fn alloc_num_equals<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: AllocatedNum<F>,
  b: AllocatedNum<F>,
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

  //
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

pub fn conditionally_select<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  condition: &Boolean,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c =
    AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
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
