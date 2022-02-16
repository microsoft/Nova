use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    Assignment,
  },
  ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::{PrimeField, PrimeFieldBits};

///Gets as input the little indian representation of a number and spits out the number
#[allow(dead_code)]
pub fn le_bits_to_num<Scalar, CS>(
  mut cs: CS,
  bits: Vec<AllocatedBit>,
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  //We loop over the input bits and construct the constraint and the field element that corresponds
  //to the result
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

///Allocate a variable that is set to zero
pub fn alloc_zero<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let zero = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(F::zero()))?;
  //TODO: How do we enforce that it is zero?
  //cs.enforce(
  //  || "check zero is valid",
  //  |lc| lc + zero.get_variable(),
  //  |lc| lc + CS::one(),
  //  |lc| lc ,
  //);

  Ok(zero)
}

///Allocate a variable that is set to one
pub fn alloc_one<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let one = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(F::one()))?;
  //TODO: How to enforce that it is one?
  //cs.enforce(
  //  || "check one is valid",
  //  |lc| lc,
  //  |lc| lc,
  //  |lc| lc + one.get_variable() - CS::one(),
  //);

  Ok(one)
}

//The next two functions are borrowed from sapling-crypto crate

///Check that two numbers are equal and return a bit
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

///If condition return a otherwise b
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

///Same as the above but Condition is an AllocatedNum that needs to be 
///0 or 1. 1 => True, 0 => False
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
