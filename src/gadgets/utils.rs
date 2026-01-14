//! This module implements various low-level gadgets
use super::nonnative::bignat::{nat_to_limbs, BigNat};
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  frontend::{
    num::AllocatedNum, AllocatedBit, Assignment, Boolean, ConstraintSystem, LinearCombination,
    SynthesisError,
  },
  gadgets::nonnative::util::f_to_nat,
  traits::Engine,
};
use ff::{Field, PrimeField, PrimeFieldBits};
use num_bigint::BigInt;

/// Gets as input the little indian representation of a number and spits out the number
pub fn le_bits_to_num<Scalar, CS>(
  mut cs: CS,
  bits: &[AllocatedBit],
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  // We loop over the input bits and construct the constraint
  // and the field element that corresponds to the result
  let mut lc = LinearCombination::zero();
  let mut coeff = Scalar::ONE;
  let mut fe = Some(Scalar::ZERO);
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
pub fn alloc_zero<F: PrimeField, CS: ConstraintSystem<F>>(mut cs: CS) -> AllocatedNum<F> {
  let zero = AllocatedNum::alloc_infallible(cs.namespace(|| "alloc"), || F::ZERO);
  cs.enforce(
    || "check zero is valid",
    |lc| lc,
    |lc| lc,
    |lc| lc + zero.get_variable(),
  );
  zero
}

/// Allocate a variable that is set to one
pub fn alloc_one<F: PrimeField, CS: ConstraintSystem<F>>(mut cs: CS) -> AllocatedNum<F> {
  let one = AllocatedNum::alloc_infallible(cs.namespace(|| "alloc"), || F::ONE);
  cs.enforce(
    || "check one is valid",
    |lc| lc + CS::one(),
    |lc| lc + CS::one(),
    |lc| lc + one.get_variable(),
  );

  one
}

/// Allocate a scalar as a base. Only to be used is the scalar fits in base!
pub fn alloc_scalar_as_base<E, CS>(
  mut cs: CS,
  input: Option<E::Scalar>,
) -> Result<AllocatedNum<E::Base>, SynthesisError>
where
  E: Engine,
  <E as Engine>::Scalar: PrimeFieldBits,
  CS: ConstraintSystem<<E as Engine>::Base>,
{
  AllocatedNum::alloc(cs.namespace(|| "allocate scalar as base"), || {
    let input_bits = input.unwrap_or(E::Scalar::ZERO).clone().to_le_bits();
    let mut mult = E::Base::ONE;
    let mut val = E::Base::ZERO;
    for bit in input_bits {
      if bit {
        val += mult;
      }
      mult = mult + mult;
    }
    Ok(val)
  })
}

/// interpret scalar as base
/// Only to be used is the scalar fits in base!
pub fn scalar_as_base<E: Engine>(input: E::Scalar) -> E::Base {
  field_switch::<E::Scalar, E::Base>(input)
}

/// interpret base as scalar
/// Only to be used is the scalar fits in base!
pub fn base_as_scalar<E: Engine>(input: E::Base) -> E::Scalar {
  field_switch::<E::Base, E::Scalar>(input)
}

/// Switch between two fields
pub fn field_switch<F1: PrimeField + PrimeFieldBits, F2: PrimeField>(x: F1) -> F2 {
  let input_bits = x.to_le_bits();
  let mut mult = F2::ONE;
  let mut val = F2::ZERO;
  for bit in input_bits {
    if bit {
      val += mult;
    }
    mult += mult;
  }
  val
}

/// Provide a bignat representation of one field in another field
pub fn to_bignat_repr<F1: PrimeField + PrimeFieldBits, F2: PrimeField>(x: &F1) -> Vec<F2> {
  let limbs: Vec<F1> = nat_to_limbs(&f_to_nat(x), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
  limbs
    .iter()
    .map(|limb| field_switch::<F1, F2>(*limb))
    .collect()
}

/// Allocate bignat a constant
pub fn alloc_bignat_constant<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  val: &BigInt,
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
  (0..n_limbs).for_each(|i| {
    cs.enforce(
      || format!("check limb {i}"),
      |lc| lc + &bignat.limbs[i],
      |lc| lc + CS::one(),
      |lc| lc + (limbs[i], CS::one()),
    );
  });
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

  // Allocate t s.t. t=1 if z1 == z2 else 1/(z1 - z2)

  let t = AllocatedNum::alloc(cs.namespace(|| "t"), || {
    Ok(if *a.get_value().get()? == *b.get_value().get()? {
      F::ONE
    } else {
      (*a.get_value().get()? - *b.get_value().get()?)
        .invert()
        .unwrap()
    })
  })?;

  cs.enforce(
    || "t*(a - b) = 1 - r",
    |lc| lc + t.get_variable(),
    |lc| lc + a.get_variable() - b.get_variable(),
    |lc| lc + CS::one() - r.get_variable(),
  );

  cs.enforce(
    || "r*(a - b) = 0",
    |lc| lc + r.get_variable(),
    |lc| lc + a.get_variable() - b.get_variable(),
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
    |_| condition.lc(CS::one(), F::ONE),
    |lc| lc + c.get_variable() - b.get_variable(),
  );

  Ok(c)
}

/// If condition return a otherwise b
pub fn conditionally_select_vec<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &[AllocatedNum<F>],
  b: &[AllocatedNum<F>],
  condition: &Boolean,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
  a.iter()
    .zip(b.iter())
    .enumerate()
    .map(|(i, (a, b))| {
      conditionally_select(cs.namespace(|| format!("select_{i}")), a, b, condition)
    })
    .collect::<Result<Vec<AllocatedNum<F>>, SynthesisError>>()
}

/// If condition return a otherwise b where a and b are `BigNats`
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
      || format!("conditional select constraint {i}"),
      |lc| lc + &a.limbs[i] - &b.limbs[i],
      |_| condition.lc(CS::one(), F::ONE),
      |lc| lc + &c.limbs[i] - &b.limbs[i],
    );
  }
  Ok(c)
}

/// Same as the above but Condition is an `AllocatedNum` that needs to be
/// 0 or 1. 1 => True, 0 => False
pub fn conditionally_select2<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::ONE {
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

/// If condition set to 0 otherwise a. Condition is an allocated num
pub fn select_zero_or_num2<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::ONE {
      Ok(F::ZERO)
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

/// If condition set to a otherwise 0. Condition is an allocated num
pub fn select_num_or_zero2<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::ONE {
      Ok(*a.get_value().get()?)
    } else {
      Ok(F::ZERO)
    }
  })?;

  cs.enforce(
    || "conditional select constraint",
    |lc| lc + a.get_variable(),
    |lc| lc + condition.get_variable(),
    |lc| lc + c.get_variable(),
  );

  Ok(c)
}

/// If condition set to a otherwise 0
pub fn select_num_or_zero<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  condition: &Boolean,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? {
      Ok(*a.get_value().get()?)
    } else {
      Ok(F::ZERO)
    }
  })?;

  cs.enforce(
    || "conditional select constraint",
    |lc| lc + a.get_variable(),
    |_| condition.lc(CS::one(), F::ONE),
    |lc| lc + c.get_variable(),
  );

  Ok(c)
}

/// If condition set to 1 otherwise a
pub fn select_one_or_num2<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::ONE {
      Ok(F::ONE)
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

/// If condition set to 1 otherwise a - b
pub fn select_one_or_diff2<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  b: &AllocatedNum<F>,
  condition: &AllocatedNum<F>,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? == F::ONE {
      Ok(F::ONE)
    } else {
      Ok(*a.get_value().get()? - *b.get_value().get()?)
    }
  })?;

  cs.enforce(
    || "conditional select constraint",
    |lc| lc + CS::one() - a.get_variable() + b.get_variable(),
    |lc| lc + condition.get_variable(),
    |lc| lc + c.get_variable() - a.get_variable() + b.get_variable(),
  );
  Ok(c)
}

/// If condition set to a otherwise 1 for boolean conditions
pub fn select_num_or_one<F: PrimeField, CS: ConstraintSystem<F>>(
  mut cs: CS,
  a: &AllocatedNum<F>,
  condition: &Boolean,
) -> Result<AllocatedNum<F>, SynthesisError> {
  let c = AllocatedNum::alloc(cs.namespace(|| "conditional select result"), || {
    if *condition.get_value().get()? {
      Ok(*a.get_value().get()?)
    } else {
      Ok(F::ONE)
    }
  })?;

  cs.enforce(
    || "conditional select constraint",
    |lc| lc + a.get_variable() - CS::one(),
    |_| condition.lc(CS::one(), F::ONE),
    |lc| lc + c.get_variable() - CS::one(),
  );

  Ok(c)
}

/// Allocate a constant value in the circuit
pub fn alloc_constant<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  c: &Scalar,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
  let constant = AllocatedNum::alloc(cs.namespace(|| "constant"), || Ok(*c))?;

  cs.enforce(
    || "check eq given constant",
    |lc| lc + constant.get_variable(),
    |lc| lc + CS::one(),
    |lc| lc + (*c, CS::one()),
  );

  Ok(constant)
}

/// Convert a base field element to a BigInt representation.
pub fn base_as_bigint<E: Engine>(input: E::Base) -> BigInt {
  let input_bits = input.to_le_bits();
  let mut mult = BigInt::from(1);
  let mut val = BigInt::from(0);
  for bit in input_bits {
    if bit {
      val += &mult;
    }
    mult *= BigInt::from(2);
  }
  val
}

/// Gets as input the little endian representation of a number (as AllocatedNum bits)
/// and outputs the number.
///
/// Unlike le_bits_to_num which takes AllocatedBit, this takes AllocatedNum where
/// each num is constrained to be 0 or 1.
pub fn le_num_to_num<Scalar, CS>(
  mut cs: CS,
  bits: &[AllocatedNum<Scalar>],
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  // We loop over the input bits and construct the constraint
  // and the field element that corresponds to the result
  let mut lc = LinearCombination::zero();
  let mut coeff = Scalar::ONE;
  let mut fe = Some(Scalar::ZERO);
  for bit in bits.iter() {
    lc = lc + (coeff, bit.get_variable());
    fe = bit.get_value().map(|val| {
      if val == Scalar::ONE {
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

/// Compute acc_out = acc + c_i * v and c_i_out = c * c_i
///
/// This is used to fingerprint values in a circuit by computing a running sum.
pub fn fingerprint<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  acc: &AllocatedNum<Scalar>,
  c: &AllocatedNum<Scalar>,
  c_i: &AllocatedNum<Scalar>,
  v: &AllocatedNum<Scalar>,
) -> Result<(AllocatedNum<Scalar>, AllocatedNum<Scalar>), SynthesisError> {
  // we need to compute acc_out <- acc + c_i * v (the initial value of acc is zero and c_i = 1)
  // we then return acc_out and c_i_out = c_i * c

  // (1) we first compute acc_out <- acc + c_i * v
  let acc_out = AllocatedNum::alloc(cs.namespace(|| "acc_out"), || {
    Ok(*acc.get_value().get()? + *c_i.get_value().get()? * *v.get_value().get()?)
  })?;
  cs.enforce(
    || "acc_out = acc + c_i * v",
    |lc| lc + c_i.get_variable(),
    |lc| lc + v.get_variable(),
    |lc| lc + acc_out.get_variable() - acc.get_variable(),
  );

  // (2) we then compute c_i_out <- c * c^i
  let c_i_out = AllocatedNum::alloc(cs.namespace(|| "c_i_out"), || {
    Ok(*c_i.get_value().get()? * *c.get_value().get()?)
  })?;
  cs.enforce(
    || "c_i_out = c * c^i",
    |lc| lc + c.get_variable(),
    |lc| lc + c_i.get_variable(),
    |lc| lc + c_i_out.get_variable(),
  );

  Ok((acc_out, c_i_out))
}
