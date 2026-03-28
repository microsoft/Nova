use super::OptionExt;
use crate::frontend::{
  num::AllocatedNum, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use ff::PrimeField;
use num_bigint::{BigInt, Sign};
use std::convert::From;
#[derive(Clone)]
/// A representation of a bit
pub struct Bit<Scalar: PrimeField> {
  /// The linear combination which constrains the value of the bit
  pub bit: LinearCombination<Scalar>,
}

#[derive(Clone)]
/// A representation of a bit-vector
pub struct Bitvector<Scalar: PrimeField> {
  /// The linear combination which constrains the values of the bits
  pub bits: Vec<LinearCombination<Scalar>>,
  /// The value of the bits (filled at witness-time)
  pub values: Option<Vec<bool>>,
  /// Allocated bit variables
  pub allocations: Vec<Bit<Scalar>>,
}

impl<Scalar: PrimeField> Bit<Scalar> {
  /// Allocate a variable in the constraint system which can only be a
  /// boolean value.
  pub fn alloc<CS: ConstraintSystem<Scalar>>(
    mut cs: CS,
    value: Option<bool>,
  ) -> Result<Self, SynthesisError> {
    let var = cs.alloc(
      || "boolean",
      || {
        if *value.grab()? {
          Ok(Scalar::ONE)
        } else {
          Ok(Scalar::ZERO)
        }
      },
    )?;

    // Constrain: (1 - a) * a = 0
    // This constrains a to be either 0 or 1.
    cs.enforce(
      || "boolean constraint",
      |lc| lc + CS::one() - var,
      |lc| lc + var,
      |lc| lc,
    );

    Ok(Self {
      bit: LinearCombination::zero() + var,
    })
  }
}

/// A representation of a field element as a linear combination with an optional value.
pub struct Num<Scalar: PrimeField> {
  /// The linear combination representing the number.
  pub num: LinearCombination<Scalar>,
  /// The value of the number (filled at witness-time).
  pub value: Option<Scalar>,
}

impl<Scalar: PrimeField> Num<Scalar> {
  /// Creates a new `Num` with the given value and linear combination.
  pub const fn new(value: Option<Scalar>, num: LinearCombination<Scalar>) -> Self {
    Self { value, num }
  }
  /// Allocates a new `Num` in the constraint system with the given value.
  pub fn alloc<CS, F>(mut cs: CS, value: F) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    F: FnOnce() -> Result<Scalar, SynthesisError>,
  {
    let mut new_value = None;
    let var = cs.alloc(
      || "num",
      || {
        let tmp = value()?;

        new_value = Some(tmp);

        Ok(tmp)
      },
    )?;

    Ok(Num {
      value: new_value,
      num: LinearCombination::zero() + var,
    })
  }

  /// Checks that the `Num` fits in the given number of bits.
  pub fn fits_in_bits<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    n_bits: usize,
  ) -> Result<(), SynthesisError> {
    let v = self.value;

    // Fast witness path: batch-allocate all bit variables at once
    if cs.is_witness_generator() {
      let bit_values: Vec<Scalar> = if let Some(val) = v {
        let repr = val.to_repr();
        let bytes = repr.as_ref();
        (1..n_bits)
          .map(|i| {
            let (byte_pos, bit_pos) = (i / 8, i % 8);
            if byte_pos < bytes.len() && (bytes[byte_pos] >> bit_pos) & 1 == 1 {
              Scalar::ONE
            } else {
              Scalar::ZERO
            }
          })
          .collect()
      } else {
        vec![Scalar::ZERO; n_bits - 1]
      };
      cs.extend_aux(&bit_values);
      return Ok(());
    }

    // Pre-compute all bit values from the field element's byte representation
    // to avoid calling to_repr() per bit (which does Montgomery reduction each time).
    let bit_values: Option<Vec<bool>> = v.map(|val| {
      let repr = val.to_repr();
      let bytes = repr.as_ref();
      (0..n_bits)
        .map(|i| {
          let (byte_pos, bit_pos) = (i / 8, i % 8);
          if byte_pos < bytes.len() {
            (bytes[byte_pos] >> bit_pos) & 1 == 1
          } else {
            false
          }
        })
        .collect()
    });

    // Allocate all but the first bit.
    let bits: Vec<Variable> = (1..n_bits)
      .map(|i| {
        cs.alloc(
          || format!("bit {i}"),
          || {
            let r = if bit_values.as_ref().ok_or(SynthesisError::AssignmentMissing)?[i] {
              Scalar::ONE
            } else {
              Scalar::ZERO
            };
            Ok(r)
          },
        )
      })
      .collect::<Result<_, _>>()?;

    for (i, v) in bits.iter().enumerate() {
      cs.enforce(
        || format!("{i} is bit"),
        |lc| lc + *v,
        |lc| lc + CS::one() - *v,
        |lc| lc,
      )
    }

    // Last bit
    cs.enforce(
      || "last bit",
      |mut lc| {
        let mut f = Scalar::ONE;
        lc = lc + &self.num;
        for v in bits.iter() {
          f = f.double();
          lc = lc - (f, *v);
        }
        lc
      },
      |mut lc| {
        lc = lc + CS::one();
        let mut f = Scalar::ONE;
        lc = lc - &self.num;
        for v in bits.iter() {
          f = f.double();
          lc = lc + (f, *v);
        }
        lc
      },
      |lc| lc,
    );
    Ok(())
  }

  /// Computes the natural number represented by an array of bits.
  /// Checks if the natural number equals `self`
  pub fn is_equal<CS: ConstraintSystem<Scalar>>(&self, mut cs: CS, other: &Bitvector<Scalar>) {
    let allocations = other.allocations.clone();
    let mut f = Scalar::ONE;
    let sum = allocations
      .iter()
      .fold(LinearCombination::zero(), |lc, bit| {
        let l = lc + (f, &bit.bit);
        f = f.double();
        l
      });
    let sum_lc = LinearCombination::zero() + &self.num - &sum;
    cs.enforce(|| "sum", |lc| lc + &sum_lc, |lc| lc + CS::one(), |lc| lc);
  }

  /// Compute the natural number represented by an array of limbs.
  /// The limbs are assumed to be based the `limb_width` power of 2.
  /// Low-index bits are low-order
  pub fn decompose<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    n_bits: usize,
  ) -> Result<Bitvector<Scalar>, SynthesisError> {
    // Pre-compute all bit values with a single to_repr() call
    let values: Option<Vec<bool>> = self.value.as_ref().map(|v| {
      let repr = v.to_repr();
      let bytes = repr.as_ref();
      (0..n_bits)
        .map(|i| {
          let (byte_pos, bit_pos) = (i / 8, i % 8);
          if byte_pos < bytes.len() {
            (bytes[byte_pos] >> bit_pos) & 1 == 1
          } else {
            false
          }
        })
        .collect()
    });

    // Fast witness path: batch-allocate all bit variables
    if cs.is_witness_generator() {
      let field_vals: Vec<Scalar> = if let Some(ref bv) = values {
        bv.iter()
          .map(|&b| if b { Scalar::ONE } else { Scalar::ZERO })
          .collect()
      } else {
        vec![Scalar::ZERO; n_bits]
      };
      let base_idx = cs.aux_slice().len();
      cs.extend_aux(&field_vals);

      let allocations: Vec<Bit<Scalar>> = (0..n_bits)
        .map(|i| {
          let var = Variable::new_unchecked(Index::Aux(base_idx + i));
          Bit {
            bit: LinearCombination::zero() + var,
          }
        })
        .collect();
      let bits: Vec<LinearCombination<Scalar>> = allocations
        .iter()
        .map(|a| LinearCombination::zero() + &a.bit)
        .collect();
      return Ok(Bitvector {
        allocations,
        values,
        bits,
      });
    }

    let allocations: Vec<Bit<Scalar>> = (0..n_bits)
      .map(|bit_i| {
        Bit::alloc(
          cs.namespace(|| format!("bit{bit_i}")),
          values.as_ref().map(|vs| vs[bit_i]),
        )
      })
      .collect::<Result<Vec<_>, _>>()?;
    let mut f = Scalar::ONE;
    let sum = allocations
      .iter()
      .fold(LinearCombination::zero(), |lc, bit| {
        let l = lc + (f, &bit.bit);
        f = f.double();
        l
      });
    let sum_lc = LinearCombination::zero() + &self.num - &sum;
    cs.enforce(|| "sum", |lc| lc + &sum_lc, |lc| lc + CS::one(), |lc| lc);
    let bits: Vec<LinearCombination<Scalar>> = allocations
      .clone()
      .into_iter()
      .map(|a| LinearCombination::zero() + &a.bit)
      .collect();
    Ok(Bitvector {
      allocations,
      values,
      bits,
    })
  }

  /// Allocates bit decomposition variables for range checking only.
  ///
  /// In witness mode, skips `LinearCombination` and `Bit` construction since
  /// range-check callers discard the returned `Bitvector`. Falls back to
  /// full `decompose` in constraint-generation mode.
  pub fn decompose_for_range_check<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    n_bits: usize,
  ) -> Result<(), SynthesisError> {
    if cs.is_witness_generator() {
      let field_vals: Vec<Scalar> = if let Some(v) = self.value.as_ref() {
        let repr = v.to_repr();
        let bytes = repr.as_ref();
        (0..n_bits)
          .map(|i| {
            let (byte_pos, bit_pos) = (i / 8, i % 8);
            if byte_pos < bytes.len() && (bytes[byte_pos] >> bit_pos) & 1 == 1 {
              Scalar::ONE
            } else {
              Scalar::ZERO
            }
          })
          .collect()
      } else {
        vec![Scalar::ZERO; n_bits]
      };
      cs.extend_aux(&field_vals);
      return Ok(());
    }
    // Full decompose path for constraint generation
    self.decompose(cs, n_bits)?;
    Ok(())
  }

  /// Converts the `Num` to an `AllocatedNum` in the constraint system.
  pub fn as_allocated_num<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    let new = AllocatedNum::alloc(cs.namespace(|| "alloc"), || Ok(*self.value.grab()?))?;
    cs.enforce(
      || "eq",
      |lc| lc,
      |lc| lc,
      |lc| lc + new.get_variable() - &self.num,
    );
    Ok(new)
  }
}

impl<Scalar: PrimeField> From<AllocatedNum<Scalar>> for Num<Scalar> {
  fn from(a: AllocatedNum<Scalar>) -> Self {
    Self::new(a.get_value(), LinearCombination::zero() + a.get_variable())
  }
}

/// Convert a field element to a natural number
pub fn f_to_nat<Scalar: PrimeField>(f: &Scalar) -> BigInt {
  BigInt::from_bytes_le(Sign::Plus, f.to_repr().as_ref())
}

/// Convert a natural number to a field element.
/// Returns `None` if the number is too big for the field.
pub fn nat_to_f<Scalar: PrimeField>(n: &BigInt) -> Option<Scalar> {
  let (sign, bytes) = n.to_bytes_le();
  if sign == Sign::Minus {
    return None;
  }
  let mut repr = Scalar::Repr::default();
  let repr_bytes = repr.as_mut();
  if bytes.len() > repr_bytes.len() {
    return None;
  }
  repr_bytes[..bytes.len()].copy_from_slice(&bytes);
  Scalar::from_repr(repr).into()
}

use super::bignat::BigNat;
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  gadgets::utils::fingerprint,
  traits::{Engine, Group, ROCircuitTrait, ROTrait},
};

/// Get the base field modulus as a BigInt.
pub fn get_base_modulus<E: Engine>() -> BigInt {
  E::GE::group_params().3
}

/// Absorb a BigNat into a random oracle circuit (base field version).
pub fn absorb_bignat_in_ro<E: Engine, CS: ConstraintSystem<E::Base>>(
  n: &BigNat<E::Base>,
  mut cs: CS,
  ro: &mut E::ROCircuit,
) -> Result<(), SynthesisError> {
  let limbs = n
    .as_limbs()
    .iter()
    .enumerate()
    .map(|(i, limb)| limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of num"))))
    .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

  // absorb each limb directly (no packing - limbs are not constrained to be small)
  for limb in limbs {
    ro.absorb(&limb);
  }

  Ok(())
}

/// Absorb a BigNat into a random oracle circuit (scalar field version).
pub fn absorb_bignat_in_ro_scalar<E: Engine, CS: ConstraintSystem<E::Scalar>>(
  n: &BigNat<E::Scalar>,
  mut cs: CS,
  ro: &mut E::RO2Circuit,
) -> Result<(), SynthesisError> {
  let limbs = n
    .as_limbs()
    .iter()
    .enumerate()
    .map(|(i, limb)| limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of num"))))
    .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;

  // absorb each limb directly (no packing - limbs are not constrained to be small)
  for limb in limbs {
    ro.absorb(&limb);
  }
  Ok(())
}

/// Absorb a scalar field element into a random oracle (native version).
pub fn absorb_bignat_in_ro_native<E: Engine>(
  e: &E::Scalar,
  ro: &mut E::RO,
) -> Result<(), SynthesisError> {
  use super::bignat::nat_to_limbs;
  // absorb each element of x in bignum format
  let limbs: Vec<E::Base> = nat_to_limbs(&f_to_nat(e), BN_LIMB_WIDTH, BN_N_LIMBS)?;
  // absorb each limb directly (no packing - limbs are not constrained to be small)
  for limb in limbs {
    ro.absorb(limb);
  }
  Ok(())
}

/// Fingerprint a BigNat by fingerprinting each of its limbs.
pub fn fingerprint_bignat<E: Engine, CS: ConstraintSystem<E::Base>>(
  mut cs: CS,
  acc: &AllocatedNum<E::Base>,
  c: &AllocatedNum<E::Base>,
  c_i: &AllocatedNum<E::Base>,
  bn: &BigNat<E::Base>,
) -> Result<(AllocatedNum<E::Base>, AllocatedNum<E::Base>), SynthesisError> {
  // Analyze bignat as limbs
  let limbs = bn
    .as_limbs()
    .iter()
    .enumerate()
    .map(|(i, limb)| {
      limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of x to num")))
    })
    .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

  // fingerprint the limbs
  let mut acc_out = acc.clone();
  let mut c_i_out = c_i.clone();
  for (i, limb) in limbs.iter().enumerate() {
    (acc_out, c_i_out) = fingerprint::<E::Base, _>(
      cs.namespace(|| format!("output limb_{i}")),
      &acc_out,
      c,
      &c_i_out,
      limb,
    )?;
  }

  Ok((acc_out, c_i_out))
}
