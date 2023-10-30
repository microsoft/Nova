use super::{
  util::{
    Bitvector, Num, {f_to_nat, nat_to_f},
  },
  OptionExt,
};
use bellpepper_core::{ConstraintSystem, LinearCombination, SynthesisError};
use ff::PrimeField;
use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;
use std::borrow::Borrow;
use std::cmp::{max, min};
use std::convert::From;

/// Compute the natural number represented by an array of limbs.
/// The limbs are assumed to be based the `limb_width` power of 2.
pub fn limbs_to_nat<Scalar: PrimeField, B: Borrow<Scalar>, I: DoubleEndedIterator<Item = B>>(
  limbs: I,
  limb_width: usize,
) -> BigInt {
  limbs.rev().fold(BigInt::from(0), |mut acc, limb| {
    acc <<= limb_width as u32;
    acc += f_to_nat(limb.borrow());
    acc
  })
}

fn int_with_n_ones(n: usize) -> BigInt {
  let mut m = BigInt::from(1);
  m <<= n as u32;
  m -= 1;
  m
}

/// Compute the limbs encoding a natural number.
/// The limbs are assumed to be based the `limb_width` power of 2.
pub fn nat_to_limbs<Scalar: PrimeField>(
  nat: &BigInt,
  limb_width: usize,
  n_limbs: usize,
) -> Result<Vec<Scalar>, SynthesisError> {
  let mask = int_with_n_ones(limb_width);
  let mut nat = nat.clone();
  if nat.bits() as usize <= n_limbs * limb_width {
    Ok(
      (0..n_limbs)
        .map(|_| {
          let r = &nat & &mask;
          nat >>= limb_width as u32;
          nat_to_f(&r).unwrap()
        })
        .collect(),
    )
  } else {
    eprintln!("nat {nat} does not fit in {n_limbs} limbs of width {limb_width}");
    Err(SynthesisError::Unsatisfiable)
  }
}

#[derive(Clone, PartialEq, Eq)]
pub struct BigNatParams {
  pub min_bits: usize,
  pub max_word: BigInt,
  pub limb_width: usize,
  pub n_limbs: usize,
}

impl BigNatParams {
  pub fn new(limb_width: usize, n_limbs: usize) -> Self {
    let mut max_word = BigInt::from(1) << limb_width as u32;
    max_word -= 1;
    BigNatParams {
      max_word,
      n_limbs,
      limb_width,
      min_bits: 0,
    }
  }
}

/// A representation of a large natural number (a member of {0, 1, 2, ... })
#[derive(Clone)]
pub struct BigNat<Scalar: PrimeField> {
  /// The linear combinations which constrain the value of each limb of the number
  pub limbs: Vec<LinearCombination<Scalar>>,
  /// The witness values for each limb (filled at witness-time)
  pub limb_values: Option<Vec<Scalar>>,
  /// The value of the whole number (filled at witness-time)
  pub value: Option<BigInt>,
  /// Parameters
  pub params: BigNatParams,
}

impl<Scalar: PrimeField> PartialEq for BigNat<Scalar> {
  fn eq(&self, other: &Self) -> bool {
    self.value == other.value && self.params == other.params
  }
}
impl<Scalar: PrimeField> Eq for BigNat<Scalar> {}

impl<Scalar: PrimeField> From<BigNat<Scalar>> for Polynomial<Scalar> {
  fn from(other: BigNat<Scalar>) -> Polynomial<Scalar> {
    Polynomial {
      coefficients: other.limbs,
      values: other.limb_values,
    }
  }
}

impl<Scalar: PrimeField> BigNat<Scalar> {
  /// Allocates a `BigNat` in the circuit with `n_limbs` limbs of width `limb_width` each.
  /// If `max_word` is missing, then it is assumed to be `(2 << limb_width) - 1`.
  /// The value is provided by a closure returning limb values.
  pub fn alloc_from_limbs<CS, F>(
    mut cs: CS,
    f: F,
    max_word: Option<BigInt>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    F: FnOnce() -> Result<Vec<Scalar>, SynthesisError>,
  {
    let values_cell = f();
    let mut value = None;
    let mut limb_values = None;
    let limbs = (0..n_limbs)
      .map(|limb_i| {
        cs.alloc(
          || format!("limb {limb_i}"),
          || match values_cell {
            Ok(ref vs) => {
              if vs.len() != n_limbs {
                eprintln!("Values do not match stated limb count");
                return Err(SynthesisError::Unsatisfiable);
              }
              if value.is_none() {
                value = Some(limbs_to_nat::<Scalar, _, _>(vs.iter(), limb_width));
              }
              if limb_values.is_none() {
                limb_values = Some(vs.clone());
              }
              Ok(vs[limb_i])
            }
            // Hack b/c SynthesisError and io::Error don't implement Clone
            Err(ref e) => Err(SynthesisError::from(std::io::Error::new(
              std::io::ErrorKind::Other,
              format!("{e}"),
            ))),
          },
        )
        .map(|v| LinearCombination::zero() + v)
      })
      .collect::<Result<Vec<_>, _>>()?;
    Ok(Self {
      value,
      limb_values,
      limbs,
      params: BigNatParams {
        min_bits: 0,
        n_limbs,
        max_word: max_word.unwrap_or_else(|| int_with_n_ones(limb_width)),
        limb_width,
      },
    })
  }

  /// Allocates a `BigNat` in the circuit with `n_limbs` limbs of width `limb_width` each.
  /// The `max_word` is guaranteed to be `(2 << limb_width) - 1`.
  /// The value is provided by a closure returning a natural number.
  pub fn alloc_from_nat<CS, F>(
    mut cs: CS,
    f: F,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    F: FnOnce() -> Result<BigInt, SynthesisError>,
  {
    let all_values_cell =
      f().and_then(|v| Ok((nat_to_limbs::<Scalar>(&v, limb_width, n_limbs)?, v)));
    let mut value = None;
    let mut limb_values = Vec::new();
    let limbs = (0..n_limbs)
      .map(|limb_i| {
        cs.alloc(
          || format!("limb {limb_i}"),
          || match all_values_cell {
            Ok((ref vs, ref v)) => {
              if value.is_none() {
                value = Some(v.clone());
              }
              limb_values.push(vs[limb_i]);
              Ok(vs[limb_i])
            }
            // Hack b/c SynthesisError and io::Error don't implement Clone
            Err(ref e) => Err(SynthesisError::from(std::io::Error::new(
              std::io::ErrorKind::Other,
              format!("{e}"),
            ))),
          },
        )
        .map(|v| LinearCombination::zero() + v)
      })
      .collect::<Result<Vec<_>, _>>()?;
    Ok(Self {
      value,
      limb_values: if !limb_values.is_empty() {
        Some(limb_values)
      } else {
        None
      },
      limbs,
      params: BigNatParams::new(limb_width, n_limbs),
    })
  }

  /// Allocates a `BigNat` in the circuit with `n_limbs` limbs of width `limb_width` each.
  /// The `max_word` is guaranteed to be `(2 << limb_width) - 1`.
  /// The value is provided by an allocated number
  pub fn from_num<CS: ConstraintSystem<Scalar>>(
    mut cs: CS,
    n: &Num<Scalar>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    let bignat = Self::alloc_from_nat(
      cs.namespace(|| "bignat"),
      || {
        Ok({
          n.value
            .as_ref()
            .map(|n| f_to_nat(n))
            .ok_or(SynthesisError::AssignmentMissing)?
        })
      },
      limb_width,
      n_limbs,
    )?;

    // check if bignat equals n
    // (1) decompose `bignat` into a bitvector `bv`
    let bv = bignat.decompose(cs.namespace(|| "bv"))?;
    // (2) recompose bits and check if it equals n
    n.is_equal(cs.namespace(|| "n"), &bv);

    Ok(bignat)
  }

  pub fn as_limbs(&self) -> Vec<Num<Scalar>> {
    let mut limbs = Vec::new();
    for (i, lc) in self.limbs.iter().enumerate() {
      limbs.push(Num::new(
        self.limb_values.as_ref().map(|vs| vs[i]),
        lc.clone(),
      ));
    }
    limbs
  }

  pub fn assert_well_formed<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<(), SynthesisError> {
    // swap the option and iterator
    let limb_values_split =
      (0..self.limbs.len()).map(|i| self.limb_values.as_ref().map(|vs| vs[i]));
    for (i, (limb, limb_value)) in self.limbs.iter().zip(limb_values_split).enumerate() {
      Num::new(limb_value, limb.clone())
        .fits_in_bits(cs.namespace(|| format!("{i}")), self.params.limb_width)?;
    }
    Ok(())
  }

  /// Break `self` up into a bit-vector.
  pub fn decompose<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<Bitvector<Scalar>, SynthesisError> {
    let limb_values_split =
      (0..self.limbs.len()).map(|i| self.limb_values.as_ref().map(|vs| vs[i]));
    let bitvectors: Vec<Bitvector<Scalar>> = self
      .limbs
      .iter()
      .zip(limb_values_split)
      .enumerate()
      .map(|(i, (limb, limb_value))| {
        Num::new(limb_value, limb.clone()).decompose(
          cs.namespace(|| format!("subdecmop {i}")),
          self.params.limb_width,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;
    let mut bits = Vec::new();
    let mut values = Vec::new();
    let mut allocations = Vec::new();
    for bv in bitvectors {
      bits.extend(bv.bits);
      if let Some(vs) = bv.values {
        values.extend(vs)
      };
      allocations.extend(bv.allocations);
    }
    let values = if !values.is_empty() {
      Some(values)
    } else {
      None
    };
    Ok(Bitvector {
      bits,
      values,
      allocations,
    })
  }

  pub fn enforce_limb_width_agreement(
    &self,
    other: &Self,
    location: &str,
  ) -> Result<usize, SynthesisError> {
    if self.params.limb_width == other.params.limb_width {
      Ok(self.params.limb_width)
    } else {
      eprintln!(
        "Limb widths {}, {}, do not agree at {}",
        self.params.limb_width, other.params.limb_width, location
      );
      Err(SynthesisError::Unsatisfiable)
    }
  }

  pub fn from_poly(poly: Polynomial<Scalar>, limb_width: usize, max_word: BigInt) -> Self {
    Self {
      params: BigNatParams {
        min_bits: 0,
        max_word,
        n_limbs: poly.coefficients.len(),
        limb_width,
      },
      limbs: poly.coefficients,
      value: poly
        .values
        .as_ref()
        .map(|limb_values| limbs_to_nat::<Scalar, _, _>(limb_values.iter(), limb_width)),
      limb_values: poly.values,
    }
  }

  /// Constrain `self` to be equal to `other`, after carrying both.
  pub fn equal_when_carried<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<(), SynthesisError> {
    self.enforce_limb_width_agreement(other, "equal_when_carried")?;

    // We'll propegate carries over the first `n` limbs.
    let n = min(self.limbs.len(), other.limbs.len());
    let target_base = BigInt::from(1u8) << self.params.limb_width as u32;
    let mut accumulated_extra = BigInt::from(0usize);
    let max_word = max(&self.params.max_word, &other.params.max_word);
    let carry_bits = (((max_word.to_f64().unwrap() * 2.0).log2() - self.params.limb_width as f64)
      .ceil()
      + 0.1) as usize;
    let mut carry_in = Num::new(Some(Scalar::ZERO), LinearCombination::zero());

    for i in 0..n {
      let carry = Num::alloc(cs.namespace(|| format!("carry value {i}")), || {
        Ok(
          nat_to_f(
            &((f_to_nat(&self.limb_values.grab()?[i])
              + f_to_nat(&carry_in.value.unwrap())
              + max_word
              - f_to_nat(&other.limb_values.grab()?[i]))
              / &target_base),
          )
          .unwrap(),
        )
      })?;
      accumulated_extra += max_word;

      cs.enforce(
        || format!("carry {i}"),
        |lc| lc,
        |lc| lc,
        |lc| {
          lc + &carry_in.num + &self.limbs[i] - &other.limbs[i]
            + (nat_to_f(max_word).unwrap(), CS::one())
            - (nat_to_f(&target_base).unwrap(), &carry.num)
            - (
              nat_to_f(&(&accumulated_extra % &target_base)).unwrap(),
              CS::one(),
            )
        },
      );

      accumulated_extra /= &target_base;

      if i < n - 1 {
        carry.fits_in_bits(cs.namespace(|| format!("carry {i} decomp")), carry_bits)?;
      } else {
        cs.enforce(
          || format!("carry {i} is out"),
          |lc| lc,
          |lc| lc,
          |lc| lc + &carry.num - (nat_to_f(&accumulated_extra).unwrap(), CS::one()),
        );
      }
      carry_in = carry;
    }

    for (i, zero_limb) in self.limbs.iter().enumerate().skip(n) {
      cs.enforce(
        || format!("zero self {i}"),
        |lc| lc,
        |lc| lc,
        |lc| lc + zero_limb,
      );
    }
    for (i, zero_limb) in other.limbs.iter().enumerate().skip(n) {
      cs.enforce(
        || format!("zero other {i}"),
        |lc| lc,
        |lc| lc,
        |lc| lc + zero_limb,
      );
    }
    Ok(())
  }

  /// Constrain `self` to be equal to `other`, after carrying both.
  /// Uses regrouping internally to take full advantage of the field size and reduce the amount
  /// of carrying.
  pub fn equal_when_carried_regroup<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<(), SynthesisError> {
    self.enforce_limb_width_agreement(other, "equal_when_carried_regroup")?;
    let max_word = max(&self.params.max_word, &other.params.max_word);
    let carry_bits = (((max_word.to_f64().unwrap() * 2.0).log2() - self.params.limb_width as f64)
      .ceil()
      + 0.1) as usize;
    let limbs_per_group = (Scalar::CAPACITY as usize - carry_bits) / self.params.limb_width;
    let self_grouped = self.group_limbs(limbs_per_group);
    let other_grouped = other.group_limbs(limbs_per_group);
    self_grouped.equal_when_carried(cs.namespace(|| "grouped"), &other_grouped)
  }

  pub fn add(&self, other: &Self) -> Result<BigNat<Scalar>, SynthesisError> {
    self.enforce_limb_width_agreement(other, "add")?;
    let n_limbs = max(self.params.n_limbs, other.params.n_limbs);
    let max_word = &self.params.max_word + &other.params.max_word;
    let limbs: Vec<LinearCombination<Scalar>> = (0..n_limbs)
      .map(|i| match (self.limbs.get(i), other.limbs.get(i)) {
        (Some(a), Some(b)) => a.clone() + b,
        (Some(a), None) => a.clone(),
        (None, Some(b)) => b.clone(),
        (None, None) => unreachable!(),
      })
      .collect();
    let limb_values: Option<Vec<Scalar>> = self.limb_values.as_ref().and_then(|x| {
      other.limb_values.as_ref().map(|y| {
        (0..n_limbs)
          .map(|i| match (x.get(i), y.get(i)) {
            (Some(a), Some(b)) => {
              let mut t = *a;
              t.add_assign(b);
              t
            }
            (Some(a), None) | (None, Some(a)) => *a,
            (None, None) => unreachable!(),
          })
          .collect()
      })
    });
    let value = self
      .value
      .as_ref()
      .and_then(|x| other.value.as_ref().map(|y| x + y));
    Ok(Self {
      limb_values,
      value,
      limbs,
      params: BigNatParams {
        min_bits: max(self.params.min_bits, other.params.min_bits),
        n_limbs,
        max_word,
        limb_width: self.params.limb_width,
      },
    })
  }

  /// Compute a `BigNat` contrained to be equal to `self * other % modulus`.
  pub fn mult_mod<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
    modulus: &Self,
  ) -> Result<(BigNat<Scalar>, BigNat<Scalar>), SynthesisError> {
    self.enforce_limb_width_agreement(other, "mult_mod")?;
    let limb_width = self.params.limb_width;
    let quotient_bits = (self.n_bits() + other.n_bits()).saturating_sub(modulus.params.min_bits);
    let quotient_limbs = quotient_bits.saturating_sub(1) / limb_width + 1;
    let quotient = BigNat::alloc_from_nat(
      cs.namespace(|| "quotient"),
      || {
        Ok({
          let mut x = self.value.grab()?.clone();
          x *= other.value.grab()?;
          x /= modulus.value.grab()?;
          x
        })
      },
      self.params.limb_width,
      quotient_limbs,
    )?;
    quotient.assert_well_formed(cs.namespace(|| "quotient rangecheck"))?;
    let remainder = BigNat::alloc_from_nat(
      cs.namespace(|| "remainder"),
      || {
        Ok({
          let mut x = self.value.grab()?.clone();
          x *= other.value.grab()?;
          x %= modulus.value.grab()?;
          x
        })
      },
      self.params.limb_width,
      modulus.limbs.len(),
    )?;
    remainder.assert_well_formed(cs.namespace(|| "remainder rangecheck"))?;
    let a_poly = Polynomial::from(self.clone());
    let b_poly = Polynomial::from(other.clone());
    let mod_poly = Polynomial::from(modulus.clone());
    let q_poly = Polynomial::from(quotient.clone());
    let r_poly = Polynomial::from(remainder.clone());

    // a * b
    let left = a_poly.alloc_product(cs.namespace(|| "left"), &b_poly)?;
    let right_product = q_poly.alloc_product(cs.namespace(|| "right_product"), &mod_poly)?;
    // q * m + r
    let right = right_product.sum(&r_poly);

    let left_max_word = {
      let mut x = BigInt::from(min(self.limbs.len(), other.limbs.len()));
      x *= &self.params.max_word;
      x *= &other.params.max_word;
      x
    };
    let right_max_word = {
      let mut x = BigInt::from(min(quotient.limbs.len(), modulus.limbs.len()));
      x *= &quotient.params.max_word;
      x *= &modulus.params.max_word;
      x += &remainder.params.max_word;
      x
    };

    let left_int = BigNat::from_poly(left, limb_width, left_max_word);
    let right_int = BigNat::from_poly(right, limb_width, right_max_word);
    left_int.equal_when_carried_regroup(cs.namespace(|| "carry"), &right_int)?;
    Ok((quotient, remainder))
  }

  /// Compute a `BigNat` contrained to be equal to `self * other % modulus`.
  pub fn red_mod<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    modulus: &Self,
  ) -> Result<BigNat<Scalar>, SynthesisError> {
    self.enforce_limb_width_agreement(modulus, "red_mod")?;
    let limb_width = self.params.limb_width;
    let quotient_bits = self.n_bits().saturating_sub(modulus.params.min_bits);
    let quotient_limbs = quotient_bits.saturating_sub(1) / limb_width + 1;
    let quotient = BigNat::alloc_from_nat(
      cs.namespace(|| "quotient"),
      || Ok(self.value.grab()? / modulus.value.grab()?),
      self.params.limb_width,
      quotient_limbs,
    )?;
    quotient.assert_well_formed(cs.namespace(|| "quotient rangecheck"))?;
    let remainder = BigNat::alloc_from_nat(
      cs.namespace(|| "remainder"),
      || Ok(self.value.grab()? % modulus.value.grab()?),
      self.params.limb_width,
      modulus.limbs.len(),
    )?;
    remainder.assert_well_formed(cs.namespace(|| "remainder rangecheck"))?;
    let mod_poly = Polynomial::from(modulus.clone());
    let q_poly = Polynomial::from(quotient.clone());
    let r_poly = Polynomial::from(remainder.clone());

    // q * m + r
    let right_product = q_poly.alloc_product(cs.namespace(|| "right_product"), &mod_poly)?;
    let right = right_product.sum(&r_poly);

    let right_max_word = {
      let mut x = BigInt::from(min(quotient.limbs.len(), modulus.limbs.len()));
      x *= &quotient.params.max_word;
      x *= &modulus.params.max_word;
      x += &remainder.params.max_word;
      x
    };

    let right_int = BigNat::from_poly(right, limb_width, right_max_word);
    self.equal_when_carried_regroup(cs.namespace(|| "carry"), &right_int)?;
    Ok(remainder)
  }

  /// Combines limbs into groups.
  pub fn group_limbs(&self, limbs_per_group: usize) -> BigNat<Scalar> {
    let n_groups = (self.limbs.len() - 1) / limbs_per_group + 1;
    let limb_values = self.limb_values.as_ref().map(|vs| {
      let mut values: Vec<Scalar> = vec![Scalar::ZERO; n_groups];
      let mut shift = Scalar::ONE;
      let limb_block = (0..self.params.limb_width).fold(Scalar::ONE, |mut l, _| {
        l = l.double();
        l
      });
      for (i, v) in vs.iter().enumerate() {
        if i % limbs_per_group == 0 {
          shift = Scalar::ONE;
        }
        let mut a = shift;
        a *= v;
        values[i / limbs_per_group].add_assign(&a);
        shift.mul_assign(&limb_block);
      }
      values
    });
    let limbs = {
      let mut limbs: Vec<LinearCombination<Scalar>> = vec![LinearCombination::zero(); n_groups];
      let mut shift = Scalar::ONE;
      let limb_block = (0..self.params.limb_width).fold(Scalar::ONE, |mut l, _| {
        l = l.double();
        l
      });
      for (i, limb) in self.limbs.iter().enumerate() {
        if i % limbs_per_group == 0 {
          shift = Scalar::ONE;
        }
        limbs[i / limbs_per_group] =
          std::mem::replace(&mut limbs[i / limbs_per_group], LinearCombination::zero())
            + (shift, limb);
        shift.mul_assign(&limb_block);
      }
      limbs
    };
    let max_word = (0..limbs_per_group).fold(BigInt::from(0u8), |mut acc, i| {
      acc.set_bit((i * self.params.limb_width) as u64, true);
      acc
    }) * &self.params.max_word;
    BigNat {
      params: BigNatParams {
        min_bits: self.params.min_bits,
        limb_width: self.params.limb_width * limbs_per_group,
        n_limbs: limbs.len(),
        max_word,
      },
      limbs,
      limb_values,
      value: self.value.clone(),
    }
  }

  pub fn n_bits(&self) -> usize {
    assert!(self.params.n_limbs > 0);
    self.params.limb_width * (self.params.n_limbs - 1) + self.params.max_word.bits() as usize
  }
}

pub struct Polynomial<Scalar: PrimeField> {
  pub coefficients: Vec<LinearCombination<Scalar>>,
  pub values: Option<Vec<Scalar>>,
}

impl<Scalar: PrimeField> Polynomial<Scalar> {
  pub fn alloc_product<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
  ) -> Result<Polynomial<Scalar>, SynthesisError> {
    let n_product_coeffs = self.coefficients.len() + other.coefficients.len() - 1;
    let values = self.values.as_ref().and_then(|self_vs| {
      other.values.as_ref().map(|other_vs| {
        let mut values: Vec<Scalar> = std::iter::repeat_with(|| Scalar::ZERO)
          .take(n_product_coeffs)
          .collect();
        for (self_i, self_v) in self_vs.iter().enumerate() {
          for (other_i, other_v) in other_vs.iter().enumerate() {
            let mut v = *self_v;
            v.mul_assign(other_v);
            values[self_i + other_i].add_assign(&v);
          }
        }
        values
      })
    });
    let coefficients = (0..n_product_coeffs)
      .map(|i| {
        Ok(LinearCombination::zero() + cs.alloc(|| format!("prod {i}"), || Ok(values.grab()?[i]))?)
      })
      .collect::<Result<Vec<LinearCombination<Scalar>>, SynthesisError>>()?;
    let product = Polynomial {
      coefficients,
      values,
    };
    let one = Scalar::ONE;
    let mut x = Scalar::ZERO;
    for _ in 1..(n_product_coeffs + 1) {
      x.add_assign(&one);
      cs.enforce(
        || format!("pointwise product @ {x:?}"),
        |lc| {
          let mut i = Scalar::ONE;
          self.coefficients.iter().fold(lc, |lc, c| {
            let r = lc + (i, c);
            i.mul_assign(&x);
            r
          })
        },
        |lc| {
          let mut i = Scalar::ONE;
          other.coefficients.iter().fold(lc, |lc, c| {
            let r = lc + (i, c);
            i.mul_assign(&x);
            r
          })
        },
        |lc| {
          let mut i = Scalar::ONE;
          product.coefficients.iter().fold(lc, |lc, c| {
            let r = lc + (i, c);
            i.mul_assign(&x);
            r
          })
        },
      )
    }
    Ok(product)
  }

  pub fn sum(&self, other: &Self) -> Self {
    let n_coeffs = max(self.coefficients.len(), other.coefficients.len());
    let values = self.values.as_ref().and_then(|self_vs| {
      other.values.as_ref().map(|other_vs| {
        (0..n_coeffs)
          .map(|i| {
            let mut s = Scalar::ZERO;
            if i < self_vs.len() {
              s.add_assign(&self_vs[i]);
            }
            if i < other_vs.len() {
              s.add_assign(&other_vs[i]);
            }
            s
          })
          .collect()
      })
    });
    let coefficients = (0..n_coeffs)
      .map(|i| {
        let mut lc = LinearCombination::zero();
        if i < self.coefficients.len() {
          lc = lc + &self.coefficients[i];
        }
        if i < other.coefficients.len() {
          lc = lc + &other.coefficients[i];
        }
        lc
      })
      .collect();
    Polynomial {
      coefficients,
      values,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use bellpepper_core::{test_cs::TestConstraintSystem, Circuit};
  use pasta_curves::pallas::Scalar;
  use proptest::prelude::*;

  pub struct PolynomialMultiplier<Scalar: PrimeField> {
    pub a: Vec<Scalar>,
    pub b: Vec<Scalar>,
  }

  impl<Scalar: PrimeField> Circuit<Scalar> for PolynomialMultiplier<Scalar> {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      let a = Polynomial {
        coefficients: self
          .a
          .iter()
          .enumerate()
          .map(|(i, x)| {
            Ok(LinearCombination::zero() + cs.alloc(|| format!("coeff_a {i}"), || Ok(*x))?)
          })
          .collect::<Result<Vec<LinearCombination<Scalar>>, SynthesisError>>()?,
        values: Some(self.a),
      };
      let b = Polynomial {
        coefficients: self
          .b
          .iter()
          .enumerate()
          .map(|(i, x)| {
            Ok(LinearCombination::zero() + cs.alloc(|| format!("coeff_b {i}"), || Ok(*x))?)
          })
          .collect::<Result<Vec<LinearCombination<Scalar>>, SynthesisError>>()?,
        values: Some(self.b),
      };
      let _prod = a.alloc_product(cs.namespace(|| "product"), &b)?;
      Ok(())
    }
  }

  #[test]
  fn test_polynomial_multiplier_circuit() {
    let mut cs = TestConstraintSystem::<Scalar>::new();

    let circuit = PolynomialMultiplier {
      a: [1, 1, 1].iter().map(|i| Scalar::from_u128(*i)).collect(),
      b: [1, 1].iter().map(|i| Scalar::from_u128(*i)).collect(),
    };

    circuit.synthesize(&mut cs).expect("synthesis failed");

    if let Some(token) = cs.which_is_unsatisfied() {
      eprintln!("Error: {} is unsatisfied", token);
    }
  }

  #[derive(Debug)]
  pub struct BigNatBitDecompInputs {
    pub n: BigInt,
  }

  pub struct BigNatBitDecompParams {
    pub limb_width: usize,
    pub n_limbs: usize,
  }

  pub struct BigNatBitDecomp {
    inputs: Option<BigNatBitDecompInputs>,
    params: BigNatBitDecompParams,
  }

  impl<Scalar: PrimeField> Circuit<Scalar> for BigNatBitDecomp {
    fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
      let n = BigNat::alloc_from_nat(
        cs.namespace(|| "n"),
        || Ok(self.inputs.grab()?.n.clone()),
        self.params.limb_width,
        self.params.n_limbs,
      )?;
      n.decompose(cs.namespace(|| "decomp"))?;
      Ok(())
    }
  }

  proptest! {

    #![proptest_config(ProptestConfig {
      cases: 10, // this test is costlier as max n gets larger
      .. ProptestConfig::default()
    })]
    #[test]
    fn test_big_nat_can_decompose(n in any::<u16>(), limb_width in 40u8..200) {
        let n = n as usize;

        let n_limbs = if n == 0 {
            1
        } else {
            (n - 1) / limb_width as usize + 1
        };

        let circuit = BigNatBitDecomp {
           inputs: Some(BigNatBitDecompInputs {
                n: BigInt::from(n),
            }),
            params: BigNatBitDecompParams {
                limb_width: limb_width as usize,
                n_limbs,
            },
        };
        let mut cs = TestConstraintSystem::<Scalar>::new();
        circuit.synthesize(&mut cs).expect("synthesis failed");
        prop_assert!(cs.is_satisfied());
    }
  }
}
