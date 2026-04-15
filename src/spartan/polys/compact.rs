//! Compact multilinear polynomial: stores evaluations as small integer types
//! (bool, u8, u16, u32, u64, i64) and defers conversion to field elements.
//!
//! After the first `bind` round, coefficients are promoted to field elements.
//! The first sumcheck eval round can use integer arithmetic to avoid expensive
//! field multiplications, reducing both memory and compute.

use crate::constants::PARALLEL_THRESHOLD;
use crate::spartan::math::Math;
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Trait for small scalar types that can be stored compactly and converted to field elements.
pub trait SmallScalar: Copy + Send + Sync + Default + 'static {
  /// Convert this value to a field element.
  fn to_field<F: PrimeField>(self) -> F;

  /// Compute `(hi - lo)` as a field element, potentially using cheaper integer arithmetic.
  fn diff_to_field<F: PrimeField>(lo: Self, hi: Self) -> F;

  /// Maximum number of bits needed to represent any value of this type.
  /// Used by MSM to select optimal algorithm.
  fn max_bits() -> u32;
}

impl SmallScalar for bool {
  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    if self {
      F::ONE
    } else {
      F::ZERO
    }
  }

  #[inline(always)]
  fn diff_to_field<F: PrimeField>(lo: Self, hi: Self) -> F {
    match (lo, hi) {
      (false, false) | (true, true) => F::ZERO,
      (false, true) => F::ONE,
      (true, false) => F::ZERO - F::ONE,
    }
  }

  fn max_bits() -> u32 {
    1
  }
}

impl SmallScalar for u8 {
  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    F::from(self as u64)
  }

  #[inline(always)]
  fn diff_to_field<F: PrimeField>(lo: Self, hi: Self) -> F {
    if hi >= lo {
      F::from((hi - lo) as u64)
    } else {
      F::ZERO - F::from((lo - hi) as u64)
    }
  }

  fn max_bits() -> u32 {
    8
  }
}

impl SmallScalar for u16 {
  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    F::from(self as u64)
  }

  #[inline(always)]
  fn diff_to_field<F: PrimeField>(lo: Self, hi: Self) -> F {
    if hi >= lo {
      F::from((hi - lo) as u64)
    } else {
      F::ZERO - F::from((lo - hi) as u64)
    }
  }

  fn max_bits() -> u32 {
    16
  }
}

impl SmallScalar for u32 {
  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    F::from(self as u64)
  }

  #[inline(always)]
  fn diff_to_field<F: PrimeField>(lo: Self, hi: Self) -> F {
    if hi >= lo {
      F::from((hi - lo) as u64)
    } else {
      F::ZERO - F::from((lo - hi) as u64)
    }
  }

  fn max_bits() -> u32 {
    32
  }
}

impl SmallScalar for u64 {
  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    F::from(self)
  }

  #[inline(always)]
  fn diff_to_field<F: PrimeField>(lo: Self, hi: Self) -> F {
    if hi >= lo {
      F::from(hi - lo)
    } else {
      F::ZERO - F::from(lo - hi)
    }
  }

  fn max_bits() -> u32 {
    64
  }
}

impl SmallScalar for i64 {
  #[inline(always)]
  fn to_field<F: PrimeField>(self) -> F {
    if self >= 0 {
      F::from(self as u64)
    } else {
      F::ZERO - F::from(self.unsigned_abs())
    }
  }

  #[inline(always)]
  fn diff_to_field<F: PrimeField>(lo: Self, hi: Self) -> F {
    let diff = hi as i128 - lo as i128;
    if diff >= 0 {
      F::from(diff as u64)
    } else {
      F::ZERO - F::from((-diff) as u64)
    }
  }

  fn max_bits() -> u32 {
    64
  }
}

/// A multilinear polynomial stored in compact form using small scalar type `T`.
///
/// Coefficients are stored as `Vec<T>` until the first bind operation, at which
/// point they are promoted to `Vec<Scalar>`. This saves both memory (1-8 bytes
/// per entry instead of 32) and compute (integer diff instead of field sub).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactPolynomial<T: SmallScalar, Scalar: PrimeField> {
  num_vars: usize,
  /// Compact coefficients (before any binding).
  /// `None` after the first bind promotes them to `bound_coeffs`.
  coeffs: Option<Vec<T>>,
  /// Field-element coefficients (populated after first bind, or on demand).
  bound_coeffs: Vec<Scalar>,
}

impl<T: SmallScalar, Scalar: PrimeField> CompactPolynomial<T, Scalar> {
  /// Create a new compact polynomial from small-type evaluations.
  pub fn new(coeffs: Vec<T>) -> Self {
    let num_vars = coeffs.len().log_2();
    assert_eq!(coeffs.len(), 1 << num_vars);
    CompactPolynomial {
      num_vars,
      coeffs: Some(coeffs),
      bound_coeffs: Vec::new(),
    }
  }

  /// Returns the number of variables.
  pub const fn get_num_vars(&self) -> usize {
    self.num_vars
  }

  /// Returns the total number of evaluations.
  pub fn len(&self) -> usize {
    if let Some(ref c) = self.coeffs {
      c.len()
    } else {
      self.bound_coeffs.len()
    }
  }

  /// Returns true if the polynomial has no evaluations.
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// Returns true if still in compact (pre-bind) form.
  pub fn is_compact(&self) -> bool {
    self.coeffs.is_some()
  }

  /// Access compact coefficients (panics if already promoted).
  pub fn compact_coeffs(&self) -> &[T] {
    self
      .coeffs
      .as_ref()
      .expect("already promoted to field elements")
  }

  /// Access field-element coefficients (panics if still compact).
  pub fn field_coeffs(&self) -> &[Scalar] {
    assert!(self.coeffs.is_none(), "still in compact form");
    &self.bound_coeffs
  }

  /// Get coefficients as field elements, converting if necessary.
  /// Does not consume or modify the polynomial.
  pub fn to_field_vec(&self) -> Vec<Scalar> {
    if let Some(ref c) = self.coeffs {
      if c.len() < PARALLEL_THRESHOLD {
        c.iter().map(|v| v.to_field()).collect()
      } else {
        c.par_iter().map(|v| v.to_field()).collect()
      }
    } else {
      self.bound_coeffs.clone()
    }
  }

  /// Bind the top variable with challenge `r`.
  ///
  /// First call: promotes compact `Vec<T>` → `Vec<Scalar>` of half size.
  /// Subsequent calls: operates on `Vec<Scalar>` like standard MLE bind.
  pub fn bind_poly_var_top(&mut self, r: &Scalar) {
    assert!(self.num_vars > 0);

    if let Some(coeffs) = self.coeffs.take() {
      // First bind: compact → field, computing lo + r*(hi-lo) in one pass
      let n = coeffs.len() / 2;
      if n < PARALLEL_THRESHOLD {
        self.bound_coeffs = coeffs[..n]
          .iter()
          .zip(coeffs[n..].iter())
          .map(|(&lo, &hi)| {
            let lo_f: Scalar = lo.to_field();
            lo_f + *r * T::diff_to_field::<Scalar>(lo, hi)
          })
          .collect();
      } else {
        self.bound_coeffs = coeffs[..n]
          .par_iter()
          .zip(coeffs[n..].par_iter())
          .map(|(&lo, &hi)| {
            let lo_f: Scalar = lo.to_field();
            lo_f + *r * T::diff_to_field::<Scalar>(lo, hi)
          })
          .collect();
      }
    } else {
      // Subsequent binds: standard field-element bind
      let n = self.bound_coeffs.len() / 2;
      let (left, right) = self.bound_coeffs.split_at_mut(n);

      if n < PARALLEL_THRESHOLD {
        left.iter_mut().zip(right.iter()).for_each(|(a, b)| {
          *a += *r * (*b - *a);
        });
      } else {
        left
          .par_iter_mut()
          .zip(right.par_iter())
          .for_each(|(a, b)| {
            *a += *r * (*b - *a);
          });
      }

      self.bound_coeffs.truncate(n);
    }

    self.num_vars -= 1;
  }

  /// Evaluate at a point without binding (non-mutating).
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(r.len(), self.num_vars);
    if let Some(ref c) = self.coeffs {
      Self::evaluate_compact(c, r)
    } else {
      // Already promoted, use standard evaluation
      crate::spartan::polys::multilinear::MultilinearPolynomial::evaluate_with(
        &self.bound_coeffs,
        r,
      )
    }
  }

  /// Evaluate compact coefficients at a point using sqrt decomposition.
  fn evaluate_compact(z: &[T], r: &[Scalar]) -> Scalar {
    use crate::spartan::polys::eq::EqPolynomial;

    let s = r.len();
    let s_right = s / 2;
    let s_left = s - s_right;
    let n_left = 1 << s_left;
    let n_right = 1 << s_right;

    let eq_left = EqPolynomial::evals_from_points(&r[..s_left]);
    let eq_right = EqPolynomial::evals_from_points(&r[s_left..]);

    let reduced: Vec<Scalar> = (0..n_left)
      .into_par_iter()
      .map(|i| {
        let chunk = &z[i * n_right..(i + 1) * n_right];
        chunk
          .iter()
          .zip(eq_right.iter())
          .map(|(v, e)| v.to_field::<Scalar>() * *e)
          .sum()
      })
      .collect();

    reduced
      .into_par_iter()
      .zip(eq_left.into_par_iter())
      .map(|(r, e)| r * e)
      .sum()
  }

  /// Compute the quadratic evaluation points for the first sumcheck round
  /// when the polynomial is still in compact form.
  ///
  /// Returns (eval_0, eval_2) where:
  /// - eval_0 = Σ_i Z[i] * eq[i]  (lower half, r=0)
  /// - eval_2 = Σ_i (2*Z[n+i] - Z[i]) * (2*eq[n+i] - eq[i])  (r=2)
  ///
  /// This avoids field multiplications for the Z terms in compact form.
  pub fn evaluation_points_quadratic_with_eq(&self, eq: &[Scalar]) -> (Scalar, Scalar) {
    if let Some(ref c) = self.coeffs {
      let n = c.len() / 2;
      assert_eq!(eq.len(), c.len());

      if n < PARALLEL_THRESHOLD {
        let (e0, e2) = c[..n]
          .iter()
          .zip(c[n..].iter())
          .zip(eq[..n].iter())
          .zip(eq[n..].iter())
          .fold(
            (Scalar::ZERO, Scalar::ZERO),
            |(mut e0, mut e2), (((&lo, &hi), &eq_lo), &eq_hi)| {
              let lo_f: Scalar = lo.to_field();
              e0 += lo_f * eq_lo;
              let hi_f: Scalar = hi.to_field();
              let z_2 = hi_f + hi_f - lo_f;
              let eq_2 = eq_hi + eq_hi - eq_lo;
              e2 += z_2 * eq_2;
              (e0, e2)
            },
          );
        (e0, e2)
      } else {
        let (e0, e2): (Scalar, Scalar) = c[..n]
          .par_iter()
          .zip(c[n..].par_iter())
          .zip(eq[..n].par_iter())
          .zip(eq[n..].par_iter())
          .map(|(((&lo, &hi), &eq_lo), &eq_hi)| {
            let lo_f: Scalar = lo.to_field();
            let e0 = lo_f * eq_lo;
            let hi_f: Scalar = hi.to_field();
            let z_2 = hi_f + hi_f - lo_f;
            let eq_2 = eq_hi + eq_hi - eq_lo;
            let e2 = z_2 * eq_2;
            (e0, e2)
          })
          .reduce(
            || (Scalar::ZERO, Scalar::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
          );
        (e0, e2)
      }
    } else {
      // Already promoted — use field arithmetic
      let n = self.bound_coeffs.len() / 2;
      assert_eq!(eq.len(), self.bound_coeffs.len());
      let z = &self.bound_coeffs;

      if n < PARALLEL_THRESHOLD {
        let (e0, e2) = z[..n]
          .iter()
          .zip(z[n..].iter())
          .zip(eq[..n].iter())
          .zip(eq[n..].iter())
          .fold(
            (Scalar::ZERO, Scalar::ZERO),
            |(mut e0, mut e2), (((&lo, &hi), &eq_lo), &eq_hi)| {
              e0 += lo * eq_lo;
              let z_2 = hi + hi - lo;
              let eq_2 = eq_hi + eq_hi - eq_lo;
              e2 += z_2 * eq_2;
              (e0, e2)
            },
          );
        (e0, e2)
      } else {
        let (e0, e2): (Scalar, Scalar) = z[..n]
          .par_iter()
          .zip(z[n..].par_iter())
          .zip(eq[..n].par_iter())
          .zip(eq[n..].par_iter())
          .map(|(((&lo, &hi), &eq_lo), &eq_hi)| {
            let e0 = lo * eq_lo;
            let z_2 = hi + hi - lo;
            let eq_2 = eq_hi + eq_hi - eq_lo;
            let e2 = z_2 * eq_2;
            (e0, e2)
          })
          .reduce(
            || (Scalar::ZERO, Scalar::ZERO),
            |(a0, a2), (b0, b2)| (a0 + b0, a2 + b2),
          );
        (e0, e2)
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::pasta::pallas;
  use crate::spartan::polys::multilinear::MultilinearPolynomial;
  use ff::Field;

  type F = pallas::Scalar;

  #[test]
  fn test_compact_bool_bind() {
    let vals = vec![true, false, true, true, false, false, true, false];
    let mut compact = CompactPolynomial::<bool, F>::new(vals.clone());

    let dense_vals: Vec<F> = vals
      .iter()
      .map(|&b| if b { F::ONE } else { F::ZERO })
      .collect();
    let mut dense = MultilinearPolynomial::new(dense_vals);

    let r = F::from(7u64);

    compact.bind_poly_var_top(&r);
    dense.bind_poly_var_top(&r);

    assert!(!compact.is_compact());
    assert_eq!(compact.bound_coeffs, dense.Z[..dense.len()].to_vec());
  }

  #[test]
  fn test_compact_u64_bind() {
    let vals: Vec<u64> = vec![100, 200, 300, 400];
    let mut compact = CompactPolynomial::<u64, F>::new(vals.clone());

    let dense_vals: Vec<F> = vals.iter().map(|&v| F::from(v)).collect();
    let mut dense = MultilinearPolynomial::new(dense_vals);

    let r = F::from(42u64);

    compact.bind_poly_var_top(&r);
    dense.bind_poly_var_top(&r);

    assert_eq!(compact.bound_coeffs, dense.Z[..dense.len()].to_vec());
  }

  #[test]
  fn test_compact_evaluate() {
    let vals: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];
    let compact = CompactPolynomial::<u8, F>::new(vals.clone());

    let dense_vals: Vec<F> = vals.iter().map(|&v| F::from(v as u64)).collect();
    let dense = MultilinearPolynomial::new(dense_vals);

    let point = vec![F::from(3u64), F::from(5u64), F::from(7u64)];

    let eval_compact = compact.evaluate(&point);
    let eval_dense = dense.evaluate(&point);

    assert_eq!(eval_compact, eval_dense);
  }

  #[test]
  fn test_compact_i64_negative() {
    let vals: Vec<i64> = vec![-5, 10, -3, 7];
    let mut compact = CompactPolynomial::<i64, F>::new(vals.clone());

    let dense_vals: Vec<F> = vals
      .iter()
      .map(|&v| {
        if v >= 0 {
          F::from(v as u64)
        } else {
          F::ZERO - F::from((-v) as u64)
        }
      })
      .collect();
    let mut dense = MultilinearPolynomial::new(dense_vals);

    let r = F::from(11u64);

    compact.bind_poly_var_top(&r);
    dense.bind_poly_var_top(&r);

    assert_eq!(compact.bound_coeffs, dense.Z[..dense.len()].to_vec());
  }

  #[test]
  fn test_compact_multi_bind() {
    let vals: Vec<u8> = (0..16).collect();
    let mut compact = CompactPolynomial::<u8, F>::new(vals.clone());

    let dense_vals: Vec<F> = vals.iter().map(|&v| F::from(v as u64)).collect();
    let mut dense = MultilinearPolynomial::new(dense_vals);

    for i in 0..4 {
      let r = F::from((i * 3 + 7) as u64);
      compact.bind_poly_var_top(&r);
      dense.bind_poly_var_top(&r);
    }

    assert_eq!(compact.bound_coeffs.len(), 1);
    assert_eq!(compact.bound_coeffs[0], dense.Z[0]);
  }
}
