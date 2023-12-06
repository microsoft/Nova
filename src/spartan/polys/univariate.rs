//! Main components:
//! - `UniPoly`: an univariate dense polynomial in coefficient form (big endian),
//! - `CompressedUniPoly`: a univariate dense polynomial, compressed (omitted linear term), in coefficient form (little endian),
use std::{
  cmp::Ordering,
  ops::{AddAssign, Index, IndexMut, MulAssign},
};

use ff::PrimeField;
use rayon::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use ref_cast::RefCast;
use serde::{Deserialize, Serialize};

use crate::traits::{Group, TranscriptReprTrait};

// ax^2 + bx + c stored as vec![c, b, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, b, a]
#[derive(Debug, Clone, PartialEq, Eq, RefCast)]
#[repr(transparent)]
pub struct UniPoly<Scalar: PrimeField> {
  pub coeffs: Vec<Scalar>,
}

// ax^2 + bx + c stored as vec![c, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, a]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedUniPoly<Scalar: PrimeField> {
  coeffs_except_linear_term: Vec<Scalar>,
}

impl<Scalar: PrimeField> UniPoly<Scalar> {
  pub fn new(coeffs: Vec<Scalar>) -> Self {
    let mut res = UniPoly { coeffs };
    res.truncate_leading_zeros();
    res
  }

  fn zero() -> Self {
    UniPoly::new(Vec::new())
  }

  /// Divide self by another polynomial, and returns the
  /// quotient and remainder.
  pub fn divide_with_q_and_r(&self, divisor: &Self) -> Option<(UniPoly<Scalar>, UniPoly<Scalar>)> {
    if self.is_zero() {
      Some((UniPoly::zero(), UniPoly::zero()))
    } else if divisor.is_zero() {
      panic!("Dividing by zero polynomial")
    } else if self.degree() < divisor.degree() {
      Some((UniPoly::zero(), self.clone()))
    } else {
      // Now we know that self.degree() >= divisor.degree();
      let mut quotient = vec![Scalar::ZERO; self.degree() - divisor.degree() + 1];
      let mut remainder: UniPoly<Scalar> = self.clone();
      // Can unwrap here because we know self is not zero.
      let divisor_leading_inv = divisor.leading_coefficient().unwrap().invert().unwrap();
      while !remainder.is_zero() && remainder.degree() >= divisor.degree() {
        let cur_q_coeff = *remainder.leading_coefficient().unwrap() * divisor_leading_inv;
        let cur_q_degree = remainder.degree() - divisor.degree();
        quotient[cur_q_degree] = cur_q_coeff;

        for (i, div_coeff) in divisor.coeffs.iter().enumerate() {
          remainder.coeffs[cur_q_degree + i] -= &(cur_q_coeff * div_coeff);
        }
        while let Some(true) = remainder.coeffs.last().map(|c| c == &Scalar::ZERO) {
          remainder.coeffs.pop();
        }
      }
      Some((UniPoly::new(quotient), remainder))
    }
  }

  pub fn is_zero(&self) -> bool {
    self.coeffs.is_empty() || self.coeffs.iter().all(|c| c == &Scalar::ZERO)
  }

  fn truncate_leading_zeros(&mut self) {
    while self.coeffs.last().map_or(false, |c| c == &Scalar::ZERO) {
      self.coeffs.pop();
    }
  }

  pub fn leading_coefficient(&self) -> Option<&Scalar> {
    self.coeffs.last()
  }

  pub fn from_evals(evals: &[Scalar]) -> Self {
    // we only support degree-2 or degree-3 univariate polynomials
    assert!(evals.len() == 3 || evals.len() == 4);
    let two_inv = Scalar::from(2).invert().unwrap();
    let coeffs = if evals.len() == 3 {
      // ax^2 + bx + c
      let c = evals[0];
      let a = two_inv * (evals[2] - evals[1] - evals[1] + c);
      let b = evals[1] - c - a;
      vec![c, b, a]
    } else {
      // ax^3 + bx^2 + cx + d
      let six_inv = Scalar::from(6).invert().unwrap();

      let d = evals[0];
      let a = six_inv
        * (evals[3] - evals[2] - evals[2] - evals[2] + evals[1] + evals[1] + evals[1] - evals[0]);
      let b = two_inv
        * (evals[0] + evals[0] - evals[1] - evals[1] - evals[1] - evals[1] - evals[1]
          + evals[2]
          + evals[2]
          + evals[2]
          + evals[2]
          - evals[3]);
      let c = evals[1] - d - a - b;
      vec![d, c, b, a]
    };

    UniPoly { coeffs }
  }

  pub fn degree(&self) -> usize {
    self.coeffs.len() - 1
  }

  pub fn eval_at_zero(&self) -> Scalar {
    self.coeffs[0]
  }

  pub fn eval_at_one(&self) -> Scalar {
    (0..self.coeffs.len())
      .into_par_iter()
      .map(|i| self.coeffs[i])
      .sum()
  }

  pub fn evaluate(&self, r: &Scalar) -> Scalar {
    let mut eval = self.coeffs[0];
    let mut power = *r;
    for coeff in self.coeffs.iter().skip(1) {
      eval += power * coeff;
      power *= r;
    }
    eval
  }

  pub fn compress(&self) -> CompressedUniPoly<Scalar> {
    let coeffs_except_linear_term = [&self.coeffs[0..1], &self.coeffs[2..]].concat();
    assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
    CompressedUniPoly {
      coeffs_except_linear_term,
    }
  }
}

impl<Scalar: PrimeField> CompressedUniPoly<Scalar> {
  // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
  // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
  pub fn decompress(&self, hint: &Scalar) -> UniPoly<Scalar> {
    let mut linear_term =
      *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
    for i in 1..self.coeffs_except_linear_term.len() {
      linear_term -= self.coeffs_except_linear_term[i];
    }

    let mut coeffs: Vec<Scalar> = Vec::new();
    coeffs.push(self.coeffs_except_linear_term[0]);
    coeffs.push(linear_term);
    coeffs.extend(&self.coeffs_except_linear_term[1..]);
    assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
    UniPoly { coeffs }
  }
}

impl<G: Group> TranscriptReprTrait<G> for UniPoly<G::Scalar> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let coeffs = self.compress().coeffs_except_linear_term;
    coeffs
      .iter()
      .flat_map(|&t| t.to_repr().as_ref().to_vec())
      .collect::<Vec<u8>>()
  }
}

impl<Scalar: PrimeField> Index<usize> for UniPoly<Scalar> {
  type Output = Scalar;

  fn index(&self, index: usize) -> &Self::Output {
    &self.coeffs[index]
  }
}

impl<Scalar: PrimeField> IndexMut<usize> for UniPoly<Scalar> {
  fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    &mut self.coeffs[index]
  }
}

impl<Scalar: PrimeField> AddAssign<&Scalar> for UniPoly<Scalar> {
  fn add_assign(&mut self, rhs: &Scalar) {
    self.coeffs.par_iter_mut().for_each(|c| *c += rhs);
  }
}

impl<Scalar: PrimeField> MulAssign<&Scalar> for UniPoly<Scalar> {
  fn mul_assign(&mut self, rhs: &Scalar) {
    self.coeffs.par_iter_mut().for_each(|c| *c *= rhs);
  }
}

impl<Scalar: PrimeField> AddAssign<&Self> for UniPoly<Scalar> {
  fn add_assign(&mut self, rhs: &Self) {
    let ordering = self.coeffs.len().cmp(&rhs.coeffs.len());
    #[allow(clippy::disallowed_methods)]
    for (lhs, rhs) in self.coeffs.iter_mut().zip(&rhs.coeffs) {
      *lhs += rhs;
    }
    if matches!(ordering, Ordering::Less) {
      self
        .coeffs
        .extend(rhs.coeffs[self.coeffs.len()..].iter().cloned());
    }
    if matches!(ordering, Ordering::Equal) {
      self.truncate_leading_zeros();
    }
  }
}

impl<Scalar: PrimeField> AsRef<Vec<Scalar>> for UniPoly<Scalar> {
  fn as_ref(&self) -> &Vec<Scalar> {
    &self.coeffs
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{bn256_grumpkin, secp_secq::secp256k1};

  fn test_from_evals_quad_with<F: PrimeField>() {
    // polynomial is 2x^2 + 3x + 1
    let e0 = F::ONE;
    let e1 = F::from(6);
    let e2 = F::from(15);
    let evals = vec![e0, e1, e2];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 3);
    assert_eq!(poly.coeffs[0], F::ONE);
    assert_eq!(poly.coeffs[1], F::from(3));
    assert_eq!(poly.coeffs[2], F::from(2));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e3 = F::from(28);
    assert_eq!(poly.evaluate(&F::from(3)), e3);
  }

  #[test]
  fn test_from_evals_quad() {
    test_from_evals_quad_with::<pasta_curves::pallas::Scalar>();
    test_from_evals_quad_with::<bn256_grumpkin::bn256::Scalar>();
    test_from_evals_quad_with::<secp256k1::Scalar>();
  }

  fn test_from_evals_cubic_with<F: PrimeField>() {
    // polynomial is x^3 + 2x^2 + 3x + 1
    let e0 = F::ONE;
    let e1 = F::from(7);
    let e2 = F::from(23);
    let e3 = F::from(55);
    let evals = vec![e0, e1, e2, e3];
    let poly = UniPoly::from_evals(&evals);

    assert_eq!(poly.eval_at_zero(), e0);
    assert_eq!(poly.eval_at_one(), e1);
    assert_eq!(poly.coeffs.len(), 4);

    assert_eq!(poly.coeffs[1], F::from(3));
    assert_eq!(poly.coeffs[2], F::from(2));
    assert_eq!(poly.coeffs[3], F::from(1));

    let hint = e0 + e1;
    let compressed_poly = poly.compress();
    let decompressed_poly = compressed_poly.decompress(&hint);
    for i in 0..decompressed_poly.coeffs.len() {
      assert_eq!(decompressed_poly.coeffs[i], poly.coeffs[i]);
    }

    let e4 = F::from(109);
    assert_eq!(poly.evaluate(&F::from(4)), e4);
  }

  #[test]
  fn test_from_evals_cubic() {
    test_from_evals_cubic_with::<pasta_curves::pallas::Scalar>();
    test_from_evals_cubic_with::<bn256_grumpkin::bn256::Scalar>();
    test_from_evals_cubic_with::<secp256k1::Scalar>()
  }
}
