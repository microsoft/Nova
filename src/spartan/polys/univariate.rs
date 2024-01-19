//! Main components:
//! - `UniPoly`: an univariate dense polynomial in coefficient form (big endian),
//! - `CompressedUniPoly`: a univariate dense polynomial, compressed (omitted linear term), in coefficient form (little endian),
use ff::PrimeField;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::traits::{Group, TranscriptReprTrait};

// ax^2 + bx + c stored as vec![c, b, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, b, a]
#[derive(Debug, PartialEq, Eq)]
pub struct UniPoly<Scalar: PrimeField> {
  coeffs: Vec<Scalar>,
}

// ax^2 + bx + c stored as vec![c, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, a]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedUniPoly<Scalar: PrimeField> {
  coeffs_except_linear_term: Vec<Scalar>,
}

impl<Scalar: PrimeField> UniPoly<Scalar> {
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
#[cfg(test)]
mod tests {
  use crate::provider::{bn256_grumpkin, secp_secq::secp256k1};

  use super::*;

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
