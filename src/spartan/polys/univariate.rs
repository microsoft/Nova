//! Main components:
//! - `UniPoly`: an univariate dense polynomial in coefficient form (big endian),
//! - `CompressedUniPoly`: a univariate dense polynomial, compressed (omitted linear term), in coefficient form (little endian),
use crate::traits::{AbsorbInRO2Trait, Engine, Group, ROTrait, TranscriptReprTrait};
use core::panic;
use ff::PrimeField;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

// ax^2 + bx + c stored as vec![c, b, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, b, a]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniPoly<Scalar: PrimeField> {
  pub(crate) coeffs: Vec<Scalar>,
}

// ax^2 + bx + c stored as vec![c, a]
// ax^3 + bx^2 + cx + d stored as vec![d, c, a]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedUniPoly<Scalar: PrimeField> {
  coeffs_except_linear_term: Vec<Scalar>,
}

impl<Scalar: PrimeField> UniPoly<Scalar> {
  #[cfg(feature = "experimental")]
  pub fn from_evals(evals: &[Scalar]) -> Self {
    let n = evals.len();
    let xs: Vec<Scalar> = (0..n).map(|x| Scalar::from(x as u64)).collect();

    let mut matrix: Vec<Vec<Scalar>> = Vec::with_capacity(n);
    for i in 0..n {
      let mut row = Vec::with_capacity(n);
      let x = xs[i];
      row.push(Scalar::ONE);
      row.push(x);
      for j in 2..n {
        row.push(row[j - 1] * x);
      }
      row.push(evals[i]);
      matrix.push(row);
    }

    let coeffs = gaussian_elimination(&mut matrix);
    Self { coeffs }
  }

  // a x^2 + b x + c
  // evals: [c, a + b + c, a]
  pub fn from_evals_deg2(evals: &[Scalar]) -> Self {
    let c = evals[0];
    let a = evals[2];
    let a_b_c = evals[1];
    let b = a_b_c - a - c;
    Self {
      coeffs: vec![c, b, a],
    }
  }

  // a x^3 + b x^2 + c x + d
  // evals: [d, a + b + c, a, -a + b - c + d]
  pub fn from_evals_deg3(evals: &[Scalar]) -> Self {
    let d = evals[0];
    let a = evals[2];
    let a_b_c_d = evals[1];
    let b2_d2 = a_b_c_d + evals[3];
    let b = b2_d2 * Scalar::TWO_INV - d;
    let c = a_b_c_d - a - d - b;
    Self {
      coeffs: vec![d, c, b, a],
    }
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

impl<E: Engine> AbsorbInRO2Trait<E> for UniPoly<E::Scalar> {
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    for coeff in &self.coeffs {
      ro.absorb(*coeff);
    }
  }
}

// This code is based on code from https://github.com/a16z/jolt/blob/main/jolt-core/src/utils/gaussian_elimination.rs, which itself is
// inspired by https://github.com/TheAlgorithms/Rust/blob/master/src/math/gaussian_elimination.rs
pub fn gaussian_elimination<F: PrimeField>(matrix: &mut [Vec<F>]) -> Vec<F> {
  let size = matrix.len();
  assert_eq!(size, matrix[0].len() - 1);

  for i in 0..size - 1 {
    for j in i..size - 1 {
      echelon(matrix, i, j);
    }
  }

  for i in (1..size).rev() {
    eliminate(matrix, i);
  }

  let mut result: Vec<F> = vec![F::ZERO; size];
  for i in 0..size {
    result[i] = div_f(matrix[i][size], matrix[i][i]);
  }

  result
}

fn echelon<F: PrimeField>(matrix: &mut [Vec<F>], i: usize, j: usize) {
  let size = matrix.len();
  if matrix[i][i] != F::ZERO {
    let factor = div_f(matrix[j + 1][i], matrix[i][i]);
    (i..size + 1).for_each(|k| {
      let tmp = matrix[i][k];
      matrix[j + 1][k] -= factor * tmp;
    });
  }
}

fn eliminate<F: PrimeField>(matrix: &mut [Vec<F>], i: usize) {
  let size = matrix.len();
  if matrix[i][i] != F::ZERO {
    for j in (1..i + 1).rev() {
      let factor = div_f(matrix[j - 1][i], matrix[i][i]);
      for k in (0..size + 1).rev() {
        let tmp = matrix[i][k];
        matrix[j - 1][k] -= factor * tmp;
      }
    }
  }
}

/// Division of two prime fields
///
/// # Panics
///
/// Panics if `b` is zero.
pub fn div_f<F: PrimeField>(a: F, b: F) -> F {
  let inverse_b = b.invert();

  if inverse_b.into_option().is_none() {
    panic!("Division by zero");
  }

  a * inverse_b.unwrap()
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{bn256_grumpkin::bn256, pasta::pallas, secp_secq::secp256k1};

  fn test_from_evals_quad_with<F: PrimeField>() {
    // polynomial is 2x^2 + 3x + 1
    let e0 = F::ONE;
    let e1 = F::from(6);
    let e2 = F::from(2);
    let evals = vec![e0, e1, e2];
    let poly = UniPoly::from_evals_deg2(&evals);

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
    test_from_evals_quad_with::<pallas::Scalar>();
    test_from_evals_quad_with::<bn256::Scalar>();
    test_from_evals_quad_with::<secp256k1::Scalar>();
  }

  fn test_from_evals_cubic_with<F: PrimeField>() {
    // polynomial is x^3 + 2x^2 + 3x + 1
    let e0 = F::ONE; // f(0)
    let e1 = F::from(7); // f(1)
    let e2 = F::ONE; // cubic term
    let e3 = -F::ONE; // f(-1)
    let evals = vec![e0, e1, e2, e3];
    let poly = UniPoly::from_evals_deg3(&evals);

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
    test_from_evals_cubic_with::<pallas::Scalar>();
    test_from_evals_cubic_with::<bn256::Scalar>();
    test_from_evals_cubic_with::<secp256k1::Scalar>()
  }
}
