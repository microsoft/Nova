//! This module provides foundational types and functions for manipulating multilinear polynomials in the context of cryptographic computations.
//!
//! Main components:
//! - `EqPolynomial`: Represents multilinear extension of equality polynomials, evaluated based on binary input values.
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.
use core::ops::Index;
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::{Add, Mul};

use crate::spartan::math::Math;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $eq(x,e)$, denoted as $\tilde{eq}(x, e)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{eq}(x, e) = \prod_{i=0}^m(e_i * x_i + (1 - e_i) * (1 - x_i))
/// $$
///
/// Each element in the vector `r` corresponds to a component $e_i$, representing a bit from the binary representation of an input value $e$.
/// This polynomial evaluates to 1 if every component $x_i$ equals its corresponding $e_i$, and 0 otherwise.
///
/// For instance, for e = 6 (with a binary representation of 0b110), the vector r would be [1, 1, 0].
pub struct EqPolynomial<Scalar: PrimeField> {
  r: Vec<Scalar>,
}

impl<Scalar: PrimeField> EqPolynomial<Scalar> {
  /// Creates a new `EqPolynomial` from a vector of Scalars `r`.
  ///
  /// Each Scalar in `r` corresponds to a bit from the binary representation of an input value `e`.
  pub fn new(r: Vec<Scalar>) -> Self {
    EqPolynomial { r }
  }

  /// Evaluates the `EqPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `r`.
  ///
  /// Panics if `rx` and `r` have different lengths.
  pub fn evaluate(&self, rx: &[Scalar]) -> Scalar {
    assert_eq!(self.r.len(), rx.len());
    (0..rx.len())
      .map(|i| rx[i] * self.r[i] + (Scalar::ONE - rx[i]) * (Scalar::ONE - self.r[i]))
      .fold(Scalar::ONE, |acc, item| acc * item)
  }

  /// Evaluates the `EqPolynomial` at all the `2^|r|` points in its domain.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals(&self) -> Vec<Scalar> {
    let ell = self.r.len();
    let mut evals: Vec<Scalar> = vec![Scalar::ZERO; (2_usize).pow(ell as u32)];
    let mut size = 1;
    evals[0] = Scalar::ONE;

    for r in self.r.iter().rev() {
      let (evals_left, evals_right) = evals.split_at_mut(size);
      let (evals_right, _) = evals_right.split_at_mut(size);

      evals_left
        .par_iter_mut()
        .zip(evals_right.par_iter_mut())
        .for_each(|(x, y)| {
          *y = *x * r;
          *x -= &*y;
        });

      size *= 2;
    }

    evals
  }
}

/// A multilinear extension of a polynomial $Z(\cdot)$, denote it as $\tilde{Z}(x_1, ..., x_m)$
/// where the degree of each variable is at most one.
///
/// This is the dense representation of a multilinear poynomial.
/// Let it be $\mathbb{G}(\cdot): \mathbb{F}^m \rightarrow \mathbb{F}$, it can be represented uniquely by the list of
/// evaluations of $\mathbb{G}(\cdot)$ over the Boolean hypercube $\{0, 1\}^m$.
///
/// For example, a 3 variables multilinear polynomial can be represented by evaluation
/// at points $[0, 2^3-1]$.
///
/// The implementation follows
/// $$
/// \tilde{Z}(x_1, ..., x_m) = \sum_{e\in {0,1}^m}Z(e)\cdot \prod_{i=0}^m(x_i\cdot e_i)\cdot (1-e_i)
/// $$
///
/// Vector $Z$ indicates $Z(e)$ where $e$ ranges from $0$ to $2^m-1$.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultilinearPolynomial<Scalar: PrimeField> {
  num_vars: usize,           // the number of variables in the multilinear polynomial
  pub(crate) Z: Vec<Scalar>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<Scalar: PrimeField> MultilinearPolynomial<Scalar> {
  /// Creates a new MultilinearPolynomial from the given evaluations.
  ///
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<Scalar>) -> Self {
    assert_eq!(Z.len(), (2_usize).pow((Z.len() as f64).log2() as u32));
    MultilinearPolynomial {
      num_vars: (Z.len() as f64).log2() as usize,
      Z,
    }
  }

  /// Returns the number of variables in the multilinear polynomial
  pub fn get_num_vars(&self) -> usize {
    self.num_vars
  }

  /// Returns the total number of evaluations.
  pub fn len(&self) -> usize {
    self.Z.len()
  }

  /// Checks if the multilinear polynomial is empty.
  ///
  /// This method returns true if the polynomial has no evaluations, and false otherwise.
  pub fn is_empty(&self) -> bool {
    self.Z.is_empty()
  }

  /// Bounds the polynomial's top variable using the given scalar.
  ///
  /// This operation modifies the polynomial in-place.
  pub fn bound_poly_var_top(&mut self, r: &Scalar) {
    let n = self.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);
    let (right, _) = right.split_at(n);

    left
      .par_iter_mut()
      .zip(right.par_iter())
      .for_each(|(a, b)| {
        *a += *r * (*b - *a);
      });

    self.Z.resize(n, Scalar::ZERO);
    self.num_vars -= 1;
  }

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time.
  ///
  /// The point must have a value for each variable.
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    // r must have a value for each variable
    assert_eq!(r.len(), self.get_num_vars());
    let chis = EqPolynomial::new(r.to_vec()).evals();
    assert_eq!(chis.len(), self.Z.len());

    (0..chis.len())
      .into_par_iter()
      .map(|i| chis[i] * self.Z[i])
      .reduce(|| Scalar::ZERO, |x, y| x + y)
  }

  /// Evaluates the polynomial with the given evaluations and point.
  pub fn evaluate_with(Z: &[Scalar], r: &[Scalar]) -> Scalar {
    EqPolynomial::new(r.to_vec())
      .evals()
      .into_par_iter()
      .zip(Z.into_par_iter())
      .map(|(a, b)| a * b)
      .reduce(|| Scalar::ZERO, |x, y| x + y)
  }

  /// Multiplies the polynomial by a scalar.
  #[allow(unused)]
  pub fn scalar_mul(&self, scalar: &Scalar) -> Self {
    let mut new_poly = self.clone();
    for z in &mut new_poly.Z {
      *z *= scalar;
    }
    new_poly
  }
}

impl<Scalar: PrimeField> Index<usize> for MultilinearPolynomial<Scalar> {
  type Output = Scalar;

  #[inline(always)]
  fn index(&self, _index: usize) -> &Scalar {
    &(self.Z[_index])
  }
}

/// Sparse multilinear polynomial, which means the $Z(\cdot)$ is zero at most points.
/// So we do not have to store every evaluations of $Z(\cdot)$, only store the non-zero points.
///
/// For example, the evaluations are [0, 0, 0, 1, 0, 1, 0, 2].
/// The sparse polynomial only store the non-zero values, [(3, 1), (5, 1), (7, 2)].
/// In the tuple, the first is index, the second is value.
pub(crate) struct SparsePolynomial<Scalar: PrimeField> {
  num_vars: usize,
  Z: Vec<(usize, Scalar)>,
}

impl<Scalar: PrimeField> SparsePolynomial<Scalar> {
  pub fn new(num_vars: usize, Z: Vec<(usize, Scalar)>) -> Self {
    SparsePolynomial { num_vars, Z }
  }

  /// Computes the $\tilde{eq}$ extension polynomial.
  /// return 1 when a == r, otherwise return 0.
  fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
    assert_eq!(a.len(), r.len());
    let mut chi_i = Scalar::ONE;
    for j in 0..r.len() {
      if a[j] {
        chi_i *= r[j];
      } else {
        chi_i *= Scalar::ONE - r[j];
      }
    }
    chi_i
  }

  // Takes O(n log n)
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars, r.len());

    (0..self.Z.len())
      .into_par_iter()
      .map(|i| {
        let bits = (self.Z[i].0).get_bits(r.len());
        SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
      })
      .reduce(|| Scalar::ZERO, |x, y| x + y)
  }
}

/// Adds another multilinear polynomial to `self`.
/// Assumes the two polynomials have the same number of variables.
impl<Scalar: PrimeField> Add for MultilinearPolynomial<Scalar> {
  type Output = Result<Self, &'static str>;

  fn add(self, other: Self) -> Self::Output {
    if self.get_num_vars() != other.get_num_vars() {
      return Err("The two polynomials must have the same number of variables");
    }

    let sum: Vec<Scalar> = self
      .Z
      .iter()
      .zip(other.Z.iter())
      .map(|(a, b)| *a + *b)
      .collect();

    Ok(MultilinearPolynomial::new(sum))
  }
}

/// Multiplies `self` by another multilinear polynomial.
/// Assumes the two polynomials have the same number of variables.
impl<Scalar: PrimeField> Mul for MultilinearPolynomial<Scalar> {
  type Output = Result<Self, &'static str>;

  fn mul(self, other: Self) -> Self::Output {
    if self.get_num_vars() != other.get_num_vars() {
      return Err("The two polynomials must have the same number of variables");
    }

    let product: Vec<Scalar> = self
      .Z
      .iter()
      .zip(other.Z.iter())
      .map(|(a, b)| *a * *b)
      .collect();

    Ok(MultilinearPolynomial::new(product))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use pasta_curves::Fp;

  fn make_mlp<F: PrimeField>(len: usize, value: F) -> MultilinearPolynomial<F> {
    MultilinearPolynomial {
      num_vars: len.count_ones() as usize,
      Z: vec![value; len],
    }
  }

  fn test_eq_polynomial_with<F: PrimeField>() {
    let eq_poly = EqPolynomial::<F>::new(vec![F::ONE, F::ZERO, F::ONE]);
    let y = eq_poly.evaluate(vec![F::ONE, F::ONE, F::ONE].as_slice());
    assert_eq!(y, F::ZERO);

    let y = eq_poly.evaluate(vec![F::ONE, F::ZERO, F::ONE].as_slice());
    assert_eq!(y, F::ONE);

    let eval_list = eq_poly.evals();
    for (i, &coeff) in eval_list.iter().enumerate().take((2_usize).pow(3)) {
      if i == 5 {
        assert_eq!(coeff, F::ONE);
      } else {
        assert_eq!(coeff, F::ZERO);
      }
    }
  }

  fn test_multilinear_polynomial_with<F: PrimeField>() {
    // Let the polynomial has 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

    let TWO = F::from(2);

    let Z = vec![
      F::ZERO,
      F::ZERO,
      F::ZERO,
      F::ONE,
      F::ZERO,
      F::ONE,
      F::ZERO,
      TWO,
    ];
    let m_poly = MultilinearPolynomial::<F>::new(Z.clone());
    assert_eq!(m_poly.get_num_vars(), 3);

    let x = vec![F::ONE, F::ONE, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), TWO);

    let y = MultilinearPolynomial::<F>::evaluate_with(Z.as_slice(), x.as_slice());
    assert_eq!(y, TWO);
  }

  fn test_sparse_polynomial_with<F: PrimeField>() {
    // Let the polynomial have 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

    let TWO = F::from(2);
    let Z = vec![(3, F::ONE), (5, F::ONE), (7, TWO)];
    let m_poly = SparsePolynomial::<F>::new(3, Z);

    let x = vec![F::ONE, F::ONE, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), TWO);

    let x = vec![F::ONE, F::ZERO, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), F::ONE);
  }

  #[test]
  fn test_eq_polynomial() {
    test_eq_polynomial_with::<Fp>();
  }

  #[test]
  fn test_multilinear_polynomial() {
    test_multilinear_polynomial_with::<Fp>();
  }

  #[test]
  fn test_sparse_polynomial() {
    test_sparse_polynomial_with::<Fp>();
  }

  fn test_mlp_add_with<F: PrimeField>() {
    let mlp1 = make_mlp(4, F::from(3));
    let mlp2 = make_mlp(4, F::from(7));

    let mlp3 = mlp1.add(mlp2).unwrap();

    assert_eq!(mlp3.Z, vec![F::from(10); 4]);
  }

  fn test_mlp_scalar_mul_with<F: PrimeField>() {
    let mlp = make_mlp(4, F::from(3));

    let mlp2 = mlp.scalar_mul(&F::from(2));

    assert_eq!(mlp2.Z, vec![F::from(6); 4]);
  }

  fn test_mlp_mul_with<F: PrimeField>() {
    let mlp1 = make_mlp(4, F::from(3));
    let mlp2 = make_mlp(4, F::from(7));

    let mlp3 = mlp1.mul(mlp2).unwrap();

    assert_eq!(mlp3.Z, vec![F::from(21); 4]);
  }

  #[test]
  fn test_mlp_add() {
    test_mlp_add_with::<Fp>();
  }

  #[test]
  fn test_mlp_scalar_mul() {
    test_mlp_scalar_mul_with::<Fp>();
  }

  #[test]
  fn test_mlp_mul() {
    test_mlp_mul_with::<Fp>();
  }
}
