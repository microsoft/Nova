//! `PowPolynomial`: Represents multilinear extension of power polynomials

use crate::spartan::polys::eq::EqPolynomial;
use ff::PrimeField;
use std::iter::successors;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $pow(x,t)$, denoted as $\tilde{pow}(x, t)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{power}(x, t) = \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
/// $$
pub struct PowPolynomial<Scalar: PrimeField> {
  eq: EqPolynomial<Scalar>,
}

impl<Scalar: PrimeField> PowPolynomial<Scalar> {
  /// Creates a new `PowPolynomial` from a Scalars `t`.
  pub fn new(t: &Scalar, ell: usize) -> Self {
    // t_pow = [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
    let t_pow = Self::squares(t, ell);

    PowPolynomial {
      eq: EqPolynomial::new(t_pow),
    }
  }

  /// Create powers the following powers of `t`:
  /// [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
  pub fn squares(t: &Scalar, ell: usize) -> Vec<Scalar> {
    successors(Some(*t), |p: &Scalar| Some(p.square()))
      .take(ell)
      .collect::<Vec<_>>()
  }

  /// Evaluates the `PowPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `t_pow`.
  ///
  /// Panics if `rx` and `t_pow` have different lengths.
  #[allow(dead_code)]
  pub fn evaluate(&self, rx: &[Scalar]) -> Scalar {
    self.eq.evaluate(rx)
  }

  pub fn coordinates(self) -> Vec<Scalar> {
    self.eq.r
  }

  /// Evaluates the `PowPolynomial` at all the `2^|t_pow|` points in its domain.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals(&self) -> Vec<Scalar> {
    self.eq.evals()
  }
}
