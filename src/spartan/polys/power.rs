//! `PowPolynomial`: Represents multilinear extension of power polynomials

use crate::spartan::polys::eq::EqPolynomial;
use ff::PrimeField;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $pow(x,t)$, denoted as $\tilde{pow}(x, t)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{power}(x, t) = \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
/// $$
pub struct PowPolynomial<Scalar: PrimeField> {
  t_pow: Vec<Scalar>,
  eq: EqPolynomial<Scalar>,
}

impl<Scalar: PrimeField> PowPolynomial<Scalar> {
  /// Creates a new `PowPolynomial` from a Scalars `t`.
  pub fn new(t: &Scalar, ell: usize) -> Self {
    // t_pow = [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
    let mut t_pow = vec![Scalar::ONE; ell];
    t_pow[0] = *t;
    for i in 1..ell {
      t_pow[i] = t_pow[i - 1].square();
    }

    PowPolynomial {
      t_pow: t_pow.clone(),
      eq: EqPolynomial::new(t_pow),
    }
  }

  /// Evaluates the `PowPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `t_pow`.
  ///
  /// Panics if `rx` and `t_pow` have different lengths.
  pub fn evaluate(&self, rx: &[Scalar]) -> Scalar {
    self.eq.evaluate(rx)
  }

  pub fn coordinates(&self) -> Vec<Scalar> {
    self.t_pow.clone()
  }

  /// Evaluates the `PowPolynomial` at all the `2^|t_pow|` points in its domain.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals(&self) -> Vec<Scalar> {
    self.eq.evals()
  }
}
