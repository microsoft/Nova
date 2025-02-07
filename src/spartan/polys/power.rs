//! `PowPolynomial`: Represents multilinear extension of power polynomials

use core::iter::successors;
use ff::PrimeField;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $pow(x,t)$, denoted as $\tilde{pow}(x, t)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{power}(x, t) = \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
/// $$
pub struct PowPolynomial<Scalar: PrimeField> {
  t_pow: Vec<Scalar>,
}

#[allow(dead_code)]
impl<Scalar: PrimeField> PowPolynomial<Scalar> {
  /// Creates a new `PowPolynomial` from a Scalars `t`.
  pub fn new(t: &Scalar, ell: usize) -> Self {
    // t_pow = [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
    let t_pow = successors(Some(*t), |p: &Scalar| Some(p.square()))
      .take(ell)
      .collect::<Vec<_>>();

    PowPolynomial { t_pow }
  }

  /// Evaluates the `PowPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `t_pow`.
  ///
  /// Panics if `rx` and `t_pow` have different lengths.
  pub fn evaluate(&self, rx: &[Scalar]) -> Scalar {
    assert_eq!(rx.len(), self.t_pow.len());

    // compute the polynomial evaluation using \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
    let mut result = Scalar::ONE;
    for (t_pow, x) in self.t_pow.iter().zip(rx.iter()) {
      result *= Scalar::ONE + (*t_pow - Scalar::ONE) * x;
    }
    result
  }

  pub fn coordinates(self) -> Vec<Scalar> {
    self.t_pow
  }

  /// Evaluates the `PowPolynomial` at all the `2^|t_pow|` points in its domain.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals(&self) -> Vec<Scalar> {
    (0..(1 << self.t_pow.len()))
      .scan(Scalar::ONE, |state, _| {
        let current = *state;
        *state *= self.t_pow[0];
        Some(current)
      })
      .collect::<Vec<_>>()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{bn256_grumpkin::bn256, pasta::pallas, secp_secq::secp256k1};
  use rand::rngs::OsRng;

  fn test_evals_with<Scalar: PrimeField>() {
    let t = Scalar::random(&mut OsRng);
    let ell = 4;
    let pow = PowPolynomial::new(&t, ell);

    // compute evaluations which should be of length 2^ell
    let evals = pow.evals();
    assert_eq!(evals.len(), 1 << ell);

    let mut evals_alt = vec![Scalar::ONE; 1 << ell];
    evals_alt[0] = Scalar::ONE;
    for i in 1..(1 << ell) {
      evals_alt[i] = evals_alt[i - 1] * t;
    }
    for i in 0..(1 << ell) {
      if evals[i] != evals_alt[i] {
        println!(
          "Mismatch at index {}: expected {:?}, got {:?}",
          i, evals_alt[i], evals[i]
        );
      }
      assert_eq!(evals[i], evals_alt[i]);
    }
  }

  #[test]
  fn test_evals() {
    test_evals_with::<bn256::Scalar>();
    test_evals_with::<pallas::Scalar>();
    test_evals_with::<secp256k1::Scalar>();
  }
}
