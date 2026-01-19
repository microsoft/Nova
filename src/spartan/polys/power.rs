//! `PowPolynomial`: Represents multilinear extension of power polynomials

use core::iter::successors;
use ff::PrimeField;

/// Represents the multilinear extension polynomial (MLE) of the power polynomial $pow(x,t)$, denoted as $\tilde{pow}(x, t)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{power}(x, t) = \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
/// $$
pub struct PowPolynomial<Scalar: PrimeField> {
  t_pow: Vec<Scalar>,
}

impl<Scalar: PrimeField> PowPolynomial<Scalar> {
  /// Creates a new `PowPolynomial` from a scalar `t` and number of variables `ell`.
  ///
  /// The internal representation stores `[t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]`.
  pub fn new(t: &Scalar, ell: usize) -> Self {
    // t_pow = [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
    let t_pow = successors(Some(*t), |p: &Scalar| Some(p.square()))
      .take(ell)
      .collect::<Vec<_>>();

    PowPolynomial { t_pow }
  }

  /// Evaluates the `PowPolynomial` at all the `2^|t_pow|` points in its domain.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals(&self) -> Vec<Scalar> {
    successors(Some(Scalar::ONE), |p| Some(*p * self.t_pow[0]))
      .take(1 << self.t_pow.len())
      .collect::<Vec<_>>()
  }

  /// Returns the coordinates (powers of t) used in this polynomial.
  pub fn coordinates(&self) -> &[Scalar] {
    &self.t_pow
  }

  /// Computes two vectors such that their outer product equals the output of the [`evals`](Self::evals) function.
  ///
  /// # Parameters
  ///
  /// - `len_left`: Length of the first (left) vector factor. This must be chosen
  ///   together with `len_right` so that `len_left * len_right == 2^{|t_pow|}`,
  ///   where `|t_pow|` is the number of variables in the polynomial
  ///   (`self.t_pow.len()`). If this condition is not satisfied, the function
  ///   will panic due to the internal assertion.
  /// - `len_right`: Length of the second (right) vector factor. See `len_left`
  ///   for the required relation between the two lengths.
  ///
  /// # Returns
  ///
  /// A vector containing the concatenation of the two factor vectors:
  /// first all entries of the left vector, followed by all entries of the
  /// right vector. Conceptually, if `L` and `R` denote these two vectors,
  /// then their outer product `L ⊗ R` (viewed as a flattened vector) equals
  /// the evaluations returned by [`evals`](Self::evals).
  pub fn split_evals(&self, len_left: usize, len_right: usize) -> Vec<Scalar> {
    // Compute the number of elements in the left and right halves
    let ell = self.t_pow.len();
    assert_eq!(len_left * len_right, 1 << ell);

    let t = self.t_pow[0];

    // Compute the left and right halves of the evaluations
    // left = [1, t, t^2, ..., t^{2^{ell/2} - 1}]
    let left = successors(Some(Scalar::ONE), |p| Some(*p * t))
      .take(len_left)
      .collect::<Vec<_>>();

    // right = [1, t^{2^{ell/2}}, t^{2^{ell/2 + 1}}, ..., t^{2^{ell} - 1}]
    // take the last entry from left, multiply with t to get the second entry in right
    let left_last_times_t = left[left.len() - 1] * t;
    let mut right = vec![Scalar::ONE; len_right];
    right[0] = Scalar::ONE;
    right[1] = left_last_times_t;
    for i in 2..len_right {
      right[i] = right[i - 1] * left_last_times_t;
    }

    [left, right].concat()
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

  fn test_split_evals_with<Scalar: PrimeField>() {
    let t = Scalar::random(&mut OsRng);
    let ell = 4;
    let pow = PowPolynomial::new(&t, ell);

    // compute evaluations which should be of length 2^ell
    let evals = pow.evals();
    assert_eq!(evals.len(), 1 << ell);

    // now compute split evals
    let split_evals = pow.split_evals(1 << (ell / 2), 1 << (ell - ell / 2));
    let (left, right) = split_evals.split_at(1 << (ell / 2));

    // check that the outer product of left and right equals evals
    let mut evals_iter = evals.iter();
    for (i, l) in right.iter().enumerate() {
      for (j, r) in left.iter().enumerate() {
        let eval = evals_iter.next().unwrap();
        if eval != &(*l * r) {
          println!(
            "Mismatch at left index {}, right index {}: expected {:?}, got {:?}",
            i,
            j,
            *l * r,
            eval
          );
        }
        assert_eq!(eval, &(*l * r));
      }
    }
  }

  #[test]
  fn test_split_evals() {
    test_split_evals_with::<bn256::Scalar>();
    test_split_evals_with::<pallas::Scalar>();
    test_split_evals_with::<secp256k1::Scalar>();
  }
}
