//! `PowPolynomial`: Represents multilinear extension of power polynomials

use ff::PrimeField;
//use std::iter::successors;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $pow(x,t)$, denoted as $\tilde{pow}(x, t)$.
///
/// The polynomial is defined by the formula:
/// $$
/// \tilde{power}(x, t) = \prod_{i=1}^m(1 + (t^{2^i} - 1) * x_i)
/// $$
pub struct PowPolynomial<Scalar: PrimeField> {
  t: Scalar,
  t_pow: Vec<Scalar>,
}

#[allow(dead_code)]
impl<Scalar: PrimeField> PowPolynomial<Scalar> {
  /// Creates a new `PowPolynomial` from a Scalars `t`.
  pub fn new(t: &Scalar, ell: usize) -> Self {
    // t_pow = [t^{2^0}, t^{2^1}, ..., t^{2^{ell-1}}]
    let mut t_pow = vec![Scalar::ONE; ell];
    t_pow[0] = *t;
    for i in 1..ell {
      t_pow[i] = t_pow[i-1].square();
    }

    /*let t_pow = successors(Some(*t), |p: &Scalar| Some(p.square()))
      .take(ell)
      .collect::<Vec<_>>();*/

    PowPolynomial {
      t: *t,
      t_pow
    }
  }

  /// Evaluates the `PowPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `t_pow`.
  ///
  /// Panics if `rx` and `t_pow` have different lengths.
  #[allow(dead_code)]
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
    let mut powers = vec![Scalar::ONE; 1 << self.t_pow.len()];
    powers[0] = Scalar::ONE;
    for i in 1..(1 << self.t_pow.len()) {
      powers[i] = powers[i-1] * self.t;
    }
    powers
  }

  /// Computes two vectors such that their outer product equals the output of the `evals` function.
  pub fn split_evals(&self) -> (Vec<Scalar>, Vec<Scalar>) {
    // Compute the number of elements in the left and right halves
    let ell = self.t_pow.len();
    let (len_left, len_right) = (1 << ell/2, 1 << (ell - ell/2));

    // Compute the left and right halves of the evaluations
    // left = [1, t, t^2, ..., t^{2^{ell/2} - 1}]
    //let left = successors(Some(Scalar::ONE), |p| Some(*p * self.t))
    //  .take(len_left)
    //  .collect::<Vec<_>>();
    let mut left = vec![Scalar::ONE; len_left];
    left[0] = Scalar::ONE;
    for i in 1..len_left {
      left[i] = left[i-1] * self.t;
    }


    // right = [1, t^{2^{ell/2}}, t^{2^{ell/2 + 1}}, ..., t^{2^{ell} - 1}]
    // take the last entry from left, multiply with t to get the second entry in right
    let left_last_times_t = left[left.len()-1] * self.t;
    let mut right = vec![Scalar::ONE; len_right];
    right[0] = Scalar::ONE;
    right[1] = left_last_times_t;
    for i in 2..len_right {
      right[i] = right[i-1] * left_last_times_t;
    }

    (left, right)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{bn256_grumpkin::bn256, pasta::pallas, secp_secq::secp256k1};
  use rand::rngs::OsRng;

  fn test_split_evals_with<Scalar: PrimeField>() {
    let t = Scalar::random(&mut OsRng);
    let ell = 4;
    let pow = PowPolynomial::new(&t, ell);

    // compute evaluations which should be of length 2^ell
    let evals = pow.evals();
    assert_eq!(evals.len(), 1 << ell);

    let mut evals_alt = vec![Scalar::ONE; 1 << ell];
    evals_alt[0] = Scalar::ONE;
    for i in 1..(1 << ell) {
      evals_alt[i] = evals_alt[i-1] * t;
    }
    for i in 0..(1 << ell) {
      if evals[i] != evals_alt[i] {
        println!("Mismatch at index {}: expected {:?}, got {:?}", i, evals_alt[i], evals[i]);
      }
      assert_eq!(evals[i], evals_alt[i]);
    }

    // now compute split evals
    let (left, right) = pow.split_evals();
    assert_eq!(left.len() * right.len(), evals.len());

    // check that the outer product of left and right equals evals
    let mut evals_iter = evals.iter();
    for (i, l) in right.iter().enumerate() {
      for (j, r) in left.iter().enumerate() {
      let eval = evals_iter.next().unwrap();
      if eval != &(*l * r) {
        println!("Mismatch at left index {}, right index {}: expected {:?}, got {:?}", i, j, *l * r, eval);
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
