//! `MaskedEqPolynomial`: Represents the `eq` polynomial over n variables, where the first 2^m entries are 0.

use crate::spartan::polys::eq::EqPolynomial;
use ff::PrimeField;
use itertools::zip_eq;

/// Represents the multilinear extension polynomial (MLE) of the equality polynomial $eqₘ(x,r)$
/// over n variables, where the first 2^m evaluations are 0.
///
/// The polynomial is defined by the formula:
/// eqₘ(x,r) = eq(x,r) - ( ∏_{0 ≤ i < n-m} (1−rᵢ)(1−xᵢ) )⋅( ∏_{n-m ≤ i < n} (1−rᵢ)(1−xᵢ) + rᵢ⋅xᵢ )
#[derive(Debug)]
pub struct MaskedEqPolynomial<'a, Scalar: PrimeField> {
  eq: &'a EqPolynomial<Scalar>,
  num_masked_vars: usize,
}

impl<'a, Scalar: PrimeField> MaskedEqPolynomial<'a, Scalar> {
  /// Creates a new `MaskedEqPolynomial` from a vector of Scalars `r` of size n, with the number of
  /// masked variables m = `num_masked_vars`.
  pub const fn new(eq: &'a EqPolynomial<Scalar>, num_masked_vars: usize) -> Self {
    MaskedEqPolynomial {
      eq,
      num_masked_vars,
    }
  }

  /// Evaluates the `MaskedEqPolynomial` at a given point `rx`.
  ///
  /// This function computes the value of the polynomial at the point specified by `rx`.
  /// It expects `rx` to have the same length as the internal vector `r`.
  ///
  /// Panics if `rx` and `r` have different lengths.
  pub fn evaluate(&self, rx: &[Scalar]) -> Scalar {
    let r = &self.eq.r;
    assert_eq!(r.len(), rx.len());
    let split_idx = r.len() - self.num_masked_vars;

    let (r_lo, r_hi) = r.split_at(split_idx);
    let (rx_lo, rx_hi) = rx.split_at(split_idx);
    let eq_lo = zip_eq(r_lo, rx_lo)
      .map(|(r, rx)| *r * rx + (Scalar::ONE - r) * (Scalar::ONE - rx))
      .product::<Scalar>();
    let eq_hi = zip_eq(r_hi, rx_hi)
      .map(|(r, rx)| *r * rx + (Scalar::ONE - r) * (Scalar::ONE - rx))
      .product::<Scalar>();
    let mask_lo = zip_eq(r_lo, rx_lo)
      .map(|(r, rx)| (Scalar::ONE - r) * (Scalar::ONE - rx))
      .product::<Scalar>();

    (eq_lo - mask_lo) * eq_hi
  }

  /// Evaluates the `MaskedEqPolynomial` at all the `2^|r|` points in its domain.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  pub fn evals(&self) -> Vec<Scalar> {
    Self::evals_from_points(&self.eq.r, self.num_masked_vars)
  }

  /// Evaluates the `MaskedEqPolynomial` from the `2^|r|` points in its domain, without creating an intermediate polynomial
  /// representation.
  ///
  /// Returns a vector of Scalars, each corresponding to the polynomial evaluation at a specific point.
  fn evals_from_points(r: &[Scalar], num_masked_vars: usize) -> Vec<Scalar> {
    let mut evals = EqPolynomial::evals_from_points(r);

    // replace the first 2^m evaluations with 0
    let num_masked_evals = 1 << num_masked_vars;
    evals[..num_masked_evals]
      .iter_mut()
      .for_each(|e| *e = Scalar::ZERO);

    evals
  }
}

#[cfg(test)]
mod tests {
  use crate::provider;

  use super::*;
  use crate::spartan::polys::eq::EqPolynomial;
  use pasta_curves::Fp;
  use rand_chacha::ChaCha20Rng;
  use rand_core::{CryptoRng, RngCore, SeedableRng};

  fn test_masked_eq_polynomial_with<F: PrimeField, R: RngCore + CryptoRng>(
    num_vars: usize,
    num_masked_vars: usize,
    mut rng: &mut R,
  ) {
    let num_masked_evals = 1 << num_masked_vars;

    // random point
    let r = std::iter::from_fn(|| Some(F::random(&mut rng)))
      .take(num_vars)
      .collect::<Vec<_>>();
    // evaluation point
    let rx = std::iter::from_fn(|| Some(F::random(&mut rng)))
      .take(num_vars)
      .collect::<Vec<_>>();

    let poly_eq = EqPolynomial::new(r);
    let poly_eq_evals = poly_eq.evals();

    let masked_eq_poly = MaskedEqPolynomial::new(&poly_eq, num_masked_vars);
    let masked_eq_poly_evals = masked_eq_poly.evals();

    // ensure the first 2^m entries are 0
    assert_eq!(
      masked_eq_poly_evals[..num_masked_evals],
      vec![F::ZERO; num_masked_evals]
    );
    // ensure the remaining evaluations match eq(r)
    assert_eq!(
      masked_eq_poly_evals[num_masked_evals..],
      poly_eq_evals[num_masked_evals..]
    );

    // compute the evaluation at rx succinctly
    let masked_eq_eval = masked_eq_poly.evaluate(&rx);

    // compute the evaluation as a MLE
    let rx_evals = EqPolynomial::evals_from_points(&rx);
    let expected_masked_eq_eval = zip_eq(rx_evals, masked_eq_poly_evals)
      .map(|(rx, r)| rx * r)
      .sum();

    assert_eq!(masked_eq_eval, expected_masked_eq_eval);
  }

  #[test]
  fn test_masked_eq_polynomial() {
    let mut rng = ChaCha20Rng::from_seed([0u8; 32]);
    let num_vars = 5;
    let num_masked_vars = 2;
    test_masked_eq_polynomial_with::<Fp, _>(num_vars, num_masked_vars, &mut rng);
    test_masked_eq_polynomial_with::<provider::bn256_grumpkin::bn256::Scalar, _>(
      num_vars,
      num_masked_vars,
      &mut rng,
    );
    test_masked_eq_polynomial_with::<provider::secp_secq::secp256k1::Scalar, _>(
      num_vars,
      num_masked_vars,
      &mut rng,
    );
  }
}
