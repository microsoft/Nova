use core::marker::PhantomData;
use ff::PrimeField;

/// Represents an identity polynomial that maps a point to its index in binary representation.
pub struct IdentityPolynomial<Scalar: PrimeField> {
  ell: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField> IdentityPolynomial<Scalar> {
  /// Creates a new identity polynomial with the specified number of variables.
  pub fn new(ell: usize) -> Self {
    IdentityPolynomial {
      ell,
      _p: PhantomData,
    }
  }

  /// Evaluates the identity polynomial at the given point.
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.ell, r.len());
    let mut power_of_two = 1_u64;
    (0..self.ell)
      .rev()
      .map(|i| {
        let result = Scalar::from(power_of_two) * r[i];
        power_of_two *= 2;
        result
      })
      .fold(Scalar::ZERO, |acc, item| acc + item)
  }
}
