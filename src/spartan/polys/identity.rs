use core::marker::PhantomData;
use ff::PrimeField;

pub struct IdentityPolynomial<Scalar: PrimeField> {
  ell: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField> IdentityPolynomial<Scalar> {
  pub fn new(ell: usize) -> Self {
    IdentityPolynomial {
      ell,
      _p: PhantomData,
    }
  }

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
