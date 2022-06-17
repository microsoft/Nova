use core::ops::Index;
use ff::PrimeField;

pub struct EqPolynomial<Scalar: PrimeField> {
  r: Vec<Scalar>,
}

impl<Scalar: PrimeField> EqPolynomial<Scalar> {
  pub fn new(r: Vec<Scalar>) -> Self {
    EqPolynomial { r }
  }

  pub fn evaluate(&self, rx: &[Scalar]) -> Scalar {
    assert_eq!(self.r.len(), rx.len());
    let mut prod = Scalar::one();
    for i in 0..rx.len() {
      prod = prod * (self.r[i] * rx[i] + (Scalar::one() - self.r[i]) * (Scalar::one() - rx[i]));
    }
    prod
    // (0..rx.len())
    //  .map(|i| self.r[i] * rx[i] + (Scalar::one() - self.r[i]) * (Scalar::one() - rx[i]))
    //  .product()
  }

  pub fn evals(&self) -> Vec<Scalar> {
    let ell = self.r.len();

    let mut evals: Vec<Scalar> = vec![Scalar::one(); (2 as usize).pow(ell as u32) as usize];
    let mut size = 1;
    for j in 0..ell {
      // in each iteration, we double the size of chis
      size *= 2;
      for i in (0..size).rev().step_by(2) {
        // copy each element from the prior iteration twice
        let scalar = evals[i / 2];
        evals[i] = scalar * self.r[j];
        evals[i - 1] = scalar - evals[i];
      }
    }
    evals
  }
}

#[derive(Debug)]
pub struct MultilinearPolynomial<Scalar: PrimeField> {
  num_vars: usize, // the number of variables in the multilinear polynomial
  len: usize,
  Z: Vec<Scalar>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<Scalar: PrimeField> MultilinearPolynomial<Scalar> {
  pub fn new(Z: Vec<Scalar>) -> Self {
    MultilinearPolynomial {
      num_vars: Z.len().log2() as usize,
      len: Z.len(),
      Z,
    }
  }

  pub fn get_num_vars(&self) -> usize {
    self.num_vars
  }

  pub fn len(&self) -> usize {
    self.len
  }

  pub fn clone(&self) -> MultilinearPolynomial<Scalar> {
    Self::new(self.Z[0..self.len].to_vec())
  }

  pub fn split(
    &self,
    idx: usize,
  ) -> (MultilinearPolynomial<Scalar>, MultilinearPolynomial<Scalar>) {
    assert!(idx < self.len());
    (
      MultilinearPolynomial::new(self.Z[..idx].to_vec()),
      MultilinearPolynomial::new(self.Z[idx..2 * idx].to_vec()),
    )
  }

  pub fn bound_poly_var_top(&mut self, r: &Scalar) {
    let n = self.len() / 2;
    for i in 0..n {
      self.Z[i] = self.Z[i] + *r * (self.Z[i + n] - self.Z[i]);
    }
    self.num_vars -= 1;
    self.len = n;
  }

  pub fn bound_poly_var_bot(&mut self, r: &Scalar) {
    let n = self.len() / 2;
    for i in 0..n {
      self.Z[i] = self.Z[2 * i] + *r * (self.Z[2 * i + 1] - self.Z[2 * i]);
    }
    self.num_vars -= 1;
    self.len = n;
  }

  // returns Z(r) in O(n) time
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    // r must have a value for each variable
    assert_eq!(r.len(), self.get_num_vars());
    let chis = EqPolynomial::new(r.to_vec()).evals();
    assert_eq!(chis.len(), self.Z.len());
    let mut sum = Scalar::zero();
    for i in 0..self.Z.len() {
      sum = sum + chis[i] * self.Z[i];
    }
    sum
  }

  fn vec(&self) -> &Vec<Scalar> {
    &self.Z
  }
}

impl<Scalar: PrimeField> Index<usize> for MultilinearPolynomial<Scalar> {
  type Output = Scalar;

  #[inline(always)]
  fn index(&self, _index: usize) -> &Scalar {
    &(self.Z[_index])
  }
}
