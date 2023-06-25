//! This module defines basic types related to Boolean hypercubes.
use std::marker::PhantomData;

use crate::utils::*;
/// There's some overlap with polynomial.rs.
use ff::PrimeField;

#[derive(Debug)]
pub(crate) struct BooleanHypercube<F: PrimeField> {
  n_vars: usize,
  current: u64,
  max: u64,
  _f: PhantomData<F>,
}

impl<F: PrimeField> BooleanHypercube<F> {
  pub(crate) fn new(n_vars: usize) -> Self {
    Self {
      _f: PhantomData::<F>,
      n_vars,
      current: 0,
      max: 2_u32.pow(n_vars as u32) as u64,
    }
  }

  /// returns the entry at given i (which is the little-endian bit representation of i)
  pub(crate) fn evaluate_at(&self, i: usize) -> Vec<F> {
    assert!(i < self.max as usize);
    let bits = bit_decompose((i) as u64, self.n_vars);
    bits.iter().map(|&x| F::from(x as u64)).collect()
  }
}

impl<Scalar: PrimeField> Iterator for BooleanHypercube<Scalar> {
  type Item = Vec<Scalar>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.current > self.max {
      None
    } else {
      let bits = bit_decompose(self.current, self.n_vars);
      let point: Vec<Scalar> = bits.iter().map(|&bit| Scalar::from(bit as u64)).collect();
      self.current += 1;
      Some(point)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use pasta_curves::Fq;

  #[test]
  fn test_evaluate() {
    // Declare the coefficients in the order 1, x, y, xy, z, xz, yz, xyz.
    let poly = BooleanHypercube::<Fq>::new(3);

    let point = 7usize;
    // So, f(1, 1, 1) = 5.
    assert_eq!(poly.evaluate_at(point), vec![Fq::ONE, Fq::ONE, Fq::ONE]);
  }
}
