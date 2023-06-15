//! This module defines basic types related to Boolean hypercubes.
/// There's some overlap with polynomial.rs.
use ff::PrimeField;

#[derive(Debug)]
pub struct BooleanHypercube<Scalar: PrimeField> {
  dimensions: usize,
  current: u64,
  max: u64,
  coefficients: Vec<Scalar>,
}

impl<Scalar: PrimeField> BooleanHypercube<Scalar> {
  pub fn new(dimensions: usize, coefficients: Vec<Scalar>) -> Self {
    assert!(coefficients.len() == 2_usize.pow(dimensions as u32));

    BooleanHypercube {
      dimensions,
      current: 0,
      max: 2_u32.pow(dimensions as u32) as u64,
      coefficients,
    }
  }

  // Evaluate the multilinear polynomial at the given point
  pub fn evaluate(&self, point: &[Scalar]) -> Scalar {
    assert!(point.len() == self.dimensions);

    let mut result = Scalar::ZERO;

    for i in 0..self.max as usize {
      let monomial = self.monomial(i, point);
      result = result + self.coefficients[i] * monomial;
    }

    result
  }

  // This calculates a single monomial of the multilinear polynomial
  fn monomial(&self, i: usize, point: &[Scalar]) -> Scalar {
    assert!(i < self.max as usize);
    let mut result = Scalar::ONE;

    let bits = bit_decompose(i as u64, self.dimensions);

    for j in 0..self.dimensions {
      if bits[j] {
        result = result * point[j];
      }
    }

    result
  }
}

/// Decompose an integer into a binary vector in little endian.
pub fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
  let mut res = Vec::with_capacity(num_var);
  let mut i = input;
  for _ in 0..num_var {
    res.push(i & 1 == 1);
    i >>= 1;
  }
  res
}

mod tests {
  use super::*;
  use pasta_curves::Fp;

  #[test]
  fn test_evaluate() {
    // Declare the coefficients in the order 1, x, y, xy, z, xz, yz, xyz.
    let poly = BooleanHypercube::<Fp>::new(
      3,
      vec![
        Fp::from(0u64),
        Fp::from(4u64),
        Fp::from(2u64),
        Fp::from(0u64),
        Fp::from(1u64),
        Fp::from(0u64),
        Fp::from(0u64),
        Fp::from(0u64),
      ],
    );

    let point = vec![Fp::from(1u64), Fp::from(1u64), Fp::from(1u64)];

    // The polynomial would be f(x, y, z) = 4x + 2y + z.
    // So, f(1, 1, 1) = 4*1 + 2*1 + 1 = 7.
    assert_eq!(poly.evaluate(&point), Fp::from(7u64));
  }
}
