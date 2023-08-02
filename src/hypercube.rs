//! This module defines basic types related to Boolean hypercubes.
#![allow(unused)]
use std::marker::PhantomData;

/// There's some overlap with polynomial.rs.
use ff::PrimeField;
use itertools::Itertools;

#[derive(Debug)]
pub(crate) struct BooleanHypercube<F: PrimeField> {
  pub(crate) n_vars: usize,
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

  /// returns the entry at given i (which is the big-endian bit representation of i)
  pub(crate) fn evaluate_at_big(&self, i: usize) -> Vec<F> {
    assert!(i < self.max as usize);
    let bits = bit_decompose((i) as u64, self.n_vars);
    bits.iter().map(|&x| F::from(x as u64)).collect()
  }

  /// returns the entry at given i (which is the little-endian bit representation of i)
  pub(crate) fn evaluate_at_little(&self, i: usize) -> Vec<F> {
    assert!(i < self.max as usize);
    let bits = bit_decompose((i) as u64, self.n_vars);
    bits.iter().map(|&x| F::from(x as u64)).rev().collect()
  }

  pub(crate) fn evaluate_at(&self, i: usize) -> Vec<F> {
    // This is what we are currently using
    self.evaluate_at_little(i)
  }
}

impl<Scalar: PrimeField> Iterator for BooleanHypercube<Scalar> {
  type Item = Vec<Scalar>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.current >= self.max {
      None
    } else {
      let bits = bit_decompose(self.current, self.n_vars);
      let point: Vec<Scalar> = bits.iter().map(|&bit| Scalar::from(bit as u64)).collect();
      self.current += 1;
      Some(point)
    }
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

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use pasta_curves::Fq;

  fn test_evaluate_with<F: PrimeField>() {
    let poly = BooleanHypercube::<F>::new(3);

    let point = 7usize;
    // So, f(1, 1, 1) = 5.
    assert_eq!(poly.evaluate_at(point), vec![F::ONE, F::ONE, F::ONE]);
  }

  fn test_big_endian_eval_with<F: PrimeField>() {
    let mut hypercube = BooleanHypercube::<F>::new(3);

    let expected_outputs = vec![
      vec![F::ZERO, F::ZERO, F::ZERO],
      vec![F::ONE, F::ZERO, F::ZERO],
      vec![F::ZERO, F::ONE, F::ZERO],
      vec![F::ONE, F::ONE, F::ZERO],
      vec![F::ZERO, F::ZERO, F::ONE],
      vec![F::ONE, F::ZERO, F::ONE],
      vec![F::ZERO, F::ONE, F::ONE],
      vec![F::ONE, F::ONE, F::ONE],
    ];

    for (i, _) in expected_outputs
      .iter()
      .enumerate()
      .take(hypercube.max as usize)
    {
      assert_eq!(hypercube.evaluate_at_big(i), expected_outputs[i]);
    }
  }

  fn test_big_endian_next_with<F: PrimeField>() {
    let mut hypercube = BooleanHypercube::<F>::new(3);

    let expected_outputs = vec![
      vec![F::ZERO, F::ZERO, F::ZERO],
      vec![F::ONE, F::ZERO, F::ZERO],
      vec![F::ZERO, F::ONE, F::ZERO],
      vec![F::ONE, F::ONE, F::ZERO],
      vec![F::ZERO, F::ZERO, F::ONE],
      vec![F::ONE, F::ZERO, F::ONE],
      vec![F::ZERO, F::ONE, F::ONE],
      vec![F::ONE, F::ONE, F::ONE],
    ];

    for expected_output in expected_outputs {
      let actual_output = hypercube.next().unwrap();
      assert_eq!(actual_output, expected_output);
    }
  }

  fn test_little_endian_eval_with<F: PrimeField>() {
    let mut hypercube = BooleanHypercube::<F>::new(3);

    let expected_outputs = vec![
      vec![F::ZERO, F::ZERO, F::ZERO],
      vec![F::ZERO, F::ZERO, F::ONE],
      vec![F::ZERO, F::ONE, F::ZERO],
      vec![F::ZERO, F::ONE, F::ONE],
      vec![F::ONE, F::ZERO, F::ZERO],
      vec![F::ONE, F::ZERO, F::ONE],
      vec![F::ONE, F::ONE, F::ZERO],
      vec![F::ONE, F::ONE, F::ONE],
    ];

    for (i, _) in expected_outputs
      .iter()
      .enumerate()
      .take(hypercube.max as usize)
    {
      assert_eq!(hypercube.evaluate_at_little(i), expected_outputs[i]);
    }
  }

  #[test]
  fn test_evaluate() {
    test_evaluate_with::<Fq>();
  }
  #[test]
  fn test_big_endian_eval() {
    test_big_endian_eval_with::<Fq>();
  }

  #[test]
  fn test_big_endian_next() {
    test_big_endian_next_with::<Fq>();
  }

  #[test]
  fn test_little_endian_eval() {
    test_little_endian_eval_with::<Fq>();
  }
}
