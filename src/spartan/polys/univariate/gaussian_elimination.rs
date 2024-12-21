//! # NOTE: this code was originally from: https://github.com/a16z/jolt/blob/f1e5ab3cb2f55eb49c17c249039294184d27fdba/jolt-core/src/poly/unipoly.rs#L38

// Inspired by: https://github.com/TheAlgorithms/Rust/blob/master/src/math/gaussian_elimination.rs
// Gaussian Elimination of Quadratic Matrices
// Takes an augmented matrix as input, returns vector of results
// Wikipedia reference: augmented matrix: https://en.wikipedia.org/wiki/Augmented_matrix
// Wikipedia reference: algorithm: https://en.wikipedia.org/wiki/Gaussian_elimination

use core::panic;

use ff::PrimeField;

pub fn gaussian_elimination<F: PrimeField>(matrix: &mut [Vec<F>]) -> Vec<F> {
  let size = matrix.len();
  assert_eq!(size, matrix[0].len() - 1);

  for i in 0..size - 1 {
    for j in i..size - 1 {
      echelon(matrix, i, j);
    }
  }

  for i in (1..size).rev() {
    eliminate(matrix, i);
  }

  // Disable cargo clippy warnings about needless range loops.
  // Checking the diagonal like this is simpler than any alternative.
  #[allow(clippy::needless_range_loop)]
  for i in 0..size {
    if matrix[i][i] == F::ZERO {
      println!("Infinitely many solutions");
    }
  }

  let mut result: Vec<F> = vec![F::ZERO; size];
  for i in 0..size {
    result[i] = div_f(matrix[i][size], matrix[i][i]);
  }

  result
}

fn echelon<F: PrimeField>(matrix: &mut [Vec<F>], i: usize, j: usize) {
  let size = matrix.len();
  if matrix[i][i] == F::ZERO {
  } else {
    let factor = div_f(matrix[j + 1][i], matrix[i][i]);
    (i..size + 1).for_each(|k| {
      let tmp = matrix[i][k];
      matrix[j + 1][k] -= factor * tmp;
    });
  }
}

fn eliminate<F: PrimeField>(matrix: &mut [Vec<F>], i: usize) {
  let size = matrix.len();
  if matrix[i][i] == F::ZERO {
  } else {
    for j in (1..i + 1).rev() {
      let factor = div_f(matrix[j - 1][i], matrix[i][i]);
      for k in (0..size + 1).rev() {
        let tmp = matrix[i][k];
        matrix[j - 1][k] -= factor * tmp;
      }
    }
  }
}

/// Division of two prime fields
///
/// # Panics
///
/// Panics if `b` is zero.
pub fn div_f<F: PrimeField>(a: F, b: F) -> F {
  let inverse_b = b.invert();

  if inverse_b.into_option().is_none() {
    panic!("Division by zero");
  }

  a * inverse_b.unwrap()
}
