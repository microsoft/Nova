//! Basic utils
use std::sync::Arc;

use crate::errors::NovaError;
use crate::spartan::polynomial::MultilinearPolynomial;
use crate::traits::Group;
use ff::{Field, PrimeField};
use rand_core::RngCore;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

/// A matrix structure represented on a sparse form.
/// First element is row index, second column, third value stored
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparseMatrix<G: Group> {
  n_rows: usize,
  n_cols: usize,
  coeffs: Vec<(usize, usize, G::Scalar)>,
}

impl<G: Group> SparseMatrix<G> {
  pub fn new(n_rows: usize, n_cols: usize) -> Self {
    Self {
      n_rows,
      n_cols,
      coeffs: vec![],
    }
  }

  pub fn with_coeffs(n_rows: usize, n_cols: usize, coeffs: Vec<(usize, usize, G::Scalar)>) -> Self {
    Self {
      n_rows,
      n_cols,
      coeffs,
    }
  }

  // Return the number of rows of this matrix.
  pub fn n_rows(&self) -> usize {
    self.n_rows
  }

  // Returns a mutable reference to the number of rows of this matrix.
  pub fn n_rows_mut(&mut self) -> &mut usize {
    &mut self.n_rows
  }

  // Return the number of cols of this matrix.
  pub fn n_cols(&self) -> usize {
    self.n_cols
  }

  // Returns a mutable reference to the number of cols of this matrix.
  pub fn n_cols_mut(&mut self) -> &mut usize {
    &mut self.n_cols
  }

  // Return the non-0 coefficients of this matrix.
  pub fn coeffs(&self) -> &[(usize, usize, G::Scalar)] {
    self.coeffs.as_slice()
  }

  pub(crate) fn is_valid(
    &self,
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
  ) -> Result<(), NovaError> {
    if self.n_rows >= num_cons || self.n_cols > num_io + num_vars {
      Err(NovaError::InvalidIndex)
    } else {
      Ok(())
    }
  }

  /// Pad matrix so that its columns and rows are powers of two
  pub(crate) fn pad(&mut self) {
    // Find the desired dimensions after padding
    let rows = self.n_rows();
    let cols = self.n_cols();

    // Since we padd with 0's and our matrix repr is sparse, we just need
    // to update the rows and cols attrs of the matrix.
    *self.n_rows_mut() = rows.next_power_of_two();
    *self.n_cols_mut() = cols.next_power_of_two();
  }

  // Gives the MLE of the given matrix.
  pub fn to_mle(&self) -> MultilinearPolynomial<G::Scalar> {
    // Matrices might need to get padded before turned into an MLE
    let mut padded_matrix = self.clone();
    padded_matrix.pad();

    sparse_vec_to_mle::<G>(
      padded_matrix.n_rows(),
      padded_matrix.n_cols(),
      padded_matrix.coeffs().to_vec(),
    )
  }
}

// NOTE: the method is called "sparse_vec_to_mle", but inputs are n_rows & n_cols, and a normal
// vector does not have rows&cols. This is because in this case the vec comes from matrix
// coefficients, maybe the method should be renamed, because is not to convert 'any' vector but a
// vector of matrix coefficients. A better option probably is to replace the two inputs n_rows &
// n_cols by directly the n_vars.
pub fn sparse_vec_to_mle<G: Group>(
  n_rows: usize,
  n_cols: usize,
  v: Vec<(usize, usize, G::Scalar)>,
) -> MultilinearPolynomial<G::Scalar> {
  let n_vars: usize = (log2(n_rows) + log2(n_cols)) as usize; // n_vars = s + s'
  let mut padded_vec = vec![G::Scalar::ZERO; 1 << n_vars];
  v.iter().copied().for_each(|(row, col, coeff)| {
    padded_vec[(n_cols * row) + col] = coeff;
  });

  dense_vec_to_mle(n_vars, &padded_vec)
}

pub fn dense_vec_to_mle<F: PrimeField>(n_vars: usize, v: &Vec<F>) -> MultilinearPolynomial<F> {
  // Pad to 2^n_vars
  let v_padded: Vec<F> = [
    v.clone(),
    std::iter::repeat(F::ZERO)
      .take((1 << n_vars) - v.len())
      .collect(),
  ]
  .concat();
  MultilinearPolynomial::new(v_padded)
}

pub fn vector_add<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Vec<F> {
  assert_eq!(a.len(), b.len(), "Vector addition with different lengths");
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] + b[i]);
  }

  res
}

pub fn vector_elem_product<F: PrimeField>(a: &Vec<F>, e: &F) -> Vec<F> {
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * e);
  }

  res
}

// XXX: This could be implemented via Mul trait in the lib. We should consider as it will reduce imports.
#[allow(dead_code)]
pub fn matrix_vector_product<F: PrimeField>(matrix: &Vec<Vec<F>>, vector: &Vec<F>) -> Vec<F> {
  assert_ne!(matrix.len(), 0, "empty-row matrix");
  assert_ne!(matrix[0].len(), 0, "empty-col  matrix");
  assert_eq!(
    matrix[0].len(),
    vector.len(),
    "matrix rows != vector length"
  );
  let mut res = Vec::with_capacity(matrix.len());
  for i in 0..matrix.len() {
    let mut sum = F::ZERO;
    for j in 0..matrix[i].len() {
      sum += matrix[i][j] * vector[j];
    }
    res.push(sum);
  }

  res
}

// Matrix vector product where matrix is sparse
// First element is row index, second column, third value stored
// XXX: This could be implemented via Mul trait in the lib. We should consider as it will reduce imports.
pub fn matrix_vector_product_sparse<G: Group>(
  matrix: &SparseMatrix<G>,
  vector: &Vec<G::Scalar>,
) -> Vec<G::Scalar> {
  let mut res = vec![G::Scalar::ZERO; matrix.n_rows()];
  for &(row, col, value) in matrix.coeffs.iter() {
    res[row] += value * vector[col];
  }
  res
}

pub fn hadamard_product<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Vec<F> {
  assert_eq!(a.len(), b.len(), "Hadamard needs same len vectors");
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * b[i]);
  }

  res
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

/// Sample a random list of multilinear polynomials.
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube.
pub fn random_mle_list<F: PrimeField, R: RngCore>(
  nv: usize,
  degree: usize,
  mut rng: &mut R,
) -> (Vec<Arc<MultilinearPolynomial<F>>>, F) {
  let mut multiplicands = Vec::with_capacity(degree);
  for _ in 0..degree {
    multiplicands.push(Vec::with_capacity(1 << nv))
  }
  let mut sum = F::ZERO;

  for _ in 0..(1 << nv) {
    let mut product = F::ONE;

    for e in multiplicands.iter_mut() {
      let val = F::random(&mut rng);
      e.push(val);
      product *= val;
    }
    sum += product;
  }

  let list = multiplicands
    .into_iter()
    .map(|x| Arc::new(MultilinearPolynomial::new(x)))
    .collect();

  (list, sum)
}

// Build a randomize list of mle-s whose sum is zero.
pub fn random_zero_mle_list<F: PrimeField, R: RngCore>(
  nv: usize,
  degree: usize,
  mut rng: &mut R,
) -> Vec<Arc<MultilinearPolynomial<F>>> {
  let mut multiplicands = Vec::with_capacity(degree);
  for _ in 0..degree {
    multiplicands.push(Vec::with_capacity(1 << nv))
  }
  for _ in 0..(1 << nv) {
    multiplicands[0].push(F::ZERO);
    for e in multiplicands.iter_mut().skip(1) {
      e.push(F::random(&mut rng));
    }
  }

  let list = multiplicands
    .into_iter()
    .map(|x| Arc::new(MultilinearPolynomial::new(x)))
    .collect();

  list
}

pub(crate) fn log2(x: usize) -> u32 {
  if x == 0 {
    0
  } else if x.is_power_of_two() {
    1usize.leading_zeros() - x.leading_zeros()
  } else {
    0usize.leading_zeros() - x.leading_zeros()
  }
}

#[cfg(test)]
mod tests {
  use crate::hypercube::BooleanHypercube;

  use super::*;
  use pasta_curves::{Ep, Fq};

  fn to_F_vec<F: PrimeField>(v: Vec<u64>) -> Vec<F> {
    v.iter().map(|x| F::from(*x)).collect()
  }

  fn to_F_matrix<F: PrimeField>(m: Vec<Vec<u64>>) -> Vec<Vec<F>> {
    m.iter().map(|x| to_F_vec(x.clone())).collect()
  }

  #[test]
  fn test_vector_add() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let b = to_F_vec::<Fq>(vec![4, 5, 6]);
    let res = vector_add(&a, &b);
    assert_eq!(res, to_F_vec::<Fq>(vec![5, 7, 9]));
  }

  #[test]
  fn test_vector_elem_product() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let e = Fq::from(2);
    let res = vector_elem_product(&a, &e);
    assert_eq!(res, to_F_vec::<Fq>(vec![2, 4, 6]));
  }

  #[test]
  fn test_matrix_vector_product() {
    let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let vector = vec![1, 2, 3];
    let A = to_F_matrix::<Fq>(matrix);
    let z = to_F_vec::<Fq>(vector);
    let res = matrix_vector_product(&A, &z);

    assert_eq!(res, to_F_vec::<Fq>(vec![14, 32]));
  }

  #[test]
  fn test_hadamard_product() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let b = to_F_vec::<Fq>(vec![4, 5, 6]);
    let res = hadamard_product(&a, &b);
    assert_eq!(res, to_F_vec::<Fq>(vec![4, 10, 18]));
  }

  #[test]
  fn test_matrix_vector_product_sparse() {
    let matrix = vec![
      (0, 0, Fq::from(1)),
      (0, 1, Fq::from(2)),
      (0, 2, Fq::from(3)),
      (1, 0, Fq::from(4)),
      (1, 1, Fq::from(5)),
      (1, 2, Fq::from(6)),
    ];

    let z = to_F_vec::<Fq>(vec![1, 2, 3]);
    let res =
      matrix_vector_product_sparse::<Ep>(&SparseMatrix::<Ep>::with_coeffs(2, 3, matrix), &z);

    assert_eq!(res, to_F_vec::<Fq>(vec![14, 32]));
  }

  #[test]
  fn test_sparse_matrix_n_cols_rows() {
    let matrix = vec![
      (0, 0, Fq::from(1u64)),
      (0, 1, Fq::from(2u64)),
      (0, 2, Fq::from(3u64)),
      (1, 0, Fq::from(4u64)),
      (1, 1, Fq::from(5u64)),
      (1, 2, Fq::from(6u64)),
      (4, 5, Fq::from(1u64)),
    ];
    let A = SparseMatrix::<Ep>::with_coeffs(5, 6, matrix.clone());
    assert_eq!(A.n_cols(), 6);
    assert_eq!(A.n_rows(), 5);

    // Since is sparse, the empty rows/cols at the end are not accounted unless we provide the info.
    let A = SparseMatrix::<Ep>::with_coeffs(10, 10, matrix);
    assert_eq!(A.n_cols(), 10);
    assert_eq!(A.n_rows(), 10);
  }

  #[test]
  fn test_matrix_to_mle() {
    let A = SparseMatrix::<Ep>::with_coeffs(
      5,
      5,
      vec![
        (0usize, 0usize, Fq::from(1u64)),
        (0, 1, Fq::from(2u64)),
        (0, 2, Fq::from(3u64)),
        (1, 0, Fq::from(4u64)),
        (1, 1, Fq::from(5u64)),
        (1, 2, Fq::from(6u64)),
        (3, 4, Fq::from(1u64)),
      ],
    );

    let A_mle = A.to_mle();
    assert_eq!(A_mle.len(), 64); // 5x5 matrix, thus 3bit x 3bit, thus 2^6=64 evals

    // hardcoded testvector to ensure that in the future the SparseMatrix.to_mle method holds
    let expected = vec![
      Fq::from(1u64),
      Fq::from(2u64),
      Fq::from(3u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(4u64),
      Fq::from(5u64),
      Fq::from(6u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(0u64),
      Fq::from(1u64),
      // the rest are zeroes
    ];
    assert_eq!(A_mle.Z[..29], expected);
    assert_eq!(A_mle.Z[29..], vec![Fq::ZERO; 64 - 29]);

    // check that the A_mle evaluated over the boolean hypercube equals the matrix A_i_j values
    let bhc = BooleanHypercube::<Fq>::new(A_mle.get_num_vars());
    let mut A_padded = A.clone();
    A_padded.pad();
    for term in A_padded.coeffs.iter() {
      let (i, j, coeff) = term;
      let s_i_j = bhc.evaluate_at(i * A_padded.n_cols + j);
      assert_eq!(&A_mle.evaluate(&s_i_j), coeff)
    }
  }
}
