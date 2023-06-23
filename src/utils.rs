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

  // XXX: Double check this
  pub(crate) fn pad(&mut self, n: usize) {
    let prev_n = self.n_cols;
    self.coeffs.par_iter_mut().for_each(|(_, c, _)| {
      *c = if *c >= prev_n { *c + n - prev_n } else { *c };
    });
  }
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
  let mut res = vec![G::Scalar::ZERO; 4];
  for &(row, col, value) in matrix.coeffs.iter() {
    res[row] += value * vector[col];
  }
  res
}

pub fn hadamard_product<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Vec<F> {
  assert_eq!(a.len(), b.len(), "Haddamard needs same len vectors");
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * b[i]);
  }

  res
}

pub fn sparse_matrix_to_mlp<G: Group>(
  matrix: &SparseMatrix<G>,
) -> MultilinearPolynomial<G::Scalar> {
  let n_rows = 4;
  let n_cols = 6usize;

  let n_vars: usize = n_cols.next_power_of_two().trailing_zeros() as usize;

  // Create a vector of zeros with size 2^n_vars
  let mut vec: Vec<G::Scalar> = vec![G::Scalar::ZERO; 2_usize.pow(n_vars as u32)];

  // Pad to 2^n_vars
  let vec_padded: Vec<G::Scalar> = [
    vec.clone(),
    std::iter::repeat(G::Scalar::ZERO)
      .take((1 << n_vars) - vec.len())
      .collect(),
  ]
  .concat();

  // Convert this vector into a MultilinearPolynomial
  MultilinearPolynomial::new(vec_padded)
}

// XXX: Create vec_to_mlp method and estract the padd

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

#[cfg(test)]
mod tests {
  use super::*;
  use pasta_curves::{Ep, Fq};

  fn to_F_vec<F: PrimeField>(v: Vec<u64>) -> Vec<F> {
    v.iter().map(|x| F::from(*x)).collect()
  }

  fn to_F_matrix<F: PrimeField>(m: Vec<Vec<u64>>) -> Vec<Vec<F>> {
    m.iter().map(|x| to_F_vec(x.clone())).collect()
  }

  #[test]
  fn test_n_cols_sparse_matrix() {
    let one = Fq::ONE;
    let A = vec![
      (0, 1, one),
      (1, 3, one),
      (2, 1, one),
      (2, 4, one),
      (3, 0, Fq::from(5u64)),
      (3, 5, one),
    ];

    assert_eq!(6, SparseMatrix::<Ep>::with_coeffs(4, 6, A).n_cols());
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

    assert_eq!(res, to_F_vec::<Fq>(vec![14, 32, 0]));
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

  // XXX this test is not really  testing much. Improve.
  #[test]
  fn test_sparse_matrix_to_mlp() {
    let matrix = vec![
      (0, 0, Fq::from(2)),
      (0, 1, Fq::from(3)),
      (0, 2, Fq::from(4)),
      (0, 3, Fq::from(4)),
      (1, 0, Fq::from(4)),
      (1, 1, Fq::from(11)),
      (1, 2, Fq::from(14)),
      (1, 3, Fq::from(14)),
      (2, 0, Fq::from(2)),
      (2, 1, Fq::from(8)),
      (2, 2, Fq::from(17)),
      (2, 3, Fq::from(17)),
      (3, 0, Fq::from(420)),
      (3, 1, Fq::from(4)),
      (3, 2, Fq::from(2)),
    ];
    let A = SparseMatrix::<Ep>::with_coeffs(4, 4, matrix);

    // Convert the sparse matrix to a multilinear polynomial
    let mlp = sparse_matrix_to_mlp(&A);

    // A 4x4 matrix, thus 2bit x 2bit, thus 2^4=16 evals
    assert_eq!(mlp.len(), 16);
  }
}
