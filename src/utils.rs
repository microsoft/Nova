//! Basic utils
use crate::errors::NovaError;
use crate::spartan::polynomial::MultilinearPolynomial;
use crate::traits::Group;
use ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};

/// A matrix structure represented on a sparse form.
/// First element is row index, second column, third value stored
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparseMatrix<G: Group>(pub(crate) Vec<(usize, usize, G::Scalar)>);

impl<G: Group> SparseMatrix<G> {
  pub fn new() -> Self {
    Self(vec![])
  }

  pub fn with_capacity(n: usize) -> Self {
    Self(Vec::with_capacity(n))
  }

  // Find the maximum row index in the matrix
  pub fn n_rows(&self) -> usize {
    self.0.iter().map(|r| r.0).max().unwrap() + 1
  }

  // Find the maximum column index in the matrix
  pub fn n_cols(&self) -> usize {
    self.0.iter().map(|r| r.1).max().unwrap() + 1
  }
}

impl<G: Group> From<Vec<(usize, usize, G::Scalar)>> for SparseMatrix<G> {
  fn from(matrix: Vec<(usize, usize, G::Scalar)>) -> SparseMatrix<G> {
    SparseMatrix(matrix)
  }
}

impl<G: Group> From<&Vec<(usize, usize, G::Scalar)>> for SparseMatrix<G> {
  fn from(matrix: &Vec<(usize, usize, G::Scalar)>) -> SparseMatrix<G> {
    SparseMatrix(matrix.clone())
  }
}

pub fn vector_add<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Result<Vec<F>, NovaError> {
  if a.len() != b.len() {
    return Err(NovaError::InvalidIndex);
  }

  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] + b[i]);
  }

  Ok(res)
}

pub fn vector_elem_product<F: PrimeField>(a: &Vec<F>, e: &F) -> Result<Vec<F>, NovaError> {
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * e);
  }

  Ok(res)
}

#[allow(dead_code)]
pub fn matrix_vector_product<F: PrimeField>(
  matrix: &Vec<Vec<F>>,
  vector: &Vec<F>,
) -> Result<Vec<F>, NovaError> {
  if matrix.len() == 0 || matrix[0].len() == 0 {
    return Err(NovaError::InvalidIndex);
  }

  if matrix[0].len() != vector.len() {
    return Err(NovaError::InvalidIndex);
  }

  let mut res = Vec::with_capacity(matrix.len());
  for i in 0..matrix.len() {
    let mut sum = F::ZERO;
    for j in 0..matrix[i].len() {
      sum += matrix[i][j] * vector[j];
    }
    res.push(sum);
  }

  Ok(res)
}

// Matrix vector product where matrix is sparse
// First element is row index, second column, third value stored
pub fn matrix_vector_product_sparse<G: Group>(
  matrix: &SparseMatrix<G>,
  vector: &Vec<G::Scalar>,
) -> Result<Vec<G::Scalar>, NovaError> {
  if matrix.0.len() == 0 {
    return Err(NovaError::InvalidIndex);
  }

  // Find the maximum row index in the matrix
  let max_row = matrix.0.iter().map(|r| r.0).max().unwrap() + 1;
  if max_row > vector.len() {
    return Err(NovaError::InvalidIndex);
  }

  let mut res = vec![G::Scalar::ZERO; max_row];
  for &(row, col, value) in matrix.0.iter() {
    res[row] += value * vector[col];
  }

  Ok(res)
}

pub fn hadamard_product<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Result<Vec<F>, NovaError> {
  if a.len() != b.len() {
    return Err(NovaError::InvalidIndex);
  }

  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * b[i]);
  }

  Ok(res)
}

#[allow(dead_code)]
pub fn to_F_vec<F: PrimeField>(v: Vec<u64>) -> Vec<F> {
  v.iter().map(|x| F::from(*x)).collect()
}

#[allow(dead_code)]
pub fn to_F_matrix<F: PrimeField>(m: Vec<Vec<u64>>) -> Vec<Vec<F>> {
  m.iter().map(|x| to_F_vec(x.clone())).collect()
}

#[allow(dead_code)]
pub fn to_F_matrix_sparse<F: PrimeField>(m: Vec<(usize, usize, u64)>) -> Vec<(usize, usize, F)> {
  m.iter().map(|x| (x.0, x.1, F::from(x.2))).collect()
}

pub fn sparse_matrix_to_mlp<G: Group>(
  matrix: &SparseMatrix<G>,
) -> MultilinearPolynomial<G::Scalar> {
  let n_rows = matrix.n_rows();
  let n_cols = matrix.n_cols();

  // Since n_rows and n_cols already account for 0 indexing,
  // The total number of elements would be n_rows * n_cols
  let total_elements: usize = n_rows * n_cols;
  let n_vars: usize = total_elements.next_power_of_two().trailing_zeros() as usize;

  // Create a vector of zeros with size 2^n_vars
  let mut vec: Vec<G::Scalar> = vec![G::Scalar::ZERO; 2_usize.pow(n_vars as u32)];

  // Assign non-zero entries from the sparse matrix to the vector
  for &(i, j, val) in matrix.0.iter() {
    let index = i * n_cols + j; // Convert (i, j) into an index for a flat vector
    vec[index] = val;
  }

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

// FIXME: Dummy function
pub fn compute_sum_Mz<G: Group, Z>(
  _M_j: MultilinearPolynomial<G::Scalar>,
  _z_mle: &Z,
) -> Result<MultilinearPolynomial<<G as Group>::Scalar>, &'static str> {
  Ok(MultilinearPolynomial::new(vec![]))
}

#[cfg(test)]
mod tests {
  use super::*;
  use pasta_curves::{Ep, Fq};

  #[test]
  fn test_vector_add() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let b = to_F_vec::<Fq>(vec![4, 5, 6]);
    let res = vector_add(&a, &b).unwrap();
    assert_eq!(res, to_F_vec::<Fq>(vec![5, 7, 9]));
  }

  #[test]
  fn test_vector_elem_product() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let e = Fq::from(2);
    let res = vector_elem_product(&a, &e).unwrap();
    assert_eq!(res, to_F_vec::<Fq>(vec![2, 4, 6]));
  }

  #[test]
  fn test_matrix_vector_product() {
    let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let vector = vec![1, 2, 3];
    let A = to_F_matrix::<Fq>(matrix);
    let z = to_F_vec::<Fq>(vector);
    let res = matrix_vector_product(&A, &z).unwrap();

    assert_eq!(res, to_F_vec::<Fq>(vec![14, 32]));
  }

  #[test]
  fn test_hadamard_product() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let b = to_F_vec::<Fq>(vec![4, 5, 6]);
    let res = hadamard_product(&a, &b).unwrap();
    assert_eq!(res, to_F_vec::<Fq>(vec![4, 10, 18]));
  }

  #[test]
  fn test_matrix_vector_product_sparse() {
    let matrix = vec![
      (0, 0, 1),
      (0, 1, 2),
      (0, 2, 3),
      (1, 0, 4),
      (1, 1, 5),
      (1, 2, 6),
    ];
    let vector = vec![1, 2, 3];
    let A = to_F_matrix_sparse::<Fq>(matrix);
    let z = to_F_vec::<Fq>(vector);
    let res = matrix_vector_product_sparse::<Ep>(&(A.into()), &z).unwrap();

    assert_eq!(res, to_F_vec::<Fq>(vec![14, 32]));
  }

  #[test]
  fn test_sparse_matrix_n_rows() {
    let matrix = vec![
      (0, 0, 1),
      (0, 1, 2),
      (0, 2, 3),
      (1, 0, 4),
      (1, 1, 5),
      (1, 2, 6),
    ];
    let A: SparseMatrix<Ep> = to_F_matrix_sparse::<Fq>(matrix).into();
    assert_eq!(A.n_rows(), 2);
  }

  #[test]
  fn test_sparse_matrix_n_cols() {
    let matrix = vec![
      (0, 0, 1),
      (0, 1, 2),
      (0, 2, 3),
      (1, 0, 4),
      (1, 1, 5),
      (1, 2, 6),
    ];
    let A: SparseMatrix<Ep> = to_F_matrix_sparse::<Fq>(matrix).into();
    assert_eq!(A.n_cols(), 3);
  }

  #[test]
  fn test_sparse_matrix_to_mlp() {
    let matrix = vec![
      (0, 0, 2),
      (0, 1, 3),
      (0, 2, 4),
      (0, 3, 4),
      (1, 0, 4),
      (1, 1, 11),
      (1, 2, 14),
      (1, 3, 14),
      (2, 0, 2),
      (2, 1, 8),
      (2, 2, 17),
      (2, 3, 17),
      (3, 0, 420),
      (3, 1, 4),
      (3, 2, 2),
    ];
    let A: SparseMatrix<Ep> = to_F_matrix_sparse::<Fq>(matrix).into();

    // Convert the sparse matrix to a multilinear polynomial
    let mlp = sparse_matrix_to_mlp(&A);

    // A 4x4 matrix, thus 2bit x 2bit, thus 2^4=16 evals
    assert_eq!(mlp.len(), 16);
  }
}
