// Allow `&Matrix` in function signatures.
#![allow(clippy::ptr_arg)]

use ff::PrimeField;
use serde::{Deserialize, Serialize};

use super::{
  matrix,
  matrix::{
    invert, is_identity, is_invertible, is_square, left_apply_matrix, mat_mul, minor, transpose,
    Matrix,
  },
};
use crate::errors::NovaError;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MdsMatrices<F: PrimeField> {
  pub m: Matrix<F>,
  pub m_inv: Matrix<F>,
  pub m_hat: Matrix<F>,
  pub m_hat_inv: Matrix<F>,
  pub m_prime: Matrix<F>,
  pub m_double_prime: Matrix<F>,
}

pub(crate) fn derive_mds_matrices<F: PrimeField>(m: Matrix<F>) -> Result<MdsMatrices<F>, NovaError> {
  let m_inv = invert(&m).ok_or_else(|| NovaError::InvalidMatrix {
    reason: "MDS matrix is not invertible".to_string(),
  })?;
  let m_hat = minor(&m, 0, 0)?;
  let m_hat_inv = invert(&m_hat).ok_or_else(|| NovaError::InvalidMatrix {
    reason: "MDS minor matrix is not invertible".to_string(),
  })?;
  let m_prime = make_prime(&m)?;
  let m_double_prime = make_double_prime(&m, &m_hat_inv)?;

  Ok(MdsMatrices {
    m,
    m_inv,
    m_hat,
    m_hat_inv,
    m_prime,
    m_double_prime,
  })
}

/// A `SparseMatrix` is specifically one of the form of M''.
/// This means its first row and column are each dense, and the interior matrix
/// (minor to the element in both the row and column) is the identity.
/// We will pluralize this compact structure `sparse_matrixes` to distinguish from `sparse_matrices` from which they are created.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<F: PrimeField> {
  /// `w_hat` is the first column of the M'' matrix. It will be directly multiplied (scalar product) with a row of state elements.
  pub w_hat: Vec<F>,
  /// `v_rest` contains all but the first (already included in `w_hat`).
  pub v_rest: Vec<F>,
}

impl<F: PrimeField> SparseMatrix<F> {
  pub fn new_from_ref(m_double_prime: &Matrix<F>) -> Result<Self, NovaError> {
    if !Self::is_sparse_matrix(m_double_prime) {
      return Err(NovaError::InvalidMatrix {
        reason: "Matrix is not a valid sparse matrix".to_string(),
      });
    }
    
    let size = matrix::rows(m_double_prime);

    let w_hat = (0..size).map(|i| m_double_prime[i][0]).collect::<Vec<_>>();
    let v_rest = m_double_prime[0][1..].to_vec();

    Ok(Self { w_hat, v_rest })
  }

  pub fn is_sparse_matrix(m: &Matrix<F>) -> bool {
    match minor(m, 0, 0) {
      Ok(minor_matrix) => is_square(m) && is_identity(&minor_matrix),
      Err(_) => false,
    }
  }
}

// - Having effectively moved the round-key additions into the S-boxes, refactor MDS matrices used for partial-round mix layer to use sparse matrices.
// - This requires using a different (sparse) matrix at each partial round, rather than the same dense matrix at each.
//   - The MDS matrix, M, for each such round, starting from the last, is factored into two components, such that M' x M'' = M.
//   - M'' is sparse and replaces M for the round.
//   - The previous layer's M is then replaced by M x M' = M*.
//   - M* is likewise factored into M*' and M*'', and the process continues.
pub(crate) fn factor_to_sparse_matrixes<F: PrimeField>(
  base_matrix: &Matrix<F>,
  n: usize,
) -> Result<(Matrix<F>, Vec<SparseMatrix<F>>), NovaError> {
  let (pre_sparse, sparse_matrices) = factor_to_sparse_matrices(base_matrix, n)?;
  let sparse_matrixes = sparse_matrices
    .iter()
    .map(|m| SparseMatrix::<F>::new_from_ref(m))
    .collect::<Result<Vec<_>, _>>()?;

  Ok((pre_sparse, sparse_matrixes))
}

pub(crate) fn factor_to_sparse_matrices<F: PrimeField>(
  base_matrix: &Matrix<F>,
  n: usize,
) -> Result<(Matrix<F>, Vec<Matrix<F>>), NovaError> {
  let (pre_sparse, mut all) =
    (0..n).try_fold((base_matrix.clone(), Vec::new()), |(curr, mut acc), _| -> Result<_, NovaError> {
      let derived = derive_mds_matrices(curr)?;
      acc.push(derived.m_double_prime);
      let new = mat_mul(base_matrix, &derived.m_prime).ok_or_else(|| NovaError::InvalidMatrix {
        reason: "Matrix multiplication failed in sparse matrix factorization".to_string(),
      })?;
      Ok((new, acc))
    })?;
  all.reverse();
  Ok((pre_sparse, all))
}

pub(crate) fn generate_mds<F: PrimeField>(t: usize) -> Matrix<F> {
  // Source: https://github.com/dusk-network/dusk-poseidon-merkle/commit/776c37734ea2e71bb608ce4bc58fdb5f208112a7#diff-2eee9b20fb23edcc0bf84b14167cbfdc
  // Generate x and y values deterministically for the cauchy matrix
  // where x[i] != y[i] to allow the values to be inverted
  // and there are no duplicates in the x vector or y vector, so that the determinant is always non-zero
  // [a b]
  // [c d]
  // det(M) = (ad - bc) ; if a == b and c == d => det(M) =0
  // For an MDS matrix, every possible mxm submatrix, must have det(M) != 0
  let xs: Vec<F> = (0..t as u64).map(F::from).collect();
  let ys: Vec<F> = (t as u64..2 * t as u64).map(F::from).collect();

  let matrix = xs
    .iter()
    .map(|xs_item| {
      ys.iter()
        .map(|ys_item| {
          // Generate the entry at (i,j)
          let mut tmp = *xs_item;
          tmp.add_assign(ys_item);
          tmp.invert().unwrap()
        })
        .collect()
    })
    .collect();

  // To ensure correctness, we would check all sub-matrices for invertibility. Meanwhile, this is a simple sanity check.
  assert!(is_invertible(&matrix));

  //  `poseidon::product_mds_with_matrix` relies on the constructed MDS matrix being symmetric, so ensure it is.
  assert_eq!(matrix, transpose(&matrix));
  matrix
}

fn make_prime<F: PrimeField>(m: &Matrix<F>) -> Result<Matrix<F>, NovaError> {
  let result = m.iter()
    .enumerate()
    .map(|(i, row)| match i {
      0 => {
        let mut new_row = vec![F::ZERO; row.len()];
        new_row[0] = F::ONE;
        new_row
      }
      _ => {
        let mut new_row = vec![F::ZERO; row.len()];
        new_row[1..].copy_from_slice(&row[1..]);
        new_row
      }
    })
    .collect();
  Ok(result)
}

fn make_double_prime<F: PrimeField>(m: &Matrix<F>, m_hat_inv: &Matrix<F>) -> Result<Matrix<F>, NovaError> {
  let (v, w) = make_v_w(m);
  let w_hat = left_apply_matrix(m_hat_inv, &w)?;

  let result = m.iter()
    .enumerate()
    .map(|(i, row)| match i {
      0 => {
        let mut new_row = Vec::with_capacity(row.len());
        new_row.push(row[0]);
        new_row.extend(&v);
        new_row
      }
      _ => {
        let mut new_row = vec![F::ZERO; row.len()];
        new_row[0] = w_hat[i - 1];
        new_row[i] = F::ONE;
        new_row
      }
    })
    .collect();
  Ok(result)
}

fn make_v_w<F: PrimeField>(m: &Matrix<F>) -> (Vec<F>, Vec<F>) {
  let v = m[0][1..].to_vec();
  let w = m.iter().skip(1).map(|column| column[0]).collect();
  (v, w)
}
