//! # Sparse Matrices
//!
//! This module defines a custom implementation of CSR/CSC sparse matrices.
//! Specifically, we implement sparse matrix / dense vector multiplication
//! to compute the `A z`, `B z`, and `C z` in Nova.
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// CSR format sparse matrix, We follow the names used by scipy.
/// Detailed explanation here: https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseMatrix<F: PrimeField> {
  /// all non-zero values in the matrix
  pub data: Vec<F>,
  /// column indices
  pub indices: Vec<usize>,
  /// row information
  pub indptr: Vec<usize>,
  /// number of columns
  pub cols: usize,
}

impl<F: PrimeField> SparseMatrix<F> {
  /// 0x0 empty matrix
  pub fn empty() -> Self {
    SparseMatrix {
      data: vec![],
      indices: vec![],
      indptr: vec![0],
      cols: 0,
    }
  }

  /// Construct from the COO representation; Vec<usize(row), usize(col), F>.
  /// We assume that the rows are sorted during construction.
  pub fn new(matrix: &[(usize, usize, F)], rows: usize, cols: usize) -> Self {
    let mut new_matrix = vec![vec![]; rows];
    for (row, col, val) in matrix {
      new_matrix[*row].push((*col, *val));
    }

    for row in new_matrix.iter() {
      assert!(row.windows(2).all(|w| w[0].0 < w[1].0));
    }

    let mut indptr = vec![0; rows + 1];
    for (i, col) in new_matrix.iter().enumerate() {
      indptr[i + 1] = indptr[i] + col.len();
    }

    let mut indices = vec![];
    let mut data = vec![];
    for col in new_matrix {
      let (idx, val): (Vec<_>, Vec<_>) = col.into_iter().unzip();
      indices.extend(idx);
      data.extend(val);
    }

    SparseMatrix {
      data,
      indices,
      indptr,
      cols,
    }
  }

  /// Retrieves the data for row slice [i..j] from `ptrs`.
  /// We assume that `ptrs` is indexed from `indptrs` and do not check if the
  /// returned slice is actually a valid row.
  pub fn get_row_unchecked(&self, ptrs: &[usize; 2]) -> impl Iterator<Item = (&F, &usize)> {
    self.data[ptrs[0]..ptrs[1]]
      .iter()
      .zip(&self.indices[ptrs[0]..ptrs[1]])
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  pub fn multiply_vec(&self, vector: &[F]) -> Vec<F> {
    assert_eq!(self.cols, vector.len(), "invalid shape");

    self.multiply_vec_unchecked(vector)
  }

  /// Multiply by a dense vector; uses rayon/gpu.
  /// This does not check that the shape of the matrix/vector are compatible.
  pub fn multiply_vec_unchecked(&self, vector: &[F]) -> Vec<F> {
    self
      .indptr
      .par_windows(2)
      .map(|ptrs| {
        self
          .get_row_unchecked(ptrs.try_into().unwrap())
          .map(|(val, col_idx)| *val * vector[*col_idx])
          .sum()
      })
      .collect()
  }

  /// number of non-zero entries
  pub fn len(&self) -> usize {
    *self.indptr.last().unwrap()
  }

  /// empty matrix
  pub fn is_empty(&self) -> bool {
    self.len() == 0
  }

  /// returns a custom iterator
  pub fn iter(&self) -> Iter<'_, F> {
    let mut row = 0;
    while self.indptr[row + 1] == 0 {
      row += 1;
    }
    Iter {
      matrix: self,
      row,
      i: 0,
      nnz: *self.indptr.last().unwrap(),
    }
  }
}

/// Iterator for sparse matrix
pub struct Iter<'a, F: PrimeField> {
  matrix: &'a SparseMatrix<F>,
  row: usize,
  i: usize,
  nnz: usize,
}

impl<'a, F: PrimeField> Iterator for Iter<'a, F> {
  type Item = (usize, usize, F);

  fn next(&mut self) -> Option<Self::Item> {
    // are we at the end?
    if self.i == self.nnz {
      return None;
    }

    // compute current item
    let curr_item = (
      self.row,
      self.matrix.indices[self.i],
      self.matrix.data[self.i],
    );

    // advance the iterator
    self.i += 1;
    // edge case at the end
    if self.i == self.nnz {
      return Some(curr_item);
    }
    // if `i` has moved to next row
    while self.i >= self.matrix.indptr[self.row + 1] {
      self.row += 1;
    }

    Some(curr_item)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::PallasEngine,
    traits::{Engine, Group},
  };
  use ff::PrimeField;
  use proptest::{
    prelude::*,
    strategy::{BoxedStrategy, Just, Strategy},
  };

  type G = <PallasEngine as Engine>::GE;
  type Fr = <G as Group>::Scalar;

  /// Wrapper struct around a field element that implements additional traits
  #[derive(Clone, Debug, PartialEq, Eq)]
  pub struct FWrap<F: PrimeField>(pub F);

  impl<F: PrimeField> Copy for FWrap<F> {}

  #[cfg(not(target_arch = "wasm32"))]
  /// Trait implementation for generating `FWrap<F>` instances with proptest
  impl<F: PrimeField> Arbitrary for FWrap<F> {
    type Parameters = ();
    type Strategy = BoxedStrategy<Self>;

    fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
      use rand::rngs::StdRng;
      use rand_core::SeedableRng;

      let strategy = any::<[u8; 32]>()
        .prop_map(|seed| FWrap(F::random(StdRng::from_seed(seed))))
        .no_shrink();
      strategy.boxed()
    }
  }

  #[test]
  fn test_matrix_creation() {
    let matrix_data = vec![
      (0, 1, Fr::from(2)),
      (1, 2, Fr::from(3)),
      (2, 0, Fr::from(4)),
    ];
    let sparse_matrix = SparseMatrix::<Fr>::new(&matrix_data, 3, 3);

    assert_eq!(
      sparse_matrix.data,
      vec![Fr::from(2), Fr::from(3), Fr::from(4)]
    );
    assert_eq!(sparse_matrix.indices, vec![1, 2, 0]);
    assert_eq!(sparse_matrix.indptr, vec![0, 1, 2, 3]);
  }

  #[test]
  fn test_matrix_vector_multiplication() {
    let matrix_data = vec![
      (0, 1, Fr::from(2)),
      (0, 2, Fr::from(7)),
      (1, 2, Fr::from(3)),
      (2, 0, Fr::from(4)),
    ];
    let sparse_matrix = SparseMatrix::<Fr>::new(&matrix_data, 3, 3);
    let vector = vec![Fr::from(1), Fr::from(2), Fr::from(3)];

    let result = sparse_matrix.multiply_vec(&vector);

    assert_eq!(result, vec![Fr::from(25), Fr::from(9), Fr::from(4)]);
  }

  fn coo_strategy() -> BoxedStrategy<Vec<(usize, usize, FWrap<Fr>)>> {
    let coo_strategy = any::<FWrap<Fr>>().prop_flat_map(|f| (0usize..100, 0usize..100, Just(f)));
    proptest::collection::vec(coo_strategy, 10).boxed()
  }

  proptest! {
      #[test]
      fn test_matrix_iter(mut coo_matrix in coo_strategy()) {
        // process the randomly generated coo matrix
        coo_matrix.sort_by_key(|(row, col, _val)| (*row, *col));
        coo_matrix.dedup_by_key(|(row, col, _val)| (*row, *col));
        let coo_matrix = coo_matrix.into_iter().map(|(row, col, val)| { (row, col, val.0) }).collect::<Vec<_>>();

        let matrix = SparseMatrix::new(&coo_matrix, 100, 100);

        prop_assert_eq!(coo_matrix, matrix.iter().collect::<Vec<_>>());
    }
  }
}
