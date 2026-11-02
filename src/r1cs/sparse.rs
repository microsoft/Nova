//! # Sparse Matrices
//!
//! This module defines a custom implementation of CSR/CSC sparse matrices.
//! Specifically, we implement sparse matrix / dense vector multiplication
//! to compute the `A z`, `B z`, and `C z` in Nova.
use crate::constants::PARALLEL_THRESHOLD;
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Precomputed SpMV accelerator for a fixed sparse matrix.
///
/// Classifies entries by coefficient magnitude to avoid expensive field
/// multiplications for the common cases in R1CS:
/// - +/-1: just add/subtract (no multiplication)
/// - small +/-k (|k| <= 7): repeated addition (cheaper than field mul)
/// - general: full field multiplication
#[derive(Clone, Debug)]
pub(super) struct PrecomputedSparseMatrix<F: PrimeField> {
  /// One compact class per entry in the source matrix's canonical CSR order.
  /// Values in `-7..=-1` and `1..=7` encode small signed coefficients;
  /// zero means that the source coefficient requires a general field
  /// multiplication. The source remains the sole owner of rows, columns, and
  /// coefficient values.
  coefficient_classes: Vec<i8>,
  _p: PhantomData<F>,
}

impl<F: PrimeField> PrecomputedSparseMatrix<F> {
  /// Build from a CSR SparseMatrix by classifying entries.
  pub(super) fn from_sparse(m: &SparseMatrix<F>) -> Self {
    let one = F::ONE;
    let neg_one = -F::ONE;

    // Precompute small positive/negative field elements for comparison
    let small_pos: Vec<F> = (2u64..=7).map(F::from).collect();
    let small_neg: Vec<F> = (2u64..=7).map(|k| -F::from(k)).collect();

    let coefficient_classes = m
      .data
      .iter()
      .map(|&val| {
        if val == one {
          1
        } else if val == neg_one {
          -1
        } else if let Some(k) = small_pos.iter().position(|&v| v == val) {
          (k as i8) + 2
        } else if let Some(k) = small_neg.iter().position(|&v| v == val) {
          -((k as i8) + 2)
        } else {
          0
        }
      })
      .collect();

    Self {
      coefficient_classes,
      _p: PhantomData,
    }
  }

  #[inline(always)]
  fn small_mul(coeff: i8, x: F) -> F {
    // For |coeff| in 2..=7, use doubling + addition instead of field mul
    let abs = coeff.unsigned_abs();
    let result = match abs {
      2 => x.double(),
      3 => x.double() + x,
      4 => x.double().double(),
      5 => x.double().double() + x,
      6 => {
        let d = x.double();
        d.double() + d
      }
      7 => {
        let d = x.double();
        d.double() + d + x
      }
      _ => unreachable!(),
    };
    if coeff < 0 {
      -result
    } else {
      result
    }
  }

  #[inline(always)]
  fn product(class: i8, coefficient: F, value: F) -> F {
    match class {
      1 => value,
      -1 => -value,
      -7..=-2 | 2..=7 => Self::small_mul(class, value),
      0 => coefficient * value,
      _ => unreachable!("invalid compact coefficient class"),
    }
  }

  #[inline(always)]
  pub(super) fn compute_row_single(&self, matrix: &SparseMatrix<F>, row: usize, v: &[F]) -> F {
    let mut sum = F::ZERO;
    for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
      sum += Self::product(
        self.coefficient_classes[entry],
        matrix.data[entry],
        v[matrix.indices[entry]],
      );
    }
    sum
  }

  #[inline(always)]
  fn compute_row_pair(&self, matrix: &SparseMatrix<F>, row: usize, v1: &[F], v2: &[F]) -> (F, F) {
    let mut s1 = F::ZERO;
    let mut s2 = F::ZERO;
    for entry in matrix.indptr[row]..matrix.indptr[row + 1] {
      let class = self.coefficient_classes[entry];
      let coefficient = matrix.data[entry];
      let column = matrix.indices[entry];
      s1 += Self::product(class, coefficient, v1[column]);
      s2 += Self::product(class, coefficient, v2[column]);
    }
    (s1, s2)
  }

  /// Fast SpMV using precomputed coefficient classification.
  ///
  /// `matrix` must be the owning `R1CSShape` field from which this private
  /// cache was built. Keeping this API within the parent R1CS module makes
  /// that association explicit at each of its few call sites.
  pub(super) fn multiply_vec(&self, matrix: &SparseMatrix<F>, vector: &[F]) -> Vec<F> {
    self.validate_source(matrix);
    assert_eq!(matrix.cols, vector.len(), "invalid shape");
    let rows = matrix.indptr.len() - 1;
    if rows <= PARALLEL_THRESHOLD {
      (0..rows)
        .map(|r| self.compute_row_single(matrix, r, vector))
        .collect()
    } else {
      (0..rows)
        .into_par_iter()
        .map(|r| self.compute_row_single(matrix, r, vector))
        .collect()
    }
  }

  /// Fast dual-vector SpMV: compute (M*v1, M*v2) in a single pass.
  ///
  /// `matrix` must be the owning `R1CSShape` field from which this private
  /// cache was built.
  pub(super) fn multiply_vec_pair(
    &self,
    matrix: &SparseMatrix<F>,
    v1: &[F],
    v2: &[F],
  ) -> (Vec<F>, Vec<F>) {
    self.validate_source(matrix);
    assert_eq!(matrix.cols, v1.len(), "invalid shape for v1");
    assert_eq!(matrix.cols, v2.len(), "invalid shape for v2");
    let rows = matrix.indptr.len() - 1;
    if rows <= PARALLEL_THRESHOLD {
      (0..rows)
        .map(|r| self.compute_row_pair(matrix, r, v1, v2))
        .unzip()
    } else {
      (0..rows)
        .into_par_iter()
        .map(|r| self.compute_row_pair(matrix, r, v1, v2))
        .unzip()
    }
  }

  fn validate_source(&self, matrix: &SparseMatrix<F>) {
    assert_eq!(
      self.coefficient_classes.len(),
      matrix.data.len(),
      "stale compact coefficient classification"
    );
  }
}

/// CSR format sparse matrix, We follow the names used by scipy.
/// Detailed explanation here: <https://stackoverflow.com/questions/52299420/scipy-csr-matrix-understand-indptr>
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
    let nnz = *self.indptr.last().unwrap();
    if nnz == 0 {
      return Iter {
        matrix: self,
        row: 0,
        i: 0,
        nnz,
      };
    }

    let mut row = 0;
    while row + 1 < self.indptr.len() && self.indptr[row + 1] == 0 {
      row += 1;
    }
    Iter {
      matrix: self,
      row,
      i: 0,
      nnz,
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

impl<F: PrimeField> Iterator for Iter<'_, F> {
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
  use ff::{Field, PrimeField};
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

  /// Build a sparse matrix with entries spanning all coefficient categories:
  /// +1, -1, small (2..=7), small negative (-2..=-7), and general.
  fn mixed_coefficient_matrix() -> (SparseMatrix<Fr>, Vec<Fr>) {
    let one = Fr::ONE;
    let neg_one = -Fr::ONE;
    let entries = vec![
      // Row 0: +1, -1, general
      (0, 0, one),
      (0, 1, neg_one),
      (0, 2, Fr::from(42)),
      // Row 1: small positive 2..=7
      (1, 0, Fr::from(2)),
      (1, 1, Fr::from(3)),
      (1, 2, Fr::from(4)),
      (1, 3, Fr::from(5)),
      (1, 4, Fr::from(6)),
      (1, 5, Fr::from(7)),
      // Row 2: small negative -2..=-7
      (2, 0, -Fr::from(2)),
      (2, 1, -Fr::from(3)),
      (2, 2, -Fr::from(4)),
      (2, 3, -Fr::from(5)),
      (2, 4, -Fr::from(6)),
      (2, 5, -Fr::from(7)),
      // Row 3: empty row (no entries)
      // Row 4: mixed
      (4, 0, one),
      (4, 1, Fr::from(3)),
      (4, 2, -Fr::from(5)),
      (4, 3, neg_one),
      (4, 4, Fr::from(100)),
    ];
    let cols = 6;
    let rows = 5;
    let matrix = SparseMatrix::new(&entries, rows, cols);
    let vector: Vec<Fr> = (1..=cols as u64).map(Fr::from).collect();
    (matrix, vector)
  }

  #[test]
  fn test_precomputed_multiply_vec() {
    let (matrix, vector) = mixed_coefficient_matrix();
    let expected = matrix.multiply_vec(&vector);
    let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix);
    let result = precomputed.multiply_vec(&matrix, &vector);
    assert_eq!(expected, result);
    assert_eq!(precomputed.coefficient_classes.len(), matrix.data.len());
  }

  #[test]
  fn test_precomputed_multiply_vec_pair() {
    let (matrix, v1) = mixed_coefficient_matrix();
    let v2: Vec<Fr> = v1.iter().map(|x| *x + Fr::from(10)).collect();

    let expected1 = matrix.multiply_vec(&v1);
    let expected2 = matrix.multiply_vec(&v2);

    let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix);
    let (result1, result2) = precomputed.multiply_vec_pair(&matrix, &v1, &v2);

    assert_eq!(expected1, result1);
    assert_eq!(expected2, result2);
  }

  #[test]
  fn test_precomputed_handles_leading_and_trailing_empty_rows() {
    let entries = vec![(2, 0, Fr::ONE), (4, 1, -Fr::from(7))];
    let matrix = SparseMatrix::new(&entries, 7, 2);
    let vector = vec![Fr::from(9), Fr::from(3)];
    let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix);

    assert_eq!(
      precomputed.multiply_vec(&matrix, &vector),
      matrix.multiply_vec(&vector)
    );
  }

  #[test]
  #[should_panic(expected = "stale compact coefficient classification")]
  fn test_precomputed_rejects_stale_source_length() {
    let original = SparseMatrix::new(&[(0, 0, Fr::ONE)], 1, 2);
    let stale = SparseMatrix::new(&[(0, 0, Fr::ONE), (0, 1, Fr::ONE)], 1, 2);
    let precomputed = PrecomputedSparseMatrix::from_sparse(&original);

    let _ = precomputed.multiply_vec(&stale, &[Fr::ONE, Fr::ONE]);
  }

  #[test]
  fn test_precomputed_parallel_row_path_matches_sparse() {
    let rows = PARALLEL_THRESHOLD + 1;
    let entries = vec![
      (0, 0, Fr::ONE),
      (PARALLEL_THRESHOLD / 2, 1, -Fr::from(4)),
      (PARALLEL_THRESHOLD, 2, Fr::from(29)),
    ];
    let matrix = SparseMatrix::new(&entries, rows, 3);
    let v1 = vec![Fr::from(2), Fr::from(3), Fr::from(5)];
    let v2 = vec![Fr::from(7), Fr::from(11), Fr::from(13)];
    let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix);

    assert_eq!(
      precomputed.multiply_vec(&matrix, &v1),
      matrix.multiply_vec(&v1)
    );
    assert_eq!(
      precomputed.multiply_vec_pair(&matrix, &v1, &v2),
      (matrix.multiply_vec(&v1), matrix.multiply_vec(&v2))
    );
  }

  proptest! {
    #[test]
    fn test_precomputed_matches_sparse(mut coo_matrix in coo_strategy()) {
      coo_matrix.sort_by_key(|(row, col, _val)| (*row, *col));
      coo_matrix.dedup_by_key(|(row, col, _val)| (*row, *col));
      let coo_matrix = coo_matrix.into_iter().map(|(row, col, val)| (row, col, val.0)).collect::<Vec<_>>();

      let matrix = SparseMatrix::new(&coo_matrix, 100, 100);
      let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix);

      let v1: Vec<Fr> = (0..100).map(|i| Fr::from(i + 1)).collect();
      let v2: Vec<Fr> = (0..100).map(|i| Fr::from(i * 3 + 7)).collect();

      // Single-vector: precomputed must match SparseMatrix
      let expected = matrix.multiply_vec(&v1);
      let result = precomputed.multiply_vec(&matrix, &v1);
      prop_assert_eq!(&expected, &result);

      // Dual-vector: both outputs must match
      let expected1 = matrix.multiply_vec(&v1);
      let expected2 = matrix.multiply_vec(&v2);
      let (result1, result2) = precomputed.multiply_vec_pair(&matrix, &v1, &v2);
      prop_assert_eq!(&expected1, &result1);
      prop_assert_eq!(&expected2, &result2);
    }
  }
}
