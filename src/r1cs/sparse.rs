//! # Sparse Matrices
//!
//! This module defines a custom implementation of CSR/CSC sparse matrices.
//! Specifically, we implement sparse matrix / dense vector multiplication
//! to compute the `A z`, `B z`, and `C z` in Nova.
use ff::PrimeField;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Precomputed SpMV accelerator for a fixed sparse matrix.
///
/// Classifies entries by coefficient magnitude to avoid expensive field
/// multiplications for the common cases in R1CS:
/// - ±1: just add/subtract (no multiplication)
/// - small ±k (|k| ≤ 7): repeated addition (cheaper than field mul)
/// - general: full field multiplication
///
/// For a typical R1CS circuit, 60-80% of entries are ±1, and another
/// 10-20% are small integers, leaving only ~10-20% needing full mul.
#[derive(Clone, Debug)]
pub struct PrecomputedSparseMatrix<F: PrimeField> {
  num_rows: usize,
  num_cols: usize,
  /// Per-row spans into each category's column array: (start, count)
  row_unit_pos: Vec<(u32, u16)>,
  row_unit_neg: Vec<(u32, u16)>,
  row_small: Vec<(u32, u16)>,
  row_general: Vec<(u32, u16)>,
  /// Column indices for val=+1 entries
  unit_pos_cols: Vec<u32>,
  /// Column indices for val=-1 entries
  unit_neg_cols: Vec<u32>,
  /// (column_index, signed_coeff) for small integer entries (|coeff| in 2..=7)
  small_cols: Vec<u32>,
  small_coeffs: Vec<i8>,
  /// (column_index, coefficient) for general entries
  general_cols: Vec<u32>,
  general_vals: Vec<F>,
}

impl<F: PrimeField> PrecomputedSparseMatrix<F> {
  /// Build from a CSR SparseMatrix by classifying entries.
  /// Returns `None` if the matrix dimensions exceed compact index limits
  /// (>4B columns or >65K entries per row per category), in which case
  /// callers should fall back to the standard SparseMatrix path.
  pub fn from_sparse(m: &SparseMatrix<F>) -> Option<Self> {
    let num_rows = m.indptr.len() - 1;
    let num_cols = m.cols;

    if num_cols > u32::MAX as usize {
      return None;
    }

    let one = F::ONE;
    let neg_one = -F::ONE;

    // Precompute small positive/negative field elements for comparison
    let small_pos: Vec<F> = (2u64..=7).map(F::from).collect();
    let small_neg: Vec<F> = (2u64..=7).map(|k| -F::from(k)).collect();

    let mut row_unit_pos = Vec::with_capacity(num_rows);
    let mut row_unit_neg = Vec::with_capacity(num_rows);
    let mut row_small = Vec::with_capacity(num_rows);
    let mut row_general = Vec::with_capacity(num_rows);
    let mut unit_pos_cols = Vec::new();
    let mut unit_neg_cols = Vec::new();
    let mut small_cols = Vec::new();
    let mut small_coeffs: Vec<i8> = Vec::new();
    let mut general_cols = Vec::new();
    let mut general_vals = Vec::new();

    for ptrs in m.indptr.windows(2) {
      let up_start = unit_pos_cols.len() as u32;
      let un_start = unit_neg_cols.len() as u32;
      let sm_start = small_cols.len() as u32;
      let g_start = general_cols.len() as u32;

      for (&val, &col) in m.data[ptrs[0]..ptrs[1]].iter().zip(&m.indices[ptrs[0]..ptrs[1]]) {
        if val == one {
          unit_pos_cols.push(col as u32);
        } else if val == neg_one {
          unit_neg_cols.push(col as u32);
        } else if let Some(k) = small_pos.iter().position(|&v| v == val) {
          small_cols.push(col as u32);
          small_coeffs.push((k as i8) + 2);
        } else if let Some(k) = small_neg.iter().position(|&v| v == val) {
          small_cols.push(col as u32);
          small_coeffs.push(-((k as i8) + 2));
        } else {
          general_cols.push(col as u32);
          general_vals.push(val);
        }
      }

      let up_count = unit_pos_cols.len() as u32 - up_start;
      let un_count = unit_neg_cols.len() as u32 - un_start;
      let sm_count = small_cols.len() as u32 - sm_start;
      let g_count = general_cols.len() as u32 - g_start;
      if up_count > u16::MAX as u32
        || un_count > u16::MAX as u32
        || sm_count > u16::MAX as u32
        || g_count > u16::MAX as u32
      {
        return None;
      }
      row_unit_pos.push((up_start, up_count as u16));
      row_unit_neg.push((un_start, un_count as u16));
      row_small.push((sm_start, sm_count as u16));
      row_general.push((g_start, g_count as u16));
    }


    Some(Self {
      num_rows,
      num_cols,
      row_unit_pos,
      row_unit_neg,
      row_small,
      row_general,
      unit_pos_cols,
      unit_neg_cols,
      small_cols,
      small_coeffs,
      general_cols,
      general_vals,
    })
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
      6 => { let d = x.double(); d.double() + d },
      7 => { let d = x.double(); d.double() + d + x },
      _ => unreachable!(),
    };
    if coeff < 0 { -result } else { result }
  }

  #[inline(always)]
  fn compute_row_single(&self, row: usize, v: &[F]) -> F {
    let mut sum = F::ZERO;

    let (start, count) = self.row_unit_pos[row];
    for i in start..(start + count as u32) {
      sum += v[self.unit_pos_cols[i as usize] as usize];
    }

    let (start, count) = self.row_unit_neg[row];
    for i in start..(start + count as u32) {
      sum -= v[self.unit_neg_cols[i as usize] as usize];
    }

    let (start, count) = self.row_small[row];
    for i in start..(start + count as u32) {
      let idx = i as usize;
      sum += Self::small_mul(self.small_coeffs[idx], v[self.small_cols[idx] as usize]);
    }

    let (start, count) = self.row_general[row];
    for i in start..(start + count as u32) {
      let idx = i as usize;
      sum += self.general_vals[idx] * v[self.general_cols[idx] as usize];
    }

    sum
  }

  #[inline(always)]
  fn compute_row_pair(&self, row: usize, v1: &[F], v2: &[F]) -> (F, F) {
    let mut s1 = F::ZERO;
    let mut s2 = F::ZERO;

    let (start, count) = self.row_unit_pos[row];
    for i in start..(start + count as u32) {
      let col = self.unit_pos_cols[i as usize] as usize;
      s1 += v1[col];
      s2 += v2[col];
    }

    let (start, count) = self.row_unit_neg[row];
    for i in start..(start + count as u32) {
      let col = self.unit_neg_cols[i as usize] as usize;
      s1 -= v1[col];
      s2 -= v2[col];
    }

    let (start, count) = self.row_small[row];
    for i in start..(start + count as u32) {
      let idx = i as usize;
      let col = self.small_cols[idx] as usize;
      let c = self.small_coeffs[idx];
      s1 += Self::small_mul(c, v1[col]);
      s2 += Self::small_mul(c, v2[col]);
    }

    let (start, count) = self.row_general[row];
    for i in start..(start + count as u32) {
      let idx = i as usize;
      let col = self.general_cols[idx] as usize;
      let val = self.general_vals[idx];
      s1 += val * v1[col];
      s2 += val * v2[col];
    }

    (s1, s2)
  }

  /// Fast SpMV using precomputed coefficient classification.
  pub fn multiply_vec(&self, vector: &[F]) -> Vec<F> {
    debug_assert_eq!(self.num_cols, vector.len(), "SpMV: vector length mismatch");
    if self.num_rows <= 65536 {
      (0..self.num_rows).map(|r| self.compute_row_single(r, vector)).collect()
    } else {
      (0..self.num_rows).into_par_iter().map(|r| self.compute_row_single(r, vector)).collect()
    }
  }

  /// Fast dual-vector SpMV: compute (M*v1, M*v2) in a single pass.
  pub fn multiply_vec_pair(&self, v1: &[F], v2: &[F]) -> (Vec<F>, Vec<F>) {
    debug_assert_eq!(self.num_cols, v1.len(), "SpMV pair: v1 length mismatch");
    debug_assert_eq!(self.num_cols, v2.len(), "SpMV pair: v2 length mismatch");
    if self.num_rows <= 65536 {
      (0..self.num_rows).map(|r| self.compute_row_pair(r, v1, v2)).unzip()
    } else {
      (0..self.num_rows).into_par_iter().map(|r| self.compute_row_pair(r, v1, v2)).unzip()
    }
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
    if self.indptr.len() <= 65537 {
      self
        .indptr
        .windows(2)
        .map(|ptrs| {
          self
            .get_row_unchecked(ptrs.try_into().unwrap())
            .map(|(val, col_idx)| *val * vector[*col_idx])
            .sum()
        })
        .collect()
    } else {
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
  }

  /// Multiply by two dense vectors into pre-allocated output buffers.
  /// Avoids allocation when the same-sized outputs are reused across steps.
  pub fn multiply_vec_pair_into(
    &self,
    v1: &[F],
    v2: &[F],
    out1: &mut [F],
    out2: &mut [F],
  ) {
    debug_assert_eq!(self.cols, v1.len());
    debug_assert_eq!(self.cols, v2.len());
    let n = self.indptr.len() - 1;
    debug_assert!(out1.len() >= n);
    debug_assert!(out2.len() >= n);

    if n <= 4096 {
      for (row_idx, ptrs) in self.indptr.windows(2).enumerate() {
        let mut s1 = F::ZERO;
        let mut s2 = F::ZERO;
        for (val, col_idx) in self.get_row_unchecked(ptrs.try_into().unwrap()) {
          s1 += *val * v1[*col_idx];
          s2 += *val * v2[*col_idx];
        }
        out1[row_idx] = s1;
        out2[row_idx] = s2;
      }
    } else {
      use rayon::prelude::*;
      // Split output into chunks that can be written independently
      let results: Vec<(F, F)> = self
        .indptr
        .par_windows(2)
        .map(|ptrs| {
          let mut s1 = F::ZERO;
          let mut s2 = F::ZERO;
          for (val, col_idx) in self.get_row_unchecked(ptrs.try_into().unwrap()) {
            s1 += *val * v1[*col_idx];
            s2 += *val * v2[*col_idx];
          }
          (s1, s2)
        })
        .collect();
      for (row_idx, (s1, s2)) in results.into_iter().enumerate() {
        out1[row_idx] = s1;
        out2[row_idx] = s2;
      }
    }
  }

  /// Multiply by two dense vectors simultaneously in a single pass over the matrix.
  /// Returns (M*v1, M*v2). This is faster than two separate multiply_vec calls
  /// because the sparse matrix data is only loaded from memory once.
  pub fn multiply_vec_pair(&self, v1: &[F], v2: &[F]) -> (Vec<F>, Vec<F>) {
    assert_eq!(self.cols, v1.len(), "invalid shape for v1");
    assert_eq!(self.cols, v2.len(), "invalid shape for v2");

    if self.indptr.len() <= 65537 {
      self
        .indptr
        .windows(2)
        .map(|ptrs| {
          let row = self.get_row_unchecked(ptrs.try_into().unwrap());
          let mut s1 = F::ZERO;
          let mut s2 = F::ZERO;
          for (val, col_idx) in row {
            s1 += *val * v1[*col_idx];
            s2 += *val * v2[*col_idx];
          }
          (s1, s2)
        })
        .unzip()
    } else {
      self
        .indptr
        .par_windows(2)
        .map(|ptrs| {
          let row = self.get_row_unchecked(ptrs.try_into().unwrap());
          let mut s1 = F::ZERO;
          let mut s2 = F::ZERO;
          for (val, col_idx) in row {
            s1 += *val * v1[*col_idx];
            s2 += *val * v2[*col_idx];
          }
          (s1, s2)
        })
        .unzip()
    }
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
      // Row 2: small negative
      (2, 0, -Fr::from(2)),
      (2, 1, -Fr::from(3)),
      (2, 2, -Fr::from(7)),
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
    let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix).unwrap();
    let result = precomputed.multiply_vec(&vector);
    assert_eq!(expected, result);
  }

  #[test]
  fn test_precomputed_multiply_vec_pair() {
    let (matrix, v1) = mixed_coefficient_matrix();
    let v2: Vec<Fr> = v1.iter().map(|x| *x + Fr::from(10)).collect();

    let expected1 = matrix.multiply_vec(&v1);
    let expected2 = matrix.multiply_vec(&v2);

    let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix).unwrap();
    let (result1, result2) = precomputed.multiply_vec_pair(&v1, &v2);

    assert_eq!(expected1, result1);
    assert_eq!(expected2, result2);
  }

  proptest! {
    #[test]
    fn test_precomputed_matches_sparse(mut coo_matrix in coo_strategy()) {
      coo_matrix.sort_by_key(|(row, col, _val)| (*row, *col));
      coo_matrix.dedup_by_key(|(row, col, _val)| (*row, *col));
      let coo_matrix = coo_matrix.into_iter().map(|(row, col, val)| (row, col, val.0)).collect::<Vec<_>>();

      let matrix = SparseMatrix::new(&coo_matrix, 100, 100);
      let precomputed = PrecomputedSparseMatrix::from_sparse(&matrix).unwrap();

      let v1: Vec<Fr> = (0..100).map(|i| Fr::from(i + 1)).collect();
      let v2: Vec<Fr> = (0..100).map(|i| Fr::from(i * 3 + 7)).collect();

      // Single-vector: precomputed must match SparseMatrix
      let expected = matrix.multiply_vec(&v1);
      let result = precomputed.multiply_vec(&v1);
      prop_assert_eq!(&expected, &result);

      // Dual-vector: both outputs must match
      let expected1 = matrix.multiply_vec(&v1);
      let expected2 = matrix.multiply_vec(&v2);
      let (result1, result2) = precomputed.multiply_vec_pair(&v1, &v2);
      prop_assert_eq!(&expected1, &result1);
      prop_assert_eq!(&expected2, &result2);
    }
  }
}
