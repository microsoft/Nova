// Allow `&Matrix` in function signatures.
#![allow(clippy::ptr_arg)]

use ff::PrimeField;

/// Matrix functions here are, at least for now, quick and dirty — intended only to support precomputation of poseidon optimization.
///
/// Matrix represented as a Vec of rows, so that m[i][j] represents the jth column of the ith row in Matrix, m.
pub(crate) type Matrix<T> = Vec<Vec<T>>;

pub(crate) fn rows<T>(matrix: &Matrix<T>) -> usize {
  matrix.len()
}

/// Panics if `matrix` is not actually a matrix. So only use any of these functions on well-formed data.
/// Only use during constant calculation on matrices known to have been constructed correctly.
fn columns<T>(matrix: &Matrix<T>) -> usize {
  if matrix.is_empty() {
    0
  } else {
    let column_length = matrix[0].len();
    for row in matrix {
      assert!(row.len() == column_length, "not a matrix");
    }
    column_length
  }
}

// This wastefully discards the actual inverse, if it exists, so in general callers should
// just call `invert` if that result will be needed.
pub(crate) fn is_invertible<F: PrimeField>(matrix: &Matrix<F>) -> bool {
  is_square(matrix) && invert(matrix).is_some()
}

fn scalar_vec_mul<F: PrimeField>(scalar: F, vec: &[F]) -> Vec<F> {
  vec
    .iter()
    .map(|val| {
      let mut prod = scalar;
      prod.mul_assign(val);
      prod
    })
    .collect::<Vec<_>>()
}

pub(crate) fn mat_mul<F: PrimeField>(a: &Matrix<F>, b: &Matrix<F>) -> Option<Matrix<F>> {
  if rows(a) != columns(b) {
    return None;
  };

  let b_t = transpose(b);

  let res = a
    .iter()
    .map(|input_row| {
      b_t
        .iter()
        .map(|transposed_column| vec_mul(input_row, transposed_column))
        .collect()
    })
    .collect();

  Some(res)
}

fn vec_mul<F: PrimeField>(a: &[F], b: &[F]) -> F {
  a.iter().zip(b).fold(F::ZERO, |mut acc, (v1, v2)| {
    let mut tmp = *v1;
    tmp.mul_assign(v2);
    acc.add_assign(&tmp);
    acc
  })
}

pub(crate) fn vec_add<F: PrimeField>(a: &[F], b: &[F]) -> Vec<F> {
  a.iter()
    .zip(b.iter())
    .map(|(a, b)| {
      let mut res = *a;
      res.add_assign(b);
      res
    })
    .collect::<Vec<_>>()
}

pub(crate) fn vec_sub<F: PrimeField>(a: &[F], b: &[F]) -> Vec<F> {
  a.iter()
    .zip(b.iter())
    .map(|(a, b)| {
      let mut res = *a;
      res.sub_assign(b);
      res
    })
    .collect::<Vec<_>>()
}

/// Left-multiply a vector by a square matrix of same size: MV where V is considered a column vector.
pub(crate) fn left_apply_matrix<F: PrimeField>(m: &Matrix<F>, v: &[F]) -> Vec<F> {
  assert!(is_square(m), "Only square matrix can be applied to vector.");
  assert_eq!(
    rows(m),
    v.len(),
    "Matrix can only be applied to vector of same size."
  );

  let mut result = vec![F::ZERO; v.len()];

  for (result, row) in result.iter_mut().zip(m.iter()) {
    for (mat_val, vec_val) in row.iter().zip(v) {
      let mut tmp = *mat_val;
      tmp.mul_assign(vec_val);
      result.add_assign(&tmp);
    }
  }
  result
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn transpose<F: PrimeField>(matrix: &Matrix<F>) -> Matrix<F> {
  let size = rows(matrix);
  let mut new = Vec::with_capacity(size);
  for j in 0..size {
    let mut row = Vec::with_capacity(size);
    for i in 0..size {
      row.push(matrix[i][j])
    }
    new.push(row);
  }
  new
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn make_identity<F: PrimeField>(size: usize) -> Matrix<F> {
  let mut result = vec![vec![F::ZERO; size]; size];
  for i in 0..size {
    result[i][i] = F::ONE;
  }
  result
}

pub(crate) fn kronecker_delta<F: PrimeField>(i: usize, j: usize) -> F {
  if i == j {
    F::ONE
  } else {
    F::ZERO
  }
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn is_identity<F: PrimeField>(matrix: &Matrix<F>) -> bool {
  for i in 0..rows(matrix) {
    for j in 0..columns(matrix) {
      if matrix[i][j] != kronecker_delta(i, j) {
        return false;
      }
    }
  }
  true
}

pub(crate) fn is_square<T>(matrix: &Matrix<T>) -> bool {
  rows(matrix) == columns(matrix)
}

pub(crate) fn minor<F: PrimeField>(matrix: &Matrix<F>, i: usize, j: usize) -> Matrix<F> {
  assert!(is_square(matrix));
  let size = rows(matrix);
  assert!(size > 0);
  let new = matrix
    .iter()
    .enumerate()
    .filter_map(|(ii, row)| {
      if ii == i {
        None
      } else {
        let mut new_row = row.clone();
        new_row.remove(j);
        Some(new_row)
      }
    })
    .collect();
  assert!(is_square(&new));
  new
}

// Assumes matrix is partially reduced to upper triangular. `column` is the column to eliminate from all rows.
// Returns `None` if either:
//   - no non-zero pivot can be found for `column`
//   - `column` is not the first
fn eliminate<F: PrimeField>(
  matrix: &Matrix<F>,
  column: usize,
  shadow: &mut Matrix<F>,
) -> Option<Matrix<F>> {
  let zero = F::ZERO;
  let pivot_index = (0..rows(matrix))
    .find(|&i| matrix[i][column] != zero && (0..column).all(|j| matrix[i][j] == zero))?;

  let pivot = &matrix[pivot_index];
  let pivot_val = pivot[column];

  // This should never fail since we have a non-zero `pivot_val` if we got here.
  let inv_pivot = Option::from(pivot_val.invert())?;
  let mut result = Vec::with_capacity(matrix.len());
  result.push(pivot.clone());

  for (i, row) in matrix.iter().enumerate() {
    if i == pivot_index {
      continue;
    };
    let val = row[column];
    if val == zero {
      // Value is already eliminated.
      result.push(row.to_vec());
    } else {
      let mut factor = val;
      factor.mul_assign(&inv_pivot);

      let scaled_pivot = scalar_vec_mul(factor, pivot);
      let eliminated = vec_sub(row, &scaled_pivot);
      result.push(eliminated);

      let shadow_pivot = &shadow[pivot_index];
      let scaled_shadow_pivot = scalar_vec_mul(factor, shadow_pivot);
      let shadow_row = &shadow[i];
      shadow[i] = vec_sub(shadow_row, &scaled_shadow_pivot);
    }
  }

  let pivot_row = shadow.remove(pivot_index);
  shadow.insert(0, pivot_row);

  Some(result)
}

// `matrix` must be square.
fn upper_triangular<F: PrimeField>(
  matrix: &Matrix<F>,
  shadow: &mut Matrix<F>,
) -> Option<Matrix<F>> {
  assert!(is_square(matrix));
  let mut result = Vec::with_capacity(matrix.len());
  let mut shadow_result = Vec::with_capacity(matrix.len());

  let mut curr = matrix.clone();
  let mut column = 0;
  while curr.len() > 1 {
    let initial_rows = curr.len();

    curr = eliminate(&curr, column, shadow)?;
    result.push(curr[0].clone());
    shadow_result.push(shadow[0].clone());
    column += 1;

    curr = curr[1..].to_vec();
    *shadow = shadow[1..].to_vec();
    assert_eq!(curr.len(), initial_rows - 1);
  }
  result.push(curr[0].clone());
  shadow_result.push(shadow[0].clone());

  *shadow = shadow_result;

  Some(result)
}

// `matrix` must be upper triangular.
fn reduce_to_identity<F: PrimeField>(
  matrix: &Matrix<F>,
  shadow: &mut Matrix<F>,
) -> Option<Matrix<F>> {
  let size = rows(matrix);
  let mut result: Matrix<F> = Vec::new();
  let mut shadow_result: Matrix<F> = Vec::new();

  for i in 0..size {
    let idx = size - i - 1;
    let row = &matrix[idx];
    let shadow_row = &shadow[idx];

    let val = row[idx];
    let inv = {
      let inv = val.invert();
      // If `val` is zero, then there is no inverse, and we cannot compute a result.
      if inv.is_none().into() {
        return None;
      }
      inv.unwrap()
    };

    let mut normalized = scalar_vec_mul(inv, row);
    let mut shadow_normalized = scalar_vec_mul(inv, shadow_row);

    for j in 0..i {
      let idx = size - j - 1;
      let val = normalized[idx];
      let subtracted = scalar_vec_mul(val, &result[j]);
      let result_subtracted = scalar_vec_mul(val, &shadow_result[j]);

      normalized = vec_sub(&normalized, &subtracted);
      shadow_normalized = vec_sub(&shadow_normalized, &result_subtracted);
    }

    result.push(normalized);
    shadow_result.push(shadow_normalized);
  }

  result.reverse();
  shadow_result.reverse();

  *shadow = shadow_result;
  Some(result)
}

//
pub(crate) fn invert<F: PrimeField>(matrix: &Matrix<F>) -> Option<Matrix<F>> {
  let mut shadow = make_identity(columns(matrix));
  let ut = upper_triangular(matrix, &mut shadow);

  ut.and_then(|x| reduce_to_identity(&x, &mut shadow))
    .and(Some(shadow))
}
