/// # Fixed-base Scalar Multiplication
///
/// This module provides an implementation of fixed-base scalar multiplication on elliptic curves.
///
/// The multiplication is optimized through a windowed method, where scalars are broken into fixed-size
/// windows, pre-computation tables are generated, and results are efficiently combined.
use ff::{PrimeField, PrimeFieldBits};
use group::{prime::PrimeCurve, Curve};

use rayon::prelude::*;

/// Determines the window size for scalar multiplication based on the number of scalars.
///
/// This is used to balance between pre-computation and number of point additions.
pub(crate) fn get_mul_window_size(num_scalars: usize) -> usize {
  if num_scalars < 32 {
    3
  } else {
    (num_scalars as f64).ln().ceil() as usize
  }
}

/// Generates a table of multiples of a base point `g` for use in windowed scalar multiplication.
///
/// This pre-computes multiples of a base point for each window and organizes them
/// into a table for quick lookup during the scalar multiplication process. The table is a vector
/// of vectors, each inner vector corresponding to a window and containing the multiples of `g`
/// for that window.
pub(crate) fn get_window_table<T>(
  scalar_size: usize,
  window: usize,
  g: T,
) -> Vec<Vec<T::AffineRepr>>
where
  T: Curve,
  T::AffineRepr: Send,
{
  let in_window = 1 << window;
  // Number of outer iterations needed to cover the entire scalar
  let outerc = (scalar_size + window - 1) / window;

  // Number of multiples of the window's "outer point" needed for each window (fewer for the last window)
  let last_in_window = 1 << (scalar_size - (outerc - 1) * window);

  let mut multiples_of_g = vec![vec![T::identity(); in_window]; outerc];

  // Compute the multiples of g for each window
  // g_outers = [ 2^{k*window}*g for k in 0..outerc]
  let mut g_outer = g;
  let mut g_outers = Vec::with_capacity(outerc);
  for _ in 0..outerc {
    g_outers.push(g_outer);
    for _ in 0..window {
      g_outer = g_outer.double();
    }
  }
  multiples_of_g
    .par_iter_mut()
    .enumerate()
    .zip_eq(g_outers)
    .for_each(|((outer, multiples_of_g), g_outer)| {
      let cur_in_window = if outer == outerc - 1 {
        last_in_window
      } else {
        in_window
      };

      // multiples_of_g = [id, g_outer, 2*g_outer, 3*g_outer, ...],
      // where g_outer = 2^{outer*window}*g
      let mut g_inner = T::identity();
      for inner in multiples_of_g.iter_mut().take(cur_in_window) {
        *inner = g_inner;
        g_inner.add_assign(&g_outer);
      }
    });
  multiples_of_g
    .par_iter()
    .map(|s| s.iter().map(|s| s.to_affine()).collect())
    .collect()
}

/// Performs the actual windowed scalar multiplication using a pre-computed table of points.
///
/// Given a scalar and a table of pre-computed multiples of a base point, this function
/// efficiently computes the scalar multiplication by breaking the scalar into windows and
/// adding the corresponding multiples from the table.
pub(crate) fn windowed_mul<T>(
  outerc: usize,
  window: usize,
  multiples_of_g: &[Vec<T::Affine>],
  scalar: &T::Scalar,
) -> T
where
  T: PrimeCurve,
  T::Scalar: PrimeFieldBits,
{
  let modulus_size = <T::Scalar as PrimeField>::NUM_BITS as usize;
  let scalar_val: Vec<bool> = scalar.to_le_bits().into_iter().collect();

  let mut res = T::identity();
  for outer in 0..outerc {
    let mut inner = 0usize;
    for i in 0..window {
      if outer * window + i < modulus_size && scalar_val[outer * window + i] {
        inner |= 1 << i;
      }
    }
    res.add_assign(&multiples_of_g[outer][inner]);
  }
  res
}

/// Computes multiple scalar multiplications simultaneously using the windowed method.
pub(crate) fn multi_scalar_mul<T>(
  scalar_size: usize,
  window: usize,
  table: &[Vec<T::AffineRepr>],
  v: &[T::Scalar],
) -> Vec<T>
where
  T: PrimeCurve,
  T::Scalar: PrimeFieldBits,
{
  let outerc = (scalar_size + window - 1) / window;
  assert!(outerc <= table.len());

  v.par_iter()
    .map(|e| windowed_mul::<T>(outerc, window, table, e))
    .collect::<Vec<_>>()
}
