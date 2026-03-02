//! GPU-accelerated MSM using sppark's sort-based Pippenger kernel for BN254.
//!
//! Uses a modified `accumulate_parallel` kernel that handles both normal and
//! pathological scalar distributions via parallel tree reduction for large buckets.
//! Generators are cached on GPU across calls (Nova's commitment key is fixed).

#![allow(unsafe_code)]

use halo2curves::bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point};
use std::sync::Mutex;

extern "C" {
  fn sppark_msm_with_generators(
    points: *const u64,
    scalars: *const u64,
    result: *mut u64,
    n: i32,
  ) -> i32;
}

/// GPU access must be serialized — sppark's kernels are not thread-safe
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Convert sppark's Jacobian (X,Y,Z) to halo2curves G1 (homogeneous projective).
///
/// sppark Jacobian: affine_x = X/Z², affine_y = Y/Z³
/// halo2curves projective: affine_x = X/Z, affine_y = Y/Z
///
/// Conversion: Xp = X·Z, Yp = Y, Zp = Z³
fn jacobian_to_point(result: &[u64; 12]) -> Point {
  use ff::Field;
  use halo2curves::bn256::Fq;

  let x = unsafe { std::ptr::read(result.as_ptr().add(0) as *const Fq) };
  let y = unsafe { std::ptr::read(result.as_ptr().add(4) as *const Fq) };
  let z = unsafe { std::ptr::read(result.as_ptr().add(8) as *const Fq) };

  if z.is_zero().into() {
    return Point::default();
  }

  let xp = x * z;
  let yp = y;
  let zp = z * z * z;

  let mut point_bytes = [0u64; 12];
  unsafe {
    std::ptr::write(point_bytes.as_mut_ptr().add(0) as *mut Fq, xp);
    std::ptr::write(point_bytes.as_mut_ptr().add(4) as *mut Fq, yp);
    std::ptr::write(point_bytes.as_mut_ptr().add(8) as *mut Fq, zp);
    std::ptr::read(point_bytes.as_ptr() as *const Point)
  }
}

use crate::provider::msm::msm;

const GPU_MSM_THRESHOLD: usize = 256;

/// Trim trailing zeros from a scalar slice.
fn trim_trailing_zeros(scalars: &[Scalar]) -> usize {
  let ptr = scalars.as_ptr() as *const u64;
  let mut len = scalars.len();
  while len > 0 {
    let i = len - 1;
    let base = unsafe { ptr.add(i * 4) };
    if unsafe { *base | *base.add(1) | *base.add(2) | *base.add(3) } != 0 {
      break;
    }
    len -= 1;
  }
  len
}

/// Perform GPU MSM with cached generators and parallel accumulate kernel.
/// Caller must hold GPU_LOCK.
fn gpu_msm(scalars: &[Scalar], bases: &[Affine], effective_len: usize) -> Point {
  let mut result = [0u64; 12];
  let err = unsafe {
    sppark_msm_with_generators(
      bases.as_ptr() as *const u64,
      scalars.as_ptr() as *const u64,
      result.as_mut_ptr(),
      effective_len as i32,
    )
  };
  assert_eq!(err, 0, "sppark MSM failed with error code {}", err);
  jacobian_to_point(&result)
}

/// Perform MSM using sppark's GPU kernel with cached generators.
/// Falls back to CPU for small inputs.
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  if scalars.is_empty() {
    return Point::default();
  }

  let effective_len = trim_trailing_zeros(scalars);

  if effective_len < GPU_MSM_THRESHOLD {
    return msm(&scalars[..effective_len], &bases[..effective_len]);
  }

  let _gpu = GPU_LOCK.lock().unwrap();
  gpu_msm(scalars, bases, effective_len)
}

/// Perform batch MSM — each scalar vector is a separate MSM with the same generators.
pub fn batch_vartime_multiscalar_mul(scalars: &[Vec<Scalar>], bases: &[Affine]) -> Vec<Point> {
  if scalars.is_empty() {
    return vec![];
  }

  let _gpu = GPU_LOCK.lock().unwrap();

  scalars
    .iter()
    .map(|s| {
      if s.is_empty() {
        return Point::default();
      }
      let effective_len = trim_trailing_zeros(s);
      if effective_len < GPU_MSM_THRESHOLD {
        return msm(&s[..effective_len], &bases[..effective_len]);
      }
      gpu_msm(s, bases, effective_len)
    })
    .collect()
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use halo2curves::msm::msm_best;

  #[test]
  fn test_vartime_multiscalar_mul_empty() {
    let scalars = vec![];
    let bases = vec![];
    let result = vartime_multiscalar_mul(&scalars, &bases);
    assert_eq!(result, Point::default());
  }

  #[test]
  fn test_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();
    let scalars = vec![Scalar::random(&mut rng), Scalar::random(&mut rng)];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];

    let result = vartime_multiscalar_mul(&scalars, &bases);
    let expected = bases[0] * scalars[0] + bases[1] * scalars[1];

    assert_eq!(result, expected);
  }

  #[test]
  fn test_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);
    let expected = msm_best(&scalars, &bases);

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let batch_len = 5;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();
    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);
    let expected: Vec<Point> = scalars
      .iter()
      .map(|s| msm_best(s, &bases))
      .collect();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);
    let expected = msm_best(&scalars, &bases);
    assert_eq!(result, expected);
  }

  #[test]
  fn test_gpu_msm_large() {
    use halo2curves::group::Curve;
    let mut rng = rand::thread_rng();
    let sample_len = 1 << 12; // 4096 — above GPU threshold

    let scalars: Vec<Scalar> = (0..sample_len).map(|_| Scalar::random(&mut rng)).collect();
    let bases: Vec<Affine> = (0..sample_len)
      .map(|_| (Affine::generator() * Scalar::random(&mut rng)).to_affine())
      .collect();

    let result = vartime_multiscalar_mul(&scalars, &bases);
    let expected = msm_best(&scalars, &bases);
    assert_eq!(result.to_affine(), expected.to_affine());
  }
}
