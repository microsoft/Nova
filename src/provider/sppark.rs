//! GPU-accelerated MSM using sppark's sort-based Pippenger kernel for BN254.
//!
//! Uses a modified `accumulate_parallel` kernel that handles both normal and
//! pathological scalar distributions via parallel tree reduction for large buckets.
//! Generators are cached on GPU across calls (Nova's commitment key is fixed).

use ff::Field;
use halo2curves::{
  bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point},
  CurveAffine,
};
use std::sync::Mutex;

#[allow(unsafe_code)]
extern "C" {
  /// `n_bases`:  number of generators to upload (full slice length).
  /// `n_scalars`: number of scalars for this MSM (≤ n_bases).
  /// `label`:    identity of the generator set (pointer of the backing `Vec`).
  fn sppark_msm_with_generators(
    points: *const u64,
    scalars: *const u64,
    result: *mut u64,
    n_bases: u32,
    n_scalars: u32,
    label: u64,
  ) -> i32;
}

/// GPU access must be serialized — sppark's kernels are not thread-safe
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Convert sppark's Jacobian (X,Y,Z) to halo2curves G1 (homogeneous projective).
///
/// sppark Jacobian: affine_x = X/Z², affine_y = Y/Z³
/// halo2curves projective: affine_x = X/Z, affine_y = Y/Z
///
/// We go through `Fq::from_raw` to avoid relying on the internal memory layout
/// of `Fq` and `G1`, then reconstruct the projective point via explicit
/// coordinate conversion.
///
/// Returns `None` if the GPU output is malformed (zero Z after `from_raw`,
/// non-invertible Z, or off-curve affine point).
fn jacobian_to_point(result: &[u64; 12]) -> Option<Point> {
  use halo2curves::bn256::Fq;

  // Interpret input limbs as three Fq elements X, Y, Z using the public API.
  let x = Fq::from_raw([result[0], result[1], result[2], result[3]]);
  let y = Fq::from_raw([result[4], result[5], result[6], result[7]]);
  let z = Fq::from_raw([result[8], result[9], result[10], result[11]]);

  if z.is_zero().into() {
    return Some(Point::default());
  }

  // sppark Jacobian: affine_x = X / Z^2, affine_y = Y / Z^3.
  let z_inv: Fq = Option::from(z.invert())?;
  let z_inv2 = z_inv.square();
  let z_inv3 = z_inv2 * z_inv;

  let x_aff = x * z_inv2;
  let y_aff = y * z_inv3;

  let affine = Option::from(Affine::from_xy(x_aff, y_aff))?;
  Some(Point::from(affine))
}

use crate::provider::msm::msm;

const GPU_MSM_THRESHOLD: usize = 256;

/// Trim trailing zero scalars from a slice.
/// Uses the safe `Scalar::ZERO` comparison instead of raw pointer arithmetic.
fn trim_trailing_zeros(scalars: &[Scalar]) -> usize {
  let mut len = scalars.len();
  while len > 0 && scalars[len - 1] == Scalar::ZERO {
    len -= 1;
  }
  len
}

/// Perform GPU MSM with cached generators and parallel accumulate kernel.
/// Caller must hold GPU_LOCK.
/// Falls back to CPU MSM on any GPU error rather than panicking.
///
/// The generator identity is `bases.as_ptr()` — the address of the backing
/// `Vec` inside `CommitmentKey`.  Every prefix slice `&ck.ck[..v.len()]`
/// shares the same base pointer, so the GPU can recognise the same
/// generator set even when the slice length varies between calls.
#[allow(unsafe_code)]
fn gpu_msm(scalars: &[Scalar], bases: &[Affine], effective_len: usize) -> Point {
  // Use the slice's base pointer as the label.  All prefix slices of the
  // same CommitmentKey share the same pointer, so the GPU recognises
  // them as the same generator set regardless of slice length.
  let label = bases.as_ptr() as u64;
  let mut result = [0u64; 12];
  // SAFETY: `bases` points to `bases.len()` contiguous Affine elements
  // (for upload).  `scalars` points to at least `effective_len` elements
  // (for MSM).  The FFI function writes 12 u64s to `result`.
  let err = unsafe {
    sppark_msm_with_generators(
      bases.as_ptr() as *const u64,
      scalars.as_ptr() as *const u64,
      result.as_mut_ptr(),
      bases.len() as u32,
      effective_len as u32,
      label,
    )
  };
  if err != 0 {
    // GPU MSM failed; fall back to CPU MSM for this call.
    return msm(&scalars[..effective_len], &bases[..effective_len]);
  }
  // If the GPU returned malformed data, fall back to CPU.
  match jacobian_to_point(&result) {
    Some(p) => p,
    None => msm(&scalars[..effective_len], &bases[..effective_len]),
  }
}

/// Perform MSM using sppark's GPU kernel with cached generators.
/// Falls back to CPU for small inputs.
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  if scalars.is_empty() {
    return Point::default();
  }

  assert!(
    bases.len() >= scalars.len(),
    "bases.len() ({}) must be >= scalars.len() ({})",
    bases.len(),
    scalars.len()
  );

  let effective_len = trim_trailing_zeros(scalars);

  if effective_len == 0 {
    return Point::default();
  }

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
      assert!(
        bases.len() >= s.len(),
        "bases.len() ({}) must be >= scalars row len ({})",
        bases.len(),
        s.len()
      );
      let effective_len = trim_trailing_zeros(s);
      if effective_len == 0 {
        return Point::default();
      }
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
    let expected: Vec<Point> = scalars.iter().map(|s| msm_best(s, &bases)).collect();

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
