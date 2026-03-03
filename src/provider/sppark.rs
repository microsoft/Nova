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

  /// Free the cached generator data on GPU.
  /// Called automatically on process exit, but can be invoked explicitly
  /// to release GPU memory earlier (e.g., after proving is done).
  fn sppark_msm_free();

  fn sppark_msm_from_device(
    d_scalars: *mut u8,
    result: *mut u64,
    n: i32,
  ) -> i32;

  fn sppark_ensure_generators(
    points: *const u64,
    n: i32,
  );

  fn sppark_sync_device();
}

/// GPU access must be serialized — sppark's kernels are not thread-safe
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Free the cached GPU generator data.
///
/// Generators are cached on the GPU for the lifetime of the process.
/// Call this after proving is complete to release GPU memory earlier.
/// It is safe to call multiple times; subsequent MSM calls will re-upload.
#[allow(unsafe_code)]
pub fn free_gpu_generators() {
  let _gpu = GPU_LOCK.lock().unwrap();
  unsafe { sppark_msm_free() };
}

/// Convert sppark's Jacobian (X,Y,Z) to halo2curves G1 (homogeneous projective).
///
/// sppark Jacobian: affine_x = X/Z², affine_y = Y/Z³
/// halo2curves projective: affine_x = X/Z, affine_y = Y/Z
///
/// sppark's `mont_t` stores field elements in Montgomery form (`a·R mod p`).
/// We use `from_raw_bytes` (which wraps raw limbs directly) rather than
/// `from_raw` (which multiplies by R², assuming standard-form input).
///
/// Returns `None` if the GPU output is malformed (z off-range, zero Z,
/// non-invertible Z, or off-curve affine point).
#[allow(unsafe_code)]
fn jacobian_to_point(result: &[u64; 12]) -> Option<Point> {
  use halo2curves::{bn256::Fq, serde::SerdeObject};

  // Re-interpret the u64 limbs as little-endian bytes (same layout).
  let bytes: &[u8] = unsafe { std::slice::from_raw_parts(result.as_ptr() as *const u8, 96) };

  // `from_raw_bytes` wraps Montgomery-form limbs directly and checks < modulus.
  let x = Fq::from_raw_bytes(&bytes[0..32])?;
  let y = Fq::from_raw_bytes(&bytes[32..64])?;
  let z = Fq::from_raw_bytes(&bytes[64..96])?;

  if z.is_zero().into() {
    return Some(Point::default());
  }

  // sppark Jacobian: affine_x = X / Z^2, affine_y = Y / Z^3.
  let z_inv: Fq = Option::from(z.invert())?;
  let z_inv2 = z_inv.square();
  let z_inv3 = z_inv2 * z_inv;

  let x_aff = x * z_inv2;
  let y_aff = y * z_inv3;

  let affine: Affine = Option::from(Affine::from_xy(x_aff, y_aff))?;
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
  // Guard against silent truncation if lengths ever exceed u32::MAX.
  let Ok(n_bases) = u32::try_from(bases.len()) else {
    return msm(&scalars[..effective_len], &bases[..effective_len]);
  };
  let Ok(n_scalars) = u32::try_from(effective_len) else {
    return msm(&scalars[..effective_len], &bases[..effective_len]);
  };

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
      n_bases,
      n_scalars,
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

/// Perform MSM using scalars already on GPU device memory.
/// Generators must be cached from a prior MSM call.
/// Caller must hold GPU_LOCK.
fn gpu_msm_device(d_scalars: *mut u8, n: usize) -> Point {
  let mut result = [0u64; 12];
  let err = unsafe { sppark_msm_from_device(d_scalars, result.as_mut_ptr(), n as i32) };
  assert_eq!(err, 0, "sppark MSM from device failed with error code {}", err);
  jacobian_to_point(&result)
}

/// Ensure generators are cached on GPU at the specified size.
/// Must be called before `msm_from_device` if no prior `vartime_multiscalar_mul` has been done.
pub fn ensure_generators_cached(bases: &[Affine]) {
  let _gpu = GPU_LOCK.lock().unwrap();
  unsafe { sppark_ensure_generators(bases.as_ptr() as *const u64, bases.len() as i32) };
}

/// Synchronize GPU device (wait for all pending work to complete).
pub fn sync_device() {
  unsafe { sppark_sync_device() };
}

/// Perform MSM using scalars already on GPU (device pointer).
/// Generators must be cached from a prior GPU MSM call or `ensure_generators_cached`.
/// Acquires GPU lock internally.
pub fn msm_from_device(d_scalars: *mut u8, n: usize) -> Point {
  let _gpu = GPU_LOCK.lock().unwrap();
  gpu_msm_device(d_scalars, n)
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
/// Uses a higher GPU threshold than single MSMs to avoid per-call GPU overhead
/// when processing many small polynomials (e.g., HyperKZG folded polynomials).
pub fn batch_vartime_multiscalar_mul(scalars: &[Vec<Scalar>], bases: &[Affine]) -> Vec<Point> {
  if scalars.is_empty() {
    return vec![];
  }

  // Higher threshold for batch: GPU kernel launch overhead (~5ms) dominates for small MSMs.
  // CPU MSM for <64K elements is faster than GPU when called in a batch of many MSMs.
  const BATCH_GPU_THRESHOLD: usize = 1 << 16; // 65536

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
      if effective_len < BATCH_GPU_THRESHOLD {
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

  /// Validate that `jacobian_to_point` correctly interprets Montgomery-form
  /// Jacobian coordinates.  We construct a known projective point, extract its
  /// internal Montgomery-form limbs via `to_raw_bytes`, pack them as the
  /// `[u64; 12]` layout that sppark would produce, and round-trip through
  /// `jacobian_to_point`.
  #[test]
  fn test_jacobian_to_point_roundtrip() {
    use halo2curves::{
      bn256::{Fq, G1},
      group::Curve,
      serde::SerdeObject,
    };

    // Choose a known non-identity point: generator * 42
    let scalar = Scalar::from(42u64);
    let proj: G1 = Affine::generator() * scalar;
    let affine_expected = proj.to_affine();

    // Extract affine coordinates in Montgomery form
    let coords = affine_expected.coordinates().unwrap();
    let x: Fq = *coords.x();
    let y: Fq = *coords.y();

    // Build Jacobian (X, Y, Z) = (x, y, 1) in Montgomery form
    let x_bytes = x.to_raw_bytes();
    let y_bytes = y.to_raw_bytes();
    let one_bytes = Fq::ONE.to_raw_bytes();

    // Pack into [u64; 12] — each Fq is 4 limbs (32 bytes), little-endian
    let mut packed = [0u64; 12];
    for i in 0..4 {
      packed[i] = u64::from_le_bytes(x_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
      packed[4 + i] = u64::from_le_bytes(y_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
      packed[8 + i] = u64::from_le_bytes(one_bytes[i * 8..(i + 1) * 8].try_into().unwrap());
    }

    let result = jacobian_to_point(&packed);
    assert!(result.is_some(), "jacobian_to_point returned None");
    let result_affine = result.unwrap().to_affine();
    assert_eq!(
      result_affine, affine_expected,
      "round-tripped point does not match"
    );
  }

  /// Verify that `jacobian_to_point` returns `None` for clearly invalid data
  /// (all 0xFF bytes exceed the BN254 modulus).
  #[test]
  fn test_jacobian_to_point_rejects_invalid() {
    let bad = [u64::MAX; 12];
    assert!(
      jacobian_to_point(&bad).is_none(),
      "should reject out-of-range limbs"
    );
  }
}
