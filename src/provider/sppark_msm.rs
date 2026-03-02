//! GPU-accelerated MSM using sppark's sort-based Pippenger kernel for BN254.
//!
//! Uses sppark's `mult_pippenger` (fresh handle per call).
//!
//! Handles pathological scalar distributions (where 25%+ of scalars share the same
//! Pippenger bucket digits across all windows) via **scalar decomposition**:
//! split MSM(s, G) into MSM(a, G) + MSM(s-a, G) where a_i = r*(i+1) is an
//! arithmetic progression with random stride r. Both halves have uniform digit
//! distribution, converting a 7-second pathological MSM into two 55ms fast MSMs.

#![allow(unsafe_code)]

use halo2curves::bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point};
use rayon::prelude::*;
use std::sync::Mutex;

extern "C" {
  fn sppark_msm_with_generators(
    points: *const u64,
    scalars: *const u64,
    result: *mut u64,
    n: i32,
  ) -> i32;
}

/// GPU access must be serialized — sppark's mult_pippenger is not thread-safe
static GPU_LOCK: Mutex<()> = Mutex::new(());

use std::sync::atomic::{AtomicUsize, Ordering};
static GPU_CALLS: AtomicUsize = AtomicUsize::new(0);
static CPU_CALLS: AtomicUsize = AtomicUsize::new(0);
static DECOMP_CALLS: AtomicUsize = AtomicUsize::new(0);

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
const PATHOLOGICAL_CHECK_THRESHOLD: usize = 100_000;

/// Print MSM call statistics
pub fn print_stats() {
  eprintln!(
    "[sppark_msm stats] GPU={} CPU={} decomposed={}",
    GPU_CALLS.load(Ordering::Relaxed),
    CPU_CALLS.load(Ordering::Relaxed),
    DECOMP_CALLS.load(Ordering::Relaxed),
  );
}

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

/// Detect pathological scalar distribution by sampling window-digit histograms.
///
/// sppark's sort-based Pippenger degrades when many scalars share the same
/// digit in any window — this causes one bucket to accumulate millions of
/// points sequentially. We sample scalars, compute a histogram for one
/// representative window, and check if the max bucket exceeds 5% of samples.
fn is_pathological(scalars: &[Scalar], effective_len: usize) -> bool {
  if effective_len < PATHOLOGICAL_CHECK_THRESHOLD {
    return false;
  }
  use ff::PrimeField;
  let sample_size = std::cmp::min(500, effective_len);
  let step = effective_len / sample_size;

  // Check window 5 (bits 65-77) — representative middle window
  let wbits = 13u32;
  let win = 5u32;
  let bit_off = (win * wbits) as usize;
  let byte_off = bit_off / 8;
  let bit_shift = bit_off % 8;
  let mask = (1u64 << wbits) - 1;
  let num_buckets = 1usize << wbits;
  let mut bucket_counts = vec![0u32; num_buckets];

  for i in 0..sample_size {
    let repr = scalars[i * step].to_repr();
    let bytes = repr.as_ref();
    let mut val: u64 = 0;
    for j in 0..3usize {
      if byte_off + j < bytes.len() {
        val |= (bytes[byte_off + j] as u64) << (j * 8);
      }
    }
    let digit = ((val >> bit_shift) & mask) as usize;
    bucket_counts[digit] += 1;
  }

  let max_bucket = *bucket_counts.iter().max().unwrap() as usize;
  // Pathological if >5% of samples land in one bucket (expected: ~0.012%)
  max_bucket > sample_size / 20
}

/// Perform GPU MSM using sppark's mult_pippenger.
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

/// MSM with scalar decomposition to break pathological bucket imbalance.
///
/// Decomposes: MSM(s, G) = MSM(a, G) + MSM(s-a, G)
/// where a_i = r * (i+1) is an arithmetic progression with random stride r.
/// Field multiplication scrambles bits, so both a_i and (s_i - a_i) have
/// uniform digit distribution across all Pippenger windows.
///
/// Caller must hold GPU_LOCK.
fn gpu_msm_decomposed(
  scalars: &[Scalar],
  bases: &[Affine],
  effective_len: usize,
) -> Point {
  use ff::Field;

  let r = Scalar::random(&mut rand_core::OsRng);

  // Compute a_i = r * (i+1) using parallel running sums
  let num_threads = rayon::current_num_threads();
  let chunk_size = (effective_len + num_threads - 1) / num_threads;
  let mut buf = vec![Scalar::ZERO; effective_len];

  buf
    .par_chunks_mut(chunk_size)
    .enumerate()
    .for_each(|(chunk_idx, chunk)| {
      let start_idx = chunk_idx * chunk_size + 1;
      let mut acc = r * Scalar::from(start_idx as u64);
      for elem in chunk.iter_mut() {
        *elem = acc;
        acc += r;
      }
    });

  // First MSM: sum(a_i * G_i)
  let result1 = gpu_msm(&buf, bases, effective_len);

  // Compute b_i = s_i - a_i (overwrite buf in-place)
  buf
    .par_iter_mut()
    .zip(scalars[..effective_len].par_iter())
    .for_each(|(a, s)| *a = *s - *a);

  // Second MSM: sum(b_i * G_i)
  let result2 = gpu_msm(&buf, bases, effective_len);

  use std::ops::Add;
  result1.add(result2)
}

/// Perform MSM using sppark's GPU kernel.
/// Falls back to CPU for small inputs, uses scalar decomposition for pathological distributions.
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  if scalars.is_empty() {
    return Point::default();
  }

  let effective_len = trim_trailing_zeros(scalars);

  if effective_len < GPU_MSM_THRESHOLD {
    CPU_CALLS.fetch_add(1, Ordering::Relaxed);
    return msm(&scalars[..effective_len], &bases[..effective_len]);
  }

  let pathological = is_pathological(scalars, effective_len);

  let _gpu = GPU_LOCK.lock().unwrap();
  let t = std::time::Instant::now();

  let result = if pathological {
    DECOMP_CALLS.fetch_add(1, Ordering::Relaxed);
    gpu_msm_decomposed(scalars, bases, effective_len)
  } else {
    GPU_CALLS.fetch_add(1, Ordering::Relaxed);
    gpu_msm(scalars, bases, effective_len)
  };

  let elapsed = t.elapsed();
  drop(_gpu);
  eprintln!(
    "[sppark_msm] n={} effective={} {} time={:.1}ms",
    scalars.len(),
    effective_len,
    if pathological { "DECOMPOSED" } else { "GPU" },
    elapsed.as_secs_f64() * 1000.0
  );
  result
}

/// Perform batch MSM — each scalar vector is a separate MSM with the same generators.
pub fn batch_vartime_multiscalar_mul(scalars: &[Vec<Scalar>], bases: &[Affine]) -> Vec<Point> {
  if scalars.is_empty() {
    return vec![];
  }

  // Hold GPU lock for the entire batch to avoid per-call lock overhead
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
      if is_pathological(s, effective_len) {
        gpu_msm_decomposed(s, bases, effective_len)
      } else {
        gpu_msm(s, bases, effective_len)
      }
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
