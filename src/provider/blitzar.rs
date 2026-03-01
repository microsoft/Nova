//! GPU-accelerated multi-scalar multiplication using Blitzar.
//!
//! Optimizations over the naive per-call approach:
//! 1. **Cached generator conversion**: The halo2→arkworks generator conversion is done once
//!    and cached globally via `Arc`, avoiding repeated conversion on every MSM call.
//! 2. **Batch MSM**: Multiple MSMs are batched into a single GPU call using the
//!    `compute_bn254_g1_uncompressed_commitments_with_generators` API with multiple `Sequence` entries.
use blitzar::compute::{
  compute_bn254_g1_uncompressed_commitments_with_generators, convert_to_ark_bn254_g1_affine,
};
use blitzar::sequence::Sequence;
use halo2curves::bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Cached arkworks generators: (pointer address, generator count, first gen x-coord, arc-wrapped generators).
/// The Arc allows multiple threads to share the generators without cloning the data.
static CACHED_GENERATORS: Mutex<Option<(usize, usize, [u8; 32], Arc<Vec<ark_bn254::G1Affine>>)>> =
  Mutex::new(None);

/// Compute a fingerprint from the first and last generators for cache invalidation.
fn generator_fingerprint(bases: &[Affine]) -> [u8; 32] {
  use ff::PrimeField;
  if bases.is_empty() {
    return [0u8; 32];
  }
  // Use first generator's x-coordinate as fingerprint
  let repr = bases[0].x.to_repr();
  let bytes: &[u8] = repr.as_ref();
  let mut fp = [0u8; 32];
  let len = bytes.len().min(32);
  fp[..len].copy_from_slice(&bytes[..len]);
  // Mix in last generator to reduce collision probability
  if bases.len() > 1 {
    let last_repr = bases[bases.len() - 1].x.to_repr();
    let last_bytes: &[u8] = last_repr.as_ref();
    for i in 0..len.min(last_bytes.len()) {
      fp[i] ^= last_bytes[i];
    }
  }
  fp
}

/// Returns cached arkworks generators, converting and caching if necessary.
/// Returns an Arc to avoid cloning the generator vector (~64MB at 2^20).
fn get_ark_generators(bases: &[Affine]) -> Arc<Vec<ark_bn254::G1Affine>> {
  let n = bases.len();
  let fp = generator_fingerprint(bases);

  {
    let guard = CACHED_GENERATORS.lock().unwrap();
    if let Some((_, cached_n, cached_fp, ref gens)) = *guard {
      if cached_n >= n && cached_fp == fp {
        return Arc::clone(gens);
      }
    }
  }

  // Convert all generators (parallel for speed)
  let ark_generators: Vec<ark_bn254::G1Affine> = bases
    .par_iter()
    .map(|g| convert_to_ark_bn254_g1_affine(g))
    .collect();
  let arc = Arc::new(ark_generators);

  let mut guard = CACHED_GENERATORS.lock().unwrap();
  *guard = Some((bases.as_ptr() as usize, n, fp, Arc::clone(&arc)));
  arc
}

/// Performs a single multi-scalar multiplication using GPU acceleration.
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  if scalars.is_empty() {
    return Point::default();
  }

  let ark_generators = get_ark_generators(bases);

  let scalar_bytes: Vec<[u8; 32]> = scalars.par_iter().map(|s| s.to_bytes()).collect();

  let mut commitments = vec![ark_bn254::G1Affine::default(); 1];
  let sequences = [Sequence::from_raw_parts(&scalar_bytes, false)];

  compute_bn254_g1_uncompressed_commitments_with_generators(
    &mut commitments,
    &sequences,
    &ark_generators[..scalars.len()],
  );

  blitzar::compute::convert_to_halo2_bn256_g1_affine(&commitments[0]).into()
}

/// Performs a single multi-scalar multiplication with small (≤64-bit) scalars using GPU acceleration.
/// The scalars are passed to the GPU at their native byte width (e.g., 2 bytes for u16),
/// reducing data transfer and allowing the GPU to skip high-order zero buckets.
pub fn vartime_multiscalar_mul_small<T: num_integer::Integer + Into<u64> + Copy + Sync>(
  scalars: &[T],
  bases: &[Affine],
) -> Point {
  if scalars.is_empty() {
    return Point::default();
  }

  let ark_generators = get_ark_generators(bases);

  let mut commitments = vec![ark_bn254::G1Affine::default(); 1];
  let sequences = [Sequence::from_raw_parts(scalars, false)];

  compute_bn254_g1_uncompressed_commitments_with_generators(
    &mut commitments,
    &sequences,
    &ark_generators[..scalars.len()],
  );

  blitzar::compute::convert_to_halo2_bn256_g1_affine(&commitments[0]).into()
}

/// Performs a batch of multi-scalar multiplications with small (≤64-bit) scalars using GPU acceleration.
/// All MSMs are sent to the GPU in a single call for maximum throughput.
pub fn batch_vartime_multiscalar_mul_small<T: num_integer::Integer + Into<u64> + Copy + Sync>(
  scalars: &[&[T]],
  bases: &[Affine],
) -> Vec<Point> {
  if scalars.is_empty() {
    return vec![];
  }
  let num_outputs = scalars.len();
  let max_len = scalars.iter().map(|s| s.len()).max().unwrap_or(0);
  if max_len == 0 {
    return vec![Point::default(); num_outputs];
  }

  let ark_generators = get_ark_generators(bases);
  let sequences: Vec<Sequence<'_>> = scalars
    .iter()
    .map(|s| Sequence::from_raw_parts(*s, false))
    .collect();
  let mut commitments = vec![ark_bn254::G1Affine::default(); num_outputs];

  compute_bn254_g1_uncompressed_commitments_with_generators(
    &mut commitments,
    &sequences,
    &ark_generators[..max_len],
  );

  commitments
    .par_iter()
    .map(|c| blitzar::compute::convert_to_halo2_bn256_g1_affine(c).into())
    .collect()
}

/// Performs a batch of multi-scalar multiplications using GPU acceleration.
/// All MSMs are sent to the GPU in a single call for maximum throughput.
pub fn batch_vartime_multiscalar_mul(scalars: &[&[Scalar]], bases: &[Affine]) -> Vec<Point> {
  if scalars.is_empty() {
    return vec![];
  }

  let num_outputs = scalars.len();
  let max_len = scalars.iter().map(|s| s.len()).max().unwrap_or(0);
  if max_len == 0 {
    return vec![Point::default(); num_outputs];
  }

  let ark_generators = get_ark_generators(bases);

  // Convert scalars to byte arrays (parallel)
  let scalar_bytes: Vec<Vec<[u8; 32]>> = scalars
    .par_iter()
    .map(|s| s.par_iter().map(|v| v.to_bytes()).collect())
    .collect();

  // Create Sequence descriptors for each MSM
  let sequences: Vec<Sequence<'_>> = scalar_bytes.iter().map(|s| s.into()).collect();

  let mut commitments = vec![ark_bn254::G1Affine::default(); num_outputs];

  compute_bn254_g1_uncompressed_commitments_with_generators(
    &mut commitments,
    &sequences,
    &ark_generators,
  );

  commitments
    .par_iter()
    .map(|c| blitzar::compute::convert_to_halo2_bn256_g1_affine(c).into())
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
  fn test_batch_vartime_multiscalar_mul_empty() {
    let scalars: Vec<Vec<Scalar>> = vec![vec![]];
    let bases = vec![];
    let scalar_slices: Vec<&[Scalar]> = scalars.iter().map(|s| s.as_slice()).collect();

    let result = batch_vartime_multiscalar_mul(&scalar_slices, &bases);

    assert_eq!(result, [Point::default(); 1]);
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
  fn test_batch_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();

    let scalars = vec![
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
    ];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];
    let scalar_slices: Vec<&[Scalar]> = scalars.iter().map(|s| s.as_slice()).collect();

    let result = batch_vartime_multiscalar_mul(&scalar_slices, &bases);

    assert_eq!(
      result[0],
      bases[0] * scalars[0][0] + bases[1] * scalars[0][1]
    );
    assert_eq!(
      result[1],
      bases[0] * scalars[1][0] + bases[1] * scalars[1][1]
    );
  }

  #[test]
  fn test_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);

    let mut expected = Point::default();
    for i in 0..sample_len {
      expected += bases[i] * scalars[i];
    }

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();
    let scalar_slices: Vec<&[Scalar]> = scalars.iter().map(|s| s.as_slice()).collect();

    let result = batch_vartime_multiscalar_mul(&scalar_slices, &bases);

    let expected: Vec<Point> = scalars
      .iter()
      .map(|scalar_row| {
        scalar_row
          .iter()
          .enumerate()
          .map(|(i, scalar)| bases[i] * scalar)
          .sum()
      })
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
  fn test_batch_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();
    let scalar_slices: Vec<&[Scalar]> = scalars.iter().map(|s| s.as_slice()).collect();

    let result = batch_vartime_multiscalar_mul(&scalar_slices, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_of_varying_sized_scalars_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_lens: Vec<usize> = (0..batch_len).map(|i| i * 100 / (batch_len - 1)).collect();

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|i| {
        (0..sample_lens[i])
          .map(|_| Scalar::random(&mut rng))
          .collect()
      })
      .collect();

    let bases: Vec<Affine> = (0..sample_lens[batch_len - 1])
      .map(|_| Affine::random(&mut rng))
      .collect();
    let scalar_slices: Vec<&[Scalar]> = scalars.iter().map(|s| s.as_slice()).collect();

    let result = batch_vartime_multiscalar_mul(&scalar_slices, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases[..scalar.len()]))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }
}
