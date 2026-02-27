//! This module implements variable time multi-scalar multiplication using Blitzar's GPU acceleration.
//!
//! It uses Blitzar's `MsmHandle` (fixed-base MSM) to precompute and cache generator tables on the GPU.
//! This avoids redundant generator-to-GPU transfers and halo2↔arkworks conversions across multiple
//! MSM calls that share the same commitment key, which is the common case in ppsnark proving.
use blitzar::compute::{
  convert_to_ark_bn254_g1_affine, convert_to_halo2_bn256_g1_affine, ElementP2, MsmHandle,
  SwMsmHandle,
};
use halo2curves::bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

type Bn254MsmHandle = MsmHandle<ElementP2<ark_bn254::g1::Config>>;

/// Cached GPU handle: stores (generator_count, handle).
/// The handle is reused as long as the requested MSM length fits within the cached generator count.
/// If a longer MSM is requested, the handle is rebuilt with the new (larger) set of generators.
///
/// Uses `Mutex<Option<...>>` rather than `OnceLock` to avoid deadlocks when called from
/// within rayon parallel contexts (e.g., `rayon::join` or `par_iter` in ppsnark).
static GPU_HANDLE: Mutex<Option<(usize, Arc<Bn254MsmHandle>)>> = Mutex::new(None);

/// Returns a cached `MsmHandle` that covers at least `bases.len()` generators.
/// On first call (or when the bases grow), converts halo2 generators to arkworks and
/// installs them on the GPU. Subsequent calls reuse the cached handle.
fn get_or_create_handle(bases: &[Affine]) -> Arc<Bn254MsmHandle> {
  let n = bases.len();

  // Check if existing handle is big enough
  {
    let guard = GPU_HANDLE.lock().unwrap();
    if let Some((cached_n, ref handle)) = *guard {
      if cached_n >= n {
        return Arc::clone(handle);
      }
    }
  }
  // Mutex is released here before the heavy GPU work

  // Build handle outside any lock to avoid blocking rayon threads
  let handle = Arc::new(build_handle(bases));

  let mut guard = GPU_HANDLE.lock().unwrap();
  // Double-check: another thread may have built a sufficient handle while we were building
  if let Some((cached_n, ref existing)) = *guard {
    if cached_n >= n {
      return Arc::clone(existing);
    }
  }
  *guard = Some((n, Arc::clone(&handle)));
  handle
}

/// Converts halo2 generators to arkworks affine points and creates an `MsmHandle`.
/// Uses sequential iteration to avoid rayon nesting issues when called from parallel contexts.
fn build_handle(bases: &[Affine]) -> Bn254MsmHandle {
  let ark_generators: Vec<ark_bn254::G1Affine> = bases
    .iter()
    .map(|g| convert_to_ark_bn254_g1_affine(g))
    .collect();
  MsmHandle::new_with_affine(&ark_generators)
}

/// Performs a single multi-scalar multiplication using GPU-cached generators.
///
/// The generators (bases) are installed on the GPU once and reused across calls.
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  if scalars.is_empty() {
    return Point::default();
  }

  let handle = get_or_create_handle(bases);

  // Lay out scalars as contiguous bytes: n scalars × 32 bytes each
  let n = scalars.len();
  let mut scalar_bytes = vec![0u8; n * 32];
  scalar_bytes
    .par_chunks_mut(32)
    .zip(scalars.par_iter())
    .for_each(|(chunk, s)| {
      chunk.copy_from_slice(&s.to_bytes());
    });

  let mut results = vec![ark_bn254::G1Affine::default(); 1];
  handle.affine_msm(&mut results, 32, &scalar_bytes);

  convert_to_halo2_bn256_g1_affine(&results[0]).into()
}

/// Performs a batch of multi-scalar multiplications using GPU-cached generators.
///
/// All MSMs in the batch share the same generator set. The generators are installed
/// on the GPU once and reused across calls.
pub fn batch_vartime_multiscalar_mul(scalars: &[Vec<Scalar>], bases: &[Affine]) -> Vec<Point> {
  if scalars.is_empty() {
    return vec![];
  }

  let handle = get_or_create_handle(bases);
  let num_outputs = scalars.len();

  // Find the max scalar vector length (they may differ across outputs)
  let max_len = scalars.iter().map(|s| s.len()).max().unwrap_or(0);

  if max_len == 0 {
    return vec![Point::default(); num_outputs];
  }

  // Check if all scalar vectors have the same length (uniform case)
  let all_same_len = scalars.iter().all(|s| s.len() == max_len);

  if all_same_len {
    // Uniform length: use regular msm with interleaved scalars
    // Layout: for each generator i, all m scalars for that generator are contiguous
    // s_11, s_21, ..., s_m1, s_12, s_22, ..., s_m2, ..., s_mn
    let mut scalar_bytes = vec![0u8; num_outputs * max_len * 32];
    scalar_bytes
      .par_chunks_mut(num_outputs * 32)
      .enumerate()
      .for_each(|(gen_idx, gen_chunk)| {
        for (out_idx, scalar_vec) in scalars.iter().enumerate() {
          let src = scalar_vec[gen_idx].to_bytes();
          let dst_offset = out_idx * 32;
          gen_chunk[dst_offset..dst_offset + 32].copy_from_slice(&src);
        }
      });

    let mut results = vec![ark_bn254::G1Affine::default(); num_outputs];
    handle.affine_msm(&mut results, 32, &scalar_bytes);

    results
      .par_iter()
      .map(|r| convert_to_halo2_bn256_g1_affine(r).into())
      .collect()
  } else {
    // Variable length: use vlen_msm
    // First, sort by length (vlen_msm requires ascending order)
    let mut indexed: Vec<(usize, &Vec<Scalar>)> = scalars.iter().enumerate().collect();
    indexed.sort_by_key(|(_, s)| s.len());

    let output_lengths: Vec<u32> = indexed.iter().map(|(_, s)| s.len() as u32).collect();
    let output_bit_table: Vec<u32> = vec![256; num_outputs];

    // Layout: interleaved column-major, padded to max_len
    let mut scalar_bytes = vec![0u8; num_outputs * max_len * 32];
    scalar_bytes
      .par_chunks_mut(num_outputs * 32)
      .enumerate()
      .for_each(|(gen_idx, gen_chunk)| {
        for (sorted_idx, (_, scalar_vec)) in indexed.iter().enumerate() {
          if gen_idx < scalar_vec.len() {
            let src = scalar_vec[gen_idx].to_bytes();
            let dst_offset = sorted_idx * 32;
            gen_chunk[dst_offset..dst_offset + 32].copy_from_slice(&src);
          }
        }
      });

    let mut results = vec![ark_bn254::G1Affine::default(); num_outputs];
    handle.affine_vlen_msm(&mut results, &output_bit_table, &output_lengths, &scalar_bytes);

    // Unsort the results back to original order
    let mut final_results = vec![Point::default(); num_outputs];
    for (sorted_idx, (orig_idx, _)) in indexed.iter().enumerate() {
      final_results[*orig_idx] = convert_to_halo2_bn256_g1_affine(&results[sorted_idx]).into();
    }
    final_results
  }
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
    let scalars = vec![vec![]];
    let bases = vec![];

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

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

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

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

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

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

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

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

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases[..scalar.len()]))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }
}
