//! This module provides a multi-scalar multiplication routine
//! The generic implementation is adapted from halo2; we add an optimization to commit to bits more efficiently
//! The specialized implementations are adapted from jolt, with additional optimizations and parallelization.
use ff::{Field, PrimeField};
use halo2curves::{group::Group, CurveAffine};
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};
use rayon::{current_num_threads, prelude::*};

#[derive(Clone, Copy)]
enum Bucket<C: CurveAffine> {
  None,
  Affine(C),
  Projective(C::Curve),
}

impl<C: CurveAffine> Bucket<C> {
  fn add_assign(&mut self, other: &C) {
    *self = match *self {
      Bucket::None => Bucket::Affine(*other),
      Bucket::Affine(a) => Bucket::Projective(a + *other),
      Bucket::Projective(a) => Bucket::Projective(a + other),
    }
  }

  fn add(self, other: C::Curve) -> C::Curve {
    match self {
      Bucket::None => other,
      Bucket::Affine(a) => other + a,
      Bucket::Projective(a) => other + a,
    }
  }
}

fn cpu_msm_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  let c = if bases.len() < 4 {
    1
  } else if bases.len() < 32 {
    3
  } else {
    (f64::from(bases.len() as u32)).ln().ceil() as usize
  };

  fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
    let skip_bits = segment * c;
    let skip_bytes = skip_bits / 8;

    if skip_bytes >= 32 {
      return 0;
    }

    let mut v = [0; 8];
    for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
      *v = *o;
    }

    let mut tmp = u64::from_le_bytes(v);
    tmp >>= skip_bits - (skip_bytes * 8);
    tmp %= 1 << c;

    tmp as usize
  }

  let boolean_sum = coeffs
    .iter()
    .zip(bases.iter())
    .filter(|(scalar, _)| *scalar == &C::Scalar::ONE)
    .fold(C::Curve::identity(), |mut acc, (_, base)| {
      acc += *base;
      acc
    });
  let non_boolean_sum = {
    let segments = (256 / c) + 1;
    (0..segments)
      .rev()
      .fold(C::Curve::identity(), |mut acc, segment| {
        (0..c).for_each(|_| acc = acc.double());

        let mut buckets = vec![Bucket::None; (1 << c) - 1];

        for (coeff, base) in coeffs.iter().zip(bases.iter()) {
          // skip Booleans
          if *coeff != C::Scalar::ZERO && *coeff != C::Scalar::ONE {
            let coeff = get_at::<C::Scalar>(segment, c, &coeff.to_repr());
            if coeff != 0 {
              buckets[coeff - 1].add_assign(base);
            }
          }
        }

        // Summation by parts
        // e.g. 3a + 2b + 1c = a +
        //                    (a) + b +
        //                    ((a) + b) + c
        let mut running_sum = C::Curve::identity();
        for exp in buckets.into_iter().rev() {
          running_sum = exp.add(running_sum);
          acc += &running_sum;
        }
        acc
      })
  };

  boolean_sum + non_boolean_sum
}

/// Performs a multi-scalar-multiplication operation without GPU acceleration.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
/// Adapted from zcash/halo2
pub fn msm<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());

  let num_threads = current_num_threads();
  if coeffs.len() > num_threads {
    let chunk = coeffs.len() / num_threads;
    coeffs
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(coeffs, bases)| cpu_msm_serial(coeffs, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    cpu_msm_serial(coeffs, bases)
  }
}

fn num_bits(n: usize) -> usize {
  if n == 0 {
    0
  } else {
    (n.ilog2() + 1) as usize
  }
}

/// Multi-scalar multiplication using the best algorithm for the given scalars.
pub fn msm_small<C: CurveAffine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
  scalars: &[T],
  bases: &[C],
) -> C::Curve {
  let max_num_bits = num_bits(scalars.iter().max().unwrap().to_usize().unwrap());
  msm_small_with_max_num_bits(scalars, bases, max_num_bits)
}

/// Multi-scalar multiplication using the best algorithm for the given scalars.
pub fn msm_small_with_max_num_bits<
  C: CurveAffine,
  T: Integer + Into<u64> + Copy + Sync + ToPrimitive,
>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  assert_eq!(bases.len(), scalars.len());

  match max_num_bits {
    0 => C::identity().into(),
    1 => msm_binary(scalars, bases),
    2..=10 => msm_10(scalars, bases, max_num_bits),
    _ => msm_small_rest(scalars, bases, max_num_bits),
  }
}

fn msm_binary<C: CurveAffine, T: Integer + Sync>(scalars: &[T], bases: &[C]) -> C::Curve {
  assert_eq!(scalars.len(), bases.len());
  let num_threads = current_num_threads();
  let process_chunk = |scalars: &[T], bases: &[C]| {
    let mut acc = C::Curve::identity();
    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _)| !scalar.is_zero())
      .for_each(|(_, base)| {
        acc += *base;
      });
    acc
  };

  if scalars.len() > num_threads {
    let chunk = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(scalars, bases)| process_chunk(scalars, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    process_chunk(scalars, bases)
  }
}

/// MSM optimized for up to 10-bit scalars
fn msm_10<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  fn msm_10_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let num_buckets: usize = 1 << max_num_bits;
    let mut buckets = vec![Bucket::None; num_buckets];

    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _base)| !scalar.is_zero())
      .for_each(|(scalar, base)| {
        let bucket_index: u64 = (*scalar).into();
        buckets[bucket_index as usize].add_assign(base);
      });

    let mut result = C::Curve::identity();
    let mut running_sum = C::Curve::identity();
    buckets.iter().skip(1).rev().for_each(|exp| {
      running_sum = exp.add(running_sum);
      result += &running_sum;
    });
    result
  }

  let num_threads = current_num_threads();
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| msm_10_serial(scalars_chunk, bases_chunk, max_num_bits))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_10_serial(scalars, bases, max_num_bits)
  }
}

fn msm_small_rest<C: CurveAffine, T: Into<u64> + Zero + Copy + Sync>(
  scalars: &[T],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  fn msm_small_rest_serial<C: CurveAffine, T: Into<u64> + Zero + Copy>(
    scalars: &[T],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let mut c = if bases.len() < 32 {
      3
    } else {
      compute_ln(bases.len()) + 2
    };

    if max_num_bits == 32 || max_num_bits == 64 {
      c = 8;
    }

    let zero = C::Curve::identity();

    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _base)| !s.is_zero());
    let window_starts = (0..max_num_bits).step_by(c);

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // in parallel process each such window.
    let window_sums: Vec<_> = window_starts
      .map(|w_start| {
        let mut res = zero;
        // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
        let mut buckets = vec![zero; (1 << c) - 1];
        // This clone is cheap, because the iterator contains just a
        // pointer and an index into the original vectors.
        scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
          let scalar: u64 = scalar.into();
          if scalar == 1 {
            // We only process unit scalars once in the first window.
            if w_start == 0 {
              res += base;
            }
          } else {
            let mut scalar = scalar;

            // We right-shift by w_start, thus getting rid of the
            // lower bits.
            scalar >>= w_start;

            // We mod the remaining bits by 2^{window size}, thus taking `c` bits.
            scalar %= 1 << c;

            // If the scalar is non-zero, we update the corresponding
            // bucket.
            // (Recall that `buckets` doesn't have a zero bucket.)
            if scalar != 0 {
              buckets[(scalar - 1) as usize] += base;
            }
          }
        });

        // Compute sum_{i in 0..num_buckets} (sum_{j in i..num_buckets} bucket[j])
        // This is computed below for b buckets, using 2b curve additions.
        //
        // We could first normalize `buckets` and then use mixed-addition
        // here, but that's slower for the kinds of groups we care about
        // (Short Weierstrass curves and Twisted Edwards curves).
        // In the case of Short Weierstrass curves,
        // mixed addition saves ~4 field multiplications per addition.
        // However normalization (with the inversion batched) takes ~6
        // field multiplications per element,
        // hence batch normalization is a slowdown.

        // `running_sum` = sum_{j in i..num_buckets} bucket[j],
        // where we iterate backward from i = num_buckets to 0.
        let mut running_sum = C::Curve::identity();
        buckets.into_iter().rev().for_each(|b| {
          running_sum += &b;
          res += &running_sum;
        });
        res
      })
      .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
    lowest
      + window_sums[1..]
        .iter()
        .rev()
        .fold(zero, |mut total, sum_i| {
          total += sum_i;
          for _ in 0..c {
            total = total.double();
          }
          total
        })
  }

  let num_threads = current_num_threads();
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(scalars_chunk, bases_chunk)| {
        msm_small_rest_serial(scalars_chunk, bases_chunk, max_num_bits)
      })
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    msm_small_rest_serial(scalars, bases, max_num_bits)
  }
}

fn compute_ln(a: usize) -> usize {
  // log2(a) * ln(2)
  if a == 0 {
    0 // Handle edge case where log2 is undefined
  } else {
    a.ilog2() as usize * 69 / 100
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{
    bn256_grumpkin::{bn256, grumpkin},
    pasta::{pallas, vesta},
    secp_secq::{secp256k1, secq256k1},
  };
  use ff::Field;
  use halo2curves::{group::Group, CurveAffine};
  use rand_core::OsRng;

  fn test_general_msm_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    assert_eq!(coeffs.len(), bases.len());
    let naive = coeffs
      .iter()
      .zip(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let msm = msm(&coeffs, &bases);

    assert_eq!(naive, msm)
  }

  #[test]
  fn test_general_msm() {
    test_general_msm_with::<pallas::Scalar, pallas::Affine>();
    test_general_msm_with::<vesta::Scalar, vesta::Affine>();
    test_general_msm_with::<bn256::Scalar, bn256::Affine>();
    test_general_msm_with::<grumpkin::Scalar, grumpkin::Affine>();
    test_general_msm_with::<secp256k1::Scalar, secp256k1::Affine>();
    test_general_msm_with::<secq256k1::Scalar, secq256k1::Affine>();
  }

  fn test_msm_ux_with<F: PrimeField, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    for bit_width in [1, 4, 8, 10, 16, 20, 32, 40, 64] {
      println!("bit_width: {bit_width}");
      assert!(bit_width <= 64); // Ensure we don't overflow F::from
      let coeffs: Vec<u64> = (0..n)
        .map(|_| rand::random::<u64>() % (1 << bit_width))
        .collect::<Vec<_>>();
      let coeffs_scalar: Vec<F> = coeffs.iter().map(|b| F::from(*b)).collect::<Vec<_>>();
      let general = msm(&coeffs_scalar, &bases);
      let integer = msm_small(&coeffs, &bases);

      assert_eq!(general, integer);
    }
  }

  #[test]
  fn test_msm_ux() {
    test_msm_ux_with::<pallas::Scalar, pallas::Affine>();
    test_msm_ux_with::<vesta::Scalar, vesta::Affine>();
    test_msm_ux_with::<bn256::Scalar, bn256::Affine>();
    test_msm_ux_with::<grumpkin::Scalar, grumpkin::Affine>();
    test_msm_ux_with::<secp256k1::Scalar, secp256k1::Affine>();
    test_msm_ux_with::<secq256k1::Scalar, secq256k1::Affine>();
  }
}
