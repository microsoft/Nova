//! This module provides a multi-scalar multiplication routine.
//!
//! For full-field-element MSM, we use a signed-decomposition + bit-width-partitioning strategy
//!
//! 1. **Signed scalar decomposition**: For each scalar `s`, we compare `num_bits(s)` vs
//!    `num_bits(p - s)` and use whichever representation is smaller, negating the base point
//!    if we use `p - s`. This halves the effective scalar range.
//! 2. **Bit-width partitioning**: Scalars are routed to the optimal algorithm based on their
//!    actual bit-width after signed reduction: binary accumulation for 0/1, single-window
//!    bucket sort for ≤10 bits, multi-window Pippenger for ≤64 bits, and wNAF for the rest.
//! 3. **wNAF (windowed non-adjacent form)**: Large scalars use signed digit decomposition
//!    with XYZZ bucket coordinates for ~18% faster accumulation.
//! 4. **XYZZ bucket coordinates**: Extended Jacobian `(X, Y, ZZ, ZZZ)` provides cheaper
//!    mixed addition (7M + 2S) compared to standard Jacobian (~11M + 5S for proj+affine).
//!
//! The MSM implementations (for integer types and field types) are adapted from halo2/jolt.
use ff::{Field, PrimeField};
use halo2curves::{group::Group, CurveAffine};
use num_integer::Integer;
use num_traits::{ToPrimitive, Zero};
use rayon::{current_num_threads, prelude::*};

// ==================================================================================
// XYZZ (Extended Jacobian) Bucket coordinates
// ==================================================================================

/// Extended Jacobian (XYZZ) coordinates for efficient MSM bucket accumulation.
///
/// Stores `(X, Y, ZZ, ZZZ)` where `ZZ = Z²` and `ZZZ = Z³` for a Jacobian point
/// with coordinates `(X/ZZ, Y/ZZZ)` in affine.
///
/// Mixed addition (affine + XYZZ) costs 7M + 2S vs ~11M + 5S for standard projective+affine.
/// Formula source: <https://www.hyperelliptic.org/EFD/g1p/auto-shortw-xyzz.html>
///
/// **Assumes `a = 0`** in the curve equation `y² = x³ + ax + b`, which holds for
/// all curves used in Nova (BN254, Grumpkin, Pallas, Vesta, secp256k1, secq256k1).
#[derive(Copy, Clone)]
struct BucketXYZZ<F: Field> {
  x: F,
  y: F,
  zz: F,
  zzz: F,
}

impl<F: Field> BucketXYZZ<F> {
  /// The point at infinity (identity).
  #[inline]
  fn zero() -> Self {
    Self {
      x: F::ONE,
      y: F::ONE,
      zz: F::ZERO,
      zzz: F::ZERO,
    }
  }

  /// Check if this is the identity.
  #[inline]
  fn is_zero(&self) -> bool {
    self.zz == F::ZERO
  }

  /// Double in place (dbl-2008-s-1, assumes a=0).
  /// Cost: 2M + 5S + 7add
  fn double_in_place(&mut self) {
    if self.is_zero() {
      return;
    }
    // U = 2*Y1
    let u = self.y.double();
    // V = U^2
    let v = u.square();
    // W = U*V
    let w = u * v;
    // S = X1*V
    let s = self.x * v;
    // M = 3*X1^2 (a=0, so no a*ZZ^2 term)
    let x_sq = self.x.square();
    let m = x_sq.double() + x_sq;
    // X3 = M^2 - 2*S
    self.x = m.square() - s.double();
    // Y3 = M*(S - X3) - W*Y1
    self.y = m * (s - self.x) - w * self.y;
    // ZZ3 = V*ZZ1
    self.zz *= v;
    // ZZZ3 = W*ZZZ1
    self.zzz *= w;
  }

  /// XYZZ += XYZZ (full addition, add-2008-s).
  fn add_assign_bucket(&mut self, other: &Self) {
    if other.is_zero() {
      return;
    }
    if self.is_zero() {
      *self = *other;
      return;
    }
    // U1 = X1*ZZ2, U2 = X2*ZZ1
    let u1 = self.x * other.zz;
    let u2 = other.x * self.zz;
    // S1 = Y1*ZZZ2, S2 = Y2*ZZZ1
    let s1 = self.y * other.zzz;
    let s2 = other.y * self.zzz;

    if u1 == u2 {
      if s1 == s2 {
        self.double_in_place();
      } else {
        *self = Self::zero();
      }
      return;
    }
    let p = u2 - u1;
    let r = s2 - s1;
    let pp = p.square();
    let ppp = p * pp;
    let q = u1 * pp;
    self.x = r.square() - ppp - q.double();
    self.y = r * (q - self.x) - s1 * ppp;
    self.zz = self.zz * other.zz * pp;
    self.zzz = self.zzz * other.zzz * ppp;
  }
}

/// Mixed addition: BucketXYZZ += CurveAffine point (madd-2008-s).
/// Cost: 7M + 2S
#[inline]
fn bucket_add_affine<C: CurveAffine>(bucket: &mut BucketXYZZ<C::Base>, p: &C) {
  if bool::from(p.is_identity()) {
    return;
  }
  let coords = p.coordinates().unwrap();
  let px = *coords.x();
  let py = *coords.y();

  if bucket.is_zero() {
    bucket.x = px;
    bucket.y = py;
    bucket.zz = C::Base::ONE;
    bucket.zzz = C::Base::ONE;
    return;
  }
  // U2 = X2*ZZ1, S2 = Y2*ZZZ1
  let u2 = px * bucket.zz;
  let s2 = py * bucket.zzz;

  if bucket.x == u2 {
    if bucket.y == s2 {
      bucket.double_in_place();
    } else {
      *bucket = BucketXYZZ::zero();
    }
    return;
  }
  let p_val = u2 - bucket.x;
  let r = s2 - bucket.y;
  let pp = p_val.square();
  let ppp = p_val * pp;
  let q = bucket.x * pp;
  bucket.x = r.square() - ppp - q.double();
  bucket.y = r * (q - bucket.x) - bucket.y * ppp;
  bucket.zz *= pp;
  bucket.zzz *= ppp;
}

/// Mixed subtraction: BucketXYZZ -= CurveAffine point.
#[inline]
fn bucket_sub_affine<C: CurveAffine>(bucket: &mut BucketXYZZ<C::Base>, p: &C) {
  if bool::from(p.is_identity()) {
    return;
  }
  let coords = p.coordinates().unwrap();
  let px = *coords.x();
  let py = -(*coords.y()); // negate y to subtract

  if bucket.is_zero() {
    bucket.x = px;
    bucket.y = py;
    bucket.zz = C::Base::ONE;
    bucket.zzz = C::Base::ONE;
    return;
  }
  let u2 = px * bucket.zz;
  let s2 = py * bucket.zzz;

  if bucket.x == u2 {
    if bucket.y == s2 {
      bucket.double_in_place();
    } else {
      *bucket = BucketXYZZ::zero();
    }
    return;
  }
  let p_val = u2 - bucket.x;
  let r = s2 - bucket.y;
  let pp = p_val.square();
  let ppp = p_val * pp;
  let q = bucket.x * pp;
  bucket.x = r.square() - ppp - q.double();
  bucket.y = r * (q - bucket.x) - bucket.y * ppp;
  bucket.zz *= pp;
  bucket.zzz *= ppp;
}

/// Convert XYZZ bucket to projective curve point.
///
/// Computes affine coordinates `(X/ZZ, Y/ZZZ)` then converts to projective.
/// Only called O(windows) times per thread, so the field inversion cost is negligible.
#[inline]
fn bucket_to_curve<C: CurveAffine>(bucket: &BucketXYZZ<C::Base>) -> C::CurveExt {
  if bucket.is_zero() {
    return C::CurveExt::identity();
  }
  let zz_inv = bucket.zz.invert().unwrap();
  let zzz_inv = bucket.zzz.invert().unwrap();
  let x = bucket.x * zz_inv;
  let y = bucket.y * zzz_inv;
  C::from_xy(x, y)
    .expect("XYZZ bucket should produce a valid curve point")
    .into()
}

// ==================================================================================
// Scalar utilities
// ==================================================================================

/// Count significant bits in a field element (from its little-endian repr).
#[inline]
fn scalar_num_bits<F: PrimeField>(s: &F) -> u32 {
  let repr = s.to_repr();
  let bytes = repr.as_ref();
  for i in (0..bytes.len()).rev() {
    if bytes[i] != 0 {
      return i as u32 * 8 + (8 - bytes[i].leading_zeros());
    }
  }
  0
}

/// Extract the low 64 bits from a field element's little-endian representation.
#[inline]
fn repr_low_u64<F: PrimeField>(s: &F) -> u64 {
  let repr = s.to_repr();
  let bytes = repr.as_ref();
  let mut buf = [0u8; 8];
  let len = bytes.len().min(8);
  buf[..len].copy_from_slice(&bytes[..len]);
  u64::from_le_bytes(buf)
}

/// Convert little-endian bytes to u64 limbs.
#[inline]
fn bytes_to_limbs(bytes: &[u8]) -> Vec<u64> {
  bytes
    .chunks(8)
    .map(|chunk| {
      let mut buf = [0u8; 8];
      buf[..chunk.len()].copy_from_slice(chunk);
      u64::from_le_bytes(buf)
    })
    .collect()
}

// ==================================================================================
// wNAF MSM (for large scalars, >64 bits)
// ==================================================================================

/// Extract wNAF (windowed non-adjacent form) signed digits from a scalar.
///
/// Given a scalar represented as u64 limbs, window size `w`, and `num_bits`,
/// produces signed digits in `[-(2^(w-1)), ..., 2^(w-1)]`.
fn make_wnaf_digits(limbs: &[u64], w: usize, num_bits: usize) -> Vec<i64> {
  let radix: u64 = 1 << w;
  let window_mask: u64 = radix - 1;
  let num_bits = if num_bits == 0 { 1 } else { num_bits };
  let digits_count = num_bits.div_ceil(w);
  let mut carry = 0u64;
  let mut digits = Vec::with_capacity(digits_count);

  for i in 0..digits_count {
    let bit_offset = i * w;
    let u64_idx = bit_offset / 64;
    let bit_idx = bit_offset % 64;

    let bit_buf = if u64_idx >= limbs.len() {
      0u64
    } else if bit_idx + w <= 64 || u64_idx == limbs.len() - 1 {
      limbs[u64_idx] >> bit_idx
    } else {
      (limbs[u64_idx] >> bit_idx) | (limbs[u64_idx + 1] << (64 - bit_idx))
    };

    let coef = carry + (bit_buf & window_mask);
    carry = (coef + radix / 2) >> w;
    let mut digit = (coef as i64) - (carry << w) as i64;

    if i == digits_count - 1 {
      digit += (carry << w) as i64;
    }
    digits.push(digit);
  }
  digits
}

/// wNAF MSM using XYZZ buckets. For large scalars (>64 bits).
/// Parallel over windows for a single chunk of bases/scalars.
fn msm_wnaf_serial<C: CurveAffine>(bases: &[C], scalar_reprs: &[Vec<u64>]) -> C::CurveExt {
  let size = bases.len().min(scalar_reprs.len());
  if size == 0 {
    return C::CurveExt::identity();
  }

  let c = if size < 32 { 3 } else { compute_ln(size) + 2 };
  let num_bits = C::ScalarExt::NUM_BITS as usize;
  let digits_count = num_bits.div_ceil(c);

  // Pre-compute all wNAF digits
  let scalar_digits: Vec<Vec<i64>> = scalar_reprs
    .iter()
    .map(|limbs| make_wnaf_digits(limbs, c, num_bits))
    .collect();

  // Process each window in parallel, using XYZZ buckets
  let window_sums: Vec<C::CurveExt> = (0..digits_count)
    .into_par_iter()
    .map(|i| {
      let mut buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); 1 << c];
      for (digits, base) in scalar_digits.iter().zip(bases.iter()) {
        let digit = digits[i];
        if digit > 0 {
          bucket_add_affine::<C>(&mut buckets[(digit - 1) as usize], base);
        } else if digit < 0 {
          bucket_sub_affine::<C>(&mut buckets[(-digit - 1) as usize], base);
        }
      }
      // Prefix sum
      let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
      let mut res: BucketXYZZ<C::Base> = BucketXYZZ::zero();
      for b in buckets.into_iter().rev() {
        running_sum.add_assign_bucket(&b);
        res.add_assign_bucket(&running_sum);
      }
      bucket_to_curve::<C>(&res)
    })
    .collect();

  // Combine window sums: lowest + Horner evaluation for higher windows
  let lowest = *window_sums.first().unwrap();
  lowest
    + window_sums[1..]
      .iter()
      .rev()
      .fold(C::CurveExt::identity(), |mut total, sum_i| {
        total += sum_i;
        for _ in 0..c {
          total = total.double();
        }
        total
      })
}

/// Parallel wNAF MSM: chunks the input for better parallelism.
fn msm_wnaf<C: CurveAffine>(bases: &[C], scalar_reprs: &[Vec<u64>]) -> C::CurveExt {
  let size = bases.len().min(scalar_reprs.len());
  if size == 0 {
    return C::CurveExt::identity();
  }

  // For parallelism, split across threads; each chunk runs msm_wnaf_serial
  // which internally parallelizes over windows
  let num_threads = current_num_threads();
  let chunk_size = size.div_ceil(num_threads);
  if chunk_size >= size {
    return msm_wnaf_serial(bases, scalar_reprs);
  }

  bases
    .par_chunks(chunk_size)
    .zip(scalar_reprs.par_chunks(chunk_size))
    .map(|(b, s)| msm_wnaf_serial(b, s))
    .reduce(C::CurveExt::identity, |sum, evl| sum + evl)
}

// ==================================================================================
// Main MSM with signed decomposition + bit-width partitioning
// ==================================================================================

/// Performs an optimized multi-scalar multiplication for full field element scalars.
///
/// This function uses signed scalar decomposition to halve the effective scalar range,
/// then partitions scalars by bit-width to route each group to the optimal MSM algorithm.
///
/// Adapted from the Jolt MSM implementation, with optimizations for Nova's use cases.
pub fn msm<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());
  let n = coeffs.len();
  if n == 0 {
    return C::Curve::identity();
  }

  // For very small inputs, use the simple fallback
  if n <= 16 {
    return msm_simple(coeffs, bases);
  }

  // Group indices: 0=unit_pos, 1=unit_neg, 2=pos≤8, 3=neg≤8,
  // 4=pos≤16, 5=neg≤16, 6=pos≤32, 7=neg≤32, 8=pos≤64, 9=neg≤64, 10=large
  const NUM_GROUPS: usize = 11;

  // Phase 1: Classify each scalar in parallel
  // Encode as u64: top 4 bits = group, bottom 60 bits = original index
  let classified: Vec<u64> = coeffs
    .par_iter()
    .enumerate()
    .filter_map(|(i, s)| {
      if bool::from(s.is_zero()) {
        return None;
      }
      let neg_s = -(*s);
      let bits_s = scalar_num_bits(s);
      let bits_neg = scalar_num_bits(&neg_s);

      let group = if bits_s <= 1 {
        0u8 // unit positive
      } else if bits_neg <= 1 {
        1u8 // unit negative
      } else if bits_s <= 8 {
        2u8
      } else if bits_neg <= 8 {
        3u8
      } else if bits_s <= 16 {
        4u8
      } else if bits_neg <= 16 {
        5u8
      } else if bits_s <= 32 {
        6u8
      } else if bits_neg <= 32 {
        7u8
      } else if bits_s <= 64 {
        8u8
      } else if bits_neg <= 64 {
        9u8
      } else {
        10u8 // large
      };
      Some(((i as u64) & 0x0FFF_FFFF_FFFF_FFFF) | ((group as u64) << 60))
    })
    .collect();

  if classified.is_empty() {
    return C::Curve::identity();
  }

  // Phase 2: Sort by group for efficient partitioning
  let mut classified = classified;
  classified.par_sort_unstable_by_key(|v| (v >> 60) as u8);

  let extract_group = |v: u64| (v >> 60) as u8;
  let extract_index = |v: u64| (v & 0x0FFF_FFFF_FFFF_FFFF) as usize;

  // Find partition boundaries
  let mut boundaries = [0usize; NUM_GROUPS + 1];
  {
    let mut pos = 0;
    for g in 0..NUM_GROUPS as u8 {
      boundaries[g as usize] = pos;
      pos += classified[pos..].partition_point(|v| extract_group(*v) <= g);
    }
    boundaries[NUM_GROUPS] = classified.len();
  }

  // Helper to extract (bases, u64_scalars) for a group range
  let extract_u64_group = |start: usize, end: usize, negate: bool| -> (Vec<C>, Vec<u64>) {
    classified[start..end]
      .iter()
      .map(|&v| {
        let idx = extract_index(v);
        let b = bases[idx];
        let s = if negate { -coeffs[idx] } else { coeffs[idx] };
        (b, repr_low_u64(&s))
      })
      .unzip()
  };

  // Helper to extract (bases, bool_scalars) for unit groups
  let extract_binary_group = |start: usize, end: usize| -> Vec<C> {
    classified[start..end]
      .iter()
      .map(|&v| bases[extract_index(v)])
      .collect()
  };

  // Phase 3: Compute MSM for each group in parallel
  // Binary groups (unit scalars): just accumulate bases
  let (g0_start, g0_end) = (boundaries[0], boundaries[1]);
  let (g1_start, g1_end) = (boundaries[1], boundaries[2]);

  // Small-scalar groups
  let (g2_start, g2_end) = (boundaries[2], boundaries[3]);
  let (g3_start, g3_end) = (boundaries[3], boundaries[4]);
  let (g4_start, g4_end) = (boundaries[4], boundaries[5]);
  let (g5_start, g5_end) = (boundaries[5], boundaries[6]);
  let (g6_start, g6_end) = (boundaries[6], boundaries[7]);
  let (g7_start, g7_end) = (boundaries[7], boundaries[8]);
  let (g8_start, g8_end) = (boundaries[8], boundaries[9]);
  let (g9_start, g9_end) = (boundaries[9], boundaries[10]);

  // Large scalar group
  let (g10_start, g10_end) = (boundaries[10], boundaries[11]);

  // Execute all groups in parallel using nested rayon joins
  let (binary_result, small_and_large_result) = rayon::join(
    || {
      // Binary: pos - neg
      let (pos, neg) = rayon::join(
        || {
          let bases_pos = extract_binary_group(g0_start, g0_end);
          accumulate_bases::<C>(&bases_pos)
        },
        || {
          let bases_neg = extract_binary_group(g1_start, g1_end);
          accumulate_bases::<C>(&bases_neg)
        },
      );
      pos - neg
    },
    || {
      let (small_result, large_result) = rayon::join(
        || {
          // Small scalar groups: compute each pair (positive - negative)
          let ((r8, r16), (r32, r64)) = rayon::join(
            || {
              rayon::join(
                || {
                  let (pos_b, pos_s) = extract_u64_group(g2_start, g2_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g3_start, g3_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 8)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 8)
                },
                || {
                  let (pos_b, pos_s) = extract_u64_group(g4_start, g4_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g5_start, g5_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 16)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 16)
                },
              )
            },
            || {
              rayon::join(
                || {
                  let (pos_b, pos_s) = extract_u64_group(g6_start, g6_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g7_start, g7_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 32)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 32)
                },
                || {
                  let (pos_b, pos_s) = extract_u64_group(g8_start, g8_end, false);
                  let (neg_b, neg_s) = extract_u64_group(g9_start, g9_end, true);
                  msm_small_with_max_num_bits(&pos_s, &pos_b, 64)
                    - msm_small_with_max_num_bits(&neg_s, &neg_b, 64)
                },
              )
            },
          );
          r8 + r16 + r32 + r64
        },
        || {
          // Large scalars: wNAF
          if g10_start >= g10_end {
            return C::Curve::identity();
          }
          let (large_bases, large_reprs): (Vec<C>, Vec<Vec<u64>>) = classified[g10_start..g10_end]
            .iter()
            .map(|&v| {
              let idx = extract_index(v);
              let b = bases[idx];
              let s = coeffs[idx];
              let repr = bytes_to_limbs(s.to_repr().as_ref());
              (b, repr)
            })
            .unzip();
          msm_wnaf::<C>(&large_bases, &large_reprs)
        },
      );
      small_result + large_result
    },
  );

  binary_result + small_and_large_result
}

/// Simple MSM fallback for very small inputs.
fn msm_simple<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  coeffs
    .iter()
    .zip(bases.iter())
    .fold(C::Curve::identity(), |acc, (coeff, base)| {
      acc + *base * coeff
    })
}

/// Accumulate bases (sum of affine points, for binary MSM).
fn accumulate_bases<C: CurveAffine>(bases: &[C]) -> C::Curve {
  let num_threads = current_num_threads();
  if bases.is_empty() {
    return C::Curve::identity();
  }
  if bases.len() > num_threads {
    let chunk = bases.len().div_ceil(num_threads);
    bases
      .par_chunks(chunk)
      .map(|chunk| {
        chunk.iter().fold(C::Curve::identity(), |mut acc, b| {
          acc += *b;
          acc
        })
      })
      .reduce(C::Curve::identity, |a, b| a + b)
  } else {
    bases.iter().fold(C::Curve::identity(), |mut acc, b| {
      acc += *b;
      acc
    })
  }
}

fn num_bits(n: usize) -> usize {
  if n == 0 {
    0
  } else {
    (n.ilog2() + 1) as usize
  }
}

// ==================================================================================
// Small-scalar bucket type (for integer MSMs)
// ==================================================================================

#[derive(Clone, Copy)]
enum SmallBucket<C: CurveAffine> {
  None,
  Affine(C),
  Projective(C::Curve),
}

impl<C: CurveAffine> SmallBucket<C> {
  fn add_assign(&mut self, other: &C) {
    *self = match *self {
      SmallBucket::None => SmallBucket::Affine(*other),
      SmallBucket::Affine(a) => SmallBucket::Projective(a + *other),
      SmallBucket::Projective(a) => SmallBucket::Projective(a + other),
    }
  }

  fn add(self, other: C::Curve) -> C::Curve {
    match self {
      SmallBucket::None => other,
      SmallBucket::Affine(a) => other + a,
      SmallBucket::Projective(a) => other + a,
    }
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
    let mut buckets = vec![SmallBucket::None; num_buckets];

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

#[inline(always)]
pub(crate) fn batch_add<C: CurveAffine>(bases: &[C], one_indices: &[usize]) -> C::Curve {
  fn add_chunk<C: CurveAffine>(bases: impl Iterator<Item = C>) -> C::Curve {
    let mut acc = C::Curve::identity();
    for base in bases {
      acc += base;
    }
    acc
  }

  let num_chunks = rayon::current_num_threads();
  let chunk_size = (one_indices.len() + num_chunks).div_ceil(num_chunks);

  let comm = one_indices
    .par_chunks(chunk_size)
    .into_par_iter()
    .map(|chunk| add_chunk(chunk.iter().map(|index| bases[*index])))
    .reduce(C::Curve::identity, |sum, evl| sum + evl);

  comm
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
      let mask = if bit_width == 64 {
        u64::MAX
      } else {
        (1u64 << bit_width) - 1
      };
      let coeffs: Vec<u64> = (0..n)
        .map(|_| rand::random::<u64>() & mask)
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
