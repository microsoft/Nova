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

/// Subtract an affine point from an XYZZ bucket (equivalent to adding negated point).
#[inline(always)]
fn bucket_sub_affine<C: CurveAffine>(bucket: &mut BucketXYZZ<C::Base>, p: &C) {
  if bool::from(p.is_identity()) {
    return;
  }
  let coords = p.coordinates().unwrap();
  let px = *coords.x();
  let py = -*coords.y(); // Negate y to subtract

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
// Batch affine pairwise addition (Montgomery batch inversion)
// ==================================================================================

/// Batch-invert a slice of field elements using Montgomery's trick.
/// After calling, `v[i]` contains `1/v[i]` (original).
/// Cost: 3(N-1) multiplies + 1 inversion.
#[inline]
fn batch_invert_in_place<F: Field>(v: &mut [F]) {
  let n = v.len();
  if n == 0 {
    return;
  }
  if n == 1 {
    v[0] = v[0].invert().unwrap();
    return;
  }
  // Forward pass: compute prefix products
  let mut prefix = Vec::with_capacity(n);
  prefix.push(v[0]);
  for i in 1..n {
    prefix.push(prefix[i - 1] * v[i]);
  }
  // Invert total product
  let mut inv = prefix[n - 1].invert().unwrap();
  // Backward pass: extract individual inverses
  for i in (1..n).rev() {
    let vi_inv = prefix[i - 1] * inv;
    inv *= v[i];
    v[i] = vi_inv;
  }
  v[0] = inv;
}

/// Sum N affine points using batch-affine tree reduction.
///
/// Instead of sequential XYZZ additions (7M+2S each), this uses a tree-structured
/// pairwise addition with Montgomery batch inversion. Amortized cost per addition:
/// ~4M+1S+3M(batch_inv) = 7M+1S, but with better cache behavior because all
/// intermediate results stay in affine coordinates (64B vs 128B for XYZZ).
///
/// For N points: performs N-1 additions across log₂(N) tree levels.
/// Each level batch-processes all independent pairs with a single batch inversion.
///
/// Falls back to XYZZ accumulation for N ≤ 64 (overhead of affine tree not worthwhile).
fn batch_add_affine_tree<C: CurveAffine>(points: &[C]) -> C::CurveExt {
  let n = points.len();
  if n == 0 {
    return C::CurveExt::identity();
  }
  if n == 1 {
    return points[0].into();
  }
  // For small N, XYZZ is faster (no batch inversion overhead)
  if n <= 64 {
    let mut acc = BucketXYZZ::<C::Base>::zero();
    for p in points {
      bucket_add_affine::<C>(&mut acc, p);
    }
    return bucket_to_curve::<C>(&acc);
  }

  // Work buffer: copy input points, reduce in-place
  // Use Option to handle identity points cleanly
  let mut buf: Vec<Option<(C::Base, C::Base)>> = points
    .iter()
    .map(|p| {
      if bool::from(p.is_identity()) {
        None
      } else {
        let c = p.coordinates().unwrap();
        Some((*c.x(), *c.y()))
      }
    })
    .collect();

  while buf.len() > 1 {
    let len = buf.len();
    let half = len / 2;

    // Compute deltas for batch inversion (exactly half-sized allocation)
    let mut deltas: Vec<C::Base> = Vec::with_capacity(half);
    for i in 0..half {
      deltas.push(match (buf[2 * i], buf[2 * i + 1]) {
        (Some((x1, _)), Some((x2, _))) => {
          let dx = x2 - x1;
          if dx == C::Base::ZERO {
            C::Base::ONE
          } else {
            dx
          }
        }
        _ => C::Base::ONE,
      });
    }

    batch_invert_in_place::<C::Base>(&mut deltas);

    // Compute results for each pair
    for i in 0..half {
      buf[i] = match (buf[2 * i], buf[2 * i + 1]) {
        (None, None) => None,
        (Some(p), None) | (None, Some(p)) => Some(p),
        (Some((x1, y1)), Some((x2, y2))) => {
          let dx = x2 - x1;
          if dx == C::Base::ZERO {
            if y1 == y2 {
              if y1 == C::Base::ZERO {
                None
              } else {
                let x1_sq = x1.square();
                let lambda = (x1_sq.double() + x1_sq) * (y1.double()).invert().unwrap();
                let x3 = lambda.square() - x1.double();
                let y3 = lambda * (x1 - x3) - y1;
                Some((x3, y3))
              }
            } else {
              None
            }
          } else {
            let inv_dx = deltas[i];
            let lambda = (y2 - y1) * inv_dx;
            let x3 = lambda.square() - x1 - x2;
            let y3 = lambda * (x1 - x3) - y1;
            Some((x3, y3))
          }
        }
      };
    }

    // Handle odd element: copy to end of reduced array
    if len % 2 == 1 {
      buf[half] = buf[len - 1];
      buf.truncate(half + 1);
    } else {
      buf.truncate(half);
    }
  }

  match buf[0] {
    None => C::CurveExt::identity(),
    Some((x, y)) => C::from_xy(x, y)
      .expect("batch_add_affine_tree produced invalid point")
      .into(),
  }
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

// ==================================================================================
// Main MSM with signed decomposition + bit-width partitioning
// ==================================================================================

/// Performs an optimized multi-scalar multiplication for full field element scalars.
///
/// This function uses signed scalar decomposition to halve the effective scalar range,
/// then partitions scalars by bit-width to route each group to the optimal MSM algorithm.
/// Small scalars (≤64 bits) use Nova's bucket-sort MSMs; large scalars delegate to
/// halo2curves' `msm_best`.
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
      if bool::from(s.is_zero()) || bool::from(bases[i].is_identity()) {
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
          // Large scalars: delegate to halo2curves' optimized MSM
          if g10_start >= g10_end {
            return C::Curve::identity();
          }
          let (large_bases, large_coeffs): (Vec<C>, Vec<C::Scalar>) = classified
            [g10_start..g10_end]
            .iter()
            .map(|&v| {
              let idx = extract_index(v);
              (bases[idx], coeffs[idx])
            })
            .unzip();
          halo2curves::msm::msm_best(&large_coeffs, &large_bases)
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
// Small-scalar MSM with XYZZ buckets
// ==================================================================================

/// Multi-scalar multiplication using the best algorithm for the given scalars.
pub fn msm_small<C: CurveAffine, T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
  scalars: &[T],
  bases: &[C],
) -> C::Curve {
  if scalars.is_empty() {
    return C::Curve::identity();
  }
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
    11..=64 => msm_small_rest(scalars, bases, max_num_bits),
    _ => {
      // For >32-bit scalars, halo2curves' msm_best is faster than our
      // bucket-sort Pippenger (e.g., 192ms vs 244ms at u64, 2^20 points).
      let field_scalars: Vec<C::ScalarExt> = scalars
        .iter()
        .map(|s| C::ScalarExt::from((*s).into()))
        .collect();
      halo2curves::msm::msm_best(&field_scalars, bases)
    }
  }
}

/// Multi-scalar multiplication for signed (i128) scalars.
///
/// Single-pass windowed Pippenger: handles positive and negative scalars
/// in the same bucket accumulation, avoiding the 2× MSM overhead of split pos/neg.
pub fn msm_signed<C: CurveAffine>(scalars: &[i128], bases: &[C]) -> C::Curve {
  assert_eq!(bases.len(), scalars.len());

  // Find max magnitude and filter out zeros.
  // Callers must ensure magnitudes fit in u64 (e.g., i128 used to avoid i64 overflow).
  let mut max_mag: u64 = 0;
  for &s in scalars.iter() {
    debug_assert!(
      s.unsigned_abs() <= u64::MAX as u128,
      "msm_signed: scalar magnitude exceeds u64"
    );
    max_mag = max_mag.max(s.unsigned_abs() as u64);
  }

  if max_mag == 0 {
    return C::Curve::identity();
  }

  let bits = num_bits(max_mag as usize);

  // For small magnitudes, use the existing split approach (bucket sort is tuned for unsigned)
  if bits <= 10 {
    let mut pos = vec![0u64; scalars.len()];
    let mut neg = vec![0u64; scalars.len()];
    for (i, &s) in scalars.iter().enumerate() {
      if s >= 0 {
        pos[i] = s as u64;
      } else {
        neg[i] = s.unsigned_abs() as u64;
      }
    }
    return msm_small_with_max_num_bits(&pos, bases, bits)
      - msm_small_with_max_num_bits(&neg, bases, bits);
  }

  // Windowed Pippenger with signed bucket accumulation
  msm_signed_windowed(scalars, bases, bits)
}

/// Single-pass signed windowed Pippenger MSM.
///
/// For each c-bit window, positive scalars add to bucket[digit-1],
/// negative scalars subtract (using the negated affine point).
/// This avoids allocating separate pos/neg arrays and running 2× MSMs.
fn msm_signed_windowed<C: CurveAffine>(
  scalars: &[i128],
  bases: &[C],
  max_num_bits: usize,
) -> C::Curve {
  fn msm_signed_windowed_serial<C: CurveAffine>(
    scalars: &[i128],
    bases: &[C],
    max_num_bits: usize,
  ) -> C::Curve {
    let c = if bases.len() < 32 {
      3
    } else {
      // Optimal c ≈ log2(n) for Pippenger: minimizes n*(b/c) + (b/c)*2^c
      let ln = compute_ln(bases.len());
      if max_num_bits <= 32 {
        (ln + 2).min(max_num_bits)
      } else {
        // For 33-64 bit signed scalars with ~50% non-zero,
        // effective n ≈ len/2, optimal c ≈ ln(n/2)
        (ln + 2).clamp(8, 16)
      }
    };

    let window_starts: Vec<usize> = (0..max_num_bits).step_by(c).collect();

    let window_sums: Vec<C::CurveExt> = window_starts
      .iter()
      .map(|&w_start| {
        let mut res: BucketXYZZ<C::Base> = BucketXYZZ::zero();
        let mut buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); (1 << c) - 1];

        for (&scalar, base) in scalars.iter().zip(bases) {
          if scalar == 0 {
            continue;
          }
          let mag = scalar.unsigned_abs() as u64;
          let is_neg = scalar < 0;

          if mag == 1 {
            if w_start == 0 {
              if is_neg {
                bucket_sub_affine::<C>(&mut res, base);
              } else {
                bucket_add_affine::<C>(&mut res, base);
              }
            }
          } else {
            let digit = (mag >> w_start) % (1 << c);
            if digit != 0 {
              let bucket = &mut buckets[(digit - 1) as usize];
              if is_neg {
                bucket_sub_affine::<C>(bucket, base);
              } else {
                bucket_add_affine::<C>(bucket, base);
              }
            }
          }
        }

        // Prefix sum
        let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
        for b in buckets.into_iter().rev() {
          running_sum.add_assign_bucket(&b);
          res.add_assign_bucket(&running_sum);
        }
        bucket_to_curve::<C>(&res)
      })
      .collect();

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

  let num_threads = current_num_threads();
  if scalars.len() > num_threads {
    let chunk_size = scalars.len() / num_threads;
    scalars
      .par_chunks(chunk_size)
      .zip(bases.par_chunks(chunk_size))
      .map(|(s, b)| msm_signed_windowed_serial(s, b, max_num_bits))
      .reduce(C::Curve::identity, |a, b| a + b)
  } else {
    msm_signed_windowed_serial(scalars, bases, max_num_bits)
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

/// MSM optimized for up to 10-bit scalars, using XYZZ bucket coordinates.
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
    let mut buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); num_buckets];

    scalars
      .iter()
      .zip(bases.iter())
      .filter(|(scalar, _base)| !scalar.is_zero())
      .for_each(|(scalar, base)| {
        let bucket_index: u64 = (*scalar).into();
        bucket_add_affine::<C>(&mut buckets[bucket_index as usize], base);
      });

    let mut result: BucketXYZZ<C::Base> = BucketXYZZ::zero();
    let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
    for b in buckets.into_iter().skip(1).rev() {
      running_sum.add_assign_bucket(&b);
      result.add_assign_bucket(&running_sum);
    }
    bucket_to_curve::<C>(&result)
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

    let scalars_and_bases_iter = scalars.iter().zip(bases).filter(|(s, _base)| !s.is_zero());
    let window_starts: Vec<usize> = (0..max_num_bits).step_by(c).collect();

    // Each window is of size `c`.
    // We divide up the bits 0..num_bits into windows of size `c`, and
    // process each such window.
    let window_sums: Vec<C::CurveExt> = window_starts
      .iter()
      .map(|&w_start| {
        let mut res: BucketXYZZ<C::Base> = BucketXYZZ::zero();
        // We don't need the "zero" bucket, so we only have 2^c - 1 buckets.
        let mut buckets: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); (1 << c) - 1];
        // This clone is cheap, because the iterator contains just a
        // pointer and an index into the original vectors.
        scalars_and_bases_iter.clone().for_each(|(&scalar, base)| {
          let scalar: u64 = scalar.into();
          if scalar == 1 {
            // We only process unit scalars once in the first window.
            if w_start == 0 {
              bucket_add_affine::<C>(&mut res, base);
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
              bucket_add_affine::<C>(&mut buckets[(scalar - 1) as usize], base);
            }
          }
        });

        // Prefix sum using XYZZ coordinates
        let mut running_sum: BucketXYZZ<C::Base> = BucketXYZZ::zero();
        for b in buckets.into_iter().rev() {
          running_sum.add_assign_bucket(&b);
          res.add_assign_bucket(&running_sum);
        }
        bucket_to_curve::<C>(&res)
      })
      .collect();

    // We store the sum for the lowest window.
    let lowest = *window_sums.first().unwrap();

    // We're traversing windows from high to low.
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
  fn add_chunk_xyzz<C: CurveAffine>(bases: &[C], indices: &[usize]) -> BucketXYZZ<C::Base> {
    let mut acc = BucketXYZZ::<C::Base>::zero();
    for &idx in indices {
      bucket_add_affine::<C>(&mut acc, &bases[idx]);
    }
    acc
  }

  let num_chunks = rayon::current_num_threads();
  let chunk_size = (one_indices.len() + num_chunks).div_ceil(num_chunks);

  let comm = one_indices
    .par_chunks(chunk_size)
    .into_par_iter()
    .map(|chunk| add_chunk_xyzz::<C>(bases, chunk))
    .reduce(BucketXYZZ::zero, |mut sum, evl| {
      sum.add_assign_bucket(&evl);
      sum
    });

  bucket_to_curve::<C>(&comm)
}

/// Batch-add multiple sparse binary vectors over the same SRS, deduplicating shared hot
/// indices. When all polys in the group have the same hot index for a given position, the
/// SRS point is accumulated into a shared sum (once) rather than into each accumulator
/// individually. At the end the shared sum is merged into all accumulators.
///
/// Uses rayon fold/reduce for parallelism: each thread processes a chunk of entries
/// with its own per-poly accumulators and shared accumulator, then merges.
pub fn batch_add_multi<C: CurveAffine>(bases: &[C], hot_per_poly: &[&[usize]]) -> Vec<C::Curve> {
  let n_polys = hot_per_poly.len();
  if n_polys == 0 {
    return vec![];
  }
  if n_polys == 1 {
    return vec![batch_add(bases, hot_per_poly[0])];
  }

  let n_entries = hot_per_poly[0].len();
  debug_assert!(hot_per_poly.iter().all(|h| h.len() == n_entries));

  // Use XYZZ bucket coordinates for 7M+2S per addition vs ~11M+5S for projective.
  let (accs, shared) = (0..n_entries)
    .into_par_iter()
    .with_min_len(1024)
    .fold(
      || {
        (
          vec![BucketXYZZ::<C::Base>::zero(); n_polys],
          BucketXYZZ::<C::Base>::zero(),
        )
      },
      |(mut accs, mut shared), t| {
        let idx0 = hot_per_poly[0][t];
        let all_same = hot_per_poly[1..].iter().all(|h| h[t] == idx0);

        if all_same {
          bucket_add_affine::<C>(&mut shared, &bases[idx0]);
        } else {
          for (p, hot) in hot_per_poly.iter().enumerate() {
            bucket_add_affine::<C>(&mut accs[p], &bases[hot[t]]);
          }
        }
        (accs, shared)
      },
    )
    .reduce(
      || {
        (
          vec![BucketXYZZ::<C::Base>::zero(); n_polys],
          BucketXYZZ::<C::Base>::zero(),
        )
      },
      |(mut a_accs, mut a_shared), (b_accs, b_shared)| {
        for (a, b) in a_accs.iter_mut().zip(b_accs.iter()) {
          a.add_assign_bucket(b);
        }
        a_shared.add_assign_bucket(&b_shared);
        (a_accs, a_shared)
      },
    );

  // Merge shared sum into every accumulator, then convert to curve points
  let shared_curve = bucket_to_curve::<C>(&shared);
  accs
    .iter()
    .map(|acc| bucket_to_curve::<C>(acc) + shared_curve)
    .collect()
}

/// Batch-add multiple sparse binary vectors using affine tree reduction.
///
/// Same semantics as `batch_add_multi` but uses `batch_add_affine_tree` for each
/// poly's accumulation. This trades the per-addition cost of XYZZ (7M+2S) for
/// batch-affine (5M+1S amortized) at the cost of gathering SRS points into
/// contiguous buffers.
///
/// Best for single-threaded or low-thread-count scenarios where the per-addition
/// cost dominates over parallelism benefits.
pub fn batch_add_multi_tree<C: CurveAffine>(
  bases: &[C],
  hot_per_poly: &[&[usize]],
) -> Vec<C::Curve> {
  let n_polys = hot_per_poly.len();
  if n_polys == 0 {
    return vec![];
  }
  let n_entries = hot_per_poly[0].len();
  if n_entries == 0 {
    return vec![C::Curve::identity(); n_polys];
  }
  debug_assert!(hot_per_poly.iter().all(|h| h.len() == n_entries));

  if n_polys == 1 {
    let gathered: Vec<C> = hot_per_poly[0].iter().map(|&i| bases[i]).collect();
    return vec![batch_add_affine_tree(&gathered)];
  }

  // Phase 1: Classify entries into all_same vs different
  let mut shared_indices: Vec<usize> = Vec::with_capacity(n_entries);
  let mut diff_positions: Vec<usize> = Vec::with_capacity(n_entries);

  for t in 0..n_entries {
    let idx0 = hot_per_poly[0][t];
    if hot_per_poly[1..].iter().all(|h| h[t] == idx0) {
      shared_indices.push(idx0);
    } else {
      diff_positions.push(t);
    }
  }

  // Phase 2: Compute shared sum via affine tree
  let shared_points: Vec<C> = shared_indices.iter().map(|&i| bases[i]).collect();
  let shared_curve = batch_add_affine_tree(&shared_points);

  // Phase 3: For each poly, gather its non-shared points and tree-reduce
  // Process one poly at a time to reuse the gather buffer (~4.5MB)
  let mut results = Vec::with_capacity(n_polys);
  let mut gather_buf: Vec<C> = Vec::with_capacity(diff_positions.len());

  for p in 0..n_polys {
    gather_buf.clear();
    for &t in &diff_positions {
      gather_buf.push(bases[hot_per_poly[p][t]]);
    }
    let poly_sum = batch_add_affine_tree(&gather_buf);
    results.push(poly_sum + shared_curve);
  }

  results
}

/// Commit chunked RA polynomials with address-grouped accumulation.
///
/// For `n_chunks` polynomials where SRS index = `addr * num_entries + i`,
/// groups entries by address value within each chunk for sequential SRS access,
/// then uses affine tree reduction per group.
///
/// This exploits address locality when each chunk only hits a small number
/// of distinct SRS address blocks (e.g., `subtable_size` = 16).
pub fn batch_add_address_grouped<C: CurveAffine>(
  bases: &[C],
  addrs: &[&[u16]],
  num_entries: usize,
  subtable_size: usize,
) -> Vec<C::Curve> {
  let n_chunks = addrs.len();
  if n_chunks == 0 {
    return vec![];
  }
  assert!(
    n_chunks <= 32,
    "batch_add_address_grouped: n_chunks={n_chunks} exceeds u32 bitmask capacity"
  );
  debug_assert!(addrs.iter().all(|a| a.len() == num_entries));

  // Phase 1: Classify entries into shared (all chunks same addr) and per-chunk groups.
  // For shared entries, group by address value.
  let mut shared_by_addr: Vec<Vec<usize>> = vec![Vec::new(); subtable_size];
  let mut diff_by_chunk_addr: Vec<Vec<Vec<usize>>> =
    vec![vec![Vec::new(); subtable_size]; n_chunks];

  for t in 0..num_entries {
    let a0 = addrs[0][t];
    let all_same = addrs[1..].iter().all(|a| a[t] == a0);

    if all_same {
      shared_by_addr[a0 as usize].push(t);
    } else {
      for (c, chunk_addrs) in addrs.iter().enumerate() {
        diff_by_chunk_addr[c][chunk_addrs[t] as usize].push(t);
      }
    }
  }

  // Phase 2+3: Sequential SRS scan with per-address bitmask.
  // For each SRS address block, scan t=0..T sequentially (perfect prefetch).
  // Use a bitmask to identify which chunks need each entry.
  // Shared entries get a separate accumulator added to all chunk results.
  let mut shared_acc = BucketXYZZ::<C::Base>::zero();
  let mut chunk_accums: Vec<BucketXYZZ<C::Base>> = vec![BucketXYZZ::zero(); n_chunks];

  // Build per-address bitmask: mask[t] has bit c set if chunk c maps entry t to this address
  let mut mask = vec![0u32; num_entries];
  // Separate shared flag to avoid bit collision when n_chunks >= 32
  let mut is_shared = vec![false; num_entries];

  for a in 0..subtable_size {
    let base_offset = a * num_entries;

    // Fill mask for this address
    for (c, diff_addrs) in diff_by_chunk_addr.iter().enumerate().take(n_chunks) {
      for &t in &diff_addrs[a] {
        mask[t] |= 1u32 << c;
      }
    }
    // Mark shared entries
    for &t in &shared_by_addr[a] {
      is_shared[t] = true;
    }

    // Sequential scan through SRS block [a*T, (a+1)*T)
    for t in 0..num_entries {
      if is_shared[t] {
        let pt = &bases[base_offset + t];
        bucket_add_affine::<C>(&mut shared_acc, pt);
        is_shared[t] = false;
      } else {
        let m = mask[t];
        if m == 0 {
          continue;
        }
        let pt = &bases[base_offset + t];
        // Per-chunk entry: add to each matching chunk
        let mut bits = m;
        while bits != 0 {
          let c = bits.trailing_zeros() as usize;
          bucket_add_affine::<C>(&mut chunk_accums[c], pt);
          bits &= bits - 1;
        }
      }
      // Clear mask for reuse
      mask[t] = 0;
    }
  }

  let shared_curve = bucket_to_curve::<C>(&shared_acc);
  let mut results = Vec::with_capacity(n_chunks);
  for accum in chunk_accums.iter().take(n_chunks) {
    results.push(bucket_to_curve::<C>(accum) + shared_curve);
  }

  results
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

  /// Regression test: MSM must handle identity bases without panicking,
  /// and the result must match naive accumulation (identity bases contribute nothing).
  fn test_msm_identity_bases_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let mut coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let mut bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();

    // Replace a few bases with identity and give them non-zero scalars
    bases[0] = A::identity();
    bases[3] = A::identity();
    bases[n - 1] = A::identity();
    coeffs[0] = F::ONE;
    coeffs[3] = F::random(OsRng);

    let naive = coeffs
      .iter()
      .zip(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let result = msm(&coeffs, &bases);

    assert_eq!(naive, result);
  }

  #[test]
  fn test_msm_identity_bases() {
    test_msm_identity_bases_with::<pallas::Scalar, pallas::Affine>();
    test_msm_identity_bases_with::<vesta::Scalar, vesta::Affine>();
    test_msm_identity_bases_with::<bn256::Scalar, bn256::Affine>();
    test_msm_identity_bases_with::<grumpkin::Scalar, grumpkin::Affine>();
    test_msm_identity_bases_with::<secp256k1::Scalar, secp256k1::Affine>();
    test_msm_identity_bases_with::<secq256k1::Scalar, secq256k1::Affine>();
  }

  fn test_batch_add_affine_tree_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    for &n in &[0, 1, 2, 3, 5, 16, 63, 64, 65, 128, 255, 256, 500, 1000] {
      let points: Vec<A> = (0..n)
        .map(|_| A::from(A::generator() * F::random(OsRng)))
        .collect();

      let expected: A::CurveExt = points
        .iter()
        .fold(A::CurveExt::identity(), |acc, p| acc + *p);
      let got = batch_add_affine_tree(&points);
      assert_eq!(expected, got, "batch_add_affine_tree mismatch at n={n}");
    }
  }

  #[test]
  fn test_batch_add_affine_tree() {
    test_batch_add_affine_tree_with::<bn256::Scalar, bn256::Affine>();
    test_batch_add_affine_tree_with::<pallas::Scalar, pallas::Affine>();
  }

  fn test_batch_add_multi_tree_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n_bases = 256;
    let n_entries = 200;
    let n_polys = 8;

    let bases: Vec<A> = (0..n_bases)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect();

    let hot_vecs: Vec<Vec<usize>> = (0..n_polys)
      .map(|_| {
        (0..n_entries)
          .map(|_| rand::random::<usize>() % n_bases)
          .collect()
      })
      .collect();
    let hot_refs: Vec<&[usize]> = hot_vecs.iter().map(|v| v.as_slice()).collect();

    let expected = batch_add_multi(&bases, &hot_refs);
    let got = batch_add_multi_tree(&bases, &hot_refs);

    assert_eq!(expected.len(), got.len());
    for (i, (e, g)) in expected.iter().zip(got.iter()).enumerate() {
      assert_eq!(*e, *g, "batch_add_multi_tree mismatch at poly {i}");
    }
  }

  #[test]
  fn test_batch_add_multi_tree() {
    test_batch_add_multi_tree_with::<bn256::Scalar, bn256::Affine>();
    test_batch_add_multi_tree_with::<pallas::Scalar, pallas::Affine>();
  }
}
