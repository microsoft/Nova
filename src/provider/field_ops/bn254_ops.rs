//! BN254 scalar field (Fr) low-level Montgomery arithmetic primitives.
//!
//! These operate directly on the `[u64; 4]` limb representation used by
//! `halo2curves::bn256::Fr`, which stores field elements in Montgomery form.
//!
//! Key primitives:
//! - `mac`: multiply-accumulate with carry (the inner-loop workhorse)
//! - `montgomery_reduce`: reduce an 8-limb product back to 4-limb Montgomery form
//! - `mul_by_2limbs`: interleaved Montgomery multiply for `[0, 0, lo, hi]` challenges
//! - `schoolbook_4x4`: full 4×4 multiply returning unreduced 8-limb product

/// BN254 scalar field modulus p
pub(crate) const MODULUS: [u64; 4] = [
  0x43e1f593f0000001,
  0x2833e84879b97091,
  0xb85045b68181585d,
  0x30644e72e131a029,
];

/// -p^{-1} mod 2^64 (Montgomery reduction constant)
pub(crate) const INV: u64 = 0xc2e1f593efffffff;

/// R = 2^256 mod p (Montgomery form of 1)
#[allow(dead_code)]
pub(crate) const R: [u64; 4] = [
  0xac96341c4ffffffb,
  0x36fc76959f60cd29,
  0x666ea36f7879462e,
  0x0e0a77c19a07df2f,
];

/// R^2 = (2^256)^2 mod p (for converting integers to Montgomery form)
pub(crate) const R2: [u64; 4] = [
  0x1bb8e645ae216da7,
  0x53fe3ab1e35c59e3,
  0x8c49833d53bb8085,
  0x0216d0b17f4e44a5,
];

/// Multiply-accumulate with carry: returns (lo, carry) where
/// `lo + carry * 2^64 = a + b * c + carry_in`
#[inline(always)]
pub(crate) const fn mac(a: u64, b: u64, c: u64, carry: u64) -> (u64, u64) {
  let ret = (a as u128) + (b as u128) * (c as u128) + (carry as u128);
  (ret as u64, (ret >> 64) as u64)
}

/// Add with carry: returns (sum, carry)
#[inline(always)]
pub(crate) const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
  let ret = (a as u128) + (b as u128) + (carry as u128);
  (ret as u64, (ret >> 64) as u64)
}

/// Subtract with borrow: returns (diff, borrow)
#[inline(always)]
pub(crate) const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
  let ret = (a as u128).wrapping_sub((b as u128) + ((borrow >> 63) as u128));
  (ret as u64, (ret >> 64) as u64)
}

/// Conditionally subtract the modulus if `result >= MODULUS`.
#[inline(always)]
fn subtract_modulus_if_needed(result: &mut [u64; 4]) {
  // Test if result >= MODULUS
  let (_, borrow) = sbb(result[0], MODULUS[0], 0);
  let (_, borrow) = sbb(result[1], MODULUS[1], borrow);
  let (_, borrow) = sbb(result[2], MODULUS[2], borrow);
  let (_, borrow) = sbb(result[3], MODULUS[3], borrow);

  // borrow >> 63 == 1 means result < modulus (no subtraction needed)
  // borrow >> 63 == 0 means result >= modulus (must subtract)
  let mask = 0u64.wrapping_sub((borrow >> 63) ^ 1);

  let (r0, borrow2) = sbb(result[0], MODULUS[0] & mask, 0);
  let (r1, borrow2) = sbb(result[1], MODULUS[1] & mask, borrow2);
  let (r2, borrow2) = sbb(result[2], MODULUS[2] & mask, borrow2);
  let (r3, _) = sbb(result[3], MODULUS[3] & mask, borrow2);

  result[0] = r0;
  result[1] = r1;
  result[2] = r2;
  result[3] = r3;
}

/// Montgomery reduction: reduce an 8-limb product `t[0..8]` to a 4-limb
/// Montgomery-form field element, computing `t * R^{-1} mod p`.
///
/// Requires `t < R * p` (i.e., t must be a single product of two field
/// elements, not an accumulated sum). For accumulated sums that may exceed
/// this bound, use `montgomery_reduce_9`.
#[inline]
pub(crate) fn montgomery_reduce(t: [u64; 8]) -> [u64; 4] {
  let mut r = t;

  macro_rules! reduce_round {
    ($i:expr) => {
      let k = r[$i].wrapping_mul(INV);
      let (_, carry) = mac(r[$i], k, MODULUS[0], 0);
      let (r1, carry) = mac(r[$i + 1], k, MODULUS[1], carry);
      let (r2, carry) = mac(r[$i + 2], k, MODULUS[2], carry);
      let (r3, carry) = mac(r[$i + 3], k, MODULUS[3], carry);
      let (r4, carry2) = adc(r[$i + 4], carry, 0);
      r[$i + 1] = r1;
      r[$i + 2] = r2;
      r[$i + 3] = r3;
      r[$i + 4] = r4;
      if $i < 3 {
        r[$i + 5] = r[$i + 5].wrapping_add(carry2);
      }
      let _ = carry2;
    };
  }

  reduce_round!(0);
  reduce_round!(1);
  reduce_round!(2);
  reduce_round!(3);

  let mut result = [r[4], r[5], r[6], r[7]];
  subtract_modulus_if_needed(&mut result);
  result
}

/// Montgomery reduction for a 9-limb accumulated value.
///
/// Unlike `montgomery_reduce` (which requires input < R * p), this handles
/// arbitrary 9-limb inputs (up to 2^576) by:
/// 1. Running 4 REDC rounds with full carry propagation through all upper limbs
/// 2. Fully reducing the 4-limb base result via multiple modular subtractions
/// 3. Handling the overflow contribution (limbs beyond 4) via `overflow * R^2 → REDC`
///
/// The result is a canonical 4-limb field element in `[0, p)`.
#[inline]
pub(crate) fn montgomery_reduce_9(t: [u64; 9]) -> [u64; 4] {
  let mut r: [u64; 10] = [t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], 0];

  // 4 rounds of Montgomery reduction with carry propagation through all upper limbs.
  // This differs from `montgomery_reduce` which uses wrapping_add for carry propagation
  // (sufficient for single products but not for accumulated sums that may have full limbs).
  for i in 0..4usize {
    let k = r[i].wrapping_mul(INV);
    let (_, carry) = mac(r[i], k, MODULUS[0], 0);
    let (v1, carry) = mac(r[i + 1], k, MODULUS[1], carry);
    let (v2, carry) = mac(r[i + 2], k, MODULUS[2], carry);
    let (v3, carry) = mac(r[i + 3], k, MODULUS[3], carry);
    r[i + 1] = v1;
    r[i + 2] = v2;
    r[i + 3] = v3;
    // Propagate carry through all remaining upper limbs
    let mut c = carry;
    for j in (i + 4)..10 {
      if c == 0 {
        break;
      }
      let (v, nc) = adc(r[j], c, 0);
      r[j] = v;
      c = nc;
    }
  }

  // After 4 rounds: result in r[4..7], overflow in r[8] (r[9] negligible for practical K).
  //
  // V = r[9]*2^320 + r[8]*2^256 + [r4,r5,r6,r7]
  // V ≡ T * R^{-1} (mod p), V < T/R + p
  //
  // The 4-limb portion can be up to 2^256 - 1. Since p ≈ 0.189 * 2^256,
  // we have 2^256 / p ≈ 5.29, so up to 5 subtract_modulus_if_needed calls
  // are needed to bring the value into [0, p).
  let mut result = [r[4], r[5], r[6], r[7]];
  subtract_modulus_if_needed(&mut result);
  subtract_modulus_if_needed(&mut result);
  subtract_modulus_if_needed(&mut result);
  subtract_modulus_if_needed(&mut result);
  subtract_modulus_if_needed(&mut result);

  // Handle overflow from r[8]: represents overflow * 2^256 in the REDC output.
  // overflow * 2^256 mod p (in Montgomery form) = REDC(overflow * R^2).
  if r[8] > 0 {
    let overflow_r2 = mul_scalar_1x4(r[8], &R2);
    let overflow_mont = montgomery_reduce(overflow_r2);
    result = field_add(&result, &overflow_mont);
  }

  // r[9] should always be 0 for practical input sizes (K < 2^64 products)
  debug_assert_eq!(r[9], 0, "extreme overflow in montgomery_reduce_9");

  // Verify result is canonically reduced
  debug_assert!(
    result[3] < MODULUS[3]
      || (result[3] == MODULUS[3]
        && (result[2] < MODULUS[2]
          || (result[2] == MODULUS[2]
            && (result[1] < MODULUS[1]
              || (result[1] == MODULUS[1] && result[0] < MODULUS[0]))))),
    "montgomery_reduce_9: non-canonical result {:?} (overflow r[8]={}, r[9]={})",
    result,
    r[8],
    r[9]
  );

  result
}

/// 4×4 schoolbook multiplication returning an unreduced 8-limb product.
///
/// Given `a` and `b` both in Montgomery form (4 limbs each),
/// returns `a * b` as 8 limbs WITHOUT Montgomery reduction.
/// Caller must eventually reduce via `montgomery_reduce`.
#[inline]
pub(crate) fn schoolbook_4x4(a: &[u64; 4], b: &[u64; 4]) -> [u64; 8] {
  let (r0, carry) = mac(0, a[0], b[0], 0);
  let (r1, carry) = mac(0, a[0], b[1], carry);
  let (r2, carry) = mac(0, a[0], b[2], carry);
  let (r3, r4) = mac(0, a[0], b[3], carry);

  let (r1, carry) = mac(r1, a[1], b[0], 0);
  let (r2, carry) = mac(r2, a[1], b[1], carry);
  let (r3, carry) = mac(r3, a[1], b[2], carry);
  let (r4, r5) = mac(r4, a[1], b[3], carry);

  let (r2, carry) = mac(r2, a[2], b[0], 0);
  let (r3, carry) = mac(r3, a[2], b[1], carry);
  let (r4, carry) = mac(r4, a[2], b[2], carry);
  let (r5, r6) = mac(r5, a[2], b[3], carry);

  let (r3, carry) = mac(r3, a[3], b[0], 0);
  let (r4, carry) = mac(r4, a[3], b[1], carry);
  let (r5, carry) = mac(r5, a[3], b[2], carry);
  let (r6, r7) = mac(r6, a[3], b[3], carry);

  [r0, r1, r2, r3, r4, r5, r6, r7]
}

/// Interleaved Montgomery multiply for a challenge with Montgomery repr `[0, 0, lo, hi]`.
///
/// Computes `a * [0, 0, b_lo, b_hi] * R^{-1} mod p` using only 8 `mac` ops
/// (vs 16 for a full 4×4) plus 2 reduction rounds (vs 4).
///
/// The challenge is stored as a Montgomery-form value `[0, 0, lo, hi]`.
/// The actual field element it represents is `(lo·2^128 + hi·2^192) · R^{-1} mod p`.
/// Both prover and verifier construct the same representation, preserving soundness.
///
/// This is the same approach as Jolt's `mul_by_hi_2limbs`: the first two rounds
/// of interleaved Montgomery multiplication are no-ops (b[0]=b[1]=0), so we skip
/// directly to rounds 2 and 3.
#[inline]
pub(crate) fn mul_by_2limbs(a: &[u64; 4], b_lo: u64, b_hi: u64) -> [u64; 4] {
  // Rounds 0-1 skipped: b[0]=b[1]=0, so T stays zero, k=0, no-op.

  // Round 2: b[2] = b_lo
  let (t0, carry) = mac(0, a[0], b_lo, 0);
  let (t1, carry) = mac(0, a[1], b_lo, carry);
  let (t2, carry) = mac(0, a[2], b_lo, carry);
  let (t3, t4) = mac(0, a[3], b_lo, carry);
  let k = t0.wrapping_mul(INV);
  let (_, carry) = mac(t0, k, MODULUS[0], 0);
  let (t0, carry) = mac(t1, k, MODULUS[1], carry);
  let (t1, carry) = mac(t2, k, MODULUS[2], carry);
  let (t2, carry) = mac(t3, k, MODULUS[3], carry);
  let (t3, t4_carry) = adc(t4, 0, carry);

  // Round 3: b[3] = b_hi
  let (t0, carry) = mac(t0, a[0], b_hi, 0);
  let (t1, carry) = mac(t1, a[1], b_hi, carry);
  let (t2, carry) = mac(t2, a[2], b_hi, carry);
  let (t3, carry) = mac(t3, a[3], b_hi, carry);
  let (t4, _) = adc(t4_carry, carry, 0);
  let k = t0.wrapping_mul(INV);
  let (_, carry) = mac(t0, k, MODULUS[0], 0);
  let (r0, carry) = mac(t1, k, MODULUS[1], carry);
  let (r1, carry) = mac(t2, k, MODULUS[2], carry);
  let (r2, carry) = mac(t3, k, MODULUS[3], carry);
  let (r3, _) = adc(t4, 0, carry);

  let mut result = [r0, r1, r2, r3];
  subtract_modulus_if_needed(&mut result);
  result
}

/// Full interleaved Montgomery multiplication.
/// Computes `a * b * R^{-1} mod p`.
#[inline]
#[allow(dead_code)]
pub(crate) fn mont_mul(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
  montgomery_reduce(schoolbook_4x4(a, b))
}

/// Add two 9-limb values with carry propagation.
///
/// Used for accumulating unreduced products. The 9th limb catches overflow
/// that would be lost with only 8 limbs when accumulating many products.
#[inline]
pub(crate) fn add_9limb(a: &[u64; 9], b: &[u64; 9]) -> [u64; 9] {
  let mut r = [0u64; 9];
  let mut carry = 0u64;
  for i in 0..9 {
    let (sum, c) = adc(a[i], b[i], carry);
    r[i] = sum;
    carry = c;
  }
  r
}

/// Multiply a u64 scalar by a 4-limb value, returning an 8-limb result.
///
/// Used to compute `overflow * R^2` when reducing 9-limb accumulators.
#[inline]
pub(crate) fn mul_scalar_1x4(s: u64, a: &[u64; 4]) -> [u64; 8] {
  let (r0, carry) = mac(0, s, a[0], 0);
  let (r1, carry) = mac(0, s, a[1], carry);
  let (r2, carry) = mac(0, s, a[2], carry);
  let (r3, r4) = mac(0, s, a[3], carry);
  [r0, r1, r2, r3, r4, 0, 0, 0]
}

/// Add two 4-limb field elements mod p.
///
/// Both inputs must be in `[0, p)`. The result is in `[0, p)`.
#[inline]
pub(crate) fn field_add(a: &[u64; 4], b: &[u64; 4]) -> [u64; 4] {
  let (r0, carry) = adc(a[0], b[0], 0);
  let (r1, carry) = adc(a[1], b[1], carry);
  let (r2, carry) = adc(a[2], b[2], carry);
  let (r3, _) = adc(a[3], b[3], carry);
  let mut result = [r0, r1, r2, r3];
  subtract_modulus_if_needed(&mut result);
  result
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use halo2curves::bn256::Fr;
  use rand_chacha::ChaCha20Rng;
  use rand_core::SeedableRng;

  #[test]
  fn test_constants_match_halo2curves() {
    assert_eq!(Fr::ONE.0, R, "R constant mismatch");
    assert_eq!(Fr::ZERO.0, [0u64; 4]);

    // MODULUS[0] * INV ≡ -1 (mod 2^64)
    assert_eq!(MODULUS[0].wrapping_mul(INV), u64::MAX);
  }

  #[test]
  fn test_mont_mul_matches_halo2curves() {
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    for _ in 0..1000 {
      let a = Fr::random(&mut rng);
      let b = Fr::random(&mut rng);
      let expected = a * b;
      assert_eq!(Fr(mont_mul(&a.0, &b.0)), expected);
    }
  }

  #[test]
  fn test_schoolbook_reduce_matches() {
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    for _ in 0..1000 {
      let a = Fr::random(&mut rng);
      let b = Fr::random(&mut rng);
      let expected = a * b;
      assert_eq!(Fr(montgomery_reduce(schoolbook_4x4(&a.0, &b.0))), expected);
    }
  }

  #[test]
  fn test_mul_by_2limbs() {
    let mut rng = ChaCha20Rng::from_seed([13u8; 32]);
    for _ in 0..1000 {
      let a = Fr::random(&mut rng);
      let lo = rand_core::RngCore::next_u64(&mut rng);
      let hi = rand_core::RngCore::next_u64(&mut rng) >> 3;

      // [0, 0, lo, hi] as Montgomery representation
      let challenge_mont = Fr([0, 0, lo, hi]);
      let expected = a * challenge_mont;
      assert_eq!(Fr(mul_by_2limbs(&a.0, lo, hi)), expected);
    }
  }

  #[test]
  fn test_mul_by_2limbs_edge_cases() {
    assert_eq!(Fr(mul_by_2limbs(&Fr::ONE.0, 0, 0)), Fr::ZERO);
    assert_eq!(Fr(mul_by_2limbs(&Fr::ZERO.0, 42, 7)), Fr::ZERO);

    let a = Fr::from(7u64);
    let c = Fr([0, 0, 1, 0]);
    assert_eq!(Fr(mul_by_2limbs(&a.0, 1, 0)), a * c);
  }

  #[test]
  fn test_unreduced_accumulation_9limb() {
    let mut rng = ChaCha20Rng::from_seed([99u8; 32]);

    for n in [1, 2, 10, 50, 100, 500, 1000] {
      let mut acc_9 = [0u64; 9];
      let mut acc_field = Fr::ZERO;

      for _ in 0..n {
        let a = Fr::random(&mut rng);
        let b = Fr::random(&mut rng);
        acc_field += a * b;
        let prod = schoolbook_4x4(&a.0, &b.0);
        let prod9 = [prod[0], prod[1], prod[2], prod[3], prod[4], prod[5], prod[6], prod[7], 0];
        acc_9 = add_9limb(&acc_9, &prod9);
      }

      assert_eq!(Fr(montgomery_reduce_9(acc_9)), acc_field, "Mismatch for n={n}");
    }
  }
}
