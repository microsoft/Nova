//! Challenge128: a 125-bit sumcheck challenge optimized for BN254.
//!
//! Instead of using a full 254-bit field element as the sumcheck challenge,
//! we use a 125-bit value stored in the Montgomery representation `[0, 0, lo, hi]`.
//! This allows multiplying a field element by a challenge using only 8 `mac`
//! operations instead of 16 (a 4×2 schoolbook multiply vs 4×4).
//!
//! The approach follows Jolt's `MontU128Challenge`: the challenge is squeezed
//! from the transcript as 125 raw bits, then placed in the upper two limbs of
//! a 4-limb Montgomery representation. The resulting field element is
//! `(lo·2^128 + hi·2^192) · R^{-1} mod p`, which is a valid, uniformly
//! distributed (up to 125 bits of entropy) field element.

use super::bn254_ops;
use halo2curves::bn256::Fr;

/// A 125-bit challenge stored as two u64 limbs.
///
/// Represents the field element whose Montgomery form is `[0, 0, lo, hi]`.
/// `hi` has its top 3 bits always zero, giving 64 + 61 = 125 bits of entropy.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Challenge128 {
  /// Lower 64 bits of the challenge
  pub lo: u64,
  /// Upper 61 bits of the challenge (top 3 bits always zero)
  pub hi: u64,
}

/// Number of challenge bits used in sumcheck
pub const CHALLENGE_BITS: usize = 125;

impl Challenge128 {
  /// Construct from two limbs. `hi` must have top 3 bits zero.
  #[inline]
  pub fn new(lo: u64, hi: u64) -> Self {
    debug_assert_eq!(hi >> 61, 0, "top 3 bits of hi must be zero");
    Self { lo, hi }
  }

  /// Construct from raw transcript bytes (little-endian).
  ///
  /// Takes the first 16 bytes, masks `hi` to 61 bits.
  pub fn from_bytes(bytes: &[u8]) -> Self {
    assert!(bytes.len() >= 16);
    let lo = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
    let hi = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) & (u64::MAX >> 3);
    Self { lo, hi }
  }

  /// Convert to the full `Fr` field element.
  ///
  /// The Montgomery representation is `[0, 0, lo, hi]`, so the actual
  /// field element is `(lo·2^128 + hi·2^192) · R^{-1} mod p`.
  #[inline]
  pub fn to_scalar(self) -> Fr {
    Fr([0, 0, self.lo, self.hi])
  }

  /// Multiply a field element by this challenge.
  ///
  /// Uses the optimized 4×2 interleaved Montgomery multiply (8 macs vs 16).
  #[inline]
  pub fn mul_scalar(self, a: &Fr) -> Fr {
    Fr(bn254_ops::mul_by_2limbs(&a.0, self.lo, self.hi))
  }
}

impl From<Challenge128> for Fr {
  #[inline]
  fn from(c: Challenge128) -> Fr {
    c.to_scalar()
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use rand_chacha::ChaCha20Rng;
  use rand_core::SeedableRng;

  #[test]
  fn test_challenge_mul_consistency() {
    let mut rng = ChaCha20Rng::from_seed([55u8; 32]);
    for _ in 0..1000 {
      let a = Fr::random(&mut rng);
      let lo = rand_core::RngCore::next_u64(&mut rng);
      let hi = rand_core::RngCore::next_u64(&mut rng) >> 3;
      let c = Challenge128::new(lo, hi);

      // Optimized path
      let result = c.mul_scalar(&a);

      // Standard path: convert to Fr, then multiply
      let expected = a * c.to_scalar();

      assert_eq!(result, expected);
    }
  }

  #[test]
  fn test_challenge_to_scalar() {
    let c = Challenge128::new(42, 7);
    let fr = c.to_scalar();
    assert_eq!(fr.0, [0, 0, 42, 7]);
  }

  #[test]
  fn test_challenge_from_bytes() {
    let bytes = [0u8; 32];
    let c = Challenge128::from_bytes(&bytes);
    assert_eq!(c.lo, 0);
    assert_eq!(c.hi, 0);

    let mut bytes = [0xFFu8; 32];
    let c = Challenge128::from_bytes(&bytes);
    assert_eq!(c.lo, u64::MAX);
    assert_eq!(c.hi, u64::MAX >> 3); // top 3 bits masked

    // Specific pattern
    bytes[0..8].copy_from_slice(&42u64.to_le_bytes());
    bytes[8..16].copy_from_slice(&100u64.to_le_bytes());
    let c = Challenge128::from_bytes(&bytes);
    assert_eq!(c.lo, 42);
    assert_eq!(c.hi, 100);
  }
}
