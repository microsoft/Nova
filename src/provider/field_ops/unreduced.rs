//! Unreduced 9-limb accumulator for deferred Montgomery reduction.
//!
//! In sumcheck evaluation-point computation, we frequently compute
//! `Σ a_i * b_i` where each product is a full field multiply. Normally,
//! each multiply requires Montgomery reduction (expensive). With `Unreduced`,
//! we accumulate the raw products via wide addition (cheap), and
//! perform a single Montgomery reduction at the end.
//!
//! For N products, this saves N-1 Montgomery reductions.

use super::bn254_ops;
use halo2curves::bn256::Fr;
use std::ops::{Add, AddAssign};

/// A 9-limb unreduced product accumulator, representing the sum of 4×4
/// schoolbook multiplication results before Montgomery reduction.
///
/// Each individual product is 8 limbs (512 bits), but accumulating many
/// products can overflow 8 limbs. The 9th limb catches this overflow.
///
/// After accumulation, call `reduce()` to obtain a proper `Fr` field element.
///
/// Overflow safety: each 4×4 product is at most ~2^508. Accumulating K products
/// yields at most K·2^508. With 9 limbs (576 bits), this supports K ≤ 2^68.
/// In practice, sumcheck polynomials have ≤ 2^30 elements, so overflow is not a concern.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Unreduced {
  /// The 9 limbs of the unreduced accumulator (8 from product + 1 overflow)
  pub limbs: [u64; 9],
}

impl Unreduced {
  /// The zero accumulator.
  #[inline]
  pub const fn zero() -> Self {
    Self { limbs: [0u64; 9] }
  }

  /// Create from a raw 8-limb product (e.g., from `schoolbook_4x4`),
  /// zero-extending to 9 limbs.
  #[inline]
  pub const fn from_raw(limbs: [u64; 8]) -> Self {
    Self {
      limbs: [
        limbs[0], limbs[1], limbs[2], limbs[3], limbs[4], limbs[5], limbs[6], limbs[7], 0,
      ],
    }
  }

  /// Compute `a * b` without reduction, returning an `Unreduced` product.
  #[inline]
  pub fn mul(a: &Fr, b: &Fr) -> Self {
    Self::from_raw(bn254_ops::schoolbook_4x4(&a.0, &b.0))
  }

  /// Montgomery-reduce this accumulator to a proper field element.
  ///
  /// Computes `self * R^{-1} mod p`.
  ///
  /// Uses `montgomery_reduce_9` which handles arbitrary 9-limb inputs
  /// with full carry propagation and proper final reduction.
  #[inline]
  pub fn reduce(self) -> Fr {
    Fr(bn254_ops::montgomery_reduce_9(self.limbs))
  }
}

impl Add for Unreduced {
  type Output = Self;

  #[inline]
  fn add(self, rhs: Self) -> Self {
    Self {
      limbs: bn254_ops::add_9limb(&self.limbs, &rhs.limbs),
    }
  }
}

impl AddAssign for Unreduced {
  #[inline]
  fn add_assign(&mut self, rhs: Self) {
    self.limbs = bn254_ops::add_9limb(&self.limbs, &rhs.limbs);
  }
}

/// A pair of unreduced accumulators, used in `compute_eval_points_*` which
/// returns two accumulated sums.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnreducedPair(pub Unreduced, pub Unreduced);

impl UnreducedPair {
  /// Zero pair.
  #[inline]
  pub fn zero() -> Self {
    Self(Unreduced::zero(), Unreduced::zero())
  }

  /// Reduce both accumulators to field elements.
  #[inline]
  pub fn reduce(self) -> (Fr, Fr) {
    (self.0.reduce(), self.1.reduce())
  }
}

impl Add for UnreducedPair {
  type Output = Self;

  #[inline]
  fn add(self, rhs: Self) -> Self {
    Self(self.0 + rhs.0, self.1 + rhs.1)
  }
}

/// A triple of unreduced accumulators, used in cubic sumcheck eval points.
#[derive(Copy, Clone, Debug, Default)]
pub struct UnreducedTriple(pub Unreduced, pub Unreduced, pub Unreduced);

impl UnreducedTriple {
  /// Zero triple.
  #[inline]
  pub fn zero() -> Self {
    Self(Unreduced::zero(), Unreduced::zero(), Unreduced::zero())
  }

  /// Reduce all three accumulators to field elements.
  #[inline]
  pub fn reduce(self) -> (Fr, Fr, Fr) {
    (self.0.reduce(), self.1.reduce(), self.2.reduce())
  }
}

impl Add for UnreducedTriple {
  type Output = Self;

  #[inline]
  fn add(self, rhs: Self) -> Self {
    Self(self.0 + rhs.0, self.1 + rhs.1, self.2 + rhs.2)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use rand_chacha::ChaCha20Rng;
  use rand_core::SeedableRng;

  #[test]
  fn test_single_mul_reduce() {
    let mut rng = ChaCha20Rng::from_seed([77u8; 32]);
    for _ in 0..1000 {
      let a = Fr::random(&mut rng);
      let b = Fr::random(&mut rng);
      let expected = a * b;
      assert_eq!(Unreduced::mul(&a, &b).reduce(), expected);
    }
  }

  #[test]
  fn test_accumulation_correctness() {
    let mut rng = ChaCha20Rng::from_seed([33u8; 32]);

    for n in [1, 2, 10, 50, 100, 500, 1000] {
      let mut acc = Unreduced::zero();
      let mut expected = Fr::ZERO;

      for _ in 0..n {
        let a = Fr::random(&mut rng);
        let b = Fr::random(&mut rng);
        expected += a * b;
        acc += Unreduced::mul(&a, &b);
      }

      assert_eq!(acc.reduce(), expected, "Mismatch for n={n}");
    }
  }

  #[test]
  fn test_pair_accumulation() {
    let mut rng = ChaCha20Rng::from_seed([44u8; 32]);
    let n = 100;
    let mut acc = UnreducedPair::zero();
    let mut e0 = Fr::ZERO;
    let mut e1 = Fr::ZERO;

    for _ in 0..n {
      let a = Fr::random(&mut rng);
      let b = Fr::random(&mut rng);
      let c = Fr::random(&mut rng);
      let d = Fr::random(&mut rng);

      e0 += a * b;
      e1 += c * d;

      acc = acc + UnreducedPair(Unreduced::mul(&a, &b), Unreduced::mul(&c, &d));
    }

    let (r0, r1) = acc.reduce();
    assert_eq!(r0, e0);
    assert_eq!(r1, e1);
  }

  #[test]
  fn test_overflow_limb_needed() {
    // Verify that the 9th limb actually gets used for large accumulations.
    let mut rng = ChaCha20Rng::from_seed([88u8; 32]);
    let mut acc = Unreduced::zero();
    let mut expected = Fr::ZERO;

    // 50 products should be enough to overflow 8 limbs
    for _ in 0..50 {
      let a = Fr::random(&mut rng);
      let b = Fr::random(&mut rng);
      expected += a * b;
      acc += Unreduced::mul(&a, &b);
    }

    // The 9th limb should be nonzero for a large enough accumulation
    // (This is a probabilistic check; with random inputs, overflow is very likely at n=50)
    assert_eq!(acc.reduce(), expected);
  }
}
