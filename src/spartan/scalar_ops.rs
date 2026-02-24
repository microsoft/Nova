//! Deferred-reduction multiply-accumulate for sumcheck inner loops.
//!
//! The key optimization: instead of performing a Montgomery reduction after
//! every field multiplication in the sumcheck accumulation loops, we accumulate
//! raw wide products and reduce once at the end. For N products, this saves
//! N-1 Montgomery reductions.
//!
//! The trait `ScalarMulAccum` abstracts this pattern. Engines with optimized
//! field arithmetic (BN254) use a wide accumulator type; other engines use
//! the scalar type directly (multiply-then-reduce, no overhead).

use crate::{
  provider::{Bn256EngineIPA, Bn256EngineKZG},
  traits::Engine,
};

/// Deferred-reduction multiply-accumulate operations for sumcheck.
///
/// In sumcheck inner loops, we compute `Σ a_i * b_i` where each product
/// is a field multiplication. With `ScalarMulAccum`, each multiply returns
/// an unreduced accumulator (`MulAccum`) that supports cheap wide addition.
/// A single `reduce` call at the end converts the accumulated sum to a
/// proper field element.
///
/// For engines without specialized implementations, `MulAccum = Scalar`
/// and multiply-accumulate is just standard field multiplication (no overhead).
pub trait ScalarMulAccum: Engine {
  /// Accumulator type for unreduced products.
  ///
  /// For BN254: a 9-limb wide representation (`Unreduced`).
  /// For other engines: `Self::Scalar` (immediate reduction).
  type MulAccum: Copy
    + Clone
    + Send
    + Sync
    + Default
    + std::ops::Add<Output = Self::MulAccum>
    + std::ops::AddAssign;

  /// Multiply two field elements, returning an unreduced accumulator.
  fn mul_accum(a: &Self::Scalar, b: &Self::Scalar) -> Self::MulAccum;

  /// Reduce an accumulator to a field element.
  fn reduce(u: Self::MulAccum) -> Self::Scalar;
}

// -- BN254 optimized implementations --

use crate::provider::field_ops::Unreduced;

impl ScalarMulAccum for Bn256EngineKZG {
  type MulAccum = Unreduced;

  #[inline]
  fn mul_accum(a: &Self::Scalar, b: &Self::Scalar) -> Self::MulAccum {
    Unreduced::mul(a, b)
  }

  #[inline]
  fn reduce(u: Self::MulAccum) -> Self::Scalar {
    u.reduce()
  }
}

impl ScalarMulAccum for Bn256EngineIPA {
  type MulAccum = Unreduced;

  #[inline]
  fn mul_accum(a: &Self::Scalar, b: &Self::Scalar) -> Self::MulAccum {
    Unreduced::mul(a, b)
  }

  #[inline]
  fn reduce(u: Self::MulAccum) -> Self::Scalar {
    u.reduce()
  }
}

// -- Default (no-op) implementations for other engines --

macro_rules! impl_default_scalar_mul_accum {
  ($engine:ty) => {
    impl ScalarMulAccum for $engine {
      type MulAccum = <Self as Engine>::Scalar;

      #[inline]
      fn mul_accum(a: &Self::Scalar, b: &Self::Scalar) -> Self::MulAccum {
        *a * *b
      }

      #[inline]
      fn reduce(u: Self::MulAccum) -> Self::Scalar {
        u
      }
    }
  };
}

use crate::provider::{
  GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
};

impl_default_scalar_mul_accum!(GrumpkinEngine);
impl_default_scalar_mul_accum!(PallasEngine);
impl_default_scalar_mul_accum!(VestaEngine);
impl_default_scalar_mul_accum!(Secp256k1Engine);
impl_default_scalar_mul_accum!(Secq256k1Engine);
