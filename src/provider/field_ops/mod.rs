//! Fast field arithmetic for sumcheck-intensive provers.
//!
//! This module provides BN254-specific optimizations for the inner loops
//! of Spartan's sumcheck prover:
//!
//! - `Challenge128`: 125-bit challenges enabling 4×2 Montgomery multiply
//! - `Unreduced`: deferred-reduction accumulator for multiply-sum patterns
//! - `bn254_ops`: raw limb-level Montgomery arithmetic primitives
//!
//! These are concrete types specific to `halo2curves::bn256::Fr`.
//! The abstract interface consumed by generic sumcheck code is defined
//! in `crate::spartan::SumcheckScalarOps`.

pub mod bn254_ops;
pub mod challenge;
pub mod unreduced;

pub use challenge::Challenge128;
pub use unreduced::{Unreduced, UnreducedPair, UnreducedTriple};
