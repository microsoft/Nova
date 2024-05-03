//! This module contains the definitions of polynomial types used in the Spartan SNARK.
pub(crate) mod eq;
pub(crate) mod identity;
pub(crate) mod masked_eq;
#[cfg(feature = "bench")]
pub(crate) mod multilinear;
#[cfg(not(feature = "bench"))]
pub mod multilinear;
pub(crate) mod power;
pub(crate) mod univariate;
