//! This module contains the definitions of polynomial types used in the Spartan SNARK.

/// Module providing the equality polynomial.
pub mod eq;

pub(crate) mod identity;
pub(crate) mod masked_eq;

/// Module providing multilinear polynomial types.
pub mod multilinear;

/// Module providing power polynomial.
pub mod power;

pub(crate) mod univariate;
