//! This module implements various gadgets necessary for Nova and applications built with Nova.

/// Elliptic curve gadgets for in-circuit point operations.
pub mod ecc;

/// Non-native field arithmetic gadgets for operations on fields with different characteristics.
pub mod nonnative;

/// Utility gadgets for common operations like conditional selection, bit manipulation, etc.
pub mod utils;
