//! This module defines constants used throughout the library.

/// Number of bits used for challenge generation in the protocol.
pub const NUM_CHALLENGE_BITS: usize = 128;

/// Number of bits used for hash output sizing.
pub const NUM_HASH_BITS: usize = 250;

/// Width of each limb in bignat representation.
pub const BN_LIMB_WIDTH: usize = 64;

/// Number of limbs in bignat representation.
pub const BN_N_LIMBS: usize = 4;
