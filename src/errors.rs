//! This module defines errors returned by the library.
use core::fmt::Debug;

/// Errors returned by Nova
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NovaError {
  /// returned if the supplied row or col in (row,col,val) tuple is out of range
  InvalidIndex,
  /// returned if the supplied input is not even-sized
  OddInputLength,
  /// returned if the supplied input is not of the right length
  InvalidInputLength,
  /// returned if the supplied witness is not of the right length
  InvalidWitnessLength,
  /// returned if the supplied instance input does not match the previous instance output
  InputOutputMismatch,
  /// returned if the supplied random vector is not of the right length
  InvalidRandomVectorLength,
  /// returned if the supplied witness is not a satisfying witness to a given shape and instance
  UnSat,
  /// returned when the supplied compressed commitment cannot be decompressed
  DecompressionError,
  /// returned when an invalid inner product argument is provided
  InvalidIPA,
  /// returned when the Hadamard check fails
  HadamardCheckFailed,
}
