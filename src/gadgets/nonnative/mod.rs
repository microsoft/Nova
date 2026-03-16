//! This module implements various gadgets necessary for doing non-native arithmetic
//! Code in this module is adapted from [bellman-bignat](https://github.com/alex-ozdemir/bellman-bignat), which is licenced under MIT

use crate::frontend::SynthesisError;

trait OptionExt<T> {
  fn grab(&self) -> Result<&T, SynthesisError>;
}

impl<T> OptionExt<T> for Option<T> {
  fn grab(&self) -> Result<&T, SynthesisError> {
    self.as_ref().ok_or(SynthesisError::AssignmentMissing)
  }
}

/// Module providing big natural number arithmetic in circuits.
pub mod bignat;

/// Module providing utility types and functions for non-native arithmetic.
pub mod util;
