//! Self-contained sub-circuit implementations for various primitives.

use super::SynthesisError;

mod multieq;
mod uint32;

pub mod boolean;
pub mod num;
pub mod poseidon;
pub mod sha256;

/// A trait for representing an assignment to a variable.
pub trait Assignment<T> {
  /// Get the value of the assigned variable.
  fn get(&self) -> Result<&T, SynthesisError>;
}

impl<T> Assignment<T> for Option<T> {
  fn get(&self) -> Result<&T, SynthesisError> {
    match *self {
      Some(ref v) => Ok(v),
      None => Err(SynthesisError::AssignmentMissing),
    }
  }
}
