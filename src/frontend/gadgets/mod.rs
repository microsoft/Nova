//! Self-contained sub-circuit implementations for various primitives.
use super::SynthesisError;
pub mod boolean;
mod multieq;
pub mod num;
pub mod sha256;
mod uint32;

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
