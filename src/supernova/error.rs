//! This module defines errors returned by the library.
use core::fmt::Debug;
use thiserror::Error;

use crate::errors::NovaError;

/// Errors returned by Nova
#[derive(Clone, Debug, Eq, PartialEq, Error)]
pub enum SuperNovaError {
  /// Nova error
  #[error("NovaError")]
  NovaError(NovaError),
  /// missig commitment key
  #[error("MissingCK")]
  MissingCK,
  /// Extended error for supernova
  #[error("UnSatIndex")]
  UnSatIndex(&'static str, usize),
}
