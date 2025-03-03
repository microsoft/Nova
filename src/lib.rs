//! This library implements Nova, a high-speed recursive SNARK.
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![forbid(unsafe_code)]
#![cfg_attr(not(feature = "std"), no_std)]

// main APIs exposed by this library
pub mod nova;

// public modules
pub mod errors;
pub mod frontend;
pub mod gadgets;
pub mod provider;
pub mod spartan;
pub mod traits;

// private modules
mod constants;
mod digest;
mod r1cs;

use traits::{commitment::CommitmentEngineTrait, Engine};

// some type aliases
type CommitmentKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey;
type DerandKey<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::DerandKey;
type Commitment<E> = <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment;
type CE<E> = <E as Engine>::CE;

// TODO -> See how to handle this in best way possible
#[cfg(not(feature = "std"))]
pub(crate) mod prelude {
  extern crate alloc;
  pub use alloc::borrow::ToOwned;
  pub use alloc::boxed::Box;
  pub use alloc::format;
  pub use alloc::string::String;
  pub use alloc::string::ToString;
  pub use alloc::vec;
  pub use alloc::vec::Vec;

  pub use alloc::collections::BTreeMap;
  pub use alloc::collections::VecDeque;
  // #[cfg(feature = "std")]
  // pub use std::collections::{BTreeMap, HashMap, VecDeque};
}
