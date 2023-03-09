//! This module defines a collection of traits that define the behavior of a commitment engine
//! We require the commitment engine to provide a commitment to vectors with a single group element
use crate::{
  errors::NovaError,
  traits::{AbsorbInROTrait, Group, TranscriptReprTrait},
};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign},
};
use serde::{Deserialize, Serialize};

/// Defines basic operations on commitments
pub trait CommitmentOps<Rhs = Self, Output = Self>:
  Add<Rhs, Output = Output> + AddAssign<Rhs>
{
}

impl<T, Rhs, Output> CommitmentOps<Rhs, Output> for T where
  T: Add<Rhs, Output = Output> + AddAssign<Rhs>
{
}

/// A helper trait for references with a commitment operation
pub trait CommitmentOpsOwned<Rhs = Self, Output = Self>:
  for<'r> CommitmentOps<&'r Rhs, Output>
{
}
impl<T, Rhs, Output> CommitmentOpsOwned<Rhs, Output> for T where
  T: for<'r> CommitmentOps<&'r Rhs, Output>
{
}

/// A helper trait for types implementing a multiplication of a commitment with a scalar
pub trait ScalarMul<Rhs, Output = Self>: Mul<Rhs, Output = Output> + MulAssign<Rhs> {}

impl<T, Rhs, Output> ScalarMul<Rhs, Output> for T where T: Mul<Rhs, Output = Output> + MulAssign<Rhs>
{}

/// This trait defines the behavior of the commitment
pub trait CommitmentTrait<G: Group>:
  Clone
  + Copy
  + Debug
  + Default
  + PartialEq
  + Eq
  + Send
  + Sync
  + TranscriptReprTrait<G>
  + Serialize
  + for<'de> Deserialize<'de>
  + AbsorbInROTrait<G>
  + CommitmentOps
  + CommitmentOpsOwned
  + ScalarMul<G::Scalar>
{
  /// Holds the type of the compressed commitment
  type CompressedCommitment: Clone
    + Debug
    + PartialEq
    + Eq
    + Send
    + Sync
    + TranscriptReprTrait<G>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// Compresses self into a compressed commitment
  fn compress(&self) -> Self::CompressedCommitment;

  /// Returns the coordinate representation of the commitment
  fn to_coordinates(&self) -> (G::Base, G::Base, bool);

  /// Decompresses a compressed commitment into a commitment
  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError>;
}

/// A trait that ties different pieces of the commitment generation together
pub trait CommitmentEngineTrait<G: Group>:
  Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// Holds the type of the commitment key
  type CommitmentKey: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Holds the type of the commitment
  type Commitment: CommitmentTrait<G>;

  /// Samples a new commitment key of a specified size
  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey;

  /// Commits to the provided vector using the provided generators
  fn commit(ck: &Self::CommitmentKey, v: &[G::Scalar]) -> Self::Commitment;
}
