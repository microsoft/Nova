//! This module defines a collection of traits that define the behavior of a commitment engine
//! We require the commitment engine to provide a commitment to vectors with a single group element
use crate::{
  errors::NovaError,
  traits::{AbsorbInROTrait, AppendToTranscriptTrait, CompressedGroup, Group},
};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign},
};
use serde::{Deserialize, Serialize};

/// This trait defines the behavior of commitment key
#[allow(clippy::len_without_is_empty)]
pub trait CommitmentGensTrait<G: Group>:
  Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// Holds the type of the commitment that can be produced
  type Commitment;

  /// Holds the type of the compressed commitment
  type CompressedCommitment;

  /// Samples a new commitment key of a specified size
  fn new(label: &'static [u8], n: usize) -> Self;

  /// Returns the vector length that can be committed
  fn len(&self) -> usize;

  /// Commits to a vector using the commitment key
  fn commit(&self, v: &[G::Scalar]) -> Self::Commitment;
}

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
  + Serialize
  + for<'de> Deserialize<'de>
  + AbsorbInROTrait<G>
  + AppendToTranscriptTrait<G>
  + CommitmentOps
  + CommitmentOpsOwned
  + ScalarMul<G::Scalar>
{
  /// Holds the type of the compressed commitment
  type CompressedCommitment;

  /// Compresses self into a compressed commitment
  fn compress(&self) -> Self::CompressedCommitment;

  /// Returns the coordinate representation of the commitment
  fn to_coordinates(&self) -> (G::Base, G::Base, bool);
}

/// This trait defines the behavior of a compressed commitment
pub trait CompressedCommitmentTrait<C: CompressedGroup>:
  Clone
  + Debug
  + PartialEq
  + Eq
  + Send
  + Sync
  + Serialize
  + for<'de> Deserialize<'de>
  + AppendToTranscriptTrait<C::GroupElement>
{
  /// Holds the type of the commitment that can be decompressed into
  type Commitment;

  /// Decompresses self into a commitment
  fn decompress(&self) -> Result<Self::Commitment, NovaError>;
}

/// A trait that ties different pieces of the commitment generation together
pub trait CommitmentEngineTrait<G: Group>:
  Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// Holds the type of the commitment key
  type CommitmentGens: CommitmentGensTrait<
    G,
    Commitment = Self::Commitment,
    CompressedCommitment = Self::CompressedCommitment,
  >;

  /// Holds the type of the commitment
  type Commitment: CommitmentTrait<G, CompressedCommitment = Self::CompressedCommitment>;

  /// Holds the type of the compressed commitment
  type CompressedCommitment: CompressedCommitmentTrait<
    G::CompressedGroupElement,
    Commitment = Self::Commitment,
  >;

  /// Commits to the provided vector using the provided generators
  fn commit(gens: &Self::CommitmentGens, v: &[G::Scalar]) -> Self::Commitment;
}
