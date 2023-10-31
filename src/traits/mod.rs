//! This module defines various traits required by the users of the library to implement.
use crate::errors::NovaError;
use bellpepper_core::{boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::fmt::Debug;
use ff::{PrimeField, PrimeFieldBits};
use group::Group;
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

pub mod circuit;
pub mod commitment;
pub mod evaluation;
pub mod snark;

use commitment::CommitmentEngineTrait;

/// Represents an element of a group
/// This is currently tailored for an elliptic curve group
pub trait GroupExt:
  Group<Scalar = Self::ScalarExt> + Serialize + for<'de> Deserialize<'de>
{
  /// A type representing an element of the base field of the group
  type Base: PrimeFieldBits + TranscriptReprTrait<Self> + Serialize + for<'de> Deserialize<'de>;

  /// A type representing an element of the scalar field of the group
  type ScalarExt: PrimeFieldBits
    + PrimeFieldExt
    + TranscriptReprTrait<Self>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// A type representing the compressed version of the group element
  type CompressedGroupElement: CompressedGroup<GroupElement = Self>;

  /// A type representing preprocessed group element
  type PreprocessedGroupElement: Clone + Debug + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that represents a circuit-friendly sponge that consumes elements
  /// from the base field and squeezes out elements of the scalar field
  type RO: ROTrait<Self::Base, Self::Scalar>;

  /// An alternate implementation of `Self::RO` in the circuit model
  type ROCircuit: ROCircuitTrait<Self::Base>;

  /// A type that provides a generic Fiat-Shamir transcript to be used when externalizing proofs
  type TE: TranscriptEngineTrait<Self>;

  /// A type that defines a commitment engine over scalars in the group
  type CE: CommitmentEngineTrait<Self>;

  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self;

  /// Compresses the group element
  fn compress(&self) -> Self::CompressedGroupElement;

  /// Produces a preprocessed element
  fn preprocessed(&self) -> Self::PreprocessedGroupElement;

  /// Produce a vector of group elements using a static label
  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement>;

  /// Returns the affine coordinates (x, y, infinty) for the point
  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool);

  /// Returns A, B, and the order of the group as a big integer
  fn get_curve_params() -> (Self::Base, Self::Base, BigInt);
}

/// Represents a compressed version of a group element
pub trait CompressedGroup:
  Clone
  + Copy
  + Debug
  + Eq
  + Send
  + Sync
  + TranscriptReprTrait<Self::GroupElement>
  + Serialize
  + for<'de> Deserialize<'de>
  + 'static
{
  /// A type that holds the decompressed version of the compressed group element
  type GroupElement: GroupExt;

  /// Decompresses the compressed group element
  fn decompress(&self) -> Option<Self::GroupElement>;
}

/// A helper trait to absorb different objects in RO
pub trait AbsorbInROTrait<G: GroupExt> {
  /// Absorbs the value in the provided RO
  fn absorb_in_ro(&self, ro: &mut G::RO);
}

/// A helper trait that defines the behavior of a hash function that we use as an RO
pub trait ROTrait<Base: PrimeField, Scalar> {
  /// The circuit alter ego of this trait impl - this constrains it to use the same constants
  type CircuitRO: ROCircuitTrait<Base, Constants = Self::Constants>;

  /// A type representing constants/parameters associated with the hash function
  type Constants: Default + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Initializes the hash function
  fn new(constants: Self::Constants, num_absorbs: usize) -> Self;

  /// Adds a scalar to the internal state
  fn absorb(&mut self, e: Base);

  /// Returns a challenge of `num_bits` by hashing the internal state
  fn squeeze(&mut self, num_bits: usize) -> Scalar;
}

/// A helper trait that defines the behavior of a hash function that we use as an RO in the circuit model
pub trait ROCircuitTrait<Base: PrimeField> {
  /// the vanilla alter ego of this trait - this constrains it to use the same constants
  type NativeRO<T: PrimeField>: ROTrait<Base, T, Constants = Self::Constants>;

  /// A type representing constants/parameters associated with the hash function on this Base field
  type Constants: Default + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Initializes the hash function
  fn new(constants: Self::Constants, num_absorbs: usize) -> Self;

  /// Adds a scalar to the internal state
  fn absorb(&mut self, e: &AllocatedNum<Base>);

  /// Returns a challenge of `num_bits` by hashing the internal state
  fn squeeze<CS>(&mut self, cs: CS, num_bits: usize) -> Result<Vec<AllocatedBit>, SynthesisError>
  where
    CS: ConstraintSystem<Base>;
}

/// An alias for constants associated with G::RO
pub type ROConstants<G> =
  <<G as GroupExt>::RO as ROTrait<<G as GroupExt>::Base, <G as Group>::Scalar>>::Constants;

/// An alias for constants associated with `G::ROCircuit`
pub type ROConstantsCircuit<G> =
  <<G as GroupExt>::ROCircuit as ROCircuitTrait<<G as GroupExt>::Base>>::Constants;

/// This trait allows types to implement how they want to be added to `TranscriptEngine`
pub trait TranscriptReprTrait<G: GroupExt>: Send + Sync {
  /// returns a byte representation of self to be added to the transcript
  fn to_transcript_bytes(&self) -> Vec<u8>;
}

/// This trait defines the behavior of a transcript engine compatible with Spartan
pub trait TranscriptEngineTrait<G: GroupExt>: Send + Sync {
  /// initializes the transcript
  fn new(label: &'static [u8]) -> Self;

  /// returns a scalar element of the group as a challenge
  fn squeeze(&mut self, label: &'static [u8]) -> Result<G::Scalar, NovaError>;

  /// absorbs any type that implements `TranscriptReprTrait` under a label
  fn absorb<T: TranscriptReprTrait<G>>(&mut self, label: &'static [u8], o: &T);

  /// adds a domain separator
  fn dom_sep(&mut self, bytes: &'static [u8]);
}

/// Defines additional methods on `PrimeField` objects
pub trait PrimeFieldExt: PrimeField {
  /// Returns a scalar representing the bytes
  fn from_uniform(bytes: &[u8]) -> Self;
}

impl<G: GroupExt, T: TranscriptReprTrait<G>> TranscriptReprTrait<G> for &[T] {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self
      .iter()
      .flat_map(|t| t.to_transcript_bytes())
      .collect::<Vec<u8>>()
  }
}
