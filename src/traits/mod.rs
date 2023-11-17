//! This module defines various traits required by the users of the library to implement.
use crate::errors::NovaError;
use bellpepper_core::{boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::fmt::Debug;
use ff::{PrimeField, PrimeFieldBits};
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

pub mod commitment;

use commitment::CommitmentEngineTrait;

/// Represents an element of a group
/// This is currently tailored for an elliptic curve group
pub trait Group: Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq {
  /// A type representing an element of the base field of the group
  type Base: PrimeFieldBits + Serialize + for<'de> Deserialize<'de>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeFieldBits + PrimeFieldExt + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Returns A, B, the order of the group, the size of the base field as big integers
  fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt);
}

/// A collection of engines that are required by the library
pub trait Engine: Clone + Copy + Debug + Send + Sync + Sized + Eq + PartialEq {
  /// A type representing an element of the base field of the group
  type Base: PrimeFieldBits + TranscriptReprTrait<Self::GE> + Serialize + for<'de> Deserialize<'de>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeFieldBits
    + PrimeFieldExt
    + Send
    + Sync
    + TranscriptReprTrait<Self::GE>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// A type that represents an element of the group
  type GE: Group<Base = Self::Base, Scalar = Self::Scalar> + Serialize + for<'de> Deserialize<'de>;

  /// A type that represents a circuit-friendly sponge that consumes elements
  /// from the base field and squeezes out elements of the scalar field
  type RO: ROTrait<Self::Base, Self::Scalar>;

  /// An alternate implementation of `Self::RO` in the circuit model
  type ROCircuit: ROCircuitTrait<Self::Base>;

  /// A type that provides a generic Fiat-Shamir transcript to be used when externalizing proofs
  type TE: TranscriptEngineTrait<Self>;

  /// A type that defines a commitment engine over scalars in the group
  type CE: CommitmentEngineTrait<Self>;
}

/// A helper trait to absorb different objects in RO
pub trait AbsorbInROTrait<E: Engine> {
  /// Absorbs the value in the provided RO
  fn absorb_in_ro(&self, ro: &mut E::RO);
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
  fn squeeze<CS: ConstraintSystem<Base>>(
    &mut self,
    cs: CS,
    num_bits: usize,
  ) -> Result<Vec<AllocatedBit>, SynthesisError>;
}

/// An alias for constants associated with E::RO
pub type ROConstants<E> =
  <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants;

/// An alias for constants associated with `E::ROCircuit`
pub type ROConstantsCircuit<E> =
  <<E as Engine>::ROCircuit as ROCircuitTrait<<E as Engine>::Base>>::Constants;

/// This trait allows types to implement how they want to be added to `TranscriptEngine`
pub trait TranscriptReprTrait<G: Group>: Send + Sync {
  /// returns a byte representation of self to be added to the transcript
  fn to_transcript_bytes(&self) -> Vec<u8>;
}

/// This trait defines the behavior of a transcript engine compatible with Spartan
pub trait TranscriptEngineTrait<E: Engine>: Send + Sync {
  /// initializes the transcript
  fn new(label: &'static [u8]) -> Self;

  /// returns a scalar element of the group as a challenge
  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, NovaError>;

  /// absorbs any type that implements `TranscriptReprTrait` under a label
  fn absorb<T: TranscriptReprTrait<E::GE>>(&mut self, label: &'static [u8], o: &T);

  /// adds a domain separator
  fn dom_sep(&mut self, bytes: &'static [u8]);
}

/// Defines additional methods on `PrimeField` objects
pub trait PrimeFieldExt: PrimeField {
  /// Returns a scalar representing the bytes
  fn from_uniform(bytes: &[u8]) -> Self;
}

impl<G: Group, T: TranscriptReprTrait<G>> TranscriptReprTrait<G> for &[T] {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self
      .iter()
      .flat_map(|t| t.to_transcript_bytes())
      .collect::<Vec<u8>>()
  }
}

pub mod circuit;
pub mod evaluation;
pub mod snark;
