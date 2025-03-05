//! This module defines various traits required by the users of the library to implement.
use crate::{
  errors::NovaError,
  frontend::{num::AllocatedNum, AllocatedBit, ConstraintSystem, SynthesisError},
};
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
  type Base: PrimeFieldBits
    + TranscriptReprTrait
    + Serialize
    + for<'de> Deserialize<'de>
    + ReprTrait<Self::Base>
    + ReprTrait<Self::Scalar>;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeFieldBits
    + PrimeFieldExt
    + Send
    + Sync
    + TranscriptReprTrait
    + Serialize
    + for<'de> Deserialize<'de>
    + ReprTrait<Self::Base>
    + ReprTrait<Self::Scalar>;

  /// A type that represents an element of the group
  type GE: Group<Base = Self::Base, Scalar = Self::Scalar> + Serialize + for<'de> Deserialize<'de>;

  /// A type that represents a circuit-friendly sponge that consumes
  /// elements from the base field
  type RO: ROTrait<Self::Base>;

  /// An alternate implementation of `Self::RO` in the circuit model
  type ROCircuit: ROCircuitTrait<Self::Base>;

  /// A type that represents a circuit-friendly sponge that consumes
  /// elements from the scalar field
  type RO2: ROTrait<Self::Scalar>;

  /// An alternate implementation of `Self::RO2` in the circuit model
  type RO2Circuit: ROCircuitTrait<Self::Scalar>;

  /// A type that provides a generic Fiat-Shamir transcript to be used when externalizing proofs
  type TE: TranscriptEngineTrait<Self>;

  /// A type that defines a commitment engine over scalars in the group
  type CE: CommitmentEngineTrait<Self>;
}

/// This trait allows types to implement how they want to be added to `RO`
pub trait ReprTrait<F: PrimeField>: Send + Sync {
  /// returns a representation of self as a vector of base field elements
  fn to_vec(&self) -> Vec<F>;
}

/// A helper trait that defines the behavior of a hash function that we use as an RO
pub trait ROTrait<F: PrimeField> {
  /// The circuit alter ego of this trait impl - this constrains it to use the same constants
  type CircuitRO: ROCircuitTrait<F, Constants = Self::Constants>;

  /// A type representing constants/parameters associated with the hash function
  type Constants: Default + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Initializes the hash function
  fn new(constants: Self::Constants) -> Self;

  /// Adds a scalar to the internal state
  fn absorb<T: ReprTrait<F>>(&mut self, o: &T);

  /// Returns a challenge of `num_bits` by hashing the internal state
  fn squeeze(&mut self, num_bits: usize) -> F;
}

/// A helper trait that defines the behavior of a hash function that we use as an RO in the circuit model
pub trait ROCircuitTrait<F: PrimeField> {
  /// the vanilla alter ego of this trait - this constrains it to use the same constants
  type NativeRO: ROTrait<F, Constants = Self::Constants>;

  /// A type representing constants/parameters associated with the hash function on this Base field
  type Constants: Default + Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Initializes the hash function
  fn new(constants: Self::Constants) -> Self;

  /// Adds a scalar to the internal state
  fn absorb(&mut self, e: &AllocatedNum<F>);

  /// Returns a challenge of `num_bits` by hashing the internal state
  fn squeeze<CS: ConstraintSystem<F>>(
    &mut self,
    cs: CS,
    num_bits: usize,
  ) -> Result<Vec<AllocatedBit>, SynthesisError>;
}

/// An alias for constants associated with E::RO
pub type ROConstants<E> = <<E as Engine>::RO as ROTrait<<E as Engine>::Base>>::Constants;

/// An alias for constants associated with `E::ROCircuit`
pub type ROConstantsCircuit<E> =
  <<E as Engine>::ROCircuit as ROCircuitTrait<<E as Engine>::Base>>::Constants;

/// An alias for constants associated with E::RO2
//pub type RO2Constants<E> = <<E as Engine>::RO as ROTrait<<E as Engine>::Scalar>>::Constants;

/// An alias for constants associated with `E::RO2Circuit`
//pub type RO2ConstantsCircuit<E> =
//  <<E as Engine>::ROCircuit as ROCircuitTrait<<E as Engine>::Scalar>>::Constants;

/// This trait allows types to implement how they want to be added to `TranscriptEngine`
pub trait TranscriptReprTrait: Send + Sync {
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
  fn absorb<T: TranscriptReprTrait>(&mut self, label: &'static [u8], o: &T);

  /// adds a domain separator
  fn dom_sep(&mut self, bytes: &'static [u8]);
}

/// Defines additional methods on `PrimeField` objects
pub trait PrimeFieldExt: PrimeField {
  /// Returns a scalar representing the bytes
  fn from_uniform(bytes: &[u8]) -> Self;
}

/*impl<E: Engine> ReprTrait<E::Base> for E::Base {
  fn to_vec(&self) -> Vec<E::Base> {
    vec![*self]
  }
}

impl<E: Engine> ReprTrait<E::Scalar> for E::Base {
  fn to_vec(&self) -> Vec<E::Scalar> {
    let mut v = Vec::new();
    // analyze self in bignum format
    let limbs: Vec<E::Scalar> = nat_to_limbs(&f_to_nat(self), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
    for limb in limbs {
      v.push(scalar_as_base::<E>(limb));
    }
    v
  }
}*/

impl<T: TranscriptReprTrait> TranscriptReprTrait for &[T] {
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
