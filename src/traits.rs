//! This module defines various traits required by the users of the library to implement.
use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign},
};
use ff::{PrimeField, PrimeFieldBits};
use merlin::Transcript;
use rug::Integer;

/// Represents an element of a group
pub trait Group:
  Clone
  + Copy
  + Debug
  + Eq
  + Sized
  + GroupOps
  + GroupOpsOwned
  + ScalarMul<<Self as Group>::Scalar>
  + ScalarMulOwned<<Self as Group>::Scalar>
{
  /// A type representing an element of the base field of the group
  type Base: PrimeField + PrimeFieldBits;

  /// A type representing an element of the scalar field of the group
  type Scalar: PrimeField + PrimeFieldBits + ChallengeTrait;

  /// A type representing the compressed version of the group element
  type CompressedGroupElement: CompressedGroup<GroupElement = Self>;

  /// A type representing preprocessed group element
  type PreprocessedGroupElement;

  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self;

  /// Compresses the group element
  fn compress(&self) -> Self::CompressedGroupElement;

  /// Attempts to create a group element from a sequence of bytes,
  /// failing with a `None` if the supplied bytes do not encode the group element
  fn from_uniform_bytes(bytes: &[u8]) -> Option<Self::PreprocessedGroupElement>;

  /// Returns the affine coordinates (x, y, infinty) for the point
  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool);

  /// Returns the order of the group as a big integer
  fn get_order() -> Integer;
}

/// Represents a compressed version of a group element
pub trait CompressedGroup: Clone + Copy + Debug + Eq + Sized + Send + Sync + 'static {
  /// A type that holds the decompressed version of the compressed group element
  type GroupElement: Group;

  /// Decompresses the compressed group element
  fn decompress(&self) -> Option<Self::GroupElement>;

  /// Returns a byte array representing the compressed group element
  fn as_bytes(&self) -> &[u8];
}

/// A helper trait to generate challenges using a transcript object
pub trait ChallengeTrait {
  /// Returns a Scalar representing the challenge using the transcript
  fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self;
}

/// A helper trait for types with a group operation.
pub trait GroupOps<Rhs = Self, Output = Self>:
  Add<Rhs, Output = Output> + Sub<Rhs, Output = Output> + AddAssign<Rhs> + SubAssign<Rhs>
{
}

impl<T, Rhs, Output> GroupOps<Rhs, Output> for T where
  T: Add<Rhs, Output = Output> + Sub<Rhs, Output = Output> + AddAssign<Rhs> + SubAssign<Rhs>
{
}

/// A helper trait for references with a group operation.
pub trait GroupOpsOwned<Rhs = Self, Output = Self>: for<'r> GroupOps<&'r Rhs, Output> {}
impl<T, Rhs, Output> GroupOpsOwned<Rhs, Output> for T where T: for<'r> GroupOps<&'r Rhs, Output> {}

/// A helper trait for types implementing group scalar multiplication.
pub trait ScalarMul<Rhs, Output = Self>: Mul<Rhs, Output = Output> + MulAssign<Rhs> {}

impl<T, Rhs, Output> ScalarMul<Rhs, Output> for T where T: Mul<Rhs, Output = Output> + MulAssign<Rhs>
{}

/// A helper trait for references implementing group scalar multiplication.
pub trait ScalarMulOwned<Rhs, Output = Self>: for<'r> ScalarMul<&'r Rhs, Output> {}
impl<T, Rhs, Output> ScalarMulOwned<Rhs, Output> for T where T: for<'r> ScalarMul<&'r Rhs, Output> {}

/// A helper trait for a step of the incremental computation (i.e., circuit for F)
pub trait StepCircuit<F: PrimeField> {
  /// Sythesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step z_{i+1}
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError>;
}
