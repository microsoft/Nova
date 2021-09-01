use core::borrow::Borrow;
use core::fmt::Debug;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use merlin::Transcript;
use rand::{CryptoRng, RngCore};

/// Represents an element of a prime field
pub trait PrimeField:
  Sized
  + Eq
  + Copy
  + Clone
  + Default
  + Send
  + Sync
  + Debug
  + Add<Output = Self>
  + Sub<Output = Self>
  + Mul<Output = Self>
  + Neg<Output = Self>
  + for<'a> Add<&'a Self, Output = Self>
  + for<'a> Mul<&'a Self, Output = Self>
  + for<'a> Sub<&'a Self, Output = Self>
  + AddAssign
  + MulAssign
  + SubAssign
  + for<'a> AddAssign<&'a Self>
  + for<'a> MulAssign<&'a Self>
  + for<'a> SubAssign<&'a Self>
{
  /// returns the additive identity of the field
  fn zero() -> Self;

  /// returns the multiplicative identity of the field
  fn one() -> Self;

  /// converts the supplied bytes into an element of the field
  fn from_bytes_mod_order_wide(bytes: &[u8]) -> Option<Self>;

  /// returns an uniformly random element from the finite field
  fn random(rng: &mut (impl RngCore + CryptoRng)) -> Self;
}

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
  type Scalar: PrimeField + ChallengeTrait;
  type CompressedGroupElement: CompressedGroup<GroupElement = Self>;

  fn vartime_multiscalar_mul<I, J>(scalars: I, points: J) -> Self
  where
    I: IntoIterator,
    I::Item: Borrow<Self::Scalar>,
    J: IntoIterator,
    J::Item: Borrow<Self>,
    Self: Clone;

  fn compress(&self) -> Self::CompressedGroupElement;

  fn from_uniform_bytes(bytes: &[u8]) -> Option<Self>;
}

/// Represents a compressed version of a group element
pub trait CompressedGroup: Clone + Copy + Debug + Eq + Sized + Send + Sync + 'static {
  type GroupElement: Group;

  fn decompress(&self) -> Option<Self::GroupElement>;

  fn as_bytes(&self) -> &[u8];
}

/// A helper trait to generate challenges using a transcript object
pub trait ChallengeTrait {
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
