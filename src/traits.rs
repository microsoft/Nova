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
  + Mul<<Self as Group>::Scalar, Output = Self>
  + MulAssign<<Self as Group>::Scalar>
  + for<'a> MulAssign<&'a <Self as Group>::Scalar>
  + for<'a> Mul<&'a <Self as Group>::Scalar, Output = Self>
  + Add<Self, Output = Self>
  + AddAssign<Self>
  + for<'a> AddAssign<&'a Self>
  + for<'a> Add<&'a Self, Output = Self>
{
  type Scalar: PrimeField + ChallengeTrait;
  type CompressedGroupElement: CompressedGroup;

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
