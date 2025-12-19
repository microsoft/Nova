use crate::traits::{commitment::ScalarMul, Group, TranscriptReprTrait};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Sub, SubAssign},
};
use halo2curves::{serde::SerdeObject, CurveAffine};
use num_integer::Integer;
use num_traits::ToPrimitive;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

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

/// A helper trait for references implementing group scalar multiplication.
pub trait ScalarMulOwned<Rhs, Output = Self>: for<'r> ScalarMul<&'r Rhs, Output> {}
impl<T, Rhs, Output> ScalarMulOwned<Rhs, Output> for T where T: for<'r> ScalarMul<&'r Rhs, Output> {}

/// A trait that defines the core discrete logarithm group functionality
pub trait DlogGroup:
  Group
  + Serialize
  + for<'de> Deserialize<'de>
  + GroupOps
  + GroupOpsOwned
  + ScalarMul<<Self as Group>::Scalar>
  + ScalarMulOwned<<Self as Group>::Scalar>
{
  /// A type representing preprocessed group element
  type AffineGroupElement: Clone
    + Debug
    + PartialEq
    + Eq
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>
    + TranscriptReprTrait<Self>
    + CurveAffine
    + SerdeObject;

  /// Produce a vector of group elements using a static label
  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::AffineGroupElement>;

  /// Produces a preprocessed element
  fn affine(&self) -> Self::AffineGroupElement;

  /// Returns a group element from a preprocessed group element
  fn group(p: &Self::AffineGroupElement) -> Self;

  /// Returns an element that is the additive identity of the group
  fn zero() -> Self;

  /// Returns the generator of the group
  fn gen() -> Self;

  /// Returns the affine coordinates (x, y, infinity) for the point
  fn to_coordinates(&self) -> (<Self as Group>::Base, <Self as Group>::Base, bool);
}

/// Extension trait for DlogGroup that provides multi-scalar multiplication operations
pub trait DlogGroupExt: DlogGroup {
  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self;

  /// A method to compute a batch of multiexponentations
  fn batch_vartime_multiscalar_mul(
    scalars: &[Vec<Self::Scalar>],
    bases: &[Self::AffineGroupElement],
  ) -> Vec<Self> {
    scalars
      .par_iter()
      .map(|scalar| Self::vartime_multiscalar_mul(scalar, &bases[..scalar.len()]))
      .collect::<Vec<_>>()
  }

  /// A method to compute a multiexponentation with small scalars
  fn vartime_multiscalar_mul_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    scalars: &[T],
    bases: &[Self::AffineGroupElement],
  ) -> Self;

  /// A method to compute a multiexponentation with small scalars
  fn vartime_multiscalar_mul_small_with_max_num_bits<
    T: Integer + Into<u64> + Copy + Sync + ToPrimitive,
  >(
    scalars: &[T],
    bases: &[Self::AffineGroupElement],
    max_num_bits: usize,
  ) -> Self;

  /// A method to compute a batch of multiexponentations with small scalars
  fn batch_vartime_multiscalar_mul_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    scalars: &[Vec<T>],
    bases: &[Self::AffineGroupElement],
  ) -> Vec<Self> {
    scalars
      .par_iter()
      .map(|scalar| Self::vartime_multiscalar_mul_small(scalar, &bases[..scalar.len()]))
      .collect::<Vec<_>>()
  }
}

/// A trait that defines extensions to the DlogGroup trait, to be implemented for
/// elliptic curve groups that are pairing friendly
pub trait PairingGroup: DlogGroupExt {
  /// A type representing the second group
  type G2: DlogGroup<Scalar = Self::Scalar, Base = Self::Base>;

  /// A type representing the target group
  type GT: PartialEq + Eq;

  /// A method to compute a pairing
  fn pairing(p: &Self, q: &Self::G2) -> Self::GT;
}

/// Implements Nova's traits except DlogGroupExt so that the MSM can be implemented differently
#[macro_export]
macro_rules! impl_traits_no_dlog_ext {
  (
    $name:ident,
    $name_curve:ident,
    $name_curve_affine:ident,
    $order_str:literal,
    $base_str:literal
  ) => {
    impl Group for $name::Point {
      type Base = $name::Base;
      type Scalar = $name::Scalar;

      fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt) {
        let A = $name::Point::a();
        let B = $name::Point::b();
        let order = BigInt::from_str_radix($order_str, 16).unwrap();
        let base = BigInt::from_str_radix($base_str, 16).unwrap();

        (A, B, order, base)
      }
    }

    impl DlogGroup for $name::Point {
      type AffineGroupElement = $name::Affine;

      fn affine(&self) -> Self::AffineGroupElement {
        self.to_affine()
      }

      fn group(p: &Self::AffineGroupElement) -> Self {
        $name::Point::from(*p)
      }

      fn from_label(label: &'static [u8], n: usize) -> Vec<Self::AffineGroupElement> {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut reader = shake.finalize_xof();
        let mut uniform_bytes_vec = Vec::new();
        for _ in 0..n {
          let mut uniform_bytes = [0u8; 32];
          digest::XofReader::read(&mut reader, &mut uniform_bytes);
          uniform_bytes_vec.push(uniform_bytes);
        }
        let gens_proj: Vec<$name_curve> = (0..n)
          .into_par_iter()
          .map(|i| {
            let hash = $name_curve::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes_vec[i])
          })
          .collect();

        let num_threads = rayon::current_num_threads();
        if gens_proj.len() > num_threads {
          let chunk = (gens_proj.len() as f64 / num_threads as f64).ceil() as usize;
          (0..num_threads)
            .into_par_iter()
            .flat_map(|i| {
              let start = i * chunk;
              let end = if i == num_threads - 1 {
                gens_proj.len()
              } else {
                core::cmp::min((i + 1) * chunk, gens_proj.len())
              };
              if end > start {
                let mut gens = vec![$name_curve_affine::identity(); end - start];
                <Self as Curve>::batch_normalize(&gens_proj[start..end], &mut gens);
                gens
              } else {
                vec![]
              }
            })
            .collect()
        } else {
          let mut gens = vec![$name_curve_affine::identity(); n];
          <Self as Curve>::batch_normalize(&gens_proj, &mut gens);
          gens
        }
      }

      fn zero() -> Self {
        $name::Point::identity()
      }

      fn gen() -> Self {
        $name::Point::generator()
      }

      fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
        let coordinates = self.affine().coordinates();
        if coordinates.is_some().unwrap_u8() == 1
          && ($name_curve_affine::identity() != self.affine())
        {
          (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
        } else {
          (Self::Base::zero(), Self::Base::zero(), true)
        }
      }
    }

    impl PrimeFieldExt for $name::Scalar {
      fn from_uniform(bytes: &[u8]) -> Self {
        let bytes_arr: [u8; 64] = bytes.try_into().unwrap();
        $name::Scalar::from_uniform_bytes(&bytes_arr)
      }
    }

    impl<G: Group> TranscriptReprTrait<G> for $name::Scalar {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.to_bytes().into_iter().rev().collect()
      }
    }

    impl<G: DlogGroup> TranscriptReprTrait<G> for $name::Affine {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        let coords = self.coordinates().unwrap();
        let x_bytes = coords.x().to_bytes().into_iter();
        let y_bytes = coords.y().to_bytes().into_iter();
        x_bytes.rev().chain(y_bytes.rev()).collect()
      }
    }
  };
}

/// Implements Nova's traits
#[macro_export]
macro_rules! impl_traits {
  (
    $name:ident,
    $name_curve:ident,
    $name_curve_affine:ident,
    $order_str:literal,
    $base_str:literal
  ) => {
    $crate::impl_traits_no_dlog_ext!(
      $name,
      $name_curve,
      $name_curve_affine,
      $order_str,
      $base_str
    );

    impl DlogGroupExt for $name::Point {
      fn vartime_multiscalar_mul(
        scalars: &[Self::Scalar],
        bases: &[Self::AffineGroupElement],
      ) -> Self {
        msm(scalars, bases)
      }

      fn vartime_multiscalar_mul_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
        scalars: &[T],
        bases: &[Self::AffineGroupElement],
      ) -> Self {
        msm_small(scalars, bases)
      }

      /// A method to compute a multiexponentation with small scalars
      fn vartime_multiscalar_mul_small_with_max_num_bits<
        T: Integer + Into<u64> + Copy + Sync + ToPrimitive,
      >(
        scalars: &[T],
        bases: &[Self::AffineGroupElement],
        max_num_bits: usize,
      ) -> Self {
        msm_small_with_max_num_bits(scalars, bases, max_num_bits)
      }
    }
  };
}
