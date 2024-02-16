use crate::{
  errors::NovaError,
  traits::{commitment::ScalarMul, Group, TranscriptReprTrait},
};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Sub, SubAssign},
};
use serde::{Deserialize, Serialize};

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
  type GroupElement: DlogGroup;

  /// Decompresses the compressed group element
  fn decompress(&self) -> Result<Self::GroupElement, NovaError>;
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

/// A helper trait for references implementing group scalar multiplication.
pub trait ScalarMulOwned<Rhs, Output = Self>: for<'r> ScalarMul<&'r Rhs, Output> {}
impl<T, Rhs, Output> ScalarMulOwned<Rhs, Output> for T where T: for<'r> ScalarMul<&'r Rhs, Output> {}

/// A trait that defines extensions to the Group trait
pub trait DlogGroup:
  Group
  + Serialize
  + for<'de> Deserialize<'de>
  + GroupOps
  + GroupOpsOwned
  + ScalarMul<<Self as Group>::Scalar>
  + ScalarMulOwned<<Self as Group>::Scalar>
{
  /// A type representing the compressed version of the group element
  type CompressedGroupElement: CompressedGroup<GroupElement = Self>;

  /// A type representing preprocessed group element
  type AffineGroupElement: Clone
    + Debug
    + PartialEq
    + Eq
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>
    + TranscriptReprTrait<Self>;

  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self;

  /// Produce a vector of group elements using a static label
  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::AffineGroupElement>;

  /// Compresses the group element
  fn compress(&self) -> Self::CompressedGroupElement;

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

/// A trait that defines extensions to the DlogGroup trait, to be implemented for
/// elliptic curve groups that are pairing friendly
pub trait PairingGroup: DlogGroup {
  /// A type representing the second group
  type G2: DlogGroup<Scalar = Self::Scalar, Base = Self::Base>;

  /// A type representing the target group
  type GT: PartialEq + Eq;

  /// A method to compute a pairing
  fn pairing(p: &Self, q: &Self::G2) -> Self::GT;
}

/// This implementation behaves in ways specific to the halo2curves suite of curves in:
// - to_coordinates,
// - vartime_multiscalar_mul, where it does not call into accelerated implementations.
// A specific reimplementation exists for the pasta curves in their own module.
#[macro_export]
macro_rules! impl_traits {
  (
    $name:ident,
    $name_compressed:ident,
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
      type CompressedGroupElement = $name_compressed;
      type AffineGroupElement = $name::Affine;

      fn vartime_multiscalar_mul(
        scalars: &[Self::Scalar],
        bases: &[Self::AffineGroupElement],
      ) -> Self {
        best_multiexp(scalars, bases)
      }

      fn affine(&self) -> Self::AffineGroupElement {
        self.to_affine()
      }

      fn group(p: &Self::AffineGroupElement) -> Self {
        $name::Point::from(*p)
      }

      fn compress(&self) -> Self::CompressedGroupElement {
        self.to_bytes()
      }

      fn from_label(label: &'static [u8], n: usize) -> Vec<Self::AffineGroupElement> {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut reader = shake.finalize_xof();
        let mut uniform_bytes_vec = Vec::new();
        for _ in 0..n {
          let mut uniform_bytes = [0u8; 32];
          reader.read_exact(&mut uniform_bytes).unwrap();
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

    impl<G: DlogGroup> TranscriptReprTrait<G> for $name_compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.as_ref().to_vec()
      }
    }

    impl CompressedGroup for $name_compressed {
      type GroupElement = $name::Point;

      fn decompress(&self) -> Result<$name::Point, NovaError> {
        let d = $name_curve::from_bytes(&self);
        if d.is_some().into() {
          Ok(d.unwrap())
        } else {
          Err(NovaError::DecompressionError)
        }
      }
    }

    impl<G: Group> TranscriptReprTrait<G> for $name::Scalar {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.to_repr().to_vec()
      }
    }

    impl<G: DlogGroup> TranscriptReprTrait<G> for $name::Affine {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        let coords = self.coordinates().unwrap();

        [coords.x().to_repr(), coords.y().to_repr()].concat()
      }
    }
  };
}
