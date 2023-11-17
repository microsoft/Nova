//! This module implements Nova's traits using the following configuration:
//! `CommitmentEngine` with Pedersen's commitments
//! `Engine` with pasta curves and BN256/Grumpkin
//! `RO` traits with Poseidon
//! `EvaluationEngine` with an IPA-based polynomial evaluation argument
use crate::traits::{commitment::ScalarMul, Group, TranscriptReprTrait};
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
  fn decompress(&self) -> Option<Self::GroupElement>;
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
  type PreprocessedGroupElement: Clone
    + Debug
    + PartialEq
    + Eq
    + Send
    + Sync
    + Serialize
    + for<'de> Deserialize<'de>;

  /// A method to compute a multiexponentation
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self;

  /// Produce a vector of group elements using a static label
  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement>;

  /// Compresses the group element
  fn compress(&self) -> Self::CompressedGroupElement;

  /// Produces a preprocessed element
  fn preprocessed(&self) -> Self::PreprocessedGroupElement;

  /// Returns an element that is the additive identity of the group
  fn zero() -> Self;

  /// Returns the affine coordinates (x, y, infinty) for the point
  fn to_coordinates(&self) -> (<Self as Group>::Base, <Self as Group>::Base, bool);
}

pub mod bn256_grumpkin;
pub mod ipa_pc;
pub mod keccak;
pub mod pasta;
pub mod pedersen;
pub mod poseidon;
pub mod secp_secq;

use ff::PrimeField;
use pasta_curves::{self, arithmetic::CurveAffine, group::Group as AnotherGroup};
use rayon::{current_num_threads, prelude::*};

/// Native implementation of fast multiexp
/// Adapted from zcash/halo2
fn cpu_multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  let c = if bases.len() < 4 {
    1
  } else if bases.len() < 32 {
    3
  } else {
    (f64::from(bases.len() as u32)).ln().ceil() as usize
  };

  fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
    let skip_bits = segment * c;
    let skip_bytes = skip_bits / 8;

    if skip_bytes >= 32 {
      return 0;
    }

    let mut v = [0; 8];
    for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
      *v = *o;
    }

    let mut tmp = u64::from_le_bytes(v);
    tmp >>= skip_bits - (skip_bytes * 8);
    tmp %= 1 << c;

    tmp as usize
  }

  let segments = (256 / c) + 1;

  (0..segments)
    .rev()
    .fold(C::Curve::identity(), |mut acc, segment| {
      (0..c).for_each(|_| acc = acc.double());

      #[derive(Clone, Copy)]
      enum Bucket<C: CurveAffine> {
        None,
        Affine(C),
        Projective(C::Curve),
      }

      impl<C: CurveAffine> Bucket<C> {
        fn add_assign(&mut self, other: &C) {
          *self = match *self {
            Bucket::None => Bucket::Affine(*other),
            Bucket::Affine(a) => Bucket::Projective(a + *other),
            Bucket::Projective(a) => Bucket::Projective(a + other),
          }
        }

        fn add(self, other: C::Curve) -> C::Curve {
          match self {
            Bucket::None => other,
            Bucket::Affine(a) => other + a,
            Bucket::Projective(a) => other + a,
          }
        }
      }

      let mut buckets = vec![Bucket::None; (1 << c) - 1];

      for (coeff, base) in coeffs.iter().zip(bases.iter()) {
        let coeff = get_at::<C::Scalar>(segment, c, &coeff.to_repr());
        if coeff != 0 {
          buckets[coeff - 1].add_assign(base);
        }
      }

      // Summation by parts
      // e.g. 3a + 2b + 1c = a +
      //                    (a) + b +
      //                    ((a) + b) + c
      let mut running_sum = C::Curve::identity();
      for exp in buckets.into_iter().rev() {
        running_sum = exp.add(running_sum);
        acc += &running_sum;
      }
      acc
    })
}

/// Performs a multi-exponentiation operation without GPU acceleration.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
/// Adapted from zcash/halo2
pub(crate) fn cpu_best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());

  let num_threads = current_num_threads();
  if coeffs.len() > num_threads {
    let chunk = coeffs.len() / num_threads;
    coeffs
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(coeffs, bases)| cpu_multiexp_serial(coeffs, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    cpu_multiexp_serial(coeffs, bases)
  }
}

/// Curve ops
/// This implementation behaves in ways specific to the halo2curves suite of curves in:
// - to_coordinates,
// - vartime_multiscalar_mul, where it does not call into accelerated implementations.
// A specific reimplementation exists for the pasta curves in their own module.
#[macro_export]
macro_rules! impl_traits {
  (
    $engine:ident,
    $name:ident,
    $name_compressed:ident,
    $name_curve:ident,
    $name_curve_affine:ident,
    $order_str:literal,
    $base_str:literal
  ) => {
    impl Engine for $engine {
      type Base = $name::Base;
      type Scalar = $name::Scalar;
      type GE = $name::Point;
      type RO = PoseidonRO<Self::Base, Self::Scalar>;
      type ROCircuit = PoseidonROCircuit<Self::Base>;
      type TE = Keccak256Transcript<Self>;
      type CE = CommitmentEngine<Self>;
    }

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
      type PreprocessedGroupElement = $name::Affine;

      fn vartime_multiscalar_mul(
        scalars: &[Self::Scalar],
        bases: &[Self::PreprocessedGroupElement],
      ) -> Self {
        cpu_best_multiexp(scalars, bases)
      }

      fn preprocessed(&self) -> Self::PreprocessedGroupElement {
        self.to_affine()
      }

      fn compress(&self) -> Self::CompressedGroupElement {
        self.to_bytes()
      }

      fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement> {
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

      fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
        // see: grumpkin implementation at src/provider/bn256_grumpkin.rs
        let coordinates = self.to_affine().coordinates();
        if coordinates.is_some().unwrap_u8() == 1
          && ($name_curve_affine::identity() != self.to_affine())
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

      fn decompress(&self) -> Option<$name::Point> {
        Some($name_curve::from_bytes(&self).unwrap())
      }
    }

    impl<G: Group> TranscriptReprTrait<G> for $name::Scalar {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.to_repr().to_vec()
      }
    }
  };
}

#[cfg(test)]
mod tests {
  use super::cpu_best_multiexp;

  use crate::provider::{
    bn256_grumpkin::{bn256, grumpkin},
    secp_secq::{secp256k1, secq256k1},
  };
  use group::{ff::Field, Group};
  use halo2curves::CurveAffine;
  use pasta_curves::{pallas, vesta};
  use rand_core::OsRng;

  fn test_msm_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();
    let naive = coeffs
      .iter()
      .zip(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let msm = cpu_best_multiexp(&coeffs, &bases);

    assert_eq!(naive, msm)
  }

  #[test]
  fn test_msm() {
    test_msm_with::<pallas::Scalar, pallas::Affine>();
    test_msm_with::<vesta::Scalar, vesta::Affine>();
    test_msm_with::<bn256::Scalar, bn256::Affine>();
    test_msm_with::<grumpkin::Scalar, grumpkin::Affine>();
    test_msm_with::<secp256k1::Scalar, secp256k1::Affine>();
    test_msm_with::<secq256k1::Scalar, secq256k1::Affine>();
  }
}
