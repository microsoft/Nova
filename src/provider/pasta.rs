//! This module implements the Nova traits for `pallas::Point`, `pallas::Scalar`, `vesta::Point`, `vesta::Scalar`.
use crate::{
  errors::NovaError,
  provider::traits::{CompressedGroup, DlogGroup},
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use halo2curves::msm::best_multiexp;
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::{
  self,
  arithmetic::{CurveAffine, CurveExt},
  group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup, GroupEncoding},
  pallas, vesta, Ep, EpAffine, Eq, EqAffine,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::Shake256;
use std::io::Read;

/// A wrapper for compressed group elements of pallas
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasCompressedElementWrapper {
  repr: [u8; 32],
}

impl PallasCompressedElementWrapper {
  /// Wraps repr into the wrapper
  pub const fn new(repr: [u8; 32]) -> Self {
    Self { repr }
  }
}

/// A wrapper for compressed group elements of vesta
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaCompressedElementWrapper {
  repr: [u8; 32],
}

impl VestaCompressedElementWrapper {
  /// Wraps repr into the wrapper
  pub const fn new(repr: [u8; 32]) -> Self {
    Self { repr }
  }
}

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
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        if scalars.len() >= 128 {
          pasta_msm::$name(bases, scalars)
        } else {
          best_multiexp(scalars, bases)
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        best_multiexp(scalars, bases)
      }

      fn affine(&self) -> Self::AffineGroupElement {
        self.to_affine()
      }

      fn group(p: &Self::AffineGroupElement) -> Self {
        $name::Point::from(*p)
      }

      fn compress(&self) -> Self::CompressedGroupElement {
        $name_compressed::new(self.to_bytes())
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
        let ck_proj: Vec<$name_curve> = (0..n)
          .into_par_iter()
          .map(|i| {
            let hash = $name_curve::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes_vec[i])
          })
          .collect();

        let num_threads = rayon::current_num_threads();
        if ck_proj.len() > num_threads {
          let chunk = (ck_proj.len() as f64 / num_threads as f64).ceil() as usize;
          (0..num_threads)
            .into_par_iter()
            .flat_map(|i| {
              let start = i * chunk;
              let end = if i == num_threads - 1 {
                ck_proj.len()
              } else {
                core::cmp::min((i + 1) * chunk, ck_proj.len())
              };
              if end > start {
                let mut ck = vec![$name_curve_affine::identity(); end - start];
                <Self as Curve>::batch_normalize(&ck_proj[start..end], &mut ck);
                ck
              } else {
                vec![]
              }
            })
            .collect()
        } else {
          let mut ck = vec![$name_curve_affine::identity(); n];
          <Self as Curve>::batch_normalize(&ck_proj, &mut ck);
          ck
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
        if coordinates.is_some().unwrap_u8() == 1 {
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

    impl CompressedGroup for $name_compressed {
      type GroupElement = $name::Point;

      fn decompress(&self) -> Result<$name::Point, NovaError> {
        let d = $name_curve::from_bytes(&self.repr);
        if d.is_some().into() {
          Ok(d.unwrap())
        } else {
          Err(NovaError::DecompressionError)
        }
      }
    }

    impl<G: DlogGroup> TranscriptReprTrait<G> for $name_compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.repr.to_vec()
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

impl_traits!(
  pallas,
  PallasCompressedElementWrapper,
  Ep,
  EpAffine,
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001"
);

impl_traits!(
  vesta,
  VestaCompressedElementWrapper,
  Eq,
  EqAffine,
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001"
);
