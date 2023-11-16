//! This module implements the Nova traits for `pallas::Point`, `pallas::Scalar`, `vesta::Point`, `vesta::Scalar`.
use crate::{
  provider::{
    cpu_best_multiexp,
    keccak::Keccak256Transcript,
    pedersen::CommitmentEngine,
    poseidon::{PoseidonRO, PoseidonROCircuit},
    CompressedGroup, GroupExt,
  },
  traits::{Engine, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::{
  self,
  arithmetic::{CurveAffine, CurveExt},
  group::{cofactor::CofactorCurveAffine, Curve, Group, GroupEncoding},
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

struct PallasEngine;
struct VestaEngine;

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

      fn get_curve_params() -> (Self::Base, Self::Base, BigInt, BigInt) {
        let A = $name::Point::a();
        let B = $name::Point::b();
        let order = BigInt::from_str_radix($order_str, 16).unwrap();
        let base = BigInt::from_str_radix($base_str, 16).unwrap();

        (A, B, order, base)
      }
    }

    impl GroupExt for $name::Point {
      type CompressedGroupElement = $name_compressed;
      type PreprocessedGroupElement = $name::Affine;

      #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
      fn vartime_multiscalar_mul(
        scalars: &[Self::Scalar],
        bases: &[Self::PreprocessedGroupElement],
      ) -> Self {
        if scalars.len() >= 128 {
          pasta_msm::$name(bases, scalars)
        } else {
          cpu_best_multiexp(scalars, bases)
        }
      }

      #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
      fn vartime_multiscalar_mul(
        scalars: &[Self::Scalar],
        bases: &[Self::PreprocessedGroupElement],
      ) -> Self {
        cpu_best_multiexp(scalars, bases)
      }

      fn compress(&self) -> Self::CompressedGroupElement {
        $name_compressed::new(self.to_bytes())
      }

      fn preprocessed(&self) -> Self::PreprocessedGroupElement {
        self.to_affine()
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

      fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
        let coordinates = self.to_affine().coordinates();
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

    impl<E: Engine> TranscriptReprTrait<E> for $name_compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.repr.to_vec()
      }
    }

    impl CompressedGroup for $name_compressed {
      type GroupElement = $name::Point;

      fn decompress(&self) -> Option<$name::Point> {
        Some($name_curve::from_bytes(&self.repr).unwrap())
      }
    }

    impl<E: Engine> TranscriptReprTrait<E> for $name::Scalar {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.to_repr().to_vec()
      }
    }
  };
}

impl_traits!(
  PallasEngine,
  pallas,
  PallasCompressedElementWrapper,
  Ep,
  EpAffine,
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001"
);

impl_traits!(
  VestaEngine,
  vesta,
  VestaCompressedElementWrapper,
  Eq,
  EqAffine,
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001"
);

#[cfg(test)]
mod tests {
  use super::*;
  type E = pasta_curves::pallas::Point;

  fn from_label_serial(label: &'static [u8], n: usize) -> Vec<EpAffine> {
    let mut shake = Shake256::default();
    shake.update(label);
    let mut reader = shake.finalize_xof();
    let mut ck = Vec::new();
    for _ in 0..n {
      let mut uniform_bytes = [0u8; 32];
      reader.read_exact(&mut uniform_bytes).unwrap();
      let hash = Ep::hash_to_curve("from_uniform_bytes");
      ck.push(hash(&uniform_bytes).to_affine());
    }
    ck
  }

  #[test]
  fn test_from_label() {
    let label = b"test_from_label";
    for n in [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1021,
    ] {
      let ck_par = <E as EngineExt>::from_label(label, n);
      let ck_ser = from_label_serial(label, n);
      assert_eq!(ck_par.len(), n);
      assert_eq!(ck_ser.len(), n);
      assert_eq!(ck_par, ck_ser);
    }
  }
}
