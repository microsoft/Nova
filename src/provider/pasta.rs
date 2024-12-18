//! This module implements the Nova traits for `pallas::Point`, `pallas::Scalar`, `vesta::Point`, `vesta::Scalar`.
use crate::{
  impl_traits, impl_transcript_traits_pasta,
  provider::traits::DlogGroup,
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup};
use halo2curves::{
  msm::msm_best,
  pasta::{
    pallas::{Affine as PallasAffine, Point as Pallas},
    vesta::{Affine as VestaAffine, Point as Vesta},
  },
  CurveAffine, CurveExt,
};
use num_bigint::BigInt;
use num_traits::Num;
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

/// Re-exports that give access to the standard aliases used in the code base, for pallas
pub mod pallas {
  pub use halo2curves::pasta::{pallas::Affine, pallas::Point, Fp as Base, Fq as Scalar};
}

/// Re-exports that give access to the standard aliases used in the code base, for vesta
pub mod vesta {
  pub use halo2curves::pasta::{vesta::Affine, vesta::Point, Fp as Scalar, Fq as Base};
}

impl_traits!(
  pallas,
  Pallas,
  PallasAffine,
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001"
);

impl_traits!(
  vesta,
  Vesta,
  VestaAffine,
  "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
  "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001"
);

impl_transcript_traits_pasta!(pallas,);

impl_transcript_traits_pasta!(vesta,);
