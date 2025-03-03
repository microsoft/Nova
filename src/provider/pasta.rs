#![cfg(feature = "std")]
//! This module implements the Nova traits for `pallas::Point`, `pallas::Scalar`, `vesta::Point`, `vesta::Scalar`.
use crate::{
  impl_traits,
  provider::traits::DlogGroup,
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::FromUniformBytes;
// TODO -> Make halo2curves be optional (use only on STD)
use halo2curves::{
  group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup},
  msm::msm_best,
  pasta::{Pallas, PallasAffine, Vesta, VestaAffine},
  CurveAffine, CurveExt,
};
use num_bigint::BigInt;
use num_traits::Num;
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

/// Re-exports that give access to the standard aliases used in the code base, for pallas
pub mod pallas {
  // ! TODO Here make halo2curves only on STD and pasta_curves on NO_STD instead of having 2 separate files
  pub use halo2curves::pasta::{Fp as Base, Fq as Scalar, Pallas as Point, PallasAffine as Affine};
}

/// Re-exports that give access to the standard aliases used in the code base, for vesta
pub mod vesta {
  pub use halo2curves::pasta::{Fp as Scalar, Fq as Base, Vesta as Point, VestaAffine as Affine};
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
