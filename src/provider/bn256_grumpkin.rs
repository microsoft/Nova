//! This module implements the Nova traits for `bn256::Point`, `bn256::Scalar`, `grumpkin::Point`, `grumpkin::Scalar`.
use crate::{
  errors::NovaError,
  impl_traits,
  provider::traits::{CompressedGroup, DlogGroup, PairingGroup},
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup, GroupEncoding};
use num_bigint::BigInt;
use num_traits::Num;
// Remove this when https://github.com/zcash/pasta_curves/issues/41 resolves
use halo2curves::{
  bn256::{
    pairing, G1Affine as Bn256Affine, G1Compressed as Bn256Compressed, G2Affine, G2Compressed, Gt,
    G1 as Bn256Point, G2,
  },
  grumpkin::{G1Affine as GrumpkinAffine, G1Compressed as GrumpkinCompressed, G1 as GrumpkinPoint},
  msm::best_multiexp,
};
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

/// Re-exports that give access to the standard aliases used in the code base, for bn256
pub mod bn256 {
  pub use halo2curves::bn256::{Fq as Base, Fr as Scalar, G1Affine as Affine, G1 as Point};
}

/// Re-exports that give access to the standard aliases used in the code base, for grumpkin
pub mod grumpkin {
  pub use halo2curves::grumpkin::{Fq as Base, Fr as Scalar, G1Affine as Affine, G1 as Point};
}

impl_traits!(
  bn256,
  Bn256Compressed,
  Bn256Point,
  Bn256Affine,
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
);

impl_traits!(
  grumpkin,
  GrumpkinCompressed,
  GrumpkinPoint,
  GrumpkinAffine,
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
);

impl PairingGroup for Bn256Point {
  type G2 = G2;
  type GT = Gt;

  fn pairing(p: &Self, q: &Self::G2) -> Self::GT {
    pairing(&p.affine(), &q.affine())
  }
}

impl Group for G2 {
  type Base = bn256::Base;
  type Scalar = bn256::Scalar;

  fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt) {
    let A = bn256::Point::a();
    let B = bn256::Point::b();
    let order = BigInt::from_str_radix(
      "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
      16,
    )
    .unwrap();
    let base = BigInt::from_str_radix(
      "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
      16,
    )
    .unwrap();

    (A, B, order, base)
  }
}

impl DlogGroup for G2 {
  type CompressedGroupElement = G2Compressed;
  type AffineGroupElement = G2Affine;

  fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self {
    best_multiexp(scalars, bases)
  }

  fn affine(&self) -> Self::AffineGroupElement {
    self.to_affine()
  }

  fn group(p: &Self::AffineGroupElement) -> Self {
    G2::from(*p)
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    self.to_bytes()
  }

  fn from_label(_label: &'static [u8], _n: usize) -> Vec<Self::AffineGroupElement> {
    unimplemented!()
  }

  fn zero() -> Self {
    G2::identity()
  }

  fn gen() -> Self {
    G2::generator()
  }

  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
    unimplemented!()
  }
}

impl<G: DlogGroup> TranscriptReprTrait<G> for G2Compressed {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.as_ref().to_vec()
  }
}

impl CompressedGroup for G2Compressed {
  type GroupElement = G2;

  fn decompress(&self) -> Result<G2, NovaError> {
    let d = G2::from_bytes(self);
    if d.is_some().into() {
      Ok(d.unwrap())
    } else {
      Err(NovaError::DecompressionError)
    }
  }
}

impl<G: DlogGroup> TranscriptReprTrait<G> for G2Affine {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    unimplemented!()
  }
}
