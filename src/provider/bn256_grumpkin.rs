//! This module implements the Nova traits for `bn256::Point`, `bn256::Scalar`, `grumpkin::Point`, `grumpkin::Scalar`.
use crate::{
  impl_traits,
  provider::{
    msm::{msm, msm_small, msm_small_with_max_num_bits},
    traits::{DlogGroup, DlogGroupExt, PairingGroup},
  },
  traits::{Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{Field, FromUniformBytes};
use halo2curves::{
  bn256::{Bn256, G1Affine as Bn256Affine, G2Affine, G2Compressed, Gt, G1 as Bn256Point, G2},
  group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup},
  grumpkin::{G1Affine as GrumpkinAffine, G1 as GrumpkinPoint},
  pairing::Engine as H2CEngine,
  CurveAffine, CurveExt,
};
use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{Num, ToPrimitive};
use rayon::prelude::*;
use sha3::Shake256;

/// Re-exports that give access to the standard aliases used in the code base, for bn256
pub mod bn256 {
  pub use halo2curves::bn256::{Fq as Base, Fr as Scalar, G1Affine as Affine, G1 as Point};
}

/// Re-exports that give access to the standard aliases used in the code base, for grumpkin
pub mod grumpkin {
  pub use halo2curves::grumpkin::{Fq as Base, Fr as Scalar, G1Affine as Affine, G1 as Point};
}

crate::impl_traits_no_dlog_ext!(
  bn256,
  Bn256Point,
  Bn256Affine,
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001",
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
);

impl DlogGroupExt for bn256::Point {
  #[cfg(not(feature = "blitzar"))]
  fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self {
    msm(scalars, bases)
  }

  fn vartime_multiscalar_mul_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    scalars: &[T],
    bases: &[Self::AffineGroupElement],
  ) -> Self {
    msm_small(scalars, bases)
  }

  fn vartime_multiscalar_mul_small_with_max_num_bits<
    T: Integer + Into<u64> + Copy + Sync + ToPrimitive,
  >(
    scalars: &[T],
    bases: &[Self::AffineGroupElement],
    max_num_bits: usize,
  ) -> Self {
    msm_small_with_max_num_bits(scalars, bases, max_num_bits)
  }

  #[cfg(feature = "blitzar")]
  fn vartime_multiscalar_mul(scalars: &[Self::Scalar], bases: &[Self::AffineGroupElement]) -> Self {
    super::blitzar::vartime_multiscalar_mul(scalars, bases)
  }

  #[cfg(feature = "blitzar")]
  fn batch_vartime_multiscalar_mul(
    scalars: &[Vec<Self::Scalar>],
    bases: &[Self::AffineGroupElement],
  ) -> Vec<Self> {
    super::blitzar::batch_vartime_multiscalar_mul(scalars, bases)
  }
}

impl_traits!(
  grumpkin,
  GrumpkinPoint,
  GrumpkinAffine,
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
);

impl PairingGroup for Bn256Point {
  type G2 = G2;
  type GT = Gt;

  fn pairing(p: &Self, q: &Self::G2) -> Self::GT {
    <Bn256 as H2CEngine>::pairing(&p.affine(), &q.affine())
  }
}

impl Group for G2 {
  type Base = bn256::Base;
  type Scalar = bn256::Scalar;

  fn group_params() -> (Self::Base, Self::Base, BigInt, BigInt) {
    // G2 uses a quadratic extension field, so A and B are in QuadExtField<Fq>
    // We need to extract the constant terms that can be represented in Fq
    // For BN256 G2, the curve equation is y^2 = x^3 + Ax + B where A and B are in Fq2
    // The constant terms (real parts) are typically 0 for A and 3 for B
    let A = bn256::Base::ZERO; // Constant term of A in Fq2
    let B = bn256::Base::from(3u64); // Constant term of B in Fq2
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
  type AffineGroupElement = G2Affine;

  fn affine(&self) -> Self::AffineGroupElement {
    self.to_affine()
  }

  fn group(p: &Self::AffineGroupElement) -> Self {
    G2::from(*p)
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

impl<G: DlogGroup> TranscriptReprTrait<G> for G2Affine {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    unimplemented!()
  }
}
