//! This module implements the Nova traits for pallas::Point, pallas::Scalar, vesta::Point, vesta::Scalar.
use crate::traits::{ChallengeTrait, CompressedGroup, Group};
use core::ops::Mul;
use ff::Field;
use merlin::Transcript;
use pasta_curves::{
  self,
  arithmetic::{CurveAffine, CurveExt, Group as Grp},
  group::{Curve, GroupEncoding},
  pallas, vesta, Ep, Eq,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rug::Integer;

//////////////////////////////////////Pallas///////////////////////////////////////////////

/// A wrapper for compressed group elements that come from the pallas curve
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PallasCompressedElementWrapper {
  repr: [u8; 32],
}

impl PallasCompressedElementWrapper {
  /// Wraps repr into the wrapper
  pub fn new(repr: [u8; 32]) -> Self {
    Self { repr }
  }
}

impl Group for pallas::Point {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type CompressedGroupElement = PallasCompressedElementWrapper;
  type PreprocessedGroupElement = pallas::Affine;

  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    // Unoptimized.
    scalars
      .into_iter()
      .zip(bases)
      .map(|(scalar, base)| base.mul(scalar))
      .fold(Ep::group_zero(), |acc, x| acc + x)
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    PallasCompressedElementWrapper::new(self.to_bytes())
  }

  fn from_uniform_bytes(bytes: &[u8]) -> Option<Self::PreprocessedGroupElement> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 32];
      arr.copy_from_slice(&bytes[0..32]);

      let hash = Ep::hash_to_curve("from_uniform_bytes");
      Some(hash(&arr).to_affine())
    }
  }

  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
    let coordinates = self.to_affine().coordinates();
    if coordinates.is_some().unwrap_u8() == 1 {
      (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
    } else {
      (Self::Base::zero(), Self::Base::zero(), true)
    }
  }

  fn get_order() -> Integer {
    Integer::from_str_radix(
      "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
      16,
    )
    .unwrap()
  }
}

impl ChallengeTrait for pallas::Scalar {
  fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self {
    let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
    transcript.challenge_bytes(label, &mut key);
    let mut rng = ChaCha20Rng::from_seed(key);
    pallas::Scalar::random(&mut rng)
  }
}

impl CompressedGroup for PallasCompressedElementWrapper {
  type GroupElement = pallas::Point;

  fn decompress(&self) -> Option<pallas::Point> {
    Some(Ep::from_bytes(&self.repr).unwrap())
  }
  fn as_bytes(&self) -> &[u8] {
    &self.repr
  }
}

//////////////////////////////////////Vesta////////////////////////////////////////////////

/// A wrapper for compressed group elements that come from the vesta curve
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VestaCompressedElementWrapper {
  repr: [u8; 32],
}

impl VestaCompressedElementWrapper {
  /// Wraps repr into the wrapper
  pub fn new(repr: [u8; 32]) -> Self {
    Self { repr }
  }
}

impl Group for vesta::Point {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type CompressedGroupElement = VestaCompressedElementWrapper;
  type PreprocessedGroupElement = vesta::Affine;

  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    // Unoptimized.
    scalars
      .into_iter()
      .zip(bases)
      .map(|(scalar, base)| base.mul(scalar))
      .fold(Eq::group_zero(), |acc, x| acc + x)
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    VestaCompressedElementWrapper::new(self.to_bytes())
  }

  fn from_uniform_bytes(bytes: &[u8]) -> Option<Self::PreprocessedGroupElement> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 32];
      arr.copy_from_slice(&bytes[0..32]);

      let hash = Eq::hash_to_curve("from_uniform_bytes");
      Some(hash(&arr).to_affine())
    }
  }

  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
    let coordinates = self.to_affine().coordinates();
    if coordinates.is_some().unwrap_u8() == 1 {
      (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
    } else {
      (Self::Base::zero(), Self::Base::zero(), true)
    }
  }

  fn get_order() -> Integer {
    Integer::from_str_radix(
      "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
      16,
    )
    .unwrap()
  }
}

impl ChallengeTrait for vesta::Scalar {
  fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self {
    let mut key: <ChaCha20Rng as SeedableRng>::Seed = Default::default();
    transcript.challenge_bytes(label, &mut key);
    let mut rng = ChaCha20Rng::from_seed(key);
    vesta::Scalar::random(&mut rng)
  }
}

impl CompressedGroup for VestaCompressedElementWrapper {
  type GroupElement = vesta::Point;

  fn decompress(&self) -> Option<vesta::Point> {
    Some(Eq::from_bytes(&self.repr).unwrap())
  }
  fn as_bytes(&self) -> &[u8] {
    &self.repr
  }
}
