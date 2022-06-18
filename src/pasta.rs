//! This module implements the Nova traits for pallas::Point, pallas::Scalar, vesta::Point, vesta::Scalar.
use crate::{
  poseidon::PoseidonRO,
  traits::{ChallengeTrait, CompressedGroup, Group},
};
use digest::{ExtendableOutput, Input};
use ff::Field;
use merlin::Transcript;
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::{
  self,
  arithmetic::{CurveAffine, CurveExt},
  group::{Curve, Group as Grp, GroupEncoding},
  pallas, vesta, Ep, Eq,
};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use sha3::Shake256;
use std::io::Read;
use std::ops::Mul;
use pasta_curves::arithmetic::Group as Grp2;

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
  type HashFunc = PoseidonRO<Self::Base, Self::Scalar>;

  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    //pasta_msm::pallas(bases, scalars)
    scalars
    .iter()
    .zip(bases)
    .map(|(scalar, point)| point.mul(scalar))
    .fold(Ep::group_zero(), |acc, x| acc + x)
  }

  fn preprocessed(&self) -> Self::PreprocessedGroupElement {
    self.to_affine()
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    PallasCompressedElementWrapper::new(self.to_bytes())
  }

  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement> {
    let mut shake = Shake256::default();
    shake.input(label);
    let mut reader = shake.xof_result();
    let mut gens: Vec<Self::PreprocessedGroupElement> = Vec::new();
    let mut uniform_bytes = [0u8; 32];
    for _ in 0..n {
      reader.read_exact(&mut uniform_bytes).unwrap();
      let hash = Ep::hash_to_curve("from_uniform_bytes");
      gens.push(hash(&uniform_bytes).to_affine());
    }
    gens
  }

  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
    let coordinates = self.to_affine().coordinates();
    if coordinates.is_some().unwrap_u8() == 1 {
      (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
    } else {
      (Self::Base::zero(), Self::Base::zero(), true)
    }
  }

  fn get_order() -> BigInt {
    BigInt::from_str_radix(
      "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
      16,
    )
    .unwrap()
  }

  fn gen() -> Self {
    pallas::Point::generator()
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
  type HashFunc = PoseidonRO<Self::Base, Self::Scalar>;

  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    // pasta_msm::vesta(bases, scalars)
    scalars
    .iter()
    .zip(bases)
    .map(|(scalar, point)| point.mul(scalar))
    .fold(Eq::group_zero(), |acc, x| acc + x)
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    VestaCompressedElementWrapper::new(self.to_bytes())
  }

  fn preprocessed(&self) -> Self::PreprocessedGroupElement {
    self.to_affine()
  }

  fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement> {
    let mut shake = Shake256::default();
    shake.input(label);
    let mut reader = shake.xof_result();
    let mut gens: Vec<Self::PreprocessedGroupElement> = Vec::new();
    let mut uniform_bytes = [0u8; 32];
    for _ in 0..n {
      reader.read_exact(&mut uniform_bytes).unwrap();
      let hash = Eq::hash_to_curve("from_uniform_bytes");
      gens.push(hash(&uniform_bytes).to_affine());
    }
    gens
  }

  fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
    let coordinates = self.to_affine().coordinates();
    if coordinates.is_some().unwrap_u8() == 1 {
      (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
    } else {
      (Self::Base::zero(), Self::Base::zero(), true)
    }
  }

  fn get_order() -> BigInt {
    BigInt::from_str_radix(
      "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
      16,
    )
    .unwrap()
  }

  fn gen() -> Self {
    vesta::Point::generator()
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
