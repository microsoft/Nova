//! This module implements the Nova traits for pallas::Point and pallas::Scalar.
use crate::traits::{ChallengeTrait, CompressedGroup, Group, PrimeField};
use core::borrow::Borrow;
use core::ops::Mul;
use merlin::Transcript;
use pasta_curves::arithmetic::{CurveExt, Field, FieldExt, Group as Grp};
use pasta_curves::group::{Group as GrpTrait, GroupEncoding};
use pasta_curves::{self, pallas, Ep, Fq};
use rand::{CryptoRng, RngCore};

impl Group for pallas::Point {
  type Scalar = pallas::Scalar;
  type CompressedGroupElement = <pallas::Point as GroupEncoding>::Repr;

  fn vartime_multiscalar_mul<I, J>(scalars: I, points: J) -> Self
  where
    I: IntoIterator,
    I::Item: Borrow<Self::Scalar>,
    J: IntoIterator,
    J::Item: Borrow<Self>,
    Self: Clone,
  {
    // Unoptimized.
    scalars
      .into_iter()
      .zip(points)
      .map(|(scalar, point)| (*point.borrow()).mul(*scalar.borrow()))
      .fold(Ep::group_zero(), |acc, x| acc + x)
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    self.to_bytes()
  }

  fn from_uniform_bytes(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 32];
      arr.copy_from_slice(&bytes[0..32]);

      let hash = Ep::hash_to_curve("from_uniform_bytes");
      Some(hash(&arr))
    }
  }

  fn gen() -> Self {
    pallas::Point::generator()
  }
}

impl PrimeField for pallas::Scalar {
  fn zero() -> Self {
    Fq::zero()
  }
  fn one() -> Self {
    Fq::one()
  }
  fn from_bytes_mod_order_wide(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 64];
      arr.copy_from_slice(&bytes[0..64]);
      Some(Fq::from_bytes_wide(&arr))
    }
  }

  fn random(rng: &mut (impl RngCore + CryptoRng)) -> Self {
    <Fq as ff::Field>::random(rng)
  }

  fn inverse(&self) -> Option<Self> {
    Some(self.invert().unwrap())
  }

  fn as_bytes(&self) -> Vec<u8> {
    self.to_bytes().to_vec()
  }
}

impl ChallengeTrait for pallas::Scalar {
  fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self {
    let mut buf = [0u8; 64];
    transcript.challenge_bytes(label, &mut buf);
    pallas::Scalar::from_bytes_mod_order_wide(&buf).unwrap()
  }
}

impl CompressedGroup for <pallas::Point as GroupEncoding>::Repr {
  type GroupElement = pallas::Point;
  fn decompress(&self) -> Option<<Self as CompressedGroup>::GroupElement> {
    Some(Ep::from_bytes(self).unwrap())
  }
  fn as_bytes(&self) -> &[u8] {
    self
  }
}
