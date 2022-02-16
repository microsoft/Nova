//! This module implements the Nova traits for pallas::Point, pallas::Scalar, vesta::Point, vesta::Scalar.
use crate::traits::{ChallengeTrait, CompressedGroup, Group, PrimeField};
use core::borrow::Borrow;
use core::ops::Mul;
use merlin::Transcript;
use pasta_curves::arithmetic::{CurveExt, Field, FieldExt, Group as Grp, CurveAffine};
use pasta_curves::group::{Group as GrpTrait, GroupEncoding, Curve};
use pasta_curves::{self, pallas, vesta, Ep, Eq, Fq, Fp};
use rand::{CryptoRng, RngCore};

//////////////////////////////////////Pallas///////////////////////////////////////////////

impl Group for pallas::Point {
  type Base = pallas::Base;
	type Scalar = pallas::Scalar;
  type CompressedGroupElement = [u8; 32];

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

	///Ioanna: This is so that we can turn Points to affine coordinates 
	///We need this to implement Scalar mul in the circuit
	fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
		let coordinates = self.to_affine().coordinates();
		if coordinates.is_some().unwrap_u8() == 1{
			(*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
		}else{
			(Self::Base::zero(), Self::Base::zero(), true) 
		}
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

impl CompressedGroup for [u8; 32] {
  type GroupElement = pallas::Point;
	
	fn decompress(&self) -> Option<pallas::Point> {
    Some(Ep::from_bytes(self).unwrap())
  }
  fn as_bytes(&self) -> &[u8] {
    self
  }
}

//////////////////////////////////////Vesta////////////////////////////////////////////////

impl Group for vesta::Point {
  type Base = vesta::Base;
	type Scalar = vesta::Scalar;
  type CompressedGroupElement = [u8; 33]; //IOANNA: HACK HACK HACK 
	// Should be <vesta::Point as GroupEncoding>::Repr as above but this is [u8;32]
	// and for this type Compressed Group is already implemented but with GroupElement: pallas::Point
	// TODO: Avoid this hack. Maybe make CompressedGroup Generic?  

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
      .fold(Eq::group_zero(), |acc, x| acc + x)
  }

  fn compress(&self) -> Self::CompressedGroupElement {
		//HACK HACK HACK: Add one more element so that the compression 
		//output type is not the same for pallas and vesta
    let mut arr = [0;33];
		arr[..32].copy_from_slice(&self.to_bytes());
		return arr
  }

  fn from_uniform_bytes(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 32];
      arr.copy_from_slice(&bytes[0..32]);

      let hash = Eq::hash_to_curve("from_uniform_bytes");
      Some(hash(&arr))
    }
  }

  fn gen() -> Self {
    vesta::Point::generator()
  }

	///Ioanna: This is so that we can turn Points to affine coordinates 
	///We need this to implement Scalar mul in the circuit
	fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
		let coordinates = self.to_affine().coordinates();
		if coordinates.is_some().unwrap_u8() == 1{
			(*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
		}else{
			(Self::Base::zero(), Self::Base::zero(), true) 
		}
	}
}

impl PrimeField for vesta::Scalar {
  fn zero() -> Self {
    Fp::zero()
  }
  fn one() -> Self {
    Fp::one()
  }
  fn from_bytes_mod_order_wide(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 64];
      arr.copy_from_slice(&bytes[0..64]);
      Some(Fp::from_bytes_wide(&arr))
    }
  }

  fn random(_rng: &mut (impl RngCore + CryptoRng)) -> Self {
    Fp::rand()
  }

  fn inverse(&self) -> Option<Self> {
    Some(self.invert().unwrap())
  }

  fn as_bytes(&self) -> Vec<u8> {
    self.to_bytes().to_vec()
  }
}


impl ChallengeTrait for vesta::Scalar {
  fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self {
    let mut buf = [0u8; 64];
    transcript.challenge_bytes(label, &mut buf);
    vesta::Scalar::from_bytes_mod_order_wide(&buf).unwrap()
  }
}

impl CompressedGroup for [u8; 33] {
 	type GroupElement = vesta::Point; 
	
	fn decompress(&self) -> Option<vesta::Point> {
    //HACK HACK HACK: See compress function above
		let mut bytes = [0; 32];
		bytes[..32].copy_from_slice(self);
		Some(Eq::from_bytes(&bytes).unwrap())
  }
  fn as_bytes(&self) -> &[u8] {
    self
  }
}
