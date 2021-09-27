use crate::traits::{ChallengeTrait, CompressedGroup, Group, PrimeField};
use merlin::Transcript;
use pasta_curves::arithmetic::{CurveExt, FieldExt, Group as Grp};
use pasta_curves::group::GroupEncoding;
use pasta_curves::{self, pallas, Ep, Fq};
use rand::{CryptoRng, RngCore};
use std::borrow::Borrow;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PallasPoint(pallas::Point);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PallasScalar(pallas::Scalar);

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PallasCompressedPoint(<pallas::Point as GroupEncoding>::Repr);

impl Group for PallasPoint {
  type Scalar = PallasScalar;
  type CompressedGroupElement = PallasCompressedPoint;

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
      .fold(PallasPoint(Ep::group_zero()), |acc, x| acc + x)
  }

  fn compress(&self) -> Self::CompressedGroupElement {
    PallasCompressedPoint(self.0.to_bytes())
  }

  fn from_uniform_bytes(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 32];
      arr.copy_from_slice(&bytes[0..32]);

      let hash = Ep::hash_to_curve("from_uniform_bytes");
      Some(Self(hash(&arr)))
    }
  }
}

impl PrimeField for PallasScalar {
  fn zero() -> Self {
    Self(Fq::zero())
  }
  fn one() -> Self {
    Self(Fq::one())
  }
  fn from_bytes_mod_order_wide(bytes: &[u8]) -> Option<Self> {
    if bytes.len() != 64 {
      None
    } else {
      let mut arr = [0; 64];
      arr.copy_from_slice(&bytes[0..64]);
      Some(Self(Fq::from_bytes_wide(&arr)))
    }
  }

  fn random(_rng: &mut (impl RngCore + CryptoRng)) -> Self {
    Self(Fq::rand())
  }
}

impl From<Fq> for PallasScalar {
  fn from(s: Fq) -> Self {
    Self(s)
  }
}

impl ChallengeTrait for PallasScalar {
  fn challenge(label: &'static [u8], transcript: &mut Transcript) -> Self {
    let mut buf = [0u8; 64];
    transcript.challenge_bytes(label, &mut buf);
    PallasScalar::from_bytes_mod_order_wide(&buf).unwrap()
  }
}

impl CompressedGroup for PallasCompressedPoint {
  type GroupElement = PallasPoint;
  fn decompress(&self) -> Option<<Self as CompressedGroup>::GroupElement> {
    Some(PallasPoint(Ep::from_bytes(&self.0).unwrap()))
  }
  fn as_bytes(&self) -> &[u8] {
    &self.0
  }
}

impl Add<PallasPoint> for PallasPoint {
  type Output = PallasPoint;

  fn add(self, x: PallasPoint) -> PallasPoint {
    Self(self.0.add(x.0))
  }
}

impl<'r> Add<&'r PallasPoint> for PallasPoint {
  type Output = PallasPoint;

  fn add(self, x: &PallasPoint) -> PallasPoint {
    Self(self.0.add(x.0))
  }
}

impl AddAssign<PallasPoint> for PallasPoint {
  fn add_assign(&mut self, x: PallasPoint) {
    self.0.add_assign(x.0);
  }
}

impl<'r> AddAssign<&'r PallasPoint> for PallasPoint {
  fn add_assign(&mut self, x: &PallasPoint) {
    self.0.add_assign(x.0);
  }
}

impl Sub<PallasPoint> for PallasPoint {
  type Output = PallasPoint;

  fn sub(self, x: PallasPoint) -> PallasPoint {
    Self(self.0.sub(x.0))
  }
}

impl<'r> Sub<&'r PallasPoint> for PallasPoint {
  type Output = PallasPoint;

  fn sub(self, x: &PallasPoint) -> PallasPoint {
    Self(self.0.sub(x.0))
  }
}
impl SubAssign<PallasPoint> for PallasPoint {
  fn sub_assign(&mut self, x: PallasPoint) {
    self.0.sub_assign(x.0);
  }
}
impl<'r> SubAssign<&'r PallasPoint> for PallasPoint {
  fn sub_assign(&mut self, x: &PallasPoint) {
    self.0.sub_assign(x.0);
  }
}

impl Mul<PallasScalar> for PallasPoint {
  type Output = PallasPoint;

  fn mul(self, x: PallasScalar) -> PallasPoint {
    Self(self.0.mul(x.0))
  }
}

impl<'r> Mul<&'r PallasScalar> for PallasPoint {
  type Output = PallasPoint;

  fn mul(self, x: &PallasScalar) -> PallasPoint {
    Self(self.0.mul(x.0))
  }
}

impl MulAssign<PallasScalar> for PallasPoint {
  fn mul_assign(&mut self, x: PallasScalar) {
    self.0.mul_assign(x.0);
  }
}
impl<'r> MulAssign<&'r PallasScalar> for PallasPoint {
  fn mul_assign(&mut self, x: &PallasScalar) {
    self.0.mul_assign(x.0);
  }
}

impl Add<PallasScalar> for PallasScalar {
  type Output = PallasScalar;

  fn add(self, x: PallasScalar) -> PallasScalar {
    Self(self.0.add(x.0))
  }
}

impl<'r> Add<&'r PallasScalar> for PallasScalar {
  type Output = PallasScalar;

  fn add(self, x: &PallasScalar) -> PallasScalar {
    Self(self.0.add(x.0))
  }
}

impl AddAssign<PallasScalar> for PallasScalar {
  fn add_assign(&mut self, x: PallasScalar) {
    self.0.add_assign(x.0);
  }
}

impl<'r> AddAssign<&'r PallasScalar> for PallasScalar {
  fn add_assign(&mut self, x: &PallasScalar) {
    self.0.add_assign(x.0);
  }
}

impl Sub<PallasScalar> for PallasScalar {
  type Output = PallasScalar;

  fn sub(self, x: PallasScalar) -> PallasScalar {
    Self(self.0.sub(x.0))
  }
}

impl<'r> Sub<&'r PallasScalar> for PallasScalar {
  type Output = PallasScalar;

  fn sub(self, x: &PallasScalar) -> PallasScalar {
    Self(self.0.sub(x.0))
  }
}
impl SubAssign<PallasScalar> for PallasScalar {
  fn sub_assign(&mut self, x: PallasScalar) {
    self.0.sub_assign(x.0)
  }
}
impl<'r> SubAssign<&'r PallasScalar> for PallasScalar {
  fn sub_assign(&mut self, x: &PallasScalar) {
    self.0.sub_assign(x.0)
  }
}

impl Mul<PallasScalar> for PallasScalar {
  type Output = PallasScalar;

  fn mul(self, x: PallasScalar) -> PallasScalar {
    Self(self.0.mul(x.0))
  }
}

impl<'r> Mul<&'r PallasScalar> for PallasScalar {
  type Output = PallasScalar;

  fn mul(self, x: &PallasScalar) -> PallasScalar {
    Self(self.0.mul(x.0))
  }
}

impl MulAssign<PallasScalar> for PallasScalar {
  fn mul_assign(&mut self, x: PallasScalar) {
    self.0.mul_assign(x.0)
  }
}
impl<'r> MulAssign<&'r PallasScalar> for PallasScalar {
  fn mul_assign(&mut self, x: &PallasScalar) {
    self.0.mul_assign(x.0)
  }
}

impl Neg for PallasScalar {
  type Output = Self;
  fn neg(self) -> Self {
    Self(self.0.neg())
  }
}
