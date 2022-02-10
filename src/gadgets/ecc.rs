#![allow(non_snake_case)]
use ff::{PrimeField, PrimeFieldBits};
use rand::rngs::OsRng;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
struct Point<Fp, Fq>
where
  Fp: PrimeField,
  Fq: PrimeField + PrimeFieldBits,
{
  x: Fp,
  y: Fp,
  is_infinity: bool,
  _p: PhantomData<Fq>,
}

impl<Fp, Fq> Point<Fp, Fq>
where
  Fp: PrimeField,
  Fq: PrimeField + PrimeFieldBits,
{
  pub fn random_vartime() -> Self {
    loop {
      let x = Fp::random(&mut OsRng);
      let y = (x * x * x + Fp::one() + Fp::one() + Fp::one() + Fp::one() + Fp::one()).sqrt();
      if y.is_some().unwrap_u8() == 1 {
        return Self {
          x: x,
          y: y.unwrap(),
          is_infinity: false,
          _p: Default::default(),
        };
      }
    }
  }

  pub fn add(&self, other: &Point<Fp, Fq>) -> Self {
    if self.is_infinity {
      return other.clone();
    }

    if other.is_infinity {
      return self.clone();
    }

    let lambda = (other.y - self.y) * (other.x - self.x).invert().unwrap();
    let x = lambda * lambda - self.x - other.x;
    let y = lambda * (self.x - x) - self.y;
    Self {
      x: x,
      y: y,
      is_infinity: false,
      _p: Default::default(),
    }
  }

  pub fn double(&self) -> Self {
    if self.is_infinity {
      return Self {
        x: Fp::zero(),
        y: Fp::zero(),
        is_infinity: true,
        _p: Default::default(),
      };
    }

    let lambda = (Fp::one() + Fp::one() + Fp::one())
      * self.x
      * self.x
      * ((Fp::one() + Fp::one()) * self.y).invert().unwrap();
    let x = lambda * lambda - self.x - self.x;
    let y = lambda * (self.x - x) - self.y;
    Self {
      x: x,
      y: y,
      is_infinity: false,
      _p: Default::default(),
    }
  }

  #[allow(dead_code)]
  pub fn scalar_mul_mont(&self, scalar: &Fq) -> Self {
    let mut R0 = Self {
      x: Fp::zero(),
      y: Fp::zero(),
      is_infinity: true,
      _p: Default::default(),
    };

    let mut R1 = self.clone();
    let bits = scalar.to_le_bits();
    for i in (0..bits.len()).rev() {
      if bits[i] {
        R0 = R0.add(&R1);
        R1 = R1.double();
      } else {
        R1 = R0.add(&R1);
        R0 = R0.double();
      }
    }
    R0
  }

  pub fn scalar_mul(&self, scalar: &Fq) -> Self {
    let mut res = Self {
      x: Fp::zero(),
      y: Fp::zero(),
      is_infinity: true,
      _p: Default::default(),
    };

    let bits = scalar.to_le_bits();
    for i in (0..bits.len()).rev() {
      res = res.double();
      if bits[i] {
        res = self.add(&res);
      }
    }
    res
  }
}

#[cfg(test)]
mod fp {
  use ff::PrimeField;

  #[derive(PrimeField)]
  #[PrimeFieldModulus = "28948022309329048855892746252171976963363056481941560715954676764349967630337"]
  #[PrimeFieldGenerator = "5"]
  #[PrimeFieldReprEndianness = "little"]
  pub struct Fp([u64; 4]);
}

#[cfg(test)]
mod fq {
  use ff::PrimeField;

  #[derive(PrimeField)]
  #[PrimeFieldModulus = "28948022309329048855892746252171976963363056481941647379679742748393362948097"]
  #[PrimeFieldGenerator = "5"]
  #[PrimeFieldReprEndianness = "little"]
  pub struct Fq([u64; 4]);
}

#[cfg(test)]
mod tests {
  use super::*;
  use super::{fp::Fp, fq::Fq};
  use ff::Field;
  use pasta_curves::arithmetic::CurveAffine;
  use pasta_curves::group::Curve;
  use pasta_curves::EpAffine;
  use std::ops::Mul;

  #[test]
  fn test_ecc_ops() {
    // perform some curve arithmetic
    let a = Point::<Fp, Fq>::random_vartime();
    let b = Point::<Fp, Fq>::random_vartime();
    let c = a.add(&b);
    let d = a.double();
    let s = Fq::random(&mut OsRng);
    let e = a.scalar_mul(&s);

    // perform the same computation by translating to pasta_curve types
    let a_pasta = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(a.x.to_repr().0.clone()).unwrap(),
      pasta_curves::Fp::from_repr(a.y.to_repr().0.clone()).unwrap(),
    )
    .unwrap();
    let b_pasta = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(b.x.to_repr().0.clone()).unwrap(),
      pasta_curves::Fp::from_repr(b.y.to_repr().0.clone()).unwrap(),
    )
    .unwrap();
    let c_pasta = (a_pasta + b_pasta).to_affine();
    let d_pasta = (a_pasta + a_pasta).to_affine();
    let e_pasta = a_pasta
      .mul(pasta_curves::Fq::from_repr(s.to_repr().0.clone()).unwrap())
      .to_affine();

    // transform c, d, and e into pasta_curve types
    let c_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(c.x.to_repr().0.clone()).unwrap(),
      pasta_curves::Fp::from_repr(c.y.to_repr().0.clone()).unwrap(),
    )
    .unwrap();
    let d_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(d.x.to_repr().0.clone()).unwrap(),
      pasta_curves::Fp::from_repr(d.y.to_repr().0.clone()).unwrap(),
    )
    .unwrap();
    let e_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(e.x.to_repr().0.clone()).unwrap(),
      pasta_curves::Fp::from_repr(e.y.to_repr().0.clone()).unwrap(),
    )
    .unwrap();

    // check that we have the same outputs
    assert_eq!(c_pasta, c_pasta_2);
    assert_eq!(d_pasta, d_pasta_2);
    assert_eq!(e_pasta, e_pasta_2);
  }
}
