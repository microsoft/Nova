//! This module implements the Nova traits for pallas::Point, pallas::Scalar, vesta::Point, vesta::Scalar.
use crate::{
  poseidon::{PoseidonRO, PoseidonROCircuit},
  traits::{ChallengeTrait, CompressedGroup, Group},
};
use digest::{ExtendableOutput, Input};
use ff::Field;
use merlin::Transcript;
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::{
  self,
  arithmetic::{CurveAffine, CurveExt, Group as OtherGroup},
  group::{Curve, Group as AnotherGroup, GroupEncoding},
  pallas, vesta, Ep, Eq,
};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use sha3::Shake256;
use std::io::Read;

//////////////////////////////////////Shared MSM code for Pasta curves///////////////////////////////////////////////

/// Native implementation of fast multiexp for platforms that do not support pasta_msm/semolina
/// Forked from zcash/halo2
fn cpu_multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
  use ff::PrimeField;
  let coeffs: Vec<_> = coeffs.iter().map(|a| a.to_repr()).collect();

  let c = if bases.len() < 4 {
    1
  } else if bases.len() < 32 {
    3
  } else {
    (f64::from(bases.len() as u32)).ln().ceil() as usize
  };

  fn get_at<F: PrimeField>(segment: usize, c: usize, bytes: &F::Repr) -> usize {
    let skip_bits = segment * c;
    let skip_bytes = skip_bits / 8;

    if skip_bytes >= 32 {
      return 0;
    }

    let mut v = [0; 8];
    for (v, o) in v.iter_mut().zip(bytes.as_ref()[skip_bytes..].iter()) {
      *v = *o;
    }

    let mut tmp = u64::from_le_bytes(v);
    tmp >>= skip_bits - (skip_bytes * 8);
    tmp %= 1 << c;

    tmp as usize
  }

  let segments = (256 / c) + 1;

  for current_segment in (0..segments).rev() {
    for _ in 0..c {
      *acc = acc.double();
    }

    #[derive(Clone, Copy)]
    enum Bucket<C: CurveAffine> {
      None,
      Affine(C),
      Projective(C::Curve),
    }

    impl<C: CurveAffine> Bucket<C> {
      fn add_assign(&mut self, other: &C) {
        *self = match *self {
          Bucket::None => Bucket::Affine(*other),
          Bucket::Affine(a) => Bucket::Projective(a + *other),
          Bucket::Projective(mut a) => {
            a += *other;
            Bucket::Projective(a)
          }
        }
      }

      fn add(self, mut other: C::Curve) -> C::Curve {
        match self {
          Bucket::None => other,
          Bucket::Affine(a) => {
            other += a;
            other
          }
          Bucket::Projective(a) => other + a,
        }
      }
    }

    let mut buckets: Vec<Bucket<C>> = vec![Bucket::None; (1 << c) - 1];

    for (coeff, base) in coeffs.iter().zip(bases.iter()) {
      let coeff = get_at::<C::Scalar>(current_segment, c, coeff);
      if coeff != 0 {
        buckets[coeff - 1].add_assign(base);
      }
    }

    // Summation by parts
    // e.g. 3a + 2b + 1c = a +
    //                    (a) + b +
    //                    ((a) + b) + c
    let mut running_sum = C::Curve::identity();
    for exp in buckets.into_iter().rev() {
      running_sum = exp.add(running_sum);
      *acc += &running_sum;
    }
  }
}

/// Performs a multi-exponentiation operation without GPU acceleration.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
/// Forked from zcash/halo2
fn cpu_best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());

  let num_threads = rayon::current_num_threads();
  if coeffs.len() > num_threads {
    let chunk = coeffs.len() / num_threads;
    let num_chunks = coeffs.chunks(chunk).len();
    let mut results = vec![C::Curve::identity(); num_chunks];
    rayon::scope(|scope| {
      let chunk = coeffs.len() / num_threads;

      for ((coeffs, bases), acc) in coeffs
        .chunks(chunk)
        .zip(bases.chunks(chunk))
        .zip(results.iter_mut())
      {
        scope.spawn(move |_| {
          cpu_multiexp_serial(coeffs, bases, acc);
        });
      }
    });
    results.iter().fold(C::Curve::identity(), |a, b| a + b)
  } else {
    let mut acc = C::Curve::identity();
    cpu_multiexp_serial(coeffs, bases, &mut acc);
    acc
  }
}

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
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    if scalars.len() >= 128 {
      pasta_msm::pallas(bases, scalars)
    } else {
      cpu_best_multiexp(scalars, bases)
    }
  }

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    cpu_best_multiexp(scalars, bases)
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

  fn get_curve_params() -> (Self::Base, Self::Base, BigInt) {
    let A = Self::Base::zero();
    let B = Self::Base::from(5);
    let order = BigInt::from_str_radix(
      "40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
      16,
    )
    .unwrap();

    (A, B, order)
  }

  fn zero() -> Self {
    pallas::Point::group_zero()
  }

  fn get_generator() -> Self {
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
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;

  #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    if scalars.len() >= 128 {
      pasta_msm::vesta(bases, scalars)
    } else {
      cpu_best_multiexp(scalars, bases)
    }
  }

  #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
  fn vartime_multiscalar_mul(
    scalars: &[Self::Scalar],
    bases: &[Self::PreprocessedGroupElement],
  ) -> Self {
    cpu_best_multiexp(scalars, bases)
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

  fn get_curve_params() -> (Self::Base, Self::Base, BigInt) {
    let A = Self::Base::zero();
    let B = Self::Base::from(5);
    let order = BigInt::from_str_radix(
      "40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
      16,
    )
    .unwrap();

    (A, B, order)
  }

  fn zero() -> Self {
    vesta::Point::group_zero()
  }

  fn get_generator() -> Self {
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
