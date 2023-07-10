//! This module implements the Nova traits for secp::Point, secp::Scalar, secq::Point, secq::Scalar.
use crate::{
  provider::{
    keccak::Keccak256Transcript,
    pedersen::CommitmentEngine,
    poseidon::{PoseidonRO, PoseidonROCircuit},
  },
  traits::{CompressedGroup, Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Input};
use ff::{FromUniformBytes, PrimeField};
use halo2curves::{
  secp256k1::{Secp256k1, Secp256k1Affine, Secp256k1Compressed},
  secq256k1::{Secq256k1, Secq256k1Affine, Secq256k1Compressed},
  CurveAffine,
};
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::{
  self,
  arithmetic::CurveExt,
  group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup, GroupEncoding},
};

/// Re-exports that give access to the standard aliases used in the code base, for secp
pub mod secp256k1 {
  pub use halo2curves::secp256k1::{
    Fp as Base, Fq as Scalar, Secp256k1 as Point, Secp256k1Affine as Affine,
    Secp256k1Compressed as Compressed,
  };
}

/// Re-exports that give access to the standard aliases used in the code base, for secq
pub mod secq256k1 {
  pub use halo2curves::secq256k1::{
    Fp as Base, Fq as Scalar, Secq256k1 as Point, Secq256k1Affine as Affine,
    Secq256k1Compressed as Compressed,
  };
}

use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

impl<G: Group> TranscriptReprTrait<G> for secp256k1::Base {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().to_vec()
  }
}

impl<G: Group> TranscriptReprTrait<G> for secp256k1::Scalar {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().to_vec()
  }
}

macro_rules! impl_traits {
  (
    $name:ident,
    $name_compressed:ident,
    $name_curve:ident,
    $name_curve_affine:ident,
    $order_str:literal
  ) => {
    impl Group for $name::Point {
      type Base = $name::Base;
      type Scalar = $name::Scalar;
      type CompressedGroupElement = $name_compressed;
      type PreprocessedGroupElement = $name::Affine;
      type RO = PoseidonRO<Self::Base, Self::Scalar>;
      type ROCircuit = PoseidonROCircuit<Self::Base>;
      type TE = Keccak256Transcript<Self>;
      type CE = CommitmentEngine<Self>;

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
        self.to_bytes()
      }

      fn from_label(label: &'static [u8], n: usize) -> Vec<Self::PreprocessedGroupElement> {
        let mut shake = Shake256::default();
        shake.input(label);
        let mut reader = shake.xof_result();
        let mut uniform_bytes_vec = Vec::new();
        for _ in 0..n {
          let mut uniform_bytes = [0u8; 32];
          reader.read_exact(&mut uniform_bytes).unwrap();
          uniform_bytes_vec.push(uniform_bytes);
        }
        let gens_proj: Vec<$name_curve> = (0..n)
          .collect::<Vec<usize>>()
          .into_par_iter()
          .map(|i| {
            let hash = $name_curve::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes_vec[i])
          })
          .collect();

        let num_threads = rayon::current_num_threads();
        if gens_proj.len() > num_threads {
          let chunk = (gens_proj.len() as f64 / num_threads as f64).ceil() as usize;
          (0..num_threads)
            .collect::<Vec<usize>>()
            .into_par_iter()
            .map(|i| {
              let start = i * chunk;
              let end = if i == num_threads - 1 {
                gens_proj.len()
              } else {
                core::cmp::min((i + 1) * chunk, gens_proj.len())
              };
              if end > start {
                let mut gens = vec![$name_curve_affine::identity(); end - start];
                <Self as Curve>::batch_normalize(&gens_proj[start..end], &mut gens);
                gens
              } else {
                vec![]
              }
            })
            .collect::<Vec<Vec<$name_curve_affine>>>()
            .into_par_iter()
            .flatten()
            .collect()
        } else {
          let mut gens = vec![$name_curve_affine::identity(); n];
          <Self as Curve>::batch_normalize(&gens_proj, &mut gens);
          gens
        }
      }

      fn to_coordinates(&self) -> (Self::Base, Self::Base, bool) {
        let coordinates = self.to_affine().coordinates();
        if coordinates.is_some().unwrap_u8() == 1
          // The bn256/grumpkin convention is to define and return the identity point's affine encoding (not None)
          && (Self::PreprocessedGroupElement::identity() != self.to_affine())
        {
          (*coordinates.unwrap().x(), *coordinates.unwrap().y(), false)
        } else {
          (Self::Base::zero(), Self::Base::zero(), true)
        }
      }

      fn get_curve_params() -> (Self::Base, Self::Base, BigInt) {
        let A = $name::Point::a();
        let B = $name::Point::b();
        let order = BigInt::from_str_radix($order_str, 16).unwrap();

        (A, B, order)
      }

      fn zero() -> Self {
        $name::Point::identity()
      }

      fn get_generator() -> Self {
        $name::Point::generator()
      }
    }

    impl PrimeFieldExt for $name::Scalar {
      fn from_uniform(bytes: &[u8]) -> Self {
        let bytes_arr: [u8; 64] = bytes.try_into().unwrap();
        $name::Scalar::from_uniform_bytes(&bytes_arr)
      }
    }

    impl<G: Group> TranscriptReprTrait<G> for $name_compressed {
      fn to_transcript_bytes(&self) -> Vec<u8> {
        self.as_ref().to_vec()
      }
    }

    impl CompressedGroup for $name_compressed {
      type GroupElement = $name::Point;

      fn decompress(&self) -> Option<$name::Point> {
        Some($name_curve::from_bytes(&self).unwrap())
      }
    }
  };
}

impl_traits!(
  secp256k1,
  Secp256k1Compressed,
  Secp256k1,
  Secp256k1Affine,
  "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
);

impl_traits!(
  secq256k1,
  Secq256k1Compressed,
  Secq256k1,
  Secq256k1Affine,
  "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
);

fn cpu_multiexp_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C], acc: &mut C::Curve) {
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
/// Adapted from zcash/halo2
// TODO: update once https://github.com/privacy-scaling-explorations/halo2curves/pull/29
// (or a successor thereof) is merged
fn cpu_best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());

  let num_threads = rayon::current_num_threads();
  if coeffs.len() > num_threads {
    let chunk = coeffs.len() / num_threads;
    let num_chunks = coeffs.chunks(chunk).len();
    let mut results = vec![C::Curve::identity(); num_chunks];
    rayon::scope(|scope| {
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

#[cfg(test)]
mod tests {

  use super::*;
  type G = secp256k1::Point;

  fn from_label_serial(label: &'static [u8], n: usize) -> Vec<Secp256k1Affine> {
    let mut shake = Shake256::default();
    shake.input(label);
    let mut reader = shake.xof_result();
    let mut ck = Vec::new();
    for _ in 0..n {
      let mut uniform_bytes = [0u8; 32];
      reader.read_exact(&mut uniform_bytes).unwrap();
      let hash = secp256k1::Point::hash_to_curve("from_uniform_bytes");
      ck.push(hash(&uniform_bytes).to_affine());
    }
    ck
  }

  #[test]
  fn test_from_label() {
    let label = b"test_from_label";
    for n in [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1021,
    ] {
      let ck_par = <G as Group>::from_label(label, n);
      let ck_ser = from_label_serial(label, n);
      assert_eq!(ck_par.len(), n);
      assert_eq!(ck_ser.len(), n);
      assert_eq!(ck_par, ck_ser);
    }
  }
}
