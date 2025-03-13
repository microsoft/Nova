//! This module provides a multi-scalar multiplication routine
/// Adapted from zcash/halo2
use ff::PrimeField;
use halo2curves::{group::Group, CurveAffine};
use rayon::{current_num_threads, prelude::*};

fn cpu_msm_serial<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
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

  (0..segments)
    .rev()
    .fold(C::Curve::identity(), |mut acc, segment| {
      (0..c).for_each(|_| acc = acc.double());

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
            Bucket::Projective(a) => Bucket::Projective(a + other),
          }
        }

        fn add(self, other: C::Curve) -> C::Curve {
          match self {
            Bucket::None => other,
            Bucket::Affine(a) => other + a,
            Bucket::Projective(a) => other + a,
          }
        }
      }

      let mut buckets = vec![Bucket::None; (1 << c) - 1];

      for (coeff, base) in coeffs.iter().zip(bases.iter()) {
        let coeff = get_at::<C::Scalar>(segment, c, &coeff.to_repr());
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
        acc += &running_sum;
      }
      acc
    })
}

/// Performs a multi-scalar-multiplication operation without GPU acceleration.
///
/// This function will panic if coeffs and bases have a different length.
///
/// This will use multithreading if beneficial.
/// Adapted from zcash/halo2
pub(crate) fn cpu_best_msm<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
  assert_eq!(coeffs.len(), bases.len());

  let num_threads = current_num_threads();
  if coeffs.len() > num_threads {
    let chunk = coeffs.len() / num_threads;
    coeffs
      .par_chunks(chunk)
      .zip(bases.par_chunks(chunk))
      .map(|(coeffs, bases)| cpu_msm_serial(coeffs, bases))
      .reduce(C::Curve::identity, |sum, evl| sum + evl)
  } else {
    cpu_msm_serial(coeffs, bases)
  }
}

#[cfg(test)]
mod tests {
  use super::cpu_best_msm;
  use crate::provider::{
    bn256_grumpkin::{bn256, grumpkin},
    pasta::{pallas, vesta},
    secp_secq::{secp256k1, secq256k1},
  };
  use ff::Field;
  use halo2curves::{group::Group, CurveAffine};
  use rand_core::OsRng;

  fn test_msm_with<F: Field, A: CurveAffine<ScalarExt = F>>() {
    let n = 8;
    let coeffs = (0..n).map(|_| F::random(OsRng)).collect::<Vec<_>>();
    let bases = (0..n)
      .map(|_| A::from(A::generator() * F::random(OsRng)))
      .collect::<Vec<_>>();
    let naive = coeffs
      .iter()
      .zip(bases.iter())
      .fold(A::CurveExt::identity(), |acc, (coeff, base)| {
        acc + *base * coeff
      });
    let msm = cpu_best_msm(&coeffs, &bases);

    assert_eq!(naive, msm)
  }

  #[test]
  fn test_msm() {
    test_msm_with::<pallas::Scalar, pallas::Affine>();
    test_msm_with::<vesta::Scalar, vesta::Affine>();
    test_msm_with::<bn256::Scalar, bn256::Affine>();
    test_msm_with::<grumpkin::Scalar, grumpkin::Affine>();
    test_msm_with::<secp256k1::Scalar, secp256k1::Affine>();
    test_msm_with::<secq256k1::Scalar, secq256k1::Affine>();
  }
}
