//! This module implements Nova's traits using the following configuration:
//! `CommitmentEngine` with Pedersen's commitments
//! `Group` with pasta curves and BN256/Grumpkin
//! `RO` traits with Poseidon
//! `EvaluationEngine` with an IPA-based polynomial evaluation argument

pub mod bn256_grumpkin;
pub mod ipa_pc;
pub mod keccak;
pub mod pasta;
pub mod pedersen;
pub mod poseidon;

use ff::PrimeField;
use pasta_curves::{self, arithmetic::CurveAffine, group::Group as AnotherGroup};

/// Native implementation of fast multiexp
/// Adapted from zcash/halo2
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
pub(crate) fn cpu_best_multiexp<C: CurveAffine>(coeffs: &[C::Scalar], bases: &[C]) -> C::Curve {
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
