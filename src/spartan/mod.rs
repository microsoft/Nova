//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! We provide two implementations, one in snark.rs (which does not use any preprocessing)
//! and another in ppsnark.rs (which uses preprocessing to keep the verifier's state small if the PCS provides a succinct verifier)
//! We also provide direct.rs that allows proving a step circuit directly with either of the two SNARKs.
//!
//! In polynomial.rs we also provide foundational types and functions for manipulating multilinear polynomials.
pub mod direct;
pub(crate) mod math;
pub mod polys;
pub mod ppsnark;
pub mod snark;
mod sumcheck;

use crate::{traits::Engine, Commitment};
use ff::Field;
use polys::multilinear::SparsePolynomial;
use rayon::{iter::IntoParallelRefIterator, prelude::*};

fn powers<E: Engine>(s: &E::Scalar, n: usize) -> Vec<E::Scalar> {
  assert!(n >= 1);
  let mut powers = Vec::new();
  powers.push(E::Scalar::ONE);
  for i in 1..n {
    powers.push(powers[i - 1] * s);
  }
  powers
}

/// A type that holds a witness to a polynomial evaluation instance
pub struct PolyEvalWitness<E: Engine> {
  p: Vec<E::Scalar>, // polynomial
}

impl<E: Engine> PolyEvalWitness<E> {
  fn pad(mut W: Vec<PolyEvalWitness<E>>) -> Vec<PolyEvalWitness<E>> {
    // determine the maximum size
    if let Some(n) = W.iter().map(|w| w.p.len()).max() {
      W.iter_mut().for_each(|w| {
        w.p.resize(n, E::Scalar::ZERO);
      });
      W
    } else {
      Vec::new()
    }
  }

  fn weighted_sum(W: &[PolyEvalWitness<E>], s: &[E::Scalar]) -> PolyEvalWitness<E> {
    assert_eq!(W.len(), s.len());
    let mut p = vec![E::Scalar::ZERO; W[0].p.len()];
    for i in 0..W.len() {
      for j in 0..W[i].p.len() {
        p[j] += W[i].p[j] * s[i]
      }
    }
    PolyEvalWitness { p }
  }

  // This method panics unless all vectors in p_vec are of the same length
  fn batch(p_vec: &[&Vec<E::Scalar>], s: &E::Scalar) -> PolyEvalWitness<E> {
    p_vec
      .iter()
      .for_each(|p| assert_eq!(p.len(), p_vec[0].len()));

    let powers_of_s = powers::<E>(s, p_vec.len());

    let p = p_vec
      .par_iter()
      .zip(powers_of_s.par_iter())
      .map(|(v, &weight)| {
        // compute the weighted sum for each vector
        v.iter().map(|&x| x * weight).collect::<Vec<E::Scalar>>()
      })
      .reduce(
        || vec![E::Scalar::ZERO; p_vec[0].len()],
        |acc, v| {
          // perform vector addition to combine the weighted vectors
          acc.into_iter().zip(v).map(|(x, y)| x + y).collect()
        },
      );

    PolyEvalWitness { p }
  }
}

/// A type that holds a polynomial evaluation instance
pub struct PolyEvalInstance<E: Engine> {
  c: Commitment<E>,  // commitment to the polynomial
  x: Vec<E::Scalar>, // evaluation point
  e: E::Scalar,      // claimed evaluation
}

impl<E: Engine> PolyEvalInstance<E> {
  fn pad(U: Vec<PolyEvalInstance<E>>) -> Vec<PolyEvalInstance<E>> {
    // determine the maximum size
    if let Some(ell) = U.iter().map(|u| u.x.len()).max() {
      U.into_iter()
        .map(|mut u| {
          let mut x = vec![E::Scalar::ZERO; ell - u.x.len()];
          x.append(&mut u.x);
          PolyEvalInstance { x, ..u }
        })
        .collect()
    } else {
      Vec::new()
    }
  }

  fn batch(
    c_vec: &[Commitment<E>],
    x: &[E::Scalar],
    e_vec: &[E::Scalar],
    s: &E::Scalar,
  ) -> PolyEvalInstance<E> {
    let powers_of_s = powers::<E>(s, c_vec.len());
    let e = e_vec
      .par_iter()
      .zip(powers_of_s.par_iter())
      .map(|(e, p)| *e * p)
      .sum();
    let c = c_vec
      .par_iter()
      .zip(powers_of_s.par_iter())
      .map(|(c, p)| *c * *p)
      .reduce(Commitment::<E>::default, |acc, item| acc + item);

    PolyEvalInstance {
      c,
      x: x.to_vec(),
      e,
    }
  }
}
