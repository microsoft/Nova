//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! We provide two implementations, one in snark.rs (which does not use any preprocessing)
//! and another in ppsnark.rs (which uses preprocessing to keep the verifier's state small if the PCS provides a succinct verifier)
//! We also provide direct.rs that allows proving a step circuit directly with either of the two SNARKs.
//!
//! In polynomial.rs we also provide foundational types and functions for manipulating multilinear polynomials.
pub mod direct;
pub mod ppsnark;
pub mod snark;

#[macro_use]
mod macros;

/// Module providing the `Math` trait with `log_2()` method on `usize`.
pub mod math;

/// Module providing polynomial types for Spartan SNARKs.
pub mod polys;

/// Module providing sumcheck protocol implementation.
pub mod sumcheck;

pub use sumcheck::SumcheckEngine;

use crate::{
  errors::NovaError,
  r1cs::{R1CSShape, SparseMatrix},
  traits::Engine,
  Commitment,
};
use ff::{Field, PrimeField};
use itertools::Itertools as _;
use rayon::{iter::IntoParallelRefIterator, prelude::*};
use std::cmp::max;

/// Creates a vector of the first `n` powers of `s`.
///
/// Returns `[1, s, s^2, ..., s^{n-1}]`.
pub fn powers<E: Engine>(s: &E::Scalar, n: usize) -> Vec<E::Scalar> {
  assert!(n >= 1);
  let mut powers = Vec::with_capacity(n);
  powers.push(E::Scalar::ONE);
  for i in 1..n {
    powers.push(powers[i - 1] * s);
  }
  powers
}

/// Batch invert a vector of field elements
///
/// Uses Montgomery's trick with parallelization for large inputs.
/// Returns an error if any element is zero.
pub fn batch_invert<Scalar: PrimeField>(v: &[Scalar]) -> Result<Vec<Scalar>, NovaError> {
  const MAX_SIZE_FOR_SERIAL: usize = 4096;
  const MIN_CHUNK_SIZE: usize = 128;

  if v.len() < MAX_SIZE_FOR_SERIAL {
    return batch_invert_serial(v);
  }

  let compute_prod = |beta: &[Scalar]| -> (Vec<Scalar>, Scalar) {
    let mut prod = vec![Scalar::ZERO; beta.len()];
    let mut acc = Scalar::ONE;
    prod.iter_mut().zip(beta).for_each(|(p, e)| {
      let nxt_acc = acc * *e;
      *p = acc;
      acc = nxt_acc;
    });
    (prod, acc)
  };

  let compute_inv = |beta: &mut [Scalar], prod: &[Scalar], init_inv: Scalar| {
    let mut acc = init_inv;
    for i in (0..beta.len()).rev() {
      let nxt_acc = acc * beta[i];
      beta[i] = acc * prod[i];
      acc = nxt_acc;
    }
  };

  let mut v = v.to_vec();

  let num_chunks = rayon::current_num_threads();
  let chunk_size = max(MIN_CHUNK_SIZE, v.len() / num_chunks);

  // Phase 1: Compute Product Tree
  let (prod1, mut beta2, prod2, root) = {
    let chunks = v
      .par_chunks(chunk_size)
      .map(compute_prod)
      .collect::<Vec<_>>();

    let (prod1, beta2): (Vec<Vec<_>>, Vec<_>) = chunks.into_iter().unzip();

    let (prod2, root) = compute_prod(&beta2);

    (prod1, beta2, prod2, root)
  };

  if root == Scalar::ZERO {
    return Err(NovaError::InternalError);
  }
  let root_inv = root.invert().unwrap();

  // Phase 2: Compute Inverse Tree
  compute_inv(&mut beta2, &prod2, root_inv);

  v.par_chunks_mut(chunk_size)
    .zip(prod1)
    .zip(beta2)
    .for_each(|((v, p), b)| {
      compute_inv(v, &p, b);
    });

  Ok(v)
}

/// Serial batch invert for small inputs using Montgomery's trick
pub fn batch_invert_serial<Scalar: PrimeField>(v: &[Scalar]) -> Result<Vec<Scalar>, NovaError> {
  let mut products = vec![Scalar::ZERO; v.len()];
  let mut acc = Scalar::ONE;

  for i in 0..v.len() {
    products[i] = acc;
    acc *= v[i];
  }

  // we can compute an inversion only if acc is non-zero
  if acc == Scalar::ZERO {
    return Err(NovaError::InternalError);
  }

  // compute the inverse once for all entries
  acc = acc.invert().unwrap();

  let mut inv = vec![Scalar::ZERO; v.len()];
  for i in 0..v.len() {
    let tmp = acc * v[v.len() - 1 - i];
    inv[v.len() - 1 - i] = products[v.len() - 1 - i] * acc;
    acc = tmp;
  }

  Ok(inv)
}

/// A type that holds a witness to a polynomial evaluation instance
#[derive(Clone, Debug)]
pub struct PolyEvalWitness<E: Engine> {
  p: Vec<E::Scalar>, // polynomial
}

impl<E: Engine> PolyEvalWitness<E> {
  /// Returns a reference to the polynomial coefficients.
  pub fn p(&self) -> &[E::Scalar] {
    &self.p
  }

  /// Given \[Pᵢ\] and s, compute P = ∑ᵢ sⁱ⋅Pᵢ
  ///
  /// # Details
  ///
  /// We allow the input polynomials to have different sizes, and interpret smaller ones as
  /// being padded with 0 to the maximum size of all polynomials.
  fn batch_diff_size(W: Vec<PolyEvalWitness<E>>, s: E::Scalar) -> PolyEvalWitness<E> {
    let powers = powers::<E>(&s, W.len());

    let size_max = W.iter().map(|w| w.p.len()).max().unwrap();
    // Scale the input polynomials by the power of s
    let num_chunks = rayon::current_num_threads().next_power_of_two();
    let chunk_size = size_max / num_chunks;

    let p = if chunk_size > 0 {
      (0..num_chunks)
        .into_par_iter()
        .flat_map_iter(|chunk_index| {
          let mut chunk = vec![E::Scalar::ZERO; chunk_size];
          for (coeff, poly) in powers.iter().zip(W.iter()) {
            for (rlc, poly_eval) in chunk
              .iter_mut()
              .zip(poly.p[chunk_index * chunk_size..].iter())
            {
              if *coeff == E::Scalar::ONE {
                *rlc += *poly_eval;
              } else {
                *rlc += *coeff * poly_eval;
              };
            }
          }
          chunk
        })
        .collect::<Vec<_>>()
    } else {
      W.into_par_iter()
        .zip_eq(powers.par_iter())
        .map(|(mut w, s)| {
          if *s != E::Scalar::ONE {
            w.p.par_iter_mut().for_each(|e| *e *= s);
          }
          w.p
        })
        .reduce(
          || vec![E::Scalar::ZERO; size_max],
          |left, right| {
            // Sum into the largest polynomial
            let (mut big, small) = if left.len() > right.len() {
              (left, right)
            } else {
              (right, left)
            };

            big
              .par_iter_mut()
              .zip(small.par_iter())
              .for_each(|(b, s)| *b += s);

            big
          },
        )
    };

    PolyEvalWitness { p }
  }

  /// Given a set of polynomials \[Pᵢ\] and a scalar `s`, this method computes the weighted sum
  /// of the polynomials, where each polynomial Pᵢ is scaled by sⁱ. The method handles polynomials
  /// of different sizes by padding smaller ones with zeroes up to the size of the largest polynomial.
  ///
  /// # Panics
  ///
  /// This method panics if the polynomials in `p_vec` are not all of the same length.
  fn batch(p_vec: &[&Vec<E::Scalar>], s: &E::Scalar) -> PolyEvalWitness<E> {
    p_vec
      .iter()
      .for_each(|p| assert_eq!(p.len(), p_vec[0].len()));

    let powers_of_s = powers::<E>(s, p_vec.len());

    let num_chunks = rayon::current_num_threads().next_power_of_two();
    let chunk_size = p_vec[0].len() / num_chunks;

    let p = if chunk_size > 0 {
      (0..num_chunks)
        .into_par_iter()
        .flat_map_iter(|chunk_index| {
          let mut chunk = vec![E::Scalar::ZERO; chunk_size];
          for (coeff, poly) in powers_of_s.iter().zip(p_vec.iter()) {
            for (rlc, poly_eval) in chunk
              .iter_mut()
              .zip(poly[chunk_index * chunk_size..].iter())
            {
              if *coeff == E::Scalar::ONE {
                *rlc += *poly_eval;
              } else {
                *rlc += *coeff * poly_eval;
              };
            }
          }
          chunk
        })
        .collect::<Vec<_>>()
    } else {
      zip_with!(par_iter, (p_vec, powers_of_s), |v, weight| {
        // compute the weighted sum for each vector
        v.iter().map(|&x| x * *weight).collect::<Vec<E::Scalar>>()
      })
      .reduce(
        || vec![E::Scalar::ZERO; p_vec[0].len()],
        |acc, v| {
          // perform vector addition to combine the weighted vectors
          zip_with!((acc.into_iter(), v), |x, y| x + y).collect()
        },
      )
    };

    PolyEvalWitness { p }
  }
}

/// A type that holds a polynomial evaluation instance
#[derive(Clone, Debug)]
pub struct PolyEvalInstance<E: Engine> {
  c: Commitment<E>,  // commitment to the polynomial
  x: Vec<E::Scalar>, // evaluation point
  e: E::Scalar,      // claimed evaluation
}

impl<E: Engine> PolyEvalInstance<E> {
  /// Returns a reference to the commitment to the polynomial.
  pub fn c(&self) -> &Commitment<E> {
    &self.c
  }

  /// Returns a reference to the evaluation point.
  pub fn x(&self) -> &[E::Scalar] {
    &self.x
  }

  /// Returns the claimed evaluation.
  pub fn e(&self) -> E::Scalar {
    self.e
  }

  fn batch_diff_size(
    c_vec: &[Commitment<E>],
    e_vec: &[E::Scalar],
    num_vars: &[usize],
    x: Vec<E::Scalar>,
    s: E::Scalar,
  ) -> PolyEvalInstance<E> {
    let num_instances = num_vars.len();
    assert_eq!(c_vec.len(), num_instances);
    assert_eq!(e_vec.len(), num_instances);

    let num_vars_max = x.len();
    let powers: Vec<E::Scalar> = powers::<E>(&s, num_instances);
    // Rescale evaluations by the first Lagrange polynomial,
    // so that we can check its evaluation against x
    let evals_scaled = zip_with!(iter, (e_vec, num_vars), |eval, num_rounds| {
      // x_lo = [ x[0]   , ..., x[n-nᵢ-1] ]
      // x_hi = [ x[n-nᵢ], ..., x[n]      ]
      let (r_lo, _r_hi) = x.split_at(num_vars_max - num_rounds);
      // Compute L₀(x_lo)
      let lagrange_eval = r_lo
        .iter()
        .map(|r| E::Scalar::ONE - r)
        .product::<E::Scalar>();

      // vᵢ = L₀(x_lo)⋅Pᵢ(x_hi)
      lagrange_eval * eval
    })
    .collect::<Vec<_>>();

    // C = ∑ᵢ γⁱ⋅Cᵢ
    let comm_joint = zip_with!(iter, (c_vec, powers), |c, g_i| *c * *g_i)
      .fold(Commitment::<E>::default(), |acc, item| acc + item);

    // v = ∑ᵢ γⁱ⋅vᵢ
    let eval_joint = zip_with!((evals_scaled.into_iter(), powers.iter()), |e, g_i| e * g_i).sum();

    PolyEvalInstance {
      c: comm_joint,
      x,
      e: eval_joint,
    }
  }

  fn batch(
    c_vec: &[Commitment<E>],
    x: &[E::Scalar],
    e_vec: &[E::Scalar],
    s: &E::Scalar,
  ) -> PolyEvalInstance<E> {
    let num_instances = c_vec.len();
    assert_eq!(e_vec.len(), num_instances);

    let powers_of_s = powers::<E>(s, num_instances);
    // Weighted sum of evaluations
    let e = zip_with!(par_iter, (e_vec, powers_of_s), |e, p| *e * p).sum();
    // Weighted sum of commitments
    let c = zip_with!(par_iter, (c_vec, powers_of_s), |c, p| *c * *p)
      .reduce(Commitment::<E>::default, |acc, item| acc + item);

    PolyEvalInstance {
      c,
      x: x.to_vec(),
      e,
    }
  }
}

/// Bounds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials.
///
/// Given an R1CS shape and evaluation point `rx`, computes the evaluations of the
/// A, B, and C matrices when their row variables are bound to `rx`.
///
/// # Arguments
/// * `S` - The R1CS shape containing the A, B, C matrices
/// * `rx` - The evaluation point for row variables (length must equal num_cons)
///
/// # Returns
/// A tuple of three vectors (A_evals, B_evals, C_evals), each of length 2 * num_vars.
pub fn compute_eval_table_sparse<E: Engine>(
  S: &R1CSShape<E>,
  rx: &[E::Scalar],
) -> (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>) {
  assert_eq!(rx.len(), S.num_cons());

  let inner = |M: &SparseMatrix<E::Scalar>, M_evals: &mut Vec<E::Scalar>| {
    for (row_idx, ptrs) in M.indptr.windows(2).enumerate() {
      for (val, col_idx) in M.get_row_unchecked(ptrs.try_into().unwrap()) {
        M_evals[*col_idx] += rx[row_idx] * val;
      }
    }
  };

  let (A_evals, (B_evals, C_evals)) = rayon::join(
    || {
      let mut A_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars()];
      inner(S.A(), &mut A_evals);
      A_evals
    },
    || {
      rayon::join(
        || {
          let mut B_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars()];
          inner(S.B(), &mut B_evals);
          B_evals
        },
        || {
          let mut C_evals: Vec<E::Scalar> = vec![E::Scalar::ZERO; 2 * S.num_vars()];
          inner(S.C(), &mut C_evals);
          C_evals
        },
      )
    },
  );

  (A_evals, B_evals, C_evals)
}
