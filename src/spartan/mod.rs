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
  traits::{Engine, TranscriptEngineTrait},
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
  /// Creates a new polynomial evaluation witness from polynomial coefficients.
  pub fn new(p: Vec<E::Scalar>) -> Self {
    Self { p }
  }

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
  pub fn batch_diff_size(W: Vec<PolyEvalWitness<E>>, s: E::Scalar) -> PolyEvalWitness<E> {
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
          let start_idx = chunk_index * chunk_size;
          for (coeff, poly) in powers.iter().zip(W.iter()) {
            // Handle different-sized polynomials by treating indices beyond poly.len() as 0
            if start_idx < poly.p.len() {
              for (rlc, poly_eval) in chunk.iter_mut().zip(poly.p[start_idx..].iter()) {
                if *coeff == E::Scalar::ONE {
                  *rlc += *poly_eval;
                } else {
                  *rlc += *coeff * poly_eval;
                };
              }
            }
            // else: this chunk is beyond poly.p's length, contributes 0 (already initialized)
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
  pub fn batch(p_vec: &[&Vec<E::Scalar>], s: &E::Scalar) -> PolyEvalWitness<E> {
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
  /// Creates a new polynomial evaluation instance.
  pub fn new(c: Commitment<E>, x: Vec<E::Scalar>, e: E::Scalar) -> Self {
    Self { c, x, e }
  }

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

  /// Batches multiple polynomial evaluation instances with different sizes into a single instance.
  ///
  /// Given commitments, evaluations, and variable counts for each polynomial,
  /// combines them using random linear combination with Lagrange correction
  /// for the replication model.
  pub fn batch_diff_size(
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

  /// Batches multiple polynomial evaluation instances with the same evaluation point.
  ///
  /// All instances must share the same evaluation point `x`.
  pub fn batch(
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

/// Reduces a batch of polynomial evaluation claims using Sumcheck
/// to a single claim at the same point.
///
/// # Details
///
/// We are given as input a list of instance/witness pairs
/// u = \[(Cᵢ, xᵢ, eᵢ)\], w = \[Pᵢ\], such that
/// - nᵢ = |xᵢ|
/// - Cᵢ = Commit(Pᵢ)
/// - eᵢ = Pᵢ(xᵢ)
/// - |Pᵢ| = 2^nᵢ
///
/// We allow the polynomial Pᵢ to have different sizes, by appropriately scaling
/// the claims and resulting evaluations from Sumcheck using the replication model.
pub fn batch_eval_reduce<E: Engine>(
  u_vec: Vec<PolyEvalInstance<E>>,
  w_vec: Vec<PolyEvalWitness<E>>,
  transcript: &mut E::TE,
) -> Result<
  (
    PolyEvalInstance<E>,
    PolyEvalWitness<E>,
    E::Scalar, // chal used for batching
    sumcheck::SumcheckProof<E>,
    Vec<E::Scalar>,
  ),
  NovaError,
> {
  use polys::multilinear::MultilinearPolynomial;

  let num_claims = u_vec.len();
  assert_eq!(w_vec.len(), num_claims);

  // Compute nᵢ and n = maxᵢ{nᵢ}
  let num_rounds = u_vec.iter().map(|u| u.x.len()).collect::<Vec<_>>();

  // Check polynomials match number of variables, i.e. |Pᵢ| = 2^nᵢ
  w_vec
    .iter()
    .zip_eq(num_rounds.iter())
    .for_each(|(w, num_vars)| assert_eq!(w.p.len(), 1 << num_vars));

  // generate a challenge, and powers of it for random linear combination
  let rho = transcript.squeeze(b"r")?;
  let powers_of_rho = powers::<E>(&rho, num_claims);

  let (claims, eval_points, comms): (Vec<_>, Vec<_>, Vec<_>) =
    u_vec.into_iter().map(|u| (u.e, u.x, u.c)).multiunzip();

  // Create clones of polynomials to be given to Sumcheck
  // Pᵢ(X)
  let polys_P: Vec<MultilinearPolynomial<E::Scalar>> = w_vec
    .iter()
    .map(|w| MultilinearPolynomial::new(w.p.clone()))
    .collect();

  // For each i, check eᵢ = ∑ₓ Pᵢ(x)eq(xᵢ,x), where x ∈ {0,1}^nᵢ
  // Use split eq representation for memory efficiency (O(sqrt(N)) instead of O(N))
  let (sc_proof_batch, r, claims_batch) = sumcheck::SumcheckProof::prove_quad_batch_prod_split_eq(
    &claims,
    &num_rounds,
    polys_P,
    eval_points,
    &powers_of_rho,
    transcript,
  )?;

  let (claims_batch_left, _): (Vec<E::Scalar>, Vec<E::Scalar>) = claims_batch;

  transcript.absorb(b"l", &claims_batch_left.as_slice());

  // we now combine evaluation claims at the same point r into one
  let chal = transcript.squeeze(b"c")?;

  let u_joint = PolyEvalInstance::batch_diff_size(&comms, &claims_batch_left, &num_rounds, r, chal);

  // P = ∑ᵢ γⁱ⋅Pᵢ
  let w_joint = PolyEvalWitness::batch_diff_size(w_vec, chal);

  Ok((u_joint, w_joint, chal, sc_proof_batch, claims_batch_left))
}

/// Verifies a batch of polynomial evaluation claims using Sumcheck
/// reducing them to a single claim at the same point.
pub fn batch_eval_verify<E: Engine>(
  u_vec: Vec<PolyEvalInstance<E>>,
  transcript: &mut E::TE,
  sc_proof_batch: &sumcheck::SumcheckProof<E>,
  evals_batch: &[E::Scalar],
) -> Result<PolyEvalInstance<E>, NovaError> {
  use polys::eq::EqPolynomial;

  let num_claims = u_vec.len();
  assert_eq!(evals_batch.len(), num_claims);

  // generate a challenge
  let rho = transcript.squeeze(b"r")?;
  let powers_of_rho = powers::<E>(&rho, num_claims);

  // Compute nᵢ and n = maxᵢ{nᵢ}
  let num_rounds = u_vec.iter().map(|u| u.x.len()).collect::<Vec<_>>();
  let num_rounds_max = *num_rounds.iter().max().unwrap();

  let claims = u_vec.iter().map(|u| u.e).collect::<Vec<_>>();

  let (claim_batch_final, r) =
    sc_proof_batch.verify_batch(&claims, &num_rounds, &powers_of_rho, 2, transcript)?;

  let claim_batch_final_expected = {
    let evals_r = u_vec.iter().map(|u| {
      let (_, r_hi) = r.split_at(num_rounds_max - u.x.len());
      EqPolynomial::new(r_hi.to_vec()).evaluate(&u.x)
    });

    zip_with!(
      (evals_r, evals_batch.iter(), powers_of_rho.iter()),
      |e_i, p_i, rho_i| e_i * *p_i * rho_i
    )
    .sum()
  };

  if claim_batch_final != claim_batch_final_expected {
    return Err(NovaError::InvalidSumcheckProof);
  }

  transcript.absorb(b"l", &evals_batch);

  // we now combine evaluation claims at the same point r into one
  let chal = transcript.squeeze(b"c")?;

  let comms = u_vec.into_iter().map(|u| u.c).collect::<Vec<_>>();

  let u_joint = PolyEvalInstance::batch_diff_size(&comms, evals_batch, &num_rounds, r, chal);

  Ok(u_joint)
}

#[cfg(test)]
mod batch_invert_tests {
  use super::{batch_invert, batch_invert_serial};
  use ff::Field;
  use rand::rngs::OsRng;
  use rayon::iter::IntoParallelIterator;
  use rayon::prelude::*;

  type F = halo2curves::bn256::Fr;

  #[test]
  fn test_batch_invert() {
    let n = (1 << 15) + 5;

    let v = (0..n)
      .into_par_iter()
      .map(|_| F::random(&mut OsRng))
      .collect::<Vec<_>>();

    let res_1 = batch_invert_serial(&v);
    let res_2 = batch_invert(&v);

    assert_eq!(res_1, res_2)
  }
}

#[cfg(test)]
mod batch_eval_tests {
  use super::{batch_eval_reduce, batch_eval_verify, PolyEvalInstance, PolyEvalWitness};
  use crate::provider::Bn256EngineKZG;
  use crate::spartan::polys::multilinear::MultilinearPolynomial;
  use crate::traits::{commitment::CommitmentEngineTrait, Engine, TranscriptEngineTrait};
  use ff::Field;

  type E = Bn256EngineKZG;

  /// Test batch_eval_reduce with polynomials of significantly different sizes
  /// to verify correctness of the batching protocol.
  #[test]
  fn test_batch_eval_reduce_different_sizes() {
    // Create polynomials of different sizes
    // P1: 4 variables (16 coefficients)
    // P2: 2 variables (4 coefficients)
    let p1: Vec<<E as Engine>::Scalar> = (0..16)
      .map(|i| <E as Engine>::Scalar::from(i as u64 + 1))
      .collect();
    let p2: Vec<<E as Engine>::Scalar> = (0..4)
      .map(|i| <E as Engine>::Scalar::from(i as u64 + 100))
      .collect();

    // Evaluation points
    let x1: Vec<<E as Engine>::Scalar> = (0..4)
      .map(|i| <E as Engine>::Scalar::from(i as u64 * 7 + 3))
      .collect();
    let x2: Vec<<E as Engine>::Scalar> = (0..2)
      .map(|i| <E as Engine>::Scalar::from(i as u64 * 11 + 5))
      .collect();

    // Compute evaluations
    let e1 = MultilinearPolynomial::new(p1.clone()).evaluate(&x1);
    let e2 = MultilinearPolynomial::new(p2.clone()).evaluate(&x2);

    // Create commitment key
    let ck = <<E as Engine>::CE as CommitmentEngineTrait<E>>::setup(b"test", 16).unwrap();

    // Commit to polynomials (using trivial blinder)
    let blinder = <E as Engine>::Scalar::ZERO;
    let c1 = <<E as Engine>::CE as CommitmentEngineTrait<E>>::commit(&ck, &p1, &blinder);
    let c2 = <<E as Engine>::CE as CommitmentEngineTrait<E>>::commit(&ck, &p2, &blinder);

    // Create instances and witnesses
    let u_vec: Vec<PolyEvalInstance<E>> = vec![
      PolyEvalInstance::new(c1, x1, e1),
      PolyEvalInstance::new(c2, x2, e2),
    ];
    let w_vec: Vec<PolyEvalWitness<E>> = vec![
      PolyEvalWitness::new(p1.clone()),
      PolyEvalWitness::new(p2.clone()),
    ];

    // Run batch_eval_reduce
    let mut transcript_prover = <E as Engine>::TE::new(b"test_batch_eval");
    let (u_joint, _w_joint, _chal, sc_proof, evals_batch) =
      batch_eval_reduce(u_vec.clone(), w_vec, &mut transcript_prover).unwrap();

    // Run batch_eval_verify
    let mut transcript_verifier = <E as Engine>::TE::new(b"test_batch_eval");
    let u_joint_verify =
      batch_eval_verify(u_vec, &mut transcript_verifier, &sc_proof, &evals_batch).unwrap();

    // Check that prover and verifier agree
    assert_eq!(u_joint.c(), u_joint_verify.c(), "Commitments don't match");
    assert_eq!(
      u_joint.x(),
      u_joint_verify.x(),
      "Evaluation points don't match"
    );
    assert_eq!(u_joint.e(), u_joint_verify.e(), "Evaluations don't match");
  }
}
