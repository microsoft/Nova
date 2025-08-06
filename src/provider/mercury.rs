//! This module implements Nova's evaluation engine using `Mercury` (<https://eprint.iacr.org/2025/385.pdf>)
//! Mercury is a pairing-based polynomial commitment scheme for multilinear polynomials.
//!
//! For a polynomial of size n, the construction of an opening proof requires O(n) field operations, and 2n + O(√n) scalar multiplications.
//! The opening proof size is constant.
//! The verification consists of O(log n) field operations and O(1) scalar multiplications, along with 2 pairings.
//!
//! The batch KZG evaluation tech is from BDFG20 <https://eprint.iacr.org/2020/081.pdf>
//!
//! Mercury and HyperKZG share the same types of commitment key and engine.
//!
//! Samaritan presents a similar construction and achieves the same performance, see <https://eprint.iacr.org/2025/419.pdf>.

use std::{cmp::max, marker::PhantomData};

use ff::{Field, PrimeField};
use halo2curves::fft::best_fft;
use rayon::iter::{
  IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
  IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};

use crate::provider::traits::DlogGroupExt;
use crate::{
  errors::NovaError,
  provider::{
    hyperkzg,
    traits::{DlogGroup, PairingGroup},
  },
  spartan::polys::{
    eq::EqPolynomial,
    univariate::{gaussian_elimination, UniPoly},
  },
  traits::{
    commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine,
    TranscriptEngineTrait,
  },
};

// Transcript absorb/squeeze labels used by both prover and verifier
mod transcript_labels {
  pub const LABEL_F: &[u8] = b"f";
  pub const LABEL_U: &[u8] = b"u";
  pub const LABEL_E: &[u8] = b"e";
  pub const LABEL_H: &[u8] = b"h";
  pub const LABEL_Q: &[u8] = b"q";
  pub const LABEL_G: &[u8] = b"g";
  pub const LABEL_S: &[u8] = b"s";
  pub const LABEL_D: &[u8] = b"d";
  pub const LABEL_QUOT_F: &[u8] = b"t";
  pub const LABEL_GZ: &[u8] = b"gz";
  pub const LABEL_GZI: &[u8] = b"gzi";
  pub const LABEL_HZ: &[u8] = b"hz";
  pub const LABEL_HZI: &[u8] = b"hzi";
  pub const LABEL_SZ: &[u8] = b"sz";
  pub const LABEL_SZI: &[u8] = b"szi";
  pub const LABEL_W: &[u8] = b"w";
  pub const LABEL_W_PRIME: &[u8] = b"wp";

  pub const LABEL_ALPHA: &[u8] = b"a";
  pub const LABEL_GAMMA: &[u8] = b"gm";
  pub const LABEL_ZETA: &[u8] = b"zt";
  pub const LABEL_BETA: &[u8] = b"b";
  pub const LABEL_Z: &[u8] = b"z";
  pub const LABEL_PAIRING_D: &[u8] = b"pd";
}

type ProverKey<E> = hyperkzg::ProverKey<E>;
type VerifierKey<E> = hyperkzg::VerifierKey<E>;
type Commitment<E> = hyperkzg::Commitment<E>;
type CommitmentEngine<E> = hyperkzg::CommitmentEngine<E>;

/// Provides an implementation of a polynomial evaluation engine using KZG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

/// Provides an implementation of a polynomial evaluation argument
/// The proof consists of 8 field elements (F) for committed values and 6 field elements (F) for evaluation values,
/// corresponding to the structure of the EvaluationArgument. This reflects the number of field elements included in the proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EvaluationArgument<E: Engine>
where
  E::GE: PairingGroup,
{
  comm_h: Commitment<E>,

  comm_g: Commitment<E>,
  comm_q: Commitment<E>,

  comm_s: Commitment<E>,

  comm_d: Commitment<E>,

  comm_quot_f: Commitment<E>,

  comm_w: Commitment<E>,
  comm_w_prime: Commitment<E>,

  g_zeta: E::Scalar,
  g_zeta_inv: E::Scalar,

  h_zeta: E::Scalar,
  h_zeta_inv: E::Scalar,

  s_zeta: E::Scalar,
  s_zeta_inv: E::Scalar,
}

fn omega<Scalar: PrimeField>(log_n: u32) -> Scalar {
  Scalar::ROOT_OF_UNITY.pow([1_u64 << (Scalar::S - log_n)])
}

impl<Scalar: PrimeField> UniPoly<Scalar> {
  fn expand(&mut self, size: usize) {
    assert!(self.coeffs.len() <= size);
    self.coeffs.resize(size, Scalar::ZERO);
  }

  pub fn scale(&mut self, s: &Scalar) {
    self.coeffs.par_iter_mut().for_each(|c| *c *= *s);
  }

  // Interpolate a polynomial from its evaluations and the corresponding points
  // Only linear or quadratic polynomials are supported
  // Adapted from `UniPoly::from_evals`
  fn from_evals_with_xs(xs: &[&Scalar], evals: &[Scalar]) -> Self {
    if evals.len() == 1 {
      return Self {
        coeffs: vec![evals[0]],
      };
    }

    assert_eq!(xs.len(), evals.len());
    let n = evals.len();

    let mut matrix: Vec<Vec<Scalar>> = Vec::with_capacity(n);
    for i in 0..n {
      let mut row = Vec::with_capacity(n);
      let x = xs[i];
      row.push(Scalar::ONE);
      row.push(*x);
      for j in 2..n {
        row.push(row[j - 1] * x);
      }
      row.push(evals[i]);
      matrix.push(row);
    }

    let coeffs = gaussian_elimination(&mut matrix);
    Self { coeffs }
  }

  fn trim(&mut self) {
    while !self.coeffs.is_empty() && self.coeffs.last().unwrap() == &Scalar::ZERO {
      self.coeffs.pop();
    }
  }

  fn batch_add_with_polynomials(&mut self, polynomials: Vec<&[Scalar]>, scalars: &[Scalar]) {
    let rhs_max_len = polynomials.iter().map(|p| p.len()).max().unwrap();
    self.expand(max(self.coeffs.len(), rhs_max_len));

    self
      .coeffs
      .par_iter_mut()
      .take(rhs_max_len)
      .enumerate()
      .for_each(|(coeff_id, coeff)| {
        for (rhs_polynomial, scalar) in polynomials.iter().zip(scalars) {
          if coeff_id < rhs_polynomial.len() {
            let scalar = *scalar;
            let rhs = rhs_polynomial[coeff_id];
            if scalar == -Scalar::ONE {
              *coeff -= rhs;
            } else if scalar != Scalar::ZERO && rhs != Scalar::ZERO {
              *coeff += scalar * rhs;
            }
          }
        }
      });
  }

  // f(X) * (X + a)
  fn multiply_by_linear_polynomial(&mut self, a: &Scalar) {
    let a_f = self.coeffs.par_iter().map(|c| *c * *a).collect::<Vec<_>>();
    let n = self.coeffs.len();

    self.coeffs.insert(0, Scalar::ZERO);

    self
      .coeffs
      .par_iter_mut()
      .take(n)
      .zip(a_f)
      .for_each(|(c, rhs)| *c += rhs);
  }

  // f(X) / (X - a)
  // Returns the remainder
  // Horner's method
  fn divide_by_linear_polynomial(&mut self, a: &Scalar) -> Scalar {
    for i in (0..self.coeffs.len() - 1).rev() {
      let last = self.coeffs[i + 1] * *a;
      self.coeffs[i] += last;
    }

    self.coeffs.remove(0)
  }

  // Transpose a matrix of size num_rows * num_cols in-place
  fn transpose(&mut self, num_rows: usize, num_cols: usize) {
    assert!(num_rows <= num_cols);

    let b = num_cols;
    self.expand(b * b);

    let new_coeffs = (0..b)
      .into_par_iter()
      .map(|c| {
        self
          .coeffs
          .iter()
          .skip(c)
          .step_by(b)
          .copied()
          .collect::<Vec<_>>()
      })
      .flatten()
      .collect();

    self.coeffs = new_coeffs;
  }
}

// f(X) / (X^{num_cols} - alpha)
// returns the quotient and the remainder polynomial
// See Mercury 5. Univariate division
#[inline]
fn divide_by_binomial<Scalar: PrimeField>(
  coeffs: &[Scalar],
  num_rows: usize,
  num_cols: usize,
  alpha: &Scalar,
) -> (UniPoly<Scalar>, UniPoly<Scalar>) {
  let (quotients, remainder): (Vec<Vec<Scalar>>, Vec<Scalar>) = (0..num_cols)
    .into_par_iter()
    .map(|col_id| {
      let mut quotient = UniPoly {
        coeffs: coeffs
          .iter()
          .skip(col_id)
          .step_by(num_cols)
          .copied()
          .collect(),
      };

      assert_eq!(quotient.coeffs.len(), num_rows);

      let remainder = quotient.divide_by_linear_polynomial(alpha);

      quotient.expand(num_cols);

      (quotient.coeffs, remainder)
    })
    .unzip();

  let mut quotient = UniPoly {
    coeffs: quotients.into_iter().flatten().collect(),
  };

  quotient.transpose(num_rows, num_cols);

  let remainder = UniPoly { coeffs: remainder };

  (quotient, remainder)
}

/// Evaluate eq(r, u)
fn eval_pu_poly<Scalar: PrimeField>(u: &[Scalar], r: &Scalar) -> Scalar {
  let mut res = Scalar::ONE;
  for (i, u_i) in u.iter().rev().enumerate() {
    res *= *u_i * r.pow([1 << i]) + Scalar::ONE - u_i;
  }
  res
}

/// Compute polynomial h, See Mercury Section 2
#[inline]
fn compute_h_poly<Scalar: PrimeField>(
  f_poly: &[Scalar],
  eq_col: &[Scalar],
  num_rows: usize,
  num_cols: usize,
) -> UniPoly<Scalar> {
  let coeffs = (0..num_rows)
    .into_par_iter()
    .map(|row_id| {
      (0..num_cols)
        .into_par_iter()
        .map(|col_id| f_poly[row_id * num_cols + col_id] * eq_col[col_id])
        .sum::<Scalar>()
    })
    .collect();

  UniPoly { coeffs }
}

/// Compute polynomial s for inner product argument
/// See Mercury 4.1 and 4.2
#[inline]
fn make_s_polynomial<Scalar: PrimeField>(
  a_polys: (&[Scalar], &[Scalar]),
  b_polys: (&[Scalar], &[Scalar]),
  log_b: u32,
  gamma: &Scalar,
) -> UniPoly<Scalar> {
  let b = 1 << log_b;
  let b2 = b * 2;

  let omega = omega::<Scalar>(log_b + 1);

  let (a_poly_1, a_poly_2) = a_polys;
  let (b_poly_1, b_poly_2) = b_polys;

  assert_eq!(a_poly_1.len(), b);
  assert_eq!(a_poly_2.len(), b);
  assert_eq!(b_poly_1.len(), b);
  assert_eq!(b_poly_2.len(), b);

  let mut evals = [
    a_poly_1.to_vec(),
    a_poly_2.to_vec(),
    b_poly_1.to_vec(),
    b_poly_2.to_vec(),
  ];

  // Coefficients To Evaluations
  let [a_evals_1, a_evals_2, b_evals_1, b_evals_2] = {
    evals
      .par_iter_mut()
      .for_each(|a| a.resize(b2, Scalar::ZERO));

    evals
      .par_iter_mut()
      .for_each(|a| best_fft(a, omega, log_b + 1));

    evals
  };

  let mut evals = vec![Scalar::ZERO; b2];

  // Evaluate a1(X) * b1(1/X) + a1(1/X) * b1(X)
  // + (a2(X) * b2(1/X) + a2(1/X) * b2(X)) * gamma
  {
    evals[0] = a_evals_1[0] * b_evals_1[0] + a_evals_2[0] * b_evals_2[0] * gamma;
    evals[0] = evals[0] + evals[0];

    evals.par_iter_mut().skip(1).enumerate().for_each(|(i, a)| {
      let i = i + 1;
      let s1 = a_evals_1[i] * b_evals_1[b2 - i] + a_evals_1[b2 - i] * b_evals_1[i];
      let s2 = a_evals_2[i] * b_evals_2[b2 - i] + a_evals_2[b2 - i] * b_evals_2[i];

      *a = s1 + s2 * gamma
    });
  }

  // Evaluate X^{b-1} * LastPolynomial
  {
    let omega_n_1 = omega.pow([b as u64 - 1]);
    let mut omega_pow = omega_n_1;
    for v in evals.iter_mut().skip(1) {
      *v *= omega_pow;
      omega_pow *= omega_n_1;
    }
  }

  // Evaluations To Coefficients
  let mut res = {
    best_fft(&mut evals, omega.invert().unwrap(), log_b + 1);

    let mut res = UniPoly { coeffs: evals };
    res.trim();

    let b_inv = Scalar::from(b2 as u64).invert().unwrap();
    res.coeffs.par_iter_mut().for_each(|x| *x *= b_inv);

    res
  };

  assert!(res.coeffs.len() < b2);

  res.coeffs.drain(..b);

  res
}

/// BDFG20 4.1
mod batch_evaluation {
  use ff::{Field, PrimeField};
  use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator,
  };

  use crate::provider::mercury::Commitment;
  use crate::provider::traits::DlogGroup;
  use crate::provider::traits::DlogGroupExt;
  use crate::{
    errors::NovaError,
    provider::{mercury::CommitmentEngine, traits::PairingGroup},
    spartan::polys::univariate::UniPoly,
    traits::{commitment::CommitmentEngineTrait, Engine, TranscriptEngineTrait},
  };

  pub struct BatchEvaluations<Scalar: PrimeField> {
    pub g_zeta: Scalar,
    pub g_zeta_inv: Scalar,
    pub h_zeta: Scalar,
    pub h_zeta_inv: Scalar,
    pub h_alpha: Scalar,
    pub s_zeta: Scalar,
    pub s_zeta_inv: Scalar,
    pub d_zeta: Scalar,

    pub zeta: Scalar,
    pub zeta_inv: Scalar,
    pub alpha: Scalar,
  }

  pub struct BatchEvaluationCommitments<E: Engine> {
    pub comm_g: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    pub comm_h: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    pub comm_s: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    pub comm_d: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,

    pub comm_w: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    pub comm_w_prime: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
  }

  pub struct BatchEvaluationInput<'a, Scalar: PrimeField> {
    pub evals: BatchEvaluations<Scalar>,

    pub g_poly: &'a UniPoly<Scalar>,
    pub h_poly: &'a UniPoly<Scalar>,
    pub s_poly: &'a UniPoly<Scalar>,
    pub d_poly: &'a UniPoly<Scalar>,
  }

  pub struct BatchArg<E: Engine>
  where
    E::GE: PairingGroup,
  {
    pub comm_w: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    pub comm_w_prime: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
  }

  /// Implementation of the opening procedure from BDFG20 Section 4.1
  #[inline]
  pub fn generate_batch_evaluate_arg<E: Engine>(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
    data: BatchEvaluationInput<'_, E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<BatchArg<E>, NovaError>
  where
    E::GE: PairingGroup,
  {
    use super::batch_evaluation::*;
    use super::transcript_labels::*;

    let zeta = data.evals.zeta;
    let zeta_inv = data.evals.zeta_inv;
    let alpha = data.evals.alpha;

    let evals = data.evals;

    // * BDFG20 4.1 - 1.
    // Random challenge `beta`, which is `gamma` in BDFG20 4.1
    let beta = transcript.squeeze(LABEL_BETA)?;
    let beta_2 = beta * beta;
    let beta_3 = beta_2 * beta;

    // Interpolate g_star, h_star, s_star, d_star
    // See BDFG20 Definition 2.3 - "open" 6.
    let [g_star, h_star, s_star, d_star] = {
      let eval_domains = vec![
        vec![&zeta, &zeta_inv],
        vec![&zeta, &zeta_inv, &alpha],
        vec![&zeta, &zeta_inv],
        vec![&zeta],
      ];

      let evals = vec![
        vec![evals.g_zeta, evals.g_zeta_inv],
        vec![evals.h_zeta, evals.h_zeta_inv, evals.h_alpha],
        vec![evals.s_zeta, evals.s_zeta_inv],
        vec![evals.d_zeta],
      ];

      eval_domains
        .into_par_iter()
        .zip(evals)
        .map(|(xs, evals)| UniPoly::from_evals_with_xs(&xs, &evals))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
    };

    // * BDFG20 4.1 - 2.
    // Compute polynomial m(X) = sum{ beta_pow * ( poly(X) - poly_star(X) ) }
    // `m(X)`` is `f(X)` in BDFG20 4.1
    let m_poly = {
      // m(X) =
      //             z_poly_t_s1(X) * (g(X) - g*(X))
      // + beta   * (z_poly_t_s2(X) * (h(X) - h*(X)))
      // + beta^2 * (z_poly_t_s3(X) * (s(X) - s*(X)))
      // + beta^3 * (z_poly_t_s4(X) * (d(X) - d*(X)))

      let mut polys = [
        data.g_poly.clone(),
        data.h_poly.clone(),
        data.s_poly.clone(),
        data.d_poly.clone(),
      ];

      let rhs_polys = [
        &g_star.coeffs,
        &h_star.coeffs,
        &s_star.coeffs,
        &d_star.coeffs,
      ];

      polys.par_iter_mut().zip(rhs_polys).for_each(|(poly, rhs)| {
        poly.batch_add_with_polynomials(vec![rhs], &[-E::Scalar::ONE]);
      });

      let vanishing_points_t_s = [vec![alpha], vec![], vec![alpha], vec![alpha, zeta_inv]];

      polys
        .par_iter_mut()
        .zip(vanishing_points_t_s)
        .for_each(|(poly, vanishing_points)| {
          for p in vanishing_points {
            poly.multiply_by_linear_polynomial(&-p);
          }
        });

      let [g, h, s, d] = polys;

      let mut m_poly = g;

      m_poly.batch_add_with_polynomials(
        vec![&h.coeffs, &s.coeffs, &d.coeffs],
        &[beta, beta_2, beta_3],
      );

      m_poly
    };

    // Compute quotient of m(X): quot_m(X) = m(X) / (X - alpha) / (X - zeta) / (X - zeta_inv)
    let mut quot_m_poly = m_poly;
    let rem = quot_m_poly.divide_by_linear_polynomial(&alpha);
    assert_eq!(rem, E::Scalar::ZERO);
    let rem = quot_m_poly.divide_by_linear_polynomial(&zeta);
    assert_eq!(rem, E::Scalar::ZERO);
    let rem = quot_m_poly.divide_by_linear_polynomial(&zeta_inv);
    assert_eq!(rem, E::Scalar::ZERO);

    // W(X) = quot_m(X)
    let comm_w = E::CE::commit(ck, &quot_m_poly.coeffs, &E::Scalar::ZERO);
    transcript.absorb(LABEL_W, &[comm_w].to_vec().as_slice());

    // * BDFG20 4.1 - 3.
    // Random evaluation point z
    let z = transcript.squeeze(LABEL_Z)?;

    // * BDFG20 4.1 - 4.
    // Compute polynomial L(X) = m_z(X) - Z_T(z) * W(X)
    // m_z(X) = sum { beta_pow * ( z_poly_t_si(z) * (poly(X) - poly_star(z))  ) }
    let l_poly = {
      let t_s1_eval_at_z = z - alpha;
      let t_s2_eval_at_z = E::Scalar::ONE;
      let t_s3_eval_at_z = t_s1_eval_at_z;
      let t_s4_eval_at_z = t_s1_eval_at_z * (z - zeta_inv);
      let t_eval_at_z = t_s4_eval_at_z * (z - zeta);

      // Compute mz(X)
      // =          z_poly_t_s1(z) * (g_poly(X) - g_star(z))
      // + beta   * z_poly_t_s2(z) * (h_poly(X) - h_star(z))
      // + beta^2 * z_poly_t_s3(z) * (s_poly(X) - s_star(z))
      // + beta^3 * z_poly_t_s4(z) * (d_poly(X) - d_star(z))
      //           lhs                   rhs

      let mz_poly = {
        let [g_poly_z, h_poly_z, s_poly_z, d_poly_z] = {
          let poly_star_eval = [&g_star, &h_star, &s_star, &d_star]
            .into_par_iter()
            .map(|poly| poly.evaluate(&z))
            .collect::<Vec<_>>();

          let mut polys = [
            data.g_poly.clone(),
            data.h_poly.clone(),
            data.s_poly.clone(),
            data.d_poly.clone(),
          ];

          polys
            .par_iter_mut()
            .zip(poly_star_eval)
            .for_each(|(poly, star_eval)| {
              poly.coeffs[0] -= star_eval;
            });

          polys
        };

        let scalars = [
          t_s1_eval_at_z,
          t_s2_eval_at_z * beta,
          t_s3_eval_at_z * beta_2,
          t_s4_eval_at_z * beta_3,
        ];

        let mut mz_poly = g_poly_z;
        mz_poly.scale(&scalars[0]);

        mz_poly.batch_add_with_polynomials(
          vec![&h_poly_z.coeffs, &s_poly_z.coeffs, &d_poly_z.coeffs],
          &[scalars[1], scalars[2], scalars[3]],
        );

        mz_poly
      };

      // L(X) = mz(x) - zt(z) * quot_m(X)
      {
        let mut lhs = mz_poly;

        lhs.batch_add_with_polynomials(vec![&quot_m_poly.coeffs], &[-t_eval_at_z]);

        lhs
      }
    };

    // L(X) / (X - z)
    let quot_l_poly = {
      let mut quot_l_poly = l_poly;

      let rem = quot_l_poly.divide_by_linear_polynomial(&z);
      assert_eq!(rem, E::Scalar::ZERO);

      quot_l_poly
    };

    // W'(X) = quot_l(X)
    let comm_w_prime = E::CE::commit(ck, &quot_l_poly.coeffs, &E::Scalar::ZERO);

    Ok(BatchArg {
      comm_w,
      comm_w_prime,
    })
  }

  /// BDFG20 4.1 - 5
  pub fn extract_pairing_to_verify_batch_evaluation<E>(
    evals: BatchEvaluations<E::Scalar>,
    cm: BatchEvaluationCommitments<E>,
    g1: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    transcript: &mut E::TE,
  ) -> Result<
    (
      <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
      <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    ),
    NovaError,
  >
  where
    E: Engine<CE = CommitmentEngine<E>>,
    E::GE: PairingGroup,
  {
    use super::transcript_labels::*;

    let zeta = evals.zeta;
    let zeta_inv = evals.zeta_inv;
    let alpha = evals.alpha;

    let beta = transcript.squeeze(LABEL_BETA)?;
    let beta_2 = beta.pow([2]);
    let beta_3 = beta.pow([3]);

    transcript.absorb(LABEL_W, &[cm.comm_w].to_vec().as_slice());

    let z = transcript.squeeze(LABEL_Z)?;

    let g_star =
      UniPoly::from_evals_with_xs(&[&zeta, &zeta_inv], &[evals.g_zeta, evals.g_zeta_inv]);
    let h_star = UniPoly::from_evals_with_xs(
      &[&zeta, &zeta_inv, &alpha],
      &[evals.h_zeta, evals.h_zeta_inv, evals.h_alpha],
    );
    let s_star =
      UniPoly::from_evals_with_xs(&[&zeta, &zeta_inv], &[evals.s_zeta, evals.s_zeta_inv]);
    let d_star = UniPoly::from_evals_with_xs(&[&zeta], &[evals.d_zeta]);

    let g_star_eval = g_star.evaluate(&z);
    let h_star_eval = h_star.evaluate(&z);
    let s_star_eval = s_star.evaluate(&z);
    let d_star_eval = d_star.evaluate(&z);

    let van_zeta = z - zeta;
    let van_zeta_inv = z - zeta_inv;
    let van_alpha = z - alpha;

    let z_eval_t_s1 = van_alpha;
    let z_eval_t_s2 = E::Scalar::ONE;
    let z_eval_t_s3 = van_alpha;
    let z_eval_t_s4 = van_zeta_inv * van_alpha;
    let z_eval_t = z_eval_t_s4 * van_zeta;

    // F = sum{ beta_pow * z_eval_t_si * cm_poly_i } - [ sum{ beta_pow * z_eval_t_si * star_eval_at_z } ] - z_eval_t * comm_W
    // LHS = F + comm_w_prime * z
    // See BDFG20 4.1 - 6
    let scalars = {
      let scalar = z_eval_t_s1 * g_star_eval
        + beta * z_eval_t_s2 * h_star_eval
        + beta_2 * z_eval_t_s3 * s_star_eval
        + beta_3 * z_eval_t_s4 * d_star_eval;

      [
        z_eval_t_s1,
        beta * z_eval_t_s2,
        beta_2 * z_eval_t_s3,
        beta_3 * z_eval_t_s4,
        -scalar,
        -z_eval_t,
        z,
      ]
    };

    let bases = [
      cm.comm_g,
      cm.comm_h,
      cm.comm_s,
      cm.comm_d,
      g1,
      cm.comm_w,
      cm.comm_w_prime,
    ]
    .into_iter()
    .map(|b| b.into_inner().affine())
    .collect::<Vec<_>>();

    // Pairing 1 lhs is F + [z] comm_W_prime
    // Pairing 1 rhs is g1
    // Pairing 2 lhs = comm_W_prime
    // Pairing 2 rhs is [tau]_2

    // * Main Cost (Verifier) II: MSM of 7
    let lhs1 = Commitment::new(E::GE::vartime_multiscalar_mul(&scalars, &bases));
    let lhs2 = cm.comm_w_prime;

    Ok((lhs1, lhs2))
  }
}

impl<E: Engine> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine<CE = CommitmentEngine<E>>,
  E::GE: PairingGroup,
{
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;
  type EvaluationArgument = EvaluationArgument<E>;

  /// Reuse the setup from hyperkzg
  fn setup(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    hyperkzg::EvaluationEngine::setup(ck)
  }

  /// A method to prove the evaluation of a multilinear polynomial
  fn prove(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
    _pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comm: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    poly: &[E::Scalar],
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    use batch_evaluation::*;
    use transcript_labels::*;

    let comm_f = comm;

    transcript.absorb(LABEL_F, &[*comm_f].to_vec().as_slice());
    transcript.absorb(LABEL_U, &point.to_vec().as_slice());
    transcript.absorb(LABEL_E, &[*eval].to_vec().as_slice());

    let original_size = poly.len();

    let (f_poly, log_b, b, point) = {
      let mut log_n = point.len();

      assert!(log_n > 1);

      let mut point = point.to_vec();
      let mut f_poly = poly.to_vec();

      if log_n % 2 == 1 {
        log_n += 1;
        point.insert(0, E::Scalar::ZERO);
        f_poly.resize(1 << log_n, E::Scalar::ZERO);
      }
      let log_b = log_n / 2;
      let b = 1 << log_b;

      (f_poly, log_b, b, point)
    };

    let (u_row, u_col) = point.split_at(log_b);
    let u_row = u_row.to_owned();
    let u_col = u_col.to_owned();

    let eq_row = EqPolynomial::new(u_row.clone()).evals();
    let eq_col = EqPolynomial::new(u_col.clone()).evals();

    #[cfg(debug_assertions)]
    {
      // Check pu_row(X), pu_col(X)
      use rand_core::OsRng;

      use crate::spartan::polys::univariate::UniPoly;

      let r = E::Scalar::random(OsRng);
      let pu_row_eval_expected = eval_pu_poly(&u_row, &r);
      let pu_col_eval_expected = eval_pu_poly(&u_col, &r);

      let pu_row_eval_actual = UniPoly {
        coeffs: eq_row.clone(),
      }
      .evaluate(&r);
      let pu_col_eval_actual = UniPoly {
        coeffs: eq_col.clone(),
      }
      .evaluate(&r);

      assert_eq!(pu_row_eval_expected, pu_row_eval_actual);
      assert_eq!(pu_col_eval_expected, pu_col_eval_actual);
    }

    // * 1. Mercury Section 6. Step 1. (a)
    // Compute h(X)
    let h_poly = compute_h_poly(&f_poly, &eq_col, b, b);

    #[cfg(debug_assertions)]
    {
      // Check h, eq_evals_row ipa vs eval
      assert_eq!(eq_row.len(), b);

      let inner_product = eq_row
        .iter()
        .zip(h_poly.coeffs.iter())
        .map(|(a, b)| *a * *b)
        .sum::<E::Scalar>();

      assert_eq!(inner_product, *eval);
    }

    // * 2. Mercury Section 6. Step 1. (b)
    let comm_h = E::CE::commit(ck, &h_poly.coeffs, &E::Scalar::ZERO);
    transcript.absorb(LABEL_H, &[comm_h].to_vec().as_slice());

    // * 3. Mercury Section 6. Step 2. (a)
    let alpha = transcript.squeeze(LABEL_ALPHA)?;

    // * 4. Mercury Section 6. Step 2. (b)
    // Compute q(X) and g(X)
    let (mut q_poly, g_poly) = divide_by_binomial(&f_poly, b, b, &alpha);

    q_poly.trim();

    assert_eq!(g_poly.coeffs.len(), b);

    #[cfg(debug_assertions)]
    {
      // Check g
      let f_alphas = (0..b)
        .into_par_iter()
        .map(|col_id| {
          use crate::spartan::polys::univariate::UniPoly;

          let coeffs = (0..b)
            .into_par_iter()
            .map(|row_id| f_poly[row_id * b + col_id])
            .collect::<Vec<_>>();

          UniPoly { coeffs }.evaluate(&alpha)
        })
        .collect::<Vec<_>>();

      f_alphas
        .iter()
        .zip(g_poly.coeffs.iter())
        .for_each(|(a, b)| {
          assert_eq!(*b, *a);
        });
    }

    #[cfg(debug_assertions)]
    {
      // Check q, g
      // f(r) = (r^b - alpha) * q(r) + g(r)

      use rand_core::OsRng;
      let r = E::Scalar::random(OsRng);

      let f_r = UniPoly {
        coeffs: f_poly.to_vec(),
      }
      .evaluate(&r);
      let q_r = q_poly.evaluate(&r);
      let g_r = g_poly.evaluate(&r);

      assert_eq!(f_r, (r.pow([b as u64]) - alpha) * q_r + g_r);
    }

    // * 5. Mercury Section 6. Step 2. (c)
    // * Main Cost (Prover) I: MSM of O(N)
    let (comm_q, comm_g) = rayon::join(
      || E::CE::commit(ck, &q_poly.coeffs, &E::Scalar::ZERO),
      || E::CE::commit(ck, &g_poly.coeffs, &E::Scalar::ZERO),
    );
    transcript.absorb(LABEL_Q, &[comm_q].to_vec().as_slice());
    transcript.absorb(LABEL_G, &[comm_g].to_vec().as_slice());

    #[cfg(debug_assertions)]
    {
      // Check g eq_evals_col ipa vs h_alpha
      let h_alpha = h_poly.evaluate(&alpha);

      let ip = eq_col
        .iter()
        .zip(g_poly.coeffs.iter())
        .map(|(a, b)| *a * *b)
        .sum::<E::Scalar>();

      assert_eq!(ip, h_alpha);
    }

    // * 6. Mercury Section 6. Step 3. (a)
    let gamma = transcript.squeeze(LABEL_GAMMA)?;

    // * 7. Mercury Section 6. Step 3. (b)
    // Compute s(X)
    let s_poly = make_s_polynomial(
      (&eq_col, &eq_row),
      (&g_poly.coeffs, &h_poly.coeffs),
      log_b as u32,
      &gamma,
    );

    #[cfg(debug_assertions)]
    {
      // Check s_poly IPA proof

      use rand_core::OsRng;
      let r = E::Scalar::random(OsRng);
      let r_inv = r.invert().unwrap();

      let g_r = g_poly.evaluate(&r);
      let g_r_inv = g_poly.clone().evaluate(&r_inv);
      let h_r = h_poly.evaluate(&r);
      let h_r_inv = h_poly.evaluate(&r_inv);
      let pu_col_r = eval_pu_poly(&u_col, &r);
      let pu_col_r_inv = eval_pu_poly(&u_col, &r_inv);
      let pu_row_r = eval_pu_poly(&u_row, &r);
      let pu_row_r_inv = eval_pu_poly(&u_row, &r_inv);

      let h_alpha = h_poly.evaluate(&alpha);

      let s_r = s_poly.evaluate(&r);
      let s_r_inv = s_poly.evaluate(&r_inv);

      let mut lhs = g_r * pu_col_r_inv + g_r_inv * pu_col_r;
      lhs += gamma * (h_r * pu_row_r_inv + h_r_inv * pu_row_r);

      let mut rhs = h_alpha + gamma * eval;
      rhs += rhs;
      rhs += r * s_r + r_inv * s_r_inv;

      assert_eq!(lhs, rhs);
    }

    // * 8. Mercury Section 6. Step 3. (c)
    // Compute d(X) for degree check
    let d_poly = {
      let mut coeffs = Vec::with_capacity(g_poly.coeffs.len());
      coeffs.extend(g_poly.coeffs.iter().rev());

      assert_eq!(coeffs.len(), b);

      UniPoly { coeffs }
    };

    // * Send commitments of 7. and 8.
    let (comm_s, comm_d) = rayon::join(
      || E::CE::commit(ck, &s_poly.coeffs, &E::Scalar::ZERO),
      || E::CE::commit(ck, &d_poly.coeffs, &E::Scalar::ZERO),
    );

    transcript.absorb(LABEL_S, &[comm_s].to_vec().as_slice());
    transcript.absorb(LABEL_D, &[comm_d].to_vec().as_slice());

    // * 9. Mercury Section 6. Step 4. (a)
    let zeta = transcript.squeeze(LABEL_ZETA)?;

    let zeta_inv = &zeta.invert().unwrap();

    let [g_zeta, g_zeta_inv, h_zeta, h_zeta_inv, h_alpha, s_zeta, s_zeta_inv, d_zeta] = {
      let eval_domains = [
        vec![&zeta, &zeta_inv],
        vec![&zeta, &zeta_inv, &alpha],
        vec![&zeta, &zeta_inv],
        vec![&zeta],
      ];

      // Evaluate g(zeta), g(zeta_inv), h(zeta), h(zeta_inv), h(alpha), s(zeta), s(zeta_inv), d(zeta)

      let polys = [
        &g_poly, &g_poly, &h_poly, &h_poly, &h_poly, &s_poly, &s_poly, &d_poly,
      ];

      let xs = eval_domains.iter().flatten().collect::<Vec<_>>();

      polys
        .into_par_iter()
        .zip(xs)
        .map(|(poly, x)| poly.evaluate(x))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
    };

    // quot_f(X) = ( f(X) -  q(X) * (zeta^b - alpha) - g(zeta) ) / (X - zeta)
    let quot_f = {
      let zeta_b = zeta.pow([b as u64]);
      let zeta_b_alpha = zeta_b - alpha;

      let mut quot_f = UniPoly {
        coeffs: f_poly.to_vec(),
      };

      quot_f.batch_add_with_polynomials(vec![&q_poly.coeffs], &[-zeta_b_alpha]);

      quot_f.coeffs[0] -= g_zeta;

      let rem = quot_f.divide_by_linear_polynomial(&zeta);

      assert_eq!(rem, E::Scalar::ZERO);

      quot_f
    };

    #[cfg(debug_assertions)]
    {
      // Check quot_f
      use rand_core::OsRng;

      let r = E::Scalar::random(OsRng);

      let f_poly = UniPoly {
        coeffs: f_poly.to_vec(),
      };

      let f_r = f_poly.evaluate(&r);
      let zeta_b = zeta.pow([b as u64]);
      let zeta_b_alpha = zeta_b - alpha;
      let q_r = q_poly.evaluate(&r);
      let quot_f_r = quot_f.evaluate(&r);

      assert_eq!(quot_f_r * (r - zeta), f_r - zeta_b_alpha * q_r - g_zeta);
    }

    // * 10. Mercury Section 6. Step 4. (b)
    transcript.absorb(LABEL_GZ, &[g_zeta].to_vec().as_slice());
    transcript.absorb(LABEL_GZI, &[g_zeta_inv].to_vec().as_slice());
    transcript.absorb(LABEL_HZ, &[h_zeta].to_vec().as_slice());
    transcript.absorb(LABEL_HZI, &[h_zeta_inv].to_vec().as_slice());
    transcript.absorb(LABEL_SZ, &[s_zeta].to_vec().as_slice());
    transcript.absorb(LABEL_SZI, &[s_zeta_inv].to_vec().as_slice());

    // * 11. Mercury Section 6. Step 4. (d)
    // * Main Cost (Prover) II: MSM of O(N)
    let comm_quot_f = {
      let mut quot_f = quot_f;
      quot_f.coeffs.truncate(original_size);
      quot_f.trim();
      E::CE::commit(ck, &quot_f.coeffs, &E::Scalar::ZERO)
    };
    transcript.absorb(LABEL_QUOT_F, &[comm_quot_f].to_vec().as_slice());

    // * 12. Mercury Section 6. Step 4. (e)
    let BatchArg {
      comm_w,
      comm_w_prime,
    } = generate_batch_evaluate_arg::<E>(
      ck,
      BatchEvaluationInput {
        evals: BatchEvaluations {
          g_zeta,
          g_zeta_inv,
          h_zeta,
          h_zeta_inv,
          h_alpha,
          s_zeta,
          s_zeta_inv,
          d_zeta,
          zeta,
          zeta_inv: *zeta_inv,
          alpha,
        },
        g_poly: &g_poly,
        h_poly: &h_poly,
        s_poly: &s_poly,
        d_poly: &d_poly,
      },
      transcript,
    )?;

    // For Pairing Check Batch
    transcript.absorb(LABEL_W_PRIME, &[comm_w_prime].to_vec().as_slice());
    let _d = transcript.squeeze(LABEL_PAIRING_D)?;

    Ok(EvaluationArgument {
      comm_h,
      comm_g,
      comm_q,
      comm_s,
      comm_d,
      comm_w,
      comm_w_prime,
      comm_quot_f,
      g_zeta,
      g_zeta_inv,
      h_zeta,
      h_zeta_inv,
      s_zeta,
      s_zeta_inv,
    })
  }

  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    use batch_evaluation::*;
    use transcript_labels::*;

    let comm_f = comm;

    let (alpha, gamma, zeta) = {
      transcript.absorb(LABEL_F, &[*comm_f].to_vec().as_slice());

      transcript.absorb(LABEL_U, &point.to_vec().as_slice());

      transcript.absorb(LABEL_E, &[*eval].to_vec().as_slice());

      transcript.absorb(LABEL_H, &[arg.comm_h].to_vec().as_slice());

      // * 1. Mercury Section 6. Step 2. (a)
      let alpha = transcript.squeeze(LABEL_ALPHA)?;

      // * Mercury Section 6. Step 1. (b)
      transcript.absorb(LABEL_Q, &[arg.comm_q].to_vec().as_slice());

      // * Mercury Section 6. Step 2. (c)
      transcript.absorb(LABEL_G, &[arg.comm_g].to_vec().as_slice());

      // * 2. Mercury Section 6. Step 3. (a)
      let gamma = transcript.squeeze(LABEL_GAMMA)?;

      // * Mercury Section 6. Step 3. (b)
      transcript.absorb(LABEL_S, &[arg.comm_s].to_vec().as_slice());

      // * Mercury Section 6. Step 3. (c)
      transcript.absorb(LABEL_D, &[arg.comm_d].to_vec().as_slice());

      // * 3. Mercury Section 6. Step 4. (a)
      let zeta = transcript.squeeze(LABEL_ZETA)?;

      // * Mercury Section 6. Step 4. (c)
      transcript.absorb(LABEL_GZ, &[arg.g_zeta].to_vec().as_slice());
      transcript.absorb(LABEL_GZI, &[arg.g_zeta_inv].to_vec().as_slice());
      transcript.absorb(LABEL_HZ, &[arg.h_zeta].to_vec().as_slice());
      transcript.absorb(LABEL_HZI, &[arg.h_zeta_inv].to_vec().as_slice());
      transcript.absorb(LABEL_SZ, &[arg.s_zeta].to_vec().as_slice());
      transcript.absorb(LABEL_SZI, &[arg.s_zeta_inv].to_vec().as_slice());

      // * Mercury Section 6. Step 4. (d)
      transcript.absorb(LABEL_QUOT_F, &[arg.comm_quot_f].to_vec().as_slice());

      (alpha, gamma, zeta)
    };

    let point = {
      let log_n = point.len();
      let mut point = point.to_vec();

      if log_n % 2 == 1 {
        point.insert(0, E::Scalar::ZERO);
      }

      point
    };

    let log_n = point.len();

    let u_row = point.split_at(log_n / 2).0.to_vec();
    let log_row = u_row.len();

    let u_col = point.split_at(log_n - log_row).1.to_vec();

    let zeta_inv = zeta.invert().unwrap();
    let zeta_b_one = zeta.pow_vartime([(1_u64 << (log_n / 2)) - 1]);

    // * O(log n) Field Operations
    let pu_col_zeta = eval_pu_poly(&u_col, &zeta);
    let pu_col_zeta_inv = eval_pu_poly(&u_col, &zeta_inv);
    let pu_row_zeta = eval_pu_poly(&u_row, &zeta);
    let pu_row_zeta_inv = eval_pu_poly(&u_row, &zeta_inv);

    // Compute `d_zeta`` and `h_alpha`
    // * 4. Mercury Section 6. Step 4. (c)

    // * Implicit Degree Check
    let d_zeta = zeta_b_one * arg.g_zeta_inv;

    // * Implicit IPA Check
    let h_alpha = {
      let denom = arg.g_zeta * pu_col_zeta_inv
        + arg.g_zeta_inv * pu_col_zeta
        + gamma * (arg.h_zeta * pu_row_zeta_inv + arg.h_zeta_inv * pu_row_zeta - eval - eval)
        - zeta * arg.s_zeta
        - zeta_inv * arg.s_zeta_inv;

      denom * (E::Scalar::ONE + E::Scalar::ONE).invert().unwrap()
    };

    #[cfg(debug_assertions)]
    {
      let mut lhs = arg.g_zeta * pu_col_zeta_inv + arg.g_zeta_inv * pu_col_zeta;
      lhs += gamma * (arg.h_zeta * pu_row_zeta_inv + arg.h_zeta_inv * pu_row_zeta);

      let mut rhs = h_alpha + gamma * *eval;
      rhs += rhs;
      rhs += zeta * arg.s_zeta + zeta_inv * arg.s_zeta_inv;

      assert_eq!(lhs, rhs);

      if lhs != rhs {
        return Err(NovaError::ProofVerifyError {
          reason: "IPA check failed".to_string(),
        });
      }
    }

    let g1 = Commitment::new(DlogGroup::group(&vk.G));
    let g2 = <<E as Engine>::GE as PairingGroup>::G2::gen();
    let tau2 = <E::GE as PairingGroup>::G2::group(&vk.tau_H);
    let lr = g2;
    let rr = tau2;

    // * Check f(X) / (X^b - alpha) = (q(X), g(x))
    // * 5. Adapted from Mercury Section 6. Step 4. (f)
    // The pairing check in Step 4. (f) is
    // Pairing 1 LHS = comm_f - [zeta^b - alpha] comm_q - [g(zeta)]_1
    // Pairing 1 RHS = [1]_2
    // Pairing 2 LHS = comm_quot_f
    // Pairing 2 RHS = [tau - zeta]_2
    //
    // It is adapted into the following pairing checks:
    // Pairing 1 LHS: comm_f + [zeta^b - alpha] comm_q - [g(zeta)]_1 + [zeta] * comm_quot_f
    // Pairing 1 RHS: g1
    // Pairing 2 LHS: comm_quot_f
    // Pairing 2 RHS: [tau]_2
    //
    // The verifier's checks are batched as follows:
    // Denote the above as LHS_1, RHS_1=g1, LHS_2, RHS_2=[tau]_2.
    // Denote the KZG pairing of BDFH20 (Step 4. (g)) as LHS_1', RHS_1'=g1, LHS_2', RHS_2'=[tau]_2.
    // The verifier samples a random d and checks
    // e(LHS_1 + LHS_1' * d, g1) = e(LHS_2 + LHS_2' * d, [tau]_2)
    let (lhs_1_1, lhs_2_1) = {
      let zeta_b = zeta_b_one * zeta;
      let zeta_b_alpha = zeta_b - alpha;

      let scalars = [-zeta_b_alpha, -arg.g_zeta, zeta];

      let bases = [arg.comm_q, g1, arg.comm_quot_f]
        .into_iter()
        .map(|b| b.into_inner().affine())
        .collect::<Vec<_>>();

      // * Main Cost (Verifier) I: MSM of 3
      let ll = *comm_f + Commitment::new(E::GE::vartime_multiscalar_mul(&scalars, &bases));
      let rl = arg.comm_quot_f;

      #[cfg(debug_assertions)]
      {
        let pairing_l = E::GE::pairing(&ll.into_inner(), &lr);
        let pairing_r = E::GE::pairing(&rl.into_inner(), &rr);

        assert!(pairing_l == pairing_r);
      }

      (ll, rl)
    };

    // * Check KZG
    // * 6. Mercury Section 6. Step 4. (g)
    let (lhs_1_2, lhs_2_2) = extract_pairing_to_verify_batch_evaluation::<E>(
      BatchEvaluations {
        zeta,
        zeta_inv,
        alpha,
        g_zeta: arg.g_zeta,
        g_zeta_inv: arg.g_zeta_inv,
        h_zeta: arg.h_zeta,
        h_zeta_inv: arg.h_zeta_inv,
        h_alpha,
        s_zeta: arg.s_zeta,
        s_zeta_inv: arg.s_zeta_inv,
        d_zeta,
      },
      BatchEvaluationCommitments::<E> {
        comm_g: arg.comm_g,
        comm_h: arg.comm_h,
        comm_s: arg.comm_s,
        comm_d: arg.comm_d,
        comm_w: arg.comm_w,
        comm_w_prime: arg.comm_w_prime,
      },
      g1,
      transcript,
    )?;

    // Check Pairing of 3. and 4.
    transcript.absorb(LABEL_W_PRIME, &[arg.comm_w_prime].to_vec().as_slice());
    let d = transcript.squeeze(LABEL_PAIRING_D)?;

    // * Main Cost (Verifier) III: MSM of 2
    let ll = lhs_1_1 + lhs_1_2 * d;
    let rl = lhs_2_1 + lhs_2_2 * d;

    let pairing_l = E::GE::pairing(&ll.into_inner(), &lr);
    let pairing_r = E::GE::pairing(&rl.into_inner(), &rr);

    if pairing_l != pairing_r {
      return Err(NovaError::ProofVerifyError {
        reason: "Pairing check failed".to_string(),
      });
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use ff::Field;
  use rand_core::OsRng;
  use rayon::iter::{IntoParallelIterator, ParallelIterator};

  use crate::spartan::polys::multilinear::MultilinearPolynomial;
  use crate::traits::commitment::CommitmentEngineTrait;
  use crate::traits::evaluation::EvaluationEngineTrait;
  use crate::traits::{Engine, TranscriptEngineTrait};
  use crate::{provider::Bn256EngineKZG, spartan::polys::univariate::UniPoly};

  type F = halo2curves::bn256::Fr;
  type E = Bn256EngineKZG;
  type EE = super::EvaluationEngine<E>;

  fn prove_and_verify<EE: EvaluationEngineTrait<E>>(log_n: usize) -> EE::EvaluationArgument {
    let n = 1 << log_n;
    let poly = UniPoly {
      coeffs: (0..n)
        .into_par_iter()
        .map(|_| F::random(OsRng))
        .collect::<Vec<_>>(),
    };
    let point = (0..log_n).map(|_| F::random(OsRng)).collect::<Vec<_>>();

    let ck = <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey::setup_from_rng(
      b"test", n, OsRng,
    );

    let (pk, vk) = EE::setup(&ck);

    let eval = MultilinearPolynomial::new(poly.coeffs.clone()).evaluate(&point);

    let mut transcript = <E as Engine>::TE::new(b"test");

    let comm = <E as Engine>::CE::commit(&ck, &poly.coeffs, &F::ZERO);

    let arg = EE::prove(
      &ck,
      &pk,
      &mut transcript,
      &comm,
      &poly.coeffs,
      &point,
      &eval,
    )
    .unwrap();

    let mut transcript = <E as Engine>::TE::new(b"test");
    EE::verify(&vk, &mut transcript, &comm, &point, &eval, &arg).unwrap();

    arg
  }

  #[test]
  fn test_mercury_evaluation_engine_15() {
    prove_and_verify::<EE>(15);
  }

  #[test]
  fn test_mercury_evaluation_engine_16() {
    prove_and_verify::<EE>(16);
  }
}
