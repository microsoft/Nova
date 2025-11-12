use crate::{
  errors::NovaError,
  spartan::{
    polys::{
      multilinear::MultilinearPolynomial,
      univariate::{CompressedUniPoly, UniPoly},
    },
    sumcheck::eq_sumcheck::EqSumCheckInstance,
  },
  traits::{Engine, TranscriptEngineTrait},
};
use ff::Field;
use itertools::Itertools as _;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Defines a trait for implementing sum-check in a generic manner
pub trait SumcheckEngine<E: Engine>: Send + Sync {
  /// returns the initial claims
  fn initial_claims(&self) -> Vec<E::Scalar>;

  /// degree of the sum-check polynomial
  fn degree(&self) -> usize;

  /// the size of the polynomials
  fn size(&self) -> usize;

  /// returns evaluation points at 0, 2, d-1 (where d is the degree of the sum-check polynomial)
  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>>;

  /// bounds a variable in the constituent polynomials
  fn bound(&mut self, r: &E::Scalar);

  /// returns the final claims
  fn final_claims(&self) -> Vec<Vec<E::Scalar>>;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct SumcheckProof<E: Engine> {
  compressed_polys: Vec<CompressedUniPoly<E::Scalar>>,
}

impl<E: Engine> SumcheckProof<E> {
  pub fn new(compressed_polys: Vec<CompressedUniPoly<E::Scalar>>) -> Self {
    Self { compressed_polys }
  }

  pub fn verify(
    &self,
    claim: E::Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut E::TE,
  ) -> Result<(E::Scalar, Vec<E::Scalar>), NovaError> {
    let mut e = claim;
    let mut r: Vec<E::Scalar> = Vec::new();

    // verify that there is a univariate polynomial for each round
    if self.compressed_polys.len() != num_rounds {
      return Err(NovaError::InvalidSumcheckProof);
    }

    for i in 0..self.compressed_polys.len() {
      let poly = self.compressed_polys[i].decompress(&e);

      // verify degree bound
      if poly.degree() != degree_bound {
        return Err(NovaError::InvalidSumcheckProof);
      }

      // we do not need to check if poly(0) + poly(1) = e, as
      // decompress() call above already ensures that holds
      debug_assert_eq!(poly.eval_at_zero() + poly.eval_at_one(), e);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;

      r.push(r_i);

      // evaluate the claimed degree-ell polynomial at r_i
      e = poly.evaluate(&r_i);
    }

    Ok((e, r))
  }

  pub fn verify_batch(
    &self,
    claims: &[E::Scalar],
    num_rounds: &[usize],
    coeffs: &[E::Scalar],
    degree_bound: usize,
    transcript: &mut E::TE,
  ) -> Result<(E::Scalar, Vec<E::Scalar>), NovaError> {
    let num_instances = claims.len();
    assert_eq!(num_rounds.len(), num_instances);
    assert_eq!(coeffs.len(), num_instances);

    // n = maxᵢ{nᵢ}
    let num_rounds_max = *num_rounds.iter().max().unwrap();

    // Random linear combination of claims,
    // where each claim is scaled by 2^{n-nᵢ} to account for the padding.
    //
    // claim = ∑ᵢ coeffᵢ⋅2^{n-nᵢ}⋅cᵢ
    let claim = zip_with!(
      (
        zip_with!(iter, (claims, num_rounds), |claim, num_rounds| {
          let scaling_factor = 1 << (num_rounds_max - num_rounds);
          E::Scalar::from(scaling_factor as u64) * claim
        }),
        coeffs.iter()
      ),
      |scaled_claim, coeff| scaled_claim * coeff
    )
    .sum();

    self.verify(claim, num_rounds_max, degree_bound, transcript)
  }

  // comb_func: |p, q| p * q
  #[inline]
  fn compute_eval_points_quad_prod(
    poly_A: &MultilinearPolynomial<E::Scalar>,
    poly_B: &MultilinearPolynomial<E::Scalar>,
  ) -> (E::Scalar, E::Scalar) {
    let len = poly_A.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = poly_A[i] * poly_B[i];

        // eval quadratic coefficient: (A(high)-A(low)) * (B(high)-B(low))
        let poly_A_bound_coeff = poly_A[len + i] - poly_A[i];
        let poly_B_bound_coeff = poly_B[len + i] - poly_B[i];
        let bound_coeff = poly_A_bound_coeff * poly_B_bound_coeff;
        (eval_point_0, bound_coeff)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1),
      )
  }

  // comb_func: |p, q| p * q
  pub fn prove_quad_prod(
    claim: &E::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), NovaError> {
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();
    let mut claim_per_round = *claim;
    for _ in 0..num_rounds {
      let poly = {
        let (eval_point_0, bound_coeff) = Self::compute_eval_points_quad_prod(poly_A, poly_B);

        let evals = vec![eval_point_0, claim_per_round - eval_point_0, bound_coeff];
        UniPoly::from_evals_deg2(&evals)
      };

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);
      polys.push(poly.compress());

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);

      // bind all tables to the verifier's challenge
      rayon::join(
        || poly_A.bind_poly_var_top(&r_i),
        || poly_B.bind_poly_var_top(&r_i),
      );
    }

    Ok((
      SumcheckProof {
        compressed_polys: polys,
      },
      r,
      vec![poly_A[0], poly_B[0]],
    ))
  }

  // comb_func: |p, q| p * q
  pub fn prove_quad_batch_prod(
    claims: &[E::Scalar],
    num_rounds: &[usize],
    mut poly_A_vec: Vec<MultilinearPolynomial<E::Scalar>>,
    mut poly_B_vec: Vec<MultilinearPolynomial<E::Scalar>>,
    coeffs: &[E::Scalar],
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, (Vec<E::Scalar>, Vec<E::Scalar>)), NovaError> {
    let num_claims = claims.len();

    assert_eq!(num_rounds.len(), num_claims);
    assert_eq!(poly_A_vec.len(), num_claims);
    assert_eq!(poly_B_vec.len(), num_claims);
    assert_eq!(coeffs.len(), num_claims);

    for (i, &num_rounds) in num_rounds.iter().enumerate() {
      let expected_size = 1 << num_rounds;

      // Direct indexing with the assumption that the index will always be in bounds
      let a = &poly_A_vec[i];
      let b = &poly_B_vec[i];

      for (l, polyname) in [(a.len(), "poly_A_vec"), (b.len(), "poly_B_vec")].iter() {
        assert_eq!(
          *l, expected_size,
          "Mismatch in size for {polyname} at index {i}"
        );
      }
    }

    let num_rounds_max = *num_rounds.iter().max().unwrap();
    let mut e = zip_with!(
      iter,
      (claims, num_rounds, coeffs),
      |claim, num_rounds, coeff| {
        let scaled_claim = E::Scalar::from((1 << (num_rounds_max - num_rounds)) as u64) * claim;
        scaled_claim * coeff
      }
    )
    .sum();
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut quad_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();

    for current_round in 0..num_rounds_max {
      let remaining_rounds = num_rounds_max - current_round;
      let evals: Vec<(E::Scalar, E::Scalar)> = zip_with!(
        par_iter,
        (num_rounds, claims, poly_A_vec, poly_B_vec),
        |num_rounds, claim, poly_A, poly_B| {
          if remaining_rounds <= *num_rounds {
            Self::compute_eval_points_quad_prod(poly_A, poly_B)
          } else {
            let remaining_variables = remaining_rounds - num_rounds - 1;
            let scaled_claim = E::Scalar::from((1 << remaining_variables) as u64) * claim;
            (scaled_claim, scaled_claim)
          }
        }
      )
      .collect();

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
      let evals_combined_bound_coeff = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();

      let evals = vec![
        evals_combined_0,
        e - evals_combined_0,
        evals_combined_bound_coeff,
      ];
      let poly = UniPoly::from_evals_deg2(&evals);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      // derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);

      // bound all tables to the verifier's challenge
      zip_with_for_each!(
        (
          num_rounds.par_iter(),
          poly_A_vec.par_iter_mut(),
          poly_B_vec.par_iter_mut()
        ),
        |num_rounds, poly_A, poly_B| {
          if remaining_rounds <= *num_rounds {
            let _ = rayon::join(
              || poly_A.bind_poly_var_top(&r_i),
              || poly_B.bind_poly_var_top(&r_i),
            );
          }
        }
      );

      e = poly.evaluate(&r_i);
      quad_polys.push(poly.compress());
    }
    poly_A_vec.iter().for_each(|p| assert_eq!(p.len(), 1));
    poly_B_vec.iter().for_each(|p| assert_eq!(p.len(), 1));

    let poly_A_final = poly_A_vec
      .into_iter()
      .map(|poly| poly[0])
      .collect::<Vec<_>>();
    let poly_B_final = poly_B_vec
      .into_iter()
      .map(|poly| poly[0])
      .collect::<Vec<_>>();

    let eval_expected = zip_with!(
      iter,
      (poly_A_final, poly_B_final, coeffs),
      |eA, eB, coeff| *eA * *eB * coeff
    )
    .sum::<E::Scalar>();
    assert_eq!(e, eval_expected);

    let claims_prod = (poly_A_final, poly_B_final);

    Ok((SumcheckProof::new(quad_polys), r, claims_prod))
  }

  // DEG1: poly_A - poly_B
  // DEG2: poly_A * poly_B
  // DEG3: poly_A * poly_B * poly_C
  #[inline]
  pub fn compute_eval_points_cubic_with_deg<const DEG: usize>(
    poly_A: &MultilinearPolynomial<E::Scalar>,
    poly_B: &MultilinearPolynomial<E::Scalar>,
    poly_C: &MultilinearPolynomial<E::Scalar>,
  ) -> (E::Scalar, E::Scalar, E::Scalar) {
    let len = poly_A.len() / 2;
    let (eval_0, bound_coeff, eval_inf) = (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = match DEG {
          1 => poly_A[i] - poly_B[i],
          2 => poly_A[i] * poly_B[i],
          3 => poly_A[i] * poly_B[i] * poly_C[i],
          _ => unreachable!(),
        };

        // eval cubic coefficient: (A(high)-A(low)) * (B(high)-B(low)) * (C(high)-C(low)) if DEG = 3
        let poly_A_bound_coeff = poly_A[len + i] - poly_A[i];
        let poly_B_bound_coeff = poly_B[len + i] - poly_B[i];
        let mut poly_C_bound_coeff = E::Scalar::ZERO;
        let bound_coeff = match DEG {
          1 | 2 => E::Scalar::ZERO,
          3 => {
            poly_C_bound_coeff = poly_C[len + i] - poly_C[i];
            poly_A_bound_coeff * poly_B_bound_coeff * poly_C_bound_coeff
          }
          _ => unreachable!(),
        };

        // eval -1: comb_func(A(-1), B(-1), C(-1))
        let poly_A_inf_point = poly_A[i] - poly_A_bound_coeff;
        let poly_B_inf_point = poly_B[i] - poly_B_bound_coeff;
        let eval_point_inf = match DEG {
          1 => poly_A_inf_point - poly_B_inf_point,
          2 => poly_A_inf_point * poly_B_inf_point,
          3 => {
            let poly_C_inf_point = poly_C[i] - poly_C_bound_coeff;
            poly_A_inf_point * poly_B_inf_point * poly_C_inf_point
          }
          _ => unreachable!(),
        };

        (eval_point_0, bound_coeff, eval_point_inf)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| {
          (
            a.0 + b.0,
            match DEG {
              1 | 2 => E::Scalar::ZERO,
              3 => a.1 + b.1,
              _ => unreachable!(),
            },
            a.2 + b.2,
          )
        },
      );

    (eval_0, bound_coeff, eval_inf)
  }

  /// Prove poly_A * poly_B - poly_C
  pub fn prove_cubic_with_three_inputs(
    claim: &E::Scalar,
    taus: Vec<E::Scalar>,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    poly_C: &mut MultilinearPolynomial<E::Scalar>,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), NovaError> {
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();
    let mut claim_per_round = *claim;

    let num_rounds = taus.len();

    let mut eq_instance = EqSumCheckInstance::<E>::new(taus);

    for _ in 0..num_rounds {
      let poly = {
        // Make an iterator returning the contributions to the evaluations
        let (eval_point_0, eval_point_bound_coeff, eval_point_inf) =
          eq_instance.evaluation_points_cubic_with_three_inputs(poly_A, poly_B, poly_C);

        let evals = vec![
          eval_point_0,
          claim_per_round - eval_point_0,
          eval_point_bound_coeff,
          eval_point_inf,
        ];

        UniPoly::from_evals_deg3(&evals)
      };

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);
      polys.push(poly.compress());

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);

      // bound all tables to the verifier's challenge
      rayon::join(
        || poly_A.bind_poly_var_top(&r_i),
        || poly_B.bind_poly_var_top(&r_i),
      );
      rayon::join(
        || poly_C.bind_poly_var_top(&r_i),
        || eq_instance.bound(&r_i),
      );
    }

    Ok((
      SumcheckProof {
        compressed_polys: polys,
      },
      r,
      vec![poly_A[0], poly_B[0], poly_C[0]],
    ))
  }
}

pub(crate) mod eq_sumcheck {
  //! This module implements the sumcheck optimization for equality polynomials.
  //! The optimization is described in Section 5 of https://eprint.iacr.org/2025/1117 algorithm 5.
  use crate::{spartan::polys::multilinear::MultilinearPolynomial, traits::Engine};
  use ff::{Field, PrimeField};
  use rayon::{iter::ZipEq, prelude::*, slice::Iter};

  pub struct EqSumCheckInstance<E: Engine> {
    // number of variables at first
    init_num_vars: usize,
    first_half: usize,
    second_half: usize,
    round: usize,
    taus: Vec<E::Scalar>,
    eval_eq_left: E::Scalar,
    poly_eq_left: Vec<Vec<E::Scalar>>,
    poly_eq_right: Vec<Vec<E::Scalar>>,
    eq_tau_0_a_inf: Vec<(E::Scalar, E::Scalar, E::Scalar)>,
  }

  impl<E: Engine> EqSumCheckInstance<E> {
    pub fn new(taus: Vec<E::Scalar>) -> Self {
      let l = taus.len();
      let first_half = l / 2;

      let compute_eq_polynomials = |taus: Vec<&E::Scalar>| -> Vec<Vec<E::Scalar>> {
        let len = taus.len();
        let mut result = Vec::with_capacity(len + 1);

        result.push(vec![E::Scalar::ONE]);

        for i in 0..len {
          let tau = taus[i];

          let prev = &result[i];
          let mut v_next = prev.to_vec();
          v_next.par_extend(prev.par_iter().map(|v| *v * tau));
          let (first, last) = v_next.split_at_mut(prev.len());
          first.par_iter_mut().zip(last).for_each(|(a, b)| *a -= *b);

          result.push(v_next);
        }

        result
      };

      let (left_taus, right_taus) = taus.split_at(first_half);
      let left_taus = left_taus.iter().skip(1).rev().collect::<Vec<_>>();
      let right_taus = right_taus.iter().rev().collect::<Vec<_>>();

      let (poly_eq_left, poly_eq_right) = rayon::join(
        || compute_eq_polynomials(left_taus),
        || compute_eq_polynomials(right_taus),
      );

      let eq_tau_0_a_inf = taus
        .par_iter()
        .map(|tau| {
          // eq(tau, 0), 2tau - 1, eq(tau, -1)
          let one_minus_tau = E::Scalar::ONE - tau;
          let two_tau_minus_one = *tau - one_minus_tau;
          let two_minus_three_tau = one_minus_tau - two_tau_minus_one;
          (one_minus_tau, two_tau_minus_one, two_minus_three_tau)
        })
        .collect::<Vec<_>>();

      Self {
        init_num_vars: l,
        first_half,
        second_half: l - first_half,
        round: 1,
        taus,
        eval_eq_left: E::Scalar::ONE,
        poly_eq_left,
        poly_eq_right,
        eq_tau_0_a_inf,
      }
    }

    /// Evaluate poly_A * poly_B - poly_C
    #[inline]
    pub fn evaluation_points_cubic_with_three_inputs(
      &self,
      poly_A: &MultilinearPolynomial<E::Scalar>,
      poly_B: &MultilinearPolynomial<E::Scalar>,
      poly_C: &MultilinearPolynomial<E::Scalar>,
    ) -> (E::Scalar, E::Scalar, E::Scalar) {
      debug_assert_eq!(poly_A.len() % 2, 0);

      let in_first_half = self.round < self.first_half;

      let half_p = poly_A.Z.len() / 2;

      let [zip_A, zip_B, zip_C] = split_and_zip([&poly_A.Z, &poly_B.Z, &poly_C.Z], half_p);

      let (mut eval_0, mut bound_coeff, mut eval_inf) = if in_first_half {
        let (poly_eq_left, poly_eq_right, second_half, low_mask) = self.poly_eqs_first_half();

        zip_A
          .zip_eq(zip_B)
          .zip_eq(zip_C)
          .enumerate()
          .map(|(id, ((a, b), c))| {
            let (zero_a, one_a) = a;
            let (zero_b, one_b) = b;
            let (zero_c, one_c) = c;

            let (eval_0, bound_coeff, eval_inf) =
              eval_one_case_cubic_three_inputs(zero_a, one_a, zero_b, one_b, zero_c, one_c);

            let factor = poly_eq_left[id >> second_half] * poly_eq_right[id & low_mask];

            (eval_0 * factor, bound_coeff * factor, eval_inf * factor)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      } else {
        let poly_eq_right = self.poly_eq_right_last_half().par_iter();

        zip_A
          .zip_eq(zip_B)
          .zip_eq(zip_C)
          .zip_eq(poly_eq_right)
          .map(|(((a, b), c), poly_eq_right)| {
            let (zero_a, one_a) = a;
            let (zero_b, one_b) = b;
            let (zero_c, one_c) = c;

            let (eval_0, bound_coeff, eval_inf) =
              eval_one_case_cubic_three_inputs(zero_a, one_a, zero_b, one_b, zero_c, one_c);

            let factor = poly_eq_right;

            (eval_0 * factor, bound_coeff * factor, eval_inf * factor)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      };

      self.update_evals(&mut eval_0, &mut bound_coeff, &mut eval_inf);

      (eval_0, bound_coeff, eval_inf)
    }

    /// Evaluate poly_A * poly_B - ONE
    #[inline]
    pub fn evaluation_points_cubic_with_two_inputs(
      &self,
      poly_A: &MultilinearPolynomial<E::Scalar>,
      poly_B: &MultilinearPolynomial<E::Scalar>,
    ) -> (E::Scalar, E::Scalar, E::Scalar) {
      debug_assert_eq!(poly_A.len() % 2, 0);

      let in_first_half = self.round < self.first_half;

      let half_p = poly_A.Z.len() / 2;

      let [zip_A, zip_B] = split_and_zip([&poly_A.Z, &poly_B.Z], half_p);

      let (mut eval_0, mut bound_coeff, mut eval_inf) = if in_first_half {
        let (poly_eq_left, poly_eq_right, second_half, low_mask) = self.poly_eqs_first_half();

        zip_A
          .zip_eq(zip_B)
          .enumerate()
          .map(|(id, (a, b))| {
            let (zero_a, one_a) = a;
            let (zero_b, one_b) = b;

            let (eval_0, bound_coeff, eval_inf) =
              eval_one_case_cubic_two_inputs(zero_a, one_a, zero_b, one_b);

            let factor = poly_eq_left[id >> second_half] * poly_eq_right[id & low_mask];

            (eval_0 * factor, bound_coeff * factor, eval_inf * factor)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      } else {
        let poly_eq_right = self.poly_eq_right_last_half().par_iter();

        zip_A
          .zip_eq(zip_B)
          .zip_eq(poly_eq_right)
          .map(|((a, b), poly_eq_right)| {
            let (zero_a, one_a) = a;
            let (zero_b, one_b) = b;

            let (eval_0, bound_coeff, eval_inf) =
              eval_one_case_cubic_two_inputs(zero_a, one_a, zero_b, one_b);

            let factor = poly_eq_right;

            (eval_0 * factor, bound_coeff * factor, eval_inf * factor)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      };

      self.update_evals(&mut eval_0, &mut bound_coeff, &mut eval_inf);

      (eval_0, bound_coeff, eval_inf)
    }

    /// Evaluate poly_A
    #[inline]
    pub fn evaluation_points_cubic_with_one_input(
      &self,
      poly_A: &MultilinearPolynomial<E::Scalar>,
    ) -> (E::Scalar, E::Scalar, E::Scalar) {
      debug_assert_eq!(poly_A.len() % 2, 0);

      let in_first_half = self.round < self.first_half;

      let half_p = poly_A.Z.len() / 2;

      let [zip_A] = split_and_zip([&poly_A.Z], half_p);

      let (mut eval_0, mut eval_inf) = if in_first_half {
        let (poly_eq_left, poly_eq_right, second_half, low_mask) = self.poly_eqs_first_half();

        zip_A
          .enumerate()
          .map(|(id, a)| {
            let (zero_a, one_a) = a;

            let (eval_0, eval_inf) = eval_one_case_cubic_one_input(zero_a, one_a);

            let factor = poly_eq_left[id >> second_half] * poly_eq_right[id & low_mask];

            (eval_0 * factor, eval_inf * factor)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1),
          )
      } else {
        let poly_eq_right = self.poly_eq_right_last_half().par_iter();

        zip_A
          .zip_eq(poly_eq_right)
          .map(|(a, poly_eq_right)| {
            let (zero_a, one_a) = a;

            let (eval_0, eval_inf) = eval_one_case_cubic_one_input(zero_a, one_a);

            let factor = poly_eq_right;

            (eval_0 * factor, eval_inf * factor)
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1),
          )
      };

      let mut bound_coeff = E::Scalar::ZERO;

      self.update_evals(&mut eval_0, &mut bound_coeff, &mut eval_inf);

      (eval_0, bound_coeff, eval_inf)
    }

    #[inline]
    pub fn bound(&mut self, r: &E::Scalar) {
      let tau = self.taus[self.round - 1];
      self.eval_eq_left *= E::Scalar::ONE - tau - r + (*r * tau).double();
      self.round += 1;
    }

    #[inline]
    fn update_evals(
      &self,
      eval_0: &mut E::Scalar,
      bound_coeff: &mut E::Scalar,
      eval_inf: &mut E::Scalar,
    ) {
      let p = self.eval_eq_left;
      let eq_tau_0_a_inf = self.eq_tau_0_a_inf[self.round - 1];
      *eval_0 *= eq_tau_0_a_inf.0 * p;
      *bound_coeff *= eq_tau_0_a_inf.1 * p;
      *eval_inf *= eq_tau_0_a_inf.2 * p;
    }

    #[inline]
    fn poly_eqs_first_half(&self) -> (&Vec<E::Scalar>, &Vec<E::Scalar>, usize, usize) {
      let second_half = self.second_half;
      let poly_eq_left = &self.poly_eq_left[self.first_half - self.round];
      let poly_eq_right = &self.poly_eq_right[second_half];

      debug_assert_eq!(poly_eq_right.len(), 1 << second_half);

      (
        poly_eq_left,
        poly_eq_right,
        second_half,
        (1 << second_half) - 1,
      )
    }

    #[inline]
    fn poly_eq_right_last_half(&self) -> &Vec<E::Scalar> {
      &self.poly_eq_right[self.init_num_vars - self.round]
    }
  }

  #[inline]
  fn split_and_zip<const N: usize, T: Sync>(
    vec: [&[T]; N],
    half_size: usize,
  ) -> [ZipEq<Iter<'_, T>, Iter<'_, T>>; N] {
    std::array::from_fn(|i| {
      let (left, right) = vec[i].split_at(half_size);
      left.par_iter().zip_eq(right.par_iter())
    })
  }

  #[inline]
  fn eval_one_case_cubic_one_input<Scalar: PrimeField>(
    zero_a: &Scalar,
    one_a: &Scalar,
  ) -> (Scalar, Scalar) {
    let eval_0 = zero_a;
    let m1 = -*one_a + zero_a.double();

    (*eval_0, m1)
  }

  // a * b -1
  #[inline]
  fn eval_one_case_cubic_two_inputs<Scalar: PrimeField>(
    zero_a: &Scalar,
    one_a: &Scalar,
    zero_b: &Scalar,
    one_b: &Scalar,
  ) -> (Scalar, Scalar, Scalar) {
    let one = Scalar::ONE;
    let eval_0 = *zero_a * *zero_b - one;

    let q = (*one_a - *zero_a) * (*one_b - *zero_b);
    let m1_a = -*one_a + zero_a.double();
    let m1_b = -*one_b + zero_b.double();
    let eval_inf = m1_a * m1_b - one;

    (eval_0, q, eval_inf)
  }

  // a * b - c
  #[inline]
  fn eval_one_case_cubic_three_inputs<Scalar: PrimeField>(
    zero_a: &Scalar,
    one_a: &Scalar,
    zero_b: &Scalar,
    one_b: &Scalar,
    zero_c: &Scalar,
    one_c: &Scalar,
  ) -> (Scalar, Scalar, Scalar) {
    let eval_0 = *zero_a * *zero_b - *zero_c;

    let q = (*one_a - *zero_a) * (*one_b - *zero_b);
    let m1_a = -*one_a + zero_a.double();
    let m1_b = -*one_b + zero_b.double();
    let m1_c = -*one_c + zero_c.double();
    let eval_inf = m1_a * m1_b - m1_c;

    (eval_0, q, eval_inf)
  }
}
