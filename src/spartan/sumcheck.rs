use crate::{
  errors::NovaError,
  spartan::polys::{
    multilinear::MultilinearPolynomial,
    univariate::{CompressedUniPoly, UniPoly},
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

  #[inline]
  fn compute_eval_points_quad<F>(
    poly_A: &MultilinearPolynomial<E::Scalar>,
    poly_B: &MultilinearPolynomial<E::Scalar>,
    comb_func: &F,
  ) -> (E::Scalar, E::Scalar)
  where
    F: Fn(&E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let len = poly_A.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(&poly_A[i], &poly_B[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let eval_point_2 = comb_func(&poly_A_bound_point, &poly_B_bound_point);
        (eval_point_0, eval_point_2)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1),
      )
  }

  pub fn prove_quad<F>(
    claim: &E::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    comb_func: F,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), NovaError>
  where
    F: Fn(&E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();
    let mut claim_per_round = *claim;
    for _ in 0..num_rounds {
      let poly = {
        let (eval_point_0, eval_point_2) =
          Self::compute_eval_points_quad(poly_A, poly_B, &comb_func);

        let evals = vec![eval_point_0, claim_per_round - eval_point_0, eval_point_2];
        UniPoly::from_evals(&evals)
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

  pub fn prove_quad_batch<F>(
    claims: &[E::Scalar],
    num_rounds: &[usize],
    mut poly_A_vec: Vec<MultilinearPolynomial<E::Scalar>>,
    mut poly_B_vec: Vec<MultilinearPolynomial<E::Scalar>>,
    coeffs: &[E::Scalar],
    comb_func: F,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, (Vec<E::Scalar>, Vec<E::Scalar>)), NovaError>
  where
    F: Fn(&E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
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
            Self::compute_eval_points_quad(poly_A, poly_B, &comb_func)
          } else {
            let remaining_variables = remaining_rounds - num_rounds - 1;
            let scaled_claim = E::Scalar::from((1 << remaining_variables) as u64) * claim;
            (scaled_claim, scaled_claim)
          }
        }
      )
      .collect();

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i].0 * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i].1 * coeffs[i]).sum();

      let evals = vec![evals_combined_0, e - evals_combined_0, evals_combined_2];
      let poly = UniPoly::from_evals(&evals);

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
      |eA, eB, coeff| comb_func(eA, eB) * coeff
    )
    .sum::<E::Scalar>();
    assert_eq!(e, eval_expected);

    let claims_prod = (poly_A_final, poly_B_final);

    Ok((SumcheckProof::new(quad_polys), r, claims_prod))
  }

  #[inline]
  pub fn compute_eval_points_cubic<F>(
    poly_A: &MultilinearPolynomial<E::Scalar>,
    poly_B: &MultilinearPolynomial<E::Scalar>,
    poly_C: &MultilinearPolynomial<E::Scalar>,
    comb_func: &F,
  ) -> (E::Scalar, E::Scalar, E::Scalar)
  where
    F: Fn(&E::Scalar, &E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let len = poly_A.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
        let eval_point_2 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];
        let eval_point_3 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );
        (eval_point_0, eval_point_2, eval_point_3)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      )
  }

  #[inline]
  pub fn compute_eval_points_cubic_with_additive_term<F>(
    poly_A: &MultilinearPolynomial<E::Scalar>,
    poly_B: &MultilinearPolynomial<E::Scalar>,
    poly_C: &MultilinearPolynomial<E::Scalar>,
    poly_D: &MultilinearPolynomial<E::Scalar>,
    comb_func: &F,
  ) -> (E::Scalar, E::Scalar, E::Scalar)
  where
    F: Fn(&E::Scalar, &E::Scalar, &E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let len = poly_A.len() / 2;
    (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
        let poly_D_bound_point = poly_D[len + i] + poly_D[len + i] - poly_D[i];
        let eval_point_2 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
          &poly_D_bound_point,
        );

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];
        let poly_D_bound_point = poly_D_bound_point + poly_D[len + i] - poly_D[i];
        let eval_point_3 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
          &poly_D_bound_point,
        );
        (eval_point_0, eval_point_2, eval_point_3)
      })
      .reduce(
        || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      )
  }

  pub fn prove_cubic_with_additive_term<F>(
    claim: &E::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<E::Scalar>,
    poly_B: &mut MultilinearPolynomial<E::Scalar>,
    poly_C: &mut MultilinearPolynomial<E::Scalar>,
    poly_D: &mut MultilinearPolynomial<E::Scalar>,
    comb_func: F,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, Vec<E::Scalar>), NovaError>
  where
    F: Fn(&E::Scalar, &E::Scalar, &E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();
    let mut claim_per_round = *claim;

    for _ in 0..num_rounds {
      let poly = {
        // Make an iterator returning the contributions to the evaluations
        let (eval_point_0, eval_point_2, eval_point_3) =
          Self::compute_eval_points_cubic_with_additive_term(
            poly_A, poly_B, poly_C, poly_D, &comb_func,
          );

        let evals = vec![
          eval_point_0,
          claim_per_round - eval_point_0,
          eval_point_2,
          eval_point_3,
        ];
        UniPoly::from_evals(&evals)
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
        || {
          rayon::join(
            || poly_A.bind_poly_var_top(&r_i),
            || poly_B.bind_poly_var_top(&r_i),
          )
        },
        || {
          rayon::join(
            || poly_C.bind_poly_var_top(&r_i),
            || poly_D.bind_poly_var_top(&r_i),
          )
        },
      );
    }

    Ok((
      SumcheckProof {
        compressed_polys: polys,
      },
      r,
      vec![poly_A[0], poly_B[0], poly_C[0], poly_D[0]],
    ))
  }
}

pub(crate) mod eq_sumcheck {
  use crate::{spartan::polys::multilinear::MultilinearPolynomial, traits::Engine};
  use ff::{Field, PrimeField};
  use rayon::{iter::ZipEq, prelude::*, slice::Chunks};

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
    eq_tau_0_2_3: Vec<(E::Scalar, E::Scalar, E::Scalar)>,
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
          first
            .par_iter_mut()
            .zip(last)
            .for_each(|(a, b)| *a = *a - *b);

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

      let f2 = E::Scalar::ONE.double();
      let f1 = E::Scalar::ONE;
      let eq_tau_0_2_3 = taus
        .par_iter()
        .map(|tau| {
          let tau2 = tau.double();
          let tau3 = tau2 + tau;
          let tau5 = tau3 + tau2;
          (f1 - tau, tau3 - f1, tau5 - f2)
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
        eq_tau_0_2_3,
      }
    }

    #[inline]
    pub fn evaluation_points_cubic_with_three_inputs<F>(
      &self,
      poly_A: &MultilinearPolynomial<E::Scalar>,
      poly_B: &MultilinearPolynomial<E::Scalar>,
      poly_C: &MultilinearPolynomial<E::Scalar>,
      comb_func: F,
    ) -> (E::Scalar, E::Scalar, E::Scalar)
    where
      F: Fn([E::Scalar; 3]) -> E::Scalar + Sync,
    {
      debug_assert_eq!(poly_A.len() % 2, 0);

      let in_first_half = self.round < self.first_half;

      let half_p = poly_A.Z.len() / 2;

      let chunk_size = 100;

      let [zip_A, zip_B, zip_C] =
        split_and_zip_chunks([&poly_A.Z, &poly_B.Z, &poly_C.Z], half_p, chunk_size);

      let (mut eval_0, mut eval_2, mut eval_3) = if in_first_half {
        let (poly_eq_left, poly_eq_right, second_half, low_mask) = self.poly_eqs_first_half();

        zip_A
          .zip_eq(zip_B)
          .zip_eq(zip_C)
          .enumerate()
          .map(|(chunk_id, ((a, b), c))| {
            let (zero_a, one_a) = a;
            let (zero_b, one_b) = b;
            let (zero_c, one_c) = c;

            let offset = chunk_id * chunk_size;

            process_chunk_cubic_first_half(
              [zero_a, zero_b, zero_c],
              [one_a, one_b, one_c],
              poly_eq_left,
              poly_eq_right,
              offset,
              second_half,
              low_mask,
              &comb_func,
            )
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      } else {
        let poly_eq_right = self.poly_eq_right_last_half().par_chunks(chunk_size);

        zip_A
          .zip_eq(zip_B)
          .zip_eq(zip_C)
          .zip_eq(poly_eq_right)
          .map(|(((a, b), c), poly_eq_right)| {
            let (zero_a, one_a) = a;
            let (zero_b, one_b) = b;
            let (zero_c, one_c) = c;

            process_chunk_cubic_last_half(
              [zero_a, zero_b, zero_c],
              [one_a, one_b, one_c],
              poly_eq_right,
              &comb_func,
            )
          })
          .reduce(
            || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          )
      };

      self.update_evals(&mut eval_0, &mut eval_2, &mut eval_3);

      (eval_0, eval_2, eval_3)
    }

    #[inline]
    pub fn evaluation_points_cubic_with_two_inputs<F>(
      &self,
      poly_A: &MultilinearPolynomial<E::Scalar>,
      poly_B: &MultilinearPolynomial<E::Scalar>,
      comb_func: F,
    ) -> (E::Scalar, E::Scalar, E::Scalar)
    where
      F: Fn([E::Scalar; 2]) -> E::Scalar + Sync,
    {
      assert_eq!(poly_A.len() % 2, 0);

      let in_first_half = self.round < self.first_half;

      let (poly_eq_left, poly_eq_right, right_len, second_half) = if in_first_half {
        let second_half = self.init_num_vars - self.first_half;
        let poly_eq_left = &self.poly_eq_left[self.first_half - self.round];
        let poly_eq_right = &self.poly_eq_right[second_half];
        let right_len = poly_eq_right.len();
        assert_eq!(right_len, 1 << second_half);
        (poly_eq_left, poly_eq_right, right_len, second_half)
      } else {
        let poly_eq_right = &self.poly_eq_right[self.init_num_vars - self.round];
        (poly_eq_right, poly_eq_right, 0, 0)
      };

      let half_p = poly_A.Z.len() / 2;

      let (zero_A, one_A) = poly_A.Z.split_at(half_p);
      let (zero_B, one_B) = poly_B.Z.split_at(half_p);

      let zip_A = zero_A.par_iter().zip(one_A);
      let zip_B = zero_B.par_iter().zip(one_B);

      let (partial_eval_0, partial_eval_2, partial_eval_3) = zip_A
        .zip(zip_B)
        .enumerate()
        .map(|(i, ((zero_a, one_a), (zero_b, one_b)))| {
          let one_a_double = one_a.double();
          let one_b_double = one_b.double();
          let one_a_triple = one_a_double + one_a;
          let one_b_triple = one_b_double + one_b;
          let zero_a_double = zero_a.double();
          let zero_b_double = zero_b.double();

          let e_a_0 = *zero_a;
          let e_b_0 = *zero_b;

          let e_a_2 = one_a_double - zero_a;
          let e_b_2 = one_b_double - zero_b;

          let e_a_3 = one_a_triple - zero_a_double;
          let e_b_3 = one_b_triple - zero_b_double;

          let mut eval_0 = comb_func([e_a_0, e_b_0]);
          let mut eval_2 = comb_func([e_a_2, e_b_2]);
          let mut eval_3 = comb_func([e_a_3, e_b_3]);

          let factor = if in_first_half {
            let left_id = i >> second_half;
            let right_id = i % right_len;
            assert_eq!(i, left_id * (1 << second_half) + right_id);
            let left = poly_eq_left[left_id];
            let right = poly_eq_right[right_id];
            left * right
          } else {
            poly_eq_right[i]
          };

          eval_0 *= factor;
          eval_2 *= factor;
          eval_3 *= factor;

          (eval_0, eval_2, eval_3)
        })
        .reduce(
          || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
          |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

      let p = self.eval_eq_left;
      let eq_tau_0_2_3 = self.eq_tau_0_2_3[self.round - 1];
      let eq_tau_0_p = eq_tau_0_2_3.0 * p;
      let eq_tau_2_p = eq_tau_0_2_3.1 * p;
      let eq_tau_3_p = eq_tau_0_2_3.2 * p;

      (
        partial_eval_0 * eq_tau_0_p,
        partial_eval_2 * eq_tau_2_p,
        partial_eval_3 * eq_tau_3_p,
      )
    }

    #[inline]
    pub fn evaluation_points_cubic_with_one_input(
      &self,
      poly_A: &MultilinearPolynomial<E::Scalar>,
    ) -> (E::Scalar, E::Scalar, E::Scalar) {
      assert_eq!(poly_A.len() % 2, 0);

      let in_first_half = self.round < self.first_half;

      let (poly_eq_left, poly_eq_right, right_len, second_half) = if in_first_half {
        let second_half = self.init_num_vars - self.first_half;
        let poly_eq_left = &self.poly_eq_left[self.first_half - self.round];
        let poly_eq_right = &self.poly_eq_right[second_half];
        let right_len = poly_eq_right.len();
        assert_eq!(right_len, 1 << second_half);
        (poly_eq_left, poly_eq_right, right_len, second_half)
      } else {
        let poly_eq_right = &self.poly_eq_right[self.init_num_vars - self.round];
        (poly_eq_right, poly_eq_right, 0, 0)
      };

      let half_p = poly_A.Z.len() / 2;

      let (zero_A, one_A) = poly_A.Z.split_at(half_p);

      let zip_A = zero_A.par_iter().zip(one_A);

      let (partial_eval_0, partial_eval_2, partial_eval_3) = zip_A
        .enumerate()
        .map(|(i, (zero_a, one_a))| {
          let one_a_double = one_a.double();
          let one_a_triple = one_a_double + one_a;
          let zero_a_double = zero_a.double();

          let mut eval_0 = *zero_a;
          let mut eval_2 = one_a_double - zero_a;
          let mut eval_3 = one_a_triple - zero_a_double;

          let factor = if in_first_half {
            let left_id = i >> second_half;
            let right_id = i % right_len;
            let left = poly_eq_left[left_id];
            let right = poly_eq_right[right_id];
            assert_eq!(i, left_id * (1 << second_half) + right_id);
            left * right
          } else {
            poly_eq_right[i]
          };

          eval_0 *= factor;
          eval_2 *= factor;
          eval_3 *= factor;

          (eval_0, eval_2, eval_3)
        })
        .reduce(
          || (E::Scalar::ZERO, E::Scalar::ZERO, E::Scalar::ZERO),
          |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
        );

      let p = self.eval_eq_left;
      let eq_tau_0_2_3 = self.eq_tau_0_2_3[self.round - 1];
      let eq_tau_0_p = eq_tau_0_2_3.0 * p;
      let eq_tau_2_p = eq_tau_0_2_3.1 * p;
      let eq_tau_3_p = eq_tau_0_2_3.2 * p;

      (
        partial_eval_0 * eq_tau_0_p,
        partial_eval_2 * eq_tau_2_p,
        partial_eval_3 * eq_tau_3_p,
      )
    }

    #[inline]
    pub fn bound(&mut self, r: &E::Scalar) {
      let tau = self.taus[self.round - 1];
      self.eval_eq_left *= E::Scalar::ONE - tau - r + (*r * tau).double();
      self.round += 1;
    }

    #[inline]
    fn update_evals(&self, eval_0: &mut E::Scalar, eval_2: &mut E::Scalar, eval_3: &mut E::Scalar) {
      let p = self.eval_eq_left;
      let eq_tau_0_2_3 = self.eq_tau_0_2_3[self.round - 1];
      let eq_tau_0_p = eq_tau_0_2_3.0 * p;
      let eq_tau_2_p = eq_tau_0_2_3.1 * p;
      let eq_tau_3_p = eq_tau_0_2_3.2 * p;

      *eval_0 *= eq_tau_0_p;
      *eval_2 *= eq_tau_2_p;
      *eval_3 *= eq_tau_3_p;
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
  fn split_and_zip_chunks<const N: usize, T: Sync>(
    vec: [&[T]; N],
    half_size: usize,
    chunk_size: usize,
  ) -> [ZipEq<Chunks<'_, T>, Chunks<'_, T>>; N] {
    std::array::from_fn(|i| {
      let (left, right) = vec[i].split_at(half_size);
      left
        .par_chunks(chunk_size)
        .zip_eq(right.par_chunks(chunk_size))
    })
  }

  #[inline]
  fn process_chunk_cubic_first_half<const N: usize, Scalar: PrimeField, F>(
    zero_chunks: [&[Scalar]; N],
    one_chunks: [&[Scalar]; N],
    poly_eq_left: &[Scalar],
    poly_eq_right: &[Scalar],
    offset: usize,
    second_half: usize,
    low_mask: usize,
    comb_func: &F,
  ) -> (Scalar, Scalar, Scalar)
  where
    F: Fn([Scalar; N]) -> Scalar,
  {
    let mut acc_eval_0 = Scalar::ZERO;
    let mut acc_eval_2 = Scalar::ZERO;
    let mut acc_eval_3 = Scalar::ZERO;

    let n = zero_chunks[0].len();

    for i in 0..n {
      let zeros = std::array::from_fn(|j| zero_chunks[j][i]);
      let ones = std::array::from_fn(|j| one_chunks[j][i]);

      let (eval_0, eval_2, eval_3) = eval_one_case_cubic(zeros, ones, comb_func);

      let id = offset + i;

      let factor = poly_eq_left[id >> second_half] * poly_eq_right[id & low_mask];

      acc_eval_0 += eval_0 * factor;
      acc_eval_2 += eval_2 * factor;
      acc_eval_3 += eval_3 * factor;
    }

    (acc_eval_0, acc_eval_2, acc_eval_3)
  }

  #[inline]
  fn process_chunk_cubic_last_half<const N: usize, Scalar: PrimeField, F>(
    zero_chunks: [&[Scalar]; N],
    one_chunks: [&[Scalar]; N],
    poly_eq_right_chunk: &[Scalar],
    comb_func: &F,
  ) -> (Scalar, Scalar, Scalar)
  where
    F: Fn([Scalar; N]) -> Scalar,
  {
    let mut acc_eval_0 = Scalar::ZERO;
    let mut acc_eval_2 = Scalar::ZERO;
    let mut acc_eval_3 = Scalar::ZERO;

    let n = zero_chunks[0].len();

    for i in 0..n {
      let zeros = std::array::from_fn(|j| zero_chunks[j][i]);
      let ones: [_; N] = std::array::from_fn(|j| one_chunks[j][i]);

      let (eval_0, eval_2, eval_3) = eval_one_case_cubic(zeros, ones, comb_func);

      let factor = poly_eq_right_chunk[i];

      acc_eval_0 += eval_0 * factor;
      acc_eval_2 += eval_2 * factor;
      acc_eval_3 += eval_3 * factor;
    }

    (acc_eval_0, acc_eval_2, acc_eval_3)
  }

  #[inline]
  fn eval_one_case_cubic<const N: usize, Scalar: PrimeField, F>(
    zeros: [Scalar; N],
    ones: [Scalar; N],
    comb_func: F,
  ) -> (Scalar, Scalar, Scalar)
  where
    F: Fn([Scalar; N]) -> Scalar,
  {
    let eval_0 = {
      let poly_bound_point = zeros;
      comb_func(poly_bound_point)
    };

    let one_double = ones.iter().map(|one| one.double()).collect::<Vec<_>>();

    let eval_2 = {
      let poly_bound_point = std::array::from_fn(|i| one_double[i] - zeros[i]);
      comb_func(poly_bound_point)
    };

    let eval_3 = {
      let poly_bound_point = std::array::from_fn(|i| one_double[i] + ones[i] - zeros[i].double());
      comb_func(poly_bound_point)
    };

    (eval_0, eval_2, eval_3)
  }
}
