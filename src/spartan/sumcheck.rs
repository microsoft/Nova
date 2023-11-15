use crate::errors::NovaError;
use crate::spartan::polys::{
  multilinear::MultilinearPolynomial,
  univariate::{CompressedUniPoly, UniPoly},
};
use crate::traits::{Engine, TranscriptEngineTrait};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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

  #[inline]
  pub(in crate::spartan) fn compute_eval_points_quad<F>(
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

      // bound all tables to the verifier's challenege
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
    claim: &E::Scalar,
    num_rounds: usize,
    poly_A_vec: &mut Vec<MultilinearPolynomial<E::Scalar>>,
    poly_B_vec: &mut Vec<MultilinearPolynomial<E::Scalar>>,
    coeffs: &[E::Scalar],
    comb_func: F,
    transcript: &mut E::TE,
  ) -> Result<(Self, Vec<E::Scalar>, (Vec<E::Scalar>, Vec<E::Scalar>)), NovaError>
  where
    F: Fn(&E::Scalar, &E::Scalar) -> E::Scalar + Sync,
  {
    let mut e = *claim;
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut quad_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();

    for _ in 0..num_rounds {
      let evals: Vec<(E::Scalar, E::Scalar)> = poly_A_vec
        .par_iter()
        .zip(poly_B_vec.par_iter())
        .map(|(poly_A, poly_B)| Self::compute_eval_points_quad(poly_A, poly_B, &comb_func))
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
      poly_A_vec
        .par_iter_mut()
        .zip(poly_B_vec.par_iter_mut())
        .for_each(|(poly_A, poly_B)| {
          let _ = rayon::join(
            || poly_A.bind_poly_var_top(&r_i),
            || poly_B.bind_poly_var_top(&r_i),
          );
        });

      e = poly.evaluate(&r_i);
      quad_polys.push(poly.compress());
    }

    let poly_A_final = (0..poly_A_vec.len()).map(|i| poly_A_vec[i][0]).collect();
    let poly_B_final = (0..poly_B_vec.len()).map(|i| poly_B_vec[i][0]).collect();
    let claims_prod = (poly_A_final, poly_B_final);

    Ok((SumcheckProof::new(quad_polys), r, claims_prod))
  }

  #[inline]
  pub(in crate::spartan) fn compute_eval_points_cubic<F>(
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
  pub(in crate::spartan) fn compute_eval_points_cubic_with_additive_term<F>(
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
