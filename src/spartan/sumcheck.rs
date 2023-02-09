#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
use super::polynomial::MultilinearPolynomial;
use crate::errors::NovaError;
use crate::traits::{AppendToTranscriptTrait, ChallengeTrait, Group};
use core::marker::PhantomData;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct SumcheckProof<G: Group> {
  compressed_polys: Vec<CompressedUniPoly<G>>,
}

impl<G: Group> SumcheckProof<G> {
  pub fn verify(
    &self,
    claim: G::Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut G::TE,
  ) -> Result<(G::Scalar, Vec<G::Scalar>), NovaError> {
    let mut e = claim;
    let mut r: Vec<G::Scalar> = Vec::new();

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

      // check if G_k(0) + G_k(1) = e
      if poly.eval_at_zero() + poly.eval_at_one() != e {
        return Err(NovaError::InvalidSumcheckProof);
      }

      // append the prover's message to the transcript
      poly.append_to_transcript(b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_i = G::Scalar::challenge(b"challenge_nextround", transcript)?;

      r.push(r_i);

      // evaluate the claimed degree-ell polynomial at r_i
      e = poly.evaluate(&r_i);
    }

    Ok((e, r))
  }

  pub fn prove_quad<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<G::Scalar>,
    poly_B: &mut MultilinearPolynomial<G::Scalar>,
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, Vec<G::Scalar>), NovaError>
  where
    F: Fn(&G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<G>> = Vec::new();
    let mut claim_per_round = *claim;
    for _ in 0..num_rounds {
      let poly = {
        let len = poly_A.len() / 2;

        // Make an iterator returning the contributions to the evaluations
        let (eval_point_0, eval_point_2) = (0..len)
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
            || (G::Scalar::zero(), G::Scalar::zero()),
            |a, b| (a.0 + b.0, a.1 + b.1),
          );

        let evals = vec![eval_point_0, claim_per_round - eval_point_0, eval_point_2];
        UniPoly::from_evals(&evals)
      };

      // append the prover's message to the transcript
      poly.append_to_transcript(b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_i = G::Scalar::challenge(b"challenge_nextround", transcript)?;
      r.push(r_i);
      polys.push(poly.compress());

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);

      // bound all tables to the verifier's challenege
      poly_A.bound_poly_var_top(&r_i);
      poly_B.bound_poly_var_top(&r_i);
    }

    Ok((
      SumcheckProof {
        compressed_polys: polys,
      },
      r,
      vec![poly_A[0], poly_B[0]],
    ))
  }

  pub fn prove_cubic_with_additive_term<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<G::Scalar>,
    poly_B: &mut MultilinearPolynomial<G::Scalar>,
    poly_C: &mut MultilinearPolynomial<G::Scalar>,
    poly_D: &mut MultilinearPolynomial<G::Scalar>,
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, Vec<G::Scalar>), NovaError>
  where
    F: Fn(&G::Scalar, &G::Scalar, &G::Scalar, &G::Scalar) -> G::Scalar + Sync,
  {
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut polys: Vec<CompressedUniPoly<G>> = Vec::new();
    let mut claim_per_round = *claim;

    for _ in 0..num_rounds {
      let poly = {
        let len = poly_A.len() / 2;

        // Make an iterator returning the contributions to the evaluations
        let (eval_point_0, eval_point_2, eval_point_3) = (0..len)
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
            || (G::Scalar::zero(), G::Scalar::zero(), G::Scalar::zero()),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
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
      poly.append_to_transcript(b"poly", transcript);

      //derive the verifier's challenge for the next round
      let r_i = G::Scalar::challenge(b"challenge_nextround", transcript)?;
      r.push(r_i);
      polys.push(poly.compress());

      // Set up next round
      claim_per_round = poly.evaluate(&r_i);

      // bound all tables to the verifier's challenege
      poly_A.bound_poly_var_top(&r_i);
      poly_B.bound_poly_var_top(&r_i);
      poly_C.bound_poly_var_top(&r_i);
      poly_D.bound_poly_var_top(&r_i);
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

// ax^2 + bx + c stored as vec![a,b,c]
// ax^3 + bx^2 + cx + d stored as vec![a,b,c,d]
#[derive(Debug)]
pub struct UniPoly<G: Group> {
  coeffs: Vec<G::Scalar>,
}

// ax^2 + bx + c stored as vec![a,c]
// ax^3 + bx^2 + cx + d stored as vec![a,c,d]
#[derive(Debug, Serialize, Deserialize)]
pub struct CompressedUniPoly<G: Group> {
  coeffs_except_linear_term: Vec<G::Scalar>,
  _p: PhantomData<G>,
}

impl<G: Group> UniPoly<G> {
  pub fn from_evals(evals: &[G::Scalar]) -> Self {
    // we only support degree-2 or degree-3 univariate polynomials
    assert!(evals.len() == 3 || evals.len() == 4);
    let coeffs = if evals.len() == 3 {
      // ax^2 + bx + c
      let two_inv = G::Scalar::from(2).invert().unwrap();

      let c = evals[0];
      let a = two_inv * (evals[2] - evals[1] - evals[1] + c);
      let b = evals[1] - c - a;
      vec![c, b, a]
    } else {
      // ax^3 + bx^2 + cx + d
      let two_inv = G::Scalar::from(2).invert().unwrap();
      let six_inv = G::Scalar::from(6).invert().unwrap();

      let d = evals[0];
      let a = six_inv
        * (evals[3] - evals[2] - evals[2] - evals[2] + evals[1] + evals[1] + evals[1] - evals[0]);
      let b = two_inv
        * (evals[0] + evals[0] - evals[1] - evals[1] - evals[1] - evals[1] - evals[1]
          + evals[2]
          + evals[2]
          + evals[2]
          + evals[2]
          - evals[3]);
      let c = evals[1] - d - a - b;
      vec![d, c, b, a]
    };

    UniPoly { coeffs }
  }

  pub fn degree(&self) -> usize {
    self.coeffs.len() - 1
  }

  pub fn eval_at_zero(&self) -> G::Scalar {
    self.coeffs[0]
  }

  pub fn eval_at_one(&self) -> G::Scalar {
    (0..self.coeffs.len())
      .into_par_iter()
      .map(|i| self.coeffs[i])
      .reduce(G::Scalar::zero, |a, b| a + b)
  }

  pub fn evaluate(&self, r: &G::Scalar) -> G::Scalar {
    let mut eval = self.coeffs[0];
    let mut power = *r;
    for coeff in self.coeffs.iter().skip(1) {
      eval += power * coeff;
      power *= r;
    }
    eval
  }

  pub fn compress(&self) -> CompressedUniPoly<G> {
    let coeffs_except_linear_term = [&self.coeffs[0..1], &self.coeffs[2..]].concat();
    assert_eq!(coeffs_except_linear_term.len() + 1, self.coeffs.len());
    CompressedUniPoly {
      coeffs_except_linear_term,
      _p: Default::default(),
    }
  }
}

impl<G: Group> CompressedUniPoly<G> {
  // we require eval(0) + eval(1) = hint, so we can solve for the linear term as:
  // linear_term = hint - 2 * constant_term - deg2 term - deg3 term
  pub fn decompress(&self, hint: &G::Scalar) -> UniPoly<G> {
    let mut linear_term =
      *hint - self.coeffs_except_linear_term[0] - self.coeffs_except_linear_term[0];
    for i in 1..self.coeffs_except_linear_term.len() {
      linear_term -= self.coeffs_except_linear_term[i];
    }

    let mut coeffs: Vec<G::Scalar> = Vec::new();
    coeffs.extend(vec![&self.coeffs_except_linear_term[0]]);
    coeffs.extend(vec![&linear_term]);
    coeffs.extend(self.coeffs_except_linear_term[1..].to_vec());
    assert_eq!(self.coeffs_except_linear_term.len() + 1, coeffs.len());
    UniPoly { coeffs }
  }
}

impl<G: Group> AppendToTranscriptTrait<G> for UniPoly<G> {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut G::TE) {
    <[G::Scalar] as AppendToTranscriptTrait<G>>::append_to_transcript(
      &self.coeffs,
      label,
      transcript,
    );
  }
}
