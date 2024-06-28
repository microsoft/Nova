//! This module implements `BatchedRelaxedR1CSSNARKTrait` using Spartan that is generic over the polynomial commitment
//! and evaluation argument (i.e., a PCS) This version of Spartan does not use preprocessing so the verifier keeps the
//! entire description of R1CS matrices. This is essentially optimal for the verifier when using an IPA-based polynomial
//! commitment scheme. This batched implementation batches the outer and inner sumchecks of the Spartan SNARK.

use ff::Field;
use serde::{Deserialize, Serialize};

use itertools::Itertools;
use once_cell::sync::OnceCell;
use rayon::prelude::*;

use super::{
  compute_eval_table_sparse,
  math::Math,
  polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial},
  powers,
  snark::batch_eval_reduce,
  sumcheck::SumcheckProof,
  PolyEvalInstance, PolyEvalWitness,
};

use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix},
  spartan::{
    polys::{multilinear::SparsePolynomial, power::PowPolynomial},
    snark::batch_eval_verify,
  },
  traits::{
    evaluation::EvaluationEngineTrait,
    snark::{BatchedRelaxedR1CSSNARKTrait, DigestHelperTrait},
    Engine, TranscriptEngineTrait,
  },
  zip_with, CommitmentKey,
};

/// A succinct proof of knowledge of a witness to a batch of relaxed R1CS instances
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct BatchedRelaxedR1CSSNARK<E: Engine, EE: EvaluationEngineTrait<E>> {
  sc_proof_outer: SumcheckProof<E>,
  // Claims ([Azᵢ(τᵢ)], [Bzᵢ(τᵢ)], [Czᵢ(τᵢ)])
  claims_outer: (Vec<E::Scalar>, Vec<E::Scalar>, Vec<E::Scalar>),
  // [Eᵢ(r_x)]
  evals_E: Vec<E::Scalar>,
  sc_proof_inner: SumcheckProof<E>,
  // [Wᵢ(r_y[1..])]
  evals_W: Vec<E::Scalar>,
  sc_proof_batch: SumcheckProof<E>,
  // [Wᵢ(r_z), Eᵢ(r_z)]
  evals_batch: Vec<E::Scalar>,
  eval_arg: EE::EvaluationArgument,
}

/// A type that represents the prover's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  vk_digest: E::Scalar, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  vk_ee: EE::VerifierKey,
  S: Vec<R1CSShape<E>>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE> {
  fn new(shapes: Vec<R1CSShape<E>>, vk_ee: EE::VerifierKey) -> Self {
    VerifierKey {
      vk_ee,
      S: shapes,
      digest: OnceCell::new(),
    }
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> {}

impl<E: Engine, EE: EvaluationEngineTrait<E>> DigestHelperTrait<E> for VerifierKey<E, EE> {
  /// Returns the digest of the verifier's key.
  fn digest(&self) -> E::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::<E::Scalar, _>::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> BatchedRelaxedR1CSSNARKTrait<E>
  for BatchedRelaxedR1CSSNARK<E, EE>
{
  type ProverKey = ProverKey<E, EE>;

  type VerifierKey = VerifierKey<E, EE>;

  fn setup(
    ck: &CommitmentKey<E>,
    S: Vec<&R1CSShape<E>>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck);

    let S = S.iter().map(|s| s.pad()).collect();

    let vk = VerifierKey::new(S, vk_ee);

    let pk = ProverKey {
      pk_ee,
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: Vec<&R1CSShape<E>>,
    U: &[RelaxedR1CSInstance<E>],
    W: &[RelaxedR1CSWitness<E>],
  ) -> Result<Self, NovaError> {
    let num_instances = U.len();
    // Pad shapes and ensure their sizes are correct
    let S = S
      .iter()
      .map(|s| {
        let s = s.pad();
        if s.is_regular_shape() {
          Ok(s)
        } else {
          Err(NovaError::InternalError)
        }
      })
      .collect::<Result<Vec<_>, _>>()?;

    // Pad (W,E) for each instance
    let W = zip_with!(iter, (W, S), |w, s| w.pad(s)).collect::<Vec<RelaxedR1CSWitness<E>>>();

    let mut transcript = E::TE::new(b"BatchedRelaxedR1CSSNARK");

    transcript.absorb(b"vk", &pk.vk_digest);
    if num_instances > 1 {
      let num_instances_field = E::Scalar::from(num_instances as u64);
      transcript.absorb(b"n", &num_instances_field);
    }
    U.iter().for_each(|u| {
      transcript.absorb(b"U", u);
    });

    let (polys_W, polys_E): (Vec<_>, Vec<_>) = W.into_iter().map(|w| (w.W, w.E)).unzip();

    // Append public inputs to W: Z = [W, u, X]
    let polys_Z = zip_with!(iter, (polys_W, U), |w, u| [
      w.clone(),
      vec![u.u],
      u.X.clone()
    ]
    .concat())
    .collect::<Vec<Vec<_>>>();

    let (num_rounds_x, num_rounds_y): (Vec<_>, Vec<_>) = S
      .iter()
      .map(|s| (s.num_cons.log_2(), s.num_vars.log_2() + 1))
      .unzip();
    let num_rounds_x_max = *num_rounds_x.iter().max().unwrap();
    let num_rounds_y_max = *num_rounds_y.iter().max().unwrap();

    // Generate tau polynomial corresponding to eq(τ, τ², τ⁴ , …)
    // for a random challenge τ
    let tau = transcript.squeeze(b"t")?;
    let all_taus = PowPolynomial::squares(&tau, num_rounds_x_max);

    let polys_tau = num_rounds_x
      .iter()
      .map(|&num_rounds_x| PowPolynomial::evals_with_powers(&all_taus, num_rounds_x))
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();

    // Compute MLEs of Az, Bz, Cz, uCz + E
    let (polys_Az, polys_Bz, polys_Cz): (Vec<_>, Vec<_>, Vec<_>) =
      zip_with!(par_iter, (S, polys_Z), |s, poly_Z| {
        let (poly_Az, poly_Bz, poly_Cz) = s.multiply_vec(poly_Z)?;
        Ok((poly_Az, poly_Bz, poly_Cz))
      })
      .collect::<Result<Vec<_>, NovaError>>()?
      .into_iter()
      .multiunzip();

    let polys_uCz_E = zip_with!(par_iter, (U, polys_E, polys_Cz), |u, poly_E, poly_Cz| {
      zip_with!(par_iter, (poly_Cz, poly_E), |cz, e| u.u * cz + e).collect::<Vec<E::Scalar>>()
    })
    .collect::<Vec<_>>();

    let comb_func_outer =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    // Sample challenge for random linear-combination of outer claims
    let outer_r = transcript.squeeze(b"out_r")?;
    let outer_r_powers = powers::<E>(&outer_r, num_instances);

    // Verify outer sumcheck: Az * Bz - uCz_E for each instance
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_additive_term_batch(
      &vec![E::Scalar::ZERO; num_instances],
      &num_rounds_x,
      polys_tau,
      polys_Az
        .into_iter()
        .map(MultilinearPolynomial::new)
        .collect(),
      polys_Bz
        .into_iter()
        .map(MultilinearPolynomial::new)
        .collect(),
      polys_uCz_E
        .into_iter()
        .map(MultilinearPolynomial::new)
        .collect(),
      &outer_r_powers,
      comb_func_outer,
      &mut transcript,
    )?;

    let r_x = num_rounds_x
      .iter()
      .map(|&num_rounds| r_x[(num_rounds_x_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    // Extract evaluations of Az, Bz from Sumcheck and Cz, E at r_x
    let (evals_Az_Bz_Cz, evals_E): (Vec<_>, Vec<_>) = zip_with!(
      par_iter,
      (claims_outer[1], claims_outer[2], polys_Cz, polys_E, r_x),
      |eval_Az, eval_Bz, poly_Cz, poly_E, r_x| {
        let (eval_Cz, eval_E) = rayon::join(
          || MultilinearPolynomial::evaluate_with(poly_Cz, r_x),
          || MultilinearPolynomial::evaluate_with(poly_E, r_x),
        );
        ((*eval_Az, *eval_Bz, eval_Cz), eval_E)
      }
    )
    .unzip();

    evals_Az_Bz_Cz.iter().zip_eq(evals_E.iter()).for_each(
      |(&(eval_Az, eval_Bz, eval_Cz), &eval_E)| {
        transcript.absorb(
          b"claims_outer",
          &[eval_Az, eval_Bz, eval_Cz, eval_E].as_slice(),
        )
      },
    );

    let inner_r = transcript.squeeze(b"in_r")?;
    let inner_r_square = inner_r.square();
    let inner_r_cube = inner_r_square * inner_r;
    let inner_r_powers = powers::<E>(&inner_r_cube, num_instances);

    let claims_inner_joint = evals_Az_Bz_Cz
      .iter()
      .map(|(eval_Az, eval_Bz, eval_Cz)| *eval_Az + inner_r * eval_Bz + inner_r_square * eval_Cz)
      .collect::<Vec<_>>();

    let polys_ABCs = {
      let inner = |M_evals_As: Vec<E::Scalar>,
                   M_evals_Bs: Vec<E::Scalar>,
                   M_evals_Cs: Vec<E::Scalar>|
       -> Vec<E::Scalar> {
        zip_with!(
          into_par_iter,
          (M_evals_As, M_evals_Bs, M_evals_Cs),
          |eval_A, eval_B, eval_C| eval_A + inner_r * eval_B + inner_r_square * eval_C
        )
        .collect::<Vec<_>>()
      };

      zip_with!(par_iter, (S, r_x), |s, r_x| {
        let evals_rx = EqPolynomial::evals_from_points(r_x);
        let (eval_A, eval_B, eval_C) = compute_eval_table_sparse(s, &evals_rx);
        MultilinearPolynomial::new(inner(eval_A, eval_B, eval_C))
      })
      .collect::<Vec<_>>()
    };

    let polys_Z = polys_Z
      .into_iter()
      .zip_eq(num_rounds_y.iter())
      .map(|(mut z, &num_rounds_y)| {
        z.resize(1 << num_rounds_y, E::Scalar::ZERO);
        MultilinearPolynomial::new(z)
      })
      .collect::<Vec<_>>();

    let comb_func = |poly_A_comp: &E::Scalar, poly_B_comp: &E::Scalar| -> E::Scalar {
      *poly_A_comp * *poly_B_comp
    };

    let (sc_proof_inner, r_y, _claims_inner): (SumcheckProof<E>, Vec<E::Scalar>, (Vec<_>, Vec<_>)) =
      SumcheckProof::prove_quad_batch(
        &claims_inner_joint,
        &num_rounds_y,
        polys_ABCs,
        polys_Z,
        &inner_r_powers,
        comb_func,
        &mut transcript,
      )?;

    let r_y = num_rounds_y
      .iter()
      .map(|num_rounds| {
        let (_, r_y_hi) = r_y.split_at(num_rounds_y_max - num_rounds);
        r_y_hi
      })
      .collect::<Vec<_>>();

    let evals_W = zip_with!(par_iter, (polys_W, r_y), |poly, r_y| {
      MultilinearPolynomial::evaluate_with(poly, &r_y[1..])
    })
    .collect::<Vec<_>>();

    // Create evaluation instances for W(r_y[1..]) and E(r_x)
    let (w_vec, u_vec) = {
      let mut w_vec = Vec::with_capacity(2 * num_instances);
      let mut u_vec = Vec::with_capacity(2 * num_instances);
      w_vec.extend(polys_W.into_iter().map(|poly| PolyEvalWitness { p: poly }));
      u_vec.extend(zip_with!(iter, (evals_W, U, r_y), |eval, u, r_y| {
        PolyEvalInstance {
          c: u.comm_W,
          x: r_y[1..].to_vec(),
          e: *eval,
        }
      }));

      w_vec.extend(polys_E.into_iter().map(|poly| PolyEvalWitness { p: poly }));
      u_vec.extend(zip_with!(
        (evals_E.iter(), U.iter(), r_x),
        |eval_E, u, r_x| PolyEvalInstance {
          c: u.comm_E,
          x: r_x,
          e: *eval_E,
        }
      ));
      (w_vec, u_vec)
    };

    let (batched_u, batched_w, sc_proof_batch, claims_batch_left) =
      batch_eval_reduce(u_vec, w_vec, &mut transcript)?;

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_w.p,
      &batched_u.x,
      &batched_u.e,
    )?;

    let (evals_Az, evals_Bz, evals_Cz): (Vec<_>, Vec<_>, Vec<_>) =
      evals_Az_Bz_Cz.into_iter().multiunzip();

    Ok(BatchedRelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (evals_Az, evals_Bz, evals_Cz),
      evals_E,
      sc_proof_inner,
      evals_W,
      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
    })
  }

  fn verify(&self, vk: &Self::VerifierKey, U: &[RelaxedR1CSInstance<E>]) -> Result<(), NovaError> {
    let num_instances = U.len();
    let mut transcript = E::TE::new(b"BatchedRelaxedR1CSSNARK");

    transcript.absorb(b"vk", &vk.digest());
    if num_instances > 1 {
      let num_instances_field = E::Scalar::from(num_instances as u64);
      transcript.absorb(b"n", &num_instances_field);
    }
    U.iter().for_each(|u| {
      transcript.absorb(b"U", u);
    });

    let num_instances = U.len();

    let (num_rounds_x, num_rounds_y): (Vec<_>, Vec<_>) = vk
      .S
      .iter()
      .map(|s| (s.num_cons.log_2(), s.num_vars.log_2() + 1))
      .unzip();
    let num_rounds_x_max = *num_rounds_x.iter().max().unwrap();
    let num_rounds_y_max = *num_rounds_y.iter().max().unwrap();

    // Define τ polynomials of the appropriate size for each instance
    let polys_tau = {
      let tau = transcript.squeeze(b"t")?;

      num_rounds_x
        .iter()
        .map(|&num_rounds| PowPolynomial::new(&tau, num_rounds))
        .collect::<Vec<_>>()
    };

    // Sample challenge for random linear-combination of outer claims
    let outer_r = transcript.squeeze(b"out_r")?;
    let outer_r_powers = powers::<E>(&outer_r, num_instances);

    let (claim_outer_final, r_x) = self.sc_proof_outer.verify_batch(
      &vec![E::Scalar::ZERO; num_instances],
      &num_rounds_x,
      &outer_r_powers,
      3,
      &mut transcript,
    )?;

    // Since each instance has a different number of rounds, the Sumcheck
    // prover skips the first num_rounds_x_max - num_rounds_x rounds.
    // The evaluation point for each instance is therefore r_x[num_rounds_x_max - num_rounds_x..]
    let r_x = num_rounds_x
      .iter()
      .map(|num_rounds| r_x[(num_rounds_x_max - num_rounds)..].to_vec())
      .collect::<Vec<_>>();

    // Extract evaluations into a vector [(Azᵢ, Bzᵢ, Czᵢ, Eᵢ)]
    // TODO: This is a multizip, simplify
    let ABCE_evals = zip_with!(
      iter,
      (
        self.evals_E,
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2
      ),
      |eval_E, claim_Az, claim_Bz, claim_Cz| (*claim_Az, *claim_Bz, *claim_Cz, *eval_E)
    )
    .collect::<Vec<_>>();

    // Add evaluations of Az, Bz, Cz, E to transcript
    ABCE_evals
      .iter()
      .for_each(|(claim_Az, claim_Bz, claim_Cz, eval_E)| {
        transcript.absorb(
          b"claims_outer",
          &[*claim_Az, *claim_Bz, *claim_Cz, *eval_E].as_slice(),
        )
      });

    // Evaluate τ(rₓ) for each instance
    let evals_tau = zip_with!(iter, (polys_tau, r_x), |poly_tau, r_x| poly_tau
      .evaluate(r_x));

    // Compute expected claim for all instances ∑ᵢ rⁱ⋅τ(rₓ)⋅(Azᵢ⋅Bzᵢ − uᵢ⋅Czᵢ − Eᵢ)
    let claim_outer_final_expected = zip_with!(
      (
        ABCE_evals.iter().copied(),
        U.iter(),
        evals_tau,
        outer_r_powers.iter()
      ),
      |ABCE_eval, u, eval_tau, r| {
        let (claim_Az, claim_Bz, claim_Cz, eval_E) = ABCE_eval;
        *r * eval_tau * (claim_Az * claim_Bz - u.u * claim_Cz - eval_E)
      }
    )
    .sum::<E::Scalar>();

    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let inner_r = transcript.squeeze(b"in_r")?;
    let inner_r_square = inner_r.square();
    let inner_r_cube = inner_r_square * inner_r;
    let inner_r_powers = powers::<E>(&inner_r_cube, num_instances);

    // Compute inner claims Mzᵢ = (Azᵢ + r⋅Bzᵢ + r²⋅Czᵢ),
    // which are batched by Sumcheck into one claim:  ∑ᵢ r³ⁱ⋅Mzᵢ
    let claims_inner = ABCE_evals
      .into_iter()
      .map(|(claim_Az, claim_Bz, claim_Cz, _)| {
        claim_Az + inner_r * claim_Bz + inner_r_square * claim_Cz
      })
      .collect::<Vec<_>>();

    let (claim_inner_final, r_y) = self.sc_proof_inner.verify_batch(
      &claims_inner,
      &num_rounds_y,
      &inner_r_powers,
      2,
      &mut transcript,
    )?;
    let r_y: Vec<Vec<E::Scalar>> = num_rounds_y
      .iter()
      .map(|num_rounds| r_y[(num_rounds_y_max - num_rounds)..].to_vec())
      .collect();

    // Compute evaluations of Zᵢ = [Wᵢ, uᵢ, Xᵢ] at r_y
    // Zᵢ(r_y) = (1−r_y[0])⋅W(r_y[1..]) + r_y[0]⋅MLE([uᵢ, Xᵢ])(r_y[1..])
    let evals_Z = zip_with!(iter, (self.evals_W, U, r_y), |eval_W, U, r_y| {
      let eval_X = {
        let X = vec![U.u]
          .into_iter()
          .chain(U.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(r_y.len() - 1, X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * eval_W + r_y[0] * eval_X
    })
    .collect::<Vec<_>>();

    // compute evaluations of R1CS matrices M(r_x, r_y) = eq(r_y)ᵀ⋅M⋅eq(r_x)
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
        // TODO(@winston-h-zhang): review
        |M: &SparseMatrix<E::Scalar>, T_x: &[E::Scalar], T_y: &[E::Scalar]| -> E::Scalar {
          M.indptr
            .par_windows(2)
            .enumerate()
            .map(|(row_idx, ptrs)| {
              M.get_row_unchecked(ptrs.try_into().unwrap())
                .map(|(val, col_idx)| T_x[row_idx] * T_y[*col_idx] * val)
                .sum::<E::Scalar>()
            })
            .sum()
        };

      let (T_x, T_y) = rayon::join(
        || EqPolynomial::evals_from_points(r_x),
        || EqPolynomial::evals_from_points(r_y),
      );

      M_vec
        .par_iter()
        .map(|&M_vec| evaluate_with_table(M_vec, &T_x, &T_y))
        .collect()
    };

    // Compute inner claim ∑ᵢ r³ⁱ⋅(Aᵢ(r_x, r_y) + r⋅Bᵢ(r_x, r_y) + r²⋅Cᵢ(r_x, r_y))⋅Zᵢ(r_y)
    let claim_inner_final_expected = zip_with!(
      iter,
      (vk.S, r_x, r_y, evals_Z, inner_r_powers),
      |S, r_x, r_y, eval_Z, r_i| {
        let evals = multi_evaluate(&[&S.A, &S.B, &S.C], r_x, r_y);
        let eval = evals[0] + inner_r * evals[1] + inner_r_square * evals[2];
        eval * r_i * eval_Z
      }
    )
    .sum::<E::Scalar>();

    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // Create evaluation instances for W(r_y[1..]) and E(r_x)
    let u_vec = {
      let mut u_vec = Vec::with_capacity(2 * num_instances);
      u_vec.extend(zip_with!(iter, (self.evals_W, U, r_y), |eval, u, r_y| {
        PolyEvalInstance {
          c: u.comm_W,
          x: r_y[1..].to_vec(),
          e: *eval,
        }
      }));

      u_vec.extend(zip_with!(iter, (self.evals_E, U, r_x), |eval, u, r_x| {
        PolyEvalInstance {
          c: u.comm_E,
          x: r_x.to_vec(),
          e: *eval,
        }
      }));
      u_vec
    };

    let batched_u = batch_eval_verify(
      u_vec,
      &mut transcript,
      &self.sc_proof_batch,
      &self.evals_batch,
    )?;

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &batched_u.c,
      &batched_u.x,
      &batched_u.e,
      &self.eval_arg,
    )?;

    Ok(())
  }
}
