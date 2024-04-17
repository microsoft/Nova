//! This module implements `RelaxedR1CSSNARKTrait` using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
//! This version of Spartan does not use preprocessing so the verifier keeps the entire
//! description of R1CS matrices. This is essentially optimal for the verifier when using
//! an IPA-based polynomial commitment scheme.

use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness, SparseMatrix},
  spartan::{
    compute_eval_table_sparse,
    math::Math,
    polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, multilinear::SparsePolynomial},
    powers,
    sumcheck::SumcheckProof,
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    evaluation::EvaluationEngineTrait,
    snark::{DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine, TranscriptEngineTrait,
  },
  CommitmentKey,
};

use ff::Field;
use itertools::Itertools as _;
use once_cell::sync::OnceCell;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  vk_digest: E::Scalar, // digest of the verifier's key
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  vk_ee: EE::VerifierKey,
  S: R1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> {}

impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE> {
  fn new(shape: R1CSShape<E>, vk_ee: EE::VerifierKey) -> Self {
    VerifierKey {
      vk_ee,
      S: shape,
      digest: OnceCell::new(),
    }
  }
}

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

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<E: Engine, EE: EvaluationEngineTrait<E>> {
  sc_proof_outer: SumcheckProof<E>,
  claims_outer: (E::Scalar, E::Scalar, E::Scalar),
  eval_E: E::Scalar,
  sc_proof_inner: SumcheckProof<E>,
  eval_W: E::Scalar,
  sc_proof_batch: SumcheckProof<E>,
  evals_batch: Vec<E::Scalar>,
  eval_arg: EE::EvaluationArgument,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARKTrait<E> for RelaxedR1CSSNARK<E, EE> {
  type ProverKey = ProverKey<E, EE>;
  type VerifierKey = VerifierKey<E, EE>;

  fn setup(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck);

    let S = S.pad();

    let vk: VerifierKey<E, EE> = VerifierKey::new(S, vk_ee);

    let pk = ProverKey {
      pk_ee,
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }

  /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<Self, NovaError> {
    // pad the R1CSShape
    let S = S.pad();
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());

    let W = W.pad(&S); // pad the witness
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the digest of vk (which includes R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let mut z = [W.W.clone(), vec![U.u], U.X.clone()].concat();

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(S.num_cons.ilog2()).unwrap(),
      (usize::try_from(S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, NovaError>>()?;

    let mut poly_tau = MultilinearPolynomial::new(tau.evals());
    let (mut poly_Az, mut poly_Bz, poly_Cz, mut poly_uCz_E) = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z)?;
      let poly_uCz_E = (0..S.num_cons)
        .map(|i| U.u * poly_Cz[i] + W.E[i])
        .collect::<Vec<E::Scalar>>();
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
        MultilinearPolynomial::new(poly_uCz_E),
      )
    };

    let comb_func_outer =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_additive_term(
      &E::Scalar::ZERO, // claim is zero
      num_rounds_x,
      &mut poly_tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_uCz_E,
      comb_func_outer,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    let (claim_Az, claim_Bz): (E::Scalar, E::Scalar) = (claims_outer[1], claims_outer[2]);
    let claim_Cz = poly_Cz.evaluate(&r_x);
    let eval_E = MultilinearPolynomial::new(W.E.clone()).evaluate(&r_x);
    transcript.absorb(
      b"claims_outer",
      &[claim_Az, claim_Bz, claim_Cz, eval_E].as_slice(),
    );

    // inner sum-check
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::evals_from_points(&r_x.clone());

      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&S, &evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<E::Scalar>>()
    };

    let poly_z = {
      z.resize(S.num_vars * 2, E::Scalar::ZERO);
      z
    };

    let comb_func = |poly_A_comp: &E::Scalar, poly_B_comp: &E::Scalar| -> E::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      &mut transcript,
    )?;

    // Add additional claims about W and E polynomials to the list from CC
    // We will reduce a vector of claims of evaluations at different points into claims about them at the same point.
    // For example, eval_W =? W(r_y[1..]) and eval_E =? E(r_x) into
    // two claims: eval_W_prime =? W(rz) and eval_E_prime =? E(rz)
    // We can them combine the two into one: eval_W_prime + gamma * eval_E_prime =? (W + gamma*E)(rz),
    // where gamma is a public challenge
    // Since commitments to W and E are homomorphic, the verifier can compute a commitment
    // to the batched polynomial.
    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);

    let w_vec = vec![PolyEvalWitness { p: W.W }, PolyEvalWitness { p: W.E }];
    let u_vec = vec![
      PolyEvalInstance {
        c: U.comm_W,
        x: r_y[1..].to_vec(),
        e: eval_W,
      },
      PolyEvalInstance {
        c: U.comm_E,
        x: r_x,
        e: eval_E,
      },
    ];

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

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      eval_E,
      sc_proof_inner,
      eval_W,
      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<E>) -> Result<(), NovaError> {
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the digest of R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    let (num_rounds_x, num_rounds_y) = (
      usize::try_from(vk.S.num_cons.ilog2()).unwrap(),
      (usize::try_from(vk.S.num_vars.ilog2()).unwrap() + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<EqPolynomial<_>, NovaError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(E::Scalar::ZERO, num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = tau.evaluate(&r_x);
    let claim_outer_final_expected =
      taus_bound_rx * (claim_Az * claim_Bz - U.u * claim_Cz - self.eval_E);
    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"claims_outer",
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
        self.eval_E,
      ]
      .as_slice(),
    );

    // inner sum-check
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint =
      self.claims_outer.0 + r * self.claims_outer.1 + r * r * self.claims_outer.2;

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    // verify claim_inner_final
    let eval_Z = {
      let eval_X = {
        // public IO is (u, X)
        let X = vec![U.u]
          .into_iter()
          .chain(U.X.iter().cloned())
          .collect::<Vec<E::Scalar>>();
        SparsePolynomial::new(vk.S.num_vars.log_2(), X).evaluate(&r_y[1..])
      };
      (E::Scalar::ONE - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    // compute evaluations of R1CS matrices
    let multi_evaluate = |M_vec: &[&SparseMatrix<E::Scalar>],
                          r_x: &[E::Scalar],
                          r_y: &[E::Scalar]|
     -> Vec<E::Scalar> {
      let evaluate_with_table =
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

      (0..M_vec.len())
        .into_par_iter()
        .map(|i| evaluate_with_table(M_vec[i], &T_x, &T_y))
        .collect()
    };

    let evals = multi_evaluate(&[&vk.S.A, &vk.S.B, &vk.S.C], &r_x, &r_y);

    let claim_inner_final_expected = (evals[0] + r * evals[1] + r * r * evals[2]) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // add claims about W and E polynomials
    let u_vec: Vec<PolyEvalInstance<E>> = vec![
      PolyEvalInstance {
        c: U.comm_W,
        x: r_y[1..].to_vec(),
        e: self.eval_W,
      },
      PolyEvalInstance {
        c: U.comm_E,
        x: r_x,
        e: self.eval_E,
      },
    ];

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

/// Reduces a batch of polynomial evaluation claims using Sumcheck
/// to a single claim at the same point.
///
/// # Details
///
/// We are given as input a list of instance/witness pairs
/// u = [(Cᵢ, xᵢ, eᵢ)], w = [Pᵢ], such that
/// - nᵢ = |xᵢ|
/// - Cᵢ = Commit(Pᵢ)
/// - eᵢ = Pᵢ(xᵢ)
/// - |Pᵢ| = 2^nᵢ
///
/// We allow the polynomial Pᵢ to have different sizes, by appropriately scaling
/// the claims and resulting evaluations from Sumcheck.
fn batch_eval_reduce<E: Engine>(
  u_vec: Vec<PolyEvalInstance<E>>,
  w_vec: Vec<PolyEvalWitness<E>>,
  transcript: &mut E::TE,
) -> Result<
  (
    PolyEvalInstance<E>,
    PolyEvalWitness<E>,
    SumcheckProof<E>,
    Vec<E::Scalar>,
  ),
  NovaError,
> {
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

  let (claims, u_xs, comms): (Vec<_>, Vec<_>, Vec<_>) =
    u_vec.into_iter().map(|u| (u.e, u.x, u.c)).multiunzip();

  // Create clones of polynomials to be given to Sumcheck
  // Pᵢ(X)
  let polys_P: Vec<MultilinearPolynomial<E::Scalar>> = w_vec
    .iter()
    .map(|w| MultilinearPolynomial::new(w.p.clone()))
    .collect();
  // eq(xᵢ, X)
  let polys_eq: Vec<MultilinearPolynomial<E::Scalar>> = u_xs
    .into_iter()
    .map(|ux| MultilinearPolynomial::new(EqPolynomial::evals_from_points(&ux)))
    .collect();

  // For each i, check eᵢ = ∑ₓ Pᵢ(x)eq(xᵢ,x), where x ∈ {0,1}^nᵢ
  let comb_func = |poly_P: &E::Scalar, poly_eq: &E::Scalar| -> E::Scalar { *poly_P * *poly_eq };
  let (sc_proof_batch, r, claims_batch) = SumcheckProof::prove_quad_batch(
    &claims,
    &num_rounds,
    polys_P,
    polys_eq,
    &powers_of_rho,
    comb_func,
    transcript,
  )?;

  let (claims_batch_left, _): (Vec<E::Scalar>, Vec<E::Scalar>) = claims_batch;

  transcript.absorb(b"l", &claims_batch_left.as_slice());

  // we now combine evaluation claims at the same point r into one
  let gamma = transcript.squeeze(b"g")?;

  let u_joint =
    PolyEvalInstance::batch_diff_size(&comms, &claims_batch_left, &num_rounds, r, gamma);

  // P = ∑ᵢ γⁱ⋅Pᵢ
  let w_joint = PolyEvalWitness::batch_diff_size(w_vec, gamma);

  Ok((u_joint, w_joint, sc_proof_batch, claims_batch_left))
}

/// Verifies a batch of polynomial evaluation claims using Sumcheck
/// reducing them to a single claim at the same point.
fn batch_eval_verify<E: Engine>(
  u_vec: Vec<PolyEvalInstance<E>>,
  transcript: &mut E::TE,
  sc_proof_batch: &SumcheckProof<E>,
  evals_batch: &[E::Scalar],
) -> Result<PolyEvalInstance<E>, NovaError> {
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
  let gamma = transcript.squeeze(b"g")?;

  let comms = u_vec.into_iter().map(|u| u.c).collect::<Vec<_>>();

  let u_joint = PolyEvalInstance::batch_diff_size(&comms, evals_batch, &num_rounds, r, gamma);

  Ok(u_joint)
}
