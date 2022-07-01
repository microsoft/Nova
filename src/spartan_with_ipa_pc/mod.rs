//! This module implements RelaxedR1CSSNARKTrait using a Spartan variant
//! instantiated with an IPA-based polynomial commitment scheme
mod ipa;
mod polynomial;
mod sumcheck;

use super::{
  commitments::CommitGens,
  errors::NovaError,
  r1cs::{R1CSGens, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  snark::{ProverKeyTrait, RelaxedR1CSSNARKTrait, VerifierKeyTrait},
  traits::{AppendToTranscriptTrait, ChallengeTrait, Group},
};
use core::cmp::max;
use ff::Field;
use ipa::{
  FinalInnerProductArgument, InnerProductInstance, InnerProductWitness, StepInnerProductArgument,
};
use itertools::concat;
use merlin::Transcript;
use polynomial::{EqPolynomial, MultilinearPolynomial, SparsePolynomial};
use rayon::prelude::*;
use sumcheck::SumcheckProof;

/// A type that represents the prover's key
pub struct ProverKey<G: Group> {
  gens_r1cs: R1CSGens<G>,
  gens_ipa: CommitGens<G>,
  S: R1CSShape<G>,
}

impl<G: Group> ProverKeyTrait<G> for ProverKey<G> {
  fn new(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self {
    ProverKey {
      gens_r1cs: gens.clone(),
      gens_ipa: CommitGens::new(b"ipa", 1),
      S: S.clone(),
    }
  }
}

/// A type that represents the verifier's key
pub struct VerifierKey<G: Group> {
  gens_r1cs: R1CSGens<G>,
  gens_ipa: CommitGens<G>,
  S: R1CSShape<G>,
}

impl<G: Group> VerifierKeyTrait<G> for VerifierKey<G> {
  fn new(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self {
    VerifierKey {
      gens_r1cs: gens.clone(),
      gens_ipa: CommitGens::new(b"ipa", 1),
      S: S.clone(),
    }
  }
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
pub struct RelaxedR1CSSNARK<G: Group> {
  sc_proof_outer: SumcheckProof<G>,
  claims_outer: (G::Scalar, G::Scalar, G::Scalar),
  sc_proof_inner: SumcheckProof<G>,
  eval_E: G::Scalar,
  eval_W: G::Scalar,
  step_ipa: StepInnerProductArgument<G>,
  final_ipa: FinalInnerProductArgument<G>,
}

impl<G: Group> RelaxedR1CSSNARKTrait<G> for RelaxedR1CSSNARK<G> {
  type ProverKey = ProverKey<G>;
  type VerifierKey = VerifierKey<G>;

  /// produces a succinct proof of satisfiability of a RelaxedR1CS instance
  fn prove(
    pk: &Self::ProverKey,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError> {
    let mut transcript = Transcript::new(b"RelaxedR1CSSNARK");

    debug_assert!(pk.S.is_sat_relaxed(&pk.gens_r1cs, U, W).is_ok());

    // sanity check that R1CSShape has certain size characteristics
    assert_eq!(pk.S.num_cons.next_power_of_two(), pk.S.num_cons);
    assert_eq!(pk.S.num_vars.next_power_of_two(), pk.S.num_vars);
    assert_eq!(pk.S.num_io.next_power_of_two(), pk.S.num_io);
    assert!(pk.S.num_io < pk.S.num_vars);

    // append the R1CSShape and RelaxedR1CSInstance to the transcript
    pk.S.append_to_transcript(b"S", &mut transcript);
    U.append_to_transcript(b"U", &mut transcript);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let mut z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);

    let (num_rounds_x, num_rounds_y) = (
      (pk.S.num_cons as f64).log2() as usize,
      ((pk.S.num_vars as f64).log2() as usize + 1) as usize,
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| G::Scalar::challenge(b"challenge_tau", &mut transcript))
      .collect();

    let mut poly_tau = MultilinearPolynomial::new(EqPolynomial::new(tau).evals());
    let (mut poly_Az, mut poly_Bz, poly_Cz, mut poly_uCz_E) = {
      let (poly_Az, poly_Bz, poly_Cz) = pk.S.multiply_vec(&z)?;
      let poly_uCz_E = (0..pk.S.num_cons)
        .map(|i| U.u * poly_Cz[i] + W.E[i])
        .collect::<Vec<G::Scalar>>();
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
        MultilinearPolynomial::new(poly_uCz_E),
      )
    };

    let comb_func_outer =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_additive_term(
      &G::Scalar::zero(), // claim is zero
      num_rounds_x,
      &mut poly_tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_uCz_E,
      comb_func_outer,
      &mut transcript,
    );

    // claims from the end of sum-check
    let (claim_Az, claim_Bz): (G::Scalar, G::Scalar) = (claims_outer[1], claims_outer[2]);

    claim_Az.append_to_transcript(b"claim_Az", &mut transcript);
    claim_Bz.append_to_transcript(b"claim_Bz", &mut transcript);
    let claim_Cz = poly_Cz.evaluate(&r_x);
    let eval_E = MultilinearPolynomial::new(W.E.clone()).evaluate(&r_x);
    claim_Cz.append_to_transcript(b"claim_Cz", &mut transcript);
    eval_E.append_to_transcript(b"eval_E", &mut transcript);

    // inner sum-check
    let r_A = G::Scalar::challenge(b"challenge_rA", &mut transcript);
    let r_B = G::Scalar::challenge(b"challenge_rB", &mut transcript);
    let r_C = G::Scalar::challenge(b"challenge_rC", &mut transcript);
    let claim_inner_joint = r_A * claim_Az + r_B * claim_Bz + r_C * claim_Cz;

    let poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::new(r_x.clone()).evals();
      let (evals_A, evals_B, evals_C) = pk.S.compute_eval_table_sparse(&evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| r_A * evals_A[i] + r_B * evals_B[i] + r_C * evals_C[i])
        .collect::<Vec<G::Scalar>>()
    };

    // TODO: fix this
    let poly_z = {
      z.resize(pk.S.num_vars * 2, G::Scalar::zero());
      z
    };

    let comb_func = |poly_A_comp: &G::Scalar, poly_B_comp: &G::Scalar| -> G::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
      comb_func,
      &mut transcript,
    );

    let eval_W = MultilinearPolynomial::new(W.W.clone()).evaluate(&r_y[1..]);
    eval_W.append_to_transcript(b"eval_W", &mut transcript);

    let (step_ipa, r_U, r_W) = StepInnerProductArgument::prove(
      &InnerProductInstance::new(&U.comm_E, &EqPolynomial::new(r_x).evals(), &eval_E),
      &InnerProductWitness::new(&W.E),
      &InnerProductInstance::new(
        &U.comm_W,
        &EqPolynomial::new(r_y[1..].to_vec()).evals(),
        &eval_W,
      ),
      &InnerProductWitness::new(&W.W),
      &mut transcript,
    );

    let final_ipa = FinalInnerProductArgument::prove(
      &pk.gens_r1cs.gens,
      &pk.gens_ipa,
      &r_U,
      &r_W,
      &mut transcript,
    )?;

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      eval_W,
      eval_E,
      step_ipa,
      final_ipa,
    })
  }

  /// verifies a proof of satisfiability of a RelaxedR1CS instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError> {
    let mut transcript = Transcript::new(b"RelaxedR1CSSNARK");

    // append the R1CSShape and RelaxedR1CSInstance to the transcript
    vk.S.append_to_transcript(b"S", &mut transcript);
    U.append_to_transcript(b"U", &mut transcript);

    let (num_rounds_x, num_rounds_y) = (
      (vk.S.num_cons as f64).log2() as usize,
      ((vk.S.num_vars as f64).log2() as usize + 1) as usize,
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| G::Scalar::challenge(b"challenge_tau", &mut transcript))
      .collect::<Vec<G::Scalar>>();

    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(G::Scalar::zero(), num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let (claim_Az, claim_Bz, claim_Cz) = self.claims_outer;
    let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
    let claim_outer_final_expected =
      taus_bound_rx * (claim_Az * claim_Bz - U.u * claim_Cz - self.eval_E);
    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    self
      .claims_outer
      .0
      .append_to_transcript(b"claim_Az", &mut transcript);
    self
      .claims_outer
      .1
      .append_to_transcript(b"claim_Bz", &mut transcript);
    self
      .claims_outer
      .2
      .append_to_transcript(b"claim_Cz", &mut transcript);
    self.eval_E.append_to_transcript(b"eval_E", &mut transcript);

    // inner sum-check
    let r_A = G::Scalar::challenge(b"challenge_rA", &mut transcript);
    let r_B = G::Scalar::challenge(b"challenge_rB", &mut transcript);
    let r_C = G::Scalar::challenge(b"challenge_rC", &mut transcript);
    let claim_inner_joint =
      r_A * self.claims_outer.0 + r_B * self.claims_outer.1 + r_C * self.claims_outer.2;

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, &mut transcript)?;

    // verify claim_inner_final
    let eval_Z = {
      let eval_X = {
        // constant term
        let mut poly_X = vec![(0, U.u)];
        //remaining inputs
        poly_X.extend(
          (0..U.X.len())
            .map(|i| (i + 1, U.X[i]))
            .collect::<Vec<(usize, G::Scalar)>>(),
        );
        SparsePolynomial::new((vk.S.num_vars as f64).log2() as usize, poly_X)
          .evaluate(&r_y[1..].to_vec())
      };
      (G::Scalar::one() - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    let evaluate_as_sparse_polynomial = |S: &R1CSShape<G>,
                                         r_x: &[G::Scalar],
                                         r_y: &[G::Scalar]|
     -> (G::Scalar, G::Scalar, G::Scalar) {
      let evaluate_with_table =
        |M: &[(usize, usize, G::Scalar)], T_x: &[G::Scalar], T_y: &[G::Scalar]| -> G::Scalar {
          (0..M.len())
            .map(|i| {
              let (row, col, val) = M[i];
              T_x[row] * T_y[col] * val
            })
            .fold(G::Scalar::zero(), |acc, x| acc + x)
        };

      let T_x = EqPolynomial::new(r_x.to_vec()).evals();
      let T_y = EqPolynomial::new(r_y.to_vec()).evals();
      let eval_A_r = evaluate_with_table(&S.A, &T_x, &T_y);
      let eval_B_r = evaluate_with_table(&S.B, &T_x, &T_y);
      let eval_C_r = evaluate_with_table(&S.C, &T_x, &T_y);
      (eval_A_r, eval_B_r, eval_C_r)
    };

    let (eval_A_r, eval_B_r, eval_C_r) = evaluate_as_sparse_polynomial(&vk.S, &r_x, &r_y);
    let claim_inner_final_expected = (r_A * eval_A_r + r_B * eval_B_r + r_C * eval_C_r) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // verify eval_W and eval_E
    self.eval_W.append_to_transcript(b"eval_W", &mut transcript); //eval_E is already in the transcript

    let r_U = self.step_ipa.verify(
      &InnerProductInstance::new(&U.comm_E, &EqPolynomial::new(r_x).evals(), &self.eval_E),
      &InnerProductInstance::new(
        &U.comm_W,
        &EqPolynomial::new(r_y[1..].to_vec()).evals(),
        &self.eval_W,
      ),
      &mut transcript,
    );

    self.final_ipa.verify(
      &vk.gens_r1cs.gens,
      &vk.gens_ipa,
      max(vk.S.num_vars, vk.S.num_cons),
      &r_U,
      &mut transcript,
    )?;

    Ok(())
  }
}
