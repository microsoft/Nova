//! This module implements RelaxedR1CSSNARKTrait using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
pub mod polynomial;
mod sumcheck;

use crate::{
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    evaluation::EvaluationEngineTrait, snark::RelaxedR1CSSNARKTrait, AppendToTranscriptTrait,
    ChallengeTrait, Group, TranscriptEngineTrait,
  },
  CommitmentKey,
};
use ff::Field;
use itertools::concat;
use polynomial::{EqPolynomial, MultilinearPolynomial, SparsePolynomial};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sumcheck::SumcheckProof;

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  pk_ee: EE::ProverKey,
  S: R1CSShape<G>,
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  vk_ee: EE::VerifierKey,
  S: R1CSShape<G>,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  sc_proof_outer: SumcheckProof<G>,
  claims_outer: (G::Scalar, G::Scalar, G::Scalar),
  eval_E: G::Scalar,
  sc_proof_inner: SumcheckProof<G>,
  eval_W: G::Scalar,
  sc_proof_batch: SumcheckProof<G>,
  eval_E_prime: G::Scalar,
  eval_W_prime: G::Scalar,
  eval_arg: EE::EvaluationArgument,
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> RelaxedR1CSSNARKTrait<G>
  for RelaxedR1CSSNARK<G, EE>
{
  type ProverKey = ProverKey<G, EE>;
  type VerifierKey = VerifierKey<G, EE>;

  fn setup(ck: &CommitmentKey<G>, S: &R1CSShape<G>) -> (Self::ProverKey, Self::VerifierKey) {
    let (pk_ee, vk_ee) = EE::setup(ck);
    let pk = ProverKey { pk_ee, S: S.pad() };
    let vk = VerifierKey { vk_ee, S: S.pad() };

    (pk, vk)
  }

  /// produces a succinct proof of satisfiability of a RelaxedR1CS instance
  fn prove(
    ck: &CommitmentKey<G>,
    pk: &Self::ProverKey,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError> {
    let W = W.pad(&pk.S); // pad the witness
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");

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
      ((pk.S.num_vars as f64).log2() as usize + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| G::Scalar::challenge(b"tau", &mut transcript))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

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
    )?;

    // claims from the end of sum-check
    let (claim_Az, claim_Bz): (G::Scalar, G::Scalar) = (claims_outer[1], claims_outer[2]);
    let claim_Cz = poly_Cz.evaluate(&r_x);
    let eval_E = MultilinearPolynomial::new(W.E.clone()).evaluate(&r_x);

    <[G::Scalar] as AppendToTranscriptTrait<G>>::append_to_transcript(
      &[claim_Az, claim_Bz, claim_Cz, eval_E],
      b"claims_outer",
      &mut transcript,
    );

    // inner sum-check
    let r = G::Scalar::challenge(b"r", &mut transcript)?;
    let claim_inner_joint = claim_Az + r * claim_Bz + r * r * claim_Cz;

    let poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::new(r_x.clone()).evals();

      // Bounds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
      let compute_eval_table_sparse =
        |S: &R1CSShape<G>, rx: &[G::Scalar]| -> (Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>) {
          assert_eq!(rx.len(), S.num_cons);

          let inner = |M: &Vec<(usize, usize, G::Scalar)>, M_evals: &mut Vec<G::Scalar>| {
            for (row, col, val) in M {
              M_evals[*col] += rx[*row] * val;
            }
          };

          let (A_evals, (B_evals, C_evals)) = rayon::join(
            || {
              let mut A_evals: Vec<G::Scalar> = vec![G::Scalar::zero(); 2 * S.num_vars];
              inner(&S.A, &mut A_evals);
              A_evals
            },
            || {
              rayon::join(
                || {
                  let mut B_evals: Vec<G::Scalar> = vec![G::Scalar::zero(); 2 * S.num_vars];
                  inner(&S.B, &mut B_evals);
                  B_evals
                },
                || {
                  let mut C_evals: Vec<G::Scalar> = vec![G::Scalar::zero(); 2 * S.num_vars];
                  inner(&S.C, &mut C_evals);
                  C_evals
                },
              )
            },
          );

          (A_evals, B_evals, C_evals)
        };

      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(&pk.S, &evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .into_par_iter()
        .map(|i| evals_A[i] + r * evals_B[i] + r * r * evals_C[i])
        .collect::<Vec<G::Scalar>>()
    };

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
    )?;

    let eval_W = MultilinearPolynomial::new(W.W.clone()).evaluate(&r_y[1..]);
    <G::Scalar as AppendToTranscriptTrait<G>>::append_to_transcript(
      &eval_W,
      b"eval_W",
      &mut transcript,
    );

    // We will now reduce eval_W =? W(r_y[1..]) and eval_W =? E(r_x) into
    // two claims: eval_W_prime =? W(rz) and eval_E_prime =? E(rz)
    // We can them combine the two into one: eval_W_prime + gamma * eval_E_prime =? (W + gamma*E)(rz),
    // where gamma is a public challenge
    // Since commitments to W and E are homomorphic, the verifier can compute a commitment
    // to the batched polynomial.
    let rho = G::Scalar::challenge(b"rho", &mut transcript)?;

    let claim_batch_joint = eval_E + rho * eval_W;
    let num_rounds_z = num_rounds_x;
    let comb_func =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * *poly_B_comp + rho * *poly_C_comp * *poly_D_comp };
    let (sc_proof_batch, r_z, claims_batch) = SumcheckProof::prove_quad_sum(
      &claim_batch_joint,
      num_rounds_z,
      &mut MultilinearPolynomial::new(EqPolynomial::new(r_x).evals()),
      &mut MultilinearPolynomial::new(W.E.clone()),
      &mut MultilinearPolynomial::new(EqPolynomial::new(r_y[1..].to_vec()).evals()),
      &mut MultilinearPolynomial::new(W.W.clone()),
      comb_func,
      &mut transcript,
    )?;

    let eval_E_prime = claims_batch[1];
    let eval_W_prime = claims_batch[3];
    <[G::Scalar] as AppendToTranscriptTrait<G>>::append_to_transcript(
      &[eval_E_prime, eval_W_prime],
      b"claims_batch",
      &mut transcript,
    );

    // we now combine evaluation claims at the same point rz into one
    let gamma = G::Scalar::challenge(b"gamma", &mut transcript)?;
    let comm = U.comm_E + U.comm_W * gamma;
    let poly = W
      .E
      .iter()
      .zip(W.W.iter())
      .map(|(e, w)| *e + gamma * w)
      .collect::<Vec<G::Scalar>>();
    let eval = eval_E_prime + gamma * eval_W_prime;

    let eval_arg = EE::prove(ck, &pk.pk_ee, &mut transcript, &comm, &poly, &r_z, &eval)?;

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      eval_E,
      sc_proof_inner,
      eval_W,
      sc_proof_batch,
      eval_E_prime,
      eval_W_prime,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a RelaxedR1CS instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError> {
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");

    // append the R1CSShape and RelaxedR1CSInstance to the transcript
    vk.S.append_to_transcript(b"S", &mut transcript);
    U.append_to_transcript(b"U", &mut transcript);

    let (num_rounds_x, num_rounds_y) = (
      (vk.S.num_cons as f64).log2() as usize,
      ((vk.S.num_vars as f64).log2() as usize + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| G::Scalar::challenge(b"tau", &mut transcript))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

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

    <[G::Scalar] as AppendToTranscriptTrait<G>>::append_to_transcript(
      &[
        self.claims_outer.0,
        self.claims_outer.1,
        self.claims_outer.2,
        self.eval_E,
      ],
      b"claims_outer",
      &mut transcript,
    );

    // inner sum-check
    let r = G::Scalar::challenge(b"r", &mut transcript)?;
    let claim_inner_joint =
      self.claims_outer.0 + r * self.claims_outer.1 + r * r * self.claims_outer.2;

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
        SparsePolynomial::new((vk.S.num_vars as f64).log2() as usize, poly_X).evaluate(&r_y[1..])
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
            .collect::<Vec<usize>>()
            .par_iter()
            .map(|&i| {
              let (row, col, val) = M[i];
              T_x[row] * T_y[col] * val
            })
            .reduce(G::Scalar::zero, |acc, x| acc + x)
        };

      let T_x = EqPolynomial::new(r_x.to_vec()).evals();
      let T_y = EqPolynomial::new(r_y.to_vec()).evals();
      let eval_A_r = evaluate_with_table(&S.A, &T_x, &T_y);
      let eval_B_r = evaluate_with_table(&S.B, &T_x, &T_y);
      let eval_C_r = evaluate_with_table(&S.C, &T_x, &T_y);
      (eval_A_r, eval_B_r, eval_C_r)
    };

    let (eval_A_r, eval_B_r, eval_C_r) = evaluate_as_sparse_polynomial(&vk.S, &r_x, &r_y);
    let claim_inner_final_expected = (eval_A_r + r * eval_B_r + r * r * eval_C_r) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // batch sum-check
    <G::Scalar as AppendToTranscriptTrait<G>>::append_to_transcript(
      &self.eval_W,
      b"eval_W",
      &mut transcript,
    );

    let rho = G::Scalar::challenge(b"rho", &mut transcript)?;
    let claim_batch_joint = self.eval_E + rho * self.eval_W;
    let num_rounds_z = num_rounds_x;
    let (claim_batch_final, r_z) =
      self
        .sc_proof_batch
        .verify(claim_batch_joint, num_rounds_z, 2, &mut transcript)?;

    let claim_batch_final_expected = {
      let poly_rz = EqPolynomial::new(r_z.clone());
      let rz_rx = poly_rz.evaluate(&r_x);
      let rz_ry = poly_rz.evaluate(&r_y[1..]);
      rz_rx * self.eval_E_prime + rho * rz_ry * self.eval_W_prime
    };

    if claim_batch_final != claim_batch_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    <[G::Scalar] as AppendToTranscriptTrait<G>>::append_to_transcript(
      &[self.eval_E_prime, self.eval_W_prime],
      b"claims_batch",
      &mut transcript,
    );

    // we now combine evaluation claims at the same point rz into one
    let gamma = G::Scalar::challenge(b"gamma", &mut transcript)?;
    let comm = U.comm_E + U.comm_W * gamma;
    let eval = self.eval_E_prime + gamma * self.eval_W_prime;

    // verify eval_W and eval_E
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &comm,
      &r_z,
      &eval,
      &self.eval_arg,
    )?;

    Ok(())
  }
}
