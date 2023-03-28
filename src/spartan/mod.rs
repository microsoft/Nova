//! This module implements RelaxedR1CSSNARKTrait using Spartan that is generic
//! over the polynomial commitment and evaluation argument (i.e., a PCS)
mod math;
pub(crate) mod polynomial;
pub mod spark;
mod sumcheck;

use crate::{
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    evaluation::EvaluationEngineTrait, snark::RelaxedR1CSSNARKTrait, Group, TranscriptEngineTrait,
    TranscriptReprTrait,
  },
  Commitment, CommitmentKey,
};
use ff::Field;
use itertools::concat;
use polynomial::{EqPolynomial, MultilinearPolynomial, SparsePolynomial};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sumcheck::SumcheckProof;

/// A type that holds a witness to a polynomial evaluation instance
#[allow(dead_code)]
pub struct PolyEvalWitness<G: Group> {
  p: Vec<G::Scalar>, // polynomial
}

impl<G: Group> PolyEvalWitness<G> {
  fn pad(W: &[PolyEvalWitness<G>]) -> Vec<PolyEvalWitness<G>> {
    // determine the maximum size
    if let Some(n) = W.iter().map(|w| w.p.len()).max() {
      W.iter()
        .map(|w| {
          let mut p = w.p.clone();
          p.resize(n, G::Scalar::zero());
          PolyEvalWitness { p }
        })
        .collect()
    } else {
      Vec::new()
    }
  }

  fn weighted_sum(W: &[PolyEvalWitness<G>], s: &[G::Scalar]) -> PolyEvalWitness<G> {
    assert_eq!(W.len(), s.len());
    let mut p = vec![G::Scalar::zero(); W[0].p.len()];
    for i in 0..W.len() {
      for j in 0..W[i].p.len() {
        p[j] += W[i].p[j] * s[i]
      }
    }
    PolyEvalWitness { p }
  }
}

/// A type that holds a polynomial evaluation instance
#[allow(dead_code)]
pub struct PolyEvalInstance<G: Group> {
  c: Commitment<G>,  // commitment to the polynomial
  x: Vec<G::Scalar>, // evaluation point
  e: G::Scalar,      // claimed evaluation
}

impl<G: Group> PolyEvalInstance<G> {
  fn pad(U: &[PolyEvalInstance<G>]) -> Vec<PolyEvalInstance<G>> {
    // determine the maximum size
    if let Some(ell) = U.iter().map(|u| u.x.len()).max() {
      U.iter()
        .map(|u| {
          let mut x = vec![G::Scalar::zero(); ell - u.x.len()];
          x.extend(u.x.clone());
          PolyEvalInstance { c: u.c, x, e: u.e }
        })
        .collect()
    } else {
      Vec::new()
    }
  }
}

/// A trait that defines the behavior of a computation commitment engine
pub trait CompCommitmentEngineTrait<G: Group> {
  /// A type that holds opening hint
  type Decommitment: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds a commitment
  type Commitment: Clone
    + Send
    + Sync
    + TranscriptReprTrait<G>
    + Serialize
    + for<'de> Deserialize<'de>;

  /// A type that holds an evaluation argument
  type EvaluationArgument: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// commits to R1CS matrices
  fn commit(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
  ) -> Result<(Self::Commitment, Self::Decommitment), NovaError>;

  /// proves an evaluation of R1CS matrices viewed as polynomials
  fn prove(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
    decomm: &Self::Decommitment,
    comm: &Self::Commitment,
    r: &(&[G::Scalar], &[G::Scalar]),
    transcript: &mut G::TE,
  ) -> Result<
    (
      Self::EvaluationArgument,
      Vec<(PolyEvalWitness<G>, PolyEvalInstance<G>)>,
    ),
    NovaError,
  >;

  /// verifies an evaluation of R1CS matrices viewed as polynomials and returns verified evaluations
  fn verify(
    comm: &Self::Commitment,
    r: &(&[G::Scalar], &[G::Scalar]),
    arg: &Self::EvaluationArgument,
    transcript: &mut G::TE,
  ) -> Result<(G::Scalar, G::Scalar, G::Scalar, Vec<PolyEvalInstance<G>>), NovaError>;
}

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<
  G: Group,
  EE: EvaluationEngineTrait<G, CE = G::CE>,
  CC: CompCommitmentEngineTrait<G>,
> {
  pk_ee: EE::ProverKey,
  S: R1CSShape<G>,
  decomm: CC::Decommitment,
  comm: CC::Commitment,
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<
  G: Group,
  EE: EvaluationEngineTrait<G, CE = G::CE>,
  CC: CompCommitmentEngineTrait<G>,
> {
  num_cons: usize,
  num_vars: usize,
  vk_ee: EE::VerifierKey,
  comm: CC::Commitment,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<
  G: Group,
  EE: EvaluationEngineTrait<G, CE = G::CE>,
  CC: CompCommitmentEngineTrait<G>,
> {
  sc_proof_outer: SumcheckProof<G>,
  claims_outer: (G::Scalar, G::Scalar, G::Scalar),
  eval_E: G::Scalar,
  sc_proof_inner: SumcheckProof<G>,
  eval_W: G::Scalar,
  eval_arg_cc: CC::EvaluationArgument,
  sc_proof_batch: SumcheckProof<G>,
  evals_batch: Vec<G::Scalar>,
  eval_arg: EE::EvaluationArgument,
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>, CC: CompCommitmentEngineTrait<G>>
  RelaxedR1CSSNARKTrait<G> for RelaxedR1CSSNARK<G, EE, CC>
{
  type ProverKey = ProverKey<G, EE, CC>;
  type VerifierKey = VerifierKey<G, EE, CC>;

  fn setup(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck);

    let S = S.pad();

    let (comm, decomm) = CC::commit(ck, &S)?;

    let vk = VerifierKey {
      num_cons: S.num_cons,
      num_vars: S.num_vars,
      vk_ee,
      comm: comm.clone(),
    };

    let pk = ProverKey {
      pk_ee,
      S,
      comm,
      decomm,
    };

    Ok((pk, vk))
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

    // append the commitment to R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"C", &pk.comm);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let mut z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);

    let (num_rounds_x, num_rounds_y) = (
      (pk.S.num_cons as f64).log2() as usize,
      ((pk.S.num_vars as f64).log2() as usize + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
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
    transcript.absorb(
      b"claims_outer",
      &[claim_Az, claim_Bz, claim_Cz, eval_E].as_slice(),
    );

    // inner sum-check
    let r = transcript.squeeze(b"r")?;
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

    // we now prove evaluations of R1CS matrices at (r_x, r_y)
    let (eval_arg_cc, mut w_u_vec) = CC::prove(
      ck,
      &pk.S,
      &pk.decomm,
      &pk.comm,
      &(&r_x, &r_y),
      &mut transcript,
    )?;

    // add additional claims about W and E polynomials to the list from CC
    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_y[1..]);
    w_u_vec.push((
      PolyEvalWitness { p: W.W.clone() },
      PolyEvalInstance {
        c: U.comm_W,
        x: r_y[1..].to_vec(),
        e: eval_W,
      },
    ));

    w_u_vec.push((
      PolyEvalWitness { p: W.E },
      PolyEvalInstance {
        c: U.comm_E,
        x: r_x,
        e: eval_E,
      },
    ));

    // We will now reduce a vector of claims of evaluations at different points into claims about them at the same point.
    // For example, eval_W =? W(r_y[1..]) and eval_W =? E(r_x) into
    // two claims: eval_W_prime =? W(rz) and eval_E_prime =? E(rz)
    // We can them combine the two into one: eval_W_prime + gamma * eval_E_prime =? (W + gamma*E)(rz),
    // where gamma is a public challenge
    // Since commitments to W and E are homomorphic, the verifier can compute a commitment
    // to the batched polynomial.
    assert!(w_u_vec.len() >= 2);

    let (w_vec, u_vec): (Vec<PolyEvalWitness<G>>, Vec<PolyEvalInstance<G>>) =
      w_u_vec.into_iter().unzip();
    let w_vec_padded = PolyEvalWitness::pad(&w_vec); // pad the polynomials to be of the same size
    let u_vec_padded = PolyEvalInstance::pad(&u_vec); // pad the evaluation points

    let powers = |s: &G::Scalar, n: usize| -> Vec<G::Scalar> {
      assert!(n >= 1);
      let mut powers = Vec::new();
      powers.push(G::Scalar::one());
      for i in 1..n {
        powers.push(powers[i - 1] * s);
      }
      powers
    };

    // generate a challenge
    let rho = transcript.squeeze(b"r")?;
    let num_claims = w_vec_padded.len();
    let powers_of_rho = powers(&rho, num_claims);
    let claim_batch_joint = u_vec_padded
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(u, p)| u.e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let mut polys_left: Vec<MultilinearPolynomial<G::Scalar>> = w_vec_padded
      .iter()
      .map(|w| MultilinearPolynomial::new(w.p.clone()))
      .collect();
    let mut polys_right: Vec<MultilinearPolynomial<G::Scalar>> = u_vec_padded
      .iter()
      .map(|u| MultilinearPolynomial::new(EqPolynomial::new(u.x.clone()).evals()))
      .collect();

    let num_rounds_z = u_vec_padded[0].x.len();
    let comb_func = |poly_A_comp: &G::Scalar, poly_B_comp: &G::Scalar| -> G::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_batch, r_z, claims_batch) = SumcheckProof::prove_quad_batch(
      &claim_batch_joint,
      num_rounds_z,
      &mut polys_left,
      &mut polys_right,
      &powers_of_rho,
      comb_func,
      &mut transcript,
    )?;

    let (claims_batch_left, _): (Vec<G::Scalar>, Vec<G::Scalar>) = claims_batch;

    transcript.absorb(b"l", &claims_batch_left.as_slice());

    // we now combine evaluation claims at the same point rz into one
    let gamma = transcript.squeeze(b"g")?;
    let powers_of_gamma: Vec<G::Scalar> = powers(&gamma, num_claims);
    let comm_joint = u_vec_padded
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(u, g_i)| u.c * *g_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);
    let poly_joint = PolyEvalWitness::weighted_sum(&w_vec_padded, &powers_of_gamma);
    let eval_joint = claims_batch_left
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(e, g_i)| *e * *g_i)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &comm_joint,
      &poly_joint.p,
      &r_z,
      &eval_joint,
    )?;

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      eval_E,
      sc_proof_inner,
      eval_W,
      eval_arg_cc,
      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a RelaxedR1CS instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError> {
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");

    // append the commitment to R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"C", &vk.comm);
    transcript.absorb(b"U", U);

    let (num_rounds_x, num_rounds_y) = (
      (vk.num_cons as f64).log2() as usize,
      ((vk.num_vars as f64).log2() as usize + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
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
        // constant term
        let mut poly_X = vec![(0, U.u)];
        //remaining inputs
        poly_X.extend(
          (0..U.X.len())
            .map(|i| (i + 1, U.X[i]))
            .collect::<Vec<(usize, G::Scalar)>>(),
        );
        SparsePolynomial::new((vk.num_vars as f64).log2() as usize, poly_X).evaluate(&r_y[1..])
      };
      (G::Scalar::one() - r_y[0]) * self.eval_W + r_y[0] * eval_X
    };

    // verify evaluation argument to retrieve evaluations of R1CS matrices
    let (eval_A, eval_B, eval_C, mut u_vec) =
      CC::verify(&vk.comm, &(&r_x, &r_y), &self.eval_arg_cc, &mut transcript)?;

    let claim_inner_final_expected = (eval_A + r * eval_B + r * r * eval_C) * eval_Z;
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // add additional claims about W and E polynomials to the list from CC
    u_vec.push(PolyEvalInstance {
      c: U.comm_W,
      x: r_y[1..].to_vec(),
      e: self.eval_W,
    });

    u_vec.push(PolyEvalInstance {
      c: U.comm_E,
      x: r_x,
      e: self.eval_E,
    });

    let u_vec_padded = PolyEvalInstance::pad(&u_vec); // pad the evaluation points

    let powers = |s: &G::Scalar, n: usize| -> Vec<G::Scalar> {
      assert!(n >= 1);
      let mut powers = Vec::new();
      powers.push(G::Scalar::one());
      for i in 1..n {
        powers.push(powers[i - 1] * s);
      }
      powers
    };

    // generate a challenge
    let rho = transcript.squeeze(b"r")?;
    let num_claims = u_vec.len();
    let powers_of_rho = powers(&rho, num_claims);
    let claim_batch_joint = u_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(u, p)| u.e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let num_rounds_z = u_vec_padded[0].x.len();
    let (claim_batch_final, r_z) =
      self
        .sc_proof_batch
        .verify(claim_batch_joint, num_rounds_z, 2, &mut transcript)?;

    let claim_batch_final_expected = {
      let poly_rz = EqPolynomial::new(r_z.clone());
      let evals = u_vec_padded
        .iter()
        .map(|u| poly_rz.evaluate(&u.x))
        .collect::<Vec<G::Scalar>>();

      evals
        .iter()
        .zip(self.evals_batch.iter())
        .zip(powers_of_rho.iter())
        .map(|((e_i, p_i), rho_i)| *e_i * *p_i * rho_i)
        .fold(G::Scalar::zero(), |acc, item| acc + item)
    };

    if claim_batch_final != claim_batch_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(b"l", &self.evals_batch.as_slice());

    // we now combine evaluation claims at the same point rz into one
    let gamma = transcript.squeeze(b"g")?;
    let powers_of_gamma: Vec<G::Scalar> = powers(&gamma, num_claims);
    let comm_joint = u_vec_padded
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(u, g_i)| u.c * *g_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);
    let eval_joint = self
      .evals_batch
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(e, g_i)| *e * *g_i)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &comm_joint,
      &r_z,
      &eval_joint,
      &self.eval_arg,
    )?;

    Ok(())
  }
}
