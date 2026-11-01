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
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{de::Deserializer, ser::Serializer, Deserialize, Serialize};
use std::{ops::Deref, sync::Arc};

struct SharedR1CSShape<E: Engine>(Arc<R1CSShape<E>>);

impl<E: Engine> SharedR1CSShape<E> {
  fn new(shape: R1CSShape<E>) -> Self {
    Self(Arc::new(shape))
  }
}

impl<E: Engine> Clone for SharedR1CSShape<E> {
  fn clone(&self) -> Self {
    Self(Arc::clone(&self.0))
  }
}

impl<E: Engine> Deref for SharedR1CSShape<E> {
  type Target = R1CSShape<E>;

  fn deref(&self) -> &Self::Target {
    self.0.as_ref()
  }
}

// Delegate directly to R1CSShape so VerifierKey's wire representation and
// digest remain byte-for-byte unchanged.
impl<E: Engine> Serialize for SharedR1CSShape<E> {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    self.0.as_ref().serialize(serializer)
  }
}

impl<'de, E: Engine> Deserialize<'de> for SharedR1CSShape<E> {
  fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    R1CSShape::<E>::deserialize(deserializer).map(Self::new)
  }
}

struct CachedPaddedShape<E: Engine> {
  source_digest: E::Scalar,
  shape: SharedR1CSShape<E>,
}

/// A type that represents the prover's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  vk_digest: E::Scalar, // digest of the verifier's key
  #[serde(skip)]
  padded_shape: Option<CachedPaddedShape<E>>,
}

/// A type that represents the verifier's key
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  vk_ee: EE::VerifierKey,
  S: SharedR1CSShape<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> {}

impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE> {
  fn new(shape: SharedR1CSShape<E>, vk_ee: EE::VerifierKey) -> Self {
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
    let (pk_ee, vk_ee) = EE::setup(ck)?;

    let source_digest = S.digest();
    let S = SharedR1CSShape::new(S.pad());

    let vk: VerifierKey<E, EE> = VerifierKey::new(S.clone(), vk_ee);

    let pk = ProverKey {
      pk_ee,
      vk_digest: vk.digest(),
      padded_shape: Some(CachedPaddedShape {
        source_digest,
        shape: S,
      }),
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
    // Setup-created keys share one padded shape with the verifier key. A
    // deserialized key deliberately does not cache the first caller-supplied
    // shape: its legacy bytes contain no expected source-shape digest, so
    // caching here would let a mismatched first call poison subsequent use.
    let fallback_shape;
    let S = if let Some(cached) = &pk.padded_shape {
      if cached.source_digest != S.digest() {
        return Err(NovaError::DigestError);
      }
      cached.shape.deref()
    } else {
      fallback_shape = S.pad();
      &fallback_shape
    };
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());

    let W = W.pad(S); // pad the witness
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
      .collect::<Result<Vec<_>, NovaError>>()?;

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

    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_three_inputs(
      &E::Scalar::ZERO, // claim is zero
      tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_uCz_E,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    let (claim_Az, claim_Bz): (E::Scalar, E::Scalar) = (claims_outer[0], claims_outer[1]);
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

      let (evals_A, evals_B, evals_C) = compute_eval_table_sparse(S, &evals_rx);

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

    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad_prod(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(poly_ABC),
      &mut MultilinearPolynomial::new(poly_z),
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

    transcript.absorb(b"w", &[eval_W].as_slice());

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

    let (batched_u, batched_w, _chal, sc_proof_batch, claims_batch_left) =
      super::batch_eval_reduce(u_vec, w_vec, &mut transcript)?;

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

    transcript.absorb(b"w", &[self.eval_W].as_slice());

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

    let (batched_u, _chal) = super::batch_eval_verify(
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    digest::DigestComputer,
    provider::{ipa_pc, PallasEngine},
    r1cs::SparseMatrix,
    traits::{commitment::CommitmentEngineTrait, snark::default_ck_hint},
  };
  use ff::Field;

  type TestEngine = PallasEngine;
  type TestEvaluationEngine = ipa_pc::EvaluationEngine<TestEngine>;
  type TestSnark = RelaxedR1CSSNARK<TestEngine, TestEvaluationEngine>;
  type TestProverKey = ProverKey<TestEngine, TestEvaluationEngine>;
  type TestVerifierKey = VerifierKey<TestEngine, TestEvaluationEngine>;

  #[derive(Serialize)]
  struct LegacyProverKey<'a> {
    pk_ee: &'a <TestEvaluationEngine as EvaluationEngineTrait<TestEngine>>::ProverKey,
    vk_digest: &'a <TestEngine as Engine>::Scalar,
  }

  #[allow(non_snake_case)]
  #[derive(Serialize)]
  struct LegacyVerifierKey<'a> {
    vk_ee: &'a <TestEvaluationEngine as EvaluationEngineTrait<TestEngine>>::VerifierKey,
    S: &'a R1CSShape<TestEngine>,
  }

  impl SimpleDigestible for LegacyVerifierKey<'_> {}

  fn test_shape(changed: bool) -> R1CSShape<TestEngine> {
    // Spartan's sparse public-input polynomial needs at least one variable
    // round beyond the padded (u, X) vector.
    let num_cons = 8;
    let num_vars = 3;
    let num_io = 2;
    let cols = num_vars + num_io + 1;
    let one = <TestEngine as Engine>::Scalar::ONE;
    let five = one + one + one + one + one;
    let first = if changed { one + one } else { one };
    let A = SparseMatrix::new(
      &[
        (0, num_vars + 1, first),
        (1, 0, one),
        (2, 1, one),
        (2, num_vars + 1, one),
        (3, 2, one),
        (3, num_vars, five),
      ],
      num_cons,
      cols,
    );
    let B = SparseMatrix::new(
      &[
        (0, num_vars + 1, one),
        (1, num_vars + 1, one),
        (2, num_vars, one),
        (3, num_vars, one),
      ],
      num_cons,
      cols,
    );
    let C = SparseMatrix::new(
      &[
        (0, 0, one),
        (1, 1, one),
        (2, 2, one),
        (3, num_vars + 2, one),
      ],
      num_cons,
      cols,
    );
    R1CSShape::new(num_cons, num_vars, num_io, A, B, C).expect("test R1CS shape")
  }

  fn encode<T: Serialize>(value: &T) -> Vec<u8> {
    bincode::serde::encode_to_vec(
      value,
      bincode::config::legacy()
        .with_little_endian()
        .with_fixed_int_encoding(),
    )
    .expect("test encoding")
  }

  #[test]
  fn setup_shares_padding_and_preserves_key_bytes_and_digest() {
    let shape = test_shape(false);
    let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).expect("commitment key");
    let (pk, vk) = TestSnark::setup(&ck, &shape).expect("Spartan setup");
    let cached = pk.padded_shape.as_ref().expect("setup padding");
    assert!(Arc::ptr_eq(&cached.shape.0, &vk.S.0));

    let legacy_pk = LegacyProverKey {
      pk_ee: &pk.pk_ee,
      vk_digest: &pk.vk_digest,
    };
    let legacy_vk = LegacyVerifierKey {
      vk_ee: &vk.vk_ee,
      S: vk.S.deref(),
    };
    assert_eq!(encode(&pk), encode(&legacy_pk));
    assert_eq!(encode(&vk), encode(&legacy_vk));
    assert_eq!(
      vk.digest(),
      DigestComputer::<<TestEngine as Engine>::Scalar, _>::new(&legacy_vk)
        .digest()
        .expect("legacy verifier-key digest")
    );
  }

  #[test]
  fn deserialized_prover_key_preserves_legacy_proving_behavior() {
    let shape = test_shape(false);
    let ck = R1CSShape::commitment_key(&[&shape], &[&*default_ck_hint()]).expect("commitment key");
    let (pk, vk) = TestSnark::setup(&ck, &shape).expect("Spartan setup");
    let config = bincode::config::legacy()
      .with_little_endian()
      .with_fixed_int_encoding();
    let (decoded_pk, used): (TestProverKey, usize) =
      bincode::serde::decode_from_slice(&encode(&pk), config).expect("decode prover key");
    assert_eq!(used, encode(&pk).len());
    assert!(decoded_pk.padded_shape.is_none());
    let (decoded_vk, used): (TestVerifierKey, usize) =
      bincode::serde::decode_from_slice(&encode(&vk), config).expect("decode verifier key");
    assert_eq!(used, encode(&vk).len());

    let (random_U, random_W) = shape
      .sample_random_instance_witness(&ck)
      .expect("random relaxed instance");
    let (W, r_W, r_E) = random_W.derandomize();
    let U = random_U.derandomize(&<TestEngine as Engine>::CE::derand_key(&ck), &r_W, &r_E);
    let proof = TestSnark::prove(&ck, &pk, &shape, &U, &W).expect("in-memory proof");
    proof.verify(&vk, &U).expect("in-memory proof verification");
    let proof = TestSnark::prove(&ck, &decoded_pk, &shape, &U, &W).expect("proof");
    proof.verify(&decoded_vk, &U).expect("verification");
    assert!(decoded_pk.padded_shape.is_none());

    assert!(matches!(
      TestSnark::prove(&ck, &pk, &test_shape(true), &U, &W),
      Err(NovaError::DigestError)
    ));
  }
}
