//! This module implements Nova's evaluation engine using `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in <https://eprint.iacr.org/2022/420.pdf>).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).
#![allow(non_snake_case)]
use crate::{
  errors::NovaError,
  provider::{
    kzg_commitment::{KZGCommitmentEngine, KZGProverKey, KZGVerifierKey, UniversalKZGParam},
    pedersen::Commitment,
    traits::DlogGroup,
  },
  spartan::polys::univariate::UniPoly,
  traits::{
    commitment::{CommitmentEngineTrait, Len},
    evaluation::EvaluationEngineTrait,
    Engine as NovaEngine, Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  zip_with,
};
use core::marker::PhantomData;
use ff::Field;
use group::{Curve, Group as _};
use halo2curves::pairing::{Engine, MillerLoopResult, MultiMillerLoop};
use itertools::Itertools;
use rayon::prelude::*;
use ref_cast::RefCast;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::G1Affine: Serialize, E::Fr: Serialize",
  deserialize = "E::G1Affine: Deserialize<'de>, E::Fr: Deserialize<'de>"
))]
pub struct EvaluationArgument<E: Engine> {
  comms: Vec<E::G1Affine>,
  w: Vec<E::G1Affine>,
  evals: Vec<Vec<E::Fr>>,
}

/// Provides an implementation of a polynomial evaluation engine using KZG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E, NE> {
  _p: PhantomData<(E, NE)>,
}

// This impl block defines helper functions that are not a part of
// EvaluationEngineTrait, but that we will use to implement the trait methods.
impl<E, NE> EvaluationEngine<E, NE>
where
  E: Engine,
  NE: NovaEngine<GE = E::G1, Scalar = E::Fr>,
  E::G1: DlogGroup<AffineGroupElement = E::G1Affine, Scalar = E::Fr>,
  E::Fr: TranscriptReprTrait<E::G1>,
  E::G1Affine: TranscriptReprTrait<E::G1>,
{
  fn compute_challenge(
    com: &[E::G1Affine],
    transcript: &mut impl TranscriptEngineTrait<NE>,
  ) -> E::Fr {
    transcript.absorb(b"c", &com.to_vec().as_slice());
    transcript.squeeze(b"c").unwrap()
  }

  // Compute challenge q = Hash(vk, C0, ..., C_{k-1}, u0, ...., u_{t-1},
  // (f_i(u_j))_{i=0..k-1,j=0..t-1})
  // It is assumed that both 'C' and 'u' are already absorbed by the transcript
  fn get_batch_challenge(
    v: &[Vec<E::Fr>],
    transcript: &mut impl TranscriptEngineTrait<NE>,
  ) -> E::Fr {
    transcript.absorb(
      b"v",
      &v.iter()
        .flatten()
        .cloned()
        .collect::<Vec<E::Fr>>()
        .as_slice(),
    );

    transcript.squeeze(b"r").unwrap()
  }

  fn batch_challenge_powers(q: E::Fr, k: usize) -> Vec<E::Fr> {
    // Compute powers of q : (1, q, q^2, ..., q^(k-1))
    std::iter::successors(Some(E::Fr::ONE), |&x| Some(x * q))
      .take(k)
      .collect()
  }

  fn verifier_second_challenge(
    W: &[E::G1Affine],
    transcript: &mut impl TranscriptEngineTrait<NE>,
  ) -> E::Fr {
    transcript.absorb(b"W", &W.to_vec().as_slice());

    transcript.squeeze(b"d").unwrap()
  }
}

impl<E, NE> EvaluationEngineTrait<NE> for EvaluationEngine<E, NE>
where
  E: MultiMillerLoop,
  NE: NovaEngine<GE = E::G1, Scalar = E::Fr, CE = KZGCommitmentEngine<E>>,
  E::Fr: Serialize + DeserializeOwned,
  E::G1Affine: Serialize + DeserializeOwned,
  E::G2Affine: Serialize + DeserializeOwned,
  E::G1: DlogGroup<AffineGroupElement = E::G1Affine, Scalar = E::Fr>,
  <E::G1 as Group>::Base: TranscriptReprTrait<E::G1>, // Note: due to the move of the bound TranscriptReprTrait<G> on G::Base from Group to Engine
  E::Fr: TranscriptReprTrait<E::G1>,
  E::G1Affine: TranscriptReprTrait<E::G1>,
{
  type EvaluationArgument = EvaluationArgument<E>;
  type ProverKey = KZGProverKey<E>;
  type VerifierKey = KZGVerifierKey<E>;

  fn setup(ck: &UniversalKZGParam<E>) -> (Self::ProverKey, Self::VerifierKey) {
    ck.trim(ck.length() - 1)
  }

  fn prove(
    ck: &UniversalKZGParam<E>,
    _pk: &Self::ProverKey,
    transcript: &mut <NE as NovaEngine>::TE,
    _C: &Commitment<NE>,
    hat_P: &[E::Fr],
    point: &[E::Fr],
    _eval: &E::Fr,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let x: Vec<E::Fr> = point.to_vec();

    //////////////// begin helper closures //////////
    let kzg_open = |f: &[E::Fr], u: E::Fr| -> E::G1Affine {
      // On input f(x) and u compute the witness polynomial used to prove
      // that f(u) = v. The main part of this is to compute the
      // division (f(x) - f(u)) / (x - u), but we don't use a general
      // division algorithm, we make use of the fact that the division
      // never has a remainder, and that the denominator is always a linear
      // polynomial. The cost is (d-1) mults + (d-1) adds in E::Scalar, where
      // d is the degree of f.
      //
      // We use the fact that if we compute the quotient of f(x)/(x-u),
      // there will be a remainder, but it'll be v = f(u).  Put another way
      // the quotient of f(x)/(x-u) and (f(x) - f(v))/(x-u) is the
      // same.  One advantage is that computing f(u) could be decoupled
      // from kzg_open, it could be done later or separate from computing W.

      let compute_witness_polynomial = |f: &[E::Fr], u: E::Fr| -> Vec<E::Fr> {
        let d = f.len();

        // Compute h(x) = f(x)/(x - u)
        let mut h = vec![E::Fr::ZERO; d];
        for i in (1..d).rev() {
          h[i - 1] = f[i] + h[i] * u;
        }

        h
      };

      let h = compute_witness_polynomial(f, u);

      <NE::CE as CommitmentEngineTrait<NE>>::commit(ck, &h)
        .comm
        .affine()
    };

    let kzg_open_batch = |f: &[Vec<E::Fr>],
                          u: &[E::Fr],
                          transcript: &mut <NE as NovaEngine>::TE|
     -> (Vec<E::G1Affine>, Vec<Vec<E::Fr>>) {
      let scalar_vector_muladd = |a: &mut Vec<E::Fr>, v: &Vec<E::Fr>, s: E::Fr| {
        assert!(a.len() >= v.len());
        a.par_iter_mut()
          .zip(v.par_iter())
          .for_each(|(c, v)| *c += s * v);
      };

      let kzg_compute_batch_polynomial = |f: &[Vec<E::Fr>], q: E::Fr| -> Vec<E::Fr> {
        let k = f.len(); // Number of polynomials we're batching

        let q_powers = Self::batch_challenge_powers(q, k);

        // Compute B(x) = f[0] + q*f[1] + q^2 * f[2] + ... q^(k-1) * f[k-1]
        let mut B = f[0].clone();
        for i in 1..k {
          scalar_vector_muladd(&mut B, &f[i], q_powers[i]); // B += q_powers[i] * f[i]
        }

        B
      };
      ///////// END kzg_open_batch closure helpers

      let k = f.len();
      let t = u.len();

      // The verifier needs f_i(u_j), so we compute them here
      // (V will compute B(u_j) itself)
      let mut v = vec![vec!(E::Fr::ZERO; k); t];
      v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
        // for each point u
        v_i.par_iter_mut().zip_eq(f).for_each(|(v_ij, f)| {
          // for each poly f (except the last one - since it is constant)
          *v_ij = UniPoly::ref_cast(f).evaluate(&u[i]);
        });
      });

      let q = Self::get_batch_challenge(&v, transcript);
      let B = kzg_compute_batch_polynomial(f, q);

      // Now open B at u0, ..., u_{t-1}
      let w = u.par_iter().map(|ui| kzg_open(&B, *ui)).collect::<Vec<_>>();

      // The prover computes the challenge to keep the transcript in the same
      // state as that of the verifier
      let _d_0 = Self::verifier_second_challenge(&w, transcript);
      (w, v)
    };

    ///// END helper closures //////////

    let ell = x.len();
    let n = hat_P.len();
    assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

    // Phase 1  -- create commitments com_1, ..., com_\ell
    let mut polys: Vec<Vec<E::Fr>> = Vec::new();
    polys.push(hat_P.to_vec());

    // We don't compute final Pi (and its commitment) as it is constant and equals to 'eval'
    // also known to verifier, so can be derived on its side as well
    for i in 0..x.len() - 1 {
      let Pi_len = polys[i].len() / 2;
      let mut Pi = vec![E::Fr::ZERO; Pi_len];

      #[allow(clippy::needless_range_loop)]
      Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
        *Pi_j = x[ell - i - 1] * (polys[i][2 * j + 1] - polys[i][2 * j]) + polys[i][2 * j];
      });

      polys.push(Pi);
    }

    // We do not need to commit to the first polynomial as it is already committed.
    // Compute commitments in parallel
    let comms: Vec<E::G1Affine> = (1..polys.len())
      .into_par_iter()
      .map(|i| {
        <NE::CE as CommitmentEngineTrait<NE>>::commit(ck, &polys[i])
          .comm
          .affine()
      })
      .collect();

    // Phase 2
    // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
    // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
    let r = Self::compute_challenge(&comms, transcript);
    let u = vec![r, -r, r * r];

    // Phase 3 -- create response
    let (w, evals) = kzg_open_batch(&polys, &u, transcript);

    Ok(EvaluationArgument { comms, w, evals })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut <NE as NovaEngine>::TE,
    C: &Commitment<NE>,
    point: &[E::Fr],
    P_of_x: &E::Fr,
    pi: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let x = point.to_vec();
    let y = P_of_x;

    // vk is hashed in transcript already, so we do not add it here

    let kzg_verify_batch = |vk: &KZGVerifierKey<E>,
                            C: &Vec<E::G1Affine>,
                            W: &Vec<E::G1Affine>,
                            u: &Vec<E::Fr>,
                            v: &Vec<Vec<E::Fr>>,
                            transcript: &mut <NE as NovaEngine>::TE|
     -> bool {
      let k = C.len();
      let t = u.len();

      let q = Self::get_batch_challenge(v, transcript);
      let q_powers = Self::batch_challenge_powers(q, k); // 1, q, q^2, ..., q^(k-1)

      let d_0 = Self::verifier_second_challenge(W, transcript);
      let d_1 = d_0 * d_0;

      assert!(t == 3);
      assert!(W.len() == 3);
      // We write a special case for t=3, since this what is required for
      // hyperkzg. Following the paper directly, we must compute:
      // let L0 = C_B - vk.G * B_u[0] + W[0] * u[0];
      // let L1 = C_B - vk.G * B_u[1] + W[1] * u[1];
      // let L2 = C_B - vk.G * B_u[2] + W[2] * u[2];
      // let R0 = -W[0];
      // let R1 = -W[1];
      // let R2 = -W[2];
      // let L = L0 + L1*d_0 + L2*d_1;
      // let R = R0 + R1*d_0 + R2*d_1;
      //
      // We group terms to reduce the number of scalar mults (to seven):
      // In Rust, we could use MSMs for these, and speed up verification.
      //
      // Note, that while computing L, the intermediate computation of C_B together with computing
      // L0, L1, L2 can be replaced by single MSM of C with the powers of q multiplied by (1 + d_0 + d_1)
      // with additionally concatenated inputs for scalars/bases.

      let q_power_multiplier = E::Fr::ONE + d_0 + d_1;

      let q_powers_multiplied: Vec<E::Fr> = q_powers
        .par_iter()
        .map(|q_power| *q_power * q_power_multiplier)
        .collect();

      // Compute the batched openings
      // compute B(u_i) = v[i][0] + q*v[i][1] + ... + q^(t-1) * v[i][t-1]
      let B_u = v
        .into_par_iter()
        .map(|v_i| zip_with!(iter, (q_powers, v_i), |a, b| *a * *b).sum())
        .collect::<Vec<E::Fr>>();

      let L = NE::GE::vartime_multiscalar_mul(
        &[
          &q_powers_multiplied[..k],
          &[
            u[0],
            (u[1] * d_0),
            (u[2] * d_1),
            (B_u[0] + d_0 * B_u[1] + d_1 * B_u[2]),
          ],
        ]
        .concat(),
        &[
          &C[..k],
          &[
            E::G1::from(W[0]).into(),
            E::G1::from(W[1]).into(),
            E::G1::from(W[2]).into(),
            (-E::G1::from(vk.g)).into(),
          ],
        ]
        .concat(),
      );

      let R0 = E::G1::from(W[0]);
      let R1 = E::G1::from(W[1]);
      let R2 = E::G1::from(W[2]);
      let R = R0 + R1 * d_0 + R2 * d_1;

      // Check that e(L, vk.H) == e(R, vk.tau_H)
      let pairing_inputs = [
        (&(-L).to_affine(), &E::G2Prepared::from(vk.h)),
        (&R.to_affine(), &E::G2Prepared::from(vk.beta_h)),
      ];

      let pairing_result = E::multi_miller_loop(&pairing_inputs).final_exponentiation();
      pairing_result.is_identity().into()
    };
    ////// END verify() closure helpers

    let ell = x.len();

    let mut com = pi.comms.clone();

    // we do not need to add x to the transcript, because in our context x was
    // obtained from the transcript
    let r = Self::compute_challenge(&com, transcript);
    if r == E::Fr::ZERO || C.comm == E::G1::identity() {
      return Err(NovaError::ProofVerifyError);
    }
    com.insert(0, C.comm.affine()); // set com_0 = C, shifts other commitments to the right

    let u = vec![r, -r, r * r];

    // Setup vectors (Y, ypos, yneg) from pi.v
    let v = &pi.evals;
    if v.len() != 3 {
      return Err(NovaError::ProofVerifyError);
    }
    if v[0].len() != ell || v[1].len() != ell || v[2].len() != ell {
      return Err(NovaError::ProofVerifyError);
    }
    let ypos = &v[0];
    let yneg = &v[1];
    let mut Y = v[2].to_vec();
    Y.push(*y);

    // Check consistency of (Y, ypos, yneg)
    let two = E::Fr::from(2u64);
    for i in 0..ell {
      if two * r * Y[i + 1]
        != r * (E::Fr::ONE - x[ell - i - 1]) * (ypos[i] + yneg[i])
          + x[ell - i - 1] * (ypos[i] - yneg[i])
      {
        return Err(NovaError::ProofVerifyError);
      }
      // Note that we don't make any checks about Y[0] here, but our batching
      // check below requires it
    }

    // Check commitments to (Y, ypos, yneg) are valid
    if !kzg_verify_batch(vk, &com, &pi.w, &u, &pi.evals, transcript) {
      return Err(NovaError::ProofVerifyError);
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::test_utils::prove_verify_from_num_vars;
  use crate::{
    provider::keccak::Keccak256Transcript, CommitmentKey,
  };
  use bincode::Options;

  type E = halo2curves::bn256::Bn256;
  type NE = crate::provider::Bn256EngineKZG;
  type Fr = <NE as NovaEngine>::Scalar;

  #[test]
  fn test_hyperkzg_eval() {
    // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
    let n = 4;
    let ck = <KZGCommitmentEngine<Bn256> as CommitmentEngineTrait<NE>>::setup(b"test", n);
    let (pk, vk) = EvaluationEngine::<Bn256, NE>::setup(&ck);

    // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
    let poly = vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)];

    let C = <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::commit(&ck, &poly);

    let test_inner = |point: Vec<Fr>, eval: Fr| -> Result<(), NovaError> {
      let mut tr = Keccak256Transcript::<NE>::new(b"TestEval");
      let proof =
        EvaluationEngine::<Bn256, NE>::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).unwrap();
      let mut tr = Keccak256Transcript::<NE>::new(b"TestEval");
      EvaluationEngine::<Bn256, NE>::verify(&vk, &mut tr, &C, &point, &eval, &proof)
    };

    // Call the prover with a (point, eval) pair.
    // The prover does not recompute so it may produce a proof, but it should not verify
    let point = vec![Fr::from(0), Fr::from(0)];
    let eval = Fr::ONE;
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(1)];
    let eval = Fr::from(2);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(1), Fr::from(1)];
    let eval = Fr::from(4);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(3);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(9);
    assert!(test_inner(point, eval).is_ok());

    // Try a couple incorrect evaluations and expect failure
    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(50);
    assert!(test_inner(point, eval).is_err());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(4);
    assert!(test_inner(point, eval).is_err());
  }

  #[test]
  fn test_hyperkzg_small() {
    let n = 4;

    // poly = [1, 2, 1, 4]
    let poly = vec![Fr::ONE, Fr::from(2), Fr::from(1), Fr::from(4)];

    // point = [4,3]
    let point = vec![Fr::from(4), Fr::from(3)];

    // eval = 28
    let eval = Fr::from(28);

    let ck: CommitmentKey<NE> =
      <KZGCommitmentEngine<E> as CommitmentEngineTrait<NE>>::setup(b"test", n);
    let (pk, vk): (KZGProverKey<E>, KZGVerifierKey<E>) = EvaluationEngine::<E, NE>::setup(&ck);

    // make a commitment
    let C = KZGCommitmentEngine::commit(&ck, &poly);

    // prove an evaluation
    let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
    let proof =
      EvaluationEngine::<E, NE>::prove(&ck, &pk, &mut prover_transcript, &C, &poly, &point, &eval)
        .unwrap();
    let post_c_p = prover_transcript.squeeze(b"c").unwrap();

    // verify the evaluation
    let mut verifier_transcript = Keccak256Transcript::<NE>::new(b"TestEval");
    assert!(EvaluationEngine::<E, NE>::verify(
      &vk,
      &mut verifier_transcript,
      &C,
      &point,
      &eval,
      &proof
    )
    .is_ok());
    let post_c_v = verifier_transcript.squeeze(b"c").unwrap();

    // check if the prover transcript and verifier transcript are kept in the same state
    assert_eq!(post_c_p, post_c_v);

    let proof_bytes = bincode::DefaultOptions::new()
      .with_big_endian()
      .with_fixint_encoding()
      .serialize(&proof)
      .unwrap();
    assert_eq!(proof_bytes.len(), 368);

    // Change the proof and expect verification to fail
    let mut bad_proof = proof.clone();
    bad_proof.comms[0] = (bad_proof.comms[0] + bad_proof.comms[0] * Fr::from(123)).to_affine();
    let mut verifier_transcript2 = Keccak256Transcript::<NE>::new(b"TestEval");
    assert!(EvaluationEngine::<E, NE>::verify(
      &vk,
      &mut verifier_transcript2,
      &C,
      &point,
      &eval,
      &bad_proof
    )
    .is_err());
  }

  #[test]
  fn test_hyperkzg_large() {
    // test the mlkzg prover and verifier with random instances (derived from a seed)
    for num_vars in [4, 5, 6] {
      prove_verify_from_num_vars::<_, EvaluationEngine<E, NE>>(num_vars);
    }
  }
}
