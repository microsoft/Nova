//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme
use crate::{
  errors::NovaError,
  provider::{pedersen::CommitmentKeyExtTrait, traits::DlogGroup},
  spartan::{batch_invert, polys::eq::EqPolynomial},
  traits::{
    commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine,
    TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, CE,
};
use core::iter;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Provides an implementation of the prover key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine> {
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of the verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine> {
  ck_v: CommitmentKey<E>,
  ck_s: CommitmentKey<E>,
}

/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine,
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;
  type EvaluationArgument = InnerProductArgument<E>;

  fn setup(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    // IPA only uses Pedersen commitments which always succeed
    let ck_c = E::CE::setup(b"ipa", 1)?;

    let pk = ProverKey { ck_s: ck_c.clone() };
    let vk = VerifierKey {
      ck_v: ck.clone(),
      ck_s: ck_c,
    };

    Ok((pk, vk))
  }

  fn prove(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    poly: &[E::Scalar],
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    Self::prove_owned(ck, pk, transcript, comm, poly.to_vec(), point, eval)
  }

  fn prove_owned(
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    poly: Vec<E::Scalar>,
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let u = InnerProductInstance::new(comm, EqPolynomial::new(point.to_vec()).evals(), eval);
    let w = InnerProductWitness::new(poly);

    InnerProductArgument::prove(ck, &pk.ck_s, u, w, transcript)
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &Commitment<E>,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let u = InnerProductInstance::new(comm, EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.verify(
      &vk.ck_v,
      &vk.ck_s,
      (2_usize).pow(point.len() as u32),
      &u,
      transcript,
    )?;

    Ok(())
  }
}

fn inner_product<T: Field + Send + Sync>(a: &[T], b: &[T]) -> T {
  assert_eq!(a.len(), b.len());
  (0..a.len())
    .into_par_iter()
    .map(|i| a[i] * b[i])
    .reduce(|| T::ZERO, |x, y| x + y)
}

/// An inner product instance consists of a commitment to a vector `a` and another vector `b`
/// and the claim that c = <a, b>.
pub struct InnerProductInstance<E: Engine> {
  comm_a_vec: Commitment<E>,
  b_vec: Vec<E::Scalar>,
  c: E::Scalar,
}

impl<E> InnerProductInstance<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn new(comm_a_vec: &Commitment<E>, b_vec: Vec<E::Scalar>, c: &E::Scalar) -> Self {
    InnerProductInstance {
      comm_a_vec: *comm_a_vec,
      b_vec,
      c: *c,
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for InnerProductInstance<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // we do not need to include self.b_vec as in our context it is produced from the transcript
    [
      self.comm_a_vec.to_transcript_bytes(),
      self.c.to_transcript_bytes(),
    ]
    .concat()
  }
}

struct InnerProductWitness<E: Engine> {
  a_vec: Vec<E::Scalar>,
}

impl<E: Engine> InnerProductWitness<E> {
  fn new(a_vec: Vec<E::Scalar>) -> Self {
    InnerProductWitness { a_vec }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct InnerProductArgument<E: Engine> {
  L_vec: Vec<Commitment<E>>,
  R_vec: Vec<Commitment<E>>,
  a_hat: E::Scalar,
}

impl<E> InnerProductArgument<E>
where
  E: Engine,
  E::GE: DlogGroup,
  CommitmentKey<E>: CommitmentKeyExtTrait<E>,
{
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  fn prove(
    ck: &CommitmentKey<E>,
    ck_c: &CommitmentKey<E>,
    U: InnerProductInstance<E>,
    W: InnerProductWitness<E>,
    transcript: &mut E::TE,
  ) -> Result<Self, NovaError> {
    transcript.dom_sep(Self::protocol_name());

    let (ck, _) = ck.split_at(U.b_vec.len());

    if U.b_vec.len() != W.a_vec.len() {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", &U);

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c.scale(&r);

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[E::Scalar],
                       b_vec: &[E::Scalar],
                       ck: &CommitmentKey<E>,
                       transcript: &mut E::TE|
     -> Result<
      (
        Commitment<E>,
        Commitment<E>,
        Vec<E::Scalar>,
        Vec<E::Scalar>,
        CommitmentKey<E>,
      ),
      NovaError,
    > {
      let n = a_vec.len();
      let (ck_L, ck_R) = ck.split_at(n / 2);

      let c_L = inner_product(&a_vec[0..n / 2], &b_vec[n / 2..n]);
      let c_R = inner_product(&a_vec[n / 2..n], &b_vec[0..n / 2]);

      let L = CE::<E>::commit(&ck_R, &a_vec[0..n / 2], &E::Scalar::ZERO)
        + CE::<E>::commit(&ck_c, &[c_L], &E::Scalar::ZERO);
      let R = CE::<E>::commit(&ck_L, &a_vec[n / 2..n], &E::Scalar::ZERO)
        + CE::<E>::commit(&ck_c, &[c_R], &E::Scalar::ZERO);

      transcript.absorb(b"L", &L);
      transcript.absorb(b"R", &R);

      let r = transcript.squeeze(b"r")?;
      let r_inverse = r.invert().unwrap();

      // fold the left half and the right half
      let a_vec_folded = a_vec[0..n / 2]
        .par_iter()
        .zip(a_vec[n / 2..n].par_iter())
        .map(|(a_L, a_R)| *a_L * r + r_inverse * *a_R)
        .collect::<Vec<E::Scalar>>();

      let b_vec_folded = b_vec[0..n / 2]
        .par_iter()
        .zip(b_vec[n / 2..n].par_iter())
        .map(|(b_L, b_R)| *b_L * r_inverse + r * *b_R)
        .collect::<Vec<E::Scalar>>();

      let ck_folded = ck.fold(&r_inverse, &r);

      Ok((L, R, a_vec_folded, b_vec_folded, ck_folded))
    };

    // two vectors to hold the logarithmic number of group elements
    let mut L_vec: Vec<Commitment<E>> = Vec::new();
    let mut R_vec: Vec<Commitment<E>> = Vec::new();

    // Consume the prover-only vectors after their instance is absorbed.
    let mut a_vec = W.a_vec;
    let mut b_vec = U.b_vec;
    let mut ck = ck;
    for _i in 0..usize::try_from(b_vec.len().ilog2()).unwrap() {
      let (L, R, a_vec_folded, b_vec_folded, ck_folded) =
        prove_inner(&a_vec, &b_vec, &ck, transcript)?;
      L_vec.push(L);
      R_vec.push(R);

      a_vec = a_vec_folded;
      b_vec = b_vec_folded;
      ck = ck_folded;
    }

    Ok(InnerProductArgument {
      L_vec,
      R_vec,
      a_hat: a_vec[0],
    })
  }

  fn verify(
    &self,
    ck: &CommitmentKey<E>,
    ck_c: &CommitmentKey<E>,
    n: usize,
    U: &InnerProductInstance<E>,
    transcript: &mut E::TE,
  ) -> Result<(), NovaError> {
    let (ck, _) = ck.split_at(U.b_vec.len());

    transcript.dom_sep(Self::protocol_name());
    if U.b_vec.len() != n
      || n != (1 << self.L_vec.len())
      || self.L_vec.len() != self.R_vec.len()
      || self.L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for committing to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c.scale(&r);

    let P = U.comm_a_vec + CE::<E>::commit(&ck_c, &[U.c], &E::Scalar::ZERO);

    // compute a vector of public coins using self.L_vec and self.R_vec
    let r = (0..self.L_vec.len())
      .map(|i| {
        transcript.absorb(b"L", &self.L_vec[i]);
        transcript.absorb(b"R", &self.R_vec[i]);
        transcript.squeeze(b"r")
      })
      .collect::<Result<Vec<E::Scalar>, NovaError>>()?;

    // precompute scalars necessary for verification
    let r_square: Vec<E::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r[i] * r[i])
      .collect();
    let r_inverse = batch_invert(&r)?;
    let r_inverse_square: Vec<E::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r_inverse[i] * r_inverse[i])
      .collect();

    // compute the vector with the tensor structure
    let s = {
      let mut s = vec![E::Scalar::ZERO; n];
      s[0] = {
        let mut v = E::Scalar::ONE;
        for r_inverse_i in r_inverse {
          v *= r_inverse_i;
        }
        v
      };
      for i in 1..n {
        let pos_in_r = (31 - (i as u32).leading_zeros()) as usize;
        s[i] = s[i - (1 << pos_in_r)] * r_square[(self.L_vec.len() - 1) - pos_in_r];
      }
      s
    };

    let ck_hat = {
      let c = CE::<E>::commit(&ck, &s, &E::Scalar::ZERO);
      CommitmentKey::<E>::reinterpret_commitments_as_ck(&[c])?
    };

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let ck_folded = {
        let ck_L = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.L_vec)?;
        let ck_R = CommitmentKey::<E>::reinterpret_commitments_as_ck(&self.R_vec)?;
        let ck_P = CommitmentKey::<E>::reinterpret_commitments_as_ck(&[P])?;
        ck_L.combine(&ck_R).combine(&ck_P)
      };

      CE::<E>::commit(
        &ck_folded,
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&E::Scalar::ONE))
          .copied()
          .collect::<Vec<E::Scalar>>(),
        &E::Scalar::ZERO,
      )
    };

    if P_hat
      == CE::<E>::commit(
        &ck_hat.combine(&ck_c),
        &[self.a_hat, self.a_hat * b_hat],
        &E::Scalar::ZERO,
      )
    {
      Ok(())
    } else {
      Err(NovaError::InvalidPCS)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{keccak::Keccak256Transcript, PallasEngine},
    spartan::polys::multilinear::MultilinearPolynomial,
    traits::TranscriptEngineTrait,
  };
  use sha2::{Digest, Sha256};

  type TestEngine = PallasEngine;
  type TestScalar = <TestEngine as Engine>::Scalar;

  #[test]
  fn owned_prover_vectors_preserve_proof_bytes() {
    let ck = <TestEngine as Engine>::CE::setup(b"ipa_owned_vectors", 8).unwrap();
    let (pk, vk) = EvaluationEngine::<TestEngine>::setup(&ck).unwrap();
    let poly = (1u64..=8).map(TestScalar::from).collect::<Vec<_>>();
    let point = vec![
      TestScalar::from(2),
      TestScalar::from(3),
      TestScalar::from(5),
    ];
    let eval = MultilinearPolynomial::evaluate_with(&poly, &point);
    let comm = <TestEngine as Engine>::CE::commit(&ck, &poly, &TestScalar::from(0));

    let mut prover_transcript = Keccak256Transcript::new(b"IpaOwnedVectors");
    let proof = EvaluationEngine::<TestEngine>::prove(
      &ck,
      &pk,
      &mut prover_transcript,
      &comm,
      &poly,
      &point,
      &eval,
    )
    .unwrap();
    let mut owned_transcript = Keccak256Transcript::new(b"IpaOwnedVectors");
    let owned_proof = EvaluationEngine::<TestEngine>::prove_owned(
      &ck,
      &pk,
      &mut owned_transcript,
      &comm,
      poly.clone(),
      &point,
      &eval,
    )
    .unwrap();
    let mut verifier_transcript = Keccak256Transcript::new(b"IpaOwnedVectors");
    EvaluationEngine::<TestEngine>::verify(
      &vk,
      &mut verifier_transcript,
      &comm,
      &point,
      &eval,
      &proof,
    )
    .unwrap();

    let config = bincode::config::legacy()
      .with_little_endian()
      .with_fixed_int_encoding();
    let encoded = bincode::serde::encode_to_vec(proof, config).unwrap();
    let owned_encoded = bincode::serde::encode_to_vec(owned_proof, config).unwrap();
    assert_eq!(owned_encoded, encoded);
    assert_eq!(encoded.len(), 240);
    // Frozen before prover vectors were made move-only at this boundary.
    let digest: [u8; 32] = Sha256::digest(&encoded).into();
    assert_eq!(
      digest,
      [
        0x8a, 0xd2, 0x8a, 0x26, 0x2b, 0xbd, 0x8e, 0xd2, 0xec, 0x58, 0x82, 0x71, 0xb6, 0x05, 0x96,
        0x44, 0x0a, 0x3d, 0x4e, 0x4c, 0xab, 0x39, 0xba, 0x35, 0xf1, 0xc8, 0x5d, 0xa5, 0x75, 0x24,
        0x94, 0xa8,
      ]
    );
  }
}
