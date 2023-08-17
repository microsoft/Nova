//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::NovaError,
  provider::pedersen::CommitmentKeyExtTrait,
  spartan::polys::eq::EqPolynomial,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, CompressedCommitment, CE,
};
use core::iter;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Provides an implementation of the prover key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G: Group> {
  ck_s: CommitmentKey<G>,
}

/// Provides an implementation of the verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G: Group> {
  ck_v: CommitmentKey<G>,
  ck_s: CommitmentKey<G>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EvaluationArgument<G: Group> {
  ipa: InnerProductArgument<G>,
}

/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G> EvaluationEngineTrait<G> for EvaluationEngine<G>
where
  G: Group,
  CommitmentKey<G>: CommitmentKeyExtTrait<G>,
{
  type ProverKey = ProverKey<G>;
  type VerifierKey = VerifierKey<G>;
  type EvaluationArgument = EvaluationArgument<G>;

  fn setup(
    ck: &<<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let pk = ProverKey {
      ck_s: G::CE::setup(b"ipa", 1),
    };

    let vk = VerifierKey {
      ck_v: ck.clone(),
      ck_s: G::CE::setup(b"ipa", 1),
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<G>,
    pk: &Self::ProverKey,
    transcript: &mut G::TE,
    comm: &Commitment<G>,
    poly: &[G::Scalar],
    point: &[G::Scalar],
    eval: &G::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);
    let w = InnerProductWitness::new(poly);

    Ok(EvaluationArgument {
      ipa: InnerProductArgument::prove(ck, &pk.ck_s, &u, &w, transcript)?,
    })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut G::TE,
    comm: &Commitment<G>,
    point: &[G::Scalar],
    eval: &G::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.ipa.verify(
      &vk.ck_v,
      &vk.ck_s,
      (2_usize).pow(point.len() as u32),
      &u,
      transcript,
    )?;

    Ok(())
  }
}

fn inner_product<T>(a: &[T], b: &[T]) -> T
where
  T: Field + Send + Sync,
{
  assert_eq!(a.len(), b.len());
  (0..a.len())
    .into_par_iter()
    .map(|i| a[i] * b[i])
    .reduce(|| T::ZERO, |x, y| x + y)
}

/// An inner product instance consists of a commitment to a vector `a` and another vector `b`
/// and the claim that c = <a, b>.
pub struct InnerProductInstance<G: Group> {
  comm_a_vec: Commitment<G>,
  b_vec: Vec<G::Scalar>,
  c: G::Scalar,
}

impl<G: Group> InnerProductInstance<G> {
  fn new(comm_a_vec: &Commitment<G>, b_vec: &[G::Scalar], c: &G::Scalar) -> Self {
    InnerProductInstance {
      comm_a_vec: *comm_a_vec,
      b_vec: b_vec.to_vec(),
      c: *c,
    }
  }
}

impl<G: Group> TranscriptReprTrait<G> for InnerProductInstance<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // we do not need to include self.b_vec as in our context it is produced from the transcript
    [
      self.comm_a_vec.to_transcript_bytes(),
      self.c.to_transcript_bytes(),
    ]
    .concat()
  }
}

struct InnerProductWitness<G: Group> {
  a_vec: Vec<G::Scalar>,
}

impl<G: Group> InnerProductWitness<G> {
  fn new(a_vec: &[G::Scalar]) -> Self {
    InnerProductWitness {
      a_vec: a_vec.to_vec(),
    }
  }
}

/// An inner product argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
struct InnerProductArgument<G: Group> {
  L_vec: Vec<CompressedCommitment<G>>,
  R_vec: Vec<CompressedCommitment<G>>,
  a_hat: G::Scalar,
}

impl<G> InnerProductArgument<G>
where
  G: Group,
  CommitmentKey<G>: CommitmentKeyExtTrait<G>,
{
  const fn protocol_name() -> &'static [u8] {
    b"IPA"
  }

  fn prove(
    ck: &CommitmentKey<G>,
    ck_c: &CommitmentKey<G>,
    U: &InnerProductInstance<G>,
    W: &InnerProductWitness<G>,
    transcript: &mut G::TE,
  ) -> Result<Self, NovaError> {
    transcript.dom_sep(Self::protocol_name());

    let (ck, _) = ck.split_at(U.b_vec.len());

    if U.b_vec.len() != W.a_vec.len() {
      return Err(NovaError::InvalidInputLength);
    }

    // absorb the instance in the transcript
    transcript.absorb(b"U", U);

    // sample a random base for commiting to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c.scale(&r);

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[G::Scalar],
                       b_vec: &[G::Scalar],
                       ck: &CommitmentKey<G>,
                       transcript: &mut G::TE|
     -> Result<
      (
        CompressedCommitment<G>,
        CompressedCommitment<G>,
        Vec<G::Scalar>,
        Vec<G::Scalar>,
        CommitmentKey<G>,
      ),
      NovaError,
    > {
      let n = a_vec.len();
      let (ck_L, ck_R) = ck.split_at(n / 2);

      let c_L = inner_product(&a_vec[0..n / 2], &b_vec[n / 2..n]);
      let c_R = inner_product(&a_vec[n / 2..n], &b_vec[0..n / 2]);

      let L = CE::<G>::commit(
        &ck_R.combine(&ck_c),
        &a_vec[0..n / 2]
          .iter()
          .chain(iter::once(&c_L))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
      .compress();
      let R = CE::<G>::commit(
        &ck_L.combine(&ck_c),
        &a_vec[n / 2..n]
          .iter()
          .chain(iter::once(&c_R))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
      .compress();

      transcript.absorb(b"L", &L);
      transcript.absorb(b"R", &R);

      let r = transcript.squeeze(b"r")?;
      let r_inverse = r.invert().unwrap();

      // fold the left half and the right half
      let a_vec_folded = a_vec[0..n / 2]
        .par_iter()
        .zip(a_vec[n / 2..n].par_iter())
        .map(|(a_L, a_R)| *a_L * r + r_inverse * *a_R)
        .collect::<Vec<G::Scalar>>();

      let b_vec_folded = b_vec[0..n / 2]
        .par_iter()
        .zip(b_vec[n / 2..n].par_iter())
        .map(|(b_L, b_R)| *b_L * r_inverse + r * *b_R)
        .collect::<Vec<G::Scalar>>();

      let ck_folded = ck.fold(&r_inverse, &r);

      Ok((L, R, a_vec_folded, b_vec_folded, ck_folded))
    };

    // two vectors to hold the logarithmic number of group elements
    let mut L_vec: Vec<CompressedCommitment<G>> = Vec::new();
    let mut R_vec: Vec<CompressedCommitment<G>> = Vec::new();

    // we create mutable copies of vectors and generators
    let mut a_vec = W.a_vec.to_vec();
    let mut b_vec = U.b_vec.to_vec();
    let mut ck = ck;
    for _i in 0..usize::try_from(U.b_vec.len().ilog2()).unwrap() {
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
    ck: &CommitmentKey<G>,
    ck_c: &CommitmentKey<G>,
    n: usize,
    U: &InnerProductInstance<G>,
    transcript: &mut G::TE,
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

    // sample a random base for commiting to the inner product
    let r = transcript.squeeze(b"r")?;
    let ck_c = ck_c.scale(&r);

    let P = U.comm_a_vec + CE::<G>::commit(&ck_c, &[U.c]);

    let batch_invert = |v: &[G::Scalar]| -> Result<Vec<G::Scalar>, NovaError> {
      let mut products = vec![G::Scalar::ZERO; v.len()];
      let mut acc = G::Scalar::ONE;

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // we can compute an inversion only if acc is non-zero
      if acc == G::Scalar::ZERO {
        return Err(NovaError::InvalidInputLength);
      }

      // compute the inverse once for all entries
      acc = acc.invert().unwrap();

      let mut inv = vec![G::Scalar::ZERO; v.len()];
      for i in 0..v.len() {
        let tmp = acc * v[v.len() - 1 - i];
        inv[v.len() - 1 - i] = products[v.len() - 1 - i] * acc;
        acc = tmp;
      }

      Ok(inv)
    };

    // compute a vector of public coins using self.L_vec and self.R_vec
    let r = (0..self.L_vec.len())
      .map(|i| {
        transcript.absorb(b"L", &self.L_vec[i]);
        transcript.absorb(b"R", &self.R_vec[i]);
        transcript.squeeze(b"r")
      })
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    // precompute scalars necessary for verification
    let r_square: Vec<G::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r[i] * r[i])
      .collect();
    let r_inverse = batch_invert(&r)?;
    let r_inverse_square: Vec<G::Scalar> = (0..self.L_vec.len())
      .into_par_iter()
      .map(|i| r_inverse[i] * r_inverse[i])
      .collect();

    // compute the vector with the tensor structure
    let s = {
      let mut s = vec![G::Scalar::ZERO; n];
      s[0] = {
        let mut v = G::Scalar::ONE;
        for r_inverse_i in &r_inverse {
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
      let c = CE::<G>::commit(&ck, &s).compress();
      CommitmentKey::<G>::reinterpret_commitments_as_ck(&[c])?
    };

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let ck_folded = {
        let ck_L = CommitmentKey::<G>::reinterpret_commitments_as_ck(&self.L_vec)?;
        let ck_R = CommitmentKey::<G>::reinterpret_commitments_as_ck(&self.R_vec)?;
        let ck_P = CommitmentKey::<G>::reinterpret_commitments_as_ck(&[P.compress()])?;
        ck_L.combine(&ck_R).combine(&ck_P)
      };

      CE::<G>::commit(
        &ck_folded,
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&G::Scalar::ONE))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
    };

    if P_hat == CE::<G>::commit(&ck_hat.combine(&ck_c), &[self.a_hat, self.a_hat * b_hat]) {
      Ok(())
    } else {
      Err(NovaError::InvalidIPA)
    }
  }
}
