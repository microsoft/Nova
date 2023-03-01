//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::NovaError,
  provider::pedersen::CommitmentGensExtTrait,
  spartan::polynomial::EqPolynomial,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentGensTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    AppendToTranscriptTrait, ChallengeTrait, Group, TranscriptEngineTrait,
  },
  Commitment, CommitmentGens, CompressedCommitment, CE,
};
use core::iter;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Provides an implementation of generators for proving evaluations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EvaluationGens<G: Group> {
  gens_v: CommitmentGens<G>,
  gens_s: CommitmentGens<G>,
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
  CommitmentGens<G>: CommitmentGensExtTrait<G, CE = G::CE>,
{
  type CE = G::CE;
  type EvaluationGens = EvaluationGens<G>;
  type EvaluationArgument = EvaluationArgument<G>;

  fn setup(gens: &<Self::CE as CommitmentEngineTrait<G>>::CommitmentGens) -> Self::EvaluationGens {
    EvaluationGens {
      gens_v: gens.clone(),
      gens_s: CommitmentGens::<G>::new(b"ipa", 1),
    }
  }

  fn prove(
    gens: &Self::EvaluationGens,
    transcript: &mut G::TE,
    comm: &Commitment<G>,
    poly: &[G::Scalar],
    point: &[G::Scalar],
    eval: &G::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);
    let w = InnerProductWitness::new(poly);

    Ok(EvaluationArgument {
      ipa: InnerProductArgument::prove(&gens.gens_v, &gens.gens_s, &u, &w, transcript)?,
    })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    gens: &Self::EvaluationGens,
    transcript: &mut G::TE,
    comm: &Commitment<G>,
    point: &[G::Scalar],
    eval: &G::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let u = InnerProductInstance::new(comm, &EqPolynomial::new(point.to_vec()).evals(), eval);

    arg.ipa.verify(
      &gens.gens_v,
      &gens.gens_s,
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
    .reduce(T::zero, |x, y| x + y)
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
  _p: PhantomData<G>,
}

impl<G> InnerProductArgument<G>
where
  G: Group,
  CommitmentGens<G>: CommitmentGensExtTrait<G, CE = G::CE>,
{
  fn protocol_name() -> &'static [u8] {
    b"inner product argument"
  }

  fn prove(
    gens: &CommitmentGens<G>,
    gens_c: &CommitmentGens<G>,
    U: &InnerProductInstance<G>,
    W: &InnerProductWitness<G>,
    transcript: &mut G::TE,
  ) -> Result<Self, NovaError> {
    transcript.absorb_bytes(b"protocol-name", Self::protocol_name());

    if U.b_vec.len() != W.a_vec.len() {
      return Err(NovaError::InvalidInputLength);
    }

    U.comm_a_vec.append_to_transcript(b"comm_a_vec", transcript);
    <G::Scalar as AppendToTranscriptTrait<G>>::append_to_transcript(&U.c, b"c", transcript);

    // sample a random base for commiting to the inner product
    let r = G::Scalar::challenge(b"r", transcript)?;
    let gens_c = gens_c.scale(&r);

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[G::Scalar],
                       b_vec: &[G::Scalar],
                       gens: &CommitmentGens<G>,
                       transcript: &mut G::TE|
     -> Result<
      (
        CompressedCommitment<G>,
        CompressedCommitment<G>,
        Vec<G::Scalar>,
        Vec<G::Scalar>,
        CommitmentGens<G>,
      ),
      NovaError,
    > {
      let n = a_vec.len();
      let (gens_L, gens_R) = gens.split_at(n / 2);

      let c_L = inner_product(&a_vec[0..n / 2], &b_vec[n / 2..n]);
      let c_R = inner_product(&a_vec[n / 2..n], &b_vec[0..n / 2]);

      let L = CE::<G>::commit(
        &gens_R.combine(&gens_c),
        &a_vec[0..n / 2]
          .iter()
          .chain(iter::once(&c_L))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
      .compress();
      let R = CE::<G>::commit(
        &gens_L.combine(&gens_c),
        &a_vec[n / 2..n]
          .iter()
          .chain(iter::once(&c_R))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
      .compress();

      L.append_to_transcript(b"L", transcript);
      R.append_to_transcript(b"R", transcript);

      let r = G::Scalar::challenge(b"challenge_r", transcript)?;
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

      let gens_folded = gens.fold(&r_inverse, &r);

      Ok((L, R, a_vec_folded, b_vec_folded, gens_folded))
    };

    // two vectors to hold the logarithmic number of group elements
    let mut L_vec: Vec<CompressedCommitment<G>> = Vec::new();
    let mut R_vec: Vec<CompressedCommitment<G>> = Vec::new();

    // we create mutable copies of vectors and generators
    let mut a_vec = W.a_vec.to_vec();
    let mut b_vec = U.b_vec.to_vec();
    let mut gens = gens.clone();
    for _i in 0..(U.b_vec.len() as f64).log2() as usize {
      let (L, R, a_vec_folded, b_vec_folded, gens_folded) =
        prove_inner(&a_vec, &b_vec, &gens, transcript)?;
      L_vec.push(L);
      R_vec.push(R);

      a_vec = a_vec_folded;
      b_vec = b_vec_folded;
      gens = gens_folded;
    }

    Ok(InnerProductArgument {
      L_vec,
      R_vec,
      a_hat: a_vec[0],
      _p: Default::default(),
    })
  }

  fn verify(
    &self,
    gens: &CommitmentGens<G>,
    gens_c: &CommitmentGens<G>,
    n: usize,
    U: &InnerProductInstance<G>,
    transcript: &mut G::TE,
  ) -> Result<(), NovaError> {
    transcript.absorb_bytes(b"protocol-name", Self::protocol_name());
    if U.b_vec.len() != n
      || n != (1 << self.L_vec.len())
      || self.L_vec.len() != self.R_vec.len()
      || self.L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    U.comm_a_vec.append_to_transcript(b"comm_a_vec", transcript);
    <G::Scalar as AppendToTranscriptTrait<G>>::append_to_transcript(&U.c, b"c", transcript);

    // sample a random base for commiting to the inner product
    let r = G::Scalar::challenge(b"r", transcript)?;
    let gens_c = gens_c.scale(&r);

    let P = U.comm_a_vec + CE::<G>::commit(&gens_c, &[U.c]);

    let batch_invert = |v: &[G::Scalar]| -> Result<Vec<G::Scalar>, NovaError> {
      let mut products = vec![G::Scalar::zero(); v.len()];
      let mut acc = G::Scalar::one();

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // we can compute an inversion only if acc is non-zero
      if acc == G::Scalar::zero() {
        return Err(NovaError::InvalidInputLength);
      }

      // compute the inverse once for all entries
      acc = acc.invert().unwrap();

      let mut inv = vec![G::Scalar::zero(); v.len()];
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
        self.L_vec[i].append_to_transcript(b"L", transcript);
        self.R_vec[i].append_to_transcript(b"R", transcript);
        G::Scalar::challenge(b"challenge_r", transcript)
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
      let mut s = vec![G::Scalar::zero(); n];
      s[0] = {
        let mut v = G::Scalar::one();
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

    let gens_hat = {
      let c = CE::<G>::commit(gens, &s).compress();
      CommitmentGens::<G>::reinterpret_commitments_as_gens(&[c])?
    };

    let b_hat = inner_product(&U.b_vec, &s);

    let P_hat = {
      let gens_folded = {
        let gens_L = CommitmentGens::<G>::reinterpret_commitments_as_gens(&self.L_vec)?;
        let gens_R = CommitmentGens::<G>::reinterpret_commitments_as_gens(&self.R_vec)?;
        let gens_P = CommitmentGens::<G>::reinterpret_commitments_as_gens(&[P.compress()])?;
        gens_L.combine(&gens_R).combine(&gens_P)
      };

      CE::<G>::commit(
        &gens_folded,
        &r_square
          .iter()
          .chain(r_inverse_square.iter())
          .chain(iter::once(&G::Scalar::one()))
          .copied()
          .collect::<Vec<G::Scalar>>(),
      )
    };

    if P_hat
      == CE::<G>::commit(
        &gens_hat.combine(&gens_c),
        &[self.a_hat, self.a_hat * b_hat],
      )
    {
      Ok(())
    } else {
      Err(NovaError::InvalidIPA)
    }
  }
}
