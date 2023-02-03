//! This module implements `EvaluationEngine` using an IPA-based polynomial commitment scheme
#![allow(clippy::too_many_arguments)]
use crate::{
  errors::NovaError,
  spartan::polynomial::EqPolynomial,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentGensTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    AppendToTranscriptTrait, ChallengeTrait, Group,
  },
  Commitment, CommitmentGens, CompressedCommitment, CE,
};
use core::{cmp::max, iter};
use ff::Field;
use merlin::Transcript;
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
  nifs: Vec<NIFSForInnerProduct<G>>,
  ipa: InnerProductArgument<G>,
}

/// Provides an implementation of a polynomial evaluation engine using IPA
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> EvaluationEngineTrait<G> for EvaluationEngine<G> {
  type CE = G::CE;
  type EvaluationGens = EvaluationGens<G>;
  type EvaluationArgument = EvaluationArgument<G>;

  fn setup(gens: &<Self::CE as CommitmentEngineTrait<G>>::CommitmentGens) -> Self::EvaluationGens {
    EvaluationGens {
      gens_v: gens.clone(),
      gens_s: CommitmentGens::<G>::new(b"ipa", 1),
    }
  }

  fn prove_batch(
    gens: &Self::EvaluationGens,
    transcript: &mut Transcript,
    comms: &[Commitment<G>],
    polys: &[Vec<G::Scalar>],
    points: &[Vec<G::Scalar>],
    evals: &[G::Scalar],
  ) -> Result<Self::EvaluationArgument, NovaError> {
    // sanity checks (these should never fail)
    assert!(polys.len() >= 2);
    assert_eq!(comms.len(), polys.len());
    assert_eq!(comms.len(), points.len());
    assert_eq!(comms.len(), evals.len());

    let mut r_U = InnerProductInstance::new(
      &comms[0],
      &EqPolynomial::new(points[0].clone()).evals(),
      &evals[0],
    );
    let mut r_W = InnerProductWitness::new(&polys[0]);
    let mut nifs = Vec::new();

    for i in 1..polys.len() {
      let (n, u, w) = NIFSForInnerProduct::prove(
        &r_U,
        &r_W,
        &InnerProductInstance::new(
          &comms[i],
          &EqPolynomial::new(points[i].clone()).evals(),
          &evals[i],
        ),
        &InnerProductWitness::new(&polys[i]),
        transcript,
      );
      nifs.push(n);
      r_U = u;
      r_W = w;
    }

    let ipa = InnerProductArgument::prove(&gens.gens_v, &gens.gens_s, &r_U, &r_W, transcript)?;

    Ok(EvaluationArgument { nifs, ipa })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify_batch(
    gens: &Self::EvaluationGens,
    transcript: &mut Transcript,
    comms: &[<Self::CE as CommitmentEngineTrait<G>>::Commitment],
    points: &[Vec<G::Scalar>],
    evals: &[G::Scalar],
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    // sanity checks (these should never fail)
    assert!(comms.len() >= 2);
    assert_eq!(comms.len(), points.len());
    assert_eq!(comms.len(), evals.len());

    let mut r_U = InnerProductInstance::new(
      &comms[0],
      &EqPolynomial::new(points[0].clone()).evals(),
      &evals[0],
    );
    let mut num_vars = points[0].len();
    for i in 1..comms.len() {
      let u = arg.nifs[i - 1].verify(
        &r_U,
        &InnerProductInstance::new(
          &comms[i],
          &EqPolynomial::new(points[i].clone()).evals(),
          &evals[i],
        ),
        transcript,
      );
      r_U = u;
      num_vars = max(num_vars, points[i].len());
    }

    arg.ipa.verify(
      &gens.gens_v,
      &gens.gens_s,
      (2_usize).pow(num_vars as u32),
      &r_U,
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

  fn pad(&self, n: usize) -> InnerProductInstance<G> {
    let mut b_vec = self.b_vec.clone();
    b_vec.resize(n, G::Scalar::zero());
    InnerProductInstance {
      comm_a_vec: self.comm_a_vec,
      b_vec,
      c: self.c,
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

  fn pad(&self, n: usize) -> InnerProductWitness<G> {
    let mut a_vec = self.a_vec.clone();
    a_vec.resize(n, G::Scalar::zero());
    InnerProductWitness { a_vec }
  }
}

/// A non-interactive folding scheme (NIFS) for inner product relations
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NIFSForInnerProduct<G: Group> {
  cross_term: G::Scalar,
}

impl<G: Group> NIFSForInnerProduct<G> {
  fn protocol_name() -> &'static [u8] {
    b"NIFSForInnerProduct"
  }

  fn prove(
    U1: &InnerProductInstance<G>,
    W1: &InnerProductWitness<G>,
    U2: &InnerProductInstance<G>,
    W2: &InnerProductWitness<G>,
    transcript: &mut Transcript,
  ) -> (Self, InnerProductInstance<G>, InnerProductWitness<G>) {
    transcript.append_message(b"protocol-name", Self::protocol_name());

    // pad the instances and witness so they are of the same length
    let U1 = U1.pad(max(U1.b_vec.len(), U2.b_vec.len()));
    let U2 = U2.pad(max(U1.b_vec.len(), U2.b_vec.len()));
    let W1 = W1.pad(max(U1.b_vec.len(), U2.b_vec.len()));
    let W2 = W2.pad(max(U1.b_vec.len(), U2.b_vec.len()));

    // add the two commitments and two public vectors to the transcript
    U1.comm_a_vec
      .append_to_transcript(b"U1_comm_a_vec", transcript);
    U1.b_vec.append_to_transcript(b"U1_b_vec", transcript);
    U2.comm_a_vec
      .append_to_transcript(b"U2_comm_a_vec", transcript);
    U2.b_vec.append_to_transcript(b"U2_b_vec", transcript);

    // compute the cross-term
    let cross_term = inner_product(&W1.a_vec, &U2.b_vec) + inner_product(&W2.a_vec, &U1.b_vec);

    // add the cross-term to the transcript
    cross_term.append_to_transcript(b"cross_term", transcript);

    // obtain a random challenge
    let r = G::Scalar::challenge(b"r", transcript);

    // fold the vectors and their inner product
    let a_vec = W1
      .a_vec
      .par_iter()
      .zip(W2.a_vec.par_iter())
      .map(|(x1, x2)| *x1 + r * x2)
      .collect::<Vec<G::Scalar>>();
    let b_vec = U1
      .b_vec
      .par_iter()
      .zip(U2.b_vec.par_iter())
      .map(|(a1, a2)| *a1 + r * a2)
      .collect::<Vec<G::Scalar>>();

    let c = U1.c + r * r * U2.c + r * cross_term;
    let comm_a_vec = U1.comm_a_vec + U2.comm_a_vec * r;

    let W = InnerProductWitness { a_vec };
    let U = InnerProductInstance {
      comm_a_vec,
      b_vec,
      c,
    };

    (NIFSForInnerProduct { cross_term }, U, W)
  }

  fn verify(
    &self,
    U1: &InnerProductInstance<G>,
    U2: &InnerProductInstance<G>,
    transcript: &mut Transcript,
  ) -> InnerProductInstance<G> {
    transcript.append_message(b"protocol-name", Self::protocol_name());

    // pad the instances so they are of the same length
    let U1 = U1.pad(max(U1.b_vec.len(), U2.b_vec.len()));
    let U2 = U2.pad(max(U1.b_vec.len(), U2.b_vec.len()));

    // add the two commitments and two public vectors to the transcript
    U1.comm_a_vec
      .append_to_transcript(b"U1_comm_a_vec", transcript);
    U1.b_vec.append_to_transcript(b"U1_b_vec", transcript);
    U2.comm_a_vec
      .append_to_transcript(b"U2_comm_a_vec", transcript);
    U2.b_vec.append_to_transcript(b"U2_b_vec", transcript);

    // add the cross-term to the transcript
    self
      .cross_term
      .append_to_transcript(b"cross_term", transcript);

    // obtain a random challenge
    let r = G::Scalar::challenge(b"r", transcript);

    // fold the vectors and their inner product
    let b_vec = U1
      .b_vec
      .par_iter()
      .zip(U2.b_vec.par_iter())
      .map(|(a1, a2)| *a1 + r * a2)
      .collect::<Vec<G::Scalar>>();
    let c = U1.c + r * r * U2.c + r * self.cross_term;
    let comm_a_vec = U1.comm_a_vec + U2.comm_a_vec * r;

    InnerProductInstance {
      comm_a_vec,
      b_vec,
      c,
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

impl<G: Group> InnerProductArgument<G> {
  fn protocol_name() -> &'static [u8] {
    b"inner product argument"
  }

  fn prove(
    gens: &CommitmentGens<G>,
    gens_c: &CommitmentGens<G>,
    U: &InnerProductInstance<G>,
    W: &InnerProductWitness<G>,
    transcript: &mut Transcript,
  ) -> Result<Self, NovaError> {
    transcript.append_message(b"protocol-name", Self::protocol_name());

    if U.b_vec.len() != W.a_vec.len() {
      return Err(NovaError::InvalidInputLength);
    }

    U.comm_a_vec.append_to_transcript(b"comm_a_vec", transcript);
    U.b_vec.append_to_transcript(b"b_vec", transcript);
    U.c.append_to_transcript(b"c", transcript);

    // sample a random base for commiting to the inner product
    let r = G::Scalar::challenge(b"r", transcript);
    let gens_c = gens_c.scale(&r);

    // a closure that executes a step of the recursive inner product argument
    let prove_inner = |a_vec: &[G::Scalar],
                       b_vec: &[G::Scalar],
                       gens: &CommitmentGens<G>,
                       transcript: &mut Transcript|
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

      let r = G::Scalar::challenge(b"challenge_r", transcript);
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
    transcript: &mut Transcript,
  ) -> Result<(), NovaError> {
    transcript.append_message(b"protocol-name", Self::protocol_name());
    if U.b_vec.len() != n
      || n != (1 << self.L_vec.len())
      || self.L_vec.len() != self.R_vec.len()
      || self.L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    U.comm_a_vec.append_to_transcript(b"comm_a_vec", transcript);
    U.b_vec.append_to_transcript(b"b_vec", transcript);
    U.c.append_to_transcript(b"c", transcript);

    // sample a random base for commiting to the inner product
    let r = G::Scalar::challenge(b"r", transcript);
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
      .collect::<Vec<G::Scalar>>();

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
