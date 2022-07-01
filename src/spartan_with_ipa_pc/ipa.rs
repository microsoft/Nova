#![allow(clippy::too_many_arguments)]
use crate::commitments::{CommitGens, CommitTrait, Commitment, CompressedCommitment};
use crate::errors::NovaError;
use crate::traits::{AppendToTranscriptTrait, ChallengeTrait, Group};
use ff::Field;
use merlin::Transcript;
use rayon::prelude::*;
use std::marker::PhantomData;

pub fn inner_product<T>(a: &[T], b: &[T]) -> T
where
  T: Field + Send + Sync,
{
  assert_eq!(a.len(), b.len());
  (0..a.len())
    .into_par_iter()
    .map(|i| a[i] * b[i])
    .reduce(T::zero, |x, y| x + y)
}

pub struct InnerProductInstance<G: Group> {
  comm_a_vec: Commitment<G>,
  b_vec: Vec<G::Scalar>,
  c: G::Scalar,
}

impl<G: Group> InnerProductInstance<G> {
  pub fn new(comm_a_vec: &Commitment<G>, b_vec: &[G::Scalar], c: &G::Scalar) -> Self {
    InnerProductInstance {
      comm_a_vec: *comm_a_vec,
      b_vec: b_vec.to_vec(),
      c: *c,
    }
  }
}

pub struct InnerProductWitness<G: Group> {
  a_vec: Vec<G::Scalar>,
}

impl<G: Group> InnerProductWitness<G> {
  pub fn new(a_vec: &[G::Scalar]) -> Self {
    InnerProductWitness {
      a_vec: a_vec.to_vec(),
    }
  }
}

pub struct StepInnerProductArgument<G: Group> {
  cross_term: G::Scalar,
}

impl<G: Group> StepInnerProductArgument<G> {
  pub fn protocol_name() -> &'static [u8] {
    b"inner product argument (step)"
  }

  pub fn prove(
    U1: &InnerProductInstance<G>,
    W1: &InnerProductWitness<G>,
    U2: &InnerProductInstance<G>,
    W2: &InnerProductWitness<G>,
    transcript: &mut Transcript,
  ) -> (Self, InnerProductInstance<G>, InnerProductWitness<G>) {
    transcript.append_message(b"protocol-name", Self::protocol_name());

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

    (StepInnerProductArgument { cross_term }, U, W)
  }

  pub fn verify(
    &self,
    U1: &InnerProductInstance<G>,
    U2: &InnerProductInstance<G>,
    transcript: &mut Transcript,
  ) -> InnerProductInstance<G> {
    transcript.append_message(b"protocol-name", Self::protocol_name());

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

#[derive(Debug)]
pub struct FinalInnerProductArgument<G: Group> {
  L_vec: Vec<CompressedCommitment<G::CompressedGroupElement>>,
  R_vec: Vec<CompressedCommitment<G::CompressedGroupElement>>,
  a_hat: G::Scalar,
  _p: PhantomData<G>,
}

impl<G: Group> FinalInnerProductArgument<G> {
  fn protocol_name() -> &'static [u8] {
    b"inner product argument (final)"
  }

  pub fn prove(
    gens: &CommitGens<G>,
    gens_c: &CommitGens<G>,
    U: &InnerProductInstance<G>,
    W: &InnerProductWitness<G>,
    transcript: &mut Transcript,
  ) -> Result<Self, NovaError> {
    transcript.append_message(b"protocol-name", Self::protocol_name());

    let n = W.a_vec.len();
    if U.b_vec.len() != n || gens.len() != n || !gens.len().is_power_of_two() {
      return Err(NovaError::InvalidInputLength);
    }

    U.comm_a_vec.append_to_transcript(b"comm_a_vec", transcript);
    U.b_vec.append_to_transcript(b"b_vec", transcript);
    U.c.append_to_transcript(b"c", transcript);

    // sample a random base for commiting to the inner product
    let r = G::Scalar::challenge(b"r", transcript);
    let gens_c = gens_c.scale(&r);

    // produce a logarithmic-sized argument
    let (L_vec, R_vec, a_hat) = {
      // we create mutable copies of vectors and generators
      let mut a_vec_ref = W.a_vec.to_vec();
      let mut b_vec_ref = U.b_vec.to_vec();
      let mut gens_ref = gens.clone();

      // two vectors to hold the logarithmic number of group elements
      let mut n = gens_ref.len();
      let mut L_vec: Vec<CompressedCommitment<G::CompressedGroupElement>> = Vec::new();
      let mut R_vec: Vec<CompressedCommitment<G::CompressedGroupElement>> = Vec::new();

      for _i in 0..(U.b_vec.len() as f64).log2() as usize {
        let (a_L, a_R) = (a_vec_ref[0..n / 2].to_vec(), a_vec_ref[n / 2..n].to_vec());
        let (b_L, b_R) = (b_vec_ref[0..n / 2].to_vec(), b_vec_ref[n / 2..n].to_vec());
        let (gens_L, gens_R) = gens_ref.split_at(n / 2);

        let c_L = inner_product(&a_L, &b_R);
        let c_R = inner_product(&a_R, &b_L);

        let L = {
          let v = {
            let mut v = a_L.to_vec();
            v.push(c_L);
            v
          };
          v.commit(&gens_R.combine(&gens_c)).compress()
        };
        let R = {
          let v = {
            let mut v = a_R.to_vec();
            v.push(c_R);
            v
          };
          v.commit(&gens_L.combine(&gens_c)).compress()
        };

        L.append_to_transcript(b"L", transcript);
        R.append_to_transcript(b"R", transcript);

        L_vec.push(L);
        R_vec.push(R);

        let r = G::Scalar::challenge(b"challenge_r", transcript);
        let r_inverse = r.invert().unwrap();

        // fold the left half and the right half
        a_vec_ref = a_L
          .par_iter()
          .zip(a_R.par_iter())
          .map(|(a_L, a_R)| *a_L * r + r_inverse * *a_R)
          .collect::<Vec<G::Scalar>>();

        b_vec_ref = b_L
          .par_iter()
          .zip(b_R.par_iter())
          .map(|(b_L, b_R)| *b_L * r_inverse + r * *b_R)
          .collect::<Vec<G::Scalar>>();

        gens_ref.fold(&r_inverse, &r);
        n /= 2;
      }

      (L_vec, R_vec, a_vec_ref[0])
    };

    Ok(FinalInnerProductArgument {
      L_vec,
      R_vec,
      a_hat,
      _p: Default::default(),
    })
  }

  fn batch_invert(v: &[G::Scalar]) -> Vec<G::Scalar> {
    let mut products = vec![G::Scalar::zero(); v.len()];
    let mut acc = G::Scalar::one();

    for i in 0..v.len() {
      products[i] = acc;
      acc *= v[i];
    }

    assert_ne!(acc, G::Scalar::zero());

    // compute the inverse once for all entries
    acc = acc.invert().unwrap();

    let mut inv = vec![G::Scalar::zero(); v.len()];
    for i in 0..v.len() {
      let tmp = acc * v[v.len() - 1 - i];
      inv[v.len() - 1 - i] = products[v.len() - 1 - i] * acc;
      acc = tmp;
    }

    inv
  }

  pub fn verify(
    &self,
    gens: &CommitGens<G>,
    gens_c: &CommitGens<G>,
    n: usize,
    U: &InnerProductInstance<G>,
    transcript: &mut Transcript,
  ) -> Result<(), NovaError> {
    transcript.append_message(b"protocol-name", Self::protocol_name());
    if gens.len() != n
      || U.b_vec.len() != n
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
    let gamma = U.comm_a_vec + [U.c].commit(&gens_c);

    // verify the logarithmic-sized inner product argument
    let (gens_hat, gamma_hat, b_hat) = {
      // compute a vector of public coins using self.L_vec and self.R_vec
      let r = (0..self.L_vec.len())
        .map(|i| {
          self.L_vec[i].append_to_transcript(b"L", transcript);
          self.R_vec[i].append_to_transcript(b"R", transcript);
          G::Scalar::challenge(b"challenge_r", transcript)
        })
        .collect::<Vec<G::Scalar>>();

      // precompute scalars necessary for verification
      let (exps, r_square, r_inverse_square) = {
        let r_inverse = Self::batch_invert(&r);
        let r_square: Vec<G::Scalar> = (0..self.L_vec.len())
          .into_par_iter()
          .map(|i| r[i] * r[i])
          .collect();
        let r_inverse_square: Vec<G::Scalar> = (0..self.L_vec.len())
          .into_par_iter()
          .map(|i| r_inverse[i] * r_inverse[i])
          .collect();

        // compute the vector with the tensor structure
        let exps = {
          let mut exps = vec![G::Scalar::zero(); n];
          exps[0] = {
            let mut v = G::Scalar::one();
            for r_inverse_i in &r_inverse {
              v *= r_inverse_i;
            }
            v
          };
          for i in 1..n {
            let pos_in_u = (31 - (i as u32).leading_zeros()) as usize;
            exps[i] = exps[i - (1 << pos_in_u)] * r_square[(self.L_vec.len() - 1) - pos_in_u];
          }
          exps
        };

        (exps, r_square, r_inverse_square)
      };

      let gens_hat = {
        let c = exps.commit(gens).compress();
        CommitGens::reinterpret_commitments_as_gens(&[c])?
      };

      let b_hat = inner_product(&U.b_vec, &exps);

      let gens_folded = {
        let gens_L = CommitGens::reinterpret_commitments_as_gens(&self.L_vec)?;
        let gens_R = CommitGens::reinterpret_commitments_as_gens(&self.R_vec)?;
        let gens_gamma = CommitGens::reinterpret_commitments_as_gens(&[gamma.compress()])?;
        gens_L.combine(&gens_R).combine(&gens_gamma)
      };

      let gamma_hat = {
        let scalars: Vec<G::Scalar> = {
          let mut v = r_square;
          v.extend(r_inverse_square);
          v.push(G::Scalar::one());
          v
        };

        scalars.commit(&gens_folded)
      };

      (gens_hat, gamma_hat, b_hat)
    };

    let lhs = gamma_hat;
    let rhs = {
      let c_hat = self.a_hat * b_hat;
      let gens_hat = gens_hat.combine(&gens_c);
      [self.a_hat, c_hat].commit(&gens_hat)
    };

    if lhs == rhs {
      Ok(())
    } else {
      Err(NovaError::InvalidIPA)
    }
  }
}
