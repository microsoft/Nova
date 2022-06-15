#![allow(clippy::too_many_arguments)]
use super::commitments::{CommitGens, CommitTrait, Commitment, CompressedCommitment};
use super::errors::NovaError;
use super::traits::{AppendToTranscriptTrait, ChallengeTrait, Group, CompressedGroup};
use ff::{Field, PrimeField};
use merlin::Transcript;
use rayon::prelude::*;
use std::marker::PhantomData;
use serde::{Serialize, Deserialize};

pub struct InnerProductWitness<G: Group> {
  x_vec: Vec<G::Scalar>,
}

pub struct InnerProductInstance<G: Group> {
  comm_x_vec: Commitment<G>,
  a_vec: Vec<G::Scalar>,
  y: G::Scalar,
}

pub struct StepInnerProductArgument<G: Group> {
  C: G::Scalar,
}

#[derive(Serialize, Deserialize)]
pub struct StepInnerProductArgumentSerialized {
  C: Vec<u8>
}

#[derive(Debug)]
pub struct FinalInnerProductArgument<G: Group> {
  L_vec: Vec<CompressedCommitment<G::CompressedGroupElement>>,
  R_vec: Vec<CompressedCommitment<G::CompressedGroupElement>>,
  x_hat: G::Scalar,
  _p: PhantomData<G>,
}

#[derive(Serialize, Deserialize)]
pub struct FinalInnerProductArgumentSerialized {
    L_vec: Vec<Vec<u8>>,
    R_vec: Vec<Vec<u8>>,
    x_hat: Vec<u8>
}

impl<G: Group> InnerProductWitness<G> {
  pub fn new(x_vec: &[G::Scalar]) -> Self {
    InnerProductWitness {
      x_vec: x_vec.to_vec(),
    }
  }
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
    U1.comm_x_vec
      .append_to_transcript(b"U1_comm_x_vec", transcript);
    U1.a_vec.append_to_transcript(b"U1_a_vec", transcript);
    U2.comm_x_vec
      .append_to_transcript(b"U2_comm_x_vec", transcript);
    U2.a_vec.append_to_transcript(b"U2_a_vec", transcript);

    // commit to the cross-term
    let C = {
      let T1 = (0..W1.x_vec.len())
        .map(|i| W1.x_vec[i] * U2.a_vec[i])
        .fold(G::Scalar::zero(), |acc, item| acc + item);

      let T2 = (0..W2.x_vec.len())
        .map(|i| W2.x_vec[i] * U1.a_vec[i])
        .fold(G::Scalar::zero(), |acc, item| acc + item);

      T1 + T2
    };

    // add the cross-term to the transcript
    C.append_to_transcript(b"C", transcript);

    // obtain a random challenge
    let r = G::Scalar::challenge(b"r", transcript);

    // fold the vectors and their inner product
    let x_vec = W1
      .x_vec
      .iter()
      .zip(W2.x_vec.iter())
      .map(|(x1, x2)| *x1 + r * x2)
      .collect::<Vec<G::Scalar>>();
    let a_vec = U1
      .a_vec
      .iter()
      .zip(U2.a_vec.iter())
      .map(|(a1, a2)| *a1 + r * a2)
      .collect::<Vec<G::Scalar>>();
    let y = U1.y + r * r * U2.y + r * C;
    let comm_x_vec = U1.comm_x_vec + U2.comm_x_vec * r;

    let W = InnerProductWitness { x_vec };

    let U = InnerProductInstance {
      comm_x_vec,
      a_vec,
      y,
    };

    (StepInnerProductArgument { C }, U, W)
  }

  pub fn verify(
    &self,
    U1: &InnerProductInstance<G>,
    U2: &InnerProductInstance<G>,
    transcript: &mut Transcript,
  ) -> InnerProductInstance<G> {
    transcript.append_message(b"protocol-name", Self::protocol_name());

    // add the two commitments and two public vectors to the transcript
    U1.comm_x_vec
      .append_to_transcript(b"U1_comm_x_vec", transcript);
    U1.a_vec.append_to_transcript(b"U1_a_vec", transcript);
    U2.comm_x_vec
      .append_to_transcript(b"U2_comm_x_vec", transcript);
    U2.a_vec.append_to_transcript(b"U2_a_vec", transcript);

    // add the cross-term to the transcript
    self.C.append_to_transcript(b"C", transcript);

    // obtain a random challenge
    let r = G::Scalar::challenge(b"r", transcript);

    // fold the vectors and their inner product
    let a_vec = U1
      .a_vec
      .iter()
      .zip(U2.a_vec.iter())
      .map(|(a1, a2)| *a1 + r * a2)
      .collect::<Vec<G::Scalar>>();
    let y = U1.y + r * r * U2.y + r * self.C;
    let comm_x_vec = U1.comm_x_vec + U2.comm_x_vec * r;

    InnerProductInstance {
      comm_x_vec,
      a_vec,
      y,
    }
  }

  pub fn serialize(&self) -> StepInnerProductArgumentSerialized {
    StepInnerProductArgumentSerialized {
        C: self.C.to_repr().as_ref().to_vec(),
    }
  }
}

impl<G: Group> InnerProductInstance<G> {
  pub fn new(comm_x_vec: &Commitment<G>, a_vec: &[G::Scalar], y: &G::Scalar) -> Self {
    InnerProductInstance {
      comm_x_vec: *comm_x_vec,
      a_vec: a_vec.to_vec(),
      y: *y,
    }
  }
}

impl<G: Group> FinalInnerProductArgument<G> {
  fn protocol_name() -> &'static [u8] {
    b"inner product argument (final)"
  }

  pub fn inner_product(a: &[G::Scalar], b: &[G::Scalar]) -> G::Scalar {
    assert_eq!(a.len(), b.len());
    (0..a.len())
      .into_par_iter()
      .map(|i| a[i] * b[i])
      .reduce(G::Scalar::zero, |x, y| x + y)
  }

  pub fn prove(
    U: &InnerProductInstance<G>,
    W: &InnerProductWitness<G>,
    gens: &CommitGens<G>,
    transcript: &mut Transcript,
  ) -> Result<Self, NovaError> {
    transcript.append_message(b"protocol-name", Self::protocol_name());

    let n = W.x_vec.len();
    if U.a_vec.len() != n || gens.len() != n || !gens.len().is_power_of_two() {
      return Err(NovaError::InvalidInputLength);
    }

    U.comm_x_vec.append_to_transcript(b"comm_x_vec", transcript);
    U.a_vec.append_to_transcript(b"a_vec", transcript);
    U.y.append_to_transcript(b"y", transcript);

    // sample a random base for commiting to the inner product
    let r = G::Scalar::challenge(b"r", transcript);
    let gens_y = CommitGens::from_scalar(&r); // TODO: take externally, but randomize it

    // produce a logarithmic-sized argument
    let (L_vec, R_vec, x_hat) = {
      // we create mutable copies of vectors and generators
      let mut x_vec_ref = W.x_vec.to_vec();
      let mut a_vec_ref = U.a_vec.to_vec();
      let mut gens_ref = gens.clone();

      // two vectors to hold the logarithmic number of group elements
      let mut n = gens_ref.len();
      let mut L_vec: Vec<CompressedCommitment<G::CompressedGroupElement>> = Vec::new();
      let mut R_vec: Vec<CompressedCommitment<G::CompressedGroupElement>> = Vec::new();

      for _i in 0..(U.a_vec.len() as f64).log2() as usize {
        let (x_L, x_R) = (x_vec_ref[0..n / 2].to_vec(), x_vec_ref[n / 2..n].to_vec());
        let (a_L, a_R) = (a_vec_ref[0..n / 2].to_vec(), a_vec_ref[n / 2..n].to_vec());
        let (gens_L, gens_R) = gens_ref.split_at(n / 2);

        let c_L = Self::inner_product(&x_L, &a_R);
        let c_R = Self::inner_product(&x_R, &a_L);

        let L = {
          let v = {
            let mut v = x_L.to_vec();
            v.push(c_L);
            v
          };
          v.commit(&gens_R.combine(&gens_y)).compress()
        };
        let R = {
          let v = {
            let mut v = x_R.to_vec();
            v.push(c_R);
            v
          };
          v.commit(&gens_L.combine(&gens_y)).compress()
        };

        L.append_to_transcript(b"L", transcript);
        R.append_to_transcript(b"R", transcript);

        L_vec.push(L);
        R_vec.push(R);

        let r = G::Scalar::challenge(b"challenge_r", transcript);
        let r_inverse = r.invert().unwrap();

        // fold the left half and the right half
        x_vec_ref = x_L
          .par_iter()
          .zip(x_R.par_iter())
          .map(|(x_L, x_R)| *x_L * r + r_inverse * *x_R)
          .collect::<Vec<G::Scalar>>();

        a_vec_ref = a_L
          .par_iter()
          .zip(a_R.par_iter())
          .map(|(a_L, a_R)| *a_L * r_inverse + r * *a_R)
          .collect::<Vec<G::Scalar>>();

        gens_ref.fold(&r_inverse, &r);
        n /= 2;
      }

      (L_vec, R_vec, x_vec_ref[0])
    };

    Ok(FinalInnerProductArgument {
      L_vec,
      R_vec,
      x_hat,
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
    n: usize,
    U: &InnerProductInstance<G>,
    gens: &CommitGens<G>,
    transcript: &mut Transcript,
  ) -> Result<(), NovaError> {
    transcript.append_message(b"protocol-name", Self::protocol_name());
    if gens.len() != n
      || U.a_vec.len() != n
      || n != (1 << self.L_vec.len())
      || self.L_vec.len() != self.R_vec.len()
      || self.L_vec.len() >= 32
    {
      return Err(NovaError::InvalidInputLength);
    }

    U.comm_x_vec.append_to_transcript(b"comm_x_vec", transcript);
    U.a_vec.append_to_transcript(b"a_vec", transcript);
    U.y.append_to_transcript(b"y", transcript);

    // sample a random base for commiting to the inner product
    let r = G::Scalar::challenge(b"r", transcript);
    let gens_y = CommitGens::from_scalar(&r);

    let comm_y = [U.y].commit(&gens_y);
    let gamma = U.comm_x_vec + comm_y;

    // verify the logarithmic-sized inner product argument
    let (gens_hat, gamma_hat, a_hat) = {
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

      let a_hat = Self::inner_product(&U.a_vec, &exps);

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

      (gens_hat, gamma_hat, a_hat)
    };

    let lhs = gamma_hat;
    let rhs = {
      let y_hat = self.x_hat * a_hat;
      let gens_hat = gens_hat.combine(&gens_y);
      [self.x_hat, y_hat].commit(&gens_hat)
    };

    if lhs == rhs {
      Ok(())
    } else {
      Err(NovaError::InvalidIPA)
    }
  }

  pub fn serialize(&self) -> FinalInnerProductArgumentSerialized {
     FinalInnerProductArgumentSerialized {
        L_vec: self.L_vec.iter().map(|x| x.comm.as_bytes().to_vec()).collect(),
        R_vec: self.R_vec.iter().map(|x| x.comm.as_bytes().to_vec()).collect(),
        x_hat: self.x_hat.to_repr().as_ref().to_vec(),
     }
  }
}
