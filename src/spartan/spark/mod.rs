//! This module implements `CompCommitmentEngineTrait` using Spartan's SPARK compiler
//! We also provide a trivial implementation that has the verifier evaluate the sparse polynomials
use crate::{
  errors::NovaError,
  r1cs::R1CSShape,
  spartan::{math::Math, CompCommitmentEngineTrait, PolyEvalInstance, PolyEvalWitness},
  traits::{evaluation::EvaluationEngineTrait, Group, TranscriptReprTrait},
  CommitmentKey,
};
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

/// A trivial implementation of `ComputationCommitmentEngineTrait`
pub struct TrivialCompComputationEngine<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  _p: PhantomData<G>,
  _p2: PhantomData<EE>,
}

/// Provides an implementation of a trivial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TrivialCommitment<G: Group> {
  S: R1CSShape<G>,
}

/// Provides an implementation of a trivial decommitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TrivialDecommitment<G: Group> {
  _p: PhantomData<G>,
}

/// Provides an implementation of a trivial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TrivialEvaluationArgument<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> TranscriptReprTrait<G> for TrivialCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.S.to_transcript_bytes()
  }
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> CompCommitmentEngineTrait<G>
  for TrivialCompComputationEngine<G, EE>
{
  type Decommitment = TrivialDecommitment<G>;
  type Commitment = TrivialCommitment<G>;
  type EvaluationArgument = TrivialEvaluationArgument<G>;

  /// commits to R1CS matrices
  fn commit(
    _ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
  ) -> Result<(Self::Commitment, Self::Decommitment), NovaError> {
    Ok((
      TrivialCommitment { S: S.clone() },
      TrivialDecommitment {
        _p: Default::default(),
      },
    ))
  }

  /// proves an evaluation of R1CS matrices viewed as polynomials
  fn prove(
    _ck: &CommitmentKey<G>,
    _S: &R1CSShape<G>,
    _decomm: &Self::Decommitment,
    _comm: &Self::Commitment,
    _r: &(&[G::Scalar], &[G::Scalar]),
    _transcript: &mut G::TE,
  ) -> Result<
    (
      Self::EvaluationArgument,
      Vec<(PolyEvalWitness<G>, PolyEvalInstance<G>)>,
    ),
    NovaError,
  > {
    Ok((
      TrivialEvaluationArgument {
        _p: Default::default(),
      },
      Vec::new(),
    ))
  }

  /// verifies an evaluation of R1CS matrices viewed as polynomials
  fn verify(
    comm: &Self::Commitment,
    r: &(&[G::Scalar], &[G::Scalar]),
    _arg: &Self::EvaluationArgument,
    _transcript: &mut G::TE,
  ) -> Result<(G::Scalar, G::Scalar, G::Scalar, Vec<PolyEvalInstance<G>>), NovaError> {
    let (r_x, r_y) = r;
    let evals = SparsePolynomial::<G>::multi_evaluate(&[&comm.S.A, &comm.S.B, &comm.S.C], r_x, r_y);
    Ok((evals[0], evals[1], evals[2], Vec::new()))
  }
}

mod product;
mod sparse;

use sparse::{SparseEvaluationArgument, SparsePolynomial, SparsePolynomialCommitment};

/// A non-trivial implementation of `CompCommitmentEngineTrait` using Spartan's SPARK compiler
pub struct SparkEngine<G: Group> {
  _p: PhantomData<G>,
}

/// An implementation of Spark decommitment
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparkDecommitment<G: Group> {
  A: SparsePolynomial<G>,
  B: SparsePolynomial<G>,
  C: SparsePolynomial<G>,
}

impl<G: Group> SparkDecommitment<G> {
  fn new(S: &R1CSShape<G>) -> Self {
    let ell = (S.num_cons.log_2(), S.num_vars.log_2() + 1);
    let A = SparsePolynomial::new(ell, &S.A);
    let B = SparsePolynomial::new(ell, &S.B);
    let C = SparsePolynomial::new(ell, &S.C);

    Self { A, B, C }
  }

  fn commit(&self, ck: &CommitmentKey<G>) -> SparkCommitment<G> {
    let comm_A = self.A.commit(ck);
    let comm_B = self.B.commit(ck);
    let comm_C = self.C.commit(ck);

    SparkCommitment {
      comm_A,
      comm_B,
      comm_C,
    }
  }
}

/// An implementation of Spark commitment
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparkCommitment<G: Group> {
  comm_A: SparsePolynomialCommitment<G>,
  comm_B: SparsePolynomialCommitment<G>,
  comm_C: SparsePolynomialCommitment<G>,
}

impl<G: Group> TranscriptReprTrait<G> for SparkCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let mut bytes = self.comm_A.to_transcript_bytes();
    bytes.extend(self.comm_B.to_transcript_bytes());
    bytes.extend(self.comm_C.to_transcript_bytes());
    bytes
  }
}

/// Provides an implementation of a trivial evaluation argument
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparkEvaluationArgument<G: Group> {
  arg_A: SparseEvaluationArgument<G>,
  arg_B: SparseEvaluationArgument<G>,
  arg_C: SparseEvaluationArgument<G>,
}

impl<G: Group> CompCommitmentEngineTrait<G> for SparkEngine<G> {
  type Decommitment = SparkDecommitment<G>;
  type Commitment = SparkCommitment<G>;
  type EvaluationArgument = SparkEvaluationArgument<G>;

  /// commits to R1CS matrices
  fn commit(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
  ) -> Result<(Self::Commitment, Self::Decommitment), NovaError> {
    let sparse = SparkDecommitment::new(S);
    let comm = sparse.commit(ck);
    Ok((comm, sparse))
  }

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
  > {
    let (arg_A, u_w_vec_A) =
      SparseEvaluationArgument::prove(ck, &decomm.A, &S.A, &comm.comm_A, r, transcript)?;
    let (arg_B, u_w_vec_B) =
      SparseEvaluationArgument::prove(ck, &decomm.B, &S.B, &comm.comm_B, r, transcript)?;
    let (arg_C, u_w_vec_C) =
      SparseEvaluationArgument::prove(ck, &decomm.C, &S.C, &comm.comm_C, r, transcript)?;

    let u_w_vec = {
      let mut u_w_vec = u_w_vec_A;
      u_w_vec.extend(u_w_vec_B);
      u_w_vec.extend(u_w_vec_C);
      u_w_vec
    };

    Ok((
      SparkEvaluationArgument {
        arg_A,
        arg_B,
        arg_C,
      },
      u_w_vec,
    ))
  }

  /// verifies an evaluation of R1CS matrices viewed as polynomials
  fn verify(
    comm: &Self::Commitment,
    r: &(&[G::Scalar], &[G::Scalar]),
    arg: &Self::EvaluationArgument,
    transcript: &mut G::TE,
  ) -> Result<(G::Scalar, G::Scalar, G::Scalar, Vec<PolyEvalInstance<G>>), NovaError> {
    let (eval_A, u_vec_A) = arg.arg_A.verify(&comm.comm_A, r, transcript)?;
    let (eval_B, u_vec_B) = arg.arg_B.verify(&comm.comm_B, r, transcript)?;
    let (eval_C, u_vec_C) = arg.arg_C.verify(&comm.comm_C, r, transcript)?;

    let u_vec = {
      let mut u_vec = u_vec_A;
      u_vec.extend(u_vec_B);
      u_vec.extend(u_vec_C);
      u_vec
    };

    Ok((eval_A, eval_B, eval_C, u_vec))
  }
}
