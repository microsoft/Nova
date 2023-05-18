//! This module implements `CompCommitmentEngineTrait` using Spartan's SPARK compiler
//! We also provide a trivial implementation that has the verifier evaluate the sparse polynomials
use crate::{
  errors::NovaError,
  r1cs::R1CSShape,
  spartan::{math::Math, CompCommitmentEngineTrait},
  traits::{evaluation::EvaluationEngineTrait, Group, TranscriptReprTrait},
  CommitmentKey,
};
use core::marker::PhantomData;
use abomonation::Abomonation;
use abomonation_derive::Abomonation;
use serde::{Deserialize, Serialize};

/// A trivial implementation of `ComputationCommitmentEngineTrait`
#[derive(Clone, Debug)]
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

impl<G: Group> Abomonation for TrivialCommitment<G> {
    unsafe fn entomb<W: std::io::Write>(&self, _write: &mut W) -> std::io::Result<()> { Ok(()) }

    unsafe fn exhume<'a,'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> { Some(bytes) }

    fn extent(&self) -> usize { 0 }
}

/// Provides an implementation of a trivial decommitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TrivialDecommitment<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> Abomonation for TrivialDecommitment<G> {
    unsafe fn entomb<W: std::io::Write>(&self, _write: &mut W) -> std::io::Result<()> { Ok(()) }

    unsafe fn exhume<'a,'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> { Some(bytes) }

    fn extent(&self) -> usize { 0 }
}

/// Provides an implementation of a trivial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct TrivialEvaluationArgument<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> Abomonation for TrivialEvaluationArgument<G> {
    unsafe fn entomb<W: std::io::Write>(&self, _write: &mut W) -> std::io::Result<()> { Ok(()) }

    unsafe fn exhume<'a,'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> { Some(bytes) }

    fn extent(&self) -> usize { 0 }
}

impl<G: Group> TranscriptReprTrait<G> for TrivialCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.S.to_transcript_bytes()
  }
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> CompCommitmentEngineTrait<G, EE>
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
    _ek: &EE::ProverKey,
    _S: &R1CSShape<G>,
    _decomm: &Self::Decommitment,
    _comm: &Self::Commitment,
    _r: &(&[G::Scalar], &[G::Scalar]),
    _transcript: &mut G::TE,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    Ok(TrivialEvaluationArgument {
      _p: Default::default(),
    })
  }

  /// verifies an evaluation of R1CS matrices viewed as polynomials
  fn verify(
    _vk: &EE::VerifierKey,
    comm: &Self::Commitment,
    r: &(&[G::Scalar], &[G::Scalar]),
    _arg: &Self::EvaluationArgument,
    _transcript: &mut G::TE,
  ) -> Result<(G::Scalar, G::Scalar, G::Scalar), NovaError> {
    let (r_x, r_y) = r;
    let evals = SparsePolynomial::<G>::multi_evaluate(&[&comm.S.A, &comm.S.B, &comm.S.C], r_x, r_y);
    Ok((evals[0], evals[1], evals[2]))
  }
}

mod product;
mod sparse;

use sparse::{SparseEvaluationArgument, SparsePolynomial, SparsePolynomialCommitment};

/// A non-trivial implementation of `CompCommitmentEngineTrait` using Spartan's SPARK compiler
pub struct SparkEngine<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  _p: PhantomData<G>,
  _p2: PhantomData<EE>,
}

/// An implementation of Spark decommitment
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparkDecommitment<G: Group> {
  A: SparsePolynomial<G>,
  B: SparsePolynomial<G>,
  C: SparsePolynomial<G>,
}

impl<G: Group> Abomonation for SparkDecommitment<G> {
  #[inline]
  unsafe fn entomb<W: std::io::Write>(&self, bytes: &mut W) -> std::io::Result<()> {
    self.A.entomb(bytes)?;
    self.B.entomb(bytes)?;
    self.C.entomb(bytes)?;
    Ok(())
  }

  #[inline]
  unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
    let temp = bytes;
    bytes = self.A.exhume(temp)?;
    let temp = bytes;
    bytes = self.B.exhume(temp)?;
    let temp = bytes;
    bytes = self.C.exhume(temp)?;
    Some(bytes)
  }

  #[inline]
  fn extent(&self) -> usize {
    let mut size = 0;
    size += self.A.extent();
    size += self.B.extent();
    size += self.C.extent();
    size
  }
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

impl<G: Group> Abomonation for SparkCommitment<G> {
  #[inline]
  unsafe fn entomb<W: std::io::Write>(&self, bytes: &mut W) -> std::io::Result<()> {
    self.comm_A.entomb(bytes)?;
    self.comm_B.entomb(bytes)?;
    self.comm_C.entomb(bytes)?;
    Ok(())
  }

  #[inline]
  unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
    let temp = bytes;
    bytes = self.comm_A.exhume(temp)?;
    let temp = bytes;
    bytes = self.comm_B.exhume(temp)?;
    let temp = bytes;
    bytes = self.comm_C.exhume(temp)?;
    Some(bytes)
  }

  #[inline]
  fn extent(&self) -> usize {
    let mut size = 0;
    size += self.comm_A.extent();
    size += self.comm_B.extent();
    size += self.comm_C.extent();
    size
  }
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
pub struct SparkEvaluationArgument<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  arg_A: SparseEvaluationArgument<G, EE>,
  arg_B: SparseEvaluationArgument<G, EE>,
  arg_C: SparseEvaluationArgument<G, EE>,
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> CompCommitmentEngineTrait<G, EE>
  for SparkEngine<G, EE>
{
  type Decommitment = SparkDecommitment<G>;
  type Commitment = SparkCommitment<G>;
  type EvaluationArgument = SparkEvaluationArgument<G, EE>;

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
    pk_ee: &EE::ProverKey,
    S: &R1CSShape<G>,
    decomm: &Self::Decommitment,
    comm: &Self::Commitment,
    r: &(&[G::Scalar], &[G::Scalar]),
    transcript: &mut G::TE,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let arg_A =
      SparseEvaluationArgument::prove(ck, pk_ee, &decomm.A, &S.A, &comm.comm_A, r, transcript)?;
    let arg_B =
      SparseEvaluationArgument::prove(ck, pk_ee, &decomm.B, &S.B, &comm.comm_B, r, transcript)?;
    let arg_C =
      SparseEvaluationArgument::prove(ck, pk_ee, &decomm.C, &S.C, &comm.comm_C, r, transcript)?;

    Ok(SparkEvaluationArgument {
      arg_A,
      arg_B,
      arg_C,
    })
  }

  /// verifies an evaluation of R1CS matrices viewed as polynomials
  fn verify(
    vk_ee: &EE::VerifierKey,
    comm: &Self::Commitment,
    r: &(&[G::Scalar], &[G::Scalar]),
    arg: &Self::EvaluationArgument,
    transcript: &mut G::TE,
  ) -> Result<(G::Scalar, G::Scalar, G::Scalar), NovaError> {
    let eval_A = arg.arg_A.verify(vk_ee, &comm.comm_A, r, transcript)?;
    let eval_B = arg.arg_B.verify(vk_ee, &comm.comm_B, r, transcript)?;
    let eval_C = arg.arg_C.verify(vk_ee, &comm.comm_C, r, transcript)?;

    Ok((eval_A, eval_B, eval_C))
  }
}
