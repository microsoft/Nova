//! This module defines a collection of traits that define the behavior of a polynomial evaluation engine
//! A vector of size N is treated as a multilinear polynomial in \log{N} variables,
//! and a commitment provided by the commitment engine is treated as a multilinear polynomial commitment
use crate::{
  errors::NovaError,
  traits::{commitment::CommitmentEngineTrait, Group},
};
use abomonation::Abomonation;
use serde::{Deserialize, Serialize};

/// A trait that ties different pieces of the commitment evaluation together
pub trait EvaluationEngineTrait<G: Group>:
  Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that holds the associated commitment engine
  type CE: CommitmentEngineTrait<G>;

  /// A type that holds the prover key
  type ProverKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + Abomonation;

  /// A type that holds the verifier key
  type VerifierKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de> + Abomonation;

  /// A type that holds the evaluation argument
  type EvaluationArgument: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A method to perform any additional setup needed to produce proofs of evaluations
  fn setup(
    ck: &<Self::CE as CommitmentEngineTrait<G>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey);

  /// A method to prove the evaluation of a multilinear polynomial
  fn prove(
    ck: &<Self::CE as CommitmentEngineTrait<G>>::CommitmentKey,
    pk: &Self::ProverKey,
    transcript: &mut G::TE,
    comm: &<Self::CE as CommitmentEngineTrait<G>>::Commitment,
    poly: &[G::Scalar],
    point: &[G::Scalar],
    eval: &G::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError>;

  /// A method to verify the purported evaluation of a multilinear polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut G::TE,
    comm: &<Self::CE as CommitmentEngineTrait<G>>::Commitment,
    point: &[G::Scalar],
    eval: &G::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError>;
}
