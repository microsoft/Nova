//! This module defines a collection of traits that define the behavior of a polynomial evaluation engine
//! A vector of size N is treated as a multilinear polynomial in \log{N} variables,
//! and a commitment provided by the commitment engine is treated as a multilinear polynomial commitment
use crate::{
  errors::NovaError,
  traits::{commitment::CommitmentEngineTrait, Engine},
};
use serde::{Deserialize, Serialize};

/// A trait that ties different pieces of the commitment evaluation together
pub trait EvaluationEngineTrait<E: Engine>: Clone + Send + Sync {
  /// A type that holds the prover key
  type ProverKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the verifier key
  type VerifierKey: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that holds the evaluation argument
  type EvaluationArgument: Clone + Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A method to perform any additional setup needed to produce proofs of evaluations
  fn setup(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey);

  /// A method to prove the evaluation of a multilinear polynomial
  fn prove(
    ck: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
    pk: &Self::ProverKey,
    transcript: &mut E::TE,
    comm: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    poly: &[E::Scalar],
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError>;

  /// A method to verify the purported evaluation of a multilinear polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut E::TE,
    comm: &<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
    point: &[E::Scalar],
    eval: &E::Scalar,
    arg: &Self::EvaluationArgument,
  ) -> Result<(), NovaError>;
}
