//! This module defines a collection of traits that define the behavior of a zkSNARK for RelaxedR1CS
use crate::{
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::Group,
  CommitmentKey,
};

use serde::{Deserialize, Serialize};

/// A trait that defines the behavior of a zkSNARK
pub trait RelaxedR1CSSNARKTrait<G: Group>:
  Sized + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
  /// A type that represents the prover's key
  type ProverKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// A type that represents the verifier's key
  type VerifierKey: Send + Sync + Serialize + for<'de> Deserialize<'de>;

  /// Produces the keys for the prover and the verifier
  fn setup(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError>;

  /// Produces a new SNARK for a relaxed R1CS
  fn prove(
    ck: &CommitmentKey<G>,
    pk: &Self::ProverKey,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError>;

  /// Verifies a SNARK for a relaxed R1CS
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError>;
}
