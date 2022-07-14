//! This module defines a collection of traits that define the behavior of a zkSNARK for RelaxedR1CS
use crate::{
  errors::NovaError,
  r1cs::{R1CSGens, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::Group,
};

/// A trait that defines the behavior of a zkSNARK's prover key
pub trait ProverKeyTrait<G: Group>: Send + Sync {
  /// Produces a new prover's key
  fn new(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self;
}

/// A trait that defines the behavior of a zkSNARK's verifier key
pub trait VerifierKeyTrait<G: Group>: Send + Sync {
  /// Produces a new verifier's key
  fn new(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self;
}

/// A trait that defines the behavior of a zkSNARK
pub trait RelaxedR1CSSNARKTrait<G: Group>: Sized + Send + Sync {
  /// A type that represents the prover's key
  type ProverKey: ProverKeyTrait<G>;

  /// A type that represents the verifier's key
  type VerifierKey: VerifierKeyTrait<G>;

  /// Produces a prover key
  fn prover_key(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self::ProverKey {
    Self::ProverKey::new(gens, S)
  }

  /// Produces a verifier key
  fn verifier_key(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self::VerifierKey {
    Self::VerifierKey::new(gens, S)
  }

  /// Produces a new SNARK for a relaxed R1CS
  fn prove(
    pk: &Self::ProverKey,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError>;

  /// Verifies a SNARK for a relaxed R1CS
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError>;
}
