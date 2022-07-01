//! A collection of traits that define the behavior of a zkSNARK for RelaxedR1CS
use super::{
  errors::NovaError,
  r1cs::{R1CSGens, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::Group,
};

/// A trait that defines the behavior of a zkSNARK's prover key
pub trait ProverKeyTrait<G: Group> {
  /// Produces a new prover's key
  fn new(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self;
}

/// A trait that defines the behavior of a zkSNARK's verifier key
pub trait VerifierKeyTrait<G: Group> {
  /// Produces a new verifier's key
  fn new(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> Self;
}

/// A trait that defines the behavior of a zkSNARK
pub trait RelaxedR1CSSNARKTrait<G: Group>: Sized {
  /// A type that represents the prover's key
  type ProverKey: ProverKeyTrait<G>;

  /// A type that represents the verifier's key
  type VerifierKey: VerifierKeyTrait<G>;

  /// Produces a new SNARK for a relaxed R1CS
  fn prove(
    pk: &Self::ProverKey,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError>;

  /// Verifies a SNARK for a relaxed R1CS
  fn verify(vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError>;
}
