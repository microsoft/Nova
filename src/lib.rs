//! This library implements core components of Nova.
#![allow(non_snake_case)]
#![allow(clippy::type_complexity)]
#![deny(missing_docs)]

pub mod bellperson;
mod circuit;
mod commitments;
mod constants;
pub mod errors;
pub mod gadgets;
pub mod nifs;
pub mod pasta;
mod poseidon;
pub mod r1cs;
pub mod traits;

use errors::NovaError;
use r1cs::{R1CSGens, R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness};
use traits::Group;

/// A SNARK that proves the knowledge of a valid `RecursiveSNARK`
pub struct CompressedSNARKTrivial<G: Group> {
  W: RelaxedR1CSWitness<G>,
}

impl<G: Group> CompressedSNARKTrivial<G> {
  /// Produces a proof of a instance given its satisfying witness `W`.
  pub fn prove(W: &RelaxedR1CSWitness<G>) -> Result<CompressedSNARKTrivial<G>, NovaError> {
    Ok(Self { W: W.clone() })
  }

  /// Verifies the proof of a folded instance `U` given its shape `S` public parameters `gens`
  pub fn verify(
    &self,
    gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    U: &RelaxedR1CSInstance<G>,
  ) -> Result<(), NovaError> {
    // check that the witness is a valid witness to the folded instance `U`
    S.is_sat_relaxed(gens, U, &self.W)
  }
}
