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

use crate::bellperson::{r1cs::NovaShape, shape_cs::ShapeCS};
use ::bellperson::Circuit;
use circuit::{NIFSVerifierCircuit, NIFSVerifierCircuitParams};
use constants::{BN_LIMB_WIDTH, BN_N_LIMBS};
use errors::NovaError;
use r1cs::{
  R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
};
use traits::{Group, HashFuncConstantsTrait, HashFuncTrait, StepCircuit};

type ROConstants<G> =
  <<G as Group>::HashFunc as HashFuncTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants;

/// A type that holds public parameters of Nova
pub struct PublicParams<G1: Group, G2: Group> {
  ro_consts_primary: ROConstants<G1>,
  r1cs_gens_primary: R1CSGens<G1>,
  ro_consts_secondary: ROConstants<G2>,
  r1cs_gens_secondary: R1CSGens<G2>,
}

impl<G1: Group, G2: Group> PublicParams<G1, G2> {
  /// Create a new `PublicParams`
  pub fn new<SC1: StepCircuit<G2::Base>, SC2: StepCircuit<G1::Base>>(
    sc_primary: SC1,
    sc_secondary: SC2,
  ) -> Result<Self, NovaError> {
    let params_primary = NIFSVerifierCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params_secondary = NIFSVerifierCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);

    let ro_consts_primary: ROConstants<G1> = ROConstants::<G1>::new();
    let ro_consts_secondary: ROConstants<G2> = ROConstants::<G2>::new();

    // Initialize the shape and gens for the primary
    let circuit_primary: NIFSVerifierCircuit<G2, SC1> =
      NIFSVerifierCircuit::new(params_primary, None, sc_primary, ro_consts_primary);
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (shape_primary, r1cs_gens_primary) = (cs.r1cs_shape(), cs.r1cs_gens());
    println!(
      "circuit_primary -> Number of constraints: {}",
      cs.num_constraints()
    );

    // Initialize the shape and gens for the secondary
    let circuit_secondary: NIFSVerifierCircuit<G1, SC2> =
      NIFSVerifierCircuit::new(params_secondary, None, sc_secondary, ro_consts_secondary);
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let (shape_secondary, r1cs_gens_secondary) = (cs.r1cs_shape(), cs.r1cs_gens());
    println!(
      "circuit_secondary -> Number of constraints: {}",
      cs.num_constraints()
    );

    Self {
      ro_consts_primary,
      r1cs_gens_primary,
      ro_consts_secondary,
      r1cs_gens_secondary,
    }
  }
}

/// A SNARK that proves the correct execution of an incremental computation
pub struct RecursiveSNARK<G1: Group, G2: Group> {
  r_W_primary: RelaxedR1CSWitness<G1>,
  r_U_primary: RelaxedR1CSInstance<G1>,
  l_W_primary: RelaxedR1CSWitness<G1>,
  l_u_primary: R1CSInstance<G1>,
  r_W_secondary: RelaxedR1CSWitness<G2>,
  r_U_secondary: RelaxedR1CSInstance<G2>,
  l_W_secondary: R1CSWitness<G2>,
  l_u_secondary: R1CSInstance<G2>,
}

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
