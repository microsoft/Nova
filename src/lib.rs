//! This library implements Nova, a high-speed recursive SNARK.
#![deny(
  warnings,
  unused,
  future_incompatible,
  nonstandard_style,
  rust_2018_idioms,
  missing_docs
)]
#![allow(non_snake_case)]
#![forbid(unsafe_code)]

// private modules
mod bellpepper;
mod circuit;
mod constants;
mod digest;
mod nifs;
mod prover;
mod r1cs;

// public modules
pub mod errors;
pub mod gadgets;
pub mod provider;
pub mod public_params;
pub mod spartan;
pub mod traits;

use crate::bellpepper::{r1cs::NovaWitness, solver::SatisfyingAssignment};
use bellpepper_core::ConstraintSystem;
use circuit::{NovaAugmentedCircuit, NovaAugmentedCircuitInputs};
use constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS};
use core::marker::PhantomData;
use errors::NovaError;
use ff::Field;
use gadgets::utils::scalar_as_base;
use nifs::NIFS;
use prover::ProverKey;
use public_params::PublicParams;
use r1cs::{R1CSInstance, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness};
use serde::{Deserialize, Serialize};
use traits::{
  circuit::StepCircuit,
  commitment::{Commitment, CommitmentTrait},
  snark::RelaxedR1CSSNARKTrait,
  AbsorbInROTrait, Engine, ROConstants, ROTrait,
};

/// A SNARK that proves the correct execution of an incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E1, E2, C1, C2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
{
  z0_primary: Vec<E1::Scalar>,
  z0_secondary: Vec<E2::Scalar>,
  r_W_primary: RelaxedR1CSWitness<E1>,
  r_U_primary: RelaxedR1CSInstance<E1>,
  r_W_secondary: RelaxedR1CSWitness<E2>,
  r_U_secondary: RelaxedR1CSInstance<E2>,
  l_w_secondary: R1CSWitness<E2>,
  l_u_secondary: R1CSInstance<E2>,
  i: usize,
  zi_primary: Vec<E1::Scalar>,
  zi_secondary: Vec<E2::Scalar>,
  _p: PhantomData<(C1, C2)>,
}

impl<E1, E2, C1, C2> RecursiveSNARK<E1, E2, C1, C2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
{
  /// Create new instance of recursive SNARK
  pub fn new(
    pp: &PublicParams<E1, E2, C1, C2>,
    c_primary: &C1,
    c_secondary: &C2,
    z0_primary: &[E1::Scalar],
    z0_secondary: &[E2::Scalar],
  ) -> Result<Self, NovaError> {
    if z0_primary.len() != pp.F_arity_primary || z0_secondary.len() != pp.F_arity_secondary {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // base case for the primary
    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let inputs_primary: NovaAugmentedCircuitInputs<E2> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      E1::Scalar::ZERO,
      z0_primary.to_vec(),
      None,
      None,
      None,
      None,
    );

    let circuit_primary: NovaAugmentedCircuit<'_, E2, C1> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_primary,
      Some(inputs_primary),
      c_primary,
      pp.ro_consts_circuit_primary.clone(),
    );
    let zi_primary = circuit_primary
      .synthesize(&mut cs_primary)
      .map_err(|_| NovaError::SynthesisError)
      .expect("Nova error synthesis");
    let (u_primary, w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)
      .map_err(|_e| NovaError::UnSat)
      .expect("Nova error unsat");

    // base case for the secondary
    let mut cs_secondary = SatisfyingAssignment::<E2>::new();
    let inputs_secondary: NovaAugmentedCircuitInputs<E1> = NovaAugmentedCircuitInputs::new(
      pp.digest(),
      E2::Scalar::ZERO,
      z0_secondary.to_vec(),
      None,
      None,
      Some(u_primary.clone()),
      None,
    );
    let circuit_secondary: NovaAugmentedCircuit<'_, E1, C2> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_secondary,
      Some(inputs_secondary),
      c_secondary,
      pp.ro_consts_circuit_secondary.clone(),
    );
    let zi_secondary = circuit_secondary
      .synthesize(&mut cs_secondary)
      .map_err(|_| NovaError::SynthesisError)
      .expect("Nova error synthesis");
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)
      .map_err(|_e| NovaError::UnSat)
      .expect("Nova error unsat");

    // IVC proof for the primary circuit
    let l_w_primary = w_primary;
    let l_u_primary = u_primary;
    let r_W_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &l_w_primary);
    let r_U_primary =
      RelaxedR1CSInstance::from_r1cs_instance(&pp.ck_primary, &pp.r1cs_shape_primary, &l_u_primary);

    // IVC proof for the secondary circuit
    let l_w_secondary = w_secondary;
    let l_u_secondary = u_secondary;
    let r_W_secondary = RelaxedR1CSWitness::<E2>::default(&pp.r1cs_shape_secondary);
    let r_U_secondary =
      RelaxedR1CSInstance::<E2>::default(&pp.ck_secondary, &pp.r1cs_shape_secondary);

    assert!(
      !(zi_primary.len() != pp.F_arity_primary || zi_secondary.len() != pp.F_arity_secondary),
      "Invalid step length"
    );

    let zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(NovaError::SynthesisError))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, NovaError>>()
      .expect("Nova error synthesis");

    let zi_secondary = zi_secondary
      .iter()
      .map(|v| v.get_value().ok_or(NovaError::SynthesisError))
      .collect::<Result<Vec<<E2 as Engine>::Scalar>, NovaError>>()
      .expect("Nova error synthesis");

    Ok(Self {
      z0_primary: z0_primary.to_vec(),
      z0_secondary: z0_secondary.to_vec(),
      r_W_primary,
      r_U_primary,
      r_W_secondary,
      r_U_secondary,
      l_w_secondary,
      l_u_secondary,
      i: 0,
      zi_primary,
      zi_secondary,
      _p: Default::default(),
    })
  }

  /// Create a new `RecursiveSNARK` (or updates the provided `RecursiveSNARK`)
  /// by executing a step of the incremental computation
  pub fn prove_step(
    &mut self,
    pp: &PublicParams<E1, E2, C1, C2>,
    c_primary: &C1,
    c_secondary: &C2,
  ) -> Result<(), NovaError> {
    // first step was already done in the constructor
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // fold the secondary circuit's instance
    let (nifs_secondary, (r_U_secondary, r_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.r1cs_shape_secondary,
      &self.r_U_secondary,
      &self.r_W_secondary,
      &self.l_u_secondary,
      &self.l_w_secondary,
    )
    .expect("Unable to fold secondary");

    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let inputs_primary: NovaAugmentedCircuitInputs<E2> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      E1::Scalar::from(self.i as u64),
      self.z0_primary.to_vec(),
      Some(self.zi_primary.clone()),
      Some(self.r_U_secondary.clone()),
      Some(self.l_u_secondary.clone()),
      Some(Commitment::<E2>::decompress(&nifs_secondary.comm_T)?),
    );

    let circuit_primary: NovaAugmentedCircuit<'_, E2, C1> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_primary,
      Some(inputs_primary),
      c_primary,
      pp.ro_consts_circuit_primary.clone(),
    );
    let zi_primary = circuit_primary
      .synthesize(&mut cs_primary)
      .map_err(|_| NovaError::SynthesisError)?;

    let (l_u_primary, l_w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)
      .map_err(|_e| NovaError::UnSat)
      .expect("Nova error unsat");

    // fold the primary circuit's instance
    let (nifs_primary, (r_U_primary, r_W_primary)) = NIFS::prove(
      &pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.digest(),
      &pp.r1cs_shape_primary,
      &self.r_U_primary,
      &self.r_W_primary,
      &l_u_primary,
      &l_w_primary,
    )
    .expect("Unable to fold primary");

    let mut cs_secondary = SatisfyingAssignment::<E2>::new();
    let inputs_secondary: NovaAugmentedCircuitInputs<E1> = NovaAugmentedCircuitInputs::new(
      pp.digest(),
      E2::Scalar::from(self.i as u64),
      self.z0_secondary.to_vec(),
      Some(self.zi_secondary.clone()),
      Some(self.r_U_primary.clone()),
      Some(l_u_primary),
      Some(Commitment::<E1>::decompress(&nifs_primary.comm_T)?),
    );

    let circuit_secondary: NovaAugmentedCircuit<'_, E1, C2> = NovaAugmentedCircuit::new(
      &pp.augmented_circuit_params_secondary,
      Some(inputs_secondary),
      c_secondary,
      pp.ro_consts_circuit_secondary.clone(),
    );
    let zi_secondary = circuit_secondary
      .synthesize(&mut cs_secondary)
      .map_err(|_| NovaError::SynthesisError)?;

    let (l_u_secondary, l_w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // update the running instances and witnesses
    self.zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(NovaError::SynthesisError))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, NovaError>>()?;
    self.zi_secondary = zi_secondary
      .iter()
      .map(|v| v.get_value().ok_or(NovaError::SynthesisError))
      .collect::<Result<Vec<<E2 as Engine>::Scalar>, NovaError>>()?;

    self.l_u_secondary = l_u_secondary;
    self.l_w_secondary = l_w_secondary;

    self.r_U_primary = r_U_primary;
    self.r_W_primary = r_W_primary;

    self.i += 1;

    self.r_U_secondary = r_U_secondary;
    self.r_W_secondary = r_W_secondary;

    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<E1, E2, C1, C2>,
    num_steps: usize,
    z0_primary: &[E1::Scalar],
    z0_secondary: &[E2::Scalar],
  ) -> Result<(Vec<E1::Scalar>, Vec<E2::Scalar>), NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;

    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;

    // check if the initial inputs match
    let is_inputs_not_match = self.z0_primary != z0_primary || self.z0_secondary != z0_secondary;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outpus = self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2;

    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outpus
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <E2 as Engine>::RO::new(
        pp.ro_consts_secondary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * pp.F_arity_primary,
      );
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zi_primary {
        hasher.absorb(*e);
      }
      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 = <E1 as Engine>::RO::new(
        pp.ro_consts_primary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * pp.F_arity_secondary,
      );
      hasher2.absorb(scalar_as_base::<E1>(pp.digest()));
      hasher2.absorb(E2::Scalar::from(num_steps as u64));
      for e in z0_secondary {
        hasher2.absorb(*e);
      }
      for e in &self.zi_secondary {
        hasher2.absorb(*e);
      }
      self.r_U_primary.absorb_in_ro(&mut hasher2);

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0]
      || hash_secondary != scalar_as_base::<E2>(self.l_u_secondary.X[1])
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check the satisfiability of the provided instances
    let (res_r_primary, (res_r_secondary, res_l_secondary)) = rayon::join(
      || {
        pp.r1cs_shape_primary
          .is_sat_relaxed(&pp.ck_primary, &self.r_U_primary, &self.r_W_primary)
      },
      || {
        rayon::join(
          || {
            pp.r1cs_shape_secondary.is_sat_relaxed(
              &pp.ck_secondary,
              &self.r_U_secondary,
              &self.r_W_secondary,
            )
          },
          || {
            pp.r1cs_shape_secondary.is_sat(
              &pp.ck_secondary,
              &self.l_u_secondary,
              &self.l_w_secondary,
            )
          },
        )
      },
    );

    // check the returned res objects
    res_r_primary?;
    res_r_secondary?;
    res_l_secondary?;

    Ok((self.zi_primary.clone(), self.zi_secondary.clone()))
  }
}

/// A type that holds the verifier key for `CompressedSNARK`
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  F_arity_primary: usize,
  F_arity_secondary: usize,
  ro_consts_primary: ROConstants<E1>,
  ro_consts_secondary: ROConstants<E2>,
  pp_digest: E1::Scalar,
  vk_primary: S1::VerifierKey,
  vk_secondary: S2::VerifierKey,
  _p: PhantomData<(C1, C2)>,
}

/// A SNARK that proves the knowledge of a valid `RecursiveSNARK`
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  r_U_primary: RelaxedR1CSInstance<E1>,
  r_W_snark_primary: S1,

  r_U_secondary: RelaxedR1CSInstance<E2>,
  l_u_secondary: R1CSInstance<E2>,
  nifs_secondary: NIFS<E2>,
  f_W_snark_secondary: S2,

  zn_primary: Vec<E1::Scalar>,
  zn_secondary: Vec<E2::Scalar>,

  _p: PhantomData<(C1, C2)>,
}

impl<E1, E2, C1, C2, S1, S2> CompressedSNARK<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  /// Creates prover and verifier keys for `CompressedSNARK`
  pub fn setup(
    pp: &PublicParams<E1, E2, C1, C2>,
  ) -> Result<
    (
      ProverKey<E1, E2, C1, C2, S1, S2>,
      VerifierKey<E1, E2, C1, C2, S1, S2>,
    ),
    NovaError,
  > {
    let (pk_primary, vk_primary) = S1::setup(&pp.ck_primary, &pp.r1cs_shape_primary)?;
    let (pk_secondary, vk_secondary) = S2::setup(&pp.ck_secondary, &pp.r1cs_shape_secondary)?;

    let pk = ProverKey {
      pk_primary,
      pk_secondary,
      _p: Default::default(),
    };

    let vk = VerifierKey {
      F_arity_primary: pp.F_arity_primary,
      F_arity_secondary: pp.F_arity_secondary,
      ro_consts_primary: pp.ro_consts_primary.clone(),
      ro_consts_secondary: pp.ro_consts_secondary.clone(),
      pp_digest: pp.digest(),
      vk_primary,
      vk_secondary,
      _p: Default::default(),
    };

    Ok((pk, vk))
  }

  /// Create a new `CompressedSNARK`
  pub fn prove(
    pp: &PublicParams<E1, E2, C1, C2>,
    pk: &ProverKey<E1, E2, C1, C2, S1, S2>,
    recursive_snark: &RecursiveSNARK<E1, E2, C1, C2>,
  ) -> Result<Self, NovaError> {
    // fold the secondary circuit's instance with its running instance
    let (nifs_secondary, (f_U_secondary, f_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.r1cs_shape_secondary,
      &recursive_snark.r_U_secondary,
      &recursive_snark.r_W_secondary,
      &recursive_snark.l_u_secondary,
      &recursive_snark.l_w_secondary,
    )?;

    // create SNARKs proving the knowledge of f_W_primary and f_W_secondary
    let (r_W_snark_primary, f_W_snark_secondary) = rayon::join(
      || {
        S1::prove(
          &pp.ck_primary,
          &pk.pk_primary,
          &pp.r1cs_shape_primary,
          &recursive_snark.r_U_primary,
          &recursive_snark.r_W_primary,
        )
      },
      || {
        S2::prove(
          &pp.ck_secondary,
          &pk.pk_secondary,
          &pp.r1cs_shape_secondary,
          &f_U_secondary,
          &f_W_secondary,
        )
      },
    );

    Ok(Self {
      r_U_primary: recursive_snark.r_U_primary.clone(),
      r_W_snark_primary: r_W_snark_primary?,

      r_U_secondary: recursive_snark.r_U_secondary.clone(),
      l_u_secondary: recursive_snark.l_u_secondary.clone(),
      nifs_secondary,
      f_W_snark_secondary: f_W_snark_secondary?,

      zn_primary: recursive_snark.zi_primary.clone(),
      zn_secondary: recursive_snark.zi_secondary.clone(),

      _p: Default::default(),
    })
  }

  /// Verify the correctness of the `CompressedSNARK`
  pub fn verify(
    &self,
    vk: &VerifierKey<E1, E2, C1, C2, S1, S2>,
    num_steps: usize,
    z0_primary: &[E1::Scalar],
    z0_secondary: &[E2::Scalar],
  ) -> Result<(Vec<E1::Scalar>, Vec<E2::Scalar>), NovaError> {
    // the number of steps cannot be zero
    if num_steps == 0 {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the (relaxed) R1CS instances have two public outputs
    if self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <E2 as Engine>::RO::new(
        vk.ro_consts_secondary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * vk.F_arity_primary,
      );
      hasher.absorb(vk.pp_digest);
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zn_primary {
        hasher.absorb(*e);
      }
      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 = <E1 as Engine>::RO::new(
        vk.ro_consts_primary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * vk.F_arity_secondary,
      );
      hasher2.absorb(scalar_as_base::<E1>(vk.pp_digest));
      hasher2.absorb(E2::Scalar::from(num_steps as u64));
      for e in z0_secondary {
        hasher2.absorb(*e);
      }
      for e in &self.zn_secondary {
        hasher2.absorb(*e);
      }
      self.r_U_primary.absorb_in_ro(&mut hasher2);

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0]
      || hash_secondary != scalar_as_base::<E2>(self.l_u_secondary.X[1])
    {
      return Err(NovaError::ProofVerifyError);
    }

    // fold the secondary's running instance with the last instance to get a folded instance
    let f_U_secondary = self.nifs_secondary.verify(
      &vk.ro_consts_secondary,
      &scalar_as_base::<E1>(vk.pp_digest),
      &self.r_U_secondary,
      &self.l_u_secondary,
    )?;

    // check the satisfiability of the folded instances using
    // SNARKs proving the knowledge of their satisfying witnesses
    let (res_primary, res_secondary) = rayon::join(
      || {
        self
          .r_W_snark_primary
          .verify(&vk.vk_primary, &self.r_U_primary)
      },
      || {
        self
          .f_W_snark_secondary
          .verify(&vk.vk_secondary, &f_U_secondary)
      },
    );

    res_primary?;
    res_secondary?;

    Ok((self.zn_primary.clone(), self.zn_secondary.clone()))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::{
      pedersen::CommitmentKeyExtTrait, traits::DlogGroup, Bn256Engine, GrumpkinEngine,
      PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::{evaluation::EvaluationEngineTrait, snark::default_ck_hint},
  };
  use ::bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use core::{fmt::Write, marker::PhantomData};
  use ff::PrimeField;
  use traits::{circuit::TrivialCircuit, commitment::CommitmentEngineTrait};

  type EE<E> = provider::ipa_pc::EvaluationEngine<E>;
  type S<E, EE> = spartan::snark::RelaxedR1CSSNARK<E, EE>;
  type SPrime<E, EE> = spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for CubicCircuit<F> {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      Ok(vec![y])
    }
  }

  impl<F: PrimeField> CubicCircuit<F> {
    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }

  fn test_pp_digest_with<E1, E2, T1, T2>(circuit1: &T1, circuit2: &T2, expected: &str)
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    E1::GE: DlogGroup,
    E2::GE: DlogGroup,
    T1: StepCircuit<E1::Scalar>,
    T2: StepCircuit<E2::Scalar>,
    // required to use the IPA in the initialization of the commitment key hints below
    <E1::CE as CommitmentEngineTrait<E1>>::CommitmentKey: CommitmentKeyExtTrait<E1>,
    <E2::CE as CommitmentEngineTrait<E2>>::CommitmentKey: CommitmentKeyExtTrait<E2>,
  {
    // this tests public parameters with a size specifically intended for a spark-compressed SNARK
    let ck_hint1 = &*SPrime::<E1, EE<E1>>::ck_floor();
    let ck_hint2 = &*SPrime::<E2, EE<E2>>::ck_floor();
    let pp = PublicParams::<E1, E2, T1, T2>::setup(circuit1, circuit2, ck_hint1, ck_hint2);

    let digest_str = pp
      .digest()
      .to_repr()
      .as_ref()
      .iter()
      .fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02x}");
        output
      });
    assert_eq!(digest_str, expected);
  }

  #[test]
  fn test_pp_digest() {
    let trivial_circuit1 = TrivialCircuit::<<PallasEngine as Engine>::Scalar>::default();
    let trivial_circuit2 = TrivialCircuit::<<VestaEngine as Engine>::Scalar>::default();
    let cubic_circuit1 = CubicCircuit::<<PallasEngine as Engine>::Scalar>::default();

    test_pp_digest_with::<PallasEngine, VestaEngine, _, _>(
      &trivial_circuit1,
      &trivial_circuit2,
      "cb581e2d5c4b2ef2ddbe2d6849e0da810352f59bcdaca51476dcf9e16072f100",
    );

    test_pp_digest_with::<PallasEngine, VestaEngine, _, _>(
      &cubic_circuit1,
      &trivial_circuit2,
      "3cc29bb864910463e0501bac84cdefc1d4327e9c2ef5b0fd6d45ad1741f1a401",
    );

    let trivial_circuit1_grumpkin = TrivialCircuit::<<Bn256Engine as Engine>::Scalar>::default();
    let trivial_circuit2_grumpkin = TrivialCircuit::<<GrumpkinEngine as Engine>::Scalar>::default();
    let cubic_circuit1_grumpkin = CubicCircuit::<<Bn256Engine as Engine>::Scalar>::default();

    #[cfg(feature = "asm")]
    test_pp_digest_with::<Bn256Engine, GrumpkinEngine, _, _>(
      &trivial_circuit1_grumpkin,
      &trivial_circuit2_grumpkin,
      "c4ecd363a6c1473de7e0d24fc1dbb660f563556e2e13fb4614acdff04cab7701",
    );
    #[cfg(feature = "asm")]
    test_pp_digest_with::<Bn256Engine, GrumpkinEngine, _, _>(
      &cubic_circuit1_grumpkin,
      &trivial_circuit2_grumpkin,
      "4853a6463b6309f6ae76442934d0a423f51f1e10abaddd0d39bf5644ed589100",
    );
    #[cfg(not(feature = "asm"))]
    test_pp_digest_with::<Bn256Engine, GrumpkinEngine, _, _>(
      &trivial_circuit1_grumpkin,
      &trivial_circuit2_grumpkin,
      "c26cc841d42c19bf98bc2482e66cd30903922f2a923927b85d66f375a821f101",
    );
    #[cfg(not(feature = "asm"))]
    test_pp_digest_with::<Bn256Engine, GrumpkinEngine, _, _>(
      &cubic_circuit1_grumpkin,
      &trivial_circuit2_grumpkin,
      "4c484cab71e93dda69b420beb7276af969c2034a7ffb0ea8e6964e96a7e5a901",
    );

    let trivial_circuit1_secp = TrivialCircuit::<<Secp256k1Engine as Engine>::Scalar>::default();
    let trivial_circuit2_secp = TrivialCircuit::<<Secq256k1Engine as Engine>::Scalar>::default();
    let cubic_circuit1_secp = CubicCircuit::<<Secp256k1Engine as Engine>::Scalar>::default();

    test_pp_digest_with::<Secp256k1Engine, Secq256k1Engine, _, _>(
      &trivial_circuit1_secp,
      &trivial_circuit2_secp,
      "b794d655fb39891eaf530ca3be1ec2a5ac97f72a0d07c45dbb84529d8a611502",
    );
    test_pp_digest_with::<Secp256k1Engine, Secq256k1Engine, _, _>(
      &cubic_circuit1_secp,
      &trivial_circuit2_secp,
      "50e6acf363c31c2ac1c9c646b4494cb21aae6cb648c7b0d4c95015c811fba302",
    );
  }

  fn test_ivc_trivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
    let test_circuit2 = TrivialCircuit::<<E2 as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      TrivialCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &test_circuit1,
      &test_circuit2,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::new(
      &pp,
      &test_circuit1,
      &test_circuit2,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<E2 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    let res = recursive_snark.prove_step(&pp, &test_circuit1, &test_circuit2);

    assert!(res.is_ok());

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_trivial() {
    test_ivc_trivial_with::<PallasEngine, VestaEngine>();
    test_ivc_trivial_with::<Bn256Engine, GrumpkinEngine>();
    test_ivc_trivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_nontrivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let circuit_primary = TrivialCircuit::default();
    let circuit_secondary = CubicCircuit::default();

    // produce public parameters
    let pp = PublicParams::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &circuit_primary,
      &circuit_secondary,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::new(
      &pp,
      &circuit_primary,
      &circuit_secondary,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit_primary, &circuit_secondary);
      assert!(res.is_ok());

      // verify the recursive snark at each step of recursion
      let res = recursive_snark.verify(
        &pp,
        i + 1,
        &[<E1 as Engine>::Scalar::ONE],
        &[<E2 as Engine>::Scalar::ZERO],
      );
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    assert!(res.is_ok());

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
    let mut zn_secondary_direct = vec![<E2 as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_secondary_direct = circuit_secondary.clone().output(&zn_secondary_direct);
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(zn_secondary, vec![<E2 as Engine>::Scalar::from(2460515u64)]);
  }

  #[test]
  fn test_ivc_nontrivial() {
    test_ivc_nontrivial_with::<PallasEngine, VestaEngine>();
    test_ivc_nontrivial_with::<Bn256Engine, GrumpkinEngine>();
    test_ivc_nontrivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_nontrivial_with_compression_with<E1, E2, EE1, EE2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<E2>,
  {
    let circuit_primary = TrivialCircuit::default();
    let circuit_secondary = CubicCircuit::default();

    // produce public parameters
    let pp = PublicParams::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &circuit_primary,
      &circuit_secondary,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::new(
      &pp,
      &circuit_primary,
      &circuit_secondary,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for _i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit_primary, &circuit_secondary);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    assert!(res.is_ok());

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
    let mut zn_secondary_direct = vec![<E2 as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_secondary_direct = circuit_secondary.clone().output(&zn_secondary_direct);
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(zn_secondary, vec![<E2 as Engine>::Scalar::from(2460515u64)]);

    // produce the prover and verifier keys for compressed snark
    let (pk, vk) = CompressedSNARK::<_, _, _, _, S<E1, EE1>, S<E2, EE2>>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let res =
      CompressedSNARK::<_, _, _, _, S<E1, EE1>, S<E2, EE2>>::prove(&pp, &pk, &recursive_snark);
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    let res = compressed_snark.verify(
      &vk,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_nontrivial_with_compression() {
    test_ivc_nontrivial_with_compression_with::<PallasEngine, VestaEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_compression_with::<Bn256Engine, GrumpkinEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_compression_with::<Secp256k1Engine, Secq256k1Engine, EE<_>, EE<_>>();
  }

  fn test_ivc_nontrivial_with_spark_compression_with<E1, E2, EE1, EE2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<E2>,
  {
    let circuit_primary = TrivialCircuit::default();
    let circuit_secondary = CubicCircuit::default();

    // produce public parameters, which we'll use with a spark-compressed SNARK
    let pp = PublicParams::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &circuit_primary,
      &circuit_secondary,
      &*SPrime::<E1, EE1>::ck_floor(),
      &*SPrime::<E2, EE2>::ck_floor(),
    );

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::new(
      &pp,
      &circuit_primary,
      &circuit_secondary,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for _i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit_primary, &circuit_secondary);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    assert!(res.is_ok());

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
    let mut zn_secondary_direct = vec![<E2 as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_secondary_direct = CubicCircuit::default().output(&zn_secondary_direct);
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(zn_secondary, vec![<E2 as Engine>::Scalar::from(2460515u64)]);

    // run the compressed snark with Spark compiler
    // produce the prover and verifier keys for compressed snark
    let (pk, vk) =
      CompressedSNARK::<_, _, _, _, SPrime<E1, EE1>, SPrime<E2, EE2>>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let res = CompressedSNARK::<_, _, _, _, SPrime<E1, EE1>, SPrime<E2, EE2>>::prove(
      &pp,
      &pk,
      &recursive_snark,
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    let res = compressed_snark.verify(
      &vk,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_nontrivial_with_spark_compression() {
    test_ivc_nontrivial_with_spark_compression_with::<PallasEngine, VestaEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_spark_compression_with::<Bn256Engine, GrumpkinEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_spark_compression_with::<Secp256k1Engine, Secq256k1Engine, EE<_>, EE<_>>(
    );
  }

  fn test_ivc_nondet_with_compression_with<E1, E2, EE1, EE2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<E2>,
  {
    // y is a non-deterministic advice representing the fifth root of the input at a step.
    #[derive(Clone, Debug)]
    struct FifthRootCheckingCircuit<F: PrimeField> {
      y: F,
    }

    impl<F: PrimeField> FifthRootCheckingCircuit<F> {
      fn new(num_steps: usize) -> (Vec<F>, Vec<Self>) {
        let mut powers = Vec::new();
        let rng = &mut rand::rngs::OsRng;
        let mut seed = F::random(rng);
        for _i in 0..num_steps + 1 {
          seed *= seed.clone().square().square();

          powers.push(Self { y: seed });
        }

        // reverse the powers to get roots
        let roots = powers.into_iter().rev().collect::<Vec<Self>>();
        (vec![roots[0].y], roots[1..].to_vec())
      }
    }

    impl<F> StepCircuit<F> for FifthRootCheckingCircuit<F>
    where
      F: PrimeField,
    {
      fn arity(&self) -> usize {
        1
      }

      fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
      ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let x = &z[0];

        // we allocate a variable and set it to the provided non-deterministic advice.
        let y = AllocatedNum::alloc_infallible(cs.namespace(|| "y"), || self.y);

        // We now check if y = x^{1/5} by checking if y^5 = x
        let y_sq = y.square(cs.namespace(|| "y_sq"))?;
        let y_quad = y_sq.square(cs.namespace(|| "y_quad"))?;
        let y_pow_5 = y_quad.mul(cs.namespace(|| "y_fifth"), &y)?;

        cs.enforce(
          || "y^5 = x",
          |lc| lc + y_pow_5.get_variable(),
          |lc| lc + CS::one(),
          |lc| lc + x.get_variable(),
        );

        Ok(vec![y])
      }
    }

    let circuit_primary = FifthRootCheckingCircuit {
      y: <E1 as Engine>::Scalar::ZERO,
    };

    let circuit_secondary = TrivialCircuit::default();

    // produce public parameters
    let pp = PublicParams::<
      E1,
      E2,
      FifthRootCheckingCircuit<<E1 as Engine>::Scalar>,
      TrivialCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &circuit_primary,
      &circuit_secondary,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );

    let num_steps = 3;

    // produce non-deterministic advice
    let (z0_primary, roots) = FifthRootCheckingCircuit::new(num_steps);
    let z0_secondary = vec![<E2 as Engine>::Scalar::ZERO];

    // produce a recursive SNARK
    let mut recursive_snark: RecursiveSNARK<
      E1,
      E2,
      FifthRootCheckingCircuit<<E1 as Engine>::Scalar>,
      TrivialCircuit<<E2 as Engine>::Scalar>,
    > = RecursiveSNARK::<
      E1,
      E2,
      FifthRootCheckingCircuit<<E1 as Engine>::Scalar>,
      TrivialCircuit<<E2 as Engine>::Scalar>,
    >::new(
      &pp,
      &roots[0],
      &circuit_secondary,
      &z0_primary,
      &z0_secondary,
    )
    .unwrap();

    for circuit_primary in roots.iter().take(num_steps) {
      let res = recursive_snark.prove_step(&pp, circuit_primary, &circuit_secondary);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &z0_primary, &z0_secondary);
    assert!(res.is_ok());

    // produce the prover and verifier keys for compressed snark
    let (pk, vk) = CompressedSNARK::<_, _, _, _, S<E1, EE1>, S<E2, EE2>>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let res =
      CompressedSNARK::<_, _, _, _, S<E1, EE1>, S<E2, EE2>>::prove(&pp, &pk, &recursive_snark);
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    let res = compressed_snark.verify(&vk, num_steps, &z0_primary, &z0_secondary);
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_nondet_with_compression() {
    test_ivc_nondet_with_compression_with::<PallasEngine, VestaEngine, EE<_>, EE<_>>();
    test_ivc_nondet_with_compression_with::<Bn256Engine, GrumpkinEngine, EE<_>, EE<_>>();
    test_ivc_nondet_with_compression_with::<Secp256k1Engine, Secq256k1Engine, EE<_>, EE<_>>();
  }

  fn test_ivc_base_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
    let test_circuit2 = CubicCircuit::<<E2 as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &test_circuit1,
      &test_circuit2,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<
      E1,
      E2,
      TrivialCircuit<<E1 as Engine>::Scalar>,
      CubicCircuit<<E2 as Engine>::Scalar>,
    >::new(
      &pp,
      &test_circuit1,
      &test_circuit2,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    // produce a recursive SNARK
    let res = recursive_snark.prove_step(&pp, &test_circuit1, &test_circuit2);

    assert!(res.is_ok());

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ONE],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    assert!(res.is_ok());

    let (zn_primary, zn_secondary) = res.unwrap();

    assert_eq!(zn_primary, vec![<E1 as Engine>::Scalar::ONE]);
    assert_eq!(zn_secondary, vec![<E2 as Engine>::Scalar::from(5u64)]);
  }

  #[test]
  fn test_ivc_base() {
    test_ivc_base_with::<PallasEngine, VestaEngine>();
    test_ivc_base_with::<Bn256Engine, GrumpkinEngine>();
    test_ivc_base_with::<Secp256k1Engine, Secq256k1Engine>();
  }
}
