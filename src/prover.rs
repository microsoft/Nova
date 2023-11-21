//! A type that holds the prover key for `CompressedSNARK`
use crate::bellpepper::{r1cs::NovaWitness, solver::SatisfyingAssignment};
use crate::circuit::{NovaAugmentedCircuit, NovaAugmentedCircuitInputs};
use crate::constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS};
use crate::errors::NovaError;
use crate::gadgets::utils::scalar_as_base;
use crate::nifs::NIFS;
use crate::public_params::PublicParams;
use crate::r1cs::{R1CSInstance, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness};
use crate::traits::{
  circuit::StepCircuit,
  commitment::{Commitment, CommitmentTrait},
  snark::RelaxedR1CSSNARKTrait,
  AbsorbInROTrait, Engine, ROTrait,
};
use bellpepper_core::ConstraintSystem;
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for `CompressedSNARK`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  pub(crate) pk_primary: S1::ProverKey,
  pub(crate) pk_secondary: S2::ProverKey,
  pub(crate) _p: PhantomData<(C1, C2)>,
}

/// A type that holds the proving context for `RecursiveSNARK`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProvingContext<E1, E2, C1, C2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
{
  pub(crate) z0_primary: Vec<E1::Scalar>,
  pub(crate) z0_secondary: Vec<E2::Scalar>,
  pub(crate) r_W_primary: RelaxedR1CSWitness<E1>,
  pub(crate) r_U_primary: RelaxedR1CSInstance<E1>,
  pub(crate) r_W_secondary: RelaxedR1CSWitness<E2>,
  pub(crate) r_U_secondary: RelaxedR1CSInstance<E2>,
  pub(crate) l_w_secondary: R1CSWitness<E2>,
  pub(crate) l_u_secondary: R1CSInstance<E2>,
  pub(crate) i: usize,
  pub(crate) zi_primary: Vec<E1::Scalar>,
  pub(crate) zi_secondary: Vec<E2::Scalar>,
  pub(crate) _p: PhantomData<(C1, C2)>,
}

impl<E1, E2, C1, C2> ProvingContext<E1, E2, C1, C2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
{
  /// Create a new `RecursiveSNARK` (or updates the provided `RecursiveSNARK`)
  /// by executing a step of the incremental computation
  pub fn prove(
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
  ) -> Result<(Vec<E1::Scalar>, Vec<E2::Scalar>), NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = self.i == 0;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outpus = self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2;

    if is_num_steps_zero || is_instance_has_two_outpus {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <E2 as Engine>::RO::new(
        pp.ro_consts_secondary.clone(),
        NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * pp.F_arity_primary,
      );
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(self.i as u64));
      for e in &self.z0_primary {
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
      hasher2.absorb(E2::Scalar::from(self.i as u64));
      for e in &self.z0_secondary {
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
