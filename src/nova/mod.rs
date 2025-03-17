//! This module implements Nova's IVC scheme including its folding scheme.

use crate::{
  constants::NUM_HASH_BITS,
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  frontend::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    ConstraintSystem, SynthesisError,
  },
  gadgets::utils::{base_as_scalar, scalar_as_base},
  r1cs::{
    CommitmentKeyHint, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance,
    RelaxedR1CSWitness,
  },
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    commitment::CommitmentEngineTrait,
    snark::RelaxedR1CSSNARKTrait,
    AbsorbInROTrait, Engine, ROConstants, ROConstantsCircuit, ROTrait,
  },
  CommitmentKey, DerandKey,
};
use core::marker::PhantomData;
use ff::Field;
use once_cell::sync::OnceCell;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

mod circuit;
pub(crate) mod nifs;

use circuit::{NovaAugmentedCircuit, NovaAugmentedCircuitInputs};
use nifs::{NIFSRelaxed, NIFS};

/// A type that holds public parameters of Nova
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  F_arity: usize,

  ro_consts_primary: ROConstants<E1>,
  ro_consts_circuit_primary: ROConstantsCircuit<E2>,

  ro_consts_secondary: ROConstants<E2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<E1>,

  ck_primary: CommitmentKey<E1>,
  r1cs_shape_primary: R1CSShape<E1>,

  ck_secondary: CommitmentKey<E2>,
  r1cs_shape_secondary: R1CSShape<E2>,

  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E1::Scalar>,
  _p: PhantomData<C>,
}

impl<E1, E2, C> SimpleDigestible for PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
}

impl<E1, E2, C> PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  /// Creates a new `PublicParams` for a circuit `C`.
  ///
  /// # Note
  ///
  /// Public parameters set up a number of bases for the homomorphic commitment scheme of Nova.
  ///
  /// Some final compressing SNARKs, like variants of Spartan, use computation commitments that require
  /// larger sizes for these parameters. These SNARKs provide a hint for these values by
  /// implementing `RelaxedR1CSSNARKTrait::ck_floor()`, which can be passed to this function.
  ///
  /// If you're not using such a SNARK, pass `nova_snark::traits::snark::default_ck_hint()` instead.
  ///
  /// # Arguments
  ///
  /// * `c`: The primary circuit of type `C`.
  /// * `ck_hint`: A `CommitmentKeyHint` for `S1`, which is a function that provides a hint
  ///   for the number of generators required in the commitment scheme for the primary circuit.
  ///
  /// # Example
  ///
  /// ```rust
  /// # use nova_snark::spartan::ppsnark::RelaxedR1CSSNARK;
  /// # use nova_snark::provider::ipa_pc::EvaluationEngine;
  /// # use nova_snark::provider::{PallasEngine, VestaEngine};
  /// # use nova_snark::traits::{circuit::TrivialCircuit, Engine, snark::RelaxedR1CSSNARKTrait};
  /// # use nova_snark::nova::PublicParams;
  ///
  /// type E1 = PallasEngine;
  /// type E2 = VestaEngine;
  /// type EE<E> = EvaluationEngine<E>;
  /// type SPrime<E> = RelaxedR1CSSNARK<E, EE<E>>;
  ///
  /// let circuit = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
  /// // Only relevant for a SNARK using computational commitments, pass &(|_| 0)
  /// // or &*nova_snark::traits::snark::default_ck_hint() otherwise.
  /// let ck_hint1 = &*SPrime::<E1>::ck_floor();
  /// let ck_hint2 = &*SPrime::<E2>::ck_floor();
  ///
  /// PublicParams::setup(&circuit, ck_hint1, ck_hint2);
  /// ```
  pub fn setup(
    c: &C,
    ck_hint1: &CommitmentKeyHint<E1>,
    ck_hint2: &CommitmentKeyHint<E2>,
  ) -> Result<Self, NovaError> {
    let ro_consts_primary: ROConstants<E1> = ROConstants::<E1>::default();
    let ro_consts_secondary: ROConstants<E2> = ROConstants::<E2>::default();

    let F_arity = c.arity();

    // ro_consts_circuit_primary are parameterized by E2 because the type alias uses E2::Base = E1::Scalar
    let ro_consts_circuit_primary: ROConstantsCircuit<E2> = ROConstantsCircuit::<E2>::default();
    let ro_consts_circuit_secondary: ROConstantsCircuit<E1> = ROConstantsCircuit::<E1>::default();

    // Initialize ck for the primary
    let circuit_primary: NovaAugmentedCircuit<'_, E2, C> =
      NovaAugmentedCircuit::new(true, None, c, ro_consts_circuit_primary.clone());
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (r1cs_shape_primary, ck_primary) = cs.r1cs_shape(ck_hint1);

    // Initialize ck for the secondary
    let tc = TrivialCircuit::<E2::Scalar>::default();
    let circuit_secondary: NovaAugmentedCircuit<'_, E1, _> =
      NovaAugmentedCircuit::new(false, None, &tc, ro_consts_circuit_secondary.clone());
    let mut cs: ShapeCS<E2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let (r1cs_shape_secondary, ck_secondary) = cs.r1cs_shape(ck_hint2);

    if r1cs_shape_primary.num_io != 2 || r1cs_shape_secondary.num_io != 2 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    let pp = PublicParams {
      F_arity,

      ro_consts_primary,
      ro_consts_circuit_primary,

      ro_consts_secondary,
      ro_consts_circuit_secondary,

      ck_primary,
      r1cs_shape_primary,

      ck_secondary,
      r1cs_shape_secondary,

      digest: OnceCell::new(),
      _p: Default::default(),
    };

    // call pp.digest() so the digest is computed here rather than in RecursiveSNARK methods
    let _ = pp.digest();

    Ok(pp)
  }

  /// Retrieve the digest of the public parameters.
  pub fn digest(&self) -> E1::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }

  /// Returns the number of constraints in the primary and secondary circuits
  pub const fn num_constraints(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_cons,
      self.r1cs_shape_secondary.num_cons,
    )
  }

  /// Returns the number of variables in the primary and secondary circuits
  pub const fn num_variables(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_vars,
      self.r1cs_shape_secondary.num_vars,
    )
  }
}

/// A SNARK that proves the correct execution of an incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  z0: Vec<E1::Scalar>,

  r_W_primary: RelaxedR1CSWitness<E1>,
  r_U_primary: RelaxedR1CSInstance<E1>,
  ri_primary: E1::Scalar,

  r_W_secondary: RelaxedR1CSWitness<E2>,
  r_U_secondary: RelaxedR1CSInstance<E2>,
  ri_secondary: E2::Scalar,

  l_w_secondary: R1CSWitness<E2>,
  l_u_secondary: R1CSInstance<E2>,

  i: usize,

  zi: Vec<E1::Scalar>,

  _p: PhantomData<C>,
}

impl<E1, E2, C> RecursiveSNARK<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  /// Create new instance of recursive SNARK
  pub fn new(pp: &PublicParams<E1, E2, C>, c: &C, z0: &[E1::Scalar]) -> Result<Self, NovaError> {
    if z0.len() != pp.F_arity {
      return Err(NovaError::InvalidInitialInputLength);
    }

    let ri_primary = E1::Scalar::random(&mut OsRng);
    let ri_secondary = E2::Scalar::random(&mut OsRng);

    // base case for the primary
    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let inputs_primary: NovaAugmentedCircuitInputs<E2> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      E1::Scalar::ZERO,
      z0.to_vec(),
      None,
      None,
      None,
      ri_primary, // "r next"
      None,
      None,
    );

    let circuit_primary: NovaAugmentedCircuit<'_, E2, C> = NovaAugmentedCircuit::new(
      true,
      Some(inputs_primary),
      c,
      pp.ro_consts_circuit_primary.clone(),
    );
    let zi_primary = circuit_primary.synthesize(&mut cs_primary)?;
    let (u_primary, w_primary) =
      cs_primary.r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)?;

    // base case for the secondary
    let mut cs_secondary = SatisfyingAssignment::<E2>::new();
    let inputs_secondary: NovaAugmentedCircuitInputs<E1> = NovaAugmentedCircuitInputs::new(
      pp.digest(),
      E2::Scalar::ZERO,
      vec![E2::Scalar::ZERO],
      None,
      None,
      None,
      ri_secondary, // "r next"
      Some(u_primary.clone()),
      None,
    );
    let tc = TrivialCircuit::<E2::Scalar>::default();
    let circuit_secondary: NovaAugmentedCircuit<'_, E1, _> = NovaAugmentedCircuit::new(
      false,
      Some(inputs_secondary),
      &tc,
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary)?;
    let (u_secondary, w_secondary) =
      cs_secondary.r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)?;

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

    if zi_primary.len() != pp.F_arity {
      return Err(NovaError::InvalidStepOutputLength);
    }

    let zi_primary = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    Ok(Self {
      z0: z0.to_vec(),

      r_W_primary,
      r_U_primary,
      ri_primary,

      r_W_secondary,
      r_U_secondary,
      ri_secondary,

      l_w_secondary,
      l_u_secondary,

      i: 0,

      zi: zi_primary,

      _p: Default::default(),
    })
  }

  /// Updates the provided `RecursiveSNARK` by executing a step of the incremental computation
  pub fn prove_step(&mut self, pp: &PublicParams<E1, E2, C>, c: &C) -> Result<(), NovaError> {
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
    )?;

    let r_next_primary = E1::Scalar::random(&mut OsRng);

    let mut cs_primary = SatisfyingAssignment::<E1>::new();
    let inputs_primary: NovaAugmentedCircuitInputs<E2> = NovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(pp.digest()),
      E1::Scalar::from(self.i as u64),
      self.z0.to_vec(),
      Some(self.zi.clone()),
      Some(self.r_U_secondary.clone()),
      Some(self.ri_primary),
      r_next_primary,
      Some(self.l_u_secondary.clone()),
      Some(nifs_secondary.comm_T),
    );

    let circuit_primary: NovaAugmentedCircuit<'_, E2, C> = NovaAugmentedCircuit::new(
      true,
      Some(inputs_primary),
      c,
      pp.ro_consts_circuit_primary.clone(),
    );
    let zi_primary = circuit_primary.synthesize(&mut cs_primary)?;

    let (l_u_primary, l_w_primary) =
      cs_primary.r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)?;

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
    )?;

    let r_next_secondary = E2::Scalar::random(&mut OsRng);

    let mut cs_secondary = SatisfyingAssignment::<E2>::new();
    let inputs_secondary: NovaAugmentedCircuitInputs<E1> = NovaAugmentedCircuitInputs::new(
      pp.digest(),
      E2::Scalar::from(self.i as u64),
      vec![E2::Scalar::ZERO],
      Some(vec![E2::Scalar::ZERO]),
      Some(self.r_U_primary.clone()),
      Some(self.ri_secondary),
      r_next_secondary,
      Some(l_u_primary),
      Some(nifs_primary.comm_T),
    );

    let tc = TrivialCircuit::<E2::Scalar>::default();
    let circuit_secondary: NovaAugmentedCircuit<'_, E1, _> = NovaAugmentedCircuit::new(
      false,
      Some(inputs_secondary),
      &tc,
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary)?;

    let (l_u_secondary, l_w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness on the secondary curve".to_string(),
      })?;

    // update the running instances and witnesses
    self.zi = zi_primary
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    self.l_u_secondary = l_u_secondary;
    self.l_w_secondary = l_w_secondary;

    self.r_U_primary = r_U_primary;
    self.r_W_primary = r_W_primary;

    self.i += 1;

    self.r_U_secondary = r_U_secondary;
    self.r_W_secondary = r_W_secondary;

    self.ri_primary = r_next_primary;
    self.ri_secondary = r_next_secondary;

    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<E1, E2, C>,
    num_steps: usize,
    z0: &[E1::Scalar],
  ) -> Result<Vec<E1::Scalar>, NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;

    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;

    // check if the initial inputs match
    let is_inputs_not_match = self.z0 != z0;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2;

    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid number of steps or inputs".to_string(),
      });
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <E2 as Engine>::RO::new(pp.ro_consts_secondary.clone());
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0 {
        hasher.absorb(*e);
      }
      for e in &self.zi {
        hasher.absorb(*e);
      }
      self.r_U_secondary.absorb_in_ro(&mut hasher);
      hasher.absorb(self.ri_primary);

      let mut hasher2 = <E1 as Engine>::RO::new(pp.ro_consts_primary.clone());
      hasher2.absorb(scalar_as_base::<E1>(pp.digest()));
      hasher2.absorb(E2::Scalar::from(num_steps as u64));
      hasher2.absorb(E2::Scalar::ZERO);
      hasher2.absorb(E2::Scalar::ZERO);
      self.r_U_primary.absorb_in_ro(&mut hasher2);
      hasher2.absorb(self.ri_secondary);

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != scalar_as_base::<E2>(self.l_u_secondary.X[0])
      || hash_secondary != self.l_u_secondary.X[1]
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid output hash in R1CS instances".to_string(),
      });
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

    Ok(self.zi.clone())
  }

  /// Get the outputs after the last step of computation.
  pub fn outputs(&self) -> &[E1::Scalar] {
    &self.zi
  }

  /// The number of steps which have been executed thus far.
  pub fn num_steps(&self) -> usize {
    self.i
  }
}

/// A type that holds the prover key for `CompressedSNARK`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E1, E2, C, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  pk_primary: S1::ProverKey,
  pk_secondary: S2::ProverKey,
  _p: PhantomData<C>,
}

/// A type that holds the verifier key for `CompressedSNARK`
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E1, E2, C, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  F_arity: usize,
  ro_consts_primary: ROConstants<E1>,
  ro_consts_secondary: ROConstants<E2>,
  pp_digest: E1::Scalar,
  vk_primary: S1::VerifierKey,
  vk_secondary: S2::VerifierKey,
  dk_primary: DerandKey<E1>,
  dk_secondary: DerandKey<E2>,
  _p: PhantomData<C>,
}

/// A SNARK that proves the knowledge of a valid `RecursiveSNARK`
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E1, E2, C, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  r_U_secondary: RelaxedR1CSInstance<E2>,
  ri_secondary: E2::Scalar,
  l_u_secondary: R1CSInstance<E2>,
  nifs_Uf_secondary: NIFS<E2>,

  l_ur_secondary: RelaxedR1CSInstance<E2>,
  nifs_Un_secondary: NIFSRelaxed<E2>,

  r_U_primary: RelaxedR1CSInstance<E1>,
  ri_primary: E1::Scalar,
  l_ur_primary: RelaxedR1CSInstance<E1>,
  nifs_Un_primary: NIFSRelaxed<E1>,

  wit_blind_r_Wn_primary: E1::Scalar,
  err_blind_r_Wn_primary: E1::Scalar,
  wit_blind_r_Wn_secondary: E2::Scalar,
  err_blind_r_Wn_secondary: E2::Scalar,

  snark_primary: S1,
  snark_secondary: S2,

  zn: Vec<E1::Scalar>,

  _p: PhantomData<C>,
}

impl<E1, E2, C, S1, S2> CompressedSNARK<E1, E2, C, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  /// Creates prover and verifier keys for `CompressedSNARK`
  pub fn setup(
    pp: &PublicParams<E1, E2, C>,
  ) -> Result<(ProverKey<E1, E2, C, S1, S2>, VerifierKey<E1, E2, C, S1, S2>), NovaError> {
    let (pk_primary, vk_primary) = S1::setup(&pp.ck_primary, &pp.r1cs_shape_primary)?;
    let (pk_secondary, vk_secondary) = S2::setup(&pp.ck_secondary, &pp.r1cs_shape_secondary)?;

    let pk = ProverKey {
      pk_primary,
      pk_secondary,
      _p: Default::default(),
    };

    let vk = VerifierKey {
      F_arity: pp.F_arity,
      ro_consts_primary: pp.ro_consts_primary.clone(),
      ro_consts_secondary: pp.ro_consts_secondary.clone(),
      pp_digest: pp.digest(),
      vk_primary,
      vk_secondary,
      dk_primary: E1::CE::derand_key(&pp.ck_primary),
      dk_secondary: E2::CE::derand_key(&pp.ck_secondary),
      _p: Default::default(),
    };

    Ok((pk, vk))
  }

  /// Create a new `CompressedSNARK` (provides zero-knowledge)
  pub fn prove(
    pp: &PublicParams<E1, E2, C>,
    pk: &ProverKey<E1, E2, C, S1, S2>,
    recursive_snark: &RecursiveSNARK<E1, E2, C>,
  ) -> Result<Self, NovaError> {
    // prove three foldings

    // fold secondary U/W with secondary u/w to get Uf/Wf
    let (nifs_Uf_secondary, (r_Uf_secondary, r_Wf_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.r1cs_shape_secondary,
      &recursive_snark.r_U_secondary,
      &recursive_snark.r_W_secondary,
      &recursive_snark.l_u_secondary,
      &recursive_snark.l_w_secondary,
    )?;

    // fold Uf/Wf with random inst/wit to get U1/W1
    let (l_ur_secondary, l_wr_secondary) = pp
      .r1cs_shape_secondary
      .sample_random_instance_witness(&pp.ck_secondary)?;

    let (nifs_Un_secondary, (r_Un_secondary, r_Wn_secondary)) = NIFSRelaxed::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.r1cs_shape_secondary,
      &r_Uf_secondary,
      &r_Wf_secondary,
      &l_ur_secondary,
      &l_wr_secondary,
    )?;

    // fold primary U/W with random inst/wit to get U2/W2
    let (l_ur_primary, l_wr_primary) = pp
      .r1cs_shape_primary
      .sample_random_instance_witness(&pp.ck_primary)?;

    let (nifs_Un_primary, (r_Un_primary, r_Wn_primary)) = NIFSRelaxed::prove(
      &pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.digest(),
      &pp.r1cs_shape_primary,
      &recursive_snark.r_U_primary,
      &recursive_snark.r_W_primary,
      &l_ur_primary,
      &l_wr_primary,
    )?;

    // derandomize/unblind commitments
    let (derandom_r_Wn_primary, wit_blind_r_Wn_primary, err_blind_r_Wn_primary) =
      r_Wn_primary.derandomize();
    let derandom_r_Un_primary = r_Un_primary.derandomize(
      &E1::CE::derand_key(&pp.ck_primary),
      &wit_blind_r_Wn_primary,
      &err_blind_r_Wn_primary,
    );

    let (derandom_r_Wn_secondary, wit_blind_r_Wn_secondary, err_blind_r_Wn_secondary) =
      r_Wn_secondary.derandomize();
    let derandom_r_Un_secondary = r_Un_secondary.derandomize(
      &E2::CE::derand_key(&pp.ck_secondary),
      &wit_blind_r_Wn_secondary,
      &err_blind_r_Wn_secondary,
    );

    // create SNARKs proving the knowledge of Wn primary/secondary
    let (snark_primary, snark_secondary) = rayon::join(
      || {
        S1::prove(
          &pp.ck_primary,
          &pk.pk_primary,
          &pp.r1cs_shape_primary,
          &derandom_r_Un_primary,
          &derandom_r_Wn_primary,
        )
      },
      || {
        S2::prove(
          &pp.ck_secondary,
          &pk.pk_secondary,
          &pp.r1cs_shape_secondary,
          &derandom_r_Un_secondary,
          &derandom_r_Wn_secondary,
        )
      },
    );

    Ok(Self {
      r_U_secondary: recursive_snark.r_U_secondary.clone(),
      ri_secondary: recursive_snark.ri_secondary,
      l_u_secondary: recursive_snark.l_u_secondary.clone(),
      nifs_Uf_secondary: nifs_Uf_secondary.clone(),

      l_ur_secondary: l_ur_secondary.clone(),
      nifs_Un_secondary: nifs_Un_secondary.clone(),

      r_U_primary: recursive_snark.r_U_primary.clone(),
      ri_primary: recursive_snark.ri_primary,
      l_ur_primary: l_ur_primary.clone(),
      nifs_Un_primary: nifs_Un_primary.clone(),

      wit_blind_r_Wn_primary,
      err_blind_r_Wn_primary,
      wit_blind_r_Wn_secondary,
      err_blind_r_Wn_secondary,

      snark_primary: snark_primary?,
      snark_secondary: snark_secondary?,

      zn: recursive_snark.zi.clone(),

      _p: Default::default(),
    })
  }

  /// Verify the correctness of the `CompressedSNARK` (provides zero-knowledge)
  pub fn verify(
    &self,
    vk: &VerifierKey<E1, E2, C, S1, S2>,
    num_steps: usize,
    z0: &[E1::Scalar],
  ) -> Result<Vec<E1::Scalar>, NovaError> {
    // the number of steps cannot be zero
    if num_steps == 0 {
      return Err(NovaError::ProofVerifyError {
        reason: "Number of steps cannot be zero".to_string(),
      });
    }

    // check if the (relaxed) R1CS instances have two public outputs
    if self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2
      || self.l_ur_primary.X.len() != 2
      || self.l_ur_secondary.X.len() != 2
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid number of outputs in R1CS instances".to_string(),
      });
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <E2 as Engine>::RO::new(vk.ro_consts_secondary.clone());
      hasher.absorb(vk.pp_digest);
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0 {
        hasher.absorb(*e);
      }
      for e in &self.zn {
        hasher.absorb(*e);
      }
      self.r_U_secondary.absorb_in_ro(&mut hasher);
      hasher.absorb(self.ri_primary);

      let mut hasher2 = <E1 as Engine>::RO::new(vk.ro_consts_primary.clone());
      hasher2.absorb(scalar_as_base::<E1>(vk.pp_digest));
      hasher2.absorb(E2::Scalar::from(num_steps as u64));
      hasher2.absorb(E2::Scalar::ZERO);
      hasher2.absorb(E2::Scalar::ZERO);
      self.r_U_primary.absorb_in_ro(&mut hasher2);
      hasher2.absorb(self.ri_secondary);

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != base_as_scalar::<E1>(self.l_u_secondary.X[0])
      || hash_secondary != self.l_u_secondary.X[1]
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid output hash in R1CS instances".to_string(),
      });
    }

    // fold secondary U/W with secondary u/w to get Uf/Wf
    let r_Uf_secondary = self.nifs_Uf_secondary.verify(
      &vk.ro_consts_secondary,
      &scalar_as_base::<E1>(vk.pp_digest),
      &self.r_U_secondary,
      &self.l_u_secondary,
    )?;

    // fold Uf/Wf with random inst/wit to get U1/W1
    let r_Un_secondary = self.nifs_Un_secondary.verify(
      &vk.ro_consts_secondary,
      &scalar_as_base::<E1>(vk.pp_digest),
      &r_Uf_secondary,
      &self.l_ur_secondary,
    )?;

    // fold primary U/W with random inst/wit to get U2/W2
    let r_Un_primary = self.nifs_Un_primary.verify(
      &vk.ro_consts_primary,
      &vk.pp_digest,
      &self.r_U_primary,
      &self.l_ur_primary,
    )?;

    // derandomize/unblind commitments
    let derandom_r_Un_primary = r_Un_primary.derandomize(
      &vk.dk_primary,
      &self.wit_blind_r_Wn_primary,
      &self.err_blind_r_Wn_primary,
    );
    let derandom_r_Un_secondary = r_Un_secondary.derandomize(
      &vk.dk_secondary,
      &self.wit_blind_r_Wn_secondary,
      &self.err_blind_r_Wn_secondary,
    );

    // check the satisfiability of the folded instances using
    // SNARKs proving the knowledge of their satisfying witnesses
    let (res_primary, res_secondary) = rayon::join(
      || {
        self
          .snark_primary
          .verify(&vk.vk_primary, &derandom_r_Un_primary)
      },
      || {
        self
          .snark_secondary
          .verify(&vk.vk_secondary, &derandom_r_Un_secondary)
      },
    );

    res_primary?;
    res_secondary?;

    Ok(self.zn.clone())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    provider::{
      pedersen::CommitmentKeyExtTrait, traits::DlogGroup, Bn256EngineIPA, Bn256EngineKZG,
      GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::{circuit::TrivialCircuit, evaluation::EvaluationEngineTrait, snark::default_ck_hint},
  };
  use core::{fmt::Write, marker::PhantomData};
  use expect_test::{expect, Expect};
  use ff::PrimeField;

  type EE<E> = crate::provider::ipa_pc::EvaluationEngine<E>;
  type EEPrime<E> = crate::provider::hyperkzg::EvaluationEngine<E>;
  type S<E, EE> = crate::spartan::snark::RelaxedR1CSSNARK<E, EE>;
  type SPrime<E, EE> = crate::spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

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

  fn test_pp_digest_with<E1, E2, C>(circuit: &C, expected: &Expect)
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    E1::GE: DlogGroup,
    E2::GE: DlogGroup,
    C: StepCircuit<E1::Scalar>,
    // required to use the IPA in the initialization of the commitment key hints below
    <E1::CE as CommitmentEngineTrait<E1>>::CommitmentKey: CommitmentKeyExtTrait<E1>,
    <E2::CE as CommitmentEngineTrait<E2>>::CommitmentKey: CommitmentKeyExtTrait<E2>,
  {
    // this tests public parameters with a size specifically intended for a spark-compressed SNARK
    let ck_hint1 = &*SPrime::<E1, EE<E1>>::ck_floor();
    let ck_hint2 = &*SPrime::<E2, EE<E2>>::ck_floor();
    let pp = PublicParams::<E1, E2, C>::setup(circuit, ck_hint1, ck_hint2).unwrap();

    let digest_str = pp
      .digest()
      .to_repr()
      .as_ref()
      .iter()
      .fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02x}");
        output
      });
    expected.assert_eq(&digest_str);
  }

  #[test]
  fn test_pp_digest() {
    test_pp_digest_with::<PallasEngine, VestaEngine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["fbd08d8d030105a2fedd6c16f5964081aac34c3ee3c6080797561af57b818802"],
    );

    test_pp_digest_with::<Bn256EngineIPA, GrumpkinEngine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["99dc4a55d3e2fec50e4da7a74c9f8fa3ae61d9871d03dc7f703dd347c78f4800"],
    );

    test_pp_digest_with::<Secp256k1Engine, Secq256k1Engine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["d9fac48ccd1f55973e3fe861d35b68d56cfe1ced124555c4c27714dc1d0b2b03"],
    );
  }

  fn test_ivc_trivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit = TrivialCircuit::<<E1 as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<E1, E2, TrivialCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark =
      RecursiveSNARK::new(&pp, &test_circuit, &[<E1 as Engine>::Scalar::ZERO]).unwrap();

    let res = recursive_snark.prove_step(&pp, &test_circuit);

    assert!(res.is_ok());

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_trivial() {
    test_ivc_trivial_with::<PallasEngine, VestaEngine>();
    test_ivc_trivial_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_trivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_nontrivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let circuit = CubicCircuit::default();

    // produce public parameters
    let pp = PublicParams::<E1, E2, CubicCircuit<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &circuit,
      &[<E1 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit);
      assert!(res.is_ok());

      // verify the recursive snark at each step of recursion
      let res = recursive_snark.verify(&pp, i + 1, &[<E1 as Engine>::Scalar::ZERO]);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    let mut zn_direct = vec![<E1 as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_direct = circuit.clone().output(&zn_direct);
    }
    assert_eq!(zn, zn_direct);
    assert_eq!(zn, vec![E1::Scalar::from(2460515u64)]);
  }

  #[test]
  fn test_ivc_nontrivial() {
    test_ivc_nontrivial_with::<PallasEngine, VestaEngine>();
    test_ivc_nontrivial_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_nontrivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_nontrivial_with_compression_with<E1, E2, EE1, EE2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<E2>,
  {
    let circuit = CubicCircuit::default();

    // produce public parameters
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &circuit,
      &[<E1 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for _i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    let mut zn_direct = vec![<E1 as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_direct = circuit.clone().output(&zn_direct);
    }
    assert_eq!(zn, zn_direct);
    assert_eq!(zn, vec![<E1 as Engine>::Scalar::from(2460515u64)]);

    // produce the prover and verifier keys for compressed snark
    let (pk, vk) = CompressedSNARK::<_, _, _, S<E1, EE1>, S<E2, EE2>>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let res = CompressedSNARK::<_, _, _, S<E1, EE1>, S<E2, EE2>>::prove(&pp, &pk, &recursive_snark);
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    let res = compressed_snark.verify(&vk, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_nontrivial_with_compression() {
    test_ivc_nontrivial_with_compression_with::<PallasEngine, VestaEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_compression_with::<Bn256EngineKZG, GrumpkinEngine, EEPrime<_>, EE<_>>(
    );
    test_ivc_nontrivial_with_compression_with::<Secp256k1Engine, Secq256k1Engine, EE<_>, EE<_>>();

    test_ivc_nontrivial_with_spark_compression_with::<
      Bn256EngineKZG,
      GrumpkinEngine,
      crate::provider::hyperkzg::EvaluationEngine<_>,
      EE<_>,
    >();
  }

  fn test_ivc_nontrivial_with_spark_compression_with<E1, E2, EE1, EE2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    EE1: EvaluationEngineTrait<E1>,
    EE2: EvaluationEngineTrait<E2>,
  {
    let circuit = CubicCircuit::default();

    // produce public parameters, which we'll use with a spark-compressed SNARK
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &circuit,
      &*SPrime::<E1, EE1>::ck_floor(),
      &*SPrime::<E2, EE2>::ck_floor(),
    )
    .unwrap();

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &circuit,
      &[<E1 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    for _i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    let mut zn_direct = vec![<E1 as Engine>::Scalar::ZERO];
    for _i in 0..num_steps {
      zn_direct = CubicCircuit::default().output(&zn_direct);
    }
    assert_eq!(zn, zn_direct);
    assert_eq!(zn, vec![<E1 as Engine>::Scalar::from(2460515u64)]);

    // run the compressed snark with Spark compiler
    // produce the prover and verifier keys for compressed snark
    let (pk, vk) =
      CompressedSNARK::<_, _, _, SPrime<E1, EE1>, SPrime<E2, EE2>>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let res = CompressedSNARK::<_, _, _, SPrime<E1, EE1>, SPrime<E2, EE2>>::prove(
      &pp,
      &pk,
      &recursive_snark,
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    let res = compressed_snark.verify(&vk, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_nontrivial_with_spark_compression() {
    test_ivc_nontrivial_with_spark_compression_with::<PallasEngine, VestaEngine, EE<_>, EE<_>>();
    test_ivc_nontrivial_with_spark_compression_with::<
      Bn256EngineKZG,
      GrumpkinEngine,
      EEPrime<_>,
      EE<_>,
    >();
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

    let circuit = FifthRootCheckingCircuit {
      y: <E1 as Engine>::Scalar::ZERO,
    };

    // produce public parameters
    let pp = PublicParams::<E1, E2, FifthRootCheckingCircuit<<E1 as Engine>::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 3;

    // produce non-deterministic advice
    let (z0, roots) = FifthRootCheckingCircuit::new(num_steps);

    // produce a recursive SNARK
    let mut recursive_snark: RecursiveSNARK<
      E1,
      E2,
      FifthRootCheckingCircuit<<E1 as Engine>::Scalar>,
    > = RecursiveSNARK::<E1, E2, FifthRootCheckingCircuit<<E1 as Engine>::Scalar>>::new(
      &pp, &roots[0], &z0,
    )
    .unwrap();

    for circuit in roots.iter().take(num_steps) {
      let res = recursive_snark.prove_step(&pp, circuit);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &z0);
    assert!(res.is_ok());

    // produce the prover and verifier keys for compressed snark
    let (pk, vk) = CompressedSNARK::<_, _, _, S<E1, EE1>, S<E2, EE2>>::setup(&pp).unwrap();

    // produce a compressed SNARK
    let res = CompressedSNARK::<_, _, _, S<E1, EE1>, S<E2, EE2>>::prove(&pp, &pk, &recursive_snark);
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    let res = compressed_snark.verify(&vk, num_steps, &z0);
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_nondet_with_compression() {
    test_ivc_nondet_with_compression_with::<PallasEngine, VestaEngine, EE<_>, EE<_>>();
    test_ivc_nondet_with_compression_with::<Bn256EngineKZG, GrumpkinEngine, EEPrime<_>, EE<_>>();
    test_ivc_nondet_with_compression_with::<Secp256k1Engine, Secq256k1Engine, EE<_>, EE<_>>();
  }

  fn test_ivc_base_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit1 = CubicCircuit::<<E1 as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit1,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &test_circuit1,
      &[<E1 as Engine>::Scalar::ZERO],
    )
    .unwrap();

    // produce a recursive SNARK
    let res = recursive_snark.prove_step(&pp, &test_circuit1);

    assert!(res.is_ok());

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    assert_eq!(zn, vec![<E1 as Engine>::Scalar::from(5u64)]);
  }

  #[test]
  fn test_ivc_base() {
    test_ivc_base_with::<PallasEngine, VestaEngine>();
    test_ivc_base_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_base_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_setup_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    #[derive(Clone, Debug, Default)]
    struct CircuitWithInputize<F: PrimeField> {
      _p: PhantomData<F>,
    }

    impl<F: PrimeField> StepCircuit<F> for CircuitWithInputize<F> {
      fn arity(&self) -> usize {
        1
      }

      fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
      ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let x = &z[0];
        let y = x.square(cs.namespace(|| "x_sq"))?;
        y.inputize(cs.namespace(|| "y"))?; // inputize y
        Ok(vec![y])
      }
    }

    // produce public parameters with trivial secondary
    let circuit = CircuitWithInputize::<<E1 as Engine>::Scalar>::default();
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));

    let circuit = CircuitWithInputize::<E1::Scalar>::default();
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));
  }

  #[test]
  fn test_setup() {
    test_setup_with::<Bn256EngineKZG, GrumpkinEngine>();
  }
}
