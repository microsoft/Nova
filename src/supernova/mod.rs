//! This library implements SuperNova, a Non-Uniform IVC based on Nova.

use std::marker::PhantomData;

use crate::{
  compute_digest,
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_HASH_BITS},
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness, R1CS},
  scalar_as_base,
  traits::{
    circuit_supernova::StepCircuit, commitment::CommitmentTrait, AbsorbInROTrait, Group,
    ROConstants, ROConstantsCircuit, ROConstantsTrait, ROTrait,
  },
  Commitment, CommitmentKey,
};

use ff::Field;
use serde::{Deserialize, Serialize};

use crate::bellperson::{
  r1cs::{NovaShape, NovaWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};
use ::bellperson::{Circuit, ConstraintSystem};

use crate::nifs::NIFS;

mod circuit; // declare the module first
use circuit::{CircuitInputs, CircuitParams, SuperNovaCircuit};

/// A type that holds public parameters of Nova
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  F_arity_primary: usize,
  F_arity_secondary: usize,
  ro_consts_primary: ROConstants<G1>,
  ro_consts_circuit_primary: ROConstantsCircuit<G2>,
  ck_primary: Option<CommitmentKey<G1>>,
  r1cs_shape_primary: R1CSShape<G1>,
  constraints_path_primary: Vec<String>,
  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<G1>,
  ck_secondary: Option<CommitmentKey<G2>>,
  r1cs_shape_secondary: R1CSShape<G2>,
  constraints_path_secondary: Vec<String>,
  augmented_circuit_params_primary: CircuitParams,
  augmented_circuit_params_secondary: CircuitParams,
  digest: G1::Scalar, // digest of everything else with this field set to G1::Scalar::ZERO
}

impl<G1, G2> PublicParams<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  /// Create a new `PublicParams`
  pub fn setup_without_commitkey<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    c_primary: C1,
    c_secondary: C2,
    num_augmented_circuits: usize,
  ) -> Self where {
    let augmented_circuit_params_primary = CircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let augmented_circuit_params_secondary = CircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);

    let ro_consts_primary: ROConstants<G1> = ROConstants::<G1>::new();
    let ro_consts_secondary: ROConstants<G2> = ROConstants::<G2>::new();

    let F_arity_primary = c_primary.arity();
    let F_arity_secondary = c_secondary.arity();

    // ro_consts_circuit_primary are parameterized by G2 because the type alias uses G2::Base = G1::Scalar
    let ro_consts_circuit_primary: ROConstantsCircuit<G2> = ROConstantsCircuit::<G2>::new();
    let ro_consts_circuit_secondary: ROConstantsCircuit<G1> = ROConstantsCircuit::<G1>::new();

    // Initialize ck for the primary
    let circuit_primary: SuperNovaCircuit<G2, C1> = SuperNovaCircuit::new(
      augmented_circuit_params_primary.clone(),
      None,
      c_primary,
      ro_consts_circuit_primary.clone(),
      num_augmented_circuits,
    );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);

    // We use the largest commitment_key for all instances
    let r1cs_shape_primary = cs.r1cs_shape();
    let constraints_path_primary = cs
      .constraints
      .iter()
      .map(|constraint| constraint.3.clone())
      .collect();

    // Initialize ck for the secondary
    let circuit_secondary: SuperNovaCircuit<G1, C2> = SuperNovaCircuit::new(
      augmented_circuit_params_secondary.clone(),
      None,
      c_secondary,
      ro_consts_circuit_secondary.clone(),
      num_augmented_circuits,
    );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let constraints_path_secondary = cs
      .constraints
      .iter()
      .map(|constraint| constraint.3.clone())
      .collect();

    let r1cs_shape_secondary = cs.r1cs_shape();

    let pp = Self {
      F_arity_primary,
      F_arity_secondary,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary: None,
      r1cs_shape_primary,
      constraints_path_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary: None,
      r1cs_shape_secondary,
      constraints_path_secondary,
      augmented_circuit_params_primary,
      augmented_circuit_params_secondary,
      digest: G1::Scalar::ZERO, // digest will be set later once commitkey ready
    };

    pp
  }

  #[allow(dead_code)]
  /// Returns the number of constraints in the primary and secondary circuits
  pub fn num_constraints(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_cons,
      self.r1cs_shape_secondary.num_cons,
    )
  }

  #[allow(dead_code)]
  /// Returns the number of variables in the primary and secondary circuits
  pub fn num_variables(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_vars,
      self.r1cs_shape_secondary.num_vars,
    )
  }
}

/// SuperNova takes Ui a list of running instances.
/// One instance of Ui is a struct called RunningClaim.
pub struct RunningClaim<G1, G2, Ca, Cb>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  Ca: StepCircuit<G1::Scalar>,
  Cb: StepCircuit<G2::Scalar>,
{
  _phantom: PhantomData<G1>,
  augmented_circuit_index: usize,
  c_primary: Ca,
  c_secondary: Cb,
  params: PublicParams<G1, G2>,
}

impl<G1, G2, Ca, Cb> RunningClaim<G1, G2, Ca, Cb>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  Ca: StepCircuit<G1::Scalar>,
  Cb: StepCircuit<G2::Scalar>,
{
  /// new a running claim
  pub fn new(
    augmented_circuit_index: usize,
    circuit_primary: Ca,
    circuit_secondary: Cb,
    num_augmented_circuits: usize,
  ) -> Self {
    let claim = circuit_primary.clone();

    let pp = PublicParams::<G1, G2>::setup_without_commitkey(
      claim.clone(),
      circuit_secondary.clone(),
      num_augmented_circuits,
    );

    Self {
      augmented_circuit_index,
      _phantom: PhantomData,
      c_primary: claim,
      c_secondary: circuit_secondary.clone(),
      params: pp,
    }
  }

  /// get augmented_circuit_index
  pub fn get_augmented_circuit_index(&self) -> usize {
    self.augmented_circuit_index
  }

  /// set primary/secondary commitment key
  pub fn set_commitmentkey(
    &mut self,
    ck_primary: CommitmentKey<G1>,
    ck_secondary: CommitmentKey<G2>,
  ) {
    self.params.ck_primary = Some(ck_primary);
    self.params.ck_secondary = Some(ck_secondary);
    self.params.digest = compute_digest::<G1, PublicParams<G1, G2>>(&self.params);
  }

  /// get primary/secondary circuit r1cs shape
  pub fn get_r1cs_shape(&self) -> (&R1CSShape<G1>, &R1CSShape<G2>) {
    (
      &self.params.r1cs_shape_primary,
      &self.params.r1cs_shape_secondary,
    )
  }
}

/// A SNARK that proves the correct execution of an non-uniform incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  r_W_primary: Vec<Option<RelaxedR1CSWitness<G1>>>,
  r_U_primary: Vec<Option<RelaxedR1CSInstance<G1>>>,
  r_W_secondary: RelaxedR1CSWitness<G2>,
  r_U_secondary: RelaxedR1CSInstance<G2>,
  l_w_secondary: R1CSWitness<G2>,
  l_u_secondary: R1CSInstance<G2>,
  pp_digest: G1::Scalar,
  i: usize,
  zi_primary: Vec<G1::Scalar>,
  zi_secondary: Vec<G2::Scalar>,
  program_counter: G1::Scalar,
  augmented_circuit_index: usize,
  num_augmented_circuits: usize,
}

impl<G1, G2> RecursiveSNARK<G1, G2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
{
  /// iterate base step to get new instance of recursive SNARK
  pub fn iter_base_step<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    claim: &RunningClaim<G1, G2, C1, C2>,
    initial_program_counter: G1::Scalar,
    first_augmented_circuit_index: usize,
    num_augmented_circuits: usize,
    z0_primary: &Vec<G1::Scalar>,
    z0_secondary: &Vec<G2::Scalar>,
  ) -> Result<Self, NovaError> {
    let pp = &claim.params;
    let c_primary = &claim.c_primary;
    let c_secondary = &claim.c_secondary;
    // commitment key for primary & secondary circuit
    let ck_primary = pp.ck_primary.as_ref().ok_or_else(|| NovaError::MissingCK)?;
    let ck_secondary = pp
      .ck_secondary
      .as_ref()
      .ok_or_else(|| NovaError::MissingCK)?;

    // base case for the primary
    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs_primary: CircuitInputs<G2> = CircuitInputs::new(
      scalar_as_base::<G1>(pp.digest),
      G2::Scalar::ZERO,
      G1::Scalar::ZERO,
      z0_primary.clone(),
      None,
      None,
      None,
      None,
      initial_program_counter,
      G1::Scalar::ZERO, // set augmented circuit index selector to 0 in base case
    );

    let circuit_primary: SuperNovaCircuit<G2, C1> = SuperNovaCircuit::new(
      pp.augmented_circuit_params_primary.clone(),
      Some(inputs_primary),
      c_primary.clone(),
      pp.ro_consts_circuit_primary.clone(),
      num_augmented_circuits,
    );

    let _ = circuit_primary
      .synthesize(&mut cs_primary)
      .map_err(|_e| NovaError::SynthesisError(_e.to_string()))?;
    let (u_primary, w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &ck_primary)
      .map_err(|_e| NovaError::UnSat)?;

    // base case for the secondary
    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs_secondary: CircuitInputs<G1> = CircuitInputs::new(
      pp.digest,
      G1::Scalar::ZERO,
      G2::Scalar::ZERO,
      z0_secondary.clone(),
      None,
      None,
      Some(u_primary.clone()),
      None,
      G2::Scalar::ZERO, // secondary circuit never constrain/bump program counter
      G2::Scalar::ZERO, // set augmented circuit index selector to 0 in base case
    );
    let circuit_secondary: SuperNovaCircuit<G1, C2> = SuperNovaCircuit::new(
      pp.augmented_circuit_params_secondary.clone(),
      Some(inputs_secondary),
      c_secondary.clone(),
      pp.ro_consts_circuit_secondary.clone(),
      num_augmented_circuits,
    );
    let _ = circuit_secondary
      .synthesize(&mut cs_secondary)
      .map_err(|_e| NovaError::SynthesisError(_e.to_string()))?;
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &ck_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // IVC proof for the primary circuit
    let l_w_primary = w_primary;
    let l_u_primary = u_primary;
    let r_W_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &l_w_primary);

    let r_U_primary =
      RelaxedR1CSInstance::from_r1cs_instance(ck_primary, &pp.r1cs_shape_primary, &l_u_primary);

    // IVC proof of the secondary circuit
    let l_w_secondary = w_secondary;
    let l_u_secondary = u_secondary;
    let r_W_secondary = RelaxedR1CSWitness::<G2>::default(&pp.r1cs_shape_secondary);
    let r_U_secondary = RelaxedR1CSInstance::default(ck_secondary, &pp.r1cs_shape_secondary);

    // Outputs of the two circuits and next program counter thus far.
    let (zi_primary_pc_next, zi_primary) = c_primary.output(initial_program_counter, &z0_primary);
    let (_, zi_secondary) = c_secondary.output(G2::Scalar::ZERO, &z0_secondary);

    if zi_primary.len() != pp.F_arity_primary || zi_secondary.len() != pp.F_arity_secondary {
      return Err(NovaError::InvalidStepOutputLength);
    }

    // handle the base case by initialize U_next in next round
    let r_W_primary_initial_list = (0..num_augmented_circuits)
      .map(|i| {
        if i == first_augmented_circuit_index {
          Some(r_W_primary.clone())
        } else {
          None
        }
      })
      .collect::<Vec<Option<RelaxedR1CSWitness<G1>>>>();

    let r_U_primary_initial_list = (0..num_augmented_circuits)
      .map(|i| {
        if i == first_augmented_circuit_index {
          Some(r_U_primary.clone())
        } else {
          None
        }
      })
      .collect::<Vec<Option<RelaxedR1CSInstance<G1>>>>();

    Ok(Self {
      r_W_primary: r_W_primary_initial_list,
      r_U_primary: r_U_primary_initial_list,
      r_W_secondary: r_W_secondary,
      r_U_secondary: r_U_secondary,
      l_w_secondary,
      l_u_secondary,
      pp_digest: pp.digest,
      i: 0_usize, // after base case, next iteration start from 1
      zi_primary,
      zi_secondary,
      program_counter: zi_primary_pc_next,
      augmented_circuit_index: first_augmented_circuit_index,
      num_augmented_circuits,
    })
  }
  /// executing a step of the incremental computation
  pub fn prove_step<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    &mut self,
    claim: &RunningClaim<G1, G2, C1, C2>,
    z0_primary: &Vec<G1::Scalar>,
    z0_secondary: &Vec<G2::Scalar>,
  ) -> Result<(), NovaError> {
    // Frist step was already done in the constructor
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    let pp = &claim.params;
    let c_primary = &claim.c_primary;
    let c_secondary = &claim.c_secondary;
    // commitment key for primary & secondary circuit
    let ck_primary = pp.ck_primary.as_ref().ok_or_else(|| NovaError::MissingCK)?;
    let ck_secondary = pp
      .ck_secondary
      .as_ref()
      .ok_or_else(|| NovaError::MissingCK)?;

    if z0_primary.len() != pp.F_arity_primary || z0_secondary.len() != pp.F_arity_secondary {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // fold the secondary circuit's instance
    let (nifs_secondary, (r_U_secondary_folded, r_W_secondary_folded)) = NIFS::prove(
      ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<G1>(self.pp_digest),
      &pp.r1cs_shape_secondary,
      &self.r_U_secondary,
      &self.r_W_secondary,
      &self.l_u_secondary,
      &self.l_w_secondary,
    )?;

    // clone and updated running instance on respective circuit_index
    let r_U_secondary_next = r_U_secondary_folded;
    let r_W_secondary_next = r_W_secondary_folded;

    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs_primary: CircuitInputs<G2> = CircuitInputs::new(
      scalar_as_base::<G1>(pp.digest),
      scalar_as_base::<G1>(self.pp_digest),
      G1::Scalar::from(self.i as u64),
      z0_primary.clone(),
      Some(self.zi_primary.clone()),
      Some(vec![self.r_U_secondary.clone()]),
      Some(self.l_u_secondary.clone()),
      Some(Commitment::<G2>::decompress(&nifs_secondary.comm_T)?),
      self.program_counter,
      G1::Scalar::ZERO,
    );

    let circuit_primary: SuperNovaCircuit<G2, C1> = SuperNovaCircuit::new(
      pp.augmented_circuit_params_primary.clone(),
      Some(inputs_primary),
      c_primary.clone(),
      pp.ro_consts_circuit_primary.clone(),
      self.num_augmented_circuits,
    );

    let _ = circuit_primary.synthesize(&mut cs_primary);

    let (l_u_primary, l_w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &ck_primary)
      .map_err(|_e| NovaError::UnSat)?;

    // Performance Note: U1, W1 clone() at most happened #num_augmented_circuit times (Each clone only on specific index)
    let (nifs_primary, (r_U_primary_folded, r_W_primary_folded)) = NIFS::prove(
      ck_primary,
      &pp.ro_consts_primary,
      &self.pp_digest,
      &pp.r1cs_shape_primary,
      &self
        .r_U_primary
        .get(claim.get_augmented_circuit_index())
        .unwrap_or(&None)
        .clone()
        .unwrap_or_else(|| RelaxedR1CSInstance::default(ck_primary, &pp.r1cs_shape_primary)),
      &self
        .r_W_primary
        .get(claim.get_augmented_circuit_index())
        .unwrap_or(&None)
        .clone()
        .unwrap_or_else(|| RelaxedR1CSWitness::default(&pp.r1cs_shape_primary)),
      &l_u_primary,
      &l_w_primary,
    )?;

    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs_secondary: CircuitInputs<G1> = CircuitInputs::new(
      pp.digest,
      self.pp_digest,
      G2::Scalar::from(self.i as u64),
      z0_secondary.clone(),
      Some(self.zi_secondary.clone()),
      Some(
        self
          .r_U_primary
          .iter()
          .map(|U| {
            U.clone()
              .unwrap_or_else(|| RelaxedR1CSInstance::default(ck_primary, &pp.r1cs_shape_primary))
          })
          .collect(),
      ),
      Some(l_u_primary),
      Some(Commitment::<G1>::decompress(&nifs_primary.comm_T)?),
      G2::Scalar::ZERO, // secondary circuit never constrain/bump program counter
      G2::Scalar::from(claim.get_augmented_circuit_index() as u64),
    );

    let circuit_secondary: SuperNovaCircuit<G1, C2> = SuperNovaCircuit::new(
      pp.augmented_circuit_params_secondary.clone(),
      Some(inputs_secondary),
      c_secondary.clone(),
      pp.ro_consts_circuit_secondary.clone(),
      self.num_augmented_circuits,
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);

    let (l_u_secondary_next, l_w_secondary_next) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &ck_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // update the running instances and witnesses
    let (zi_primary_pc_next, zi_primary) = c_primary.output(self.program_counter, &self.zi_primary);
    let (_, zi_secondary) = c_secondary.output(G2::Scalar::ZERO, &self.zi_secondary);

    if zi_primary.len() != pp.F_arity_primary || zi_secondary.len() != pp.F_arity_secondary {
      return Err(NovaError::InvalidStepOutputLength);
    }

    // clone and updated running instance on respective circuit_index
    self.r_U_primary[claim.get_augmented_circuit_index()] = Some(r_U_primary_folded);
    self.r_W_primary[claim.get_augmented_circuit_index()] = Some(r_W_primary_folded);
    self.r_W_secondary = r_W_secondary_next;
    self.r_U_secondary = r_U_secondary_next;
    self.l_w_secondary = l_w_secondary_next;
    self.l_u_secondary = l_u_secondary_next;
    self.pp_digest = pp.digest;
    self.i += 1;
    self.zi_primary = zi_primary;
    self.zi_secondary = zi_secondary;
    self.program_counter = zi_primary_pc_next;
    self.augmented_circuit_index = claim.get_augmented_circuit_index();
    Ok(())
  }

  /// verify recursive snark
  pub fn verify<C1: StepCircuit<G1::Scalar>, C2: StepCircuit<G2::Scalar>>(
    &mut self,
    claim: &RunningClaim<G1, G2, C1, C2>,
    z0_primary: &Vec<G1::Scalar>,
    z0_secondary: &Vec<G2::Scalar>,
  ) -> Result<(), NovaError> {
    // number of steps cannot be zero
    if self.i == 0 {
      println!("must verify on valid RecursiveSNARK where i > 0");
      return Err(NovaError::ProofVerifyError);
    }

    // check the (relaxed) R1CS instances public outputs.
    if self.l_u_secondary.X.len() != 2 {
      return Err(NovaError::ProofVerifyError);
    }

    let pp = &claim.params;
    let ck_primary = pp.ck_primary.as_ref().ok_or_else(|| NovaError::MissingCK)?;

    self.r_U_primary.iter().try_for_each(|U| match U.clone() {
      Some(U) if U.X.len() != 2 => {
        println!("r_U_primary got instance length {:?} != {:?}", U.X.len(), 2);
        Err(NovaError::ProofVerifyError)
      }
      _ => Ok(()),
    })?;

    if self.r_U_secondary.X.len() != 2 {
      println!(
        "r_U_secondary got instance length {:?} != {:?}",
        self.r_U_secondary.X.len(),
        2
      );
      return Err(NovaError::ProofVerifyError);
    };

    // check if the output hashes in R1CS instances point to the right running instances
    let num_field_primary_ro = 3 // params_next, i_new, program_counter_new
    + 2 * pp.F_arity_primary // zo, z1
    + 1 * (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // #num_augmented_circuits * (7 + [X0, X1]*#num_limb)
    let num_field_secondary_ro = 3 // params_next, i_new, program_counter_new
    + 2 * pp.F_arity_primary // zo, z1
    + self.num_augmented_circuits * (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // #num_augmented_circuits * (7 + [X0, X1]*#num_limb)

    let (hash_primary, hash_secondary) = {
      let mut hasher = <G2 as Group>::RO::new(pp.ro_consts_secondary.clone(), num_field_primary_ro);
      hasher.absorb(pp.digest);
      hasher.absorb(G1::Scalar::from(self.i as u64));
      hasher.absorb(self.program_counter);

      for e in z0_primary {
        hasher.absorb(*e);
      }
      for e in &self.zi_primary {
        hasher.absorb(*e);
      }
      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 =
        <G1 as Group>::RO::new(pp.ro_consts_primary.clone(), num_field_secondary_ro);
      hasher2.absorb(scalar_as_base::<G1>(pp.digest));
      hasher2.absorb(G2::Scalar::from(self.i as u64));
      hasher2.absorb(G2::Scalar::ZERO);
      for e in z0_secondary {
        hasher2.absorb(*e);
      }
      for e in &self.zi_secondary {
        hasher2.absorb(*e);
      }
      let default_value = RelaxedR1CSInstance::default(ck_primary, &pp.r1cs_shape_primary);
      self.r_U_primary.iter().for_each(|U| {
        U.as_ref()
          .unwrap_or_else(|| &default_value)
          .absorb_in_ro(&mut hasher2);
      });

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    if hash_primary != self.l_u_secondary.X[0] {
      println!(
        "hash_primary {:?} not equal l_u_secondary.X[0] {:?}",
        hash_primary, self.l_u_secondary.X[0]
      );
      return Err(NovaError::ProofVerifyError);
    }
    if hash_secondary != scalar_as_base::<G2>(self.l_u_secondary.X[1]) {
      println!(
        "hash_secondary {:?} not equal l_u_secondary.X[1] {:?}",
        hash_secondary, self.l_u_secondary.X[1]
      );
      return Err(NovaError::ProofVerifyError);
    }

    // check the satisfiability of the provided `circuit_index` instance
    let default_instance = RelaxedR1CSInstance::default(ck_primary, &pp.r1cs_shape_primary);
    let default_witness = RelaxedR1CSWitness::default(&pp.r1cs_shape_primary);
    let (res_r_primary, (res_r_secondary, res_l_secondary)) = rayon::join(
      || {
        pp.r1cs_shape_primary.is_sat_relaxed(
          pp.ck_primary.as_ref().unwrap(),
          &self.r_U_primary[claim.get_augmented_circuit_index()]
            .as_ref()
            .unwrap_or_else(|| &default_instance),
          &self.r_W_primary[claim.get_augmented_circuit_index()]
            .as_ref()
            .unwrap_or_else(|| &default_witness),
        )
      },
      || {
        rayon::join(
          || {
            pp.r1cs_shape_secondary.is_sat_relaxed(
              pp.ck_secondary.as_ref().unwrap(),
              &self.r_U_secondary,
              &self.r_W_secondary,
            )
          },
          || {
            pp.r1cs_shape_secondary.is_sat(
              &pp.ck_secondary.as_ref().unwrap(),
              &self.l_u_secondary,
              &self.l_w_secondary,
            )
          },
        )
      },
    );

    res_r_primary.map_err(|err| match err {
      NovaError::UnSatIndex(i) => NovaError::UnSatMsg(format!(
        "res_r_primary is_sat_relaxed relation failed at constraint path {:?}",
        pp.constraints_path_primary
          .get(i)
          .unwrap_or(&"".to_string())
      )),
      e => e,
    })?;
    res_r_secondary.map_err(|err| match err {
      NovaError::UnSatIndex(i) => NovaError::UnSatMsg(format!(
        "res_r_secondary is_sat_relaxed relation failed at constraint path {:?}",
        pp.constraints_path_primary
          .get(i)
          .unwrap_or(&"".to_string())
      )),
      e => e,
    })?;
    res_l_secondary.map_err(|err| match err {
      NovaError::UnSatIndex(i) => NovaError::UnSatMsg(format!(
        "res_l_secondary is_sat relation failed at constraint path {:?}",
        pp.constraints_path_primary
          .get(i)
          .unwrap_or(&"".to_string())
      )),
      e => e,
    })?;

    Ok(())
  }

  /// get program counter
  pub fn get_program_counter(&self) -> G1::Scalar {
    self.program_counter
  }
}

/// genenate commitmentkey by r1cs shape
pub fn gen_commitmentkey_by_r1cs<G: Group>(shape: &R1CSShape<G>) -> CommitmentKey<G> {
  R1CS::<G>::commitment_key(shape)
}

#[cfg(test)]
mod tests {
  use crate::gadgets::utils::alloc_num_equals;
  use crate::gadgets::utils::conditionally_select;
  use crate::traits::circuit_supernova::TrivialTestCircuit;
  use crate::{
    compute_digest,
    gadgets::utils::{add_allocated_num, alloc_one, alloc_zero},
  };
  use bellperson::gadgets::boolean::Boolean;
  use core::marker::PhantomData;
  use ff::Field;
  use ff::PrimeField;

  use super::*;

  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};

  fn constraint_curcuit_index<F: PrimeField, CS: ConstraintSystem<F>>(
    mut cs: CS,
    pc_counter: &AllocatedNum<F>,
    z: &[AllocatedNum<F>],
    circuit_index: usize,
    rom_offset: usize,
  ) -> Result<(), SynthesisError> {
    let circuit_index = AllocatedNum::alloc(cs.namespace(|| "circuit_index"), || {
      Ok(F::from(circuit_index as u64))
    })?;
    let pc_offset_in_z = AllocatedNum::alloc(cs.namespace(|| "pc_offset_in_z"), || {
      Ok(pc_counter.get_value().unwrap() + F::from(rom_offset as u64))
    })?;

    // select target when index match or empty
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let _selected_circuit_index = z
      .iter()
      .enumerate()
      .map(|(i, z)| {
        let i_alloc = AllocatedNum::alloc(
          cs.namespace(|| format!("_selected_circuit_index i{:?} allocated", i)),
          || Ok(F::from(i as u64)),
        )?;
        let equal_bit = Boolean::from(alloc_num_equals(
          cs.namespace(|| format!("check selected_circuit_index {:?} equal bit", i)),
          &i_alloc,
          &pc_offset_in_z,
        )?);
        conditionally_select(
          cs.namespace(|| format!("select on index namespace {:?}", i)),
          &z,
          &zero,
          &equal_bit,
        )
      })
      .collect::<Result<Vec<AllocatedNum<F>>, SynthesisError>>()?;

    let selected_circuit_index =
      _selected_circuit_index
        .iter()
        .enumerate()
        .try_fold(zero, |agg, (i, _circuit_index)| {
          add_allocated_num(
            cs.namespace(|| format!("selected_circuit_index {:?}", i)),
            _circuit_index,
            &agg,
          )
        })?;

    cs.enforce(
      || "selected_circuit_index == circuit_index",
      |lc| lc + selected_circuit_index.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + circuit_index.get_variable(),
    );

    Ok(())
  }

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
    circuit_index: usize,
    rom_size: usize,
  }

  impl<F> CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn new(circuit_index: usize, rom_size: usize) -> Self {
      CubicCircuit {
        circuit_index,
        rom_size,
        _p: PhantomData,
      }
    }
  }

  impl<F> StepCircuit<F> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1 + self.rom_size // value + rom[].len()
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      pc_counter: &AllocatedNum<F>,
      z: &[AllocatedNum<F>],
    ) -> Result<(AllocatedNum<F>, Vec<AllocatedNum<F>>), SynthesisError> {
      // constrain rom[pc] equal to `self.circuit_index`
      constraint_curcuit_index(
        cs.namespace(|| "CubicCircuit agumented circuit constraint"),
        pc_counter,
        z,
        self.circuit_index,
        1,
      )?;

      let one = alloc_one(cs.namespace(|| "alloc one"))?;
      let pc_next = add_allocated_num(
        // pc = pc + 1
        cs.namespace(|| format!("pc = pc + 1")),
        pc_counter,
        &one,
      )?;

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

      let mut z_next = vec![y];
      z_next.extend(z[1..].iter().cloned());
      Ok((pc_next, z_next))
    }

    fn output(&self, pc: F, z: &[F]) -> (F, Vec<F>) {
      let mut z_next = vec![
        z[0] * z[0] * z[0] + z[0] + F::from(5u64), // y
      ];
      z_next.extend(z[1..].iter().cloned());
      (pc + F::from(1), z_next)
    }
  }

  #[derive(Clone, Debug, Default)]
  struct SquareCircuit<F: PrimeField> {
    _p: PhantomData<F>,
    circuit_index: usize,
    rom_size: usize,
  }

  impl<F> SquareCircuit<F>
  where
    F: PrimeField,
  {
    fn new(circuit_index: usize, rom_size: usize) -> Self {
      SquareCircuit {
        circuit_index,
        rom_size,
        _p: PhantomData,
      }
    }
  }

  impl<F> StepCircuit<F> for SquareCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1 + self.rom_size // value + rom[].len()
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      pc_counter: &AllocatedNum<F>,
      z: &[AllocatedNum<F>],
    ) -> Result<(AllocatedNum<F>, Vec<AllocatedNum<F>>), SynthesisError> {
      // constrain rom[pc] equal to `self.circuit_index`
      constraint_curcuit_index(
        cs.namespace(|| "SquareCircuit agumented circuit constraint"),
        pc_counter,
        z,
        self.circuit_index,
        1,
      )?;
      let one = alloc_one(cs.namespace(|| "alloc one"))?;
      let pc_next = add_allocated_num(
        // pc = pc + 1
        cs.namespace(|| "pc = pc + 1"),
        pc_counter,
        &one,
      )?;

      // Consider an equation: `x^2 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_sq.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

      cs.enforce(
        || "y = x^2 + x + 5",
        |lc| {
          lc + x_sq.get_variable()
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

      let mut z_next = vec![y];
      z_next.extend(z[1..].iter().cloned());
      Ok((pc_next, z_next))
    }

    fn output(&self, pc: F, z: &[F]) -> (F, Vec<F>) {
      let mut z_next = vec![
        z[0] * z[0] + z[0] + F::from(5u64), // y
      ];
      z_next.extend(z[1..].iter().cloned());
      (pc + F::from(1), z_next)
    }
  }

  const OPCODE_0: usize = 0;
  const OPCODE_1: usize = 1;
  fn test_trivial_nivc_with<G1, G2>()
  where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
  {
    // Here demo a simple RAM machine
    // - with 2 argumented circuit
    // - each argumented circuit contains primary and secondary circuit
    // - a memory commmitment via a public IO `rom` (like a program) to constraint the sequence execution

    // This test also ready to add more argumented circuit and ROM can be arbitrary length

    // ROM is for constraints the sequence of execution order for opcode
    // program counter initially point to 0

    // TODO: replace with memory commitment along with suggestion from Supernova 4.4 optimisations

    // This is mostly done with the existing Nova code. With additions of U_i[] and program_counter checks
    // in the augmented circuit.

    // To save the test time, after each step of iteration, RecursiveSNARK just verfiy the latest U_i[augmented_circuit_index] needs to be a satisfying instance.
    // TODO At the end of this test, RecursiveSNARK need to verify all U_i[] are satisfying instances

    let rom = [
      OPCODE_0, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1, OPCODE_1, OPCODE_0, OPCODE_0, OPCODE_1,
      OPCODE_1,
    ]; // Rom can be arbitrary length.
    let circuit_secondary = TrivialTestCircuit::new(rom.len());
    let num_augmented_circuit = 2;

    // Structuring running claims
    let test_circuit1 = CubicCircuit::new(OPCODE_0, rom.len());
    let mut running_claim1 = RunningClaim::<
      G1,
      G2,
      CubicCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::new(
      OPCODE_0,
      test_circuit1,
      circuit_secondary.clone(),
      num_augmented_circuit,
    );

    let test_circuit2 = SquareCircuit::new(OPCODE_1, rom.len());
    let mut running_claim2 = RunningClaim::<
      G1,
      G2,
      SquareCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::new(
      OPCODE_1,
      test_circuit2,
      circuit_secondary.clone(),
      num_augmented_circuit,
    );

    // generate the commitkey based on max num of constraints and reused it for all other augmented circuit
    let circuit_public_params = vec![&running_claim1.params, &running_claim2.params];
    let (max_index_circuit, _) = circuit_public_params
      .iter()
      .enumerate()
      .map(|(i, params)| -> (usize, usize) { (i, params.r1cs_shape_primary.num_cons) })
      .max_by(|(_, circuit_size1), (_, circuit_size2)| circuit_size1.cmp(&circuit_size2))
      .unwrap();

    let ck_primary =
      gen_commitmentkey_by_r1cs(&circuit_public_params[max_index_circuit].r1cs_shape_primary);
    let ck_secondary =
      gen_commitmentkey_by_r1cs(&circuit_public_params[max_index_circuit].r1cs_shape_secondary);

    // set unified ck_primary, ck_secondary and update digest
    running_claim1.params.ck_primary = Some(ck_primary.clone());
    running_claim1.params.ck_secondary = Some(ck_secondary.clone());
    running_claim1.params.digest =
      compute_digest::<G1, PublicParams<G1, G2>>(&running_claim1.params);

    running_claim2.params.ck_primary = Some(ck_primary.clone());
    running_claim2.params.ck_secondary = Some(ck_secondary.clone());
    running_claim2.params.digest =
      compute_digest::<G1, PublicParams<G1, G2>>(&running_claim2.params);

    let num_steps = rom.len();
    let initial_program_counter = <G1 as Group>::Scalar::from(0);

    // extend z0_primary/secondary with rom content
    let mut z0_primary = vec![<G1 as Group>::Scalar::ONE];
    z0_primary.extend(
      rom
        .iter()
        .map(|opcode| <G1 as Group>::Scalar::from(*opcode as u64)),
    );
    let mut z0_secondary = vec![<G2 as Group>::Scalar::ONE];
    z0_secondary.extend(
      rom
        .iter()
        .map(|opcode| <G2 as Group>::Scalar::from(*opcode as u64)),
    );

    let mut recursive_snark_option: Option<RecursiveSNARK<G1, G2>> = None;

    for _ in 0..num_steps {
      let program_counter = recursive_snark_option
        .as_ref()
        .map(|recursive_snark| recursive_snark.program_counter)
        .unwrap_or_else(|| initial_program_counter);
      let augmented_circuit_index = rom[u32::from_le_bytes(
        // convert program counter from field to usize (only took le 4 bytes)
        program_counter.to_repr().as_ref().clone()[0..4]
          .try_into()
          .unwrap(),
      ) as usize];

      let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
        if augmented_circuit_index == OPCODE_0 {
          RecursiveSNARK::iter_base_step(
            &running_claim1,
            program_counter,
            augmented_circuit_index,
            num_augmented_circuit,
            &z0_primary,
            &z0_secondary,
          )
          .unwrap()
        } else if augmented_circuit_index == OPCODE_1 {
          RecursiveSNARK::iter_base_step(
            &running_claim2,
            program_counter,
            augmented_circuit_index,
            num_augmented_circuit,
            &z0_primary,
            &z0_secondary,
          )
          .unwrap()
        } else {
          unimplemented!()
        }
      });

      if augmented_circuit_index == OPCODE_0 {
        let res = recursive_snark.prove_step(&running_claim1, &z0_primary, &z0_secondary);
        if let Err(e) = res.clone() {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
        let res = recursive_snark.verify(&running_claim1, &z0_primary, &z0_secondary);
        if let Err(e) = res.clone() {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
      } else if augmented_circuit_index == OPCODE_1 {
        let res = recursive_snark.prove_step(&running_claim2, &z0_primary, &z0_secondary);
        if let Err(e) = res.clone() {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
        let res = recursive_snark.verify(&running_claim2, &z0_primary, &z0_secondary);
        if let Err(e) = res.clone() {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
      }
      recursive_snark_option = Some(recursive_snark)
    }

    assert!(recursive_snark_option.is_some());

    // Now you can handle the Result using if let
    let RecursiveSNARK {
      zi_primary,
      zi_secondary,
      program_counter,
      ..
    } = &recursive_snark_option.unwrap();

    println!("zi_primary: {:?}", zi_primary);
    println!("zi_secondary: {:?}", zi_secondary);
    println!("final program_counter: {:?}", program_counter);
  }

  #[test]
  fn test_trivial_nivc() {
    type G1 = pasta_curves::pallas::Point;
    type G2 = pasta_curves::vesta::Point;

    //Expirementing with selecting the running claims for nifs
    test_trivial_nivc_with::<G1, G2>();
  }
}
