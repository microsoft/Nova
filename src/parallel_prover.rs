#![allow(unused_imports)]
#![allow(unused)]
//! There are two Verification Circuits. The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! Each recursive tree node has both an aggregated and new instance
//! of both the primary and secondary circuit. As you merge the nodes
//! the proofs verify that three folding are correct and merge the
//! running instances and new instances in a pair of nodes to a single
//! running instance.
//! We check that hash(index start, index end, z_start, z_end) has been
//! committed properly for each node.
//! The circuit also checks that when F is executed on the left nodes
//! z_end that the output is z_start of the right node

use crate::{
  bellperson::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  circuit::NovaAugmentedCircuitParams,
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
    utils::{alloc_num_equals, alloc_scalar_as_base, conditionally_select_vec, le_bits_to_num},
  },
  nifs::NIFS,
  parallel_circuit::{NovaAugmentedParallelCircuit, NovaAugmentedParallelCircuitInputs},
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    circuit::StepCircuit,
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    snark::RelaxedR1CSSNARKTrait,
    AbsorbInROTrait, Group, ROConstants, ROConstantsCircuit, ROConstantsTrait, ROTrait,
  },
  Commitment,
};
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    Assignment,
  },
  Circuit, ConstraintSystem, SynthesisError,
};
use core::marker::PhantomData;
use ff::Field;
use serde::{Deserialize, Serialize};

// TODO - This is replicated from lib but we should actually instead have another file for it and use both here and there

type CommitmentKey<G> = <<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey;

/// A type that holds public parameters of Nova
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  F_arity_primary: usize,
  F_arity_secondary: usize,
  ro_consts_primary: ROConstants<G1>,
  ro_consts_circuit_primary: ROConstantsCircuit<G2>,
  ck_primary: CommitmentKey<G1>,
  r1cs_shape_primary: R1CSShape<G1>,
  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<G1>,
  ck_secondary: CommitmentKey<G2>,
  r1cs_shape_secondary: R1CSShape<G2>,
  augmented_circuit_params_primary: NovaAugmentedCircuitParams,
  augmented_circuit_params_secondary: NovaAugmentedCircuitParams,
  _p_c1: PhantomData<C1>,
  _p_c2: PhantomData<C2>,
}

impl<G1, G2, C1, C2> PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  /// Create a new `PublicParams`
  pub fn setup(c_primary: C1, c_secondary: C2) -> Self {
    let augmented_circuit_params_primary =
      NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let augmented_circuit_params_secondary =
      NovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);

    let ro_consts_primary: ROConstants<G1> = ROConstants::<G1>::new();
    let ro_consts_secondary: ROConstants<G2> = ROConstants::<G2>::new();

    let F_arity_primary = c_primary.arity();
    let F_arity_secondary = c_secondary.arity();

    // ro_consts_circuit_primary are parameterized by G2 because the type alias uses G2::Base = G1::Scalar
    let ro_consts_circuit_primary: ROConstantsCircuit<G2> = ROConstantsCircuit::<G2>::new();
    let ro_consts_circuit_secondary: ROConstantsCircuit<G1> = ROConstantsCircuit::<G1>::new();

    // Initialize ck for the primary
    let circuit_primary: NovaAugmentedParallelCircuit<G2, C1> = NovaAugmentedParallelCircuit::new(
      augmented_circuit_params_primary.clone(),
      None,
      c_primary,
      ro_consts_circuit_primary.clone(),
    );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (r1cs_shape_primary, ck_primary) = cs.r1cs_shape();

    // Initialize ck for the secondary
    let circuit_secondary: NovaAugmentedParallelCircuit<G1, C2> = NovaAugmentedParallelCircuit::new(
      augmented_circuit_params_secondary.clone(),
      None,
      c_secondary,
      ro_consts_circuit_secondary.clone(),
    );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let (r1cs_shape_secondary, ck_secondary) = cs.r1cs_shape();

    Self {
      F_arity_primary,
      F_arity_secondary,
      ro_consts_primary,
      ro_consts_circuit_primary,
      ck_primary,
      r1cs_shape_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      ck_secondary,
      r1cs_shape_secondary,
      augmented_circuit_params_primary,
      augmented_circuit_params_secondary,
      _p_c1: Default::default(),
      _p_c2: Default::default(),
    }
  }

  /// Returns the number of constraints in the primary and secondary circuits
  pub fn num_constraints(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_cons,
      self.r1cs_shape_secondary.num_cons,
    )
  }

  /// Returns the number of variables in the primary and secondary circuits
  pub fn num_variables(&self) -> (usize, usize) {
    (
      self.r1cs_shape_primary.num_vars,
      self.r1cs_shape_secondary.num_vars,
    )
  }
}

// This ends the 1 to 1 copied code

/// A type that holds one node the tree based nova proof. This will have both running instances and fresh instances
/// of the primary and secondary circuit.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NovaTreeNode<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  // The running instance of the primary
  W_primary: RelaxedR1CSWitness<G1>,
  U_primary: RelaxedR1CSInstance<G1>,
  // The new instance of the primary
  w_primary: RelaxedR1CSWitness<G1>,
  u_primary: RelaxedR1CSInstance<G1>,
  // The running instance of the secondary
  W_secondary: RelaxedR1CSWitness<G2>,
  U_secondary: RelaxedR1CSInstance<G2>,
  // The running instance of the secondary
  w_secondary: RelaxedR1CSWitness<G2>,
  u_secondary: RelaxedR1CSInstance<G2>,
  i_start: u64,
  i_end: u64,
  z_start_primary: Vec<G1::Scalar>,
  z_end_primary: Vec<G1::Scalar>,
  z_start_secondary: Vec<G2::Scalar>,
  z_end_secondary: Vec<G2::Scalar>,
  _p_c1: PhantomData<C1>,
  _p_c2: PhantomData<C2>,
}

impl<G1, G2, C1, C2> NovaTreeNode<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  /// Creates a tree node which proves one computation and runs a base case F' proof. The running instances
  /// are set to defaults and the new proofs are set ot this base case proof.
  pub fn new(
    pp: &PublicParams<G1, G2, C1, C2>,
    c_primary: C1,
    c_secondary: C2,
    i: u64,
    z_start_primary: Vec<G1::Scalar>,
    z_end_primary: Vec<G1::Scalar>,
    z_start_secondary: Vec<G2::Scalar>,
    z_end_secondary: Vec<G2::Scalar>,
  ) -> Result<Self, NovaError> {
    // base case for the primary
    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs_primary: NovaAugmentedParallelCircuitInputs<G2> =
      NovaAugmentedParallelCircuitInputs::new(
        pp.r1cs_shape_secondary.get_digest(),
        G1::Scalar::from(i.try_into().unwrap()),
        G1::Scalar::from((i + 1).try_into().unwrap()),
        G1::Scalar::from((i).try_into().unwrap()),
        G1::Scalar::from((i + 1).try_into().unwrap()),
        z_start_primary.clone(),
        z_start_primary.clone(),
        z_end_primary.clone(),
        z_end_primary.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
      );

    let circuit_primary: NovaAugmentedParallelCircuit<G2, C1> = NovaAugmentedParallelCircuit::new(
      pp.augmented_circuit_params_primary.clone(),
      Some(inputs_primary),
      c_primary.clone(),
      pp.ro_consts_circuit_primary.clone(),
    );
    let _ = circuit_primary.synthesize(&mut cs_primary);
    let (u_primary, w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)
      .map_err(|_e| NovaError::UnSat)?;

    // base case for the secondary
    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();

    let inputs_secondary: NovaAugmentedParallelCircuitInputs<G1> =
      NovaAugmentedParallelCircuitInputs::new(
        pp.r1cs_shape_primary.get_digest(),
        G2::Scalar::from(i),
        G2::Scalar::from(i + 1),
        G2::Scalar::from(i),
        G2::Scalar::from(i + 1),
        z_start_secondary.clone(),
        z_start_secondary.clone(),
        z_end_secondary.clone(),
        z_end_secondary.clone(),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
      );
    let circuit_secondary: NovaAugmentedParallelCircuit<G1, C2> = NovaAugmentedParallelCircuit::new(
      pp.augmented_circuit_params_secondary.clone(),
      Some(inputs_secondary),
      c_secondary.clone(),
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // IVC proof for the primary circuit
    let w_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &w_primary);
    let u_primary =
      RelaxedR1CSInstance::from_r1cs_instance(&pp.ck_primary, &pp.r1cs_shape_primary, &u_primary);
    let W_primary = w_primary.clone();
    let U_primary = u_primary.clone();

    // IVC proof of the secondary circuit
    let w_secondary =
      RelaxedR1CSWitness::<G2>::from_r1cs_witness(&pp.r1cs_shape_secondary, &w_secondary);
    let u_secondary = RelaxedR1CSInstance::<G2>::from_r1cs_instance(
      &pp.ck_secondary,
      &pp.r1cs_shape_secondary,
      &u_secondary,
    );
    let W_secondary = w_secondary.clone();
    let U_secondary = u_secondary.clone();

    if z_start_primary.len() != pp.F_arity_primary
      || z_start_secondary.len() != pp.F_arity_secondary
    {
      return Err(NovaError::InvalidStepOutputLength);
    }

    let i_start = i;
    let i_end = i + 1;

    Ok(Self {
      W_primary,
      U_primary,
      w_primary,
      u_primary,
      W_secondary,
      U_secondary,
      w_secondary,
      u_secondary,
      i_start,
      i_end,
      z_start_primary,
      z_end_primary,
      z_start_secondary,
      z_end_secondary,
      _p_c1: Default::default(),
      _p_c2: Default::default(),
    })
  }

  /// Merges another node into this node. The node this is called on is treated as the left node and the node which is
  /// consumed is treated as the right node.
  pub fn merge(
    self,
    right: NovaTreeNode<G1, G2, C1, C2>,
    pp: &PublicParams<G1, G2, C1, C2>,
    c_primary: &C1,
    c_secondary: &C2,
  ) -> Result<Self, NovaError> {
    // We have to merge two proofs where the right starts one index after the left ends
    // note that this would fail in the proof step but we error earlier here for debugging clarity.
    if self.i_end + 1 != right.i_start {
      return Err(NovaError::InvalidNodeMerge);
    }

    // First we fold the secondary instances of both the left and right children in the secondary curve
    let (nifs_left_secondary, (left_U_secondary, left_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &self.U_secondary,
      &self.W_secondary,
      &self.u_secondary,
      &self.w_secondary,
    )?;
    let (nifs_right_secondary, (right_U_secondary, right_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &right.U_secondary,
      &right.W_secondary,
      &right.u_secondary,
      &right.w_secondary,
    )?;
    let (nifs_secondary, (U_secondary, W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &left_U_secondary,
      &left_W_secondary,
      &right_U_secondary,
      &right_W_secondary,
    )?;

    // Next we construct a proof of this folding and of the invocation of F

    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();

    let inputs_primary: NovaAugmentedParallelCircuitInputs<G2> =
      NovaAugmentedParallelCircuitInputs::new(
        pp.r1cs_shape_secondary.get_digest(),
        G1::Scalar::from(self.i_start as u64),
        G1::Scalar::from(self.i_end as u64),
        G1::Scalar::from(right.i_start as u64),
        G1::Scalar::from(right.i_end as u64),
        self.z_start_primary.clone(),
        self.z_end_primary,
        right.z_start_primary,
        right.z_end_primary.clone(),
        Some(self.U_secondary),
        Some(self.u_secondary),
        Some(right.U_secondary),
        Some(right.u_secondary),
        Some(Commitment::<G2>::decompress(&nifs_left_secondary.comm_T)?),
        Some(Commitment::<G2>::decompress(&nifs_right_secondary.comm_T)?),
        Some(Commitment::<G2>::decompress(&nifs_secondary.comm_T)?),
      );

    let circuit_primary: NovaAugmentedParallelCircuit<G2, C1> = NovaAugmentedParallelCircuit::new(
      pp.augmented_circuit_params_primary.clone(),
      Some(inputs_primary),
      c_primary.clone(),
      pp.ro_consts_circuit_primary.clone(),
    );
    let _ = circuit_primary.synthesize(&mut cs_primary);

    let (u_primary, w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.ck_primary)
      .map_err(|_e| NovaError::UnSat)?;

    let u_primary =
      RelaxedR1CSInstance::from_r1cs_instance(&pp.ck_primary, &pp.r1cs_shape_primary, &u_primary);
    let w_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &w_primary);

    // Now we fold the instances of the primary proof
    let (nifs_left_primary, (left_U_primary, left_W_primary)) = NIFS::prove(
      &pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.r1cs_shape_primary,
      &self.U_primary,
      &self.W_primary,
      &self.u_primary,
      &self.w_primary,
    )?;
    let (nifs_right_primary, (right_U_primary, right_W_primary)) = NIFS::prove(
      &pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.r1cs_shape_primary,
      &right.U_primary,
      &right.W_primary,
      &right.u_primary,
      &right.w_primary,
    )?;
    let (nifs_primary, (U_primary, W_primary)) = NIFS::prove(
      &pp.ck_primary,
      &pp.ro_consts_primary,
      &pp.r1cs_shape_primary,
      &left_U_primary,
      &left_W_primary,
      &right_U_primary,
      &right_W_primary,
    )?;

    // Next we construct a proof of this folding in the secondary curve
    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();

    let inputs_secondary: NovaAugmentedParallelCircuitInputs<G1> =
      NovaAugmentedParallelCircuitInputs::<G1>::new(
        pp.r1cs_shape_primary.get_digest(),
        G2::Scalar::from(self.i_start as u64),
        G2::Scalar::from(self.i_end as u64),
        G2::Scalar::from(right.i_start as u64),
        G2::Scalar::from(right.i_end as u64),
        self.z_start_secondary.clone(),
        self.z_end_secondary,
        right.z_start_secondary,
        right.z_end_secondary.clone(),
        Some(self.U_primary),
        Some(self.u_primary),
        Some(right.U_primary),
        Some(right.u_primary),
        Some(Commitment::<G1>::decompress(&nifs_left_primary.comm_T)?),
        Some(Commitment::<G1>::decompress(&nifs_right_primary.comm_T)?),
        Some(Commitment::<G1>::decompress(&nifs_primary.comm_T)?),
      );

    let circuit_secondary: NovaAugmentedParallelCircuit<G1, C2> = NovaAugmentedParallelCircuit::new(
      pp.augmented_circuit_params_primary.clone(),
      Some(inputs_secondary),
      c_secondary.clone(),
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);

    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.ck_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // Give these a trivial error vector
    let u_secondary = RelaxedR1CSInstance::from_r1cs_instance(
      &pp.ck_secondary,
      &pp.r1cs_shape_secondary,
      &u_secondary,
    );
    let w_secondary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_secondary, &w_secondary);

    // Name each of these to match struct fields
    let i_start = self.i_start.clone();
    let i_end = right.i_end.clone();
    let z_start_primary = self.z_start_primary;
    let z_end_primary = right.z_end_primary;
    let z_start_secondary = self.z_start_secondary;
    let z_end_secondary = right.z_end_secondary;

    Ok(Self {
      // Primary running instance
      W_primary,
      U_primary,
      // Primary new instance
      w_primary,
      u_primary,
      // The running instance of the secondary
      W_secondary,
      U_secondary,
      // The running instance of the secondary
      w_secondary,
      u_secondary,
      // The range data
      i_start,
      i_end,
      z_start_primary,
      z_end_primary,
      z_start_secondary,
      z_end_secondary,
      _p_c1: Default::default(),
      _p_c2: Default::default(),
    })
  }
}
