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
  Circuit, ConstraintSystem, Index, SynthesisError,
};
use core::marker::PhantomData;
use ff::Field;
use rayon::prelude::*;
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
  r_W_primary: RelaxedR1CSWitness<G1>,
  r_U_primary: RelaxedR1CSInstance<G1>,
  // The new instance of the primary
  l_w_primary: RelaxedR1CSWitness<G1>,
  l_u_primary: RelaxedR1CSInstance<G1>,
  // The running instance of the secondary
  r_W_secondary: RelaxedR1CSWitness<G2>,
  r_U_secondary: RelaxedR1CSInstance<G2>,
  // The running instance of the secondary
  l_w_secondary: RelaxedR1CSWitness<G2>,
  l_u_secondary: RelaxedR1CSInstance<G2>,
  i_start: usize,
  i_end: usize,
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
    i: usize,
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
        G2::Scalar::from(i.try_into().unwrap()),
        G2::Scalar::from((i + 1).try_into().unwrap()),
        G2::Scalar::from((i).try_into().unwrap()),
        G2::Scalar::from((i + 1).try_into().unwrap()),
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
    let l_w_primary = RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &w_primary);
    let l_u_primary =
      RelaxedR1CSInstance::from_r1cs_instance(&pp.ck_primary, &pp.r1cs_shape_primary, &u_primary);
    let r_W_primary = l_w_primary.clone();
    let r_U_primary = l_u_primary.clone();

    // IVC proof of the secondary circuit
    let l_w_secondary =
      RelaxedR1CSWitness::<G2>::from_r1cs_witness(&pp.r1cs_shape_secondary, &w_secondary);
    let l_u_secondary = RelaxedR1CSInstance::<G2>::from_r1cs_instance(
      &pp.ck_secondary,
      &pp.r1cs_shape_secondary,
      &u_secondary,
    );
    let r_W_secondary = l_w_secondary.clone();
    let r_U_secondary = l_u_secondary.clone();

    if z_start_primary.len() != pp.F_arity_primary
      || z_start_secondary.len() != pp.F_arity_secondary
    {
      return Err(NovaError::InvalidStepOutputLength);
    }

    let i_start = i;
    let i_end = i + 1;

    Ok(Self {
      r_W_primary,
      r_U_primary,
      l_w_primary,
      l_u_primary,
      r_W_secondary,
      r_U_secondary,
      l_w_secondary,
      l_u_secondary,
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
    right: &NovaTreeNode<G1, G2, C1, C2>,
    pp: &PublicParams<G1, G2, C1, C2>,
    c_primary: &C1,
    c_secondary: &C2,
  ) -> Result<Self, NovaError> {
    // We have to merge two proofs where the right starts one index after the left ends
    // note that this would fail in the proof step but we error earlier here for debugging clarity.
    if self.i_end + 1 != right.i_start {
      return Err(NovaError::InvalidNodeMerge);
    }

    // First we fold the secondary instances of both the left and right children
    let (nifs_left_secondary, (left_U_secondary, left_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &self.r_U_secondary,
      &self.r_W_secondary,
      &self.l_u_secondary,
      &self.l_w_secondary,
    )?;
    let (nifs_right_secondary, (right_U_secondary, right_W_secondary)) = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &right.r_U_secondary,
      &right.r_W_secondary,
      &right.l_u_secondary,
      &right.l_w_secondary,
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

    return Err(NovaError::InvalidStepOutputLength);
  }
}

/// Structure for parallelization
#[derive(Debug, Clone)]
pub struct ParallelSNARK<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  nodes: Vec<Vec<NovaTreeNode<G1, G2, C1, C2>>>,
}

/// Implementation for parallelization SNARK
impl<G1, G2, C1, C2> ParallelSNARK<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  /// Create a new instance of parallel SNARK
  pub fn new(
    pp: &PublicParams<G1, G2, C1, C2>,
    steps: usize,
    z0_primary: Vec<G1::Scalar>,
    z0_secondary: Vec<G2::Scalar>,
    c_primary: C1,
    c_secondary: C2,
  ) -> Self {
    // Tuple's structure is (index, zi_primary, zi_secondary)
    let mut zi = Vec::<(usize, Vec<G1::Scalar>, Vec<G2::Scalar>)>::new();
    // First input value of Z0, these steps can't be done in parallel
    zi.push((0, z0_primary.clone(), z0_secondary.clone()));
    for i in 1..steps {
      let (index, prev_primary, prev_secondary) = &zi[i - 1];
      zi.push((
        i,
        c_primary.output(&prev_primary),
        c_secondary.output(&prev_secondary),
      ));
    }
    // Do calculate node tree in parallel
    let leafs_vec = zi
      .par_chunks(2)
      .map(|item| {
        match item {
          // There are 2 nodes
          [l, r] => NovaTreeNode::new(
            &pp,
            c_primary.clone(),
            c_secondary.clone(),
            l.0,
            l.1.clone(),
            r.1.clone(),
            l.2.clone(),
            r.2.clone(),
          )
          .expect("Unable to create base node"),
          // Just 1 node left
          [l] => NovaTreeNode::new(
            &pp,
            c_primary.clone(),
            c_secondary.clone(),
            l.0,
            zi[l.0 - 1].1.clone(),
            l.1.clone(),
            zi[l.0 - 1].2.clone(),
            l.2.clone(),
          )
          .expect("Unable to create the last base node"),
          _ => panic!("Unexpected chunk size"),
        }
      })
      .collect();
    // Create a new parallel prover wit basic leafs
    Self {
      nodes: vec![leafs_vec],
    }
  }

  /// Perform the proving in parallel
  pub fn prove(&mut self, pp: &PublicParams<G1, G2, C1, C2>, c_primary: &C1, c_secondary: &C2) {
    // Calculate the max height of the tree
    // ⌈log2(n)⌉ + 1
    let max_height = ((self.nodes[0].len() as f64).log2().ceil() + 1f64) as usize;

    // Build up the tree with max given height
    for level in 0..max_height {
      // Create new instance of nodes in the next level
      let mut leafs = Vec::<(&NovaTreeNode<G1, G2, C1, C2>, &NovaTreeNode<G1, G2, C1, C2>)>::new();

      // Push leafs to the next level
      self.nodes.push(
        self.nodes[level]
          .par_chunks(2)
          .map(|item| match item {
            // There are 2 nodes in the chunk
            [vl, vr] => (*vl)
              .clone()
              .merge(vr, pp, c_primary, c_secondary)
              .expect("Merge the left and right should work"),
            // Just 1 node left, we carry it to the next level
            [vl] => (*vl).clone(),
            _ => panic!("Invalid chunk size"),
          })
          .collect(),
      );
    }
  }
}
