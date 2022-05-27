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

use crate::bellperson::{
  r1cs::{NovaShape, NovaWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};
use ::bellperson::{Circuit, ConstraintSystem};
use circuit::{NIFSVerifierCircuit, NIFSVerifierCircuitInputs, NIFSVerifierCircuitParams};
use constants::{BN_LIMB_WIDTH, BN_N_LIMBS};
use errors::NovaError;
use ff::{Field, PrimeField};
use gadgets::utils::scalar_as_base;
use neptune::{poseidon::PoseidonConstants, Arity};
use nifs::NIFS;
use poseidon::ROConstantsCircuit; // TODO: make this a trait so we can use it without the concrete implementation
use r1cs::{
  R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
};
use std::marker::PhantomData;
use traits::{
  AbsorbInROTrait, Group, HashFuncConstantsTrait, HashFuncTrait, StepCircuit, StepCompute, IO,
};

type ROConstants<G> =
  <<G as Group>::HashFunc as HashFuncTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants;

/// A type that holds public parameters of Nova
pub struct PublicParams<G1, G2, C1, C2, A1, A2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar, A1> + Clone,
  C2: StepCircuit<G2::Scalar, A2> + Clone,
  A1: Arity<<G1 as Group>::Scalar>,
  A2: Arity<<G2 as Group>::Scalar>,
{
  ro_consts_primary: ROConstants<G1>,
  ro_consts_circuit_primary: ROConstantsCircuit<<G2 as Group>::Base>,
  r1cs_gens_primary: R1CSGens<G1>,
  r1cs_shape_primary: R1CSShape<G1>,
  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<<G1 as Group>::Base>,
  r1cs_gens_secondary: R1CSGens<G2>,
  r1cs_shape_secondary: R1CSShape<G2>,
  params_primary: NIFSVerifierCircuitParams,
  params_secondary: NIFSVerifierCircuitParams,
  c_primary: C1,
  c_secondary: C2,
  a1: Option<PoseidonConstants<<G1 as Group>::Scalar, A1>>,
  a2: Option<PoseidonConstants<<G2 as Group>::Scalar, A2>>,
}

impl<G1, G2, C1, C2, A1, A2> PublicParams<G1, G2, C1, C2, A1, A2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar, A1> + Clone,
  C2: StepCircuit<G2::Scalar, A2> + Clone,
  A1: Arity<<G1 as Group>::Scalar>,
  A2: Arity<<G2 as Group>::Scalar>,
{
  /// Create a new `PublicParams`
  pub fn setup(c_primary: C1, c_secondary: C2) -> Self {
    let params_primary = NIFSVerifierCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params_secondary = NIFSVerifierCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);

    let ro_consts_primary: ROConstants<G1> = ROConstants::<G1>::new();
    let ro_consts_secondary: ROConstants<G2> = ROConstants::<G2>::new();

    let ro_consts_circuit_primary: ROConstantsCircuit<<G2 as Group>::Base> =
      ROConstantsCircuit::new();
    let ro_consts_circuit_secondary: ROConstantsCircuit<<G1 as Group>::Base> =
      ROConstantsCircuit::new();

    // Initialize gens for the primary
    let circuit_primary: NIFSVerifierCircuit<G2, C1, A1> = NIFSVerifierCircuit::new(
      params_primary.clone(),
      None,
      c_primary.clone(),
      ro_consts_circuit_primary.clone(),
    );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (r1cs_shape_primary, r1cs_gens_primary) = (cs.r1cs_shape(), cs.r1cs_gens());

    // Initialize gens for the secondary
    let circuit_secondary: NIFSVerifierCircuit<G1, C2, A2> = NIFSVerifierCircuit::new(
      params_secondary.clone(),
      None,
      c_secondary.clone(),
      ro_consts_circuit_secondary.clone(),
    );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let (r1cs_shape_secondary, r1cs_gens_secondary) = (cs.r1cs_shape(), cs.r1cs_gens());

    let a1 = if A1::to_usize() == 1 {
      None
    } else {
      Some(PoseidonConstants::<<G1 as Group>::Scalar, A1>::new())
    };
    let a2 = if A2::to_usize() == 1 {
      None
    } else {
      Some(PoseidonConstants::<<G2 as Group>::Scalar, A2>::new())
    };

    Self {
      ro_consts_primary,
      ro_consts_circuit_primary,
      r1cs_gens_primary,
      r1cs_shape_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      r1cs_gens_secondary,
      r1cs_shape_secondary,
      params_primary,
      params_secondary,
      c_primary,
      c_secondary,
      a1,
      a2,
    }
  }

  /// Create new IO::Val for primary circuit
  pub fn new_primary_val(&self, val: <G1 as Group>::Scalar) -> IO<'_, <G1 as Group>::Scalar, A1> {
    assert_eq!(1, A1::to_usize());
    IO::Val(val)
  }
  /// Create new IO::Val for secondary circuit
  pub fn new_secondary_val(&self, val: <G2 as Group>::Scalar) -> IO<<G2 as Group>::Scalar, A2> {
    assert_eq!(1, A2::to_usize());
    IO::Val(val)
  }

  /// Create new IO::Vals for primary circuit, including allocated poseidon constants
  pub fn new_primary_vals(
    &self,
    vals: Vec<<G1 as Group>::Scalar>,
  ) -> IO<<G1 as Group>::Scalar, A1> {
    assert!(A1::to_usize() != 1);
    let poseidon_constants = self.a1.as_ref().expect("poseidon constants missing");
    IO::Vals(vals, poseidon_constants)
  }

  /// Create new IO::vals for secondary circuit, including allocated poseidon constants
  pub fn new_secondary_vals(
    &self,
    vals: Vec<<G2 as Group>::Scalar>,
  ) -> IO<<G2 as Group>::Scalar, A2> {
    assert!(A2::to_usize() != 1);
    let poseidon_constants = self.a2.as_ref().expect("poseidon constants missing");
    IO::Vals(vals, poseidon_constants)
  }
}

/// State of iteration over ComputeStep
pub struct ComputeState<
  'a,
  S: StepCompute<'a, F, A> + StepCircuit<F, A>,
  F: PrimeField,
  A: Arity<F>,
> {
  circuit: S,
  val: IO<'a, F, A>,
}

impl<'a, F: PrimeField, S: StepCompute<'a, F, A> + StepCircuit<F, A> + Clone, A: Arity<F>> Iterator
  for ComputeState<'a, S, F, A>
{
  type Item = (S, F);

  fn next(&mut self) -> Option<Self::Item> {
    let current_circuit = self.circuit.clone();
    if let Some((new_circuit, output)) = self.circuit.compute_io(&self.val) {
      self.circuit = new_circuit;

      self.val = output.clone();

      Some((current_circuit, output.output()))
    } else {
      None
    }
  }
}

/// A SNARK that proves the correct execution of an incremental computation
pub struct RecursiveSNARK<G1, G2, C1, C2, A1, A2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar, A1> + Clone,
  C2: StepCircuit<G2::Scalar, A2> + Clone,
  A1: Arity<<G1 as Group>::Scalar>,
  A2: Arity<<G2 as Group>::Scalar>,
{
  r_W_primary: RelaxedR1CSWitness<G1>,
  r_U_primary: RelaxedR1CSInstance<G1>,
  l_w_primary: R1CSWitness<G1>,
  l_u_primary: R1CSInstance<G1>,
  r_W_secondary: RelaxedR1CSWitness<G2>,
  r_U_secondary: RelaxedR1CSInstance<G2>,
  l_w_secondary: R1CSWitness<G2>,
  l_u_secondary: R1CSInstance<G2>,
  zn_primary: G1::Scalar,
  zn_secondary: G2::Scalar,
  _p_c1: PhantomData<C1>,
  _p_c2: PhantomData<C2>,
  a1: PhantomData<A1>,
  a2: PhantomData<A2>,
  num_steps: usize,
}

impl<G1, G2, C1, C2, A1, A2> RecursiveSNARK<G1, G2, C1, C2, A1, A2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar, A1> + Clone,
  C2: StepCircuit<G2::Scalar, A2> + Clone,
  A1: Arity<<G1 as Group>::Scalar>,
  A2: Arity<<G2 as Group>::Scalar>,
{
  /// Initialize a `RecursiveSNARK`.
  pub fn init(
    pp: &PublicParams<G1, G2, C1, C2, A1, A2>,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
    z1_primary: G1::Scalar,
    z1_secondary: G2::Scalar,
    first_circuit_primary: &C1,
    first_circuit_secondary: &C2,
  ) -> Result<Self, NovaError> {
    // Execute the base case for the primary
    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs_primary: NIFSVerifierCircuitInputs<G2> = NIFSVerifierCircuitInputs::new(
      pp.r1cs_shape_secondary.get_digest(),
      G1::Scalar::zero(),
      z0_primary,
      None,
      None,
      None,
      None,
    );
    let circuit_primary: NIFSVerifierCircuit<G2, C1, A1> = NIFSVerifierCircuit::new(
      pp.params_primary.clone(),
      Some(inputs_primary),
      first_circuit_primary.clone(),
      pp.ro_consts_circuit_primary.clone(),
    );
    let _ = circuit_primary.synthesize(&mut cs_primary);
    let (u_primary, w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.r1cs_gens_primary)
      .map_err(|_e| NovaError::UnSat)?;

    // Execute the base case for the secondary
    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs_secondary: NIFSVerifierCircuitInputs<G1> = NIFSVerifierCircuitInputs::new(
      pp.r1cs_shape_primary.get_digest(),
      G2::Scalar::zero(),
      z0_secondary,
      None,
      None,
      Some(u_primary.clone()),
      None,
    );
    let circuit_secondary: NIFSVerifierCircuit<G1, C2, A2> = NIFSVerifierCircuit::new(
      pp.params_secondary.clone(),
      Some(inputs_secondary),
      first_circuit_secondary.clone(),
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.r1cs_gens_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    let state = Self {
      r_W_primary: RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &w_primary),
      r_U_primary: RelaxedR1CSInstance::from_r1cs_instance(
        &pp.r1cs_gens_primary,
        &pp.r1cs_shape_primary,
        &u_primary,
      ),
      l_w_primary: w_primary,
      l_u_primary: u_primary,
      r_W_secondary: RelaxedR1CSWitness::<G2>::default(&pp.r1cs_shape_secondary),
      r_U_secondary: RelaxedR1CSInstance::<G2>::default(
        &pp.r1cs_gens_secondary,
        &pp.r1cs_shape_secondary,
      ),
      l_w_secondary: w_secondary,
      l_u_secondary: u_secondary,
      zn_primary: z1_primary,
      zn_secondary: z1_secondary,
      _p_c1: Default::default(),
      _p_c2: Default::default(),
      a1: Default::default(),
      a2: Default::default(),
      num_steps: 1,
    };

    Ok(state)
  }

  /// create a new `RecursiveSNARK` for circuits implementing StepCompute
  pub fn prove<'a>(
    pp: &PublicParams<G1, G2, C1, C2, A1, A2>,
    num_steps: Option<usize>,
    z0_primary: &IO<'a, G1::Scalar, A1>,
    z0_secondary: &IO<'a, G2::Scalar, A2>,
    c_primary: Option<C1>,
    c_secondary: Option<C2>,
  ) -> Result<Self, NovaError>
  where
    C1: StepCompute<'a, G1::Scalar, A1>,
    C2: StepCompute<'a, G2::Scalar, A2>,
  {
    let c_primary = if let Some(c_primary) = c_primary {
      c_primary
    } else {
      pp.c_primary.clone()
    };
    let c_secondary = if let Some(c_secondary) = c_secondary {
      c_secondary
    } else {
      pp.c_secondary.clone()
    };

    if let Some(num_steps) = num_steps {
      let mut primary_iterator = ComputeState {
        circuit: c_primary,
        val: z0_primary.clone(),
      }
      .take(num_steps);

      let mut secondary_iterator = ComputeState {
        circuit: c_secondary,
        val: z0_secondary.clone(),
      }
      .take(num_steps);

      Self::prove_with_iterators(
        pp,
        z0_primary,
        z0_secondary,
        &mut primary_iterator,
        &mut secondary_iterator,
      )
    } else {
      let mut primary_iterator = ComputeState {
        circuit: c_primary,
        val: z0_primary.clone(),
      };

      let mut secondary_iterator = ComputeState {
        circuit: c_secondary,
        val: z0_secondary.clone(),
      };
      Self::prove_with_iterators(
        pp,
        z0_primary,
        z0_secondary,
        &mut primary_iterator,
        &mut secondary_iterator,
      )
    }
  }

  /// create a new `RecursiveSNARK` from primary and secondary iterators
  pub fn prove_with_iterators(
    pp: &PublicParams<G1, G2, C1, C2, A1, A2>,
    z0_primary: &IO<G1::Scalar, A1>,
    z0_secondary: &IO<G2::Scalar, A2>,
    primary_iterator: &mut dyn Iterator<Item = (C1, G1::Scalar)>,
    secondary_iterator: &mut dyn Iterator<Item = (C2, G2::Scalar)>,
  ) -> Result<Self, NovaError> {
    let (first_circuit_primary, z1_primary) = match primary_iterator.next() {
      Some(next) => next,
      None => return Err(NovaError::InvalidNumSteps),
    };
    let (first_circuit_secondary, z1_secondary) = match secondary_iterator.next() {
      Some(next) => next,
      None => return Err(NovaError::InvalidNumSteps),
    };

    let mut state = Self::init(
      pp,
      z0_primary.output(),
      z0_secondary.output(),
      z1_primary,
      z1_secondary,
      &first_circuit_primary,
      &first_circuit_secondary,
    )?;

    // execute the remaining steps, alternating between G1 and G2

    let mut i = 1;
    while state
      .prove_step_with_iterators(
        pp,
        i,
        z0_primary.output(),
        z0_secondary.output(),
        primary_iterator,
        secondary_iterator,
      )?
      .is_some()
    {
      i += 1;
    }
    Ok(state)
  }

  /// Prove one step of an iterative computation, mutating the RecursiveSNARK
  pub fn prove_step(
    &mut self,
    pp: &PublicParams<G1, G2, C1, C2, A1, A2>,
    i: usize,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
    next_primary: (G1::Scalar, C1),
    next_secondary: (G2::Scalar, C2),
  ) -> Result<(), NovaError> {
    let (zn_primary, new_circuit_primary) = next_primary;
    let (zn_secondary, new_circuit_secondary) = next_secondary;

    // fold the secondary circuit's instance
    let (nifs_secondary, (r_U_next_secondary, r_W_next_secondary)) = NIFS::prove(
      &pp.r1cs_gens_secondary,
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &self.r_U_secondary,
      &self.r_W_secondary,
      &self.l_u_secondary,
      &self.l_w_secondary,
    )?;

    let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs_primary: NIFSVerifierCircuitInputs<G2> = NIFSVerifierCircuitInputs::new(
      pp.r1cs_shape_secondary.get_digest(),
      G1::Scalar::from(i as u64),
      z0_primary,
      Some(self.zn_primary),
      Some(self.r_U_secondary.clone()),
      Some(self.l_u_secondary.clone()),
      Some(nifs_secondary.comm_T.decompress()?),
    );

    let circuit_primary: NIFSVerifierCircuit<G2, C1, A1> = NIFSVerifierCircuit::new(
      pp.params_primary.clone(),
      Some(inputs_primary),
      new_circuit_primary,
      pp.ro_consts_circuit_primary.clone(),
    );
    let _ = circuit_primary.synthesize(&mut cs_primary);

    (self.l_u_primary, self.l_w_primary) = cs_primary
      .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.r1cs_gens_primary)
      .map_err(|_e| NovaError::UnSat)?;

    // fold the primary circuit's instance
    let (nifs_primary, (r_U_next_primary, r_W_next_primary)) = NIFS::prove(
      &pp.r1cs_gens_primary,
      &pp.ro_consts_primary,
      &pp.r1cs_shape_primary,
      &self.r_U_primary.clone(),
      &self.r_W_primary.clone(),
      &self.l_u_primary.clone(),
      &self.l_w_primary.clone(),
    )?;

    let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs_secondary: NIFSVerifierCircuitInputs<G1> = NIFSVerifierCircuitInputs::new(
      pp.r1cs_shape_primary.get_digest(),
      G2::Scalar::from(i as u64),
      z0_secondary,
      Some(self.zn_secondary),
      Some(self.r_U_primary.clone()),
      Some(self.l_u_primary.clone()),
      Some(nifs_primary.comm_T.decompress()?),
    );

    let circuit_secondary: NIFSVerifierCircuit<G1, C2, A2> = NIFSVerifierCircuit::new(
      pp.params_secondary.clone(),
      Some(inputs_secondary),
      new_circuit_secondary,
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);

    (self.l_u_secondary, self.l_w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.r1cs_gens_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // update the running instances and witnesses
    self.r_U_secondary = r_U_next_secondary;
    self.r_W_secondary = r_W_next_secondary;
    self.r_U_primary = r_U_next_primary;
    self.r_W_primary = r_W_next_primary;

    self.zn_primary = zn_primary;
    self.zn_secondary = zn_secondary;
    self.num_steps += 1;

    Ok(())
  }

  /// Prove one step of an iterative computation, mutating the RecursiveSNARK
  pub fn prove_step_with_iterators(
    &mut self,
    pp: &PublicParams<G1, G2, C1, C2, A1, A2>,
    i: usize,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
    primary_iterator: &mut dyn Iterator<Item = (C1, G1::Scalar)>,
    secondary_iterator: &mut dyn Iterator<Item = (C2, G2::Scalar)>,
  ) -> Result<Option<()>, NovaError> {
    let (next_circuit_primary, zn_primary) = if let Some(next) = primary_iterator.next() {
      next
    } else {
      return Ok(None);
    };
    let (next_circuit_secondary, zn_secondary) = if let Some(next) = secondary_iterator.next() {
      next
    } else {
      return Ok(None);
    };
    self
      .prove_step(
        pp,
        i,
        z0_primary,
        z0_secondary,
        (zn_primary, next_circuit_primary),
        (zn_secondary, next_circuit_secondary),
      )
      .map(Some)
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<G1, G2, C1, C2, A1, A2>,
    num_steps: usize,
    z0_primary: &IO<G1::Scalar, A1>,
    z0_secondary: &IO<G2::Scalar, A2>,
  ) -> Result<(G1::Scalar, G2::Scalar), NovaError> {
    // number of steps cannot be zero
    if num_steps == 0 {
      return Err(NovaError::ProofVerifyError);
    }
    if num_steps != self.num_steps {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the (relaxed) R1CS instances have two public outputs
    if self.l_u_primary.X.len() != 2
      || self.l_u_secondary.X.len() != 2
      || self.r_U_primary.X.len() != 2
      || self.r_U_secondary.X.len() != 2
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check if the output hashes in R1CS instances point to the right running instances
    let (hash_primary, hash_secondary) = {
      let mut hasher = <G2 as Group>::HashFunc::new(pp.ro_consts_secondary.clone());
      hasher.absorb(scalar_as_base::<G2>(pp.r1cs_shape_secondary.get_digest()));
      hasher.absorb(G1::Scalar::from(num_steps as u64));
      hasher.absorb(z0_primary.output());
      hasher.absorb(self.zn_primary);
      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 = <G1 as Group>::HashFunc::new(pp.ro_consts_primary.clone());
      hasher2.absorb(scalar_as_base::<G1>(pp.r1cs_shape_primary.get_digest()));
      hasher2.absorb(G2::Scalar::from(num_steps as u64));
      hasher2.absorb(z0_secondary.output());
      hasher2.absorb(self.zn_secondary);
      self.r_U_primary.absorb_in_ro(&mut hasher2);

      (hasher.get_hash(), hasher2.get_hash())
    };

    if hash_primary != scalar_as_base::<G1>(self.l_u_primary.X[1])
      || hash_secondary != scalar_as_base::<G2>(self.l_u_secondary.X[1])
    {
      return Err(NovaError::ProofVerifyError);
    }

    // check the satisfiability of the provided instances
    pp.r1cs_shape_primary.is_sat_relaxed(
      &pp.r1cs_gens_primary,
      &self.r_U_primary,
      &self.r_W_primary,
    )?;

    pp.r1cs_shape_primary
      .is_sat(&pp.r1cs_gens_primary, &self.l_u_primary, &self.l_w_primary)?;

    pp.r1cs_shape_secondary.is_sat_relaxed(
      &pp.r1cs_gens_secondary,
      &self.r_U_secondary,
      &self.r_W_secondary,
    )?;

    pp.r1cs_shape_secondary.is_sat(
      &pp.r1cs_gens_secondary,
      &self.l_u_secondary,
      &self.l_w_secondary,
    )?;

    Ok((self.zn_primary, self.zn_secondary))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use generic_array::typenum::{U1, U2};
  use neptune::poseidon::PoseidonConstants;

  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;
  type S1 = <G1 as Group>::Scalar;
  type S2 = <G2 as Group>::Scalar;

  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::PrimeField;
  use std::marker::PhantomData;

  #[derive(Clone, Debug, Default)]
  struct TrivialTestCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F, U1> for TrivialTestCircuit<F>
  where
    F: PrimeField,
  {
    fn synthesize_step<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      z: AllocatedNum<F>,
    ) -> Result<AllocatedNum<F>, SynthesisError> {
      Ok(z)
    }
  }

  impl<'a, F> StepCompute<'a, F, U1> for TrivialTestCircuit<F>
  where
    F: PrimeField,
  {
    fn compute(&self, z: &F) -> Option<(Self, F)> {
      Some((self.clone(), *z))
    }
  }

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F, U1> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn synthesize_step<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: AllocatedNum<F>,
    ) -> Result<AllocatedNum<F>, SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = z;
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
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

      Ok(y)
    }
  }

  impl<'a, F> StepCompute<'a, F, U1> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn compute(&self, z: &F) -> Option<(Self, F)> {
      Some((self.clone(), *z * *z * *z + z + F::from(5u64)))
    }
  }

  #[derive(Clone)]
  struct CubicIncrementingCircuit<'a, F: PrimeField> {
    z_n: IO<'a, F, U2>,
    _p: PhantomData<F>,
  }

  impl<'a, F: PrimeField> CubicIncrementingCircuit<'a, F> {
    fn new(z_n: IO<'a, F, U2>) -> Self {
      Self {
        z_n,
        _p: Default::default(),
      }
    }
  }

  impl<F> StepCircuit<F, U2> for CubicIncrementingCircuit<'_, F>
  where
    F: PrimeField,
  {
    fn io<'a>(&'a self) -> &'a IO<F, U2> {
      &self.z_n
    }

    fn synthesize_step_inner<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z_vec: Vec<AllocatedNum<F>>,
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      let (x, y) = (&z_vec[0], &z_vec[1]);

      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
      let x_next = AllocatedNum::alloc(cs.namespace(|| "x_next"), || {
        Ok(x_cu.get_value().unwrap() + y.get_value().unwrap())
      })?;

      let y_next = AllocatedNum::alloc(cs.namespace(|| "y_next"), || {
        Ok(y.get_value().unwrap() + F::one())
      })?;

      cs.enforce(
        || "y_next = y + 1",
        |lc| lc + y.get_variable() + CS::one(),
        |lc| lc + CS::one(),
        |lc| lc + y_next.get_variable(),
      );

      cs.enforce(
        || "x_next = x^3 + y",
        |lc| lc + x_cu.get_variable() + y.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + x_next.get_variable(),
      );

      let z_vec_next = vec![x_next, y_next];

      Ok(z_vec_next)
    }
  }

  impl<'a, F> StepCompute<'a, F, U2> for CubicIncrementingCircuit<'a, F>
  where
    F: PrimeField,
  {
    fn compute_inner(
      &self,
      z_vec: &[F],
      p: &'a PoseidonConstants<F, U2>,
    ) -> Option<(Self, Vec<F>)> {
      let (x, y) = (z_vec[0], z_vec[1]);
      let x_next = x * x * x + y;
      let y_next = y + F::one();
      let z_vec_next = vec![x_next, y_next];
      let z_n = IO::Vals(z_vec_next.clone(), p);
      let next = Self {
        z_n,
        _p: Default::default(),
      };

      Some((next, z_vec_next))
    }
  }

  #[derive(Clone, Debug, Default)]
  struct NondeterministicCircuit<F: PrimeField> {
    roots: Vec<F>,
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> NondeterministicCircuit<F> {
    fn new(output: F, num_steps: usize) -> Self {
      // Create a vector of successive roots, whose first element is the input and last is the output.
      // The vector's size is one more than the number of steps.
      let size = num_steps + 1;
      let mut roots = Vec::with_capacity(size);
      let mut last = output;

      // Assemble in reverse order by pushing.
      roots.push(output);

      for _ in 0..num_steps {
        let next = last * last;
        // Each element is the square of the last.
        roots.push(next);
        last = next;
      }

      // So the last shall be first, and the first last: for many be called, but few chosen.
      // Now each element is the square root of its predecessor.
      roots.reverse();

      Self {
        roots,
        _p: Default::default(),
      }
    }
  }

  impl<F> StepCircuit<F, U1> for NondeterministicCircuit<F>
  where
    F: PrimeField,
  {
    fn synthesize_step<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: AllocatedNum<F>,
    ) -> Result<AllocatedNum<F>, SynthesisError> {
      // The output, x, is the square root of the input, z.
      // This is non-deterministic insofar as we are not calculating the square root directly.
      // Rather, we take advantage of the hint precomputed in self.roots.
      let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(self.roots[1]))?;

      if let (Some(z), Some(x)) = (z.get_value(), x.get_value()) {
        // Sanity check
        assert_eq!(z, x * x)
      };

      cs.enforce(
        || "z = x * x",
        |lc| lc + x.get_variable(),
        |lc| lc + x.get_variable(),
        |lc| lc + z.get_variable(),
      );

      Ok(x)
    }
  }

  impl<F> StepCompute<'_, F, U1> for NondeterministicCircuit<F>
  where
    F: PrimeField,
  {
    fn compute(&self, z: &F) -> Option<(Self, F)> {
      if self.roots.len() > 1 {
        // First in the sequence of roots is the square.
        let square = self.roots[0];
        // This is the input to the circuit
        assert_eq!(*z, square);

        let next = Self {
          roots: self.roots[1..].to_vec(),
          _p: Default::default(),
        };

        let root = self.roots[1];
        assert_eq!(square, root * root);

        Some((next, root))
      } else {
        None
      }
    }
  }

  #[test]
  fn test_ivc_trivial() {
    // produce public parameters
    let pp = PublicParams::<G1, G2, TrivialTestCircuit<S1>, TrivialTestCircuit<S2>, U1, U1>::setup(
      TrivialTestCircuit::default(),
      TrivialTestCircuit::default(),
    );

    let v1 = pp.new_primary_val(S1::zero());
    let v2 = pp.new_secondary_val(S2::zero());

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(&pp, Some(3), &v1, &v2, None, None);
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, 3, &v1, &v2);
    assert!(res.is_ok());

    // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(&pp, 2, &v1, &v2);
    assert!(bad_res.is_err());
  }

  #[test]
  fn test_ivc_nontrivial() {
    // produce public parameters
    let pp = PublicParams::<G1, G2, TrivialTestCircuit<S1>, CubicCircuit<S2>, U1, U1>::setup(
      TrivialTestCircuit::default(),
      CubicCircuit::default(),
    );

    let num_steps = 3;

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(
      &pp,
      Some(num_steps),
      &IO::Val(S1::one()),
      &IO::Val(S2::zero()),
      None,
      None,
    );
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    let secondary_z0 = S2::zero();
    let v1 = pp.new_primary_val(S1::one());
    let v1_bad = pp.new_primary_val(S1::zero());
    let v2 = pp.new_secondary_val(secondary_z0);

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &v1, &v2);
    assert!(res.is_ok());

    // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(&pp, num_steps + 1, &v1_bad, &v2);
    assert!(bad_res.is_err());

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, S1::one());
    let mut zn_secondary_direct = secondary_z0;
    for _i in 0..num_steps {
      zn_secondary_direct = CubicCircuit {
        _p: Default::default(),
      }
      .compute(&zn_secondary_direct)
      .unwrap()
      .1;
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(zn_secondary, S2::from(2460515u64));
  }

  #[test]
  fn test_ivc_binary() {
    // produce public parameters
    let p = PoseidonConstants::<S2, U2>::new();

    let pp =
      PublicParams::<G1, G2, TrivialTestCircuit<S1>, CubicIncrementingCircuit<S2>, U1, U2>::setup(
        TrivialTestCircuit::default(),
        CubicIncrementingCircuit::new(IO::Empty(&p)),
      );

    let num_steps = 1;

    let v1 = pp.new_primary_val(S1::one());
    let v2 = pp.new_secondary_vals(vec![S2::from(123), S2::zero()]);

    let c_secondary = CubicIncrementingCircuit::new(v2.clone());

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(&pp, Some(num_steps), &v1, &v2, None, Some(c_secondary));
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &v1, &v2);
    assert!(res.is_ok());

    // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(&pp, num_steps + 1, &v1, &v2);
    assert!(bad_res.is_err());

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, S1::one());
    let mut zn_secondary_direct = v2;
    for _i in 0..num_steps {
      zn_secondary_direct = CubicIncrementingCircuit {
        z_n: zn_secondary_direct.clone(),
        _p: Default::default(),
      }
      .compute_io(&zn_secondary_direct)
      .unwrap()
      .1;
    }
    assert_eq!(zn_secondary, zn_secondary_direct.output());
  }

  #[test]
  fn test_ivc_base() {
    // produce public parameters
    let pp = PublicParams::<G1, G2, TrivialTestCircuit<S1>, CubicCircuit<S2>, U1, U1>::setup(
      TrivialTestCircuit::default(),
      CubicCircuit::default(),
    );

    let num_steps = 1;

    let v1 = pp.new_primary_val(S1::one());
    let bad_v1 = pp.new_primary_val(S1::zero());
    let v2 = pp.new_secondary_val(S2::zero());

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(&pp, Some(num_steps), &v1, &v2, None, None);
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &v1, &v2);
    assert!(res.is_ok());

    // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(&pp, 0, &bad_v1, &v2);
    assert!(bad_res.is_err());

    let (zn_primary, zn_secondary) = res.unwrap();

    assert_eq!(zn_primary, S1::one());
    assert_eq!(zn_secondary, S2::from(5u64));
  }

  #[test]
  fn test_ivc_nondeterministic() {
    // Private parameters for nondeterministic function
    let num_steps = 3;
    let output = S2::from(123);

    // Create the nondeterministic circuit
    let ndc = NondeterministicCircuit::<S2>::new(output, num_steps);
    let input = ndc.roots[0];

    // produce public parameters
    let pp =
      PublicParams::<G1, G2, TrivialTestCircuit<S1>, NondeterministicCircuit<S2>, U1, U1>::setup(
        TrivialTestCircuit::default(),
        NondeterministicCircuit::<S2>::default(),
      );

    let v1 = pp.new_primary_val(S1::one());
    let bad_v1 = pp.new_primary_val(S1::zero());
    let v2 = pp.new_secondary_val(input);
    let bad_v2 = pp.new_secondary_val(S2::zero());

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(&pp, None, &v1, &v2, None, Some(ndc));
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &v1, &v2);
    assert!(res.is_ok());

    // // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(&pp, num_steps, &bad_v1, &bad_v2);
    assert!(bad_res.is_err());

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, S1::one());
    assert_eq!(zn_secondary, output);
  }
}
