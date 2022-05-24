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
use nifs::NIFS;
use poseidon::ROConstantsCircuit; // TODO: make this a trait so we can use it without the concrete implementation
use r1cs::{
  R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness,
};
use traits::{
  AbsorbInROTrait, Group, HashFuncConstantsTrait, HashFuncTrait, StepCircuit, StepCompute,
};

type ROConstants<G> =
  <<G as Group>::HashFunc as HashFuncTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants;

/// A type that holds public parameters of Nova
pub struct PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar> + Clone,
  C2: StepCircuit<G2::Scalar> + Clone,
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
}

impl<G1, G2, C1, C2> PublicParams<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar> + Clone,
  C2: StepCircuit<G2::Scalar> + Clone,
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
    let circuit_primary: NIFSVerifierCircuit<G2, C1> = NIFSVerifierCircuit::new(
      params_primary.clone(),
      None,
      c_primary.clone(),
      ro_consts_circuit_primary.clone(),
    );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit_primary.synthesize(&mut cs);
    let (r1cs_shape_primary, r1cs_gens_primary) = (cs.r1cs_shape(), cs.r1cs_gens());

    // Initialize gens for the secondary
    let circuit_secondary: NIFSVerifierCircuit<G1, C2> = NIFSVerifierCircuit::new(
      params_secondary.clone(),
      None,
      c_secondary.clone(),
      ro_consts_circuit_secondary.clone(),
    );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit_secondary.synthesize(&mut cs);
    let (r1cs_shape_secondary, r1cs_gens_secondary) = (cs.r1cs_shape(), cs.r1cs_gens());

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
    }
  }
}

/// State of iteration over ComputeStep
pub struct ComputeState<S: StepCompute<F> + StepCircuit<F>, F: PrimeField> {
  circuit: S,
  val: F,
}

impl<F: PrimeField, S: StepCompute<F> + StepCircuit<F> + Clone> Iterator for ComputeState<S, F> {
  type Item = (S, F);

  fn next(&mut self) -> Option<Self::Item> {
    let (new_circuit, output) = self.circuit.compute(&self.val);

    self.circuit = new_circuit.clone();
    self.val = output;

    Some((new_circuit, output))
  }
}

/// A SNARK that proves the correct execution of an incremental computation
pub struct RecursiveSNARK<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar> + Clone,
  C2: StepCircuit<G2::Scalar> + Clone,
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
  c1: C1,
  c2: C2,
  num_steps: usize,
}

impl<G1, G2, C1, C2> RecursiveSNARK<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar> + Clone,
  C2: StepCircuit<G2::Scalar> + Clone,
{
  /// Initialize a `RecursiveSNARK`.
  pub fn init(
    pp: &PublicParams<G1, G2, C1, C2>,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
    z1_primary: G1::Scalar,
    z1_secondary: G2::Scalar,
    first_primary_circuit: &C1,
    first_secondary_circuit: &C2,
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
    let circuit_primary: NIFSVerifierCircuit<G2, C1> = NIFSVerifierCircuit::new(
      pp.params_primary.clone(),
      Some(inputs_primary),
      first_primary_circuit.clone(),
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
    let circuit_secondary: NIFSVerifierCircuit<G1, C2> = NIFSVerifierCircuit::new(
      pp.params_secondary.clone(),
      Some(inputs_secondary),
      first_secondary_circuit.clone(),
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
      c1: first_primary_circuit.clone(),
      c2: first_secondary_circuit.clone(),
      num_steps: 1,
    };

    Ok(state)
  }

  /// create a new `RecursiveSNARK` for circuits implementing StepCompute
  pub fn prove(
    pp: &PublicParams<G1, G2, C1, C2>,
    num_steps: Option<usize>,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
  ) -> Result<Self, NovaError>
  where
    C1: StepCompute<G1::Scalar>,
    C2: StepCompute<G2::Scalar>,
  {
    if let Some(num_steps) = num_steps {
      let mut primary_iterator = ComputeState {
        circuit: pp.c_primary.clone(),
        val: z0_primary,
      }
      .take(num_steps);

      let mut secondary_iterator = ComputeState {
        circuit: pp.c_secondary.clone(),
        val: z0_secondary,
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
        circuit: pp.c_primary.clone(),
        val: z0_primary,
      };

      let mut secondary_iterator = ComputeState {
        circuit: pp.c_secondary.clone(),
        val: z0_secondary,
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
    pp: &PublicParams<G1, G2, C1, C2>,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
    primary_iterator: &mut dyn Iterator<Item = (C1, G1::Scalar)>,
    secondary_iterator: &mut dyn Iterator<Item = (C2, G2::Scalar)>,
  ) -> Result<Self, NovaError> {
    let (first_primary_circuit, z1_primary) = match primary_iterator.next() {
      Some(next) => next,
      None => return Err(NovaError::InvalidNumSteps),
    };
    let (first_secondary_circuit, z1_secondary) = match secondary_iterator.next() {
      Some(next) => next,
      None => return Err(NovaError::InvalidNumSteps),
    };

    let mut state = Self::init(
      pp,
      z0_primary,
      z0_secondary,
      z1_primary,
      z1_secondary,
      &first_primary_circuit,
      &first_secondary_circuit,
    )?;

    // execute the remaining steps, alternating between G1 and G2

    let mut i = 1;
    while state
      .prove_step_with_iterators(
        pp,
        i,
        z0_primary,
        z0_secondary,
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
    pp: &PublicParams<G1, G2, C1, C2>,
    i: usize,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
    next_primary: (G1::Scalar, C1),
    next_secondary: (G2::Scalar, C2),
  ) -> Result<(), NovaError> {
    let (zn_primary, new_primary_circuit) = next_primary;
    let (zn_secondary, new_secondary_circuit) = next_secondary;

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

    let circuit_primary: NIFSVerifierCircuit<G2, C1> = NIFSVerifierCircuit::new(
      pp.params_primary.clone(),
      Some(inputs_primary),
      self.c1.clone(),
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

    let circuit_secondary: NIFSVerifierCircuit<G1, C2> = NIFSVerifierCircuit::new(
      pp.params_secondary.clone(),
      Some(inputs_secondary),
      self.c2.clone(),
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

    self.c1 = new_primary_circuit;
    self.c2 = new_secondary_circuit;
    self.zn_primary = zn_primary;
    self.zn_secondary = zn_secondary;
    self.num_steps += 1;

    Ok(())
  }

  /// Prove one step of an iterative computation, mutating the RecursiveSNARK
  pub fn prove_step_with_iterators(
    &mut self,
    pp: &PublicParams<G1, G2, C1, C2>,
    i: usize,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
    primary_iterator: &mut dyn Iterator<Item = (C1, G1::Scalar)>,
    secondary_iterator: &mut dyn Iterator<Item = (C2, G2::Scalar)>,
  ) -> Result<Option<()>, NovaError> {
    let (next_primary_circuit, zn_primary) = if let Some(next) = primary_iterator.next() {
      next
    } else {
      return Ok(None);
    };
    let (next_secondary_circuit, zn_secondary) = if let Some(next) = secondary_iterator.next() {
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
        (zn_primary, next_primary_circuit),
        (zn_secondary, next_secondary_circuit),
      )
      .map(Some)
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<G1, G2, C1, C2>,
    num_steps: usize,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
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
      hasher.absorb(z0_primary);
      hasher.absorb(self.zn_primary);
      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 = <G1 as Group>::HashFunc::new(pp.ro_consts_primary.clone());
      hasher2.absorb(scalar_as_base::<G1>(pp.r1cs_shape_primary.get_digest()));
      hasher2.absorb(G2::Scalar::from(num_steps as u64));
      hasher2.absorb(z0_secondary);
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
  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;
  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::PrimeField;
  use std::marker::PhantomData;

  #[derive(Clone, Debug)]
  struct TrivialTestCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for TrivialTestCircuit<F>
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

  impl<F> StepCompute<F> for TrivialTestCircuit<F>
  where
    F: PrimeField,
  {
    fn compute(&self, z: &F) -> (Self, F) {
      (self.clone(), *z)
    }
  }

  #[derive(Clone, Debug)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for CubicCircuit<F>
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

  impl<F> StepCompute<F> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn compute(&self, z: &F) -> (Self, F) {
      (self.clone(), *z * *z * *z + z + F::from(5u64))
    }
  }

  #[test]
  fn test_ivc_trivial() {
    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      TrivialTestCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(
      TrivialTestCircuit {
        _p: Default::default(),
      },
      TrivialTestCircuit {
        _p: Default::default(),
      },
    );

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(
      &pp,
      Some(3),
      <G1 as Group>::Scalar::zero(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      3,
      <G1 as Group>::Scalar::zero(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());

    // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(
      &pp,
      2,
      <G1 as Group>::Scalar::zero(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(bad_res.is_err());
  }

  #[test]
  fn test_ivc_nontrivial() {
    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      TrivialTestCircuit<<G1 as Group>::Scalar>,
      CubicCircuit<<G2 as Group>::Scalar>,
    >::setup(
      TrivialTestCircuit {
        _p: Default::default(),
      },
      CubicCircuit {
        _p: Default::default(),
      },
    );

    let num_steps = 3;

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(
      &pp,
      Some(num_steps),
      <G1 as Group>::Scalar::one(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      <G1 as Group>::Scalar::one(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());

    // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(
      &pp,
      num_steps + 1,
      <G1 as Group>::Scalar::zero(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(bad_res.is_err());

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, <G1 as Group>::Scalar::one());
    let mut zn_secondary_direct = <G2 as Group>::Scalar::zero();
    for _i in 0..num_steps {
      zn_secondary_direct = CubicCircuit {
        _p: Default::default(),
      }
      .compute(&zn_secondary_direct)
      .1;
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(zn_secondary, <G2 as Group>::Scalar::from(2460515u64));
  }

  #[test]
  fn test_ivc_base() {
    // produce public parameters
    let pp = PublicParams::<
      G1,
      G2,
      TrivialTestCircuit<<G1 as Group>::Scalar>,
      CubicCircuit<<G2 as Group>::Scalar>,
    >::setup(
      TrivialTestCircuit {
        _p: Default::default(),
      },
      CubicCircuit {
        _p: Default::default(),
      },
    );

    let num_steps = 1;

    // produce a recursive SNARK
    let res = RecursiveSNARK::prove(
      &pp,
      Some(num_steps),
      <G1 as Group>::Scalar::one(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());
    let recursive_snark = res.unwrap();

    // verify the recursive SNARK
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      <G1 as Group>::Scalar::one(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());

    // verification fails when num_steps is incorrect
    let bad_res = recursive_snark.verify(
      &pp,
      0,
      <G1 as Group>::Scalar::zero(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(bad_res.is_err());

    let (zn_primary, zn_secondary) = res.unwrap();

    assert_eq!(zn_primary, <G1 as Group>::Scalar::one());
    assert_eq!(zn_secondary, <G2 as Group>::Scalar::from(5u64));
  }
}
