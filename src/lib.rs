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
mod ipa;
pub mod nifs;
pub mod pasta;
mod polynomial;
mod poseidon;
pub mod r1cs;
mod sumcheck;
pub mod traits;

use crate::bellperson::{
  r1cs::{NovaShape, NovaWitness},
  shape_cs::ShapeCS,
  solver::SatisfyingAssignment,
};
use ::bellperson::{Circuit, ConstraintSystem};
use circuit::{NIFSVerifierCircuit, NIFSVerifierCircuitInputs, NIFSVerifierCircuitParams};
use constants::{BN_LIMB_WIDTH, BN_N_LIMBS};
use core::marker::PhantomData;
use errors::NovaError;
use ff::Field;
use gadgets::utils::scalar_as_base;
use nifs::NIFS;
use poseidon::ROConstantsCircuit; // TODO: make this a trait so we can use it without the concrete implementation
use r1cs::{
  R1CSGens, R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSSNARK,
  RelaxedR1CSWitness,
};
use traits::{AbsorbInROTrait, Group, HashFuncConstantsTrait, HashFuncTrait, StepCircuit};

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
  r1cs_shape_padded_primary: R1CSShape<G1>,
  ro_consts_secondary: ROConstants<G2>,
  ro_consts_circuit_secondary: ROConstantsCircuit<<G1 as Group>::Base>,
  r1cs_gens_secondary: R1CSGens<G2>,
  r1cs_shape_secondary: R1CSShape<G2>,
  r1cs_shape_padded_secondary: R1CSShape<G2>,
  c_primary: C1,
  c_secondary: C2,
  params_primary: NIFSVerifierCircuitParams,
  params_secondary: NIFSVerifierCircuitParams,
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
    let r1cs_shape_padded_primary = r1cs_shape_primary.pad();

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
    let r1cs_shape_padded_secondary = r1cs_shape_secondary.pad();

    Self {
      ro_consts_primary,
      ro_consts_circuit_primary,
      r1cs_gens_primary,
      r1cs_shape_primary,
      r1cs_shape_padded_primary,
      ro_consts_secondary,
      ro_consts_circuit_secondary,
      r1cs_gens_secondary,
      r1cs_shape_secondary,
      r1cs_shape_padded_secondary,
      c_primary,
      c_secondary,
      params_primary,
      params_secondary,
    }
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
  _p_c1: PhantomData<C1>,
  _p_c2: PhantomData<C2>,
}

impl<G1, G2, C1, C2> RecursiveSNARK<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar> + Clone,
  C2: StepCircuit<G2::Scalar> + Clone,
{
  /// Create a new `RecursiveSNARK`
  pub fn prove(
    pp: &PublicParams<G1, G2, C1, C2>,
    num_steps: usize,
    z0_primary: G1::Scalar,
    z0_secondary: G2::Scalar,
  ) -> Result<Self, NovaError> {
    if num_steps == 0 {
      return Err(NovaError::InvalidNumSteps);
    }

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
      pp.c_primary.clone(),
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
      pp.c_secondary.clone(),
      pp.ro_consts_circuit_secondary.clone(),
    );
    let _ = circuit_secondary.synthesize(&mut cs_secondary);
    let (u_secondary, w_secondary) = cs_secondary
      .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.r1cs_gens_secondary)
      .map_err(|_e| NovaError::UnSat)?;

    // execute the remaining steps, alternating between G1 and G2
    let mut l_w_primary = w_primary;
    let mut l_u_primary = u_primary;
    let mut r_W_primary =
      RelaxedR1CSWitness::from_r1cs_witness(&pp.r1cs_shape_primary, &l_w_primary);
    let mut r_U_primary = RelaxedR1CSInstance::from_r1cs_instance(
      &pp.r1cs_gens_primary,
      &pp.r1cs_shape_primary,
      &l_u_primary,
    );

    let mut r_W_secondary = RelaxedR1CSWitness::<G2>::default(&pp.r1cs_shape_secondary);
    let mut r_U_secondary =
      RelaxedR1CSInstance::<G2>::default(&pp.r1cs_gens_secondary, &pp.r1cs_shape_secondary);
    let mut l_w_secondary = w_secondary;
    let mut l_u_secondary = u_secondary;

    let mut z_next_primary = z0_primary;
    let mut z_next_secondary = z0_secondary;
    z_next_primary = pp.c_primary.compute(&z_next_primary);
    z_next_secondary = pp.c_secondary.compute(&z_next_secondary);

    for i in 1..num_steps {
      // fold the secondary circuit's instance
      let (nifs_secondary, (r_U_next_secondary, r_W_next_secondary)) = NIFS::prove(
        &pp.r1cs_gens_secondary,
        &pp.ro_consts_secondary,
        &pp.r1cs_shape_secondary,
        &r_U_secondary,
        &r_W_secondary,
        &l_u_secondary,
        &l_w_secondary,
      )?;

      let mut cs_primary: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
      let inputs_primary: NIFSVerifierCircuitInputs<G2> = NIFSVerifierCircuitInputs::new(
        pp.r1cs_shape_secondary.get_digest(),
        G1::Scalar::from(i as u64),
        z0_primary,
        Some(z_next_primary),
        Some(r_U_secondary),
        Some(l_u_secondary),
        Some(nifs_secondary.comm_T.decompress()?),
      );

      let circuit_primary: NIFSVerifierCircuit<G2, C1> = NIFSVerifierCircuit::new(
        pp.params_primary.clone(),
        Some(inputs_primary),
        pp.c_primary.clone(),
        pp.ro_consts_circuit_primary.clone(),
      );
      let _ = circuit_primary.synthesize(&mut cs_primary);

      (l_u_primary, l_w_primary) = cs_primary
        .r1cs_instance_and_witness(&pp.r1cs_shape_primary, &pp.r1cs_gens_primary)
        .map_err(|_e| NovaError::UnSat)?;

      // fold the primary circuit's instance
      let (nifs_primary, (r_U_next_primary, r_W_next_primary)) = NIFS::prove(
        &pp.r1cs_gens_primary,
        &pp.ro_consts_primary,
        &pp.r1cs_shape_primary,
        &r_U_primary.clone(),
        &r_W_primary.clone(),
        &l_u_primary.clone(),
        &l_w_primary.clone(),
      )?;

      let mut cs_secondary: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
      let inputs_secondary: NIFSVerifierCircuitInputs<G1> = NIFSVerifierCircuitInputs::new(
        pp.r1cs_shape_primary.get_digest(),
        G2::Scalar::from(i as u64),
        z0_secondary,
        Some(z_next_secondary),
        Some(r_U_primary.clone()),
        Some(l_u_primary.clone()),
        Some(nifs_primary.comm_T.decompress()?),
      );

      let circuit_secondary: NIFSVerifierCircuit<G1, C2> = NIFSVerifierCircuit::new(
        pp.params_secondary.clone(),
        Some(inputs_secondary),
        pp.c_secondary.clone(),
        pp.ro_consts_circuit_secondary.clone(),
      );
      let _ = circuit_secondary.synthesize(&mut cs_secondary);

      (l_u_secondary, l_w_secondary) = cs_secondary
        .r1cs_instance_and_witness(&pp.r1cs_shape_secondary, &pp.r1cs_gens_secondary)
        .map_err(|_e| NovaError::UnSat)?;

      // update the running instances and witnesses
      r_U_secondary = r_U_next_secondary;
      r_W_secondary = r_W_next_secondary;
      r_U_primary = r_U_next_primary;
      r_W_primary = r_W_next_primary;
      z_next_primary = pp.c_primary.compute(&z_next_primary);
      z_next_secondary = pp.c_secondary.compute(&z_next_secondary);
    }

    Ok(Self {
      r_W_primary,
      r_U_primary,
      l_w_primary,
      l_u_primary,
      r_W_secondary,
      r_U_secondary,
      l_w_secondary,
      l_u_secondary,
      zn_primary: z_next_primary,
      zn_secondary: z_next_secondary,
      _p_c1: Default::default(),
      _p_c2: Default::default(),
    })
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

/// A SNARK that proves the knowledge of a valid `RecursiveSNARK`
pub struct CompressedSNARK<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar> + Clone,
  C2: StepCircuit<G2::Scalar> + Clone,
{
  r_U_primary: RelaxedR1CSInstance<G1>,
  l_u_primary: R1CSInstance<G1>,
  nifs_primary: NIFS<G1>,
  f_W_snark_primary: RelaxedR1CSSNARK<G1>,

  r_U_secondary: RelaxedR1CSInstance<G2>,
  l_u_secondary: R1CSInstance<G2>,
  nifs_secondary: NIFS<G2>,
  f_W_snark_secondary: RelaxedR1CSSNARK<G2>,

  zn_primary: G1::Scalar,
  zn_secondary: G2::Scalar,

  _p_c1: PhantomData<C1>,
  _p_c2: PhantomData<C2>,
}

impl<G1, G2, C1, C2> CompressedSNARK<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar> + Clone,
  C2: StepCircuit<G2::Scalar> + Clone,
{
  /// Create a new `CompressedSNARK`
  pub fn prove(
    pp: &PublicParams<G1, G2, C1, C2>,
    recursive_snark: &RecursiveSNARK<G1, G2, C1, C2>,
  ) -> Result<Self, NovaError> {
    let (res_primary, res_secondary) = rayon::join(
      // fold the primary circuit's instance
      || {
        NIFS::prove(
          &pp.r1cs_gens_primary,
          &pp.ro_consts_primary,
          &pp.r1cs_shape_primary,
          &recursive_snark.r_U_primary,
          &recursive_snark.r_W_primary,
          &recursive_snark.l_u_primary,
          &recursive_snark.l_w_primary,
        )
      },
      || {
        // fold the secondary circuit's instance
        NIFS::prove(
          &pp.r1cs_gens_secondary,
          &pp.ro_consts_secondary,
          &pp.r1cs_shape_secondary,
          &recursive_snark.r_U_secondary,
          &recursive_snark.r_W_secondary,
          &recursive_snark.l_u_secondary,
          &recursive_snark.l_w_secondary,
        )
      },
    );

    let (nifs_primary, (f_U_primary, f_W_primary)) = res_primary?;
    let (nifs_secondary, (f_U_secondary, f_W_secondary)) = res_secondary?;

    // create SNARKs proving the knowledge of f_W_primary and f_W_secondary
    let (f_W_snark_primary, f_W_snark_secondary) = rayon::join(
      || {
        RelaxedR1CSSNARK::prove(
          &pp.r1cs_gens_primary,
          &pp.r1cs_shape_padded_primary,
          &f_U_primary,
          &f_W_primary,
        )
      },
      || {
        RelaxedR1CSSNARK::prove(
          &pp.r1cs_gens_secondary,
          &pp.r1cs_shape_padded_secondary,
          &f_U_secondary,
          &f_W_secondary,
        )
      },
    );

    Ok(Self {
      r_U_primary: recursive_snark.r_U_primary.clone(),
      l_u_primary: recursive_snark.l_u_primary.clone(),
      nifs_primary,
      f_W_snark_primary: f_W_snark_primary?,

      r_U_secondary: recursive_snark.r_U_secondary.clone(),
      l_u_secondary: recursive_snark.l_u_secondary.clone(),
      nifs_secondary,
      f_W_snark_secondary: f_W_snark_secondary?,

      zn_primary: recursive_snark.zn_primary,
      zn_secondary: recursive_snark.zn_secondary,

      _p_c1: Default::default(),
      _p_c2: Default::default(),
    })
  }

  /// Verify the correctness of the `CompressedSNARK`
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

    // fold the running instance and last instance to get a folded instance
    let f_U_primary = self.nifs_primary.verify(
      &pp.ro_consts_primary,
      &pp.r1cs_shape_primary,
      &self.r_U_primary,
      &self.l_u_primary,
    )?;
    let f_U_secondary = self.nifs_secondary.verify(
      &pp.ro_consts_secondary,
      &pp.r1cs_shape_secondary,
      &self.r_U_secondary,
      &self.l_u_secondary,
    )?;

    // check the satisfiability of the folded instances using SNARKs proving the knowledge of their satisfying witnesses
    let (res_primary, res_secondary) = rayon::join(
      || {
        self.f_W_snark_primary.verify(
          &pp.r1cs_gens_primary,
          &pp.r1cs_shape_padded_primary,
          &f_U_primary,
        )
      },
      || {
        self.f_W_snark_secondary.verify(
          &pp.r1cs_gens_secondary,
          &pp.r1cs_shape_padded_secondary,
          &f_U_secondary,
        )
      },
    );

    res_primary?;
    res_secondary?;

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
    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      z: AllocatedNum<F>,
    ) -> Result<AllocatedNum<F>, SynthesisError> {
      Ok(z)
    }

    fn compute(&self, z: &F) -> F {
      *z
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
    fn synthesize<CS: ConstraintSystem<F>>(
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

    fn compute(&self, z: &F) -> F {
      *z * *z * *z + z + F::from(5u64)
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
      3,
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
      num_steps,
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

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, <G1 as Group>::Scalar::one());
    let mut zn_secondary_direct = <G2 as Group>::Scalar::zero();
    for _i in 0..num_steps {
      zn_secondary_direct = CubicCircuit {
        _p: Default::default(),
      }
      .compute(&zn_secondary_direct);
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(zn_secondary, <G2 as Group>::Scalar::from(2460515u64));
  }

  #[test]
  fn test_ivc_nontrivial_with_compression() {
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
      num_steps,
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

    let (zn_primary, zn_secondary) = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(zn_primary, <G1 as Group>::Scalar::one());
    let mut zn_secondary_direct = <G2 as Group>::Scalar::zero();
    for _i in 0..num_steps {
      zn_secondary_direct = CubicCircuit {
        _p: Default::default(),
      }
      .compute(&zn_secondary_direct);
    }
    assert_eq!(zn_secondary, zn_secondary_direct);
    assert_eq!(zn_secondary, <G2 as Group>::Scalar::from(2460515u64));

    // produce a compressed SNARK
    let res = CompressedSNARK::prove(&pp, &recursive_snark);
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    let res = compressed_snark.verify(
      &pp,
      num_steps,
      <G1 as Group>::Scalar::one(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());
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
      num_steps,
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

    let (zn_primary, zn_secondary) = res.unwrap();

    assert_eq!(zn_primary, <G1 as Group>::Scalar::one());
    assert_eq!(zn_secondary, <G2 as Group>::Scalar::from(5u64));
  }
}
