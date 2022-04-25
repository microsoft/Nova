//! There are two Verification Circuits. The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! TODO: The base case is different for the primary and the secondary.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is
//! H(params = H(shape, gens), i, z0, zi, U). Each circuit folds the last invocation of
//! the other into the running instance

use super::{
  commitments::Commitment,
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
    utils::{alloc_num_equals, alloc_zero, conditionally_select, le_bits_to_num},
  },
  poseidon::{NovaPoseidonConstants, PoseidonROGadget},
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{Group, StepCircuit},
};
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    Assignment,
  },
  Circuit, ConstraintSystem, SynthesisError,
};
use ff::{Field, PrimeField, PrimeFieldBits};

#[derive(Debug, Clone)]
pub struct NIFSVerifierCircuitParams {
  limb_width: usize,
  n_limbs: usize,
}

impl NIFSVerifierCircuitParams {
  #[allow(dead_code)]
  pub fn new(limb_width: usize, n_limbs: usize) -> Self {
    Self {
      limb_width,
      n_limbs,
    }
  }
}

pub struct NIFSVerifierCircuitInputs<G>
where
  G: Group,
{
  params: G::Base, // Hash(Shape of u2, Gens for u2). Needed for computing the challenge.
  i: G::Base,
  z0: G::Base,
  zi: G::Base,
  U: RelaxedR1CSInstance<G>,
  u: R1CSInstance<G>,
  T: Commitment<G>,
}

impl<G> NIFSVerifierCircuitInputs<G>
where
  G: Group,
{
  /// Create new inputs/witness for the verification circuit
  #[allow(dead_code, clippy::too_many_arguments)]
  pub fn new(
    params: G::Base,
    i: G::Base,
    z0: G::Base,
    zi: G::Base,
    U: RelaxedR1CSInstance<G>,
    u: R1CSInstance<G>,
    T: Commitment<G>,
  ) -> Self {
    Self {
      params,
      i,
      z0,
      zi,
      U,
      u,
      T,
    }
  }
}

/// Circuit that encodes only the folding verifier
pub struct NIFSVerifierCircuit<G, SC>
where
  G: Group,
  <G as Group>::Base: ff::PrimeField,
  SC: StepCircuit<G::Base>,
{
  params: NIFSVerifierCircuitParams,
  inputs: Option<NIFSVerifierCircuitInputs<G>>,
  step_circuit: SC, // The function that is applied for each step
  poseidon_constants: NovaPoseidonConstants<G::Base>,
}

impl<G, SC> NIFSVerifierCircuit<G, SC>
where
  G: Group,
  <G as Group>::Base: PrimeField + PrimeFieldBits,
  <G as Group>::Scalar: PrimeField + PrimeFieldBits,
  SC: StepCircuit<G::Base>,
{
  /// Create a new verification circuit for the input relaxed r1cs instances
  #[allow(dead_code)]
  pub fn new(
    params: NIFSVerifierCircuitParams,
    inputs: Option<NIFSVerifierCircuitInputs<G>>,
    step_circuit: SC,
    poseidon_constants: NovaPoseidonConstants<G::Base>,
  ) -> Self
  where
    <G as Group>::Base: ff::PrimeField,
  {
    Self {
      params,
      inputs,
      step_circuit,
      poseidon_constants,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedRelaxedR1CSInstance<G>,
      AllocatedR1CSInstance<G>,
      AllocatedPoint<G::Base>,
    ),
    SynthesisError,
  > {
    // Allocate the params
    let params = AllocatedNum::alloc(cs.namespace(|| "params"), || Ok(self.inputs.get()?.params))?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate z0
    let z_0 = AllocatedNum::alloc(cs.namespace(|| "z0"), || Ok(self.inputs.get()?.z0))?;

    // Allocate zi
    let z_i = AllocatedNum::alloc(cs.namespace(|| "zi"), || Ok(self.inputs.get()?.zi))?;

    // Allocate the running instance
    let U: AllocatedRelaxedR1CSInstance<G> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self
        .inputs
        .get()
        .map_or(None, |inputs| Some(inputs.U.clone())),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Allocate the instance to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self
        .inputs
        .get()
        .map_or(None, |inputs| Some(inputs.u.clone())),
    )?;

    // Allocate T
    let T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      self
        .inputs
        .get()
        .map_or(None, |inputs| Some(inputs.T.comm.to_coordinates())),
    )?;

    Ok((params, i, z_0, z_i, U, u, T))
  }

  /// Synthesizes base case and returns the new relaxed R1CSInstance
  fn synthesize_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedRelaxedR1CSInstance<G>, SynthesisError> {
    let U_default: AllocatedRelaxedR1CSInstance<G> = AllocatedRelaxedR1CSInstance::default(
      cs.namespace(|| "Allocate U_default"),
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    Ok(U_default)
  }

  /// Synthesizes non base case and returns the new relaxed R1CSInstance
  /// And a boolean indicating if all checks pass
  #[allow(clippy::too_many_arguments)]
  fn synthesize_non_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    params: AllocatedNum<G::Base>,
    i: AllocatedNum<G::Base>,
    z_0: AllocatedNum<G::Base>,
    z_i: AllocatedNum<G::Base>,
    U: AllocatedRelaxedR1CSInstance<G>,
    u: AllocatedR1CSInstance<G>,
    T: AllocatedPoint<G::Base>,
  ) -> Result<(AllocatedRelaxedR1CSInstance<G>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U,i,z0,zi)
    let mut ro: PoseidonROGadget<G::Base> = PoseidonROGadget::new(self.poseidon_constants.clone());
    ro.absorb(params);
    ro.absorb(i);
    ro.absorb(z_0);
    ro.absorb(z_i);
    let _ = U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;

    let hash_bits = ro.get_hash(cs.namespace(|| "Input hash"))?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;
    let check_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      u.X0.clone(),
      hash,
    )?;

    // Run NIFS Verifier
    let U_fold = U.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      u,
      T,
      self.poseidon_constants.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    Ok((U_fold, check_pass))
  }
}

impl<G, SC> Circuit<<G as Group>::Base> for NIFSVerifierCircuit<G, SC>
where
  G: Group,
  <G as Group>::Base: PrimeField + PrimeFieldBits,
  <G as Group>::Scalar: PrimeFieldBits,
  SC: StepCircuit<G::Base>,
{
  fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    // Allocate all witnesses
    let (params, i, z_0, z_i, U, u, T) =
      self.alloc_witness(cs.namespace(|| "allocate the circuit witness"))?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), i.clone(), zero)?; //TODO: maybe optimize this?

    // Synthesize the circuit for the base case and get the new running instance
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "base case"))?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      params.clone(),
      i.clone(),
      z_0.clone(),
      z_i.clone(),
      U,
      u.clone(),
      T,
    )?;

    // Either check_non_base_pass=true or we are in the base case
    let should_be_false = AllocatedBit::nor(
      cs.namespace(|| "check_non_base_pass nor base_case"),
      &check_non_base_pass,
      &is_base_case,
    )?;
    cs.enforce(
      || "check_non_base_pass nor base_case = false",
      |lc| lc + should_be_false.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc,
    );

    // Compute the U_new
    let Unew = Unew_base.conditionally_select(
      cs.namespace(|| "compute U_new"),
      Unew_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + G::Base::one())
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case),
    )?;
    let z_next = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), z_input)?;

    // Compute the new hash H(params, Unew, i+1, z0, z_{i+1})
    let mut ro: PoseidonROGadget<G::Base> = PoseidonROGadget::new(self.poseidon_constants);
    ro.absorb(params);
    ro.absorb(i_new.clone());
    ro.absorb(z_0);
    ro.absorb(z_next);
    let _ = Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    let hash_bits = ro.get_hash(cs.namespace(|| "output hash bits"))?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), hash_bits)?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    let _ = hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;
    let _ = u
      .X1
      .inputize(cs.namespace(|| "Output unmodified hash of the other circuit"))?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellperson::{shape_cs::ShapeCS, solver::SatisfyingAssignment};
  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;
  use crate::{
    bellperson::r1cs::{NovaShape, NovaWitness},
    commitments::CommitTrait,
  };
  use std::marker::PhantomData;

  struct TestCircuit<F>
  where
    F: PrimeField + ff::PrimeField,
  {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for TestCircuit<F>
  where
    F: PrimeField + ff::PrimeField,
  {
    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      z: AllocatedNum<F>,
    ) -> Result<AllocatedNum<F>, SynthesisError> {
      Ok(z)
    }
  }

  #[test]
  fn test_verification_circuit() {
    // We experiment with 8 limbs of 32 bits each
    let params = NIFSVerifierCircuitParams::new(32, 8);
    // The first circuit that verifies G2
    let poseidon_constants1: NovaPoseidonConstants<<G2 as Group>::Base> =
      NovaPoseidonConstants::new();
    let circuit1: NIFSVerifierCircuit<G2, TestCircuit<<G2 as Group>::Base>> =
      NIFSVerifierCircuit::new(
        params.clone(),
        None,
        TestCircuit {
          _p: Default::default(),
        },
        poseidon_constants1.clone(),
      );

    // First create the shape
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let (shape1, gens1) = (cs.r1cs_shape(), cs.r1cs_gens());
    println!(
      "Circuit1 -> Number of constraints: {}",
      cs.num_constraints()
    );

    // The second circuit that verifies G1
    let poseidon_constants2: NovaPoseidonConstants<<G1 as Group>::Base> =
      NovaPoseidonConstants::new();
    let circuit2: NIFSVerifierCircuit<G1, TestCircuit<<G1 as Group>::Base>> =
      NIFSVerifierCircuit::new(
        params.clone(),
        None,
        TestCircuit {
          _p: Default::default(),
        },
        poseidon_constants2,
      );
    // First create the shape
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let (shape2, gens2) = (cs.r1cs_shape(), cs.r1cs_gens());
    println!(
      "Circuit2 -> Number of constraints: {}",
      cs.num_constraints()
    );

    let zero = <<G2 as Group>::Base as Field>::zero();
    let zero_fq = <<G2 as Group>::Scalar as Field>::zero();
    let T = vec![<G2 as Group>::Scalar::zero()].commit(&gens2.gens_E);
    let w = vec![<G2 as Group>::Scalar::zero()].commit(&gens2.gens_E);
    // Now get an assignment
    let mut cs: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs: NIFSVerifierCircuitInputs<G2> = NIFSVerifierCircuitInputs::new(
      <<G2 as Group>::Base as Field>::zero(), // TODO: provide real inputs
      zero,                                   // TODO: provide real inputs
      zero,                                   // TODO: provide real inputs
      zero,                                   // TODO: provide real inputs
      RelaxedR1CSInstance::default(&gens2, &shape2),
      R1CSInstance::new(&shape2, &w, &[zero_fq, zero_fq]).unwrap(),
      T, // TODO: provide real inputs
    );

    let circuit: NIFSVerifierCircuit<G2, TestCircuit<<G2 as Group>::Base>> =
      NIFSVerifierCircuit::new(
        params,
        Some(inputs),
        TestCircuit {
          _p: Default::default(),
        },
        poseidon_constants1,
      );
    let _ = circuit.synthesize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape1, &gens1).unwrap();

    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&gens1, &inst, &witness).is_ok());
  }
}
