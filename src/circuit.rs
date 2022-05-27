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
    utils::{
      alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select, le_bits_to_num,
    },
  },
  poseidon::{PoseidonROGadget, ROConstantsCircuit},
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
use ff::Field;
use neptune::{poseidon::PoseidonConstants, Arity, Strength};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct NIFSVerifierCircuitParams {
  limb_width: usize,
  n_limbs: usize,
  is_primary_circuit: bool, // A boolean indicating if this is the primary circuit
}

impl NIFSVerifierCircuitParams {
  pub fn new(limb_width: usize, n_limbs: usize, is_primary_circuit: bool) -> Self {
    Self {
      limb_width,
      n_limbs,
      is_primary_circuit,
    }
  }
}

#[derive(Debug)]
pub struct NIFSVerifierCircuitInputs<G: Group> {
  params: G::Scalar, // Hash(Shape of u2, Gens for u2). Needed for computing the challenge.
  i: G::Base,
  z0: G::Base,
  zi: Option<G::Base>,
  U: Option<RelaxedR1CSInstance<G>>,
  u: Option<R1CSInstance<G>>,
  T: Option<Commitment<G>>,
}

impl<G> NIFSVerifierCircuitInputs<G>
where
  G: Group,
{
  /// Create new inputs/witness for the verification circuit
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    params: G::Scalar,
    i: G::Base,
    z0: G::Base,
    zi: Option<G::Base>,
    U: Option<RelaxedR1CSInstance<G>>,
    u: Option<R1CSInstance<G>>,
    T: Option<Commitment<G>>,
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
pub struct NIFSVerifierCircuit<G, SC, A>
where
  G: Group,
  SC: StepCircuit<G::Base, A>,
  A: Arity<G::Base>,
{
  params: NIFSVerifierCircuitParams,
  ro_consts: ROConstantsCircuit<G::Base>,
  arity_consts: Option<PoseidonConstants<G::Base, A>>,
  inputs: Option<NIFSVerifierCircuitInputs<G>>,
  step_circuit: SC, // The function that is applied for each step
  _a: PhantomData<A>,
}

impl<G, SC, A> NIFSVerifierCircuit<G, SC, A>
where
  G: Group,
  SC: StepCircuit<G::Base, A>,
  A: Arity<G::Base>,
{
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub fn new(
    params: NIFSVerifierCircuitParams,
    inputs: Option<NIFSVerifierCircuitInputs<G>>,
    step_circuit: SC,
    ro_consts: ROConstantsCircuit<G::Base>,
  ) -> Self {
    let arity_consts = if A::to_usize() == 1 {
      None
    } else {
      Some(PoseidonConstants::new_with_strength(Strength::Standard))
    };
    Self {
      params,
      inputs,
      step_circuit,
      ro_consts,
      arity_consts,
      _a: Default::default(),
    }
  }

  pub fn arity_consts(&self) -> Option<PoseidonConstants<G::Base, A>> {
    self.arity_consts.clone()
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
    let params = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "params"),
      self.inputs.get().map_or(None, |inputs| Some(inputs.params)),
    )?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate z0
    let z_0 = AllocatedNum::alloc(cs.namespace(|| "z0"), || Ok(self.inputs.get()?.z0))?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let z_i = AllocatedNum::alloc(cs.namespace(|| "zi"), || {
      Ok(self.inputs.get()?.zi.unwrap_or_else(G::Base::zero))
    })?;

    // Allocate the running instance
    let U: AllocatedRelaxedR1CSInstance<G> = AllocatedRelaxedR1CSInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.get().map_or(None, |inputs| {
        inputs.U.get().map_or(None, |U| Some(U.clone()))
      }),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Allocate the instance to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.get().map_or(None, |inputs| {
        inputs.u.get().map_or(None, |u| Some(u.clone()))
      }),
    )?;

    // Allocate T
    let T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      self.inputs.get().map_or(None, |inputs| {
        inputs
          .T
          .get()
          .map_or(None, |T| Some(T.comm.to_coordinates()))
      }),
    )?;

    Ok((params, i, z_0, z_i, U, u, T))
  }

  /// Synthesizes base case and returns the new relaxed R1CSInstance
  fn synthesize_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    u: AllocatedR1CSInstance<G>,
  ) -> Result<AllocatedRelaxedR1CSInstance<G>, SynthesisError> {
    let U_default: AllocatedRelaxedR1CSInstance<G> = if self.params.is_primary_circuit {
      // The primary circuit just returns the default R1CS instance
      AllocatedRelaxedR1CSInstance::default(
        cs.namespace(|| "Allocate U_default"),
        self.params.limb_width,
        self.params.n_limbs,
      )?
    } else {
      // The secondary circuit returns the incoming R1CS instance
      AllocatedRelaxedR1CSInstance::from_r1cs_instance(
        cs.namespace(|| "Allocate U_default"),
        u,
        self.params.limb_width,
        self.params.n_limbs,
      )?
    };
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
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro: PoseidonROGadget<G::Base> = PoseidonROGadget::new(self.ro_consts.clone());
    ro.absorb(params.clone());
    ro.absorb(i);
    ro.absorb(z_0);
    ro.absorb(z_i);
    let _ = U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;

    let hash_bits = ro.get_hash(cs.namespace(|| "Input hash"))?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;
    let check_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X0,
      &hash,
    )?;

    // Run NIFS Verifier
    let U_fold = U.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      params,
      u,
      T,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    Ok((U_fold, check_pass))
  }
}

impl<G, SC, A> Circuit<<G as Group>::Base> for NIFSVerifierCircuit<G, SC, A>
where
  G: Group,
  SC: StepCircuit<G::Base, A>,
  A: Arity<G::Base>,
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
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?; //TODO: maybe optimize this?

    // Synthesize the circuit for the base case and get the new running instance
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "base case"), u.clone())?;

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
    let z_next = self.step_circuit.synthesize_step_outer(
      &mut cs.namespace(|| "F"),
      z_input,
      self.arity_consts.as_ref(),
    )?;

    // Compute the new hash H(params, Unew, i+1, z0, z_{i+1})
    let mut ro: PoseidonROGadget<G::Base> = PoseidonROGadget::new(self.ro_consts);
    ro.absorb(params);
    ro.absorb(i_new.clone());
    ro.absorb(z_0);
    ro.absorb(z_next);
    let _ = Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    let hash_bits = ro.get_hash(cs.namespace(|| "output hash bits"))?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), hash_bits)?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    let _ = u
      .X1
      .inputize(cs.namespace(|| "Output unmodified hash of the other circuit"))?;
    let _ = hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellperson::{shape_cs::ShapeCS, solver::SatisfyingAssignment};
  type G1 = pasta_curves::pallas::Point;
  type G2 = pasta_curves::vesta::Point;
  use crate::constants::{BN_LIMB_WIDTH, BN_N_LIMBS};
  use crate::{
    bellperson::r1cs::{NovaShape, NovaWitness},
    traits::{HashFuncConstantsTrait, StepCompute},
  };
  use ff::PrimeField;
  use generic_array::typenum::U1;
  use std::marker::PhantomData;

  #[derive(Clone)]
  struct TestCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F, U1> for TestCircuit<F>
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
  impl<'a, F> StepCompute<'a, F, U1> for TestCircuit<F>
  where
    F: PrimeField,
  {
    fn compute(&self, z: &F) -> Option<(Self, F)> {
      Some((self.clone(), *z))
    }
  }

  #[test]
  fn test_verification_circuit() {
    // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
    let params1 = NIFSVerifierCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = NIFSVerifierCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<<G2 as Group>::Base> = ROConstantsCircuit::new();
    let ro_consts2: ROConstantsCircuit<<G1 as Group>::Base> = ROConstantsCircuit::new();

    // Initialize the shape and gens for the primary
    let circuit1: NIFSVerifierCircuit<G2, TestCircuit<<G2 as Group>::Base>, U1> =
      NIFSVerifierCircuit::new(
        params1.clone(),
        None,
        TestCircuit {
          _p: Default::default(),
        },
        ro_consts1.clone(),
      );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let (shape1, gens1) = (cs.r1cs_shape(), cs.r1cs_gens());
    assert_eq!(14914, cs.num_constraints());
    println!(
      "Circuit1 -> Number of constraints: {}",
      cs.num_constraints()
    );

    // Initialize the shape and gens for the secondary
    let circuit2: NIFSVerifierCircuit<G1, TestCircuit<<G1 as Group>::Base>, U1> =
      NIFSVerifierCircuit::new(
        params2.clone(),
        None,
        TestCircuit {
          _p: Default::default(),
        },
        ro_consts2.clone(),
      );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let (shape2, gens2) = (cs.r1cs_shape(), cs.r1cs_gens());
    assert_eq!(15454, cs.num_constraints());
    println!(
      "Circuit2 -> Number of constraints: {}",
      cs.num_constraints()
    );

    // Execute the base case for the primary
    let zero1 = <<G2 as Group>::Base as Field>::zero();
    let mut cs1: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs1: NIFSVerifierCircuitInputs<G2> = NIFSVerifierCircuitInputs::new(
      shape2.get_digest(),
      zero1,
      zero1, // TODO: Provide real input for z0
      None,
      None,
      None,
      None,
    );
    let circuit1: NIFSVerifierCircuit<G2, TestCircuit<<G2 as Group>::Base>, U1> =
      NIFSVerifierCircuit::new(
        params1,
        Some(inputs1),
        TestCircuit {
          _p: Default::default(),
        },
        ro_consts1,
      );
    let _ = circuit1.synthesize(&mut cs1);
    let (inst1, witness1) = cs1.r1cs_instance_and_witness(&shape1, &gens1).unwrap();
    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&gens1, &inst1, &witness1).is_ok());

    // Execute the base case for the secondary
    let zero2 = <<G1 as Group>::Base as Field>::zero();
    let mut cs2: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs2: NIFSVerifierCircuitInputs<G1> = NIFSVerifierCircuitInputs::new(
      shape1.get_digest(),
      zero2,
      zero2,
      None,
      None,
      Some(inst1),
      None,
    );
    let circuit: NIFSVerifierCircuit<G1, TestCircuit<<G1 as Group>::Base>, U1> =
      NIFSVerifierCircuit::new(
        params2,
        Some(inputs2),
        TestCircuit {
          _p: Default::default(),
        },
        ro_consts2,
      );
    let _ = circuit.synthesize(&mut cs2);
    let (inst2, witness2) = cs2.r1cs_instance_and_witness(&shape2, &gens2).unwrap();
    // Make sure that it is satisfiable
    assert!(shape2.is_sat(&gens2, &inst2, &witness2).is_ok());
  }
}
