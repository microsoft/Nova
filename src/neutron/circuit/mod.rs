//! There are two augmented circuits: the primary and the secondary.
//! Each of them is over a curve in a 2-cycle of elliptic curves.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is H(params = H(shape, ck), i, z0, zi, U).
//! Each circuit folds the last invocation of the other into the running instance

/*use crate::{
  constants::NUM_HASH_BITS,
  frontend::{
    num::AllocatedNum, AllocatedBit, Assignment, Boolean, ConstraintSystem, SynthesisError,
  },
  gadgets::{
    r1cs::AllocatedR1CSInstance,
    utils::{
      alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select_vec, le_bits_to_num,
    },
  },
  neutron::{nifs::NIFS, relation::FoldedInstance},
  r1cs::R1CSInstance,
  traits::{circuit::StepCircuit, Engine, ROCircuitTrait, ROConstantsCircuit},
};
use ff::Field;
use serde::{Deserialize, Serialize};*/

pub mod nifs;
pub mod r1cs;
pub mod relation;
pub mod univariate;

/*
use nifs::AllocatedNIFS;
use relation::AllocatedFoldedInstance;

/// A type that holds the parameters for the augmented circuit
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NeutronAugmentedCircuitParams {
  limb_width: usize,
  n_limbs: usize,
  is_primary_circuit: bool, // A boolean indicating if this is the primary circuit
}

impl NeutronAugmentedCircuitParams {
  /// Create new parameters for the augmented circuit
  pub const fn new(limb_width: usize, n_limbs: usize, is_primary_circuit: bool) -> Self {
    Self {
      limb_width,
      n_limbs,
      is_primary_circuit,
    }
  }
}

/// A type that holds the non-deterministic inputs for the augmented circuit
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronAugmentedCircuitInputs<E: Engine> {
  pp_digest: E::Scalar,
  i: E::Base,
  z0: Vec<E::Base>,
  zi: Option<Vec<E::Base>>,
  U: Option<FoldedInstance<E>>,
  ri: Option<E::Base>,
  r_next: E::Base,
  u: Option<R1CSInstance<E>>,
  nifs: Option<NIFS<E>>,
}

impl<E: Engine> NeutronAugmentedCircuitInputs<E> {
  /// Create new inputs/witness for the verification circuit
  pub fn new(
    pp_digest: E::Scalar,
    i: E::Base,
    z0: Vec<E::Base>,
    zi: Option<Vec<E::Base>>,
    U: Option<FoldedInstance<E>>,
    ri: Option<E::Base>,
    r_next: E::Base,
    u: Option<R1CSInstance<E>>,
    nifs: Option<NIFS<E>>,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z0,
      zi,
      U,
      ri,
      r_next,
      u,
      nifs,
    }
  }
}

/// The augmented circuit F' in Neutron that includes a step circuit F
/// and the circuit for the verifier in Neutron's non-interactive folding scheme
pub struct NeutronAugmentedCircuit<'a, E: Engine, SC: StepCircuit<E::Base>> {
  params: &'a NeutronAugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<E>,
  inputs: Option<NeutronAugmentedCircuitInputs<E>>,
  step_circuit: &'a SC, // The function that is applied for each step
}

impl<'a, E: Engine, SC: StepCircuit<E::Base>> NeutronAugmentedCircuit<'a, E, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub const fn new(
    params: &'a NeutronAugmentedCircuitParams,
    inputs: Option<NeutronAugmentedCircuitInputs<E>>,
    step_circuit: &'a SC,
    ro_consts: ROConstantsCircuit<E>,
  ) -> Self {
    Self {
      params,
      inputs,
      step_circuit,
      ro_consts,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E::Base>,
      AllocatedNum<E::Base>,
      Vec<AllocatedNum<E::Base>>,
      Vec<AllocatedNum<E::Base>>,
      AllocatedFoldedInstance<E>,
      AllocatedNum<E::Base>,
      AllocatedNum<E::Base>,
      AllocatedR1CSInstance<E>,
      AllocatedNIFS<E>,
    ),
    SynthesisError,
  > {
    // Allocate the pp_digest
    let pp_digest = alloc_scalar_as_base::<E, _>(
      cs.namespace(|| "pp_digest"),
      self.inputs.as_ref().map(|inputs| inputs.pp_digest),
    )?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate z0
    let z_0 = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
          Ok(self.inputs.get()?.z0[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![E::Base::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // Allocate the running instance
    let U: AllocatedFoldedInstance<E> = AllocatedFoldedInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // Allocate ri
    let r_i = AllocatedNum::alloc(cs.namespace(|| "ri"), || {
      Ok(self.inputs.get()?.ri.unwrap_or(E::Base::ZERO))
    })?;

    // Allocate r_i+1
    let r_next = AllocatedNum::alloc(cs.namespace(|| "r_i+1"), || Ok(self.inputs.get()?.r_next))?;

    // Allocate the instance to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.as_ref().and_then(|inputs| inputs.u.as_ref()),
    )?;

    // Allocate nifs
    let nifs = AllocatedNIFS::alloc(
      cs.namespace(|| "allocate nifs"),
      self.inputs.as_ref().and_then(|inputs| inputs.nifs.as_ref()),
      5, // TODO: take this as input
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    Ok((pp_digest, i, z_0, z_i, U, r_i, r_next, u, nifs))
  }

  /// Synthesizes non base case and returns the new relaxed `FoldedInstance`
  /// And a boolean indicating if all checks pass
  fn synthesize_non_base_case<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Base>,
    i: &AllocatedNum<E::Base>,
    z_0: &[AllocatedNum<E::Base>],
    z_i: &[AllocatedNum<E::Base>],
    U: &AllocatedFoldedInstance<E>,
    r_i: &AllocatedNum<E::Base>,
    u: &AllocatedR1CSInstance<E>,
    nifs: &AllocatedNIFS<E>,
  ) -> Result<(AllocatedFoldedInstance<E>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = E::ROCircuit::new(self.ro_consts.clone());
    ro.absorb(pp_digest);
    ro.absorb(i);
    for e in z_0 {
      ro.absorb(e);
    }
    for e in z_i {
      ro.absorb(e);
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    ro.absorb(r_i);

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;
    let check_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X0,
      &hash,
    )?;

    // Run NIFS Verifier
    let U_fold = nifs.verify(
      cs.namespace(|| "compute fold of U and u"),
      pp_digest,
      U,
      u,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    Ok((U_fold, check_pass))
  }
}

impl<E: Engine, SC: StepCircuit<E::Base>> NeutronAugmentedCircuit<'_, E, SC> {
  /// synthesize circuit giving constraint system
  pub fn synthesize<CS: ConstraintSystem<<E as Engine>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Base>>, SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (pp_digest, i, z_0, z_i, U, r_i, r_next, u, nifs) =
      self.alloc_witness(cs.namespace(|| "allocate the circuit witness"), arity)?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &pp_digest,
      &i,
      &z_0,
      &z_i,
      &U,
      &r_i,
      &u,
      &nifs,
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

    // In the primary circuit, the base case simply returns a default running instance
    // In the secondary circuit, the base case returns Unew_non_base
    let Unew = if self.params.is_primary_circuit {
      let Unew_base = AllocatedFoldedInstance::default(
        cs.namespace(|| "Allocate U_default"),
        self.params.limb_width,
        self.params.n_limbs,
      )?;
      Unew_base.conditionally_select(
        cs.namespace(|| "compute U_new"),
        &Unew_non_base,
        &Boolean::from(is_base_case.clone()),
      )?
    } else {
      // TODO: This ends up folding something with unconstrained running instance
      // enforce checks on running instance
      Unew_non_base
    };

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E::Base::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case),
    )?;

    let z_next = self
      .step_circuit
      .synthesize(&mut cs.namespace(|| "F"), &z_input)?;

    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }

    // Compute the new hash H(pp_digest, Unew, i+1, z0, z_{i+1})
    let mut ro = E::ROCircuit::new(self.ro_consts);
    ro.absorb(&pp_digest);
    ro.absorb(&i_new);
    for e in &z_0 {
      ro.absorb(e);
    }
    for e in &z_next {
      ro.absorb(e);
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    ro.absorb(&r_next);
    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), &hash_bits)?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    u.X1
      .inputize(cs.namespace(|| "Output unmodified hash of the other circuit"))?;
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(z_next)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
    },
    gadgets::utils::scalar_as_base,
    provider::{
      poseidon::PoseidonConstantsCircuit, Bn256EngineKZG, GrumpkinEngine, PallasEngine,
      Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::{circuit::TrivialCircuit, snark::default_ck_hint},
  };
  use expect_test::{expect, Expect};

  // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
  fn test_recursive_circuit_with<E1, E2>(
    primary_params: &NeutronAugmentedCircuitParams,
    secondary_params: &NeutronAugmentedCircuitParams,
    ro_consts1: ROConstantsCircuit<E2>,
    ro_consts2: ROConstantsCircuit<E1>,
    num_constraints_primary: &Expect,
    num_constraints_secondary: &Expect,
  ) where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let tc1 = TrivialCircuit::default();
    // Initialize the shape and ck for the primary
    let circuit1: NeutronAugmentedCircuit<'_, E2, TrivialCircuit<<E2 as Engine>::Base>> =
      NeutronAugmentedCircuit::new(primary_params, None, &tc1, ro_consts1.clone());
    let mut cs: TestShapeCS<E1> = TestShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let (shape1, ck1) = cs.r1cs_shape(&*default_ck_hint());
    num_constraints_primary.assert_eq(cs.num_constraints().to_string().as_str());

    let tc2 = TrivialCircuit::default();
    // Initialize the shape and ck for the secondary
    let circuit2: NeutronAugmentedCircuit<'_, E1, TrivialCircuit<<E1 as Engine>::Base>> =
      NeutronAugmentedCircuit::new(secondary_params, None, &tc2, ro_consts2.clone());
    let mut cs: TestShapeCS<E2> = TestShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let (shape2, ck2) = cs.r1cs_shape(&*default_ck_hint());
    num_constraints_secondary.assert_eq(cs.num_constraints().to_string().as_str());

    // Execute the base case for the primary
    let zero1 = <<E2 as Engine>::Base as Field>::ZERO;
    let ri_1 = <<E2 as Engine>::Base as Field>::ZERO;
    let mut cs1 = SatisfyingAssignment::<E1>::new();
    let inputs1: NeutronAugmentedCircuitInputs<E2> = NeutronAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(zero1), // pass zero for testing
      zero1,
      vec![zero1],
      None,
      None,
      None,
      ri_1,
      None,
      None,
    );
    let circuit1: NeutronAugmentedCircuit<'_, E2, TrivialCircuit<<E2 as Engine>::Base>> =
      NeutronAugmentedCircuit::new(primary_params, Some(inputs1), &tc1, ro_consts1);
    let _ = circuit1.synthesize(&mut cs1);
    let (inst1, witness1) = cs1.r1cs_instance_and_witness(&shape1, &ck1).unwrap();
    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&ck1, &inst1, &witness1).is_ok());

    // Execute the base case for the secondary
    let zero2 = <<E1 as Engine>::Base as Field>::ZERO;
    let ri_2 = <<E1 as Engine>::Base as Field>::ZERO;
    let mut cs2 = SatisfyingAssignment::<E2>::new();
    let inputs2: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      scalar_as_base::<E2>(zero2), // pass zero for testing
      zero2,
      vec![zero2],
      None,
      None,
      None,
      ri_2,
      Some(inst1),
      None,
    );
    let circuit2: NeutronAugmentedCircuit<'_, E1, TrivialCircuit<<E1 as Engine>::Base>> =
      NeutronAugmentedCircuit::new(secondary_params, Some(inputs2), &tc2, ro_consts2);
    let _ = circuit2.synthesize(&mut cs2);
    let (inst2, witness2) = cs2.r1cs_instance_and_witness(&shape2, &ck2).unwrap();
    // Make sure that it is satisfiable
    assert!(shape2.is_sat(&ck2, &inst2, &witness2).is_ok());
  }

  #[test]
  fn test_recursive_circuit_pasta() {
    // this test checks against values that must be replicated in benchmarks if changed here
    let params1 = NeutronAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = NeutronAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<VestaEngine> = PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<PallasEngine> = PoseidonConstantsCircuit::default();

    test_recursive_circuit_with::<PallasEngine, VestaEngine>(
      &params1,
      &params2,
      ro_consts1,
      ro_consts2,
      &expect!["38361"],
      &expect!["38352"],
    );
  }

  #[test]
  fn test_recursive_circuit_bn256_grumpkin() {
    let params1 = NeutronAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = NeutronAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<GrumpkinEngine> = PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<Bn256EngineKZG> = PoseidonConstantsCircuit::default();

    test_recursive_circuit_with::<Bn256EngineKZG, GrumpkinEngine>(
      &params1,
      &params2,
      ro_consts1,
      ro_consts2,
      &expect!["38641"],
      &expect!["38667"],
    );
  }

  #[test]
  fn test_recursive_circuit_secpq() {
    let params1 = NeutronAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = NeutronAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<Secq256k1Engine> = PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<Secp256k1Engine> = PoseidonConstantsCircuit::default();

    test_recursive_circuit_with::<Secp256k1Engine, Secq256k1Engine>(
      &params1,
      &params2,
      ro_consts1,
      ro_consts2,
      &expect!["39106"],
      &expect!["39372"],
    );
  }
}
*/
