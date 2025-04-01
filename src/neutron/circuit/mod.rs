//! This module implements the augmented circuit for Neutron.
//! This circuit includes an invocation of the step circuit F as well as a folding verifier circuit
//! for the non-interactive folding scheme.
//! The folding verifier cirucit uses cyclefold technique to fold claims about scalar multiplication operations
use crate::{
  constants::NUM_HASH_BITS,
  frontend::{
    num::AllocatedNum, AllocatedBit, Assignment, Boolean, ConstraintSystem, SynthesisError,
  },
  gadgets::{
    ecc::AllocatedNonnativePoint,
    utils::{
      alloc_bignat_constant, alloc_num_equals, alloc_zero, conditionally_select_vec, le_bits_to_num,
    },
  },
  neutron::{nifs::NIFS, relation::FoldedInstance},
  nova::nifs::NIFS as NovaNIFS,
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, Engine, Group, RO2ConstantsCircuit,
    ROCircuitTrait,
  },
  Commitment,
};
use ff::Field;
use num_bigint::BigInt;
use serde::{Deserialize, Serialize};

mod cyclefold;
mod nifs;
mod nifs_ec;
mod r1cs;
mod r1cs_ec;
mod relation;
mod univariate;

pub(crate) mod ec;

use nifs::AllocatedNIFS;
use nifs_ec::AllocatedNIFSEC;
use r1cs::AllocatedNonnativeR1CSInstance;
use r1cs_ec::{AllocatedECInstance, AllocatedRelaxedECInstance};
use relation::AllocatedFoldedInstance;

/// A type that holds the non-deterministic inputs for the augmented circuit
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronAugmentedCircuitInputs<E1: Engine, E2: Engine> {
  pp_digest: E1::Scalar,
  i: E1::Scalar,
  z0: Vec<E1::Scalar>,
  zi: Option<Vec<E1::Scalar>>,
  U: Option<FoldedInstance<E1>>,
  U_EC: Option<RelaxedR1CSInstance<E2>>,
  ri: Option<E1::Scalar>,
  r_next: E1::Scalar,
  u: Option<R1CSInstance<E1>>,
  nifs: Option<NIFS<E1>>,
  comm_W_fold: Option<Commitment<E1>>,
  comm_E_fold: Option<Commitment<E1>>,
  u_EC_W: Option<R1CSInstance<E2>>,
  nifs_EC: Option<NovaNIFS<E2>>,
}

impl<E1: Engine, E2: Engine> NeutronAugmentedCircuitInputs<E1, E2> {
  /// Create new inputs/witness for the verification circuit
  pub fn new(
    pp_digest: E1::Scalar,
    i: E1::Scalar,
    z0: Vec<E1::Scalar>,
    zi: Option<Vec<E1::Scalar>>,
    U: Option<FoldedInstance<E1>>,
    U_EC: Option<RelaxedR1CSInstance<E2>>,
    ri: Option<E1::Scalar>,
    r_next: E1::Scalar,
    u: Option<R1CSInstance<E1>>,
    nifs: Option<NIFS<E1>>,
    comm_W_fold: Option<Commitment<E1>>,
    comm_E_fold: Option<Commitment<E1>>,
    u_EC_W: Option<R1CSInstance<E2>>,
    nifs_EC: Option<NovaNIFS<E2>>,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z0,
      zi,
      U,
      U_EC,
      ri,
      r_next,
      u,
      nifs,
      comm_W_fold,
      comm_E_fold,
      u_EC_W,
      nifs_EC,
    }
  }
}

/// The augmented circuit F' in Neutron that includes a step circuit F
/// and the circuit for the verifier in Neutron's non-interactive folding scheme
pub struct NeutronAugmentedCircuit<'a, E1: Engine, E2: Engine, SC: StepCircuit<E1::Scalar>> {
  ro_consts: RO2ConstantsCircuit<E1>,
  inputs: Option<NeutronAugmentedCircuitInputs<E1, E2>>,
  step_circuit: &'a SC, // The function that is applied for each step
}

impl<'a, E1: Engine, E2: Engine, SC: StepCircuit<E1::Scalar>>
  NeutronAugmentedCircuit<'a, E1, E2, SC>
where
  E1: Engine<
    Base = <E2 as Engine>::Scalar,
    RO = <E2 as Engine>::RO2,
    ROCircuit = <E2 as Engine>::RO2Circuit,
  >,
  E2: Engine<
    Base = <E1 as Engine>::Scalar,
    RO = <E1 as Engine>::RO2,
    ROCircuit = <E1 as Engine>::RO2Circuit,
  >,
{
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub const fn new(
    inputs: Option<NeutronAugmentedCircuitInputs<E1, E2>>,
    step_circuit: &'a SC,
    ro_consts: RO2ConstantsCircuit<E1>,
  ) -> Self {
    Self {
      inputs,
      step_circuit,
      ro_consts,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E1::Scalar>,
      AllocatedNum<E1::Scalar>,
      Vec<AllocatedNum<E1::Scalar>>,
      Vec<AllocatedNum<E1::Scalar>>,
      AllocatedFoldedInstance<E1>,
      AllocatedRelaxedECInstance<E2>,
      AllocatedNum<E1::Scalar>,
      AllocatedNum<E1::Scalar>,
      AllocatedNonnativeR1CSInstance<E1>,
      AllocatedNIFS<E1>,
      AllocatedNonnativePoint<E1>,
      AllocatedNonnativePoint<E1>,
      AllocatedECInstance<E2>,
      AllocatedNIFSEC<E2>,
    ),
    SynthesisError,
  > {
    // Allocate the pp_digest
    let pp_digest = AllocatedNum::alloc(cs.namespace(|| "pp_digest"), || {
      Ok(self.inputs.get()?.pp_digest)
    })?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate z0
    let z_0 = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
          Ok(self.inputs.get()?.z0[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E1::Scalar>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![E1::Scalar::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E1::Scalar>>, _>>()?;

    // Allocate the running instance
    let U: AllocatedFoldedInstance<E1> = AllocatedFoldedInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
    )?;
    let U_EC = AllocatedRelaxedECInstance::alloc(
      cs.namespace(|| "allocate U_EC"),
      self.inputs.as_ref().and_then(|inputs| inputs.U_EC.as_ref()),
    )?;

    // Allocate ri
    let r_i = AllocatedNum::alloc(cs.namespace(|| "ri"), || {
      Ok(self.inputs.get()?.ri.unwrap_or(E1::Scalar::ZERO))
    })?;

    // Allocate r_i+1
    let r_next = AllocatedNum::alloc(cs.namespace(|| "r_i+1"), || Ok(self.inputs.get()?.r_next))?;

    // Allocate the instance to be folded in
    let u = AllocatedNonnativeR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.as_ref().and_then(|inputs| inputs.u.as_ref()),
    )?;

    // Allocate nifs
    let nifs = AllocatedNIFS::alloc(
      cs.namespace(|| "allocate nifs"),
      self.inputs.as_ref().and_then(|inputs| inputs.nifs.as_ref()),
      5, // TODO: take this as input
    )?;

    // Allocated comm_W_fold
    let comm_W_fold = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_W"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.comm_W_fold.as_ref().map(|c| c.to_coordinates())),
    )?;

    // Allocate comm_E_fold
    let comm_E_fold = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_E_fold"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.comm_E_fold.as_ref().map(|c| c.to_coordinates())),
    )?;

    // Allocate u_EC that contains checks the provided comm_W_fold and comm_E_fold
    let u_EC_W = AllocatedECInstance::alloc(
      cs.namespace(|| "allocate u_EC_W"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.u_EC_W.as_ref()),
      (&U.comm_W.x, &U.comm_W.y, &U.comm_W.is_infinity),
      (&u.comm_W.x, &u.comm_W.y, &u.comm_W.is_infinity),
    )?;

    // Allocate nifs_EC
    let nifs_EC = AllocatedNIFSEC::alloc(
      cs.namespace(|| "allocate nifs_EC"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.nifs_EC.as_ref()),
    )?;

    Ok((
      pp_digest,
      i,
      z_0,
      z_i,
      U,
      U_EC,
      r_i,
      r_next,
      u,
      nifs,
      comm_W_fold,
      comm_E_fold,
      u_EC_W,
      nifs_EC,
    ))
  }

  fn synthesize_base_case<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<(AllocatedFoldedInstance<E1>, AllocatedRelaxedECInstance<E2>), SynthesisError> {
    let zero = alloc_bignat_constant(cs.namespace(|| "zero"), &BigInt::from(0))?;

    // In the base case, we simply return the default running instance
    Ok((
      AllocatedFoldedInstance::default(cs.namespace(|| "Allocate U_default"))?,
      AllocatedRelaxedECInstance::default(cs.namespace(|| "Allocate U_EC_default"), &zero)?,
    ))
  }

  /// Synthesizes non base case and returns the new relaxed `FoldedInstance`
  /// And a boolean indicating if all checks pass
  fn synthesize_non_base_case<CS: ConstraintSystem<E1::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E1::Scalar>,
    i: &AllocatedNum<E1::Scalar>,
    z_0: &[AllocatedNum<E1::Scalar>],
    z_i: &[AllocatedNum<E1::Scalar>],
    U: &AllocatedFoldedInstance<E1>,
    U_EC: &AllocatedRelaxedECInstance<E2>,
    r_i: &AllocatedNum<E1::Scalar>,
    u: &AllocatedNonnativeR1CSInstance<E1>,
    nifs: &AllocatedNIFS<E1>,
    comm_W_fold: &AllocatedNonnativePoint<E1>,
    comm_E_fold: &AllocatedNonnativePoint<E1>,
    u_EC_W: &AllocatedECInstance<E2>,
    nifs_EC: &AllocatedNIFSEC<E2>,
  ) -> Result<
    (
      AllocatedFoldedInstance<E1>,
      AllocatedRelaxedECInstance<E2>,
      AllocatedBit,
    ),
    SynthesisError,
  > {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = E1::RO2Circuit::new(self.ro_consts.clone());
    ro.absorb(pp_digest);
    ro.absorb(i);
    for e in z_0 {
      ro.absorb(e);
    }
    for e in z_i {
      ro.absorb(e);
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;
    //U_EC.absorb_in_ro(cs.namespace(|| "absorb U_EC"), &mut ro)?; // TODO: bring this back
    ro.absorb(r_i);

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;
    let check_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X,
      &hash,
    )?;

    // Run NIFS Verifier
    let U_fold = nifs.verify(
      cs.namespace(|| "compute fold of U and u"),
      pp_digest,
      U,
      u,
      comm_W_fold,
      comm_E_fold,
      self.ro_consts.clone(),
    )?;

    // Run NIFS_EC Verifier
    // we first formulate the "fresh" instance to be folded into u_EC
    // allocate modulus as a constant
    let m_bn = alloc_bignat_constant(cs.namespace(|| "alloc modulus"), &E1::GE::group_params().3)?;
    let (U_EC_fold, _r) = nifs_EC.verify(
      cs.namespace(|| "compute fold of U_EC and u_EC"),
      pp_digest,
      U_EC,
      u_EC_W,
      self.ro_consts.clone(),
      &m_bn,
    )?;

    Ok((U_fold, U_EC_fold, check_pass))
  }
}

impl<E1: Engine, E2: Engine, SC: StepCircuit<E1::Scalar>> NeutronAugmentedCircuit<'_, E1, E2, SC>
where
  E1: Engine<
    Base = <E2 as Engine>::Scalar,
    RO = <E2 as Engine>::RO2,
    ROCircuit = <E2 as Engine>::RO2Circuit,
  >,
  E2: Engine<
    Base = <E1 as Engine>::Scalar,
    RO = <E1 as Engine>::RO2,
    ROCircuit = <E1 as Engine>::RO2Circuit,
  >,
{
  /// synthesize circuit giving constraint system
  pub fn synthesize<CS: ConstraintSystem<E1::Scalar>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E1::Scalar>>, SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (
      pp_digest,
      i,
      z_0,
      z_i,
      U,
      U_EC,
      r_i,
      r_next,
      u,
      nifs,
      comm_W_fold,
      comm_E_fold,
      u_EC_W,
      nifs_EC,
    ) = self.alloc_witness(cs.namespace(|| "allocate the circuit witness"), arity)?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // synthesize base case
    let (Unew_base, Unew_base_EC) =
      self.synthesize_base_case(cs.namespace(|| "synthesize base case"))?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base, Unew_non_base_EC, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &pp_digest,
      &i,
      &z_0,
      &z_i,
      &U,
      &U_EC,
      &r_i,
      &u,
      &nifs,
      &comm_W_fold,
      &comm_E_fold,
      &u_EC_W,
      &nifs_EC,
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

    let is_base_case_boolean = Boolean::from(is_base_case);

    // we pick between the base case output and the non-base case output
    let Unew = Unew_base.conditionally_select(
      cs.namespace(|| "compute U_new"),
      &Unew_non_base,
      &is_base_case_boolean,
    )?;

    let _Unew_EC = Unew_base_EC.conditionally_select(
      cs.namespace(|| "compute U_new_EC"),
      &Unew_non_base_EC,
      &is_base_case_boolean,
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E1::Scalar::ONE)
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
      &is_base_case_boolean,
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
    let mut ro = E1::RO2Circuit::new(self.ro_consts);
    ro.absorb(&pp_digest);
    ro.absorb(&i_new);
    for e in &z_0 {
      ro.absorb(e);
    }
    for e in &z_next {
      ro.absorb(e);
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    //Unew_EC.absorb_in_ro(cs.namespace(|| "absorb U_new_EC"), &mut ro)?;
    ro.absorb(&r_next);
    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), &hash_bits)?;

    // Outputs the computed hash
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(z_next)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
    },
    provider::{
      Bn256EngineKZG, GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::{circuit::TrivialCircuit, snark::default_ck_hint, RO2ConstantsCircuit},
  };
  use expect_test::{expect, Expect};

  // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
  fn test_recursive_circuit_with<E1, E2>(num_constraints: &Expect)
  where
    E1: Engine<
      Base = <E2 as Engine>::Scalar,
      RO = <E2 as Engine>::RO2,
      ROCircuit = <E2 as Engine>::RO2Circuit,
    >,
    E2: Engine<
      Base = <E1 as Engine>::Scalar,
      RO = <E1 as Engine>::RO2,
      ROCircuit = <E1 as Engine>::RO2Circuit,
    >,
  {
    let ro_consts = RO2ConstantsCircuit::<E1>::default();
    let tc = TrivialCircuit::default();

    let circuit: NeutronAugmentedCircuit<'_, E1, E2, TrivialCircuit<E1::Scalar>> =
      NeutronAugmentedCircuit::new(None, &tc, ro_consts.clone());
    let mut cs: TestShapeCS<E1> = TestShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());
    num_constraints.assert_eq(cs.num_constraints().to_string().as_str());

    // Execute the base case for the primary
    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1, E2> = NeutronAugmentedCircuitInputs::new(
      E1::Scalar::ZERO, // pass zero for testing
      E1::Scalar::ZERO,
      [E1::Scalar::ZERO].to_vec(),
      None,
      None,
      None,
      None,
      E1::Scalar::ZERO,
      None,
      None,
      None,
      None,
      None,
      None,
    );
    let circuit: NeutronAugmentedCircuit<'_, E1, E2, TrivialCircuit<E1::Scalar>> =
      NeutronAugmentedCircuit::new(Some(inputs), &tc, ro_consts);
    let _ = circuit.synthesize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_neutron_recursive_circuit_pasta() {
    test_recursive_circuit_with::<PallasEngine, VestaEngine>(&expect!["22709"]);
    test_recursive_circuit_with::<Bn256EngineKZG, GrumpkinEngine>(&expect!["23045"]);
    test_recursive_circuit_with::<Secp256k1Engine, Secq256k1Engine>(&expect!["23603"]);
  }
}
