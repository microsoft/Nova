//! There are two augmented circuits: the primary and the secondary.
//! Each of them is over a curve in a 2-cycle of elliptic curves.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is H(params = H(shape, ck), i, z0, zi, U).
//! Each circuit folds the last invocation of the other into the running instance

use crate::{
  constants::NUM_HASH_BITS,
  frontend::{
    num::AllocatedNum, AllocatedBit, Assignment, Boolean, ConstraintSystem, SynthesisError,
  },
  gadgets::{
    ecc::AllocatedNonnativePoint,
    utils::{alloc_num_equals, alloc_zero, conditionally_select_vec, le_bits_to_num},
  },
  neutron::{nifs::NIFS, relation::FoldedInstance},
  r1cs::R1CSInstance,
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, Engine, RO2ConstantsCircuit, ROCircuitTrait,
  },
  Commitment,
};
use ff::Field;
use serde::{Deserialize, Serialize};

pub mod nifs;
pub mod r1cs;
pub mod relation;
pub mod univariate;

use nifs::AllocatedNIFS;
use r1cs::AllocatedNonnativeR1CSInstance;
use relation::AllocatedFoldedInstance;

/// A type that holds the non-deterministic inputs for the augmented circuit
#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NeutronAugmentedCircuitInputs<E: Engine> {
  pp_digest: E::Scalar,
  i: E::Scalar,
  z0: Vec<E::Scalar>,
  zi: Option<Vec<E::Scalar>>,
  U: Option<FoldedInstance<E>>,
  ri: Option<E::Scalar>,
  r_next: E::Scalar,
  u: Option<R1CSInstance<E>>,
  nifs: Option<NIFS<E>>,
  comm_W_fold: Option<Commitment<E>>,
  comm_E_fold: Option<Commitment<E>>,
}

impl<E: Engine> NeutronAugmentedCircuitInputs<E> {
  /// Create new inputs/witness for the verification circuit
  pub fn new(
    pp_digest: E::Scalar,
    i: E::Scalar,
    z0: Vec<E::Scalar>,
    zi: Option<Vec<E::Scalar>>,
    U: Option<FoldedInstance<E>>,
    ri: Option<E::Scalar>,
    r_next: E::Scalar,
    u: Option<R1CSInstance<E>>,
    nifs: Option<NIFS<E>>,
    comm_W_fold: Option<Commitment<E>>,
    comm_E_fold: Option<Commitment<E>>,
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
      comm_W_fold,
      comm_E_fold,
    }
  }
}

/// The augmented circuit F' in Neutron that includes a step circuit F
/// and the circuit for the verifier in Neutron's non-interactive folding scheme
pub struct NeutronAugmentedCircuit<'a, E: Engine, SC: StepCircuit<E::Scalar>> {
  ro_consts: RO2ConstantsCircuit<E>,
  inputs: Option<NeutronAugmentedCircuitInputs<E>>,
  step_circuit: &'a SC, // The function that is applied for each step
}

impl<'a, E: Engine, SC: StepCircuit<E::Scalar>> NeutronAugmentedCircuit<'a, E, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub const fn new(
    inputs: Option<NeutronAugmentedCircuitInputs<E>>,
    step_circuit: &'a SC,
    ro_consts: RO2ConstantsCircuit<E>,
  ) -> Self {
    Self {
      inputs,
      step_circuit,
      ro_consts,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<E::Scalar>,
      AllocatedNum<E::Scalar>,
      Vec<AllocatedNum<E::Scalar>>,
      Vec<AllocatedNum<E::Scalar>>,
      AllocatedFoldedInstance<E>,
      AllocatedNum<E::Scalar>,
      AllocatedNum<E::Scalar>,
      AllocatedNonnativeR1CSInstance<E>,
      AllocatedNIFS<E>,
      AllocatedNonnativePoint<E>,
      AllocatedNonnativePoint<E>,
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
      .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![E::Scalar::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;

    // Allocate the running instance
    let U: AllocatedFoldedInstance<E> = AllocatedFoldedInstance::alloc(
      cs.namespace(|| "Allocate U"),
      self.inputs.as_ref().and_then(|inputs| inputs.U.as_ref()),
    )?;

    // Allocate ri
    let r_i = AllocatedNum::alloc(cs.namespace(|| "ri"), || {
      Ok(self.inputs.get()?.ri.unwrap_or(E::Scalar::ZERO))
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

    Ok((
      pp_digest,
      i,
      z_0,
      z_i,
      U,
      r_i,
      r_next,
      u,
      nifs,
      comm_W_fold,
      comm_E_fold,
    ))
  }

  fn synthesize_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedFoldedInstance<E>, SynthesisError> {
    // In the base case, we simply return the default running instance
    AllocatedFoldedInstance::default(cs.namespace(|| "Allocate U_default"))
  }

  /// Synthesizes non base case and returns the new relaxed `FoldedInstance`
  /// And a boolean indicating if all checks pass
  fn synthesize_non_base_case<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>,
    i: &AllocatedNum<E::Scalar>,
    z_0: &[AllocatedNum<E::Scalar>],
    z_i: &[AllocatedNum<E::Scalar>],
    U: &AllocatedFoldedInstance<E>,
    r_i: &AllocatedNum<E::Scalar>,
    u: &AllocatedNonnativeR1CSInstance<E>,
    nifs: &AllocatedNIFS<E>,
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
  ) -> Result<(AllocatedFoldedInstance<E>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = E::RO2Circuit::new(self.ro_consts.clone());
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

    Ok((U_fold, check_pass))
  }
}

impl<E: Engine, SC: StepCircuit<E::Scalar>> NeutronAugmentedCircuit<'_, E, SC> {
  /// synthesize circuit giving constraint system
  pub fn synthesize<CS: ConstraintSystem<E::Scalar>>(
    self,
    cs: &mut CS,
  ) -> Result<Vec<AllocatedNum<E::Scalar>>, SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (pp_digest, i, z_0, z_i, U, r_i, r_next, u, nifs, comm_W_fold, comm_E_fold) =
      self.alloc_witness(cs.namespace(|| "allocate the circuit witness"), arity)?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // synthesize base case
    let Unew_base = self.synthesize_base_case(cs.namespace(|| "synthesize base case"))?;

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
      &comm_W_fold,
      &comm_E_fold,
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

    // we pick between the base case output and the non-base case output
    let Unew = Unew_base.conditionally_select(
      cs.namespace(|| "compute U_new"),
      &Unew_non_base,
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_new = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E::Scalar::ONE)
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
    let mut ro = E::RO2Circuit::new(self.ro_consts);
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
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let ro_consts = RO2ConstantsCircuit::<E1>::default();
    let tc = TrivialCircuit::default();

    let circuit: NeutronAugmentedCircuit<'_, E1, TrivialCircuit<E1::Scalar>> =
      NeutronAugmentedCircuit::new(None, &tc, ro_consts.clone());
    let mut cs: TestShapeCS<E1> = TestShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());
    num_constraints.assert_eq(cs.num_constraints().to_string().as_str());

    // Execute the base case for the primary
    let zero = <E1::Scalar as Field>::ZERO;
    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      zero, // pass zero for testing
      zero,
      vec![zero],
      None,
      None,
      None,
      zero,
      None,
      None,
      None,
      None,
    );
    let circuit: NeutronAugmentedCircuit<'_, E1, TrivialCircuit<E1::Scalar>> =
      NeutronAugmentedCircuit::new(Some(inputs), &tc, ro_consts);
    let _ = circuit.synthesize(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_neutron_recursive_circuit_pasta() {
    test_recursive_circuit_with::<PallasEngine, VestaEngine>(&expect!["7048"]);
    test_recursive_circuit_with::<Bn256EngineKZG, GrumpkinEngine>(&expect!["7328"]);
    test_recursive_circuit_with::<Secp256k1Engine, Secq256k1Engine>(&expect!["7793"]);
  }
}
