//! Supernova implementation support arbitrary argumented circuits and running instances.
//! There are two Verification Circuits for each argumented circuit: The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! Each circuit takes as input 2 hashes.
//! Each circuit folds the last invocation of the other into the respective running instance, specified by `augmented_circuit_index`
//!
//! The augmented circuit F' for `SuperNova` that includes everything from Nova
//!   and additionally checks:
//!    1. Ui[] are contained in X[0] hash pre-image.
//!    2. R1CS Instance u is folded into Ui[augmented_circuit_index] correctly; just like Nova IVC.
//!    3. (optional by F logic) F circuit might check `program_counter_{i}` invoked current F circuit is legal or not.
//!    3. F circuit produce `program_counter_{i+1}` and sent to next round for optionally constraint the next F' argumented circuit.

use crate::{
  constants::NUM_HASH_BITS,
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{
      conditionally_select_alloc_relaxed_r1cs,
      conditionally_select_vec_allocated_relaxed_r1cs_instance, AllocatedR1CSInstance,
      AllocatedRelaxedR1CSInstance,
    },
    utils::{
      alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select_vec, le_bits_to_num,
    },
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    circuit_supernova::EnforcingStepCircuit, commitment::CommitmentTrait, Engine, ROCircuitTrait,
    ROConstantsCircuit,
  },
  Commitment,
};
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, SynthesisError,
};

use bellpepper::gadgets::Assignment;

use ff::Field;
use itertools::Itertools as _;
use serde::{Deserialize, Serialize};

use crate::supernova::{
  num_ro_inputs,
  utils::{get_from_vec_alloc_relaxed_r1cs, get_selector_vec_from_index},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SuperNovaAugmentedCircuitParams {
  limb_width: usize,
  n_limbs: usize,
  is_primary_circuit: bool, // A boolean indicating if this is the primary circuit
}

impl SuperNovaAugmentedCircuitParams {
  pub const fn new(limb_width: usize, n_limbs: usize, is_primary_circuit: bool) -> Self {
    Self {
      limb_width,
      n_limbs,
      is_primary_circuit,
    }
  }

  pub fn get_n_limbs(&self) -> usize {
    self.n_limbs
  }
}

#[derive(Debug)]
pub struct SuperNovaAugmentedCircuitInputs<'a, E: Engine> {
  pp_digest: E::Scalar,
  i: E::Base,
  /// Input to the circuit for the base case
  z0: &'a [E::Base],
  /// Input to the circuit for the non-base case
  zi: Option<&'a [E::Base]>,
  /// List of `RelaxedR1CSInstance`.
  /// `None` if this is the base case.
  /// Elements are `None` if the circuit at that index was not yet executed.
  U: Option<&'a [Option<RelaxedR1CSInstance<E>>]>,
  /// R1CS proof to be folded into U
  u: Option<&'a R1CSInstance<E>>,
  /// Nova folding proof for accumulating u into U[j]
  T: Option<&'a Commitment<E>>,
  /// Index of the current circuit
  program_counter: Option<E::Base>,
  /// Index j of circuit being folded into U[j]
  last_augmented_circuit_index: E::Base,
}

impl<'a, E: Engine> SuperNovaAugmentedCircuitInputs<'a, E> {
  /// Create new inputs/witness for the verification circuit
  pub fn new(
    pp_digest: E::Scalar,
    i: E::Base,
    z0: &'a [E::Base],
    zi: Option<&'a [E::Base]>,
    U: Option<&'a [Option<RelaxedR1CSInstance<E>>]>,
    u: Option<&'a R1CSInstance<E>>,
    T: Option<&'a Commitment<E>>,
    program_counter: Option<E::Base>,
    last_augmented_circuit_index: E::Base,
  ) -> Self {
    Self {
      pp_digest,
      i,
      z0,
      zi,
      U,
      u,
      T,
      program_counter,
      last_augmented_circuit_index,
    }
  }
}

/// The augmented circuit F' in `SuperNova` that includes a step circuit F
/// and the circuit for the verifier in `SuperNova`'s non-interactive folding scheme,
/// `SuperNova` NIFS will fold strictly r1cs instance u with respective relaxed r1cs instance `U[last_augmented_circuit_index]`
pub struct SuperNovaAugmentedCircuit<'a, E: Engine, SC: EnforcingStepCircuit<E::Base>> {
  params: &'a SuperNovaAugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<E>,
  inputs: Option<SuperNovaAugmentedCircuitInputs<'a, E>>,
  step_circuit: &'a SC,          // The function that is applied for each step
  num_augmented_circuits: usize, // number of overall augmented circuits
}

impl<'a, E: Engine, SC: EnforcingStepCircuit<E::Base>> SuperNovaAugmentedCircuit<'a, E, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub const fn new(
    params: &'a SuperNovaAugmentedCircuitParams,
    inputs: Option<SuperNovaAugmentedCircuitInputs<'a, E>>,
    step_circuit: &'a SC,
    ro_consts: ROConstantsCircuit<E>,
    num_augmented_circuits: usize,
  ) -> Self {
    Self {
      params,
      inputs,
      step_circuit,
      ro_consts,
      num_augmented_circuits,
    }
  }

  /// Allocate all witnesses from the augmented function's non-deterministic inputs.
  /// Optional entries are allocated as their default values.
  fn alloc_witness<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    arity: usize,
    num_augmented_circuits: usize,
  ) -> Result<
    (
      AllocatedNum<E::Base>,
      AllocatedNum<E::Base>,
      Vec<AllocatedNum<E::Base>>,
      Vec<AllocatedNum<E::Base>>,
      Vec<AllocatedRelaxedR1CSInstance<E>>,
      AllocatedR1CSInstance<E>,
      AllocatedPoint<E>,
      Option<AllocatedNum<E::Base>>,
      Vec<Boolean>,
    ),
    SynthesisError,
  > {
    let last_augmented_circuit_index =
      AllocatedNum::alloc(cs.namespace(|| "last_augmented_circuit_index"), || {
        Ok(self.inputs.get()?.last_augmented_circuit_index)
      })?;

    // Allocate the params
    let params = alloc_scalar_as_base::<E, _>(
      cs.namespace(|| "params"),
      self.inputs.as_ref().map(|inputs| inputs.pp_digest),
    )?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate program_counter only on primary circuit
    let program_counter = if self.params.is_primary_circuit {
      Some(AllocatedNum::alloc(
        cs.namespace(|| "program_counter"),
        || {
          Ok(
            self
              .inputs
              .get()?
              .program_counter
              .expect("program_counter missing"),
          )
        },
      )?)
    } else {
      None
    };

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
          Ok(self.inputs.get()?.zi.unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // Allocate the running instances
    let U = (0..num_augmented_circuits)
      .map(|i| {
        AllocatedRelaxedR1CSInstance::alloc(
          cs.namespace(|| format!("Allocate U {:?}", i)),
          self
            .inputs
            .as_ref()
            .and_then(|inputs| inputs.U.and_then(|U| U[i].as_ref())),
          self.params.limb_width,
          self.params.n_limbs,
        )
      })
      .collect::<Result<Vec<AllocatedRelaxedR1CSInstance<E>>, _>>()?;

    // Allocate the r1cs instance to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self.inputs.as_ref().and_then(|inputs| inputs.u),
    )?;

    // Allocate T
    let T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      self
        .inputs
        .as_ref()
        .and_then(|inputs| inputs.T.map(|T| T.to_coordinates())),
    )?;
    T.check_on_curve(cs.namespace(|| "check T on curve"))?;

    // Compute instance selector
    let last_augmented_circuit_selector = get_selector_vec_from_index(
      cs.namespace(|| "instance selector"),
      &last_augmented_circuit_index,
      num_augmented_circuits,
    )?;

    Ok((
      params,
      i,
      z_0,
      z_i,
      U,
      u,
      T,
      program_counter,
      last_augmented_circuit_selector,
    ))
  }

  /// Synthesizes base case and returns the new relaxed `R1CSInstance`
  fn synthesize_base_case<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    u: AllocatedR1CSInstance<E>,
    last_augmented_circuit_selector: &[Boolean],
  ) -> Result<Vec<AllocatedRelaxedR1CSInstance<E>>, SynthesisError> {
    let mut cs = cs.namespace(|| "alloc U_i default");

    // Allocate a default relaxed r1cs instance
    let default = AllocatedRelaxedR1CSInstance::default(
      cs.namespace(|| "Allocate primary U_default".to_string()),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // The primary circuit just initialize single AllocatedRelaxedR1CSInstance
    let U_default = if self.params.is_primary_circuit {
      vec![default]
    } else {
      // The secondary circuit convert the incoming R1CS instance on index which match last_augmented_circuit_index
      let incoming_r1cs = AllocatedRelaxedR1CSInstance::from_r1cs_instance(
        cs.namespace(|| "Allocate incoming_r1cs"),
        u,
        self.params.limb_width,
        self.params.n_limbs,
      )?;

      last_augmented_circuit_selector
        .iter()
        .enumerate()
        .map(|(i, equal_bit)| {
          // If index match last_augmented_circuit_index, then return incoming_r1cs,
          // otherwise return the default one
          conditionally_select_alloc_relaxed_r1cs(
            cs.namespace(|| format!("select on index namespace {:?}", i)),
            &incoming_r1cs,
            &default,
            equal_bit,
          )
        })
        .collect::<Result<Vec<AllocatedRelaxedR1CSInstance<E>>, _>>()?
    };
    Ok(U_default)
  }

  /// Synthesizes non base case and returns the new relaxed `R1CSInstance`
  /// And a boolean indicating if all checks pass
  fn synthesize_non_base_case<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    params: &AllocatedNum<E::Base>,
    i: &AllocatedNum<E::Base>,
    z_0: &[AllocatedNum<E::Base>],
    z_i: &[AllocatedNum<E::Base>],
    U: &[AllocatedRelaxedR1CSInstance<E>],
    u: &AllocatedR1CSInstance<E>,
    T: &AllocatedPoint<E>,
    arity: usize,
    last_augmented_circuit_selector: &[Boolean],
    program_counter: &Option<AllocatedNum<E::Base>>,
  ) -> Result<(Vec<AllocatedRelaxedR1CSInstance<E>>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, i, program_counter, z0, zi, U[])
    let mut ro = E::ROCircuit::new(
      self.ro_consts.clone(),
      num_ro_inputs(
        self.num_augmented_circuits,
        self.params.get_n_limbs(),
        arity,
        self.params.is_primary_circuit,
      ),
    );
    ro.absorb(params);
    ro.absorb(i);

    if self.params.is_primary_circuit {
      if let Some(program_counter) = program_counter.as_ref() {
        ro.absorb(program_counter)
      } else {
        Err(SynthesisError::AssignmentMissing)?
      }
    }

    for e in z_0 {
      ro.absorb(e);
    }
    for e in z_i {
      ro.absorb(e);
    }

    U.iter().enumerate().try_for_each(|(i, U)| {
      U.absorb_in_ro(cs.namespace(|| format!("absorb U {:?}", i)), &mut ro)
    })?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), &hash_bits)?;
    let check_pass: AllocatedBit = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X0,
      &hash,
    )?;

    // Run NIFS Verifier
    let U_to_fold = get_from_vec_alloc_relaxed_r1cs(
      cs.namespace(|| "U to fold"),
      U,
      last_augmented_circuit_selector,
    )?;
    let U_fold = U_to_fold.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      params,
      u,
      T,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // update AllocatedRelaxedR1CSInstance on index match augmented circuit index
    let U_next: Vec<AllocatedRelaxedR1CSInstance<E>> = U
      .iter()
      .zip_eq(last_augmented_circuit_selector.iter())
      .map(|(U, equal_bit)| {
        conditionally_select_alloc_relaxed_r1cs(
          cs.namespace(|| "select on index namespace"),
          &U_fold,
          U,
          equal_bit,
        )
      })
      .collect::<Result<Vec<AllocatedRelaxedR1CSInstance<E>>, _>>()?;

    Ok((U_next, check_pass))
  }

  pub fn synthesize<CS: ConstraintSystem<<E as Engine>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<(Option<AllocatedNum<E::Base>>, Vec<AllocatedNum<E::Base>>), SynthesisError> {
    let arity = self.step_circuit.arity();
    let num_augmented_circuits = if self.params.is_primary_circuit {
      // primary circuit only fold single running instance with secondary output strict r1cs instance
      1
    } else {
      // secondary circuit contains the logic to choose one of multiple augments running instance to fold
      self.num_augmented_circuits
    };

    if self.inputs.is_some() {
      // Check arity of z0
      let z0_len = self.inputs.as_ref().map_or(0, |inputs| inputs.z0.len());
      if self.step_circuit.arity() != z0_len {
        return Err(SynthesisError::IncompatibleLengthVector(format!(
          "z0_len {:?} != arity length {:?}",
          z0_len,
          self.step_circuit.arity()
        )));
      }

      // The primary curve should always fold the circuit with index 0
      let last_augmented_circuit_index = self
        .inputs
        .get()
        .map_or(E::Base::ZERO, |inputs| inputs.last_augmented_circuit_index);
      if self.params.is_primary_circuit && last_augmented_circuit_index != E::Base::ZERO {
        return Err(SynthesisError::IncompatibleLengthVector(
          "primary circuit running instance only valid on index 0".to_string(),
        ));
      }
    }

    // Allocate witnesses
    let (params, i, z_0, z_i, U, u, T, program_counter, last_augmented_circuit_selector) = self
      .alloc_witness(
        cs.namespace(|| "allocate the circuit witness"),
        arity,
        num_augmented_circuits,
      )?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // Synthesize the circuit for the non-base case and get the new running
    // instances along with a boolean indicating if all checks have passed
    // must use return `last_augmented_circuit_index_checked` since it got range checked
    let (U_next_non_base, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      &params,
      &i,
      &z_0,
      &z_i,
      &U,
      &u,
      &T,
      arity,
      &last_augmented_circuit_selector,
      &program_counter,
    )?;

    // Synthesize the circuit for the base case and get the new running instances
    let U_next_base = self.synthesize_base_case(
      cs.namespace(|| "base case"),
      u.clone(),
      &last_augmented_circuit_selector,
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

    // Compute the U_next
    let U_next = conditionally_select_vec_allocated_relaxed_r1cs_instance(
      cs.namespace(|| "U_next"),
      &U_next_base[..],
      &U_next_non_base[..],
      &Boolean::from(is_base_case.clone()),
    )?;

    // Compute i + 1
    let i_next = AllocatedNum::alloc(cs.namespace(|| "i + 1"), || {
      Ok(*i.get_value().get()? + E::Base::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc + i.get_variable() + CS::one(),
      |lc| lc + CS::one(),
      |lc| lc + i_next.get_variable(),
    );

    // Compute z_{i+1}
    let z_input = conditionally_select_vec(
      cs.namespace(|| "select input to F"),
      &z_0,
      &z_i,
      &Boolean::from(is_base_case),
    )?;

    let (program_counter_new, z_next) = self.step_circuit.enforcing_synthesize(
      &mut cs.namespace(|| "F"),
      program_counter.as_ref(),
      &z_input,
    )?;

    if z_next.len() != arity {
      return Err(SynthesisError::IncompatibleLengthVector(
        "z_next".to_string(),
      ));
    }

    // To check correct folding sequencing we are just going to make a hash.
    // The next RunningInstance folding can take the pre-image of this hash as witness and check.

    // "Finally, there is a subtle sizing issue in the above description: in each step,
    // because Ui+1 is produced as the public IO of F0 program_counter+1, it must be contained in
    // the public IO of instance ui+1. In the next iteration, because ui+1 is folded
    // into Ui+1[program_counter+1], this means that Ui+1[program_counter+1] is at least as large as Ui by the
    // properties of the folding scheme. This means that the list of running instances
    // grows in each step. To alleviate this issue, we have each F0j only produce a hash
    // of its outputs as public output. In the subsequent step, the next augmented
    // function takes as non-deterministic input a preimage to this hash." pg.16

    // https://eprint.iacr.org/2022/1758.pdf

    // Compute the new hash H(params, i+1, program_counter, z0, z_{i+1}, U_next)
    let mut ro = E::ROCircuit::new(
      self.ro_consts.clone(),
      num_ro_inputs(
        self.num_augmented_circuits,
        self.params.get_n_limbs(),
        self.step_circuit.arity(),
        self.params.is_primary_circuit,
      ),
    );
    ro.absorb(&params);
    ro.absorb(&i_next);
    // optionally absorb program counter if exist
    if program_counter.is_some() {
      ro.absorb(
        program_counter_new
          .as_ref()
          .expect("new program counter missing"),
      )
    }
    for e in &z_0 {
      ro.absorb(e);
    }
    for e in &z_next {
      ro.absorb(e);
    }
    U_next.iter().enumerate().try_for_each(|(i, U)| {
      U.absorb_in_ro(cs.namespace(|| format!("absorb U_new {:?}", i)), &mut ro)
    })?;

    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), &hash_bits)?;

    // We are cycling of curve implementation, so primary/secondary will rotate hash in IO for the others to check
    // bypass unmodified hash of other circuit as next X[0]
    // and output the computed the computed hash as next X[1]
    u.X1
      .inputize(cs.namespace(|| "bypass unmodified hash of the other circuit"))?;
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok((program_counter_new, z_next))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    bellpepper::{
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
    },
    constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
    gadgets::utils::scalar_as_base,
    provider::{
      poseidon::PoseidonConstantsCircuit,
      {Bn256EngineKZG, GrumpkinEngine}, {PallasEngine, VestaEngine},
      {Secp256k1Engine, Secq256k1Engine},
    },
    traits::{circuit_supernova::TrivialTestCircuit, snark::default_ck_hint},
  };

  // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
  fn test_supernova_recursive_circuit_with<E1, E2>(
    primary_params: &SuperNovaAugmentedCircuitParams,
    secondary_params: &SuperNovaAugmentedCircuitParams,
    ro_consts1: ROConstantsCircuit<E2>,
    ro_consts2: ROConstantsCircuit<E1>,
    num_constraints_primary: usize,
    num_constraints_secondary: usize,
    num_augmented_circuits: usize,
  ) where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let tc1 = TrivialTestCircuit::default();
    // Initialize the shape and ck for the primary
    let circuit1: SuperNovaAugmentedCircuit<'_, E2, TrivialTestCircuit<<E2 as Engine>::Base>> =
      SuperNovaAugmentedCircuit::new(
        primary_params,
        None,
        &tc1,
        ro_consts1.clone(),
        num_augmented_circuits,
      );
    let mut cs: TestShapeCS<E1> = TestShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let (shape1, ck1) = cs.r1cs_shape_and_key(&*default_ck_hint());
    assert_eq!(cs.num_constraints(), num_constraints_primary);

    let tc2 = TrivialTestCircuit::default();
    // Initialize the shape and ck for the secondary
    let circuit2: SuperNovaAugmentedCircuit<'_, E1, TrivialTestCircuit<<E1 as Engine>::Base>> =
      SuperNovaAugmentedCircuit::new(
        secondary_params,
        None,
        &tc2,
        ro_consts2.clone(),
        num_augmented_circuits,
      );
    let mut cs: TestShapeCS<E2> = TestShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let (shape2, ck2) = cs.r1cs_shape_and_key(&*default_ck_hint());
    assert_eq!(cs.num_constraints(), num_constraints_secondary);

    // Execute the base case for the primary
    let zero1 = <<E2 as Engine>::Base as Field>::ZERO;
    let mut cs1 = SatisfyingAssignment::<E1>::new();
    let vzero1 = vec![zero1];
    let inputs1: SuperNovaAugmentedCircuitInputs<'_, E2> = SuperNovaAugmentedCircuitInputs::new(
      scalar_as_base::<E1>(zero1), // pass zero for testing
      zero1,
      &vzero1,
      None,
      None,
      None,
      None,
      Some(zero1),
      zero1,
    );
    let circuit1: SuperNovaAugmentedCircuit<'_, E2, TrivialTestCircuit<<E2 as Engine>::Base>> =
      SuperNovaAugmentedCircuit::new(
        primary_params,
        Some(inputs1),
        &tc1,
        ro_consts1,
        num_augmented_circuits,
      );
    let _ = circuit1.synthesize(&mut cs1);
    let (inst1, witness1) = cs1.r1cs_instance_and_witness(&shape1, &ck1).unwrap();
    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&ck1, &inst1, &witness1).is_ok());

    // Execute the base case for the secondary
    let zero2 = <<E1 as Engine>::Base as Field>::ZERO;
    let mut cs2 = SatisfyingAssignment::<E2>::new();
    let vzero2 = vec![zero2];
    let inputs2: SuperNovaAugmentedCircuitInputs<'_, E1> = SuperNovaAugmentedCircuitInputs::new(
      scalar_as_base::<E2>(zero2), // pass zero for testing
      zero2,
      &vzero2,
      None,
      None,
      Some(&inst1),
      None,
      Some(zero2),
      zero2,
    );
    let circuit2: SuperNovaAugmentedCircuit<'_, E1, TrivialTestCircuit<<E1 as Engine>::Base>> =
      SuperNovaAugmentedCircuit::new(
        secondary_params,
        Some(inputs2),
        &tc2,
        ro_consts2,
        num_augmented_circuits,
      );
    let _ = circuit2.synthesize(&mut cs2);
    let (inst2, witness2) = cs2.r1cs_instance_and_witness(&shape2, &ck2).unwrap();
    // Make sure that it is satisfiable
    assert!(shape2.is_sat(&ck2, &inst2, &witness2).is_ok());
  }

  #[test]
  fn test_supernova_recursive_circuit_pasta() {
    // this test checks against values that must be replicated in benchmarks if changed here
    let params1 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<VestaEngine> = PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<PallasEngine> = PoseidonConstantsCircuit::default();

    test_supernova_recursive_circuit_with::<PallasEngine, VestaEngine>(
      &params1, &params2, ro_consts1, ro_consts2, 9836, 10384, 1,
    );
    // TODO: extend to num_augmented_circuits >= 2
  }

  #[test]
  fn test_supernova_recursive_circuit_grumpkin() {
    let params1 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<GrumpkinEngine> = PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<Bn256EngineKZG> = PoseidonConstantsCircuit::default();

    test_supernova_recursive_circuit_with::<Bn256EngineKZG, GrumpkinEngine>(
      &params1, &params2, ro_consts1, ro_consts2, 10004, 10573, 1,
    );
    // TODO: extend to num_augmented_circuits >= 2
  }

  #[test]
  fn test_supernova_recursive_circuit_secp() {
    let params1 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = SuperNovaAugmentedCircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<Secq256k1Engine> = PoseidonConstantsCircuit::default();
    let ro_consts2: ROConstantsCircuit<Secp256k1Engine> = PoseidonConstantsCircuit::default();

    test_supernova_recursive_circuit_with::<Secp256k1Engine, Secq256k1Engine>(
      &params1, &params2, ro_consts1, ro_consts2, 10283, 10996, 1,
    );
    // TODO: extend to num_augmented_circuits >= 2
  }
}
