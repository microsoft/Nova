//! Supernova implemetation support arbitrary argumented circuits and running instances.
//! There are two Verification Circuits for each argumented circuit: The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! Each circuit takes as input 2 hashes.
//! Each circuit folds the last invocation of the other into the respective running instance, specified by augmented_circuit_index
//!
//! The augmented circuit F' for SuperNova that includes everything from Nova
//!   and additionally checks:
//!    1. Ui[] are contained in X[0] hash pre-image.
//!    2. R1CS Instance u is folded into Ui[augmented_circuit_index] correctly; just like Nova IVC.
//!    3. (optional by F logic) F circuit might check program_counter_{i} invoked current F circuit is legal or not.
//!    3. F circuit produce program_counter_{i+1} and sent to next round for optionally constraint the next F' argumented circuit.

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
      alloc_const, alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select_vec,
      le_bits_to_num, scalar_as_base,
    },
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    circuit_supernova::StepCircuit, commitment::CommitmentTrait, Group, ROCircuitTrait,
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
use serde::{Deserialize, Serialize};

use super::utils::get_from_vec_alloc_relaxed_r1cs;

#[derive(Debug, Clone, Serialize, Deserialize)]
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
pub struct SuperNovaAugmentedCircuitInputs<'a, G: Group> {
  pp_digest: G::Scalar,
  i: G::Base,
  z0: &'a [G::Base],
  zi: Option<&'a [G::Base]>,
  U: Option<&'a [Option<RelaxedR1CSInstance<G>>]>,
  u: Option<&'a R1CSInstance<G>>,
  T: Option<&'a Commitment<G>>,
  program_counter: G::Base,
  last_augmented_circuit_index: G::Base,
}

impl<'a, G: Group> SuperNovaAugmentedCircuitInputs<'a, G> {
  /// Create new inputs/witness for the verification circuit
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    pp_digest: G::Scalar,
    i: G::Base,
    z0: &'a [G::Base],
    zi: Option<&'a [G::Base]>,
    U: Option<&'a [Option<RelaxedR1CSInstance<G>>]>,
    u: Option<&'a R1CSInstance<G>>,
    T: Option<&'a Commitment<G>>,
    program_counter: G::Base,
    last_augmented_circuit_index: G::Base,
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

/// The augmented circuit F' in SuperNova that includes a step circuit F
/// and the circuit for the verifier in SuperNova's non-interactive folding scheme,
/// SuperNova NIFS will fold strictly r1cs instance u with respective relaxed r1cs instance U[last_augmented_circuit_index]
pub struct SuperNovaAugmentedCircuit<'a, G: Group, SC: StepCircuit<G::Base>> {
  params: &'a SuperNovaAugmentedCircuitParams,
  ro_consts: ROConstantsCircuit<G>,
  inputs: Option<SuperNovaAugmentedCircuitInputs<'a, G>>,
  step_circuit: &'a SC,          // The function that is applied for each step
  num_augmented_circuits: usize, // number of overall augmented circuits
}

impl<'a, G: Group, SC: StepCircuit<G::Base>> SuperNovaAugmentedCircuit<'a, G, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub const fn new(
    params: &'a SuperNovaAugmentedCircuitParams,
    inputs: Option<SuperNovaAugmentedCircuitInputs<'a, G>>,
    step_circuit: &'a SC,
    ro_consts: ROConstantsCircuit<G>,
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

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    arity: usize,
    num_augmented_circuits: usize,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedRelaxedR1CSInstance<G>>,
      AllocatedR1CSInstance<G>,
      AllocatedPoint<G>,
      Option<AllocatedNum<G::Base>>,
      AllocatedNum<G::Base>,
    ),
    SynthesisError,
  > {
    let last_augmented_circuit_index =
      AllocatedNum::alloc(cs.namespace(|| "last_augmented_circuit_index"), || {
        Ok(self.inputs.get()?.last_augmented_circuit_index)
      })?;

    // Allocate the params
    let params = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "params"),
      self
        .inputs
        .get()
        .map_or(None, |inputs| Some(inputs.pp_digest)),
    )?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate program_counter only on primary circuit
    let program_counter = if self.params.is_primary_circuit {
      Some(AllocatedNum::alloc(
        cs.namespace(|| "program_counter"),
        || Ok(self.inputs.get()?.program_counter),
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
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![G::Base::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.inputs.get()?.zi.unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // Allocate the running instances
    let U = (0..num_augmented_circuits)
      .map(|i| {
        AllocatedRelaxedR1CSInstance::alloc(
          cs.namespace(|| format!("Allocate U {:?}", i)),
          self
            .inputs
            .get()
            .map_or(None, |inputs| inputs.U.and_then(|U| U[i].as_ref())),
          self.params.limb_width,
          self.params.n_limbs,
        )
      })
      .collect::<Result<Vec<AllocatedRelaxedR1CSInstance<G>>, _>>()?;

    // Allocate the r1cs instance to be folded in
    let u = AllocatedR1CSInstance::alloc(
      cs.namespace(|| "allocate instance u to fold"),
      self
        .inputs
        .get()
        .map_or(None, |inputs| inputs.u.get().map_or(None, |u| Some(u))),
    )?;

    // Allocate T
    let T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate T"),
      self.inputs.get().map_or(None, |inputs| {
        inputs.T.get().map_or(None, |T| Some(T.to_coordinates()))
      }),
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
      last_augmented_circuit_index,
    ))
  }

  /// Synthesizes base case and returns the new relaxed R1CSInstance
  fn synthesize_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    u: AllocatedR1CSInstance<G>,
    last_augmented_circuit_index_checked: &AllocatedNum<G::Base>,
    num_augmented_circuits: usize,
  ) -> Result<Vec<AllocatedRelaxedR1CSInstance<G>>, SynthesisError> {
    let mut cs = cs.namespace(|| "alloc U_i default");

    // The primary circuit just initialize single AllocatedRelaxedR1CSInstance
    let U_default = if self.params.is_primary_circuit {
      vec![AllocatedRelaxedR1CSInstance::default(
        cs.namespace(|| "Allocate primary U_default".to_string()),
        self.params.limb_width,
        self.params.n_limbs,
      )?]
    } else {
      // The secondary circuit convert the incoming R1CS instance on index which match last_augmented_circuit_index
      let incoming_r1cs = AllocatedRelaxedR1CSInstance::from_r1cs_instance(
        cs.namespace(|| "Allocate incoming_r1cs"),
        u,
        self.params.limb_width,
        self.params.n_limbs,
      )?;
      (0..num_augmented_circuits)
        .map(|i| {
          let i_alloc = alloc_const(
            cs.namespace(|| format!("i allocated on {:?}", i)),
            scalar_as_base::<G>(G::Scalar::from(i as u64)),
          )?;
          let equal_bit = Boolean::from(alloc_num_equals(
            cs.namespace(|| format!("check equal bit {:?}", i)),
            &i_alloc,
            last_augmented_circuit_index_checked,
          )?);
          let default = &AllocatedRelaxedR1CSInstance::default(
            cs.namespace(|| format!("Allocate U_default {:?}", i)),
            self.params.limb_width,
            self.params.n_limbs,
          )?;
          conditionally_select_alloc_relaxed_r1cs(
            cs.namespace(|| format!("select on index namespace {:?}", i)),
            &incoming_r1cs,
            default,
            &equal_bit,
          )
        })
        .collect::<Result<Vec<AllocatedRelaxedR1CSInstance<G>>, _>>()?
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
    z_0: Vec<AllocatedNum<G::Base>>,
    z_i: Vec<AllocatedNum<G::Base>>,
    U: &[AllocatedRelaxedR1CSInstance<G>],
    u: AllocatedR1CSInstance<G>,
    T: AllocatedPoint<G>,
    arity: usize,
    last_augmented_circuit_index: &AllocatedNum<G::Base>,
    program_counter: Option<AllocatedNum<G::Base>>,
    num_augmented_circuits: usize,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      Vec<AllocatedRelaxedR1CSInstance<G>>,
      AllocatedBit,
    ),
    SynthesisError,
  > {
    // Check that u.x[0] = Hash(params, i, program_counter, U[], z0, zi)
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      2 // params_next, i_new
      + program_counter.as_ref().map_or(0, |_| 1) // optional program counter
        + 2 * arity // zo, z1
        + num_augmented_circuits * (7 + 2 * self.params.n_limbs), // #num_augmented_circuits * (7 + [X0, X1]*#num_limb)
    );
    ro.absorb(&params);
    ro.absorb(&i);
    if let Some(program_counter) = program_counter.as_ref() {
      ro.absorb(program_counter)
    }

    for e in &z_0 {
      ro.absorb(e);
    }
    for e in &z_i {
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
    let (last_augmented_circuit_index_checked, U_to_fold) = get_from_vec_alloc_relaxed_r1cs(
      cs.namespace(|| "U to fold"),
      U,
      last_augmented_circuit_index,
    )?;
    let U_fold = U_to_fold.fold_with_r1cs(
      cs.namespace(|| "compute fold of U and u"),
      &params,
      &u,
      &T,
      self.ro_consts.clone(),
      self.params.limb_width,
      self.params.n_limbs,
    )?;

    // update AllocatedRelaxedR1CSInstance on index match augmented circuit index
    let U_next: Vec<AllocatedRelaxedR1CSInstance<G>> = U
      .iter()
      .enumerate()
      .map(|(i, U)| {
        let mut cs = cs.namespace(|| format!("U_i+1 non_base conditional selection {:?}", i));
        let i_alloc = alloc_const(
          cs.namespace(|| "i allocated"),
          scalar_as_base::<G>(G::Scalar::from(i as u64)),
        )?;
        let equal_bit = Boolean::from(alloc_num_equals(
          cs.namespace(|| "check equal bit"),
          &i_alloc,
          &last_augmented_circuit_index_checked,
        )?);
        conditionally_select_alloc_relaxed_r1cs(
          cs.namespace(|| "select on index namespace"),
          &U_fold,
          U,
          &equal_bit,
        )
      })
      .collect::<Result<Vec<AllocatedRelaxedR1CSInstance<G>>, _>>()?;

    Ok((last_augmented_circuit_index_checked, U_next, check_pass))
  }
}

impl<'a, G: Group, SC: StepCircuit<G::Base>> SuperNovaAugmentedCircuit<'a, G, SC> {
  pub fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<(AllocatedNum<G::Base>, Vec<AllocatedNum<G::Base>>), SynthesisError> {
    // NOTE `last_augmented_circuit_index` is aux without any constraint.
    // Reason is prover can only produce valid running instance by folding u into proper U_i[last_augmented_circuit_index]
    // However, there is crucial pre-asumption: `last_augmented_circuit_index` must within range [0, num_augmented_circuits)
    // otherwise there will be a soundness error, such that maliculous prover can choose out of range last_augmented_circuit_index.
    // The soundness error depends on how we process out-of-range condition.
    //
    // there are 2 possible solution
    // 1. range check `last_augmented_circuit_index`
    // 2. if last_augmented_circuit_index out of range, then by default select index 0
    //
    // For current version we choose 2, due to its simplicify and fit well in last_augmented_circuit_index use case.
    // Recap, the only way to pass running instance check is folding u into respective U_i[last_augmented_circuit_index]
    // So, a circuit implementing to set out-of-range last_augmented_circuit_index to index 0 is fine.
    // The illegal running instances will be propogate to later phase and finally captured with "high" probability on the basis of Nova IVC security.
    //
    // Although above "informal" analysis implies there is no `malleability` on statement (malleability refer `NARK.8 Malleability of Nova’s IVC` https://eprint.iacr.org/2023/969.pdf )
    // We need to carefully check whether it lead to other vulnerability.

    let arity = self.step_circuit.arity();
    let num_augmented_circuits = if self.params.is_primary_circuit {
      // primary circuit only fold single running instance with secondary output strict r1cs instance
      1
    } else {
      // secondary circuit contains the logic to choose one of multiple augments running instance to fold
      self.num_augmented_circuits
    };

    if self.inputs.is_some() {
      let z0_len = self.inputs.get().map_or(0, |inputs| inputs.z0.len());
      if self.step_circuit.arity() != z0_len {
        return Err(SynthesisError::IncompatibleLengthVector(format!(
          "z0_len {:?} != arity lengh {:?}",
          z0_len,
          self.step_circuit.arity()
        )));
      }
      let last_augmented_circuit_index = self
        .inputs
        .get()
        .map_or(G::Base::ZERO, |inputs| inputs.last_augmented_circuit_index);
      if self.params.is_primary_circuit && last_augmented_circuit_index != G::Base::ZERO {
        return Err(SynthesisError::IncompatibleLengthVector(
          "primary circuit running instance only valid on index 0".to_string(),
        ));
      }
    }

    // Allocate witnesses
    let (params, i, z_0, z_i, U, u, T, program_counter, last_augmented_circuit_index) = self
      .alloc_witness(
        cs.namespace(|| "allocate the circuit witness"),
        arity,
        num_augmented_circuits,
      )?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // Synthesize the circuit for the non-base case and get the new running
    // instances along with a boolean indicating if all checks have passed
    // must use return `last_augmented_circuit_index_checked` since it got range checked
    let (last_augmented_circuit_index_checked, U_next_non_base, check_non_base_pass) = self
      .synthesize_non_base_case(
        cs.namespace(|| "synthesize non base case"),
        params.clone(),
        i.clone(),
        z_0.clone(),
        z_i.clone(),
        &U,
        u.clone(),
        T,
        arity,
        &last_augmented_circuit_index,
        program_counter.clone(),
        num_augmented_circuits,
      )?;

    // Synthesize the circuit for the base case and get the new running instances
    let U_next_base = self.synthesize_base_case(
      cs.namespace(|| "base case"),
      u.clone(),
      &last_augmented_circuit_index_checked,
      num_augmented_circuits,
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
      Ok(*i.get_value().get()? + G::Base::ONE)
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

    let (program_counter_new, z_next) = if let Some(program_counter) = &program_counter {
      self
        .step_circuit
        .synthesize(&mut cs.namespace(|| "F"), program_counter, &z_input)?
    } else {
      let zero_program_counter = alloc_zero(cs.namespace(|| "zero pc"))?;
      self
        .step_circuit
        .synthesize(&mut cs.namespace(|| "F"), &zero_program_counter, &z_input)?
    };
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
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      2 // params_next, i_new
      + program_counter.as_ref().map_or(0, |_| 1) // optional program counter
        + 2 * arity // zo, z1
        + num_augmented_circuits * (7 + 2 * self.params.n_limbs), // #num_augmented_circuits * (7 + [X0, X1]*#num_limb)
    );
    ro.absorb(&params);
    ro.absorb(&i_next);
    // optionally absorb program counter if exist
    if program_counter.is_some() {
      ro.absorb(&program_counter_new)
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
