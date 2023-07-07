//! The augmented circuit F' for  SuperNova that includes everything from Nova
//!   and additionally checks:
//!    1. Ui and pci are contained in the public output.
//!    2. Instance is folded into Ui[pci] correctly; just like Nova.
//!    3. Invokes the function ϕ on input (zi, ωi) to compute pci+1, which represents
//!    the index of the function Fj currently being run. pci+1 is then sent to the
//!    next invocation of an augmented circuit (which contains a verifier circuit).

//! There are two Verification Circuits. The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is
//! H(params = H(shape, ck), i, z0, zi, U). Each circuit folds the last invocation of
//! the other into the running instance

use crate::{
  constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS},
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
    utils::{
      alloc_num_equals, alloc_scalar_as_base, alloc_zero, conditionally_select_vec, le_bits_to_num,
    },
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{
    circuit::StepCircuit, commitment::CommitmentTrait, Group, ROCircuitTrait, ROConstantsCircuit,
  },
  Commitment,
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
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitParams {
  limb_width: usize,
  n_limbs: usize,
  is_primary_circuit: bool, // A boolean indicating if this is the primary circuit
}

impl CircuitParams {
  pub fn new(limb_width: usize, n_limbs: usize, is_primary_circuit: bool) -> Self {
    Self {
      limb_width,
      n_limbs,
      is_primary_circuit,
    }
  }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CircuitInputs<G: Group> {
  params: G::Scalar, // Hash(Shape of u2, Gens for u2). Needed for computing the challenge.
  i: G::Base,
  z0: Vec<G::Base>,
  zi: Option<Vec<G::Base>>,
  U: Option<RelaxedR1CSInstance<G>>,
  u: Option<R1CSInstance<G>>,
  T: Option<Commitment<G>>,
  program_counter: G::Base,
  output_U_i: Vec<G::Base>,
}

impl<G: Group> CircuitInputs<G> {
  /// Create new inputs/witness for the verification circuit
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    params: G::Scalar,
    i: G::Base,
    z0: Vec<G::Base>,
    zi: Option<Vec<G::Base>>,
    U: Option<RelaxedR1CSInstance<G>>,
    u: Option<R1CSInstance<G>>,
    T: Option<Commitment<G>>,
    program_counter: G::Base,
    output_U_i: Vec<G::Base>,
  ) -> Self {
    Self {
      params,
      i,
      z0,
      zi,
      U,
      u,
      T,
      program_counter,
      output_U_i,
    }
  }
}

/// The augmented circuit F' in Nova that includes a step circuit F
/// and the circuit for the verifier in Nova's non-interactive folding scheme
pub struct SuperNovaCircuit<G: Group, SC: StepCircuit<G::Base>> {
  params: CircuitParams,
  ro_consts: ROConstantsCircuit<G>,
  inputs: Option<CircuitInputs<G>>,
  step_circuit: SC, // The function that is applied for each step
}

impl<G: Group, SC: StepCircuit<G::Base>> SuperNovaCircuit<G, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub fn new(
    params: CircuitParams,
    inputs: Option<CircuitInputs<G>>,
    step_circuit: SC,
    ro_consts: ROConstantsCircuit<G>,
  ) -> Self {
    Self {
      params,
      inputs,
      step_circuit,
      ro_consts,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    arity: usize,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedNum<G::Base>>,
      AllocatedRelaxedR1CSInstance<G>,
      AllocatedR1CSInstance<G>,
      AllocatedPoint<G>,
      AllocatedNum<G::Base>,
      Vec<AllocatedNum<G::Base>>,
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

    // Allocate pci
    let program_counter = AllocatedNum::alloc(cs.namespace(|| "program_counter"), || Ok(self.inputs.get()?.program_counter))?;

    // Allocate U_i
    let output_U_i = (0..arity)
    .map(|i| {
      AllocatedNum::alloc(cs.namespace(|| format!("output_U_i_{i}")), || {
        Ok(self.inputs.get()?.output_U_i[i])
      })
    })
    .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

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
          Ok(self.inputs.get()?.zi.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

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
        inputs.T.get().map_or(None, |T| Some(T.to_coordinates()))
      }),
    )?;

    Ok((params, i, z_0, z_i, U, u, T, program_counter, output_U_i))
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
    z_0: Vec<AllocatedNum<G::Base>>,
    z_i: Vec<AllocatedNum<G::Base>>,
    U: AllocatedRelaxedR1CSInstance<G>,
    u: AllocatedR1CSInstance<G>,
    T: AllocatedPoint<G>,
    arity: usize,
    program_counter: AllocatedNum<G::Base>,
    output_U_i: Vec<AllocatedNum<G::Base>>,
  ) -> Result<(AllocatedRelaxedR1CSInstance<G>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * arity,
    );
    ro.absorb(params.clone());
    ro.absorb(i);
    for e in &z_0 {
      ro.absorb(e.clone());
    }
    for e in &z_i {
      ro.absorb(e.clone());
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;
    let check_pass = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X0,
      &hash,
    )?;

    //Check that hash H(pci, z0, z_{i+1})
    let mut ro2 = G::ROCircuit::new(
      self.ro_consts.clone(),
      4 * arity,
    );
    ro2.absorb(program_counter.clone());
    for e in &z_0 {
      ro2.absorb(e.clone());
    }
    for e in &z_i {
      ro2.absorb(e.clone());
    }
    for e in &output_U_i {
      ro2.absorb(e.clone());
    }

    let supernova_hash_bits = ro2.squeeze(cs.namespace(|| "Input U_i hash"), NUM_HASH_BITS)?;
    let supernova_hash = le_bits_to_num(cs.namespace(|| "bits to U_i hash"), supernova_hash_bits)?;
    let check_pass2 = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(program_counter, z0, zi)"),
      &u.X0,
      &supernova_hash,
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

  // The output pci
  pub fn output_program_counter(&self) -> Option<G::Base> {
    self.inputs.as_ref().map(|inputs| inputs.program_counter)
  }
  // The output U_i
  pub fn output_U_i(&self) -> Option<Vec<G::Base>> {
    self.inputs.as_ref().map(|inputs| inputs.output_U_i.clone())
  }
}

impl<G: Group, SC: StepCircuit<G::Base>> Circuit<<G as Group>::Base>
  for SuperNovaCircuit<G, SC>
{

  fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (params, i, z_0, z_i, U, u, T, program_counter, output_U_i) =
      self.alloc_witness(cs.namespace(|| "allocate the circuit witness"), arity)?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

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
      arity,
      program_counter.clone(),
      output_U_i.clone(),
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
      Ok(*i.get_value().get()? + G::Base::ONE)
    })?;
    cs.enforce(
      || "check i + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + i_new.get_variable() - CS::one() - i.get_variable(),
    );

    // Compute pci + 1
    let program_counter_new = AllocatedNum::alloc(cs.namespace(|| "program_counter + 1"), || {
      Ok(*program_counter.get_value().get()? + G::Base::ONE)
    })?;
    cs.enforce(
      || "check program_counter + 1",
      |lc| lc,
      |lc| lc,
      |lc| lc + program_counter_new.get_variable() - CS::one() - program_counter.get_variable(),
    );

    program_counter
    .inputize(cs.namespace(|| "Output pci"))?;

    // Compute length of U_i and make sure it is the same as program_counter
    let output_U_i_length = AllocatedNum::alloc(cs.namespace(|| "output_U_i length"), || {
      Ok(G::Base::from(output_U_i.len() as u64))
    })?;
    cs.enforce(
      || "check output_U_i length",
      |lc| lc + output_U_i_length.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + program_counter.get_variable()
    );

    for (i, num) in output_U_i.iter().enumerate() {
      num.inputize(cs.namespace(|| format!("Output U_i_{}", i)))?;
    }

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

    // Compute the new hash H(params, Unew, i+1, z0, z_{i+1})
    let mut ro = G::ROCircuit::new(self.ro_consts.clone(), NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * arity);
    ro.absorb(params);
    ro.absorb(i_new.clone());
    for e in &z_0 {
      ro.absorb(e.clone());
    }
    for e in &z_next {
      ro.absorb(e.clone());
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), hash_bits)?;

    /*
      To check correct sequencing we are just going to make a hash with PCI and
      the other public outputs. The next RunningInstance can take the pre-image of the hash.
      *Works much like Nova but with the hash being used outside of the F'[pci].

      "Finally, there is a subtle sizing issue in the above description: in each step,
      because Ui+1 is produced as the public IO of F0 pci+1, it must be contained in
      the public IO of instance ui+1. In the next iteration, because ui+1 is folded
      into Ui+1[pci+1], this means that Ui+1[pci+1] is at least as large as Ui by the
      properties of the folding scheme. This means that the list of running instances
      grows in each step. To alleviate this issue, we have each F0j only produce a hash
      of its outputs as public output. In the subsequent step, the next augmented
      function takes as non-deterministic input a preimage to this hash." pg.16

      https://eprint.iacr.org/2022/1758.pdf
    */

    // Compute the SuperNova hash H(pci, z0, z_{i+1})
    let mut ro2 = G::ROCircuit::new(self.ro_consts.clone(), 4 * arity);
    ro2.absorb(program_counter_new.clone());
    for e in &z_0 {
      ro2.absorb(e.clone());
    }
    for e in &z_next {
      ro2.absorb(e.clone());
    }
    for e in output_U_i {
      ro2.absorb(e.clone());
    }
    let supernova_hash_bits = ro2.squeeze(cs.namespace(|| "output hash U_i"), NUM_HASH_BITS)?;
    let supernova_hash = le_bits_to_num(cs.namespace(|| "convert U_i hash to num"), supernova_hash_bits)?;

    supernova_hash.inputize(cs.namespace(|| "output new hash U_i of this circuit"))?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    u.X1
      .inputize(cs.namespace(|| "Output unmodified hash of the other circuit"))?;
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellperson::{shape_cs::ShapeCS, solver::SatisfyingAssignment};
  type PastaG1 = pasta_curves::pallas::Point;
  type PastaG2 = pasta_curves::vesta::Point;

  use crate::constants::{BN_LIMB_WIDTH, BN_N_LIMBS};
  use crate::{
    bellperson::r1cs::{NovaShape, NovaWitness},
    gadgets::utils::scalar_as_base,
    provider::poseidon::PoseidonConstantsCircuit,
    traits::{circuit::TrivialTestCircuit, ROConstantsTrait},
  };

  // In the following we use 1 to refer to the primary, and 2 to refer to the secondary circuit
  fn test_recursive_circuit_with<G1, G2>(
    primary_params: CircuitParams,
    secondary_params: CircuitParams,
    ro_consts1: ROConstantsCircuit<G2>,
    ro_consts2: ROConstantsCircuit<G1>,
    num_constraints_primary: usize,
    num_constraints_secondary: usize,
  ) where
    G1: Group<Base = <G2 as Group>::Scalar>,
    G2: Group<Base = <G1 as Group>::Scalar>,
  {
    // Initialize the shape and ck for the primary
    let circuit1: SuperNovaCircuit<G2, TrivialTestCircuit<<G2 as Group>::Base>> =
      SuperNovaCircuit::new(
        primary_params.clone(),
        None,
        TrivialTestCircuit::default(),
        ro_consts1.clone(),
      );
    let mut cs: ShapeCS<G1> = ShapeCS::new();
    let _ = circuit1.synthesize(&mut cs);
    let (shape1, ck1) = cs.r1cs_shape();
    assert_eq!(cs.num_constraints(), num_constraints_primary);

    // Initialize the shape and ck for the secondary
    let circuit2: SuperNovaCircuit<G1, TrivialTestCircuit<<G1 as Group>::Base>> =
      SuperNovaCircuit::new(
        secondary_params.clone(),
        None,
        TrivialTestCircuit::default(),
        ro_consts2.clone(),
      );
    let mut cs: ShapeCS<G2> = ShapeCS::new();
    let _ = circuit2.synthesize(&mut cs);
    let (shape2, ck2) = cs.r1cs_shape();
    assert_eq!(cs.num_constraints(), num_constraints_secondary);

    // Execute the base case for the primary
    let zero1 = <<G2 as Group>::Base as Field>::ZERO;
    let mut cs1: SatisfyingAssignment<G1> = SatisfyingAssignment::new();
    let inputs1: CircuitInputs<G2> = CircuitInputs::new(
      scalar_as_base::<G1>(zero1), // pass zero for testing
      zero1,
      vec![zero1],
      None,
      None,
      None,
      None,
      zero1,
      vec![zero1]
    );
    let circuit1: SuperNovaCircuit<G2, TrivialTestCircuit<<G2 as Group>::Base>> =
      SuperNovaCircuit::new(
        primary_params,
        Some(inputs1),
        TrivialTestCircuit::default(),
        ro_consts1,
      );
    let _ = circuit1.synthesize(&mut cs1);
    let (inst1, witness1) = cs1.r1cs_instance_and_witness(&shape1, &ck1).unwrap();
    // Make sure that this is satisfiable
    assert!(shape1.is_sat(&ck1, &inst1, &witness1).is_ok());

    // Execute the base case for the secondary
    let zero2 = <<G1 as Group>::Base as Field>::ZERO;
    let mut cs2: SatisfyingAssignment<G2> = SatisfyingAssignment::new();
    let inputs2: CircuitInputs<G1> = CircuitInputs::new(
      scalar_as_base::<G2>(zero2), // pass zero for testing
      zero2,
      vec![zero2],
      None,
      None,
      Some(inst1),
      None,
      zero2,
      vec![zero2]
    );
    let circuit2: SuperNovaCircuit<G1, TrivialTestCircuit<<G1 as Group>::Base>> =
      SuperNovaCircuit::new(
        secondary_params,
        Some(inputs2),
        TrivialTestCircuit::default(),
        ro_consts2,
      );
    let _ = circuit2.synthesize(&mut cs2);
    let (inst2, witness2) = cs2.r1cs_instance_and_witness(&shape2, &ck2).unwrap();
    // Make sure that it is satisfiable
    assert!(shape2.is_sat(&ck2, &inst2, &witness2).is_ok());
  }

  #[test]
  fn test_recursive_circuit() {
    let params1 = CircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, true);
    let params2 = CircuitParams::new(BN_LIMB_WIDTH, BN_N_LIMBS, false);
    let ro_consts1: ROConstantsCircuit<PastaG2> = PoseidonConstantsCircuit::new();
    let ro_consts2: ROConstantsCircuit<PastaG1> = PoseidonConstantsCircuit::new();

    test_recursive_circuit_with::<PastaG1, PastaG2>(
      params1, params2, ro_consts1, ro_consts2, 9815, 10347,
    );
  }
}
