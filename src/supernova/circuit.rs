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

use std::ops::Not;

use crate::{
  constants::{NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS},
  gadgets::{
    ecc::AllocatedPoint,
    r1cs::{
      conditionally_select_relaxed_r1cs_supernova, AllocatedR1CSInstanceSuperNova,
      AllocatedRelaxedR1CSInstanceSuperNova,
    },
    utils::{
      add_allocated_num, alloc_bignat_constant, alloc_num_equals, alloc_scalar_as_base, alloc_zero,
      conditionally_select, conditionally_select_vec, le_bits_to_num, scalar_as_base,
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
use itertools::Itertools;
use num_bigint::BigInt;
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
  params_next: G::Scalar,
  params: G::Scalar,
  i: G::Base,
  z0: Vec<G::Base>,
  zi: Option<Vec<G::Base>>,
  U: Option<Vec<RelaxedR1CSInstance<G>>>,
  u: Option<R1CSInstance<G>>,
  T: Option<Commitment<G>>,
  program_counter: G::Base,
  last_circuit_index_selector: G::Base,
}

impl<G: Group> CircuitInputs<G> {
  /// Create new inputs/witness for the verification circuit
  #[allow(clippy::too_many_arguments)]
  pub fn new(
    params_next: G::Scalar, // params_next can not being calculated inside, therefore pass as witness
    params: G::Scalar,
    i: G::Base,
    z0: Vec<G::Base>,
    zi: Option<Vec<G::Base>>,
    U: Option<Vec<RelaxedR1CSInstance<G>>>,
    u: Option<R1CSInstance<G>>,
    T: Option<Commitment<G>>,
    program_counter: G::Base,
    last_circuit_index_selector: G::Base,
  ) -> Self {
    Self {
      params_next,
      params,
      i,
      z0,
      zi,
      U,
      u,
      T,
      program_counter,
      last_circuit_index_selector,
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
  u_i_length: usize,
}

impl<G: Group, SC: StepCircuit<G::Base>> SuperNovaCircuit<G, SC> {
  /// Create a new verification circuit for the input relaxed r1cs instances
  pub fn new(
    params: CircuitParams,
    inputs: Option<CircuitInputs<G>>,
    step_circuit: SC,
    ro_consts: ROConstantsCircuit<G>,
    u_i_length: usize,
  ) -> Self {
    Self {
      params,
      inputs,
      step_circuit,
      ro_consts,
      u_i_length,
    }
  }

  /// Allocate all witnesses and return
  fn alloc_witness<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    arity: usize,
    u_i_length: usize,
  ) -> Result<
    (
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedNum<G::Base>>,
      Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>,
      AllocatedR1CSInstanceSuperNova<G>,
      AllocatedPoint<G>,
      AllocatedNum<G::Base>,
      AllocatedNum<G::Base>,
    ),
    SynthesisError,
  > {
    // Allocate the params
    let params = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "params"),
      self.inputs.get().map_or(None, |inputs| Some(inputs.params)),
    )?;

    // Allocate the params_next
    let params_next = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "params_next"),
      self
        .inputs
        .get()
        .map_or(None, |inputs| Some(inputs.params_next)),
    )?;

    // Allocate i
    let i = AllocatedNum::alloc(cs.namespace(|| "i"), || Ok(self.inputs.get()?.i))?;

    // Allocate pci
    let program_counter = AllocatedNum::alloc(cs.namespace(|| "program_counter"), || {
      Ok(self.inputs.get()?.program_counter)
    })?;

    // Allocate circuit_input_selector
    let last_circuit_index_selector =
      AllocatedNum::alloc(cs.namespace(|| "circuit_index_selector"), || {
        Ok(self.inputs.get()?.last_circuit_index_selector)
      })?;

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

    // Allocate the running instances
    let U = (0..u_i_length)
      .map(|i| {
        AllocatedRelaxedR1CSInstanceSuperNova::alloc(
          cs.namespace(|| format!("Allocate U {:?}", i)),
          self.inputs.get().map_or(None, |inputs| {
            inputs.U.get().map_or(None, |U| U.get(i).cloned())
          }),
          self.params.limb_width,
          self.params.n_limbs,
        )
      })
      .collect::<Result<Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>, _>>()?;

    // Allocate the instance to be folded in
    let u = AllocatedR1CSInstanceSuperNova::alloc(
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

    Ok((
      params,
      params_next,
      i,
      z_0,
      z_i,
      U,
      u,
      T,
      program_counter,
      last_circuit_index_selector,
    ))
  }

  /// Synthesizes base case and returns the new relaxed R1CSInstance
  fn synthesize_base_case<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    u: AllocatedR1CSInstanceSuperNova<G>,
    last_circuit_index_selector: &AllocatedNum<G::Base>,
    u_i_length: usize,
  ) -> Result<Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>, SynthesisError> {
    let mut cs = cs.namespace(|| "alloc U_i default");

    // The primary circuit just initialize default AllocatedRelaxedR1CSInstanceSuperNova
    let U_default = if self.params.is_primary_circuit {
      (0..u_i_length)
        .map(|i| {
          AllocatedRelaxedR1CSInstanceSuperNova::default(
            cs.namespace(|| format!("Allocate U_default {:?}", i)),
            self.params.limb_width,
            self.params.n_limbs,
          )
        })
        .collect::<Result<Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>, _>>()?
    } else {
      // The secondary circuit convert the incoming R1CS instance on index which match circuit index
      let imcomming_r1cs = AllocatedRelaxedR1CSInstanceSuperNova::from_r1cs_instance(
        cs.namespace(|| "Allocate imcomming_r1cs"),
        u,
        self.params.limb_width,
        self.params.n_limbs,
      )?;
      (0..u_i_length)
        .map(|i| {
          let i_alloc =
            AllocatedNum::alloc(cs.namespace(|| format!("i allocated on {:?}", i)), || {
              Ok(scalar_as_base::<G>(G::Scalar::from(i as u64)))
            })?;
          let equal_bit = Boolean::from(alloc_num_equals(
            cs.namespace(|| format!("check equal bit {:?}", i)),
            &i_alloc,
            &last_circuit_index_selector,
          )?);
          let default = &AllocatedRelaxedR1CSInstanceSuperNova::default(
            cs.namespace(|| format!("Allocate U_default {:?}", i)),
            self.params.limb_width,
            self.params.n_limbs,
          )?;
          conditionally_select_relaxed_r1cs_supernova(
            cs.namespace(|| format!("select on index namespace {:?}", i)),
            &imcomming_r1cs,
            default,
            &equal_bit,
          )
        })
        .collect::<Result<Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>, _>>()?
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
    U: &Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>,
    u: AllocatedR1CSInstanceSuperNova<G>,
    T: AllocatedPoint<G>,
    arity: usize,
    last_circuit_index_selector: &AllocatedNum<G::Base>,
    program_counter: AllocatedNum<G::Base>,
  ) -> Result<(AllocatedRelaxedR1CSInstanceSuperNova<G>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      NUM_FE_WITHOUT_IO_FOR_CRHF + 18 * arity,
    );
    ro.absorb(params.clone());
    ro.absorb(i);
    ro.absorb(program_counter.clone());
    // NOTE only witness and DO NOT need to constrain last_circuit_index_selector.
    // Because prover can only make valid IVC proof by folding u into correct U_i[last_circuit_index_selector]

    for e in &z_0 {
      ro.absorb(e.clone());
    }
    for e in &z_i {
      ro.absorb(e.clone());
    }

    U.iter().enumerate().try_for_each(|(i, U)| {
      U.absorb_in_ro(cs.namespace(|| format!("absorb U {:?}", i)), &mut ro)
    })?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;
    let check_pass: AllocatedBit = alloc_num_equals(
      cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
      &u.X0,
      &hash,
    )?;

    // Run NIFS Verifier
    let empty_U = AllocatedRelaxedR1CSInstanceSuperNova::alloc(
      cs.namespace(|| "empty U"),
      None,
      self.params.limb_width,
      self.params.n_limbs,
    )?;
    // select target when index match or empty
    let U: Result<Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>, SynthesisError> = U
      .iter()
      .enumerate()
      .map(|(i, U)| {
        let i_alloc =
          AllocatedNum::alloc(cs.namespace(|| format!("U_i i{:?} allocated", i)), || {
            Ok(scalar_as_base::<G>(G::Scalar::from(i as u64)))
          })?;
        let equal_bit = Boolean::from(alloc_num_equals(
          cs.namespace(|| format!("check U {:?} equal bit", i)),
          &i_alloc,
          &last_circuit_index_selector,
        )?);
        conditionally_select_relaxed_r1cs_supernova(
          cs.namespace(|| format!("select on index namespace {:?}", i)),
          &U,
          &empty_U,
          &equal_bit,
        )
      })
      .collect();

    // Here purely fold for all field is safe, because only 1 of them != G::Zero while other are all G::Zero
    let U_to_fold = U?.iter().enumerate().try_fold(empty_U, |agg, (i, U)| {
      Result::<AllocatedRelaxedR1CSInstanceSuperNova<G>, SynthesisError>::Ok(
        AllocatedRelaxedR1CSInstanceSuperNova {
          W: agg
            .W
            .add(cs.namespace(|| format!("fold W {:?}", i)), &U.W)?,
          E: agg
            .E
            .add(cs.namespace(|| format!("fold E {:?}", i)), &U.E)?,
          u: {
            let cs = cs.namespace(|| format!("fold u {:?}", i));
            add_allocated_num(cs, &agg.u, &U.u)?
          },
          X0: agg.X0.add(&U.X0)?,
          X1: agg.X1.add(&U.X1)?,
        },
      )
    })?;
    let U_fold = U_to_fold.fold_with_r1cs(
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
}

impl<G: Group, SC: StepCircuit<G::Base>> Circuit<<G as Group>::Base> for SuperNovaCircuit<G, SC> {
  fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
  ) -> Result<(), SynthesisError> {
    let arity = self.step_circuit.arity();
    let u_i_length = self.u_i_length;

    // only support 2 circuit as a demo
    assert_eq!(u_i_length, 2);

    // Allocate all witnesses
    let (params, params_next, i, z_0, z_i, U, u, T, program_counter, last_circuit_index_selector) =
      self.alloc_witness(
        cs.namespace(|| "allocate the circuit witness"),
        arity,
        u_i_length,
      )?;

    // Compute variable indicating if this is the base case
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let is_base_case = alloc_num_equals(cs.namespace(|| "Check if base case"), &i.clone(), &zero)?;

    // Synthesize the circuit for the base case and get the new running instance
    let Unew_base = self.synthesize_base_case(
      cs.namespace(|| "base case"),
      u.clone(),
      &last_circuit_index_selector,
      u_i_length,
    )?;

    // Synthesize the circuit for the non-base case and get the new running
    // instance along with a boolean indicating if all checks have passed
    let (Unew_non_base_folded, check_non_base_pass) = self.synthesize_non_base_case(
      cs.namespace(|| "synthesize non base case"),
      params.clone(),
      i.clone(),
      z_0.clone(),
      z_i.clone(),
      &U,
      u.clone(),
      T,
      arity,
      &last_circuit_index_selector,
      program_counter.clone(),
    )?;

    // update AllocatedRelaxedR1CSInstanceSuperNova on index match circuit index
    let Unew_non_base: Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>> = U
      .iter()
      .enumerate()
      .map(|(i, U)| {
        let mut cs = cs.namespace(|| format!("U_i+1 non_base conditional selection {:?}", i));
        let i_alloc = AllocatedNum::alloc(cs.namespace(|| "i allocated"), || {
          Ok(scalar_as_base::<G>(G::Scalar::from(i as u64)))
        })?;
        let equal_bit = Boolean::from(alloc_num_equals(
          cs.namespace(|| "check equal bit"),
          &i_alloc,
          &last_circuit_index_selector,
        )?);
        conditionally_select_relaxed_r1cs_supernova(
          cs.namespace(|| "select on index namespace"),
          &Unew_non_base_folded,
          &U,
          &equal_bit,
        )
      })
      .collect::<Result<Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>, _>>()?;

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
    let Unew = Unew_non_base
      .iter()
      .enumerate()
      .zip(Unew_base.iter())
      .map(|((i, Unew_non_base), Unew_base)| {
        Unew_base.conditionally_select(
          cs.namespace(|| format!("compute U_new {:?}", i)),
          Unew_non_base,
          &Boolean::from(is_base_case.clone()),
        )
      })
      .collect::<Result<Vec<AllocatedRelaxedR1CSInstanceSuperNova<G>>, _>>()?;

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

    // Compute the new hash H(params, i+1, program_counter, z0, z_{i+1}, U_new)
    let mut ro = G::ROCircuit::new(
      self.ro_consts.clone(),
      NUM_FE_WITHOUT_IO_FOR_CRHF + 18 * arity,
    );
    ro.absorb(params_next.clone());
    ro.absorb(i_new.clone());
    ro.absorb(program_counter_new.clone());
    for e in &z_0 {
      ro.absorb(e.clone());
    }
    for e in &z_next {
      ro.absorb(e.clone());
    }
    Unew.iter().enumerate().try_for_each(|(i, U)| {
      U.absorb_in_ro(cs.namespace(|| format!("absorb U_new {:?}", i)), &mut ro)
    })?;

    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), hash_bits)?;

    /*
      To check correct sequencing we are just going to make a hash with PCI and
      z_{i+1}, U_i. The next RunningInstance can take the pre-image of the hash.
      *Works much like Nova.

      TODO: figure out below description still needed or not
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

    // Bypass unmodified hash of other circuit as next X[0]
    // and output the computed the computed hash as next X[1]
    u.X1
      .inputize(cs.namespace(|| "bypass unmodified hash of the other circuit"))?;
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
        2,
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
        2,
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
      scalar_as_base::<G1>(zero1), // pass zero for testing
      zero1,
      vec![zero1],
      None,
      None,
      None,
      None,
      zero1,
      zero1,
    );
    let circuit1: SuperNovaCircuit<G2, TrivialTestCircuit<<G2 as Group>::Base>> =
      SuperNovaCircuit::new(
        primary_params,
        Some(inputs1),
        TrivialTestCircuit::default(),
        ro_consts1,
        2,
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
      scalar_as_base::<G2>(zero2), // pass zero for testing
      zero2,
      vec![zero2],
      None,
      None,
      Some(inst1),
      None,
      zero2,
      zero2,
    );
    let circuit2: SuperNovaCircuit<G1, TrivialTestCircuit<<G1 as Group>::Base>> =
      SuperNovaCircuit::new(
        secondary_params,
        Some(inputs2),
        TrivialTestCircuit::default(),
        ro_consts2,
        2,
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
      params1, params2, ro_consts1, ro_consts2, 11866, 12436,
    );
  }
}
