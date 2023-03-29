//! This library implements Nova, but with parallelized proving enabled 
#![deny(
    warnings,
    unused,
    future_incompatible,
    nonstandard_style,
    rust_2018_idioms,
    missing_docs
  )]
  #![allow(non_snake_case)]
  #![allow(clippy::type_complexity)]
  #![forbid(unsafe_code)]

use crate::bellperson::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  };
  use ::bellperson::{Circuit, ConstraintSystem};
  use circuit::{NovaAugmentedCircuit, NovaAugmentedCircuitInputs, NovaAugmentedCircuitParams};
  use constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_WITHOUT_IO_FOR_CRHF, NUM_HASH_BITS};
  use core::marker::PhantomData;
  use errors::NovaError;
  use ff::Field;
  use gadgets::utils::scalar_as_base;
  use nifs::NIFS;
  use r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness};
  use serde::{Deserialize, Serialize};
  use traits::{
    circuit::StepCircuit,
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    snark::RelaxedR1CSSNARKTrait,
    AbsorbInROTrait, Group, ROConstants, ROConstantsCircuit, ROConstantsTrait, ROTrait,
  };



//! There are two Verification Circuits. The primary and the secondary.
//! Each of them is over a Pasta curve but
//! only the primary executes the next step of the computation.
//! We have two running instances. Each circuit takes as input 2 hashes: one for each
//! of the running instances. Each of these hashes is
//! H(params = H(shape, ck), i, z0, zi, U). Each circuit folds the last invocation of
//! the other into the running instance

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NovaAugmentedCircuitParams {
limb_width: usize,
n_limbs: usize,
is_primary_circuit: bool, // A boolean indicating if this is the primary circuit
}

impl NovaAugmentedCircuitParams {
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
pub struct NovaAugmentedCircuitInputs<G: Group> {
params: G::Scalar, // Hash(Shape of u2, Gens for u2). Needed for computing the challenge.
i: G::Base,
z0: Vec<G::Base>,
zi: Option<Vec<G::Base>>,
U: Option<RelaxedR1CSInstance<G>>,
u: Option<R1CSInstance<G>>,
T: Option<Commitment<G>>,
}

impl<G: Group> NovaAugmentedCircuitInputs<G> {
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

/// The augmented circuit F' in Nova that includes a step circuit F
/// and the circuit for the verifier in Nova's non-interactive folding scheme
pub struct NovaAugmentedCircuit<G: Group, SC: StepCircuit<G::Base>> {
params: NovaAugmentedCircuitParams,
ro_consts: ROConstantsCircuit<G>,
inputs: Option<NovaAugmentedCircuitInputs<G>>,
step_circuit: SC, // The function that is applied for each step
}

impl<G: Group, SC: StepCircuit<G::Base>> NovaAugmentedCircuit<G, SC> {
/// Create a new verification circuit for the input relaxed r1cs instances
pub fn new(
    params: NovaAugmentedCircuitParams,
    inputs: Option<NovaAugmentedCircuitInputs<G>>,
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
    let z_0 = (0..arity)
    .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("z0_{i}")), || {
        Ok(self.inputs.get()?.z0[i])
        })
    })
    .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // Allocate zi. If inputs.zi is not provided (base case) allocate default value 0
    let zero = vec![G::Base::zero(); arity];
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
    z_0: Vec<AllocatedNum<G::Base>>,
    z_i: Vec<AllocatedNum<G::Base>>,
    U: AllocatedRelaxedR1CSInstance<G>,
    u: AllocatedR1CSInstance<G>,
    U_prime: AllocatedRelaxedR1CSInstance<G>,
    u_prime: AllocatedR1CSInstance<G>,
    T: AllocatedPoint<G>,
    T_prime: AllocatedPoint<G>,
    arity: usize,
) -> Result<(AllocatedRelaxedR1CSInstance<G>, AllocatedBit), SynthesisError> {
    // Check that u.x[0] = Hash(params, U, i, z0, zi)
    let mut ro = G::ROCircuit::new(
    self.ro_consts.clone(),
    NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * arity,
    );
    ro.absorb(params.clone());
    ro.absorb(i);
    for e in z_0 {
    ro.absorb(e);
    }
    for e in z_i {
    ro.absorb(e);
    }
    U.absorb_in_ro(cs.namespace(|| "absorb U"), &mut ro)?;

    let hash_bits = ro.squeeze(cs.namespace(|| "Input hash"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "bits to hash"), hash_bits)?;
    let check_pass = alloc_num_equals(
    cs.namespace(|| "check consistency of u.X[0] with H(params, U, i, z0, zi)"),
    &u.X0,
    &hash,
    )?;

    // Run NIFS Verifier
    let U_fold = U.fold_with_r1cs(
    cs.namespace(|| "compute fold of U and u"),
    params.copy(),
    u,
    T,
    self.ro_consts.clone(),
    self.params.limb_width,
    self.params.n_limbs,
    )?;

    let U_fold_prime = U_prime.fold_with_r1cs(cs.namespace(|| "compute fold of U' and u'"),
        params,
        u_prime,
        T_prime,
        self.ro_consts.clone(),
        self.params.limb_width,
        self.params.n_limbs,
    )?;

    Ok((U_fold, check_pass))
}
}
  
impl<G: Group, SC: StepCircuit<G::Base>> Circuit<<G as Group>::Base>
for NovaAugmentedCircuit<G, SC>
{
fn synthesize<CS: ConstraintSystem<<G as Group>::Base>>(
    self,
    cs: &mut CS,
) -> Result<(), SynthesisError> {
    let arity = self.step_circuit.arity();

    // Allocate all witnesses
    let (params, i, z_0, z_i, U, u, T) =
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
    let mut ro = G::ROCircuit::new(self.ro_consts, NUM_FE_WITHOUT_IO_FOR_CRHF + 2 * arity);
    ro.absorb(params);
    ro.absorb(i_new.clone());
    for e in z_0 {
    ro.absorb(e);
    }
    for e in z_next {
    ro.absorb(e);
    }
    Unew.absorb_in_ro(cs.namespace(|| "absorb U_new"), &mut ro)?;
    let hash_bits = ro.squeeze(cs.namespace(|| "output hash bits"), NUM_HASH_BITS)?;
    let hash = le_bits_to_num(cs.namespace(|| "convert hash to num"), hash_bits)?;

    // Outputs the computed hash and u.X[1] that corresponds to the hash of the other circuit
    u.X1
    .inputize(cs.namespace(|| "Output unmodified hash of the other circuit"))?;
    hash.inputize(cs.namespace(|| "output new hash of this circuit"))?;

    Ok(())
}
}

/// A SNARK that proves the correct execution of an incremental computation, designed for parallel
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveParallelProof<G1, G2, C1, C2>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
  C2: StepCircuit<G2::Scalar>,
{
  r_W: RelaxedR1CSWitness<G1>,
  r_U: RelaxedR1CSInstance<G1>,
  l_w: R1CSWitness<G1>,
  l_u: R1CSInstance<G1>,
  end: usize,
  start: usize,
  z_start_input: Vec<G1::Scalar>,
  z_end_output: Vec<G1::Scalar>,
  _p_c1: PhantomData<C1>
}


impl<G1, G2, C1> RecursiveParallelProof<G1, G2, C1>
where
  G1: Group<Base = <G2 as Group>::Scalar>,
  G2: Group<Base = <G1 as Group>::Scalar>,
  C1: StepCircuit<G1::Scalar>,
{
    pub fn new(input: Vec<G1::Scalar>, start: usize) -> Self {
        // TODO - create a trivial case where we go from start to start + 1
    }

    pub fn fold(&mut self, pp: &PublicParams<G1, G2, C1, C2>) {

    }
}