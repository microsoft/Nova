//! Circuit representation of NeutronNova's NIFS
use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{num::AllocatedNum, Assignment, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{
      bignat::BigNat,
      util::{f_to_nat, Num},
    },
    r1cs::AllocatedR1CSInstance,
    utils::{
      alloc_bignat_constant, alloc_one, alloc_scalar_as_base, conditionally_select,
      conditionally_select_bignat, le_bits_to_num,
    },
  },
  neutron::relation::FoldedInstance,
  traits::{commitment::CommitmentTrait, Engine, Group, ROCircuitTrait, ROConstantsCircuit},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's NIFS
pub struct AllocatedNIFS<E: Engine> {
  pub(crate) comm_E: AllocatedPoint<E>,
  pub(crate) poly: CompressedUniPoly<E::Scalar>,
}
