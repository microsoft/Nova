//! This module implements various gadgets necessary for folding R1CS types.
use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{ecc::AllocatedPoint, utils::alloc_scalar_as_base},
  r1cs::R1CSInstance,
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};

/// An Allocated R1CS Instance
#[derive(Clone)]
pub struct AllocatedR1CSInstance<E: Engine> {
  pub(crate) W: AllocatedPoint<E>,
  pub(crate) X0: AllocatedNum<E::Base>,
  pub(crate) X1: AllocatedNum<E::Base>,
}

impl<E: Engine> AllocatedR1CSInstance<E> {
  /// Takes the r1cs instance and creates a new allocated r1cs instance
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    u: Option<&R1CSInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    let W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate W"),
      u.map(|u| u.comm_W.to_coordinates()),
    )?;
    W.check_on_curve(cs.namespace(|| "check W on curve"))?;

    let X0 = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate X[0]"), u.map(|u| u.X[0]))?;
    let X1 = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate X[1]"), u.map(|u| u.X[1]))?;

    Ok(AllocatedR1CSInstance { W, X0, X1 })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro(&self, ro: &mut E::ROCircuit) {
    ro.absorb(&self.W.x);
    ro.absorb(&self.W.y);
    ro.absorb(&self.W.is_infinity);
    ro.absorb(&self.X0);
    ro.absorb(&self.X1);
  }
}
