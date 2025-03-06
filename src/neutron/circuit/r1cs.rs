use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::ecc::AllocatedNonnativePoint,
  r1cs::R1CSInstance,
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An Allocated R1CS Instance but with nonnative points
/// In our context, we will have a single entry for the public input, so we have X
#[derive(Clone)]
pub struct AllocatedNonnativeR1CSInstance<E: Engine> {
  pub(crate) comm_W: AllocatedNonnativePoint<E>,
  pub(crate) X: AllocatedNum<E::Scalar>,
}

impl<E: Engine> AllocatedNonnativeR1CSInstance<E> {
  /// Takes the r1cs instance and creates a new allocated r1cs instance
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    u: Option<&R1CSInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_W"),
      u.map(|u| u.comm_W.to_coordinates()),
    )?;

    let X = AllocatedNum::alloc(cs.namespace(|| "allocate X"), || {
      Ok(u.map_or(E::Scalar::ZERO, |u| u.X[0]))
    })?;

    Ok(Self { comm_W, X })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    ro: &mut E::RO2Circuit,
  ) -> Result<(), SynthesisError> {
    self.comm_W.absorb_in_ro(cs.namespace(|| "comm_W"), ro)?;
    ro.absorb(&self.X);
    Ok(())
  }
}
