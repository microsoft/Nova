//! This module implements various gadgets necessary for folding R1CS types with NeutronNova folding scheme.
use crate::{
  frontend::{num::AllocatedNum, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedNonnativePoint,
    utils::{alloc_zero, conditionally_select},
  },
  neutron::{circuit::r1cs::AllocatedNonnativeR1CSInstance, relation::FoldedInstance},
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's FoldedInstance
/// In our context, public IO of circuits folded will have only one entry, so we have X
#[derive(Clone, Debug)]
pub struct AllocatedFoldedInstance<E: Engine> {
  pub(crate) comm_W: AllocatedNonnativePoint<E>,
  pub(crate) comm_E: AllocatedNonnativePoint<E>,
  pub(crate) T: AllocatedNum<E::Scalar>,
  pub(crate) u: AllocatedNum<E::Scalar>,
  pub(crate) X: AllocatedNum<E::Scalar>,
}

impl<E: Engine> AllocatedFoldedInstance<E> {
  /// Allocates the given `FoldedInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    inst: Option<&FoldedInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.
    let comm_W = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate W"),
      inst.map(|inst| inst.comm_W.to_coordinates()),
    )?;

    let comm_E = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate E"),
      inst.map(|inst| inst.comm_E.to_coordinates()),
    )?;

    let T = AllocatedNum::alloc(cs.namespace(|| "allocate T"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.T))
    })?;

    let u = AllocatedNum::alloc(cs.namespace(|| "allocate u"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.u))
    })?;

    // Allocate X. If the input instance is None, then allocate default values 0.
    let X = AllocatedNum::alloc(cs.namespace(|| "allocate X"), || {
      Ok(inst.map_or(E::Scalar::ZERO, |inst| inst.X[0]))
    })?;

    Ok(Self {
      comm_W,
      comm_E,
      T,
      u,
      X,
    })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, T = 0, u = 0, X = 0
  pub fn default<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();

    // Allocate T = 0. Similar to X0 and X1, we do not need to check that T is well-formed
    let T = alloc_zero(cs.namespace(|| "allocate T"));

    let u = T.clone();

    // X is allocated and set to zero
    let X = T.clone();

    Ok(Self {
      comm_W,
      comm_E,
      T,
      u,
      X,
    })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    ro: &mut E::RO2Circuit,
  ) -> Result<(), SynthesisError> {
    self
      .comm_W
      .absorb_in_ro(cs.namespace(|| "absorb W in RO"), ro)?;
    self
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb E in RO"), ro)?;
    ro.absorb(&self.T);
    ro.absorb(&self.u);
    ro.absorb(&self.X);
    Ok(())
  }

  /// Folds self with an r1cs instance and returns the result
  pub fn fold<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    U2: &AllocatedNonnativeR1CSInstance<E>,
    r_b: &AllocatedNum<E::Scalar>,
    T_out: &AllocatedNum<E::Scalar>,
    comm_W_fold: &AllocatedNonnativePoint<E>,
    comm_E_fold: &AllocatedNonnativePoint<E>,
  ) -> Result<Self, SynthesisError> {
    // u_fold = (1-r_b) * self.u + r_b * U2.u
    // u_fold = self.u - r_b * self.u + r_b * U2.u
    // u_fold = self.u + r_b (U2.u - self.u)
    // In our context U2.u = 1, so u_fold = self.u + r_b (1 - self.u)
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "allocate u_fold"), || {
      let u = self
        .u
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U2_u = E::Scalar::ONE;
      Ok(u + r_b * (U2_u - u))
    })?;

    cs.enforce(
      || "enforce u_fold -self.u  = r_b (U2.u - self.u)",
      |lc| lc + r_b.get_variable(),
      |lc| lc + CS::one() - self.u.get_variable(),
      |lc| lc + u_fold.get_variable() - self.u.get_variable(),
    );

    // Fold the IO:
    // X_fold = self.X + r_b (U2.X - self.X)
    let X_fold = AllocatedNum::alloc(cs.namespace(|| "allocate X_fold"), || {
      let X = self
        .X
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U2_X = U2.X.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(X + r_b * (U2_X - X))
    })?;
    cs.enforce(
      || "enforce X_fold - self.X = r_b (U2.X - self.X)",
      |lc| lc + r_b.get_variable(),
      |lc| lc + U2.X.get_variable() - self.X.get_variable(),
      |lc| lc + X_fold.get_variable() - self.X.get_variable(),
    );

    Ok(Self {
      comm_W: comm_W_fold.clone(),
      comm_E: comm_E_fold.clone(),
      T: T_out.clone(),
      u: u_fold,
      X: X_fold,
    })
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedNonnativePoint::conditionally_select(
      cs.namespace(|| "W = cond ? self.W : other.W"),
      &self.comm_W,
      &other.comm_W,
      condition,
    )?;

    let comm_E = AllocatedNonnativePoint::conditionally_select(
      cs.namespace(|| "E = cond ? self.E : other.E"),
      &self.comm_E,
      &other.comm_E,
      condition,
    )?;

    let T = conditionally_select(
      cs.namespace(|| "T = cond ? self.T : other.T"),
      &self.T,
      &other.T,
      condition,
    )?;

    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;

    let X = conditionally_select(
      cs.namespace(|| "X[0] = cond ? self.X[0] : other.X[0]"),
      &self.X,
      &other.X,
      condition,
    )?;

    Ok(Self {
      comm_W,
      comm_E,
      T,
      u,
      X,
    })
  }
}
