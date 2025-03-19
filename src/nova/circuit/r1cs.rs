//! This module implements various gadgets necessary for folding R1CS types.
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  frontend::{num::AllocatedNum, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{
      bignat::BigNat,
      util::{f_to_nat, Num},
    },
    utils::{alloc_one, alloc_scalar_as_base, conditionally_select, conditionally_select_bignat},
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An Allocated R1CS Instance
#[derive(Clone)]
pub struct AllocatedR1CSInstance<E: Engine> {
  pub(crate) comm_W: AllocatedPoint<E>,
  pub(crate) X0: AllocatedNum<E::Base>,
  pub(crate) X1: AllocatedNum<E::Base>,
}

impl<E: Engine> AllocatedR1CSInstance<E> {
  /// Takes the r1cs instance and creates a new allocated r1cs instance
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    u: Option<&R1CSInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate comm_W"),
      u.map(|u| u.comm_W.to_coordinates()),
    )?;
    comm_W.check_on_curve(cs.namespace(|| "check comm_W on curve"))?;

    let X0 = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate X[0]"), u.map(|u| u.X[0]))?;
    let X1 = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate X[1]"), u.map(|u| u.X[1]))?;

    Ok(AllocatedR1CSInstance { comm_W, X0, X1 })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro(&self, ro: &mut E::ROCircuit) {
    self.comm_W.absorb_in_ro(ro);
    ro.absorb(&self.X0);
    ro.absorb(&self.X1);
  }
}

/// An Allocated Relaxed R1CS Instance
pub struct AllocatedRelaxedR1CSInstance<E: Engine> {
  pub(crate) W: AllocatedPoint<E>,
  pub(crate) E: AllocatedPoint<E>,
  pub(crate) u: AllocatedNum<E::Base>,
  pub(crate) X0: BigNat<E::Base>,
  pub(crate) X1: BigNat<E::Base>,
}

impl<E: Engine> AllocatedRelaxedR1CSInstance<E> {
  /// Allocates the given `RelaxedR1CSInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    inst: Option<&RelaxedR1CSInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.
    let W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate W"),
      inst.map(|inst| inst.comm_W.to_coordinates()),
    )?;

    let E = AllocatedPoint::alloc(
      cs.namespace(|| "allocate E"),
      inst.map(|inst| inst.comm_E.to_coordinates()),
    )?;

    // u << |E::Base| despite the fact that u is a scalar.
    // So we parse all of its bytes as a E::Base element
    let u = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate u"), inst.map(|inst| inst.u))?;

    // Allocate X0 and X1. If the input instance is None, then allocate default values 0.
    let X0 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[0]"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[0]))),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let X1 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[1]"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[1]))),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    Ok(AllocatedRelaxedR1CSInstance { W, E, u, X0, X1 })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, u = 0, X0 = X1 = 0
  pub fn default<CS: ConstraintSystem<E::Base>>(mut cs: CS) -> Result<Self, SynthesisError> {
    let W = AllocatedPoint::default(cs.namespace(|| "allocate W"))?;
    let E = W.clone();

    let u = W.x(); // In the default case, W.x = u = 0

    // X0 and X1 are allocated and in the honest prover case set to zero
    // If the prover is malicious, it can set to arbitrary values, but the resulting
    // relaxed R1CS instance with the checked default values of W, E, and u must still be satisfying
    let X0 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[0]"),
      || Ok(f_to_nat(&E::Scalar::ZERO)),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let X1 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[1]"),
      || Ok(f_to_nat(&E::Scalar::ZERO)),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    Ok(AllocatedRelaxedR1CSInstance { W, E, u, X0, X1 })
  }

  /// Allocates the R1CS Instance as a `RelaxedR1CSInstance` in the circuit.
  /// E = 0, u = 1
  pub fn from_r1cs_instance<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    inst: AllocatedR1CSInstance<E>,
  ) -> Result<Self, SynthesisError> {
    let E = AllocatedPoint::default(cs.namespace(|| "allocate default E"))?;

    let u = alloc_one(cs.namespace(|| "one"));

    let X0 = BigNat::from_num(
      cs.namespace(|| "allocate X0 from relaxed r1cs"),
      &Num::from(inst.X0),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let X1 = BigNat::from_num(
      cs.namespace(|| "allocate X1 from relaxed r1cs"),
      &Num::from(inst.X1),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    Ok(AllocatedRelaxedR1CSInstance {
      W: inst.comm_W,
      E,
      u,
      X0,
      X1,
    })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    ro: &mut E::ROCircuit,
  ) -> Result<(), SynthesisError> {
    self.W.absorb_in_ro(ro);
    self.E.absorb_in_ro(ro);

    ro.absorb(&self.u);

    // Analyze X0 as limbs
    let X0_bn = self
      .X0
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of X_r[0] to num")))
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // absorb each of the limbs of X[0]
    for limb in X0_bn {
      ro.absorb(&limb);
    }

    // Analyze X1 as limbs
    let X1_bn = self
      .X1
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of X_r[1] to num")))
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // absorb each of the limbs of X[1]
    for limb in X1_bn {
      ro.absorb(&limb);
    }

    Ok(())
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    other: &AllocatedRelaxedR1CSInstance<E>,
    condition: &Boolean,
  ) -> Result<AllocatedRelaxedR1CSInstance<E>, SynthesisError> {
    let W = AllocatedPoint::conditionally_select(
      cs.namespace(|| "W = cond ? self.W : other.W"),
      &self.W,
      &other.W,
      condition,
    )?;

    let E = AllocatedPoint::conditionally_select(
      cs.namespace(|| "E = cond ? self.E : other.E"),
      &self.E,
      &other.E,
      condition,
    )?;

    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;

    let X0 = conditionally_select_bignat(
      cs.namespace(|| "X[0] = cond ? self.X[0] : other.X[0]"),
      &self.X0,
      &other.X0,
      condition,
    )?;

    let X1 = conditionally_select_bignat(
      cs.namespace(|| "X[1] = cond ? self.X[1] : other.X[1]"),
      &self.X1,
      &other.X1,
      condition,
    )?;

    Ok(AllocatedRelaxedR1CSInstance { W, E, u, X0, X1 })
  }
}
