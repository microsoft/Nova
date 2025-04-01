//! This module implements R1CS gadgets for R1CS corresponding to the ec circuit
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  frontend::{
    gadgets::{boolean::Boolean, num::AllocatedNum},
    ConstraintSystem, SynthesisError,
  },
  gadgets::nonnative::{
    bignat::BigNat,
    util::{absorb_bignat_in_ro, f_to_nat},
  },
  gadgets::{
    ecc::AllocatedPoint,
    utils::{alloc_scalar_as_base, conditionally_select, conditionally_select_bignat},
  },
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An Allocated EC Instance
#[derive(Clone)]
pub struct AllocatedECInstance<E: Engine> {
  pub(crate) W: AllocatedPoint<E>,

  // public IO
  pub(crate) r: AllocatedNum<E::Base>,
  pub(crate) coords: Vec<BigNat<E::Base>>,
  pub(crate) is_inf: Vec<AllocatedNum<E::Base>>,
}

impl<E: Engine> AllocatedECInstance<E> {
  /// Takes the r1cs instance and creates a new allocated EC instance
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    inst: Option<&R1CSInstance<E>>,
    C0: (&BigNat<E::Base>, &BigNat<E::Base>, &AllocatedNum<E::Base>),
    C1: (&BigNat<E::Base>, &BigNat<E::Base>, &AllocatedNum<E::Base>),
  ) -> Result<Self, SynthesisError> {
    let W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate W {}"),
      inst.map(|inst| inst.comm_W.to_coordinates()),
    )?;
    W.check_on_curve(cs.namespace(|| "check W on curve"))?;

    // inst.X = [r, x_0, y_0, is_inf_0, x_1, y_1, is_inf_1, x_2, y_2, is_inf_2]
    //
    // We allocate the following variables
    // r
    // coords = [x_0, y_0, x_1, y_1, x_2, y_2]
    // is_inf = [is_inf_0, is_inf_1, is_inf_2]
    // But, we are given x_0, y_0, x_1, y_1, x_2, y_2 as BigNat, and is_inf_0 and is_inf_1 as AllocatedNum
    // We only need to allocate r, x_2, y_2, and is_inf_2
    let r =
      alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate r"), inst.map(|inst| inst.X[0]))?;

    let x2: BigNat<E::Base> = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x2"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[7]))),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let y2: BigNat<E::Base> = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate y2"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[8]))),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let is_inf_2 = alloc_scalar_as_base::<E, _>(
      cs.namespace(|| "allocate is_inf_2"),
      inst.map(|inst| inst.X[9]),
    )?;

    let coords = vec![
      C0.0.clone(),
      C0.1.clone(),
      C1.0.clone(),
      C1.1.clone(),
      x2,
      y2,
    ];

    let is_inf = vec![C0.2.clone(), C1.2.clone(), is_inf_2];

    Ok(Self {
      W,
      r,
      coords,
      is_inf,
    })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    ro: &mut E::ROCircuit,
  ) -> Result<(), SynthesisError> {
    self.W.absorb_in_ro(ro);

    // we only need to absorb the output point
    // we absorb r in the folding scheme, which came from another RO whose input contains all the
    // other points in the public IO
    absorb_bignat_in_ro::<E, _>(&self.coords[4], cs.namespace(|| "X5"), ro)?;
    absorb_bignat_in_ro::<E, _>(&self.coords[5], cs.namespace(|| "X6"), ro)?;
    ro.absorb(&self.is_inf[2]);

    Ok(())
  }
}

/// An Allocated Relaxed R1CS Instance
pub struct AllocatedRelaxedECInstance<E: Engine> {
  pub(crate) W: AllocatedPoint<E>,
  pub(crate) E: AllocatedPoint<E>,
  pub(crate) u: AllocatedNum<E::Base>,

  pub(crate) r: BigNat<E::Base>,
  pub(crate) coords: Vec<BigNat<E::Base>>,
  pub(crate) is_inf: Vec<AllocatedNum<E::Base>>,
}

impl<E: Engine> AllocatedRelaxedECInstance<E> {
  /// Allocates the given `RelaxedECInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
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

    let r = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate r"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[0]))),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let coords = (1..7)
      .map(|i| {
        BigNat::alloc_from_nat(
          cs.namespace(|| format!("allocate X{i}")),
          || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[i]))),
          BN_LIMB_WIDTH,
          BN_N_LIMBS,
        )
      })
      .collect::<Result<Vec<BigNat<E::Base>>, _>>()?;

    let is_inf = (7..10)
      .map(|i| {
        alloc_scalar_as_base::<E, _>(
          cs.namespace(|| format!("allocate is_inf{i}")),
          inst.map(|inst| inst.X[i]),
        )
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    Ok(Self {
      W,
      E,
      u,
      r,
      coords,
      is_inf,
    })
  }

  /// Allocates the hardcoded default `RelaxedECInstance` in the circuit.
  /// W = E = 0, u = 0, X = 0
  pub fn default<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    zero: &BigNat<E::Base>,
  ) -> Result<Self, SynthesisError> {
    let W = AllocatedPoint::default(cs.namespace(|| "allocate W"))?;
    let E = W.clone();

    let u = W.x().clone(); // In the default case, W.x = u = 0

    // X are allocated and in the honest prover case set to zero
    // If the prover is malicious, it can set to arbitrary values, but the resulting
    // relaxed R1CS instance with the the checked default values of W, E, and u must still be satisfying
    let r = zero.clone();

    let coords = vec![zero.clone(); 6];

    let is_inf = vec![u.clone(); 3];

    Ok(Self {
      W,
      E,
      u,
      r,
      coords,
      is_inf,
    })
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let W =
      AllocatedPoint::conditionally_select(cs.namespace(|| "W"), &self.W, &other.W, condition)?;
    let E =
      AllocatedPoint::conditionally_select(cs.namespace(|| "E"), &self.E, &other.E, condition)?;

    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;

    let r = conditionally_select_bignat(cs.namespace(|| "r"), &self.r, &other.r, condition)?;

    let coords = self
      .coords
      .iter()
      .zip(other.coords.iter())
      .enumerate()
      .map(|(i, (x1, x2))| {
        conditionally_select_bignat(cs.namespace(|| format!("X {i}")), x1, x2, condition)
      })
      .collect::<Result<Vec<_>, _>>()?;

    let is_inf = self
      .is_inf
      .iter()
      .zip(other.is_inf.iter())
      .enumerate()
      .map(|(i, (x1, x2))| {
        conditionally_select(cs.namespace(|| format!("is_inf {i}")), x1, x2, condition)
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Self {
      W,
      E,
      u,
      r,
      coords,
      is_inf,
    })
  }
}
