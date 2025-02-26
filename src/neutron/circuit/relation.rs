//! This module implements various gadgets necessary for folding R1CS types with NeutronNova folding scheme.
use crate::{
  frontend::{num::AllocatedNum, AllocatedBit, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{
      bignat::BigNat,
      util::{f_to_nat, Num},
    },
    r1cs::AllocatedR1CSInstance,
    utils::conditionally_select_bignat,
  },
  neutron::relation::FoldedInstance,
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's FoldedInstance
/// In our context, public IO of circuits folded will have only two entries, so we have X0 and X1
pub struct AllocatedFoldedInstance<E: Engine> {
  pub(crate) comm_W: AllocatedPoint<E>,
  pub(crate) comm_E: AllocatedPoint<E>,
  pub(crate) T: BigNat<E::Base>,
  pub(crate) u: BigNat<E::Base>,
  pub(crate) X0: BigNat<E::Base>,
  pub(crate) X1: BigNat<E::Base>,
}

impl<E: Engine> AllocatedFoldedInstance<E> {
  /// Allocates the given `FoldedInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    inst: Option<&FoldedInstance<E>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.
    let comm_W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate W"),
      inst.map(|inst| inst.comm_W.to_coordinates()),
    )?;

    let comm_E = AllocatedPoint::alloc(
      cs.namespace(|| "allocate E"),
      inst.map(|inst| inst.comm_E.to_coordinates()),
    )?;

    let T = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate T"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.T))),
      limb_width,
      n_limbs,
    )?;

    let u = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate u"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.u))),
      limb_width,
      n_limbs,
    )?;

    // Allocate X0 and X1. If the input instance is None, then allocate default values 0.
    let X0 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[0]"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[0]))),
      limb_width,
      n_limbs,
    )?;

    let X1 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[1]"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.X[1]))),
      limb_width,
      n_limbs,
    )?;

    Ok(Self {
      comm_W,
      comm_E,
      T,
      u,
      X0,
      X1,
    })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, T = 0, u = 0, X0 = X1 = 0
  pub fn default<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedPoint::default(cs.namespace(|| "allocate W"))?;
    let comm_E = comm_W.clone();

    // Allocate T = 0. Similar to X0 and X1, we do not need to check that T is well-formed
    let T = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[0]"),
      || Ok(f_to_nat(&E::Scalar::ZERO)),
      limb_width,
      n_limbs,
    )?;

    let u = T.clone();

    // X0 and X1 are allocated and in the honest prover case set to zero
    // If the prover is malicious, it can set to arbitrary values, but the resulting
    // relaxed R1CS instance with the checked default values of W, E, and u must still be satisfying
    let X0 = T.clone();
    let X1 = T.clone();

    Ok(Self {
      comm_W,
      comm_E,
      T,
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
    // TODO: refactor this code to avoid duplication
    ro.absorb(&self.comm_W.x);
    ro.absorb(&self.comm_W.y);
    ro.absorb(&self.comm_W.is_infinity);
    ro.absorb(&self.comm_E.x);
    ro.absorb(&self.comm_E.y);
    ro.absorb(&self.comm_E.is_infinity);

    // Analyze T as limbs
    let T_bn = self
      .T
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of T to num")))
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // absorb each of the limbs of T
    for limb in T_bn {
      ro.absorb(&limb);
    }

    let u_bn = self
      .u
      .as_limbs()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of u to num")))
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    // absorb each of the limbs of u
    for limb in u_bn {
      ro.absorb(&limb);
    }

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

  /// Folds self with an r1cs instance and returns the result
  pub fn fold<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    U2: &AllocatedR1CSInstance<E>,
    comm_E: &AllocatedPoint<E>,
    r_b_bits: &[AllocatedBit],
    r_b_bn: &BigNat<E::Base>,
    T_out: &BigNat<E::Base>,
    m_bn: &BigNat<E::Base>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    // comm_W_fold = self.comm_W + r * (U2.comm_W - self.comm_W)
    let neg_self_comm_W = self.comm_W.negate(cs.namespace(|| "-self.comm_W"))?;
    let sub = U2
      .comm_W
      .add(cs.namespace(|| "U2.comm_W - self.comm_W"), &neg_self_comm_W)?;
    let r_sub = sub.scalar_mul(cs.namespace(|| "r * (U2.comm_W - self.comm_W)"), &r_b_bits)?;
    let comm_W_fold = self.comm_W.add(
      cs.namespace(|| "self.comm_W + r * (U2.comm_W - self.comm_W)"),
      &r_sub,
    )?;

    // comm_E_fold = self.comm_E + r * (comm_E - self.comm_E)
    let neg_self_comm_E = self.comm_E.negate(cs.namespace(|| "-self.comm_E"))?;
    let sub = comm_E.add(cs.namespace(|| "comm_E - self.comm_E"), &neg_self_comm_E)?;
    let r_sub = sub.scalar_mul(cs.namespace(|| "r * (comm_E - self.comm_E)"), &r_b_bits)?;
    let comm_E_fold = self.comm_E.add(
      cs.namespace(|| "self.comm_E + r * (comm_E - self.comm_E)"),
      &r_sub,
    )?;

    // u_fold = self.u - self.u * r_b_bn
    let (_, mul) = self
      .u
      .mult_mod(cs.namespace(|| "self.u * r_b_bn"), &r_b_bn, &m_bn)?;
    let res = self
      .u
      .sub(cs.namespace(|| "self.u - self.u * r_b_bn"), &mul)?;
    let u_fold = res.red_mod(cs.namespace(|| "reduce folded u"), m_bn)?;

    // Fold the IO:

    // Fold self.X[0] + r_b_bn * (X[0] - self.X[0])
    let U2_X0 = BigNat::from_num(
      cs.namespace(|| "allocate U2_X0"),
      &Num::from(U2.X0.clone()),
      limb_width,
      n_limbs,
    )?;
    let sub = U2_X0.sub(cs.namespace(|| "U2.X[0] - self.X[0]"), &self.X0)?;
    let (_, r_sub) = sub.mult_mod(
      cs.namespace(|| "r_b_bn * (X[0] - self.X[0])"),
      &r_b_bn,
      m_bn,
    )?;
    let res = self.X0.add(&r_sub)?;
    let X0_fold = res.red_mod(cs.namespace(|| "reduce folded X[0]"), m_bn)?;

    // Fold self.X[1] + r_b_bn * (X[1] - self.X[1])
    let U2_X1 = BigNat::from_num(
      cs.namespace(|| "allocate U2_X1"),
      &Num::from(U2.X1.clone()),
      limb_width,
      n_limbs,
    )?;
    let sub = U2_X1.sub(cs.namespace(|| "U2.X[1] - self.X[1]"), &self.X1)?;
    let (_, r_sub) = sub.mult_mod(
      cs.namespace(|| "r_b_bn * (X[1] - self.X[1])"),
      &r_b_bn,
      m_bn,
    )?;
    let res = self.X1.add(&r_sub)?;
    let X1_fold = res.red_mod(cs.namespace(|| "reduce folded X[1]"), m_bn)?;

    Ok(Self {
      comm_W: comm_W_fold,
      comm_E: comm_E_fold,
      T: T_out.clone(),
      u: u_fold,
      X0: X0_fold,
      X1: X1_fold,
    })
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let comm_W = AllocatedPoint::conditionally_select(
      cs.namespace(|| "W = cond ? self.W : other.W"),
      &self.comm_W,
      &other.comm_W,
      condition,
    )?;

    let comm_E = AllocatedPoint::conditionally_select(
      cs.namespace(|| "E = cond ? self.E : other.E"),
      &self.comm_E,
      &other.comm_E,
      condition,
    )?;

    let T = conditionally_select_bignat(
      cs.namespace(|| "T = cond ? self.T : other.T"),
      &self.T,
      &other.T,
      condition,
    )?;

    let u = conditionally_select_bignat(
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

    Ok(Self {
      comm_W,
      comm_E,
      T,
      u,
      X0,
      X1,
    })
  }
}
