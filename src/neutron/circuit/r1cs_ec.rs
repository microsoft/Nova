//! This module implements R1CS gadgets for R1CS corresponding to the ec circuit
use crate::{
  constants::NUM_CHALLENGE_BITS,
  gadgets::nonnative::{
    bignat::BigNat,
    util::{absorb_bignat_in_ro, f_to_nat, Num},
  },
  gadgets::{
    ecc::AllocatedPoint,
    utils::{
      alloc_scalar_as_base, conditionally_select, conditionally_select_bignat, le_bits_to_num,
    },
  },
  traits::{commitment::CommitmentTrait, Engine, Engine, ROCircuitTrait, ROConstantsCircuit},
};
use ff::Field;
use frontend::gadgets::{
  boolean::Boolean, num::AllocatedNum, Assignment, ConstraintSystem, SynthesisError,
};

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
    C0: &AllocatedNonnativePoint<E>,
    C1: &AllocatedNonnativePoint<E>,
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
      alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate r"), inst.map(|inst| inst.U.X[0]))?;

    let x2 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x2"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.U.X[7]))),
      limb_width,
      n_limbs,
    )?;

    let y2 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate y2"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.U.X[8]))),
      limb_width,
      n_limbs,
    )?;

    let is_inf_2 = alloc_scalar_as_base::<E, _>(
      cs.namespace(|| "allocate is_inf_2"),
      inst.map(|inst| inst.U.X[9]),
    )?;

    let coords = vec![
      C0.x.clone(),
      C0.y.clone(),
      C1.x.clone(),
      C1.y.clone(),
      x2,
      y2,
    ];

    let is_inf = vec![C0.is_inf.clone(), C1.is_inf.clone(), is_inf_2];

    Ok(Self { W, r, coords, is_inf })
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    ro: &mut E::ROCircuit,
  ) -> Result<(), SynthesisError> {
    for (i, w) in self.W.iter().enumerate() {
      w.absorb_in_ro(cs.namespace(|| format!("W {i}")), ro)?;
    }

    // we only need to absorb the output point
    // we absorb r in the folding scheme, which came from another RO whose input contains all the
    // other points in the public IO
    absorb_bignat_in_ro::<E, _>(&self.X[4], cs.namespace(|| "X5"), ro)?;
    absorb_bignat_in_ro::<E, _>(&self.X[5], cs.namespace(|| "X6"), ro)?;
    ro.absorb(&self.is_inf[2]);

    Ok(())
  }
}

/// An Allocated Relaxed R1CS Instance
pub struct AllocatedRelaxedECInstance<E: Engine> {
  pub(crate) W: Vec<AllocatedPoint<E>>,
  pub(crate) E: Vec<AllocatedPoint<E>>,
  pub(crate) u: AllocatedNum<E::Base>,

  pub(crate) r: BigNat<E::Base>,
  pub(crate) X: Vec<BigNat<E::Base>>,
  pub(crate) is_inf: Vec<AllocatedNum<E::Base>>,
}

impl<E: Engine> AllocatedRelaxedECInstance<E> {
  /// Allocates the given `RelaxedR1CSInstance` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    inst: Option<&RelaxedR1CSInstance<E>>,
  ) -> Result<Self, SynthesisError> {
    // We do not need to check that W or E are well-formed (e.g., on the curve) as we do a hash check
    // in the Nova augmented circuit, which ensures that the relaxed instance
    // came from a prior iteration of Nova.
    let W = (0..dims.0)
      .map(|i| {
        AllocatedPoint::alloc(
          cs.namespace(|| format!("allocate W {}", i)),
          inst.map(|inst| inst.comm_W[i].to_coordinates()),
        )
      })
      .collect::<Result<Vec<AllocatedPoint<E>>, _>>()?;

    let E = (0..dims.0)
      .map(|i| {
        AllocatedPoint::alloc(
          cs.namespace(|| format!("allocate E {}", i)),
          inst.map(|inst| inst.comm_E[i].to_coordinates()),
        )
      })
      .collect::<Result<Vec<AllocatedPoint<E>>, _>>()?;

    // u << |E::Base| despite the fact that u is a scalar.
    // So we parse all of its bytes as a E::Base element
    let u = alloc_scalar_as_base::<E, _>(cs.namespace(|| "allocate u"), inst.map(|inst| inst.U.u))?;

    let r = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate r"),
      || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.U.X[0]))),
      limb_width,
      n_limbs,
    )?;

    let X = (1..7)
      .map(|i| {
        BigNat::alloc_from_nat(
          cs.namespace(|| format!("allocate X{i}")),
          || Ok(f_to_nat(&inst.map_or(E::Scalar::ZERO, |inst| inst.U.X[i]))),
          limb_width,
          n_limbs,
        )
      })
      .collect::<Result<Vec<BigNat<E::Base>>, _>>()?;

    let is_inf = (0..3)
      .map(|i| {
        alloc_scalar_as_base::<E, _>(
          cs.namespace(|| format!("allocate is_inf{i}")),
          inst.map(|inst| inst.U.X[7 + i]),
        )
      })
      .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

    Ok(Self {
      W,
      E,
      u,
      r,
      X,
      is_inf,
    })
  }

  /// Allocates the hardcoded default `RelaxedR1CSInstance` in the circuit.
  /// W = E = 0, u = 0, X = 0
  pub fn default<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    zero: &BigNat<E::Base>,
    dims: (usize, usize),
  ) -> Result<Self, SynthesisError> {
    let W = {
      let p = AllocatedPoint::default(cs.namespace(|| "allocate W"))?;
      vec![p; dims.0]
    };
    let E = W.clone();

    let u = W[0].x.clone(); // In the default case, W.x = u = 0

    // X are allocated and in the honest prover case set to zero
    // If the prover is malicious, it can set to arbitrary values, but the resulting
    // relaxed R1CS instance with the the checked default values of W, E, and u must still be satisfying
    let r = zero.clone();

    let X = vec![zero.clone(); 6];

    let is_inf = vec![u.clone(); 3];

    Ok(Self {
      W,
      E,
      u,
      r,
      X,
      is_inf,
    })
  }

  fn fold_scalar<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    U: &AllocatedNum<E::Base>,
    r: &AllocatedNum<E::Base>,
    u: &AllocatedNum<E::Base>,
  ) -> Result<AllocatedNum<E::Base>, SynthesisError> {
    // u_fold = U + r * u
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "u_fold"), || {
      Ok(*U.get_value().get()? + *r.get_value().get()? * *u.get_value().get()?)
    })?;
    cs.enforce(
      || "Check u_fold",
      |lc| lc + r.get_variable(),
      |lc| lc + u.get_variable(),
      |lc| lc + u_fold.get_variable() - U.get_variable(),
    );

    Ok(u_fold)
  }

  /// Folds self with a relaxed r1cs instance and returns the result
  pub fn fold_with_r1cs<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    params: &AllocatedNum<E::Base>, // hash of R1CSShape of F'
    u: &AllocatedECInstance<E>,
    T: &[AllocatedPoint<E>],
    ro_consts: ROConstantsCircuit<E>,
    limb_width: usize,
    n_limbs: usize,
    m_bn: &BigNat<E::Base>,
    dims: (usize, usize),
  ) -> Result<(Self, AllocatedNum<E::Base>), SynthesisError> {
    // Compute r:
    let mut ro = E::ROCircuit::new(ro_consts, num_fe_sm_for_ro::<E>());
    ro.absorb(params);

    u.absorb_in_ro(cs.namespace(|| "u"), &mut ro)?;

    for (i, Ti) in T.iter().enumerate() {
      Ti.absorb_in_ro(cs.namespace(|| format!("T{i}")), &mut ro)?;
    }

    let r_bits = ro.squeeze(cs.namespace(|| "r bits"), NUM_CHALLENGE_BITS, false)?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // W_fold = self.W + r * u.W
    let mut W_fold = Vec::with_capacity(dims.0);
    for i in 0..dims.0 {
      let rW_i = u.W[i].scalar_mul(cs.namespace(|| format!("r * u.W {i}")), &r_bits)?;
      W_fold.push(self.W[i].add(cs.namespace(|| format!("self.W {i} + r * u.W {i}")), &rW_i)?);
    }

    // E_fold = self.E + r * T
    let mut E_fold = Vec::with_capacity(dims.0);
    for (i, T_i) in T.iter().enumerate().take(dims.0) {
      let rT_i = T_i.scalar_mul(cs.namespace(|| format!("r * T {i}")), &r_bits)?;
      E_fold.push(self.E[i].add(cs.namespace(|| format!("self.E {i} + r * T {i}")), &rT_i)?);
    }

    // u_fold = u_r + r
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "u_fold"), || {
      Ok(*self.u.get_value().get()? + r.get_value().get()?)
    })?;
    cs.enforce(
      || "Check u_fold",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_fold.get_variable() - self.u.get_variable() - r.get_variable(),
    );

    // Fold the IO:
    // Analyze r into limbs
    let r_bn = BigNat::from_num(
      cs.namespace(|| "allocate r_bn"),
      &Num::from(r.clone()),
      limb_width,
      n_limbs,
    )?;

    // analyze u.r as bignat
    let u_r_bn = BigNat::from_num(
      cs.namespace(|| "allocate u_r_bn"),
      &Num::from(u.r.clone()),
      limb_width,
      n_limbs,
    )?;

    let r_fold = self
      .r
      .fold_bn(cs.namespace(|| "fold r"), &u_r_bn, &r_bn, m_bn)?;

    let X = self
      .X
      .iter()
      .zip(u.X.iter())
      .enumerate()
      .map(|(i, (u1_x, u2_x))| {
        u1_x.fold_bn(cs.namespace(|| format!("fold X{i}")), u2_x, &r_bn, m_bn)
      })
      .collect::<Result<Vec<_>, _>>()?;

    let is_inf = self
      .is_inf
      .iter()
      .zip(u.is_inf.iter())
      .enumerate()
      .map(|(i, (U, u))| Self::fold_scalar(cs.namespace(|| format!("fold is_inf{i}")), U, &r, u))
      .collect::<Result<Vec<_>, _>>()?;

    Ok((
      Self {
        W: W_fold,
        E: E_fold,
        u: u_fold,
        r: r_fold,
        X,
        is_inf,
      },
      r,
    ))
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    other: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let W = self
      .W
      .iter()
      .zip(other.W.iter())
      .enumerate()
      .map(|(i, (w1, w2))| {
        AllocatedPoint::conditionally_select(cs.namespace(|| format!("W {i}")), w1, w2, condition)
      })
      .collect::<Result<Vec<_>, _>>()?;

    let E = self
      .E
      .iter()
      .zip(other.E.iter())
      .enumerate()
      .map(|(i, (e1, e2))| {
        AllocatedPoint::conditionally_select(cs.namespace(|| format!("E {i}")), e1, e2, condition)
      })
      .collect::<Result<Vec<_>, _>>()?;

    let u = conditionally_select(
      cs.namespace(|| "u = cond ? self.u : other.u"),
      &self.u,
      &other.u,
      condition,
    )?;

    let r = conditionally_select_bignat(cs.namespace(|| "r"), &self.r, &other.r, condition)?;

    let X = self
      .X
      .iter()
      .zip(other.X.iter())
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
      X,
      is_inf,
    })
  }
}
