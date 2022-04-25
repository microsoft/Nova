use crate::{
  gadgets::{
    ecc::AllocatedPoint,
    utils::{
      alloc_bignat_constant, alloc_one, alloc_scalar_as_base, alloc_zero, conditionally_select,
      conditionally_select_bignat, le_bits_to_num,
    },
  },
  poseidon::{NovaPoseidonConstants, PoseidonROGadget},
  r1cs::{R1CSInstance, RelaxedR1CSInstance},
  traits::Group,
};
use bellperson::{
  gadgets::{boolean::Boolean, num::AllocatedNum, Assignment},
  ConstraintSystem, SynthesisError,
};
use bellperson_nonnative::{
  mp::bignat::BigNat,
  util::{convert::f_to_nat, num::Num},
};
use ff::{Field, PrimeField, PrimeFieldBits};

/// An Allocated R1CS Instance
#[derive(Clone)]
pub struct AllocatedR1CSInstance<G>
where
  G: Group,
{
  pub(crate) W: AllocatedPoint<G::Base>,
  pub(crate) X0: AllocatedNum<G::Base>,
  pub(crate) X1: AllocatedNum<G::Base>,
}

impl<G> AllocatedR1CSInstance<G>
where
  G: Group,
  <G as Group>::Base: PrimeField + PrimeFieldBits,
  <G as Group>::Scalar: PrimeFieldBits,
{
  /// Takes the r1cs instance and creates a new allocated r1cs instance
  pub fn alloc<CS: ConstraintSystem<<G as Group>::Base>>(
    mut cs: CS,
    u: Option<R1CSInstance<G>>,
  ) -> Result<Self, SynthesisError> {
    // Check that the incoming instance has exactly 2 io
    let W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate W"),
      u.get()
        .map_or(None, |u| Some(u.comm_W.comm.to_coordinates())),
    )?;

    let X0 = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "allocate X[0]"),
      u.get().map_or(None, |u| Some(u.X[0])),
    )?;
    let X1 = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "allocate X[1]"),
      u.get().map_or(None, |u| Some(u.X[1])),
    )?;

    Ok(AllocatedR1CSInstance { W, X0, X1 })
  }

  pub fn absorb_in_ro(&self, ro: &mut PoseidonROGadget<G::Base>) {
    ro.absorb(self.W.x.clone());
    ro.absorb(self.W.y.clone());
    ro.absorb(self.W.is_infinity.clone());
    ro.absorb(self.X0.clone());
    ro.absorb(self.X1.clone());
  }
}

/// An Allocated Relaxed R1CS Instance
pub struct AllocatedRelaxedR1CSInstance<G>
where
  G: Group,
  <G as Group>::Base: PrimeField + PrimeFieldBits,
  <G as Group>::Scalar: PrimeFieldBits,
{
  pub(crate) W: AllocatedPoint<G::Base>,
  pub(crate) E: AllocatedPoint<G::Base>,
  pub(crate) u: AllocatedNum<G::Base>,
  pub(crate) X0: BigNat<G::Base>,
  pub(crate) X1: BigNat<G::Base>,
}

impl<G> AllocatedRelaxedR1CSInstance<G>
where
  G: Group,
  <G as Group>::Base: PrimeField + PrimeFieldBits,
  <G as Group>::Scalar: PrimeFieldBits,
{
  /// Allocates the given RelaxedR1CSInstance as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<G as Group>::Base>>(
    mut cs: CS,
    inst: Option<RelaxedR1CSInstance<G>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    let W = AllocatedPoint::alloc(
      cs.namespace(|| "allocate W"),
      inst
        .get()
        .map_or(None, |inst| Some(inst.comm_W.comm.to_coordinates())),
    )?;

    let E = AllocatedPoint::alloc(
      cs.namespace(|| "allocate E"),
      inst
        .get()
        .map_or(None, |inst| Some(inst.comm_E.comm.to_coordinates())),
    )?;

    // u << |G::Base| despite the fact that u is a scalar.
    // So we parse all of its bytes as a G::Base element
    let u = alloc_scalar_as_base::<G, _>(
      cs.namespace(|| "allocate u"),
      inst.get().map_or(None, |inst| Some(inst.u)),
    )?;

    let X0 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[0]"),
      || Ok(f_to_nat(&inst.get()?.X[0])),
      limb_width,
      n_limbs,
    )?;

    let X1 = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate X[1]"),
      || Ok(f_to_nat(&inst.get()?.X[1])),
      limb_width,
      n_limbs,
    )?;

    Ok(AllocatedRelaxedR1CSInstance { W, E, u, X0, X1 })
  }

  /// Allocates the hardcoded default RelaxedR1CSInstance in the circuit.
  /// W = E = 0, u = 1, X0 = X1 = 0
  pub fn default<CS: ConstraintSystem<<G as Group>::Base>>(
    mut cs: CS,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let one = alloc_one(cs.namespace(|| "one"))?;

    let W_default = AllocatedPoint::new(zero.clone(), zero.clone(), one);
    let E_default = W_default.clone();

    let u_default = zero;

    let X0_default = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[0]"),
      || Ok(f_to_nat(&G::Scalar::zero())),
      limb_width,
      n_limbs,
    )?;

    let X1_default = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate x_default[1]"),
      || Ok(f_to_nat(&G::Scalar::zero())),
      limb_width,
      n_limbs,
    )?;
    Ok(AllocatedRelaxedR1CSInstance {
      W: W_default,
      E: E_default,
      u: u_default,
      X0: X0_default,
      X1: X1_default,
    })
  }

  pub fn absorb_in_ro<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    ro: &mut PoseidonROGadget<G::Base>,
  ) -> Result<(), SynthesisError> {
    ro.absorb(self.W.x.clone());
    ro.absorb(self.W.y.clone());
    ro.absorb(self.W.is_infinity.clone());
    ro.absorb(self.E.x.clone());
    ro.absorb(self.E.y.clone());
    ro.absorb(self.E.is_infinity.clone());
    ro.absorb(self.u.clone());

    // Analyze X0 as limbs
    let X0_bn = self
      .X0
      .as_limbs::<CS>()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb
          .as_sapling_allocated_num(cs.namespace(|| format!("convert limb {} of X_r[0] to num", i)))
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // absorb each of the limbs of X[0]
    for limb in X0_bn.into_iter() {
      ro.absorb(limb);
    }

    // Analyze X1 as limbs
    let X1_bn = self
      .X1
      .as_limbs::<CS>()
      .iter()
      .enumerate()
      .map(|(i, limb)| {
        limb
          .as_sapling_allocated_num(cs.namespace(|| format!("convert limb {} of X_r[1] to num", i)))
      })
      .collect::<Result<Vec<AllocatedNum<G::Base>>, _>>()?;

    // absorb each of the limbs of X[1]
    for limb in X1_bn.into_iter() {
      ro.absorb(limb);
    }

    Ok(())
  }

  /// Folds self with a relaxed r1cs instance and returns the result
  pub fn fold_with_r1cs<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    u: AllocatedR1CSInstance<G>,
    T: AllocatedPoint<G::Base>,
    poseidon_constants: NovaPoseidonConstants<G::Base>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<AllocatedRelaxedR1CSInstance<G>, SynthesisError> {
    // Compute r:
    let mut ro: PoseidonROGadget<G::Base> = PoseidonROGadget::new(poseidon_constants);
    u.absorb_in_ro(&mut ro);
    ro.absorb(T.x.clone());
    ro.absorb(T.y.clone());
    ro.absorb(T.is_infinity.clone());
    let r_bits = ro.get_challenge(cs.namespace(|| "r bits"))?;
    let r = le_bits_to_num(cs.namespace(|| "r"), r_bits.clone())?;

    // W_fold = self.W + r * u.W
    let rW = u.W.scalar_mul(cs.namespace(|| "r * u.W"), r_bits.clone())?;
    let W_fold = self.W.add(cs.namespace(|| "self.W + r * u.W"), &rW)?;

    // E_fold = self.E + r * T
    let rT = T.scalar_mul(cs.namespace(|| "r * T"), r_bits)?;
    let E_fold = self.E.add(cs.namespace(|| "self.E + r * T"), &rT)?;

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
      Num::from(r.clone()),
      limb_width,
      n_limbs,
    )?;

    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &G::get_order(),
      limb_width,
      n_limbs,
    )?;

    // Analyze X0 to bignat
    let X0_bn = BigNat::from_num(
      cs.namespace(|| "allocate X0_bn"),
      Num::from(u.X0.clone()),
      limb_width,
      n_limbs,
    )?;

    // Fold self.X[0] + r * X[0]
    let (_, r_0) = X0_bn.mult_mod(cs.namespace(|| "r*X[0]"), &r_bn, &m_bn)?;
    // add X_r[0]
    let r_new_0 = self.X0.add::<CS>(&r_0)?;
    // Now reduce
    let X0_fold = r_new_0.red_mod(cs.namespace(|| "reduce folded X[0]"), &m_bn)?;

    // Analyze X1 to bignat
    let X1_bn = BigNat::from_num(
      cs.namespace(|| "allocate X1_bn"),
      Num::from(u.X1.clone()),
      limb_width,
      n_limbs,
    )?;

    // Fold self.X[1] + r * X[1]
    let (_, r_1) = X1_bn.mult_mod(cs.namespace(|| "r*X[1]"), &r_bn, &m_bn)?;
    // add X_r[1]
    let r_new_1 = self.X1.add::<CS>(&r_1)?;
    // Now reduce
    let X1_fold = r_new_1.red_mod(cs.namespace(|| "reduce folded X[1]"), &m_bn)?;

    Ok(Self {
      W: W_fold,
      E: E_fold,
      u: u_fold,
      X0: X0_fold,
      X1: X1_fold,
    })
  }

  /// If the condition is true then returns this otherwise it returns the other
  pub fn conditionally_select<CS: ConstraintSystem<<G as Group>::Base>>(
    &self,
    mut cs: CS,
    other: AllocatedRelaxedR1CSInstance<G>,
    condition: &Boolean,
  ) -> Result<AllocatedRelaxedR1CSInstance<G>, SynthesisError> {
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
