//! Circuit representation of Nova's NIFS for EC relation
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_CHALLENGE_BITS},
  frontend::{gadgets::Assignment, num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{bignat::BigNat, util::Num},
    utils::le_bits_to_num,
  },
  neutron::circuit::r1cs_ec::{AllocatedECInstance, AllocatedRelaxedECInstance},
  nova::nifs::NIFS,
  traits::{commitment::CommitmentTrait, Engine, ROCircuitTrait, ROConstantsCircuit},
};

/// An in-circuit representation of Nova's NIFS for EC relation
pub struct AllocatedNIFSEC<E: Engine> {
  pub(crate) comm_T: AllocatedPoint<E>,
}

impl<E: Engine> AllocatedNIFSEC<E> {
  /// Allocates the given `NIFS` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    nifs: Option<&NIFS<E>>,
  ) -> Result<Self, SynthesisError> {
    let comm_T = AllocatedPoint::alloc(
      cs.namespace(|| "allocate comm_T"),
      nifs.map(|nifs| nifs.comm_T.to_coordinates()),
    )?;
    comm_T.check_on_curve(cs.namespace(|| "check comm_T on curve"))?;

    Ok(Self { comm_T })
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

  /// verify the provided NIFS inside the circuit    
  pub fn verify<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Base>, // hash of R1CSShape of F'
    U: &AllocatedRelaxedECInstance<E>, // folded instance
    u: &AllocatedECInstance<E>,
    T: &AllocatedPoint<E>,
    ro_consts: ROConstantsCircuit<E>,
    m_bn: &BigNat<E::Base>,
  ) -> Result<(AllocatedRelaxedECInstance<E>, AllocatedNum<E::Base>), SynthesisError> {
    // Compute r:
    let mut ro = E::ROCircuit::new(ro_consts);
    ro.absorb(pp_digest);

    u.absorb_in_ro(cs.namespace(|| "u"), &mut ro)?;

    T.absorb_in_ro(&mut ro);

    let r_bits = ro.squeeze(cs.namespace(|| "r bits"), NUM_CHALLENGE_BITS)?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // W_fold = U.W + r * u.W
    let rW = u.W.scalar_mul(cs.namespace(|| "r * u.W"), &r_bits)?;
    let W_fold = U.W.add(cs.namespace(|| "U.W + r * u.W"), &rW)?;

    // E_fold = U.E + r * T
    let rT = self.comm_T.scalar_mul(cs.namespace(|| "r * T"), &r_bits)?;
    let E_fold = U.E.add(cs.namespace(|| "U.E + r * T"), &rT)?;

    // u_fold = u_r + r
    let u_fold = AllocatedNum::alloc(cs.namespace(|| "u_fold"), || {
      Ok(*U.u.get_value().get()? + r.get_value().get()?)
    })?;
    cs.enforce(
      || "Check u_fold",
      |lc| lc,
      |lc| lc,
      |lc| lc + u_fold.get_variable() - U.u.get_variable() - r.get_variable(),
    );

    // Fold the IO:
    // Analyze r into limbs
    let r_bn = BigNat::from_num(
      cs.namespace(|| "allocate r_bn"),
      &Num::from(r.clone()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // analyze u.r as bignat
    let u_r_bn = BigNat::from_num(
      cs.namespace(|| "allocate u_r_bn"),
      &Num::from(u.r.clone()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    let r_fold = U
      .r
      .fold_bn(cs.namespace(|| "fold r"), &u_r_bn, &r_bn, m_bn)?;

    let coords_fold = U
      .coords
      .iter()
      .zip(u.coords.iter())
      .enumerate()
      .map(|(i, (u1_x, u2_x))| {
        u1_x.fold_bn(cs.namespace(|| format!("fold X{i}")), u2_x, &r_bn, m_bn)
      })
      .collect::<Result<Vec<_>, _>>()?;

    let is_inf_fold = U
      .is_inf
      .iter()
      .zip(u.is_inf.iter())
      .enumerate()
      .map(|(i, (U, u))| Self::fold_scalar(cs.namespace(|| format!("fold is_inf{i}")), U, &r, u))
      .collect::<Result<Vec<_>, _>>()?;

    Ok((
      AllocatedRelaxedECInstance {
        W: W_fold,
        E: E_fold,
        u: u_fold,
        r: r_fold,
        coords: coords_fold,
        is_inf: is_inf_fold,
      },
      r,
    ))
  }
}
