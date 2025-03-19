//! Circuit representation of Nova's NIFS
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_CHALLENGE_BITS},
  frontend::{gadgets::Assignment, num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{bignat::BigNat, util::Num},
    utils::{alloc_bignat_constant, le_bits_to_num},
  },
  nova::{
    circuit::r1cs::{AllocatedR1CSInstance, AllocatedRelaxedR1CSInstance},
    nifs::NIFS,
  },
  traits::{commitment::CommitmentTrait, Engine, Group, ROCircuitTrait, ROConstantsCircuit},
};

/// An in-circuit representation of NeutronNova's NIFS
pub struct AllocatedNIFS<E: Engine> {
  pub(crate) comm_T: AllocatedPoint<E>,
}

impl<E: Engine> AllocatedNIFS<E> {
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

  /// verify the provided NIFS inside the circuit
  pub fn verify<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Base>,   // verifier key
    U: &AllocatedRelaxedR1CSInstance<E>, // folded instance
    u: &AllocatedR1CSInstance<E>,
    ro_consts: ROConstantsCircuit<E>,
  ) -> Result<AllocatedRelaxedR1CSInstance<E>, SynthesisError> {
    // Compute r:
    let mut ro = E::ROCircuit::new(ro_consts);
    ro.absorb(pp_digest);

    // running instance `U` does not need to absorbed since u.X[0] = Hash(params, U, i, z0, zi)
    u.absorb_in_ro(&mut ro);
    self.comm_T.absorb_in_ro(&mut ro);

    let r_bits = ro.squeeze(cs.namespace(|| "r bits"), NUM_CHALLENGE_BITS)?;
    let r = le_bits_to_num(cs.namespace(|| "r"), &r_bits)?;

    // W_fold = U.W + r * u.W
    let rW = u.comm_W.scalar_mul(cs.namespace(|| "r * u.W"), &r_bits)?;
    let W_fold = U.W.add(cs.namespace(|| "U.W + r * u.W"), &rW)?;

    // E_fold = self.E + r * T
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
      &Num::from(r),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &E::GE::group_params().2,
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Analyze X0 to bignat
    let X0_bn = BigNat::from_num(
      cs.namespace(|| "allocate X0_bn"),
      &Num::from(u.X0.clone()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Fold U.X[0] + r * X[0]
    let (_, r_0) = X0_bn.mult_mod(cs.namespace(|| "r*X[0]"), &r_bn, &m_bn)?;
    // add X_r[0]
    let r_new_0 = U.X0.add(&r_0)?;
    // Now reduce
    let X0_fold = r_new_0.red_mod(cs.namespace(|| "reduce folded X[0]"), &m_bn)?;

    // Analyze X1 to bignat
    let X1_bn = BigNat::from_num(
      cs.namespace(|| "allocate X1_bn"),
      &Num::from(u.X1.clone()),
      BN_LIMB_WIDTH,
      BN_N_LIMBS,
    )?;

    // Fold U.X[1] + r * X[1]
    let (_, r_1) = X1_bn.mult_mod(cs.namespace(|| "r*X[1]"), &r_bn, &m_bn)?;
    // add X_r[1]
    let r_new_1 = U.X1.add(&r_1)?;
    // Now reduce
    let X1_fold = r_new_1.red_mod(cs.namespace(|| "reduce folded X[1]"), &m_bn)?;

    Ok(AllocatedRelaxedR1CSInstance {
      W: W_fold,
      E: E_fold,
      u: u_fold,
      X0: X0_fold,
      X1: X1_fold,
    })
  }
}
