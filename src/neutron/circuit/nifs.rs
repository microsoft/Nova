//! Circuit representation of NeutronNova's NIFS
use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{bignat::BigNat, util::Num},
    r1cs::AllocatedR1CSInstance,
    utils::{alloc_bignat_constant, le_bits_to_num},
  },
  neutron::{
    circuit::{relation::AllocatedFoldedInstance, univariate::AllocatedUniPoly},
    nifs::NIFS,
  },
  traits::{commitment::CommitmentTrait, Engine, Group, ROCircuitTrait, ROConstantsCircuit},
};

/// An in-circuit representation of NeutronNova's NIFS
pub struct AllocatedNIFS<E: Engine> {
  pub(crate) comm_E: AllocatedPoint<E>,
  pub(crate) poly: AllocatedUniPoly<E>,
}

impl<E: Engine> AllocatedNIFS<E> {
  /// Allocates the given `NIFS` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    nifs: Option<&NIFS<E>>,
    degree: usize,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_E = AllocatedPoint::alloc(
      cs.namespace(|| "allocate comm_E"),
      nifs.map(|nifs| nifs.comm_E.to_coordinates()),
    )?;
    comm_E.check_on_curve(cs.namespace(|| "check comm_E on curve"))?;

    // Allocate the polynomial
    let poly = AllocatedUniPoly::alloc(
      cs.namespace(|| "allocate poly"),
      degree,
      nifs.map(|nifs| &nifs.poly),
      limb_width,
      n_limbs,
    )?;

    Ok(Self { comm_E, poly })
  }

  /// verify the provided NIFS inside the circuit
  pub fn verify<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    vk: &AllocatedNum<E::Base>,      // verifier key
    U1: &AllocatedFoldedInstance<E>, // folded instance
    U2: &AllocatedR1CSInstance<E>,
    eq_rho_r_b_inv: &BigNat<E::Base>, // untrusted advice
    ro_consts: ROConstantsCircuit<E>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<AllocatedFoldedInstance<E>, SynthesisError> {
    // Compute r:
    let mut ro = E::ROCircuit::new(ro_consts);
    ro.absorb(vk);

    // running instance `U1` does not need to absorbed since U2.X[0] = Hash(vk, U1, i, z0, zi)
    U2.absorb_in_ro(&mut ro);

    // generate a challenge for the eq polynomial
    let _tau = ro.squeeze(cs.namespace(|| "tau"), NUM_CHALLENGE_BITS);

    // TODO: We will check the power-check instance contains tau as public IO later

    // absorb the commitment in the NIFS
    ro.absorb(&self.comm_E.x);
    ro.absorb(&self.comm_E.y);
    ro.absorb(&self.comm_E.is_infinity);

    // squeeze a challenge from the RO
    let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS)?;
    let _rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;

    // TODO: assemble rho as a num and check that poly(0) + poly(1) = T

    // absorb poly in the RO
    self
      .poly
      .absorb_in_ro(cs.namespace(|| "absorb poly"), &mut ro)?;

    // squeeze a challenge
    let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS)?;
    let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;
    let r_b_bn = BigNat::from_num(
      cs.namespace(|| "allocate r_bn"),
      &Num::from(r_b),
      limb_width,
      n_limbs,
    )?;

    // compute the sum-check polynomial's evaluations at r_b
    // let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    // check that eq_rho_r_b * eq_rho_r_b_inv = 1

    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &E::GE::group_params().2,
      limb_width,
      n_limbs,
    )?;

    // let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();
    let T_out = {
      let eval = {
        self
          .poly
          .evaluate(cs.namespace(|| "eval"), &r_b_bn, &m_bn)?
      };
      let (_, res) = eval.mult_mod(cs.namespace(|| "mult_mod"), &eq_rho_r_b_inv, &m_bn)?;
      res.red_mod(cs.namespace(|| "reduce"), &m_bn)?
    };

    // let U = U1.fold(U2, &self.comm_E, &r_b, &T_out)?;
    let U = U1.fold(
      cs.namespace(|| "fold"),
      U2,
      &self.comm_E,
      &r_b_bits.as_slice(),
      &r_b_bn,
      &T_out,
      &m_bn,
      limb_width,
      n_limbs,
    )?;

    // return the folded instance and witness
    Ok(U)
  }
}
