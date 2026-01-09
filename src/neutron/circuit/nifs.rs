//! Circuit representation of NeutronNova's NIFS
use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{ecc::AllocatedNonnativePoint, utils::le_bits_to_num},
  neutron::{
    circuit::{
      r1cs::AllocatedNonnativeR1CSInstance, relation::AllocatedFoldedInstance,
      univariate::AllocatedUniPoly,
    },
    nifs::NIFS,
  },
  traits::{commitment::CommitmentTrait, Engine, RO2ConstantsCircuit, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's NIFS
pub struct AllocatedNIFS<E: Engine> {
  pub(crate) comm_E: AllocatedNonnativePoint<E>,
  pub(crate) poly: AllocatedUniPoly<E>,
}

impl<E: Engine> AllocatedNIFS<E> {
  /// Allocates the given `NIFS` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    nifs: Option<&NIFS<E>>,
    degree: usize,
  ) -> Result<Self, SynthesisError> {
    let comm_E = AllocatedNonnativePoint::alloc(
      cs.namespace(|| "allocate comm_E"),
      nifs.map(|nifs| nifs.comm_E.to_coordinates()),
    )?;

    // Allocate the polynomial
    let poly = AllocatedUniPoly::alloc(
      cs.namespace(|| "allocate poly"),
      degree,
      nifs.map(|nifs| &nifs.poly),
    )?;

    Ok(Self { comm_E, poly })
  }

  /// verify the provided NIFS inside the circuit
  pub fn verify<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    pp_digest: &AllocatedNum<E::Scalar>, // verifier key
    U1: &AllocatedFoldedInstance<E>,     // folded instance
    U2: &AllocatedNonnativeR1CSInstance<E>,
    comm_W_fold: &AllocatedNonnativePoint<E>, // untrusted hint
    comm_E_fold: &AllocatedNonnativePoint<E>, // untrusted hint
    ro_consts: RO2ConstantsCircuit<E>,
  ) -> Result<AllocatedFoldedInstance<E>, SynthesisError> {
    // Compute r:
    let mut ro = E::RO2Circuit::new(ro_consts);
    ro.absorb(pp_digest);

    // running instance `U1` does not need to absorbed since U2.X[0] = Hash(vk, U1, i, z0, zi)
    U2.absorb_in_ro(cs.namespace(|| "absorb U2"), &mut ro)?;

    // generate a challenge for the eq polynomial
    let _tau = ro.squeeze(cs.namespace(|| "tau"), NUM_CHALLENGE_BITS, false);

    // TODO: We will check the power-check instance contains tau as public IO later

    // absorb the commitment in the NIFS
    self
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb comm_E"), &mut ro)?;

    // squeeze a challenge from the RO
    let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS, false)?;
    let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;

    // T = (1-rho) * U1.T + rho * U2.T, but U2.T = 0
    let T = AllocatedNum::alloc(cs.namespace(|| "allocate T"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let U1_T = U1.T.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok(U1_T * (E::Scalar::ONE - rho))
    })?;
    cs.enforce(
      || "enforce T = (1-rho) * U1.T",
      |lc| lc + U1.T.get_variable(),
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + T.get_variable(),
    );

    self
      .poly
      .check_poly_zero_poly_one_with(cs.namespace(|| "poly_at_zero + poly_at_one = T"), &T)?;

    // absorb poly in the RO
    self.poly.absorb_in_ro(&mut ro);

    // squeeze a challenge
    let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS, false)?;
    let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;

    // compute the sum-check polynomial's evaluations at r_b
    // let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let eq_rho_r_b_one = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b_one"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b))
    })?;
    cs.enforce(
      || "check eq_rho_r_b_one = (1 - rho) * (1 - r_b)",
      |lc| lc + CS::one() - rho.get_variable(),
      |lc| lc + CS::one() - r_b.get_variable(),
      |lc| lc + eq_rho_r_b_one.get_variable(),
    );

    let eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b)
    })?;

    // check eq_rho_r_b = (1 - rho) * (1 - r_b) + rho * r_b
    cs.enforce(
      || "check eq_rho_r_b = (1 - rho) * (1 - r_b) + rho * r_b",
      |lc| lc + rho.get_variable(),
      |lc| lc + r_b.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable() - eq_rho_r_b_one.get_variable(),
    );

    // let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();
    let eval = { self.poly.evaluate(cs.namespace(|| "eval"), &r_b)? };
    let T_out = AllocatedNum::alloc(cs.namespace(|| "allocate T_out"), || {
      let eval = eval.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let eq_rho_r_b_inv = eq_rho_r_b
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?
        .invert()
        .unwrap();
      Ok(eval * eq_rho_r_b_inv)
    })?;
    cs.enforce(
      || "enforce T_out * eq_rho_r_b = eval",
      |lc| lc + T_out.get_variable(),
      |lc| lc + eq_rho_r_b.get_variable(),
      |lc| lc + eval.get_variable(),
    );

    // let U = U1.fold(U2, &self.comm_E, &r_b, &T_out)?;
    let U = U1.fold(
      cs.namespace(|| "fold"),
      U2,
      &r_b,
      &T_out,
      comm_W_fold,
      comm_E_fold,
    )?;

    // return the folded instance and witness
    Ok(U)
  }
}
