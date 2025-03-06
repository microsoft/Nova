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
  pub(crate) eq_rho_r_b_inv: AllocatedNum<E::Scalar>,
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

    // Allocate the inverse
    let eq_rho_r_b_inv = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b_inv"), || {
      Ok(nifs.map_or(E::Scalar::ONE, |nifs| nifs.eq_rho_r_b_inv))
    })?;

    Ok(Self {
      comm_E,
      poly,
      eq_rho_r_b_inv,
    })
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
    let _tau = ro.squeeze(cs.namespace(|| "tau"), NUM_CHALLENGE_BITS);

    // TODO: We will check the power-check instance contains tau as public IO later

    // absorb the commitment in the NIFS
    self
      .comm_E
      .absorb_in_ro(cs.namespace(|| "absorb comm_E"), &mut ro)?;

    // squeeze a challenge from the RO
    let rho_bits = ro.squeeze(cs.namespace(|| "rho_bits"), NUM_CHALLENGE_BITS)?;
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

    let poly_at_zero = self.poly.eval_at_zero()?;
    let poly_at_one = self.poly.eval_at_one(cs.namespace(|| "poly_at_one"))?;

    // enforce that poly_at_zero + poly_at_one = T
    // TODO: avoid doing this
    cs.enforce(
      || "enforce poly_at_zero + poly_at_one = T",
      |lc| lc + poly_at_zero.get_variable() + poly_at_one.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + T.get_variable(),
    );

    // absorb poly in the RO
    self.poly.absorb_in_ro(&mut ro);

    // squeeze a challenge
    let r_b_bits = ro.squeeze(cs.namespace(|| "r_b_bits"), NUM_CHALLENGE_BITS)?;
    let r_b = le_bits_to_num(cs.namespace(|| "r_b"), &r_b_bits)?;

    // compute the sum-check polynomial's evaluations at r_b
    // let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let _eq_rho_r_b = AllocatedNum::alloc(cs.namespace(|| "allocate eq_rho_r_b"), || {
      let rho = rho.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let r_b = r_b.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      Ok((E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b)
    });

    // check eq_rho_rb = 1 - rho - r_b + 2 * rho * r_b
    // TODO: check that eq_rho_rb-1 - rho + r_b = 2 * rho * r_b

    // let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();
    let eval = { self.poly.evaluate(cs.namespace(|| "eval"), &r_b)? };
    let T_out = AllocatedNum::alloc(cs.namespace(|| "allocate T_out"), || {
      let eval = eval.get_value().ok_or(SynthesisError::AssignmentMissing)?;
      let eq_rho_r_b_inv = self
        .eq_rho_r_b_inv
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?;
      Ok(eval * eq_rho_r_b_inv)
    })?;
    cs.enforce(
      || "enforce T_out = poly(r_b) * eq_rho_r_b_inv",
      |lc| lc + eval.get_variable(),
      |lc| lc + self.eq_rho_r_b_inv.get_variable(),
      |lc| lc + T_out.get_variable(),
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
