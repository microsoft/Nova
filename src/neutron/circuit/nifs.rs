//! Circuit representation of NeutronNova's NIFS
use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{
      bignat::BigNat,
      util::{f_to_nat, Num},
    },
    r1cs::AllocatedR1CSInstance,
    utils::{alloc_bignat_constant, le_bits_to_num},
  },
  neutron::{
    circuit::{relation::AllocatedFoldedInstance, univariate::AllocatedUniPoly},
    nifs::NIFS,
  },
  traits::{commitment::CommitmentTrait, Engine, Group, ROCircuitTrait, ROConstantsCircuit},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's NIFS
pub struct AllocatedNIFS<E: Engine> {
  pub(crate) comm_E: AllocatedPoint<E>,
  pub(crate) poly: AllocatedUniPoly<E>,
  pub(crate) eq_rho_r_b_inv: BigNat<E::Base>,
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

    // Allocate the inverse
    let eq_rho_r_b_inv = BigNat::alloc_from_nat(
      cs.namespace(|| "allocate eq_rho_r_b_inv"),
      || {
        Ok(f_to_nat(
          &nifs.map_or(E::Scalar::ONE, |nifs| nifs.eq_rho_r_b_inv),
        ))
      },
      limb_width,
      n_limbs,
    )?;

    Ok(Self {
      comm_E,
      poly,
      eq_rho_r_b_inv,
    })
  }

  /// verify the provided NIFS inside the circuit
  pub fn verify<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    vk: &AllocatedNum<E::Base>,      // verifier key
    U1: &AllocatedFoldedInstance<E>, // folded instance
    U2: &AllocatedR1CSInstance<E>,
    ro_consts: ROConstantsCircuit<E>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<AllocatedFoldedInstance<E>, SynthesisError> {
    // Allocate the order of the non-native field as a constant
    let m_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc m"),
      &E::GE::group_params().2,
      limb_width,
      n_limbs,
    )?;

    let one_bn = alloc_bignat_constant(
      cs.namespace(|| "alloc one"),
      &f_to_nat(&E::Scalar::ONE),
      limb_width,
      n_limbs,
    )?;

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
    let rho = le_bits_to_num(cs.namespace(|| "rho"), &rho_bits)?;
    let rho_bn = BigNat::from_num(
      cs.namespace(|| "allocate rho_bn"),
      &Num::from(rho),
      limb_width,
      n_limbs,
    )?;

    let T = {
      // T = (1-rho) * U1.T + rho * U2.T, but U2.T = 0
      let (_, mul) = U1
        .T
        .mult_mod(cs.namespace(|| "mul_mod for T"), &rho_bn, &m_bn)?;
      let sub = U1.T.sub(cs.namespace(|| "sub U1.T - mul"), &mul)?;
      sub.red_mod(cs.namespace(|| "reduce T"), &m_bn)?
    };

    let poly_at_zero = self.poly.eval_at_zero()?;
    let poly_at_one = self
      .poly
      .eval_at_one(cs.namespace(|| "poly_at_one"), &m_bn)?;
    let expected = poly_at_zero.add(&poly_at_one)?;
    expected.equal_when_carried(cs.namespace(|| "check T"), &T)?;

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
    let eq_rho_r_b = {
      // 1 - rho - r_b + 2 * rho * r_b
      let (_, mul) = rho_bn.mult_mod(cs.namespace(|| "mul rho * r_b"), &r_b_bn, &m_bn)?;
      let two_mul = mul.add(&mul)?;
      let sub = two_mul.sub(cs.namespace(|| "sub 2*rho - rho"), &rho_bn)?;
      let sub = sub.sub(cs.namespace(|| "sub 2*rho - r_b - r_b"), &r_b_bn)?;
      sub.add(&one_bn)?
    };

    // check that eq_rho_r_b * eq_rho_r_b_inv = 1
    // TODO: we need to check this only when we are not in the base case of the primary circuit
    let (_, _mul) = eq_rho_r_b.mult_mod(
      cs.namespace(|| "mult eq_rho_r_b * eq_rho_r_b_inv "),
      &self.eq_rho_r_b_inv,
      &m_bn,
    )?;
    //mul.equal_when_carried(cs.namespace(|| "check eq_rho_r_b"), &one_bn)?;

    // let T_out = self.poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();
    let T_out = {
      let eval = {
        self
          .poly
          .evaluate(cs.namespace(|| "eval"), &r_b_bn, &m_bn)?
      };
      let (_, res) = eval.mult_mod(cs.namespace(|| "mult_mod"), &self.eq_rho_r_b_inv, &m_bn)?;
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
