//! Circuit representation of a univariate polynomial
use crate::{
  constants::NUM_CHALLENGE_BITS,
  frontend::{num::AllocatedNum, Assignment, Boolean, ConstraintSystem, SynthesisError},
  gadgets::{
    ecc::AllocatedPoint,
    nonnative::{
      bignat::BigNat,
      util::{f_to_nat, Num},
    },
    r1cs::AllocatedR1CSInstance,
    utils::{
      alloc_bignat_constant, alloc_one, alloc_scalar_as_base, conditionally_select,
      conditionally_select_bignat, le_bits_to_num,
    },
  },
  neutron::relation::FoldedInstance,
  traits::{commitment::CommitmentTrait, Engine, Group, ROCircuitTrait, ROConstantsCircuit},
};
use ff::Field;

/// An in-circuit representation of NeutronNova's NIFS
pub struct AllocatedUniPoly<E: Engine> {
  coeffs: Vec<BigNat<E::Base>>,
}

impl<E: Engine> AllocatedRelaxedR1CSInstance<E> {
  /// Allocates the given `UniPoly` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Base>>(
    mut cs: CS,
    degree: usize,
    poly: Option<&UniPoly<E::Scalar>>,
    limb_width: usize,
    n_limbs: usize,
  ) -> Result<Self, SynthesisError> {
    // we allocate degree + 1 coefficients as BigNat
    let coeffs = (0..degree + 1)
      .map(|i| {
        BigNat::alloc_from_nat(
          cs.namespace(|| format!("allocate coeff[{}]", i)),
          || {
            Ok(f_to_nat(
              &poly.map_or(E::Scalar::ZERO, |poly| poly.coeffs[i]),
            ))
          },
          limb_width,
          n_limbs,
        )
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(AllocatedUniPoly { coeffs })
  }

  /// Returns the evaluation of the polynomial at 0
  pub fn eval_at_zero(&self) -> Result<BigNat<E::Base>, SynthesisError> {
    Ok(self.coeffs[0].clone())
  }

  /// Returns the evaluation of the polynomial at 1
  pub fn eval_at_one(&self) -> Result<BigNat<E::Base>, SynthesisError> {
    let mut eval = self.coeffs[0].clone();
    for coeff in self.coeffs.iter().skip(1) {
      eval = eval.add(cs.namespace(|| "eval + coeff"), &coeff)?;
    }
    eval = eval.red_mod(cs.namespace(|| "eval reduced"))?;
    Ok(eval)
  }

  /// Evaluate the polynomial at the provided point
  /// `m_bn` is the modulus of the scalar field that is emulated by the bignat
  pub fn evaluate(
    &self,
    r: &BigNat<E::Base>,
    m_bn: &BigNat<E::Base>,
  ) -> Result<BigNat<E::Base>, SynthesisError> {
    let mut eval = self.coeffs[0].clone();
    let mut power = r.clone();
    for coeff in self.coeffs.iter().skip(1) {
      // eval = eval + power * coeff
      let (_, power_times_coeff) =
        power.mult_mod(cs.namespace(|| "power * coeff"), &coeff, &m_bn)?;
      eval = eval.add(cs.namespace(|| "eval + power * coeff"), &power_times_coeff)?;
      eval = eval.red_mod(cs.namespace(|| "eval reduced"), &m_bn)?;
    }
    Ok(eval)
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro(&self, ro: &mut E::ROCircuit) {
    for coeff in &self.coeffs {
      let coeff_bn = coeff
        .as_limbs()
        .iter()
        .enumerate()
        .map(|(i, limb)| {
          limb.as_allocated_num(cs.namespace(|| format!("convert limb {i} of coeff to num")))
        })
        .collect::<Result<Vec<AllocatedNum<E::Base>>, _>>()?;

      // absorb each of the limbs of T
      for limb in coeff_bn {
        ro.absorb(&limb);
      }
    }
  }
}
