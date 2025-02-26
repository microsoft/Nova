//! Circuit representation of a univariate polynomial
use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  gadgets::nonnative::{bignat::BigNat, util::f_to_nat},
  spartan::polys::univariate::UniPoly,
  traits::{Engine, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of `UniPoly` type
pub struct AllocatedUniPoly<E: Engine> {
  coeffs: Vec<BigNat<E::Base>>,
}

impl<E: Engine> AllocatedUniPoly<E> {
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

    Ok(Self { coeffs })
  }

  /// Returns the evaluation of the polynomial at 0
  pub fn eval_at_zero(&self) -> Result<BigNat<E::Base>, SynthesisError> {
    Ok(self.coeffs[0].clone())
  }

  /// Returns the evaluation of the polynomial at 1
  pub fn eval_at_one<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    m_bn: &BigNat<E::Base>,
  ) -> Result<BigNat<E::Base>, SynthesisError> {
    let mut eval = self.coeffs[0].clone();
    for coeff in self.coeffs.iter().skip(1) {
      eval = eval.add(&coeff)?;
    }
    eval = eval.red_mod(cs.namespace(|| "eval reduced"), &m_bn)?;
    Ok(eval)
  }

  /// Evaluate the polynomial at the provided point
  /// `m_bn` is the modulus of the scalar field that is emulated by the bignat
  pub fn evaluate<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    r: &BigNat<E::Base>,
    m_bn: &BigNat<E::Base>,
  ) -> Result<BigNat<E::Base>, SynthesisError> {
    let mut eval = self.coeffs[0].clone();
    let mut power = r.clone();
    for coeff in self.coeffs.iter().skip(1) {
      // eval = eval + power * coeff
      let (_, power_times_coeff) =
        power.mult_mod(cs.namespace(|| "power * coeff"), &coeff, &m_bn)?;
      eval = eval.add(&power_times_coeff)?;
      eval = eval.red_mod(cs.namespace(|| "eval reduced"), &m_bn)?;

      // power = power * r
      let (_, new_power) = power.mult_mod(cs.namespace(|| "power * r"), &r, &m_bn)?;
      power = new_power.red_mod(cs.namespace(|| "power reduced"), &m_bn)?;
    }
    Ok(eval)
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro<CS: ConstraintSystem<<E as Engine>::Base>>(
    &self,
    mut cs: CS,
    ro: &mut E::ROCircuit,
  ) -> Result<(), SynthesisError> {
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
    Ok(())
  }
}
