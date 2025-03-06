//! Circuit representation of a univariate polynomial
use crate::{
  frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
  spartan::polys::univariate::UniPoly,
  traits::{Engine, ROCircuitTrait},
};
use ff::Field;

/// An in-circuit representation of `UniPoly` type
pub struct AllocatedUniPoly<E: Engine> {
  coeffs: Vec<AllocatedNum<E::Scalar>>,
}

impl<E: Engine> AllocatedUniPoly<E> {
  /// Allocates the given `UniPoly` as a witness of the circuit
  pub fn alloc<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    mut cs: CS,
    degree: usize,
    poly: Option<&UniPoly<E::Scalar>>,
  ) -> Result<Self, SynthesisError> {
    // we allocate degree + 1 coefficients as BigNat
    let coeffs = (0..degree + 1)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("allocate coeff[{}]", i)), || {
          Ok(poly.map_or(E::Scalar::ZERO, |poly| poly.coeffs[i]))
        })
      })
      .collect::<Result<Vec<_>, _>>()?;

    Ok(Self { coeffs })
  }

  /// checks if poly(0) + poly(1) = c
  pub fn check_poly_zero_poly_one_with<CS: ConstraintSystem<E::Scalar>>(
    &self,
    mut cs: CS,
    c: &AllocatedNum<E::Scalar>,
  ) -> Result<(), SynthesisError> {
    // eval_at_0 = constant term and eval_at_1 = sum of all coefficients
    // eval_at_0 + eval_at_1 = constant_term + sum of all coefficients including the constant term
    cs.enforce(
      || "eval at 0 + eval at 1 = c",
      |lc| lc + c.get_variable(),
      |lc| lc + CS::one(),
      |lc| {
        self
          .coeffs
          .iter()
          .fold(lc + self.coeffs[0].get_variable(), |lc, v| {
            lc + v.get_variable()
          })
      },
    );

    Ok(())
  }

  /// Evaluate the polynomial at the provided point
  pub fn evaluate<CS: ConstraintSystem<<E as Engine>::Scalar>>(
    &self,
    mut cs: CS,
    r: &AllocatedNum<E::Scalar>,
  ) -> Result<AllocatedNum<E::Scalar>, SynthesisError> {
    let mut acc = self.coeffs[0].clone();
    let mut power = r.clone();
    for (i, coeff) in self.coeffs.iter().skip(1).enumerate() {
      // acc_new = acc_old + power * coeff
      // allocate acc_new
      let acc_new = AllocatedNum::alloc(cs.namespace(|| format!("{i} allocate acc_new")), || {
        let acc_old = acc.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let power = power.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        let coeff = coeff.get_value().ok_or(SynthesisError::AssignmentMissing)?;
        Ok(acc_old + power * coeff)
      })?;

      // check that acc_new - acc_old = power * coeff
      cs.enforce(
        || format!("{i} enforce acc_new - acc_old = power * coeff"),
        |lc| lc + power.get_variable(),
        |lc| lc + coeff.get_variable(),
        |lc| lc + acc_new.get_variable() - acc.get_variable(),
      );

      // power_new = power_old * r
      let power_new =
        AllocatedNum::alloc(cs.namespace(|| format!("{i} allocate power_new")), || {
          let power_old = power.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          let r = r.get_value().ok_or(SynthesisError::AssignmentMissing)?;
          Ok(power_old * r)
        })?;
      cs.enforce(
        || format!("{i} enforce power_new = power_old * r"),
        |lc| lc + power.get_variable(),
        |lc| lc + r.get_variable(),
        |lc| lc + power_new.get_variable(),
      );

      power = power_new;
      acc = acc_new;
    }
    Ok(acc)
  }

  /// Absorb the provided instance in the RO
  pub fn absorb_in_ro(&self, ro: &mut E::RO2Circuit) {
    for coeff in self.coeffs.iter() {
      ro.absorb(coeff);
    }
  }
}
