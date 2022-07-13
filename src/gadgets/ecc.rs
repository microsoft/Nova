//! This module implements various elliptic curve gadgets
#![allow(non_snake_case)]
use crate::gadgets::utils::{
  alloc_num_equals, alloc_one, alloc_zero, conditionally_select, conditionally_select2,
  select_num_or_one, select_num_or_zero, select_num_or_zero2, select_one_or_diff2,
  select_one_or_num2, select_zero_or_num2,
};
use bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::AllocatedNum,
    Assignment,
  },
  ConstraintSystem, SynthesisError,
};
use ff::PrimeField;

/// AllocatedPoint provides an elliptic curve abstraction inside a circuit.
#[derive(Clone)]
pub struct AllocatedPoint<Fp>
where
  Fp: PrimeField,
{
  pub(crate) x: AllocatedNum<Fp>,
  pub(crate) y: AllocatedNum<Fp>,
  pub(crate) is_infinity: AllocatedNum<Fp>,
}

impl<Fp> AllocatedPoint<Fp>
where
  Fp: PrimeField,
{
  /// Allocates a new point on the curve using coordinates provided by `coords`.
  /// If coords = None, it allocates the default infinity point
  pub fn alloc<CS>(mut cs: CS, coords: Option<(Fp, Fp, bool)>) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Fp>,
  {
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(coords.map_or(Fp::zero(), |c| c.0))
    })?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(coords.map_or(Fp::zero(), |c| c.1))
    })?;
    let is_infinity = AllocatedNum::alloc(cs.namespace(|| "is_infinity"), || {
      Ok(if coords.map_or(true, |c| c.2) {
        Fp::one()
      } else {
        Fp::zero()
      })
    })?;
    cs.enforce(
      || "is_infinity is bit",
      |lc| lc + is_infinity.get_variable(),
      |lc| lc + CS::one() - is_infinity.get_variable(),
      |lc| lc,
    );

    Ok(AllocatedPoint { x, y, is_infinity })
  }

  /// Allocates a default point on the curve.
  pub fn default<CS>(mut cs: CS) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Fp>,
  {
    let zero = alloc_zero(cs.namespace(|| "zero"))?;
    let one = alloc_one(cs.namespace(|| "one"))?;

    Ok(AllocatedPoint {
      x: zero.clone(),
      y: zero,
      is_infinity: one,
    })
  }

  /// Returns coordinates associated with the point.
  pub fn get_coordinates(&self) -> (&AllocatedNum<Fp>, &AllocatedNum<Fp>, &AllocatedNum<Fp>) {
    (&self.x, &self.y, &self.is_infinity)
  }

  // Allocate a random point. Only used for testing
  #[cfg(test)]
  pub fn random_vartime<CS: ConstraintSystem<Fp>>(mut cs: CS) -> Result<Self, SynthesisError> {
    loop {
      let x = Fp::random(&mut OsRng);
      let y = (x * x * x + Fp::one() + Fp::one() + Fp::one() + Fp::one() + Fp::one()).sqrt();
      if y.is_some().unwrap_u8() == 1 {
        let x_alloc = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(x))?;
        let y_alloc = AllocatedNum::alloc(cs.namespace(|| "y"), || Ok(y.unwrap()))?;
        let is_infinity = alloc_zero(cs.namespace(|| "Is Infinity"))?;
        return Ok(Self {
          x: x_alloc,
          y: y_alloc,
          is_infinity,
        });
      }
    }
  }

  /// Make the point io
  #[cfg(test)]
  pub fn inputize<CS: ConstraintSystem<Fp>>(&self, mut cs: CS) -> Result<(), SynthesisError> {
    let _ = self.x.inputize(cs.namespace(|| "Input point.x"));
    let _ = self.y.inputize(cs.namespace(|| "Input point.y"));
    let _ = self
      .is_infinity
      .inputize(cs.namespace(|| "Input point.is_infinity"));
    Ok(())
  }

  /// Add two points (may be equal)
  pub fn add<CS: ConstraintSystem<Fp>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<Fp>,
  ) -> Result<Self, SynthesisError> {
    // Compute boolean equal indicating if self = other

    let equal_x = alloc_num_equals(
      cs.namespace(|| "check self.x == other.x"),
      &self.x,
      &other.x,
    )?;

    let equal_y = alloc_num_equals(
      cs.namespace(|| "check self.y == other.y"),
      &self.y,
      &other.y,
    )?;

    // Compute the result of the addition and the result of double self
    let result_from_add = self.add_internal(cs.namespace(|| "add internal"), other, &equal_x)?;
    let result_from_double = self.double(cs.namespace(|| "double"))?;

    // Output:
    // If (self == other) {
    //  return double(self)
    // }else {
    //  if (self.x == other.x){
    //      return infinity [negation]
    //  } else {
    //      return add(self, other)
    //  }
    // }
    let result_for_equal_x = AllocatedPoint::select_point_or_infinity(
      cs.namespace(|| "equal_y ? result_from_double : infinity"),
      &result_from_double,
      &Boolean::from(equal_y),
    )?;

    AllocatedPoint::conditionally_select(
      cs.namespace(|| "equal ? result_from_double : result_from_add"),
      &result_for_equal_x,
      &result_from_add,
      &Boolean::from(equal_x),
    )
  }

  /// Adds other point to this point and returns the result. Assumes that the two points are
  /// different and that both other.is_infinity and this.is_infinty are bits
  pub fn add_internal<CS: ConstraintSystem<Fp>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<Fp>,
    equal_x: &AllocatedBit,
  ) -> Result<Self, SynthesisError> {
    //************************************************************************/
    // lambda = (other.y - self.y) * (other.x - self.x).invert().unwrap();
    //************************************************************************/
    // First compute (other.x - self.x).inverse()
    // If either self or other are the infinity point or self.x = other.x  then compute bogus values
    // Specifically,
    // x_diff = self != inf && other != inf && self.x == other.x ? (other.x - self.x) : 1

    // Compute self.is_infinity OR other.is_infinity =
    // NOT(NOT(self.is_ifninity) AND NOT(other.is_infinity))
    let at_least_one_inf = AllocatedNum::alloc(cs.namespace(|| "at least one inf"), || {
      Ok(
        Fp::one()
          - (Fp::one() - *self.is_infinity.get_value().get()?)
            * (Fp::one() - *other.is_infinity.get_value().get()?),
      )
    })?;
    cs.enforce(
      || "1 - at least one inf = (1-self.is_infinity) * (1-other.is_infinity)",
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + CS::one() - other.is_infinity.get_variable(),
      |lc| lc + CS::one() - at_least_one_inf.get_variable(),
    );

    // Now compute x_diff_is_actual = at_least_one_inf OR equal_x
    let x_diff_is_actual =
      AllocatedNum::alloc(cs.namespace(|| "allocate x_diff_is_actual"), || {
        Ok(if *equal_x.get_value().get()? {
          Fp::one()
        } else {
          *at_least_one_inf.get_value().get()?
        })
      })?;
    cs.enforce(
      || "1 - x_diff_is_actual = (1-equal_x) * (1-at_least_one_inf)",
      |lc| lc + CS::one() - at_least_one_inf.get_variable(),
      |lc| lc + CS::one() - equal_x.get_variable(),
      |lc| lc + CS::one() - x_diff_is_actual.get_variable(),
    );

    // x_diff = 1 if either self.is_infinity or other.is_infinity or self.x = other.x else self.x -
    // other.x
    let x_diff = select_one_or_diff2(
      cs.namespace(|| "Compute x_diff"),
      &other.x,
      &self.x,
      &x_diff_is_actual,
    )?;

    let x_diff_inv = AllocatedNum::alloc(cs.namespace(|| "x diff inverse"), || {
      if *x_diff_is_actual.get_value().get()? == Fp::one() {
        // Set to default
        Ok(Fp::one())
      } else {
        // Set to the actual inverse
        let inv = (*other.x.get_value().get()? - *self.x.get_value().get()?).invert();
        if inv.is_some().unwrap_u8() == 1 {
          Ok(inv.unwrap())
        } else {
          Err(SynthesisError::DivisionByZero)
        }
      }
    })?;

    cs.enforce(
      || "Check inverse",
      |lc| lc + x_diff.get_variable(),
      |lc| lc + x_diff_inv.get_variable(),
      |lc| lc + CS::one(),
    );

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      Ok(
        (*other.y.get_value().get()? - *self.y.get_value().get()?)
          * x_diff_inv.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "Check that lambda is correct",
      |lc| lc + other.y.get_variable() - self.y.get_variable(),
      |lc| lc + x_diff_inv.get_variable(),
      |lc| lc + lambda.get_variable(),
    );

    //************************************************************************/
    // x = lambda * lambda - self.x - other.x;
    //************************************************************************/
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(
        *lambda.get_value().get()? * lambda.get_value().get()?
          - *self.x.get_value().get()?
          - *other.x.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + other.x.get_variable(),
    );

    //************************************************************************/
    // y = lambda * (self.x - x) - self.y;
    //************************************************************************/
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(
        *lambda.get_value().get()? * (*self.x.get_value().get()? - *x.get_value().get()?)
          - *self.y.get_value().get()?,
      )
    })?;

    cs.enforce(
      || "Check that y is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    //************************************************************************/
    // We only return the computed x, y if neither of the points is infinity and self.x != other.y
    // if self.is_infinity return other.clone()
    // elif other.is_infinity return self.clone()
    // elif self.x == other.x return infinity
    // Otherwise return the computed points.
    //************************************************************************/
    // Now compute the output x

    let x1 = conditionally_select2(
      cs.namespace(|| "x1 = other.is_infinity ? self.x : x"),
      &self.x,
      &x,
      &other.is_infinity,
    )?;

    let x = conditionally_select2(
      cs.namespace(|| "x = self.is_infinity ? other.x : x1"),
      &other.x,
      &x1,
      &self.is_infinity,
    )?;

    let y1 = conditionally_select2(
      cs.namespace(|| "y1 = other.is_infinity ? self.y : y"),
      &self.y,
      &y,
      &other.is_infinity,
    )?;

    let y = conditionally_select2(
      cs.namespace(|| "y = self.is_infinity ? other.y : y1"),
      &other.y,
      &y1,
      &self.is_infinity,
    )?;

    let is_infinity1 = select_num_or_zero2(
      cs.namespace(|| "is_infinity1 = other.is_infinity ? self.is_infinity : 0"),
      &self.is_infinity,
      &other.is_infinity,
    )?;

    let is_infinity = conditionally_select2(
      cs.namespace(|| "is_infinity = self.is_infinity ? other.is_infinity : is_infinity1"),
      &other.is_infinity,
      &is_infinity1,
      &self.is_infinity,
    )?;

    Ok(Self { x, y, is_infinity })
  }

  /// Doubles the supplied point.
  pub fn double<CS: ConstraintSystem<Fp>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    //*************************************************************/
    // lambda = (Fp::one() + Fp::one() + Fp::one())
    //  * self.x
    //  * self.x
    //  * ((Fp::one() + Fp::one()) * self.y).invert().unwrap();
    /*************************************************************/

    // Compute tmp = (Fp::one() + Fp::one())* self.y ? self != inf : 1
    let tmp_actual = AllocatedNum::alloc(cs.namespace(|| "tmp_actual"), || {
      Ok(*self.y.get_value().get()? + *self.y.get_value().get()?)
    })?;
    cs.enforce(
      || "check tmp_actual",
      |lc| lc + CS::one() + CS::one(),
      |lc| lc + self.y.get_variable(),
      |lc| lc + tmp_actual.get_variable(),
    );

    let tmp = select_one_or_num2(cs.namespace(|| "tmp"), &tmp_actual, &self.is_infinity)?;

    // Compute inv = tmp.invert
    let tmp_inv = AllocatedNum::alloc(cs.namespace(|| "tmp inverse"), || {
      if *self.is_infinity.get_value().get()? == Fp::one() {
        // Return default value 1
        Ok(Fp::one())
      } else {
        // Return the actual inverse
        let inv = (*tmp.get_value().get()?).invert();
        if inv.is_some().unwrap_u8() == 1 {
          Ok(inv.unwrap())
        } else {
          Err(SynthesisError::DivisionByZero)
        }
      }
    })?;
    cs.enforce(
      || "Check inverse",
      |lc| lc + tmp.get_variable(),
      |lc| lc + tmp_inv.get_variable(),
      |lc| lc + CS::one(),
    );

    // Now compute lambda as (Fp::one() + Fp::one + Fp::one()) * self.x * self.x * tmp_inv
    let prod_1 = AllocatedNum::alloc(cs.namespace(|| "alloc prod 1"), || {
      Ok(*tmp_inv.get_value().get()? * self.x.get_value().get()?)
    })?;
    cs.enforce(
      || "Check prod 1",
      |lc| lc + self.x.get_variable(),
      |lc| lc + tmp_inv.get_variable(),
      |lc| lc + prod_1.get_variable(),
    );

    let prod_2 = AllocatedNum::alloc(cs.namespace(|| "alloc prod 2"), || {
      Ok(*prod_1.get_value().get()? * self.x.get_value().get()?)
    })?;
    cs.enforce(
      || "Check prod 2",
      |lc| lc + self.x.get_variable(),
      |lc| lc + prod_1.get_variable(),
      |lc| lc + prod_2.get_variable(),
    );

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      Ok(*prod_2.get_value().get()? * (Fp::one() + Fp::one() + Fp::one()))
    })?;
    cs.enforce(
      || "Check lambda",
      |lc| lc + CS::one() + CS::one() + CS::one(),
      |lc| lc + prod_2.get_variable(),
      |lc| lc + lambda.get_variable(),
    );

    /*************************************************************/
    //          x = lambda * lambda - self.x - self.x;
    /*************************************************************/

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(
        ((*lambda.get_value().get()?) * (*lambda.get_value().get()?))
          - *self.x.get_value().get()?
          - self.x.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "Check x",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + self.x.get_variable() + self.x.get_variable(),
    );

    /*************************************************************/
    //        y = lambda * (self.x - x) - self.y;
    /*************************************************************/

    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(
        (*lambda.get_value().get()?) * (*self.x.get_value().get()? - x.get_value().get()?)
          - self.y.get_value().get()?,
      )
    })?;
    cs.enforce(
      || "Check y",
      |lc| lc + lambda.get_variable(),
      |lc| lc + self.x.get_variable() - x.get_variable(),
      |lc| lc + y.get_variable() + self.y.get_variable(),
    );

    /*************************************************************/
    // Only return the computed x and y if the point is not infinity
    /*************************************************************/

    // x
    let x = select_zero_or_num2(cs.namespace(|| "final x"), &x, &self.is_infinity)?;

    // y
    let y = select_zero_or_num2(cs.namespace(|| "final y"), &y, &self.is_infinity)?;

    // is_infinity
    let is_infinity = self.is_infinity.clone();

    Ok(Self { x, y, is_infinity })
  }

  /// A gadget for scalar multiplication.
  pub fn scalar_mul<CS: ConstraintSystem<Fp>>(
    &self,
    mut cs: CS,
    scalar: Vec<AllocatedBit>,
  ) -> Result<Self, SynthesisError> {
    let mut res = Self::default(cs.namespace(|| "res"))?;
    for i in (0..scalar.len()).rev() {
      /*************************************************************/
      //  res = res.double();
      /*************************************************************/

      res = res.double(cs.namespace(|| format!("{}: double", i)))?;

      /*************************************************************/
      //  if scalar[i] {
      //    res = self.add(&res);
      //  }
      /*************************************************************/
      let self_and_res = self.add(cs.namespace(|| format!("{}: add", i)), &res)?;
      res = Self::conditionally_select(
        cs.namespace(|| format!("{}: Update res", i)),
        &self_and_res,
        &res,
        &Boolean::from(scalar[i].clone()),
      )?;
    }
    Ok(res)
  }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<Fp>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;

    let y = conditionally_select(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;

    let is_infinity = conditionally_select(
      cs.namespace(|| "select is_infinity"),
      &a.is_infinity,
      &b.is_infinity,
      condition,
    )?;

    Ok(Self { x, y, is_infinity })
  }

  /// If condition outputs a otherwise infinity
  pub fn select_point_or_infinity<CS: ConstraintSystem<Fp>>(
    mut cs: CS,
    a: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = select_num_or_zero(cs.namespace(|| "select x"), &a.x, condition)?;

    let y = select_num_or_zero(cs.namespace(|| "select y"), &a.y, condition)?;

    let is_infinity = select_num_or_one(
      cs.namespace(|| "select is_infinity"),
      &a.is_infinity,
      condition,
    )?;

    Ok(Self { x, y, is_infinity })
  }
}

#[cfg(test)]
use ff::PrimeFieldBits;
#[cfg(test)]
use rand::rngs::OsRng;
#[cfg(test)]
use std::marker::PhantomData;

#[cfg(test)]
#[derive(Debug, Clone)]
pub struct Point<Fp, Fq>
where
  Fp: PrimeField,
  Fq: PrimeField + PrimeFieldBits,
{
  x: Fp,
  y: Fp,
  is_infinity: bool,
  _p: PhantomData<Fq>,
}

#[cfg(test)]
impl<Fp, Fq> Point<Fp, Fq>
where
  Fp: PrimeField,
  Fq: PrimeField + PrimeFieldBits,
{
  pub fn new(x: Fp, y: Fp, is_infinity: bool) -> Self {
    Self {
      x,
      y,
      is_infinity,
      _p: Default::default(),
    }
  }

  pub fn random_vartime() -> Self {
    loop {
      let x = Fp::random(&mut OsRng);
      let y = (x * x * x + Fp::one() + Fp::one() + Fp::one() + Fp::one() + Fp::one()).sqrt();
      if y.is_some().unwrap_u8() == 1 {
        return Self {
          x,
          y: y.unwrap(),
          is_infinity: false,
          _p: Default::default(),
        };
      }
    }
  }

  /// Add any two points
  pub fn add(&self, other: &Point<Fp, Fq>) -> Self {
    if self.x == other.x {
      // If self == other then call double
      if self.y == other.y {
        self.double()
      } else {
        // if self.x == other.x and self.y != other.y then return infinity
        Self {
          x: Fp::zero(),
          y: Fp::zero(),
          is_infinity: true,
          _p: Default::default(),
        }
      }
    } else {
      self.add_internal(other)
    }
  }

  /// Add two different points
  pub fn add_internal(&self, other: &Point<Fp, Fq>) -> Self {
    if self.is_infinity {
      return other.clone();
    }

    if other.is_infinity {
      return self.clone();
    }

    let lambda = (other.y - self.y) * (other.x - self.x).invert().unwrap();
    let x = lambda * lambda - self.x - other.x;
    let y = lambda * (self.x - x) - self.y;
    Self {
      x,
      y,
      is_infinity: false,
      _p: Default::default(),
    }
  }

  pub fn double(&self) -> Self {
    if self.is_infinity {
      return Self {
        x: Fp::zero(),
        y: Fp::zero(),
        is_infinity: true,
        _p: Default::default(),
      };
    }

    let lambda = (Fp::one() + Fp::one() + Fp::one())
      * self.x
      * self.x
      * ((Fp::one() + Fp::one()) * self.y).invert().unwrap();
    let x = lambda * lambda - self.x - self.x;
    let y = lambda * (self.x - x) - self.y;
    Self {
      x,
      y,
      is_infinity: false,
      _p: Default::default(),
    }
  }

  pub fn scalar_mul(&self, scalar: &Fq) -> Self {
    let mut res = Self {
      x: Fp::zero(),
      y: Fp::zero(),
      is_infinity: true,
      _p: Default::default(),
    };

    let bits = scalar.to_le_bits();
    for i in (0..bits.len()).rev() {
      res = res.double();
      if bits[i] {
        res = self.add(&res);
      }
    }
    res
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_ecc_ops() {
    type Fp = pasta_curves::pallas::Base;
    type Fq = pasta_curves::pallas::Scalar;

    // perform some curve arithmetic
    let a = Point::<Fp, Fq>::random_vartime();
    let b = Point::<Fp, Fq>::random_vartime();
    let c = a.add(&b);
    let d = a.double();
    let s = Fq::random(&mut OsRng);
    let e = a.scalar_mul(&s);

    // perform the same computation by translating to pasta_curve types
    let a_pasta = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(a.x.to_repr()).unwrap(),
      pasta_curves::Fp::from_repr(a.y.to_repr()).unwrap(),
    )
    .unwrap();
    let b_pasta = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(b.x.to_repr()).unwrap(),
      pasta_curves::Fp::from_repr(b.y.to_repr()).unwrap(),
    )
    .unwrap();
    let c_pasta = (a_pasta + b_pasta).to_affine();
    let d_pasta = (a_pasta + a_pasta).to_affine();
    let e_pasta = a_pasta
      .mul(pasta_curves::Fq::from_repr(s.to_repr()).unwrap())
      .to_affine();

    // transform c, d, and e into pasta_curve types
    let c_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(c.x.to_repr()).unwrap(),
      pasta_curves::Fp::from_repr(c.y.to_repr()).unwrap(),
    )
    .unwrap();
    let d_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(d.x.to_repr()).unwrap(),
      pasta_curves::Fp::from_repr(d.y.to_repr()).unwrap(),
    )
    .unwrap();
    let e_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(e.x.to_repr()).unwrap(),
      pasta_curves::Fp::from_repr(e.y.to_repr()).unwrap(),
    )
    .unwrap();

    // check that we have the same outputs
    assert_eq!(c_pasta, c_pasta_2);
    assert_eq!(d_pasta, d_pasta_2);
    assert_eq!(e_pasta, e_pasta_2);
  }

  use crate::bellperson::{shape_cs::ShapeCS, solver::SatisfyingAssignment};
  use ff::{Field, PrimeFieldBits};
  use pasta_curves::{arithmetic::CurveAffine, group::Curve, EpAffine};
  use std::ops::Mul;
  type G = pasta_curves::pallas::Point;
  type Fp = pasta_curves::pallas::Scalar;
  type Fq = pasta_curves::vesta::Scalar;
  use crate::bellperson::r1cs::{NovaShape, NovaWitness};

  fn synthesize_smul<Fp, Fq, CS>(mut cs: CS) -> (AllocatedPoint<Fp>, AllocatedPoint<Fp>, Fq)
  where
    Fp: PrimeField,
    Fq: PrimeField + PrimeFieldBits,
    CS: ConstraintSystem<Fp>,
  {
    let a = AllocatedPoint::<Fp>::random_vartime(cs.namespace(|| "a")).unwrap();
    a.inputize(cs.namespace(|| "inputize a")).unwrap();
    let s = Fq::random(&mut OsRng);
    // Allocate random bits and only keep 128 bits
    let bits: Vec<AllocatedBit> = s
      .to_le_bits()
      .into_iter()
      .enumerate()
      .map(|(i, bit)| AllocatedBit::alloc(cs.namespace(|| format!("bit {}", i)), Some(bit)))
      .collect::<Result<Vec<AllocatedBit>, SynthesisError>>()
      .unwrap();
    let e = a.scalar_mul(cs.namespace(|| "Scalar Mul"), bits).unwrap();
    e.inputize(cs.namespace(|| "inputize e")).unwrap();
    (a, e, s)
  }

  #[test]
  fn test_ecc_circuit_ops() {
    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_smul::<Fp, Fq, _>(cs.namespace(|| "synthesize"));
    println!("Number of constraints: {}", cs.num_constraints());
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    // Then the satisfying assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let (a, e, s) = synthesize_smul::<Fp, Fq, _>(cs.namespace(|| "synthesize"));
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();

    let a_p: Point<Fp, Fq> = Point::new(
      a.x.get_value().unwrap(),
      a.y.get_value().unwrap(),
      a.is_infinity.get_value().unwrap() == Fp::one(),
    );
    let e_p: Point<Fp, Fq> = Point::new(
      e.x.get_value().unwrap(),
      e.y.get_value().unwrap(),
      e.is_infinity.get_value().unwrap() == Fp::one(),
    );
    let e_new = a_p.scalar_mul(&s);
    assert!(e_p.x == e_new.x && e_p.y == e_new.y);
    // Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

  fn synthesize_add_equal<Fp, Fq, CS>(mut cs: CS) -> (AllocatedPoint<Fp>, AllocatedPoint<Fp>)
  where
    Fp: PrimeField,
    Fq: PrimeField + PrimeFieldBits,
    CS: ConstraintSystem<Fp>,
  {
    let a = AllocatedPoint::<Fp>::random_vartime(cs.namespace(|| "a")).unwrap();
    a.inputize(cs.namespace(|| "inputize a")).unwrap();
    let e = a.add(cs.namespace(|| "add a to a"), &a).unwrap();
    e.inputize(cs.namespace(|| "inputize e")).unwrap();
    (a, e)
  }

  #[test]
  fn test_ecc_circuit_add_equal() {
    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_add_equal::<Fp, Fq, _>(cs.namespace(|| "synthesize add equal"));
    println!("Number of constraints: {}", cs.num_constraints());
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    // Then the satisfying assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let (a, e) = synthesize_add_equal::<Fp, Fq, _>(cs.namespace(|| "synthesize add equal"));
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();
    let a_p: Point<Fp, Fq> = Point::new(
      a.x.get_value().unwrap(),
      a.y.get_value().unwrap(),
      a.is_infinity.get_value().unwrap() == Fp::one(),
    );
    let e_p: Point<Fp, Fq> = Point::new(
      e.x.get_value().unwrap(),
      e.y.get_value().unwrap(),
      e.is_infinity.get_value().unwrap() == Fp::one(),
    );
    let e_new = a_p.add(&a_p);
    assert!(e_p.x == e_new.x && e_p.y == e_new.y);
    // Make sure that it is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }

  fn synthesize_add_negation<Fp, Fq, CS>(mut cs: CS) -> AllocatedPoint<Fp>
  where
    Fp: PrimeField,
    Fq: PrimeField + PrimeFieldBits,
    CS: ConstraintSystem<Fp>,
  {
    let a = AllocatedPoint::<Fp>::random_vartime(cs.namespace(|| "a")).unwrap();
    a.inputize(cs.namespace(|| "inputize a")).unwrap();
    let mut b = a.clone();
    b.y =
      AllocatedNum::alloc(cs.namespace(|| "allocate negation of a"), || Ok(Fp::zero())).unwrap();
    b.inputize(cs.namespace(|| "inputize b")).unwrap();
    let e = a.add(cs.namespace(|| "add a to b"), &b).unwrap();
    e
  }

  #[test]
  fn test_ecc_circuit_add_negation() {
    // First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_add_negation::<Fp, Fq, _>(cs.namespace(|| "synthesize add equal"));
    println!("Number of constraints: {}", cs.num_constraints());
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    // Then the satisfying assignment
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let e = synthesize_add_negation::<Fp, Fq, _>(cs.namespace(|| "synthesize add negation"));
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();
    let e_p: Point<Fp, Fq> = Point::new(
      e.x.get_value().unwrap(),
      e.y.get_value().unwrap(),
      e.is_infinity.get_value().unwrap() == Fp::one(),
    );
    assert!(e_p.is_infinity);
    // Make sure that it is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }
}
