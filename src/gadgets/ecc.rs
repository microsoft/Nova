//! This module implements various elliptic curve gadgets
#![allow(non_snake_case)]
use crate::{
  gadgets::utils::{
    alloc_num_equals, alloc_one, alloc_zero, conditionally_select, conditionally_select2,
    select_num_or_one, select_num_or_zero, select_num_or_zero2, select_one_or_diff2,
    select_one_or_num2, select_zero_or_num2,
  },
  traits::{Engine, Group},
};
use bellpepper::gadgets::Assignment;
use bellpepper_core::{
  boolean::{AllocatedBit, Boolean},
  num::AllocatedNum,
  ConstraintSystem, SynthesisError,
};
use ff::{Field, PrimeField};

/// `AllocatedPoint` provides an elliptic curve abstraction inside a circuit.
#[derive(Clone)]
pub struct AllocatedPoint<E: Engine> {
  pub(crate) x: AllocatedNum<E::Base>,
  pub(crate) y: AllocatedNum<E::Base>,
  pub(crate) is_infinity: AllocatedNum<E::Base>,
}

impl<E> AllocatedPoint<E>
where
  E: Engine,
{
  /// Allocates a new point on the curve using coordinates provided by `coords`.
  /// If coords = None, it allocates the default infinity point
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    coords: Option<(E::Base, E::Base, bool)>,
  ) -> Result<Self, SynthesisError> {
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(coords.map_or(E::Base::ZERO, |c| c.0))
    })?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(coords.map_or(E::Base::ZERO, |c| c.1))
    })?;
    let is_infinity = AllocatedNum::alloc(cs.namespace(|| "is_infinity"), || {
      Ok(if coords.map_or(true, |c| c.2) {
        E::Base::ONE
      } else {
        E::Base::ZERO
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

  /// checks if `self` is on the curve or if it is infinity
  pub fn check_on_curve<CS>(&self, mut cs: CS) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<E::Base>,
  {
    // check that (x,y) is on the curve if it is not infinity
    // we will check that (1- is_infinity) * y^2 = (1-is_infinity) * (x^3 + Ax + B)
    // note that is_infinity is already restricted to be in the set {0, 1}
    let y_square = self.y.square(cs.namespace(|| "y_square"))?;
    let x_square = self.x.square(cs.namespace(|| "x_square"))?;
    let x_cube = self.x.mul(cs.namespace(|| "x_cube"), &x_square)?;

    let rhs = AllocatedNum::alloc(cs.namespace(|| "rhs"), || {
      if *self.is_infinity.get_value().get()? == E::Base::ONE {
        Ok(E::Base::ZERO)
      } else {
        Ok(
          *x_cube.get_value().get()?
            + *self.x.get_value().get()? * E::GE::group_params().0
            + E::GE::group_params().1,
        )
      }
    })?;

    cs.enforce(
      || "rhs = (1-is_infinity) * (x^3 + Ax + B)",
      |lc| {
        lc + x_cube.get_variable()
          + (E::GE::group_params().0, self.x.get_variable())
          + (E::GE::group_params().1, CS::one())
      },
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + rhs.get_variable(),
    );

    // check that (1-infinity) * y_square = rhs
    cs.enforce(
      || "check that y_square * (1 - is_infinity) = rhs",
      |lc| lc + y_square.get_variable(),
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc + rhs.get_variable(),
    );

    Ok(())
  }

  /// Allocates a default point on the curve, set to the identity point.
  pub fn default<CS: ConstraintSystem<E::Base>>(mut cs: CS) -> Result<Self, SynthesisError> {
    let zero = alloc_zero(cs.namespace(|| "zero"));
    let one = alloc_one(cs.namespace(|| "one"));

    Ok(AllocatedPoint {
      x: zero.clone(),
      y: zero,
      is_infinity: one,
    })
  }

  /// Returns coordinates associated with the point.
  pub const fn get_coordinates(
    &self,
  ) -> (
    &AllocatedNum<E::Base>,
    &AllocatedNum<E::Base>,
    &AllocatedNum<E::Base>,
  ) {
    (&self.x, &self.y, &self.is_infinity)
  }

  /// Negates the provided point
  pub fn negate<CS: ConstraintSystem<E::Base>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || Ok(-*self.y.get_value().get()?))?;

    cs.enforce(
      || "check y = - self.y",
      |lc| lc + self.y.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc - y.get_variable(),
    );

    Ok(Self {
      x: self.x.clone(),
      y,
      is_infinity: self.is_infinity.clone(),
    })
  }

  /// Add two points (may be equal)
  pub fn add<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<E>,
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
  /// different and that both `other.is_infinity` and `this.is_infinity` are bits
  pub fn add_internal<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<E>,
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
        E::Base::ONE
          - (E::Base::ONE - *self.is_infinity.get_value().get()?)
            * (E::Base::ONE - *other.is_infinity.get_value().get()?),
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
          E::Base::ONE
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

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let x_diff_inv = if *x_diff_is_actual.get_value().get()? == E::Base::ONE {
        // Set to default
        E::Base::ONE
      } else {
        // Set to the actual inverse
        (*other.x.get_value().get()? - *self.x.get_value().get()?)
          .invert()
          .unwrap()
      };

      Ok((*other.y.get_value().get()? - *self.y.get_value().get()?) * x_diff_inv)
    })?;
    cs.enforce(
      || "Check that lambda is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + x_diff.get_variable(),
      |lc| lc + other.y.get_variable() - self.y.get_variable(),
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
  pub fn double<CS: ConstraintSystem<E::Base>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    //*************************************************************/
    // lambda = (E::Base::from(3) * self.x * self.x + E::GE::A())
    //  * (E::Base::from(2)) * self.y).invert().unwrap();
    /*************************************************************/

    // Compute tmp = (E::Base::ONE + E::Base::ONE)* self.y ? self != inf : 1
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

    // Now compute lambda as (E::Base::from(3) * self.x * self.x + E::GE::A()) * tmp_inv

    let prod_1 = AllocatedNum::alloc(cs.namespace(|| "alloc prod 1"), || {
      Ok(E::Base::from(3) * self.x.get_value().get()? * self.x.get_value().get()?)
    })?;
    cs.enforce(
      || "Check prod 1",
      |lc| lc + (E::Base::from(3), self.x.get_variable()),
      |lc| lc + self.x.get_variable(),
      |lc| lc + prod_1.get_variable(),
    );

    let lambda = AllocatedNum::alloc(cs.namespace(|| "alloc lambda"), || {
      let tmp_inv = if *self.is_infinity.get_value().get()? == E::Base::ONE {
        // Return default value 1
        E::Base::ONE
      } else {
        // Return the actual inverse
        (*tmp.get_value().get()?).invert().unwrap()
      };

      Ok(tmp_inv * (*prod_1.get_value().get()? + E::GE::group_params().0))
    })?;

    cs.enforce(
      || "Check lambda",
      |lc| lc + tmp.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + prod_1.get_variable() + (E::GE::group_params().0, CS::one()),
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

  /// A gadget for scalar multiplication, optimized to use incomplete addition law.
  /// The optimization here is analogous to <https://github.com/arkworks-rs/r1cs-std/blob/6d64f379a27011b3629cf4c9cb38b7b7b695d5a0/src/groups/curves/short_weierstrass/mod.rs#L295>,
  /// except we use complete addition law over affine coordinates instead of projective coordinates for the tail bits
  pub fn scalar_mul<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
    scalar_bits: &[AllocatedBit],
  ) -> Result<Self, SynthesisError> {
    let split_len = core::cmp::min(scalar_bits.len(), (E::Base::NUM_BITS - 2) as usize);
    let (incomplete_bits, complete_bits) = scalar_bits.split_at(split_len);

    // we convert AllocatedPoint into AllocatedPointNonInfinity; we deal with the case where self.is_infinity = 1 below
    let mut p = AllocatedPointNonInfinity::from_allocated_point(self);

    // we assume the first bit to be 1, so we must initialize acc to self and double it
    // we remove this assumption below
    let mut acc = p;
    p = acc.double_incomplete(cs.namespace(|| "double"))?;

    // perform the double-and-add loop to compute the scalar mul using incomplete addition law
    for (i, bit) in incomplete_bits.iter().enumerate().skip(1) {
      let temp = acc.add_incomplete(cs.namespace(|| format!("add {i}")), &p)?;
      acc = AllocatedPointNonInfinity::conditionally_select(
        cs.namespace(|| format!("acc_iteration_{i}")),
        &temp,
        &acc,
        &Boolean::from(bit.clone()),
      )?;

      p = p.double_incomplete(cs.namespace(|| format!("double {i}")))?;
    }

    // convert back to AllocatedPoint
    let res = {
      // we set acc.is_infinity = self.is_infinity
      let acc = acc.to_allocated_point(&self.is_infinity)?;

      // we remove the initial slack if bits[0] is as not as assumed (i.e., it is not 1)
      let acc_minus_initial = {
        let neg = self.negate(cs.namespace(|| "negate"))?;
        acc.add(cs.namespace(|| "res minus self"), &neg)
      }?;

      AllocatedPoint::conditionally_select(
        cs.namespace(|| "remove slack if necessary"),
        &acc,
        &acc_minus_initial,
        &Boolean::from(scalar_bits[0].clone()),
      )?
    };

    // when self.is_infinity = 1, return the default point, else return res
    // we already set res.is_infinity to be self.is_infinity, so we do not need to set it here
    let default = Self::default(cs.namespace(|| "default"))?;
    let x = conditionally_select2(
      cs.namespace(|| "check if self.is_infinity is zero (x)"),
      &default.x,
      &res.x,
      &self.is_infinity,
    )?;

    let y = conditionally_select2(
      cs.namespace(|| "check if self.is_infinity is zero (y)"),
      &default.y,
      &res.y,
      &self.is_infinity,
    )?;

    // we now perform the remaining scalar mul using complete addition law
    let mut acc = AllocatedPoint {
      x,
      y,
      is_infinity: res.is_infinity,
    };
    let mut p_complete = p.to_allocated_point(&self.is_infinity)?;

    for (i, bit) in complete_bits.iter().enumerate() {
      let temp = acc.add(cs.namespace(|| format!("add_complete {i}")), &p_complete)?;
      acc = AllocatedPoint::conditionally_select(
        cs.namespace(|| format!("acc_complete_iteration_{i}")),
        &temp,
        &acc,
        &Boolean::from(bit.clone()),
      )?;

      p_complete = p_complete.double(cs.namespace(|| format!("double_complete {i}")))?;
    }

    Ok(acc)
  }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<E::Base>>(
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
  pub fn select_point_or_infinity<CS: ConstraintSystem<E::Base>>(
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

#[derive(Clone)]
/// `AllocatedPoint` but one that is guaranteed to be not infinity
pub struct AllocatedPointNonInfinity<E: Engine> {
  x: AllocatedNum<E::Base>,
  y: AllocatedNum<E::Base>,
}

impl<E> AllocatedPointNonInfinity<E>
where
  E: Engine,
{
  /// Creates a new `AllocatedPointNonInfinity` from the specified coordinates
  pub const fn new(x: AllocatedNum<E::Base>, y: AllocatedNum<E::Base>) -> Self {
    Self { x, y }
  }

  /// Allocates a new point on the curve using coordinates provided by `coords`.
  pub fn alloc<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    coords: Option<(E::Base, E::Base)>,
  ) -> Result<Self, SynthesisError> {
    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      coords.map_or(Err(SynthesisError::AssignmentMissing), |c| Ok(c.0))
    })?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      coords.map_or(Err(SynthesisError::AssignmentMissing), |c| Ok(c.1))
    })?;

    Ok(Self { x, y })
  }

  /// Turns an `AllocatedPoint` into an `AllocatedPointNonInfinity` (assumes it is not infinity)
  pub fn from_allocated_point(p: &AllocatedPoint<E>) -> Self {
    Self {
      x: p.x.clone(),
      y: p.y.clone(),
    }
  }

  /// Returns an `AllocatedPoint` from an `AllocatedPointNonInfinity`
  pub fn to_allocated_point(
    &self,
    is_infinity: &AllocatedNum<E::Base>,
  ) -> Result<AllocatedPoint<E>, SynthesisError> {
    Ok(AllocatedPoint {
      x: self.x.clone(),
      y: self.y.clone(),
      is_infinity: is_infinity.clone(),
    })
  }

  /// Returns coordinates associated with the point.
  pub const fn get_coordinates(&self) -> (&AllocatedNum<E::Base>, &AllocatedNum<E::Base>) {
    (&self.x, &self.y)
  }

  /// Add two points assuming self != +/- other
  pub fn add_incomplete<CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<E::Base>,
  {
    // allocate a free variable that an honest prover sets to lambda = (y2-y1)/(x2-x1)
    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      if *other.x.get_value().get()? == *self.x.get_value().get()? {
        Ok(E::Base::ONE)
      } else {
        Ok(
          (*other.y.get_value().get()? - *self.y.get_value().get()?)
            * (*other.x.get_value().get()? - *self.x.get_value().get()?)
              .invert()
              .unwrap(),
        )
      }
    })?;
    cs.enforce(
      || "Check that lambda is computed correctly",
      |lc| lc + lambda.get_variable(),
      |lc| lc + other.x.get_variable() - self.x.get_variable(),
      |lc| lc + other.y.get_variable() - self.y.get_variable(),
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

    Ok(Self { x, y })
  }

  /// doubles the point; since this is called with a point not at infinity, it is guaranteed to be not infinity
  pub fn double_incomplete<CS: ConstraintSystem<E::Base>>(
    &self,
    mut cs: CS,
  ) -> Result<Self, SynthesisError> {
    // lambda = (3 x^2 + a) / 2 * y

    let x_sq = self.x.square(cs.namespace(|| "x_sq"))?;

    let lambda = AllocatedNum::alloc(cs.namespace(|| "lambda"), || {
      let n = E::Base::from(3) * x_sq.get_value().get()? + E::GE::group_params().0;
      let d = E::Base::from(2) * *self.y.get_value().get()?;
      if d == E::Base::ZERO {
        Ok(E::Base::ONE)
      } else {
        Ok(n * d.invert().unwrap())
      }
    })?;
    cs.enforce(
      || "Check that lambda is computed correctly",
      |lc| lc + lambda.get_variable(),
      |lc| lc + (E::Base::from(2), self.y.get_variable()),
      |lc| lc + (E::Base::from(3), x_sq.get_variable()) + (E::GE::group_params().0, CS::one()),
    );

    let x = AllocatedNum::alloc(cs.namespace(|| "x"), || {
      Ok(
        *lambda.get_value().get()? * *lambda.get_value().get()?
          - *self.x.get_value().get()?
          - *self.x.get_value().get()?,
      )
    })?;

    cs.enforce(
      || "check that x is correct",
      |lc| lc + lambda.get_variable(),
      |lc| lc + lambda.get_variable(),
      |lc| lc + x.get_variable() + (E::Base::from(2), self.x.get_variable()),
    );

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

    Ok(Self { x, y })
  }

  /// If condition outputs a otherwise outputs b
  pub fn conditionally_select<CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<Self, SynthesisError> {
    let x = conditionally_select(cs.namespace(|| "select x"), &a.x, &b.x, condition)?;
    let y = conditionally_select(cs.namespace(|| "select y"), &a.y, &b.y, condition)?;

    Ok(Self { x, y })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    bellpepper::{
      r1cs::{NovaShape, NovaWitness},
      {solver::SatisfyingAssignment, test_shape_cs::TestShapeCS},
    },
    provider::{
      bn256_grumpkin::{bn256, grumpkin},
      secp_secq::{secp256k1, secq256k1},
      Bn256EngineKZG, GrumpkinEngine, Secp256k1Engine, Secq256k1Engine,
      {PallasEngine, VestaEngine},
    },
    traits::snark::default_ck_hint,
  };
  use ff::{Field, PrimeFieldBits};
  use pasta_curves::{arithmetic::CurveAffine, group::Curve, pallas, vesta};
  use rand::rngs::OsRng;

  #[derive(Debug, Clone)]
  pub struct Point<E: Engine> {
    x: E::Base,
    y: E::Base,
    is_infinity: bool,
  }

  impl<E: Engine> Point<E> {
    pub fn new(x: E::Base, y: E::Base, is_infinity: bool) -> Self {
      Self { x, y, is_infinity }
    }

    pub fn random_vartime() -> Self {
      loop {
        let x = E::Base::random(&mut OsRng);
        let y = (x.square() * x + E::GE::group_params().1).sqrt();
        if y.is_some().unwrap_u8() == 1 {
          return Self {
            x,
            y: y.unwrap(),
            is_infinity: false,
          };
        }
      }
    }

    /// Add any two points
    pub fn add(&self, other: &Point<E>) -> Self {
      if self.x == other.x {
        // If self == other then call double
        if self.y == other.y {
          self.double()
        } else {
          // if self.x == other.x and self.y != other.y then return infinity
          Self {
            x: E::Base::ZERO,
            y: E::Base::ZERO,
            is_infinity: true,
          }
        }
      } else {
        self.add_internal(other)
      }
    }

    /// Add two different points
    pub fn add_internal(&self, other: &Point<E>) -> Self {
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
      }
    }

    pub fn double(&self) -> Self {
      if self.is_infinity {
        return Self {
          x: E::Base::ZERO,
          y: E::Base::ZERO,
          is_infinity: true,
        };
      }

      let lambda = E::Base::from(3)
        * self.x
        * self.x
        * ((E::Base::ONE + E::Base::ONE) * self.y).invert().unwrap();
      let x = lambda * lambda - self.x - self.x;
      let y = lambda * (self.x - x) - self.y;
      Self {
        x,
        y,
        is_infinity: false,
      }
    }

    pub fn scalar_mul(&self, scalar: &E::Scalar) -> Self {
      let mut res = Self {
        x: E::Base::ZERO,
        y: E::Base::ZERO,
        is_infinity: true,
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

  // Allocate a random point. Only used for testing
  pub fn alloc_random_point<E: Engine, CS: ConstraintSystem<E::Base>>(
    mut cs: CS,
  ) -> Result<AllocatedPoint<E>, SynthesisError> {
    // get a random point
    let p = Point::<E>::random_vartime();
    AllocatedPoint::alloc(cs.namespace(|| "alloc p"), Some((p.x, p.y, p.is_infinity)))
  }

  /// Make the point io
  pub fn inputize_allocted_point<E: Engine, CS: ConstraintSystem<E::Base>>(
    p: &AllocatedPoint<E>,
    mut cs: CS,
  ) {
    let _ = p.x.inputize(cs.namespace(|| "Input point.x"));
    let _ = p.y.inputize(cs.namespace(|| "Input point.y"));
    let _ = p
      .is_infinity
      .inputize(cs.namespace(|| "Input point.is_infinity"));
  }

  #[test]
  fn test_ecc_ops() {
    test_ecc_ops_with::<pallas::Affine, PallasEngine>();
    test_ecc_ops_with::<vesta::Affine, VestaEngine>();

    test_ecc_ops_with::<bn256::Affine, Bn256EngineKZG>();
    test_ecc_ops_with::<grumpkin::Affine, GrumpkinEngine>();

    test_ecc_ops_with::<secp256k1::Affine, Secp256k1Engine>();
    test_ecc_ops_with::<secq256k1::Affine, Secq256k1Engine>();
  }

  fn test_ecc_ops_with<C, E>()
  where
    E: Engine,
    C: CurveAffine<Base = E::Base, ScalarExt = E::Scalar>,
  {
    // perform some curve arithmetic
    let a = Point::<E>::random_vartime();
    let b = Point::<E>::random_vartime();
    let c = a.add(&b);
    let d = a.double();
    let s = <E as Engine>::Scalar::random(&mut OsRng);
    let e = a.scalar_mul(&s);

    // perform the same computation by translating to curve types
    let a_curve = C::from_xy(
      C::Base::from_repr(a.x.to_repr()).unwrap(),
      C::Base::from_repr(a.y.to_repr()).unwrap(),
    )
    .unwrap();
    let b_curve = C::from_xy(
      C::Base::from_repr(b.x.to_repr()).unwrap(),
      C::Base::from_repr(b.y.to_repr()).unwrap(),
    )
    .unwrap();
    let c_curve = (a_curve + b_curve).to_affine();
    let d_curve = (a_curve + a_curve).to_affine();
    let e_curve = a_curve
      .mul(C::Scalar::from_repr(s.to_repr()).unwrap())
      .to_affine();

    // transform c, d, and e into curve types
    let c_curve_2 = C::from_xy(
      C::Base::from_repr(c.x.to_repr()).unwrap(),
      C::Base::from_repr(c.y.to_repr()).unwrap(),
    )
    .unwrap();
    let d_curve_2 = C::from_xy(
      C::Base::from_repr(d.x.to_repr()).unwrap(),
      C::Base::from_repr(d.y.to_repr()).unwrap(),
    )
    .unwrap();
    let e_curve_2 = C::from_xy(
      C::Base::from_repr(e.x.to_repr()).unwrap(),
      C::Base::from_repr(e.y.to_repr()).unwrap(),
    )
    .unwrap();

    // check that we have the same outputs
    assert_eq!(c_curve, c_curve_2);
    assert_eq!(d_curve, d_curve_2);
    assert_eq!(e_curve, e_curve_2);
  }

  fn synthesize_smul<E, CS>(mut cs: CS) -> (AllocatedPoint<E>, AllocatedPoint<E>, E::Scalar)
  where
    E: Engine,
    CS: ConstraintSystem<E::Base>,
  {
    let a = alloc_random_point(cs.namespace(|| "a")).unwrap();
    inputize_allocted_point(&a, cs.namespace(|| "inputize a"));

    let s = E::Scalar::random(&mut OsRng);
    // Allocate bits for s
    let bits: Vec<AllocatedBit> = s
      .to_le_bits()
      .into_iter()
      .enumerate()
      .map(|(i, bit)| AllocatedBit::alloc(cs.namespace(|| format!("bit {i}")), Some(bit)))
      .collect::<Result<Vec<AllocatedBit>, SynthesisError>>()
      .unwrap();
    let e = a.scalar_mul(cs.namespace(|| "Scalar Mul"), &bits).unwrap();
    inputize_allocted_point(&e, cs.namespace(|| "inputize e"));
    (a, e, s)
  }

  #[test]
  fn test_ecc_circuit_ops() {
    test_ecc_circuit_ops_with::<PallasEngine, VestaEngine>();
    test_ecc_circuit_ops_with::<VestaEngine, PallasEngine>();

    test_ecc_circuit_ops_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ecc_circuit_ops_with::<GrumpkinEngine, Bn256EngineKZG>();

    test_ecc_circuit_ops_with::<Secp256k1Engine, Secq256k1Engine>();
    test_ecc_circuit_ops_with::<Secq256k1Engine, Secp256k1Engine>();
  }

  fn test_ecc_circuit_ops_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    // First create the shape
    let mut cs: TestShapeCS<E2> = TestShapeCS::new();
    let _ = synthesize_smul::<E1, _>(cs.namespace(|| "synthesize"));
    println!("Number of constraints: {}", cs.num_constraints());
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());

    // Then the satisfying assignment
    let mut cs = SatisfyingAssignment::<E2>::new();
    let (a, e, s) = synthesize_smul::<E1, _>(cs.namespace(|| "synthesize"));
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    let a_p: Point<E1> = Point::new(
      a.x.get_value().unwrap(),
      a.y.get_value().unwrap(),
      a.is_infinity.get_value().unwrap() == <E1 as Engine>::Base::ONE,
    );
    let e_p: Point<E1> = Point::new(
      e.x.get_value().unwrap(),
      e.y.get_value().unwrap(),
      e.is_infinity.get_value().unwrap() == <E1 as Engine>::Base::ONE,
    );
    let e_new = a_p.scalar_mul(&s);
    assert!(e_p.x == e_new.x && e_p.y == e_new.y);
    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  fn synthesize_add_equal<E, CS>(mut cs: CS) -> (AllocatedPoint<E>, AllocatedPoint<E>)
  where
    E: Engine,
    CS: ConstraintSystem<E::Base>,
  {
    let a = alloc_random_point(cs.namespace(|| "a")).unwrap();
    inputize_allocted_point(&a, cs.namespace(|| "inputize a"));
    let e = a.add(cs.namespace(|| "add a to a"), &a).unwrap();
    inputize_allocted_point(&e, cs.namespace(|| "inputize e"));
    (a, e)
  }

  #[test]
  fn test_ecc_circuit_add_equal() {
    test_ecc_circuit_add_equal_with::<PallasEngine, VestaEngine>();
    test_ecc_circuit_add_equal_with::<VestaEngine, PallasEngine>();

    test_ecc_circuit_add_equal_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ecc_circuit_add_equal_with::<GrumpkinEngine, Bn256EngineKZG>();

    test_ecc_circuit_add_equal_with::<Secp256k1Engine, Secq256k1Engine>();
    test_ecc_circuit_add_equal_with::<Secq256k1Engine, Secp256k1Engine>();
  }

  fn test_ecc_circuit_add_equal_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    // First create the shape
    let mut cs: TestShapeCS<E2> = TestShapeCS::new();
    let _ = synthesize_add_equal::<E1, _>(cs.namespace(|| "synthesize add equal"));
    println!("Number of constraints: {}", cs.num_constraints());
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());

    // Then the satisfying assignment
    let mut cs = SatisfyingAssignment::<E2>::new();
    let (a, e) = synthesize_add_equal::<E1, _>(cs.namespace(|| "synthesize add equal"));
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let a_p: Point<E1> = Point::new(
      a.x.get_value().unwrap(),
      a.y.get_value().unwrap(),
      a.is_infinity.get_value().unwrap() == <E1 as Engine>::Base::ONE,
    );
    let e_p: Point<E1> = Point::new(
      e.x.get_value().unwrap(),
      e.y.get_value().unwrap(),
      e.is_infinity.get_value().unwrap() == <E1 as Engine>::Base::ONE,
    );
    let e_new = a_p.add(&a_p);
    assert!(e_p.x == e_new.x && e_p.y == e_new.y);
    // Make sure that it is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  fn synthesize_add_negation<E, CS>(mut cs: CS) -> AllocatedPoint<E>
  where
    E: Engine,
    CS: ConstraintSystem<E::Base>,
  {
    let a = alloc_random_point(cs.namespace(|| "a")).unwrap();
    inputize_allocted_point(&a, cs.namespace(|| "inputize a"));
    let b = &mut a.clone();
    b.y = AllocatedNum::alloc(cs.namespace(|| "allocate negation of a"), || {
      Ok(E::Base::ZERO)
    })
    .unwrap();
    inputize_allocted_point(b, cs.namespace(|| "inputize b"));
    let e = a.add(cs.namespace(|| "add a to b"), b).unwrap();
    e
  }

  #[test]
  fn test_ecc_circuit_add_negation() {
    test_ecc_circuit_add_negation_with::<PallasEngine, VestaEngine>();
    test_ecc_circuit_add_negation_with::<VestaEngine, PallasEngine>();

    test_ecc_circuit_add_negation_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ecc_circuit_add_negation_with::<GrumpkinEngine, Bn256EngineKZG>();

    test_ecc_circuit_add_negation_with::<Secp256k1Engine, Secq256k1Engine>();
    test_ecc_circuit_add_negation_with::<Secq256k1Engine, Secp256k1Engine>();
  }

  fn test_ecc_circuit_add_negation_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    // First create the shape
    let mut cs: TestShapeCS<E2> = TestShapeCS::new();
    let _ = synthesize_add_negation::<E1, _>(cs.namespace(|| "synthesize add equal"));
    println!("Number of constraints: {}", cs.num_constraints());
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());

    // Then the satisfying assignment
    let mut cs = SatisfyingAssignment::<E2>::new();
    let e = synthesize_add_negation::<E1, _>(cs.namespace(|| "synthesize add negation"));
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();
    let e_p: Point<E1> = Point::new(
      e.x.get_value().unwrap(),
      e.y.get_value().unwrap(),
      e.is_infinity.get_value().unwrap() == <E1 as Engine>::Base::ONE,
    );
    assert!(e_p.is_infinity);
    // Make sure that it is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }
}
