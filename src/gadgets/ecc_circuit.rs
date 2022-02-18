#![allow(non_snake_case)]
use crate::gadgets::utils::{
  alloc_one, alloc_zero, conditionally_select, conditionally_select2, select_one_or, select_zero_or,
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
use rand::rngs::OsRng;

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
  //Creates a new allocated point from allocated nums.
  pub fn new(x: AllocatedNum<Fp>, y: AllocatedNum<Fp>, is_infinity: AllocatedNum<Fp>) -> Self {
    Self { x, y, is_infinity }
  }

  //Check that is infinity is 0/1
  #[allow(dead_code)]
  pub fn check_is_infinity<CS: ConstraintSystem<Fp>>(
    &self,
    mut cs: CS,
  ) -> Result<(), SynthesisError> {
    //Check that is_infinity * ( 1 - is_infinity ) = 0
    cs.enforce(
      || "is_infinity is bit",
      |lc| lc + self.is_infinity.get_variable(),
      |lc| lc + CS::one() - self.is_infinity.get_variable(),
      |lc| lc,
    );
    Ok(())
  }

  #[allow(dead_code)]
  //Allocate a random point. Only used for testing
  pub fn random_vartime<CS: ConstraintSystem<Fp>>(mut cs: CS) -> Result<Self, SynthesisError> {
    loop {
      let x = Fp::random(&mut OsRng);
      let y = (x * x * x + Fp::one() + Fp::one() + Fp::one() + Fp::one() + Fp::one()).sqrt();
      if y.is_some().unwrap_u8() == 1 {
        let x_alloc = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(x))?;
        let y_alloc = AllocatedNum::alloc(cs.namespace(|| "y"), || Ok(y.unwrap()))?;
        let is_infinity = alloc_zero(cs.namespace(|| "Is Infinity"))?;
        return Ok(Self::new(x_alloc, y_alloc, is_infinity));
      }
    }
  }

  //Make the point io
  #[allow(dead_code)]
  pub fn inputize<CS: ConstraintSystem<Fp>>(&self, mut cs: CS) -> Result<(), SynthesisError> {
    let _ = self.x.inputize(cs.namespace(|| "Input point.x"));
    let _ = self.y.inputize(cs.namespace(|| "Input point.y"));
    let _ = self
      .is_infinity
      .inputize(cs.namespace(|| "Input point.is_infinity"));
    Ok(())
  }

  //Adds other point to this point and returns the result
  //Assumes that both other.is_infinity and this.is_infinty are bits
  pub fn add<CS: ConstraintSystem<Fp>>(
    &self,
    mut cs: CS,
    other: &AllocatedPoint<Fp>,
  ) -> Result<Self, SynthesisError> {
    //Allocate the boolean variables that check if either of the points is infinity

    //************************************************************************/
    //lambda = (other.y - self.y) * (other.x - self.x).invert().unwrap();
    //************************************************************************/
    //First compute (other.x - self.x).inverse()
    //If either self or other are 1 then compute bogus values

    // x_diff = other != inf && self != inf ? (other.x - self.x) : 1
    let x_diff_actual = AllocatedNum::alloc(cs.namespace(|| "actual x diff"), || {
      Ok(*other.x.get_value().get()? - *self.x.get_value().get()?)
    })?;
    cs.enforce(
      || "actual x_diff is correct",
      |lc| lc + other.x.get_variable() - self.x.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + x_diff_actual.get_variable(),
    );

    //Compute self.is_infinity OR other.is_infinity
    let at_least_one_inf = AllocatedNum::alloc(cs.namespace(|| "at least one inf"), || {
      Ok(*self.is_infinity.get_value().get()? * *other.is_infinity.get_value().get()?)
    })?;
    cs.enforce(
      || "at least one inf = self.is_infinity * other.is_infinity",
      |lc| lc + self.is_infinity.get_variable(),
      |lc| lc + other.is_infinity.get_variable(),
      |lc| lc + at_least_one_inf.get_variable(),
    );

    //x_diff = 1 if either self.is_infinity or other.is_infinity else x_diff_actual
    let x_diff = select_one_or(
      cs.namespace(|| "Compute x_diff"),
      &x_diff_actual,
      &at_least_one_inf,
    )?;

    let x_diff_inv = AllocatedNum::alloc(cs.namespace(|| "x diff inverse"), || {
      if *at_least_one_inf.get_value().get()? == Fp::one() {
        //Set to default
        Ok(Fp::one())
      } else {
        //Set to the actual inverse
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
    //x = lambda * lambda - self.x - other.x;
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
    //y = lambda * (self.x - x) - self.y;
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

    let is_infinity = AllocatedNum::alloc(cs.namespace(|| "is infinity"), || Ok(Fp::zero()))?;

    //************************************************************************/
    // We only return the computed x, y if neither of the points is infinity.
    // if self.is_infinity return other.clone()
    // elif other.is_infinity return self.clone()
    // Otherwise return the computed points.
    //************************************************************************/
    //Now compute the output x
    let inner_x = conditionally_select2(
      cs.namespace(|| "final x: inner if"),
      &self.x,
      &x,
      &other.is_infinity,
    )?;
    let final_x = conditionally_select2(
      cs.namespace(|| "final x: outer if"),
      &other.x,
      &inner_x,
      &self.is_infinity,
    )?;

    //The output y
    let inner_y = conditionally_select2(
      cs.namespace(|| "final y: inner if"),
      &self.y,
      &y,
      &other.is_infinity,
    )?;
    let final_y = conditionally_select2(
      cs.namespace(|| "final y: outer if"),
      &other.y,
      &inner_y,
      &self.is_infinity,
    )?;

    //The output is_infinity
    let inner_is_infinity = conditionally_select2(
      cs.namespace(|| "final is infinity: inner if"),
      &self.is_infinity,
      &is_infinity,
      &other.is_infinity,
    )?;
    let final_is_infinity = conditionally_select2(
      cs.namespace(|| "final is infinity: outer if"),
      &other.is_infinity,
      &inner_is_infinity,
      &self.is_infinity,
    )?;
    return Ok(Self::new(final_x, final_y, final_is_infinity));
  }

  pub fn double<CS: ConstraintSystem<Fp>>(&self, mut cs: CS) -> Result<Self, SynthesisError> {
    //*************************************************************/
    // lambda = (Fp::one() + Fp::one() + Fp::one())
    //  * self.x
    //  * self.x
    //  * ((Fp::one() + Fp::one()) * self.y).invert().unwrap();
    /*************************************************************/

    //Compute tmp = (Fp::one() + Fp::one())* self.y ? self != inf : 1
    let tmp_actual = AllocatedNum::alloc(cs.namespace(|| "tmp_actual"), || {
      Ok(*self.y.get_value().get()? + *self.y.get_value().get()?)
    })?;
    cs.enforce(
      || "check tmp_actual",
      |lc| lc + CS::one() + CS::one(),
      |lc| lc + self.y.get_variable(),
      |lc| lc + tmp_actual.get_variable(),
    );

    let tmp = select_one_or(cs.namespace(|| "tmp"), &tmp_actual, &self.is_infinity)?;

    //Compute inv = tmp.invert
    let tmp_inv = AllocatedNum::alloc(cs.namespace(|| "tmp inverse"), || {
      if *self.is_infinity.get_value().get()? == Fp::one() {
        //Return default value 1
        Ok(Fp::one())
      } else {
        //Return the actual inverse
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

    //Now compute lambda as (Fp::one() + Fp::one + Fp::one()) * self.x * self.x * tmp_inv
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
    //Only return the computed x and y if the point is not infinity
    /*************************************************************/

    //x
    let final_x = select_zero_or(cs.namespace(|| "final x"), &x, &self.is_infinity)?;

    //y
    let final_y = select_zero_or(cs.namespace(|| "final y"), &y, &self.is_infinity)?;

    //is_infinity
    let final_is_infinity = self.is_infinity.clone();

    Ok(Self::new(final_x, final_y, final_is_infinity))
  }

  #[allow(dead_code)]
  pub fn scalar_mul_mont<CS: ConstraintSystem<Fp>>(
    &self,
    mut cs: CS,
    scalar: Vec<AllocatedBit>,
  ) -> Result<Self, SynthesisError> {
    /*************************************************************/
    //Initialize RO = Self {
    //  x: Fp::zero(),
    //  y: Fp::zero(),
    //  is_infinity: true,
    //  _p: Default::default(),
    //};
    /*************************************************************/

    let zero = alloc_zero(cs.namespace(|| "Allocate zero"))?;
    let one = alloc_one(cs.namespace(|| "Allocate one"))?;
    let mut R0 = Self::new(zero.clone(), zero.clone(), one.clone());

    /*************************************************************/
    //Initialize R1 and the bits of the scalar
    /*************************************************************/

    let mut R1 = self.clone();

    for i in (0..scalar.len()).rev() {
      /*************************************************************/
      //if bits[i] {
      //  R0 = R0.add(&R1);
      //  R1 = R1.double();
      //} else {
      //  R0 = R0.double();
      //  R1 = R0.add(&R1);
      //}
      /*************************************************************/

      let R0_and_R1 = R0.add(cs.namespace(|| format!("{}: R0 + R1", i)), &R1)?;
      let R0_double = R0.double(cs.namespace(|| format!("{}: 2 * R0", i)))?;
      let R1_double = R1.double(cs.namespace(|| format!("{}: 2 * R1", i)))?;

      R0 = Self::conditionally_select(
        cs.namespace(|| format!("{}: Update R0", i)),
        &R0_and_R1,
        &R0_double,
        &Boolean::from(scalar[i].clone()),
      )?;

      R1 = Self::conditionally_select(
        cs.namespace(|| format!("{}: Update R1", i)),
        &R1_double,
        &R0_and_R1,
        &Boolean::from(scalar[i].clone()),
      )?;
    }
    Ok(R0)
  }

  #[allow(dead_code)]
  pub fn scalar_mul<CS: ConstraintSystem<Fp>>(
    &self,
    mut cs: CS,
    scalar: Vec<AllocatedBit>,
  ) -> Result<Self, SynthesisError> {
    /*************************************************************/
    //Initialize res = Self {
    //  x: Fp::zero(),
    //  y: Fp::zero(),
    //  is_infinity: true,
    //  _p: Default::default(),
    //};
    /*************************************************************/

    let zero = alloc_zero(cs.namespace(|| "Allocate zero"))?;
    let one = alloc_one(cs.namespace(|| "Allocate one"))?;
    let mut res = Self::new(zero.clone(), zero.clone(), one.clone());

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

    Ok(Self::new(x, y, is_infinity))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bellperson::shape_cs::ShapeCS;
  use crate::bellperson::solver::SatisfyingAssignment;
  type G = pasta_curves::pallas::Point;
  type Fp = pasta_curves::pallas::Scalar;
  type Fq = pasta_curves::vesta::Scalar;
  use crate::bellperson::r1cs::{NovaShape, NovaWitness};
  use crate::gadgets::ecc::Point;
  use ff::PrimeFieldBits;

  fn synthesize_smul<Fp, Fq, CS>(mut cs: CS) -> (AllocatedPoint<Fp>, AllocatedPoint<Fp>, Fq)
  where
    Fp: PrimeField,
    Fq: PrimeField + PrimeFieldBits,
    CS: ConstraintSystem<Fp>,
  {
    let a = AllocatedPoint::<Fp>::random_vartime(cs.namespace(|| "a")).unwrap();
    let _ = a.inputize(cs.namespace(|| "inputize a")).unwrap();
    let s = Fq::random(&mut OsRng);
    //Allocate random bits and only keep 128 bits
    let bits: Vec<AllocatedBit> = s
      .to_le_bits()
      .into_iter()
      .enumerate()
      .map(|(i, bit)| AllocatedBit::alloc(cs.namespace(|| format!("bit {}", i)), Some(bit)))
      .collect::<Result<Vec<AllocatedBit>, SynthesisError>>()
      .unwrap();
    let e = a
      //.scalar_mul(cs.namespace(|| "Scalar Mul"), bits[..128].to_vec())
      .scalar_mul(cs.namespace(|| "Scalar Mul"), bits)
      .unwrap();
    let _ = e.inputize(cs.namespace(|| "inputize e")).unwrap();
    return (a, e, s);
  }

  #[test]
  fn test_ecc_circuit_ops() {
    //First create the shape
    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = synthesize_smul::<Fp, Fq, _>(cs.namespace(|| "synthesize"));
    println!("Number of constraints: {}", cs.num_constraints());
    let shape = cs.r1cs_shape();
    let gens = cs.r1cs_gens();

    //Then the satisfying assignment
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
    //Make sure that this is satisfiable
    assert!(shape.is_sat(&gens, &inst, &witness).is_ok());
  }
}
