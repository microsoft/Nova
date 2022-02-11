#![allow(non_snake_case)]
use ff::{PrimeField, PrimeFieldBits};
use rand::rngs::OsRng;
use std::marker::PhantomData;
use bellperson::{LinearCombination, ConstraintSystem, SynthesisError, gadgets::{num::AllocatedNum, boolean::{AllocatedBit, Boolean}, Assignment}};
use crate::gadgets::utils::{alloc_num_equals, conditionally_select};

#[derive(Clone)]
pub struct AllocatedPoint<Fp, Fq>
where
  Fp: PrimeField,
  Fq: PrimeField + PrimeFieldBits,
{
  x: AllocatedNum<Fp>,
  y: AllocatedNum<Fp>,
  is_infinity: AllocatedNum<Fp>, //TODO: Make this allocatedbit
  _p: PhantomData<Fq>,
}

impl<Fp, Fq> AllocatedPoint<Fp, Fq>
where
  Fp: PrimeField,
  Fq: PrimeField + PrimeFieldBits,
{
  //Creates a new allocated point from allocated nums
	pub fn new(x: AllocatedNum<Fp>, y: AllocatedNum<Fp>, is_infinity: AllocatedNum<Fp>) -> Self {
		//Make sure that is_infinity is either zero or 1: is_infinity * (1 - is_infinity) = 0
		Self{
			x,
			y,
			is_infinity,
			_p: Default::default(),
		}
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
				let is_infinity = AllocatedNum::alloc(cs.namespace(|| "Is Infinity"), || Ok(Fp::zero()))?;
				return Ok(Self::new(x_alloc, y_alloc, is_infinity));
			}
		}
  }

	//Make the point io
	pub fn inputize<CS:ConstraintSystem<Fp>>(&self, mut cs: CS) -> Result<(), SynthesisError> {
		let _ = self.x.inputize(cs.namespace(|| "Input point.x"));
		let _ = self.y.inputize(cs.namespace(|| "Input point.y"));
		let _ = self.is_infinity.inputize(cs.namespace(|| "Input point.is_infinity"));
		Ok(())
	}

  pub fn add<CS: ConstraintSystem<Fp>>(&self, mut cs: CS, other: &AllocatedPoint<Fp, Fq>) -> Result<Self, SynthesisError> {
    
		//First allocate the inverse of (other.x - self.x)
		let x_diff_inv = AllocatedNum::alloc(
			cs.namespace(|| "x diff inverse"),
			|| {
				let inv = (*other.x.get_value().get()? - *self.x.get_value().get()?).invert();
				if inv.is_some().unwrap_u8() == 1 {
					Ok(inv.unwrap())
				}else{
					Err(SynthesisError::DivisionByZero)
				}
			},
		)?;	
		//Enforce that (other.x - self.x)*x_diff_inv == 1
		cs.enforce(
			|| "Check inverse",
			|lc| lc + other.x.get_variable() - self.x.get_variable(),
			|lc| lc + x_diff_inv.get_variable(),
			|lc| lc + CS::one()
		);
		
		//lambda = (other.y - self.y) * (other.x - self.x).invert().unwrap();
		let lambda = AllocatedNum::alloc(
			cs.namespace(|| "lambda"),
			|| Ok((*other.y.get_value().get()? - *self.y.get_value().get()?)*x_diff_inv.get_value().get()?)
		)?;
		cs.enforce(
			|| "Check that lambda is correct",
			|lc| lc + other.y.get_variable() - self.y.get_variable(),
			|lc| lc + x_diff_inv.get_variable(),
			|lc| lc + lambda.get_variable()
		);

    //x = lambda * lambda - self.x - other.x;
		let x = AllocatedNum::alloc(
			cs.namespace(|| "x"), 
			|| Ok(*lambda.get_value().get()?*lambda.get_value().get()? - *self.x.get_value().get()? - *other.x.get_value().get()?)
		)?;
		cs.enforce(
			|| "check that x is correct",
			|lc| lc + lambda.get_variable(),
			|lc| lc + lambda.get_variable(),
			|lc| lc + x.get_variable() + self.x.get_variable() + other.x.get_variable(),
		);
    
		//y = lambda * (self.x - x) - self.y;
		let y = AllocatedNum::alloc(
			cs.namespace(|| "y"), 
			|| Ok(*lambda.get_value().get()?*(*self.x.get_value().get()? - *x.get_value().get()?) - *self.y.get_value().get()?)
		)?;
		
		cs.enforce(
			|| "Check that y is correct",
			|lc| lc + lambda.get_variable(),
			|lc| lc + self.x.get_variable() - x.get_variable(),
			|lc| lc + y.get_variable() + self.y.get_variable()
		);

		let is_infinity = AllocatedNum::alloc(cs.namespace(|| "is infinity"), || Ok(Fp::zero()))?;
		// We only return the computed x, y if neither of the points is infinity. 
		// if self.is_infinity return other.clone() 
		// elif other.is_infinity return self.clone() 
		// Otherwise return the computed points. 
		let false_repr = AllocatedNum::alloc(cs.namespace(|| "false num"), || Ok(Fp::zero()))?;
		let self_is_not_inf = Boolean::from(alloc_num_equals(
			cs.namespace(|| "self is inf"), 
			self.is_infinity.clone(), 
			false_repr.clone()
		)?);
		let other_is_not_inf = Boolean::from(alloc_num_equals(
			cs.namespace(|| "other is inf"), 
			other.is_infinity.clone(), 
			false_repr.clone()
		)?);
		
		//Now compute the output x 
		let inner_x = conditionally_select(
			cs.namespace(|| "final x: inner if"),
			&x,
			&self.x,
			&other_is_not_inf
		)?;
		let final_x = conditionally_select(
			cs.namespace(|| "final x: outer if"),
			&inner_x,
			&other.x,
			&self_is_not_inf
		)?;
		
		//The output y
		let inner_y = conditionally_select(
			cs.namespace(|| "final y: inner if"),
			&y,
			&self.y,
			&other_is_not_inf
		)?;
		let final_y = conditionally_select(
			cs.namespace(|| "final y: outer if"),
			&inner_y,
			&other.y,
			&self_is_not_inf
		)?;
		
		//The output is_infinity
		let inner_is_infinity = conditionally_select(
			cs.namespace(|| "final is infinity: inner if"),
			&is_infinity,
			&self.is_infinity,
			&other_is_not_inf
		)?;
		let final_is_infinity = conditionally_select(
			cs.namespace(|| "final is infinity: outer if"),
			&inner_is_infinity,
			&other.is_infinity,
			&self_is_not_inf
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
   
	 	//Compute tmp = (Fp::one() + Fp::one())* self.y
		let tmp = AllocatedNum::alloc(
			cs.namespace(|| "tmp"),
			|| Ok((*self.y.get_value().get()? + *self.y.get_value().get()?))
		)?;
		cs.enforce(
			|| "Compute tmp",
			|lc| lc + CS::one() + CS::one(),
			|lc| lc + self.y.get_variable(),
			|lc| lc + tmp.get_variable(),
		);

		//Compute inv = tmp.invert
		let tmp_inv = AllocatedNum::alloc(
			cs.namespace(|| "tmp inverse"),
			|| {
				let inv = (*tmp.get_value().get()?).invert();
				if inv.is_some().unwrap_u8() == 1 {
					Ok(inv.unwrap())
				}else{
					Err(SynthesisError::DivisionByZero)
				}
			},
		)?;	
		cs.enforce(
			|| "Check inverse",
			|lc| lc + tmp.get_variable(),
			|lc| lc + tmp_inv.get_variable(),
			|lc| lc + CS::one()
		);
		
		//Now compute lambda as (Fp::one() + Fp::one + Fp::one()) * self.x * self.x * tmp_inv
		let prod_1 = AllocatedNum::alloc(
			cs.namespace(|| "alloc prod 1"),
			|| Ok(*tmp_inv.get_value().get()? * self.x.get_value().get()?)
		)?;
		cs.enforce(
			|| "Check prod 1",
			|lc| lc + self.x.get_variable(),
			|lc| lc + tmp_inv.get_variable(),
			|lc| lc + prod_1.get_variable()
		);

		let prod_2 = AllocatedNum::alloc(
			cs.namespace(|| "alloc prod 2"),
			|| Ok(*prod_1.get_value().get()? * self.x.get_value().get()?)
		)?;
		cs.enforce(
			|| "Check prod 2",
			|lc| lc + self.x.get_variable(),
			|lc| lc + prod_1.get_variable(),
			|lc| lc + prod_2.get_variable()
		);

		let lambda = AllocatedNum::alloc(
			cs.namespace(|| "lambda"),
			|| Ok(*prod_2.get_value().get()? * (Fp::one() + Fp::one() + Fp::one()))
		)?;
		cs.enforce(
			|| "Check lambda",
			|lc| lc + CS::one() + CS::one() + CS::one(),
			|lc| lc + prod_2.get_variable(),
			|lc| lc + lambda.get_variable()
		);

		
		/*************************************************************/
		//          x = lambda * lambda - self.x - self.x;
		/*************************************************************/
   
	 	let x = AllocatedNum::alloc(
			cs.namespace(|| "x"),
			|| Ok(((*lambda.get_value().get()?) * (*lambda.get_value().get()?)) - *self.x.get_value().get()? - self.x.get_value().get()?)
		)?;
		cs.enforce(
			|| "Check x",
			|lc| lc + lambda.get_variable(),
			|lc| lc + lambda.get_variable(),
			|lc| lc + x.get_variable() + self.x.get_variable() + self.x.get_variable()
		);
		
		/*************************************************************/
    //        y = lambda * (self.x - x) - self.y;
		/*************************************************************/
   
		let y = AllocatedNum::alloc(
			cs.namespace(|| "y"),
			|| Ok((*lambda.get_value().get()?) * (*self.x.get_value().get()? - x.get_value().get()?) - self.y.get_value().get()?)
		)?;
		cs.enforce(
			|| "Check y",
			|lc| lc + lambda.get_variable(),
			|lc| lc + self.x.get_variable() - x.get_variable(),
			|lc| lc + y.get_variable() + self.y.get_variable()
		);

		/*************************************************************/
		//Only return the computed x and y if the point is not infinity
		/*************************************************************/
		
		let zero_repr = AllocatedNum::alloc(cs.namespace(|| "false num"), || Ok(Fp::zero()))?;
		let self_is_not_inf = Boolean::from(alloc_num_equals(
			cs.namespace(|| "self is inf"), 
			self.is_infinity.clone(), 
			zero_repr.clone()
		)?);
	
		//x 
		let final_x = conditionally_select(
			cs.namespace(|| "final x"),
			&x,
			&zero_repr,
			&self_is_not_inf
		)?;
	
		//y
		let final_y = conditionally_select(
			cs.namespace(|| "final y"),
			&y,
			&zero_repr,
			&self_is_not_inf
		)?;
		
		//is_infinity
		let final_is_infinity = self.is_infinity.clone();

		Ok(Self::new(final_x, final_y, final_is_infinity))
  }

	/*
  #[allow(dead_code)]
  pub fn scalar_mul_mont(&self, scalar: &Fq) -> Self {
    let mut R0 = Self {
      x: Fp::zero(),
      y: Fp::zero(),
      is_infinity: true,
      _p: Default::default(),
    };

    let mut R1 = self.clone();
    let bits = scalar.to_le_bits();
    for i in (0..bits.len()).rev() {
      if bits[i] {
        R0 = R0.add(&R1);
        R1 = R1.double();
      } else {
        R1 = R0.add(&R1);
        R0 = R0.double();
      }
    }
    R0
  }

  #[allow(dead_code)]
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
  }*/
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use pasta_curves::arithmetic::CurveAffine;
  use pasta_curves::group::Curve;
  use pasta_curves::EpAffine;
  use std::ops::Mul;
	use crate::bellperson::solver::SatisfyingAssignment; 
	use crate::bellperson::shape_cs::ShapeCS; 
	type G = pasta_curves::pallas::Point;
	type Fq = pasta_curves::pallas::Base;
	type Fp = pasta_curves::pallas::Scalar;
	use crate::gadgets::ecc::Point as Pp;
	use crate::bellperson::r1cs::{NovaShape, NovaWitness};
	use crate::r1cs;

  #[test]
  fn test_ecc_add_and_double() {
		//First create the shape
		let mut cs: ShapeCS<G> = ShapeCS::new();
    let a = AllocatedPoint::<Fp,Fq>::random_vartime(cs.namespace(|| "a")).unwrap();
		let _ = a.inputize(cs.namespace(|| "inputize a")).unwrap(); //Need to make something input so that the compiler is not complaining
    let b = AllocatedPoint::<Fp,Fq>::random_vartime(cs.namespace(|| "b")).unwrap();
		let _ = b.inputize(cs.namespace(|| "inputize n")).unwrap(); //Need to make something input so that the compiler is not complaining
    let c = a.add(cs.namespace(|| "c"), &b).unwrap();
		let d = a.double(cs.namespace(|| "d")).unwrap();
		let shape = cs.r1cs_shape();
   	let gens = cs.r1cs_gens(); 

		// perform some curve arithmetic
		let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let a = AllocatedPoint::<Fp,Fq>::random_vartime(cs.namespace(|| "a")).unwrap();
		let _ = a.inputize(cs.namespace(|| "inputize a")).unwrap(); //Need to make something input so that the compiler is not complaining
    let b = AllocatedPoint::<Fp,Fq>::random_vartime(cs.namespace(|| "b")).unwrap();
		let _ = b.inputize(cs.namespace(|| "inputize n")).unwrap(); //Need to make something input so that the compiler is not complaining
    let c = a.add(cs.namespace(|| "c"), &b).unwrap();
    let d = a.double(cs.namespace(|| "d")).unwrap();
		
		let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &gens).unwrap();
		
		assert!(shape.is_sat(&gens, &inst, &witness).is_ok());

		//Now create points out of the circuit to make sure we have not messed anythin up
		let a_p = Pp::<Fp, Fq>::new(a.x.get_value().unwrap(), a.y.get_value().unwrap(), a.is_infinity.get_value().unwrap() == Fp::one());
		let b_p = Pp::<Fp, Fq>::new(b.x.get_value().unwrap(), b.y.get_value().unwrap(), b.is_infinity.get_value().unwrap() == Fp::one());
		let c_p = a_p.add(&b_p);
		assert!(c.x.get_value().unwrap() == c_p.x);
		assert!(c.y.get_value().unwrap() == c_p.y);
		let d_p = a_p.double();
		assert!(d.x.get_value().unwrap() == d_p.x);
		assert!(d.y.get_value().unwrap() == d_p.y);
	}	
  /*fn test_ecc_circuit_ops() {
		//Now make sure that the circuit is satisfiable?
		let d = a.double();
    let s = Fq::random(&mut OsRng);
    let e = a.scalar_mul(&s);

    // perform the same computation by translating to pasta_curve types
    let a_pasta = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(a.x.to_repr().0).unwrap(),
      pasta_curves::Fp::from_repr(a.y.to_repr().0).unwrap(),
    )
    .unwrap();
    let b_pasta = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(b.x.to_repr().0).unwrap(),
      pasta_curves::Fp::from_repr(b.y.to_repr().0).unwrap(),
    )
    .unwrap();
    let c_pasta = (a_pasta + b_pasta).to_affine();
    let d_pasta = (a_pasta + a_pasta).to_affine();
    let e_pasta = a_pasta
      .mul(pasta_curves::Fq::from_repr(s.to_repr().0).unwrap())
      .to_affine();

    // transform c, d, and e into pasta_curve types
    let c_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(c.x.to_repr().0).unwrap(),
      pasta_curves::Fp::from_repr(c.y.to_repr().0).unwrap(),
    )
    .unwrap();
    let d_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(d.x.to_repr().0).unwrap(),
      pasta_curves::Fp::from_repr(d.y.to_repr().0).unwrap(),
    )
    .unwrap();
    let e_pasta_2 = EpAffine::from_xy(
      pasta_curves::Fp::from_repr(e.x.to_repr().0).unwrap(),
      pasta_curves::Fp::from_repr(e.y.to_repr().0).unwrap(),
    )
    .unwrap();

    // check that we have the same outputs
    assert_eq!(c_pasta, c_pasta_2);
    assert_eq!(d_pasta, d_pasta_2);
    assert_eq!(e_pasta, e_pasta_2);
  }*/
}
