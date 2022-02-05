//!This module contains useful gadgets for the verifier circuit
use bellperson::{LinearCombination, ConstraintSystem, SynthesisError, gadgets::{num::AllocatedNum,boolean::Boolean}};
use neptune::{circuit::poseidon_hash, Arity, poseidon::PoseidonConstants, Strength};
use ff::{PrimeField, PrimeFieldBits};
use std::marker::PhantomData;

///A random oracle instantiated with the poseidon hash function of a specified Arity
pub struct PoseidonRO<Scalar, A>
where 
	Scalar: PrimeField + PrimeFieldBits,
	A: Arity<Scalar>
{
	//Internal state
	state: Vec<AllocatedNum<Scalar>>,
	phantom: PhantomData<A>
}

impl<Scalar, A> PoseidonRO <Scalar, A>
where 
	Scalar: PrimeField + PrimeFieldBits,
	A: Arity<Scalar>
{
	
	///Initialize the internal state to 0
	pub fn new() -> Self {
		Self {
			state: Vec::new(),
			phantom: PhantomData
		}
	}

	///Absorb a new number into the state of the oracle
	pub fn absorb(&mut self, e: AllocatedNum<Scalar>){
		self.state.push(e.clone());
	}

	///Compute a challenge by hashing the current state
	pub fn get_challenge<CS>(self, cs: &mut CS) -> Result<AllocatedNum<Scalar>, SynthesisError> 
	where 
		CS: ConstraintSystem<Scalar>
	{
		//Make sure that the size of the state is equal to the arity
		assert_eq!(A::to_usize(), self.state.len());
		//Compute the constants and hash 
		let constants = PoseidonConstants::<Scalar, A>::new_with_strength(Strength::Strengthened); 
		let out = poseidon_hash(cs.namespace(|| "Poseidon hash"), self.state.clone(), &constants)?;
		//Only keep 254 bits. discard the lsb
		let bits = out.to_bits_le_strict(cs.namespace(|| "convert poseidon hash output to bits"))?;
		le_bits_to_num(cs.namespace(|| "le bits to nummber"), bits[1..].into())
	}
}

///Gets as input the little indian representation of a number and spits out the number
pub fn le_bits_to_num<Scalar, CS>(mut cs: CS, bits: Vec<Boolean>) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
	Scalar: PrimeField + PrimeFieldBits,
	CS: ConstraintSystem<Scalar>
{
	//We loop over the input bits and construct the constraint and the field element that corresponds 
	//to the result
	let mut lc = LinearCombination::zero();
	let mut coeff = Scalar::one();
	let mut fe = Some(Scalar::zero());
	for bit in bits.iter() {
		lc = match bit {
			Boolean::Is(ref x) => lc + (coeff, x.get_variable()),
			_ => panic!("The input to le_bits should be of the form Boolean::Is()") 
		};
		fe = bit.get_value().map(|val| if val { fe.unwrap() + coeff } else {fe.unwrap()} );
		coeff = coeff.double();
	}
	let num = AllocatedNum::alloc(cs.namespace(|| "Field element"), || fe.ok_or(SynthesisError::AssignmentMissing))?;
	lc = lc - num.get_variable();
	cs.enforce(|| "compute number from bits", |lc| lc, |lc| lc, |_| lc);
	Ok(num)
}

///Gets as input a field element of F and returns its representation to a field element of G
#[cfg(test)]
mod tests {
	use super::*;
	use crate::bellperson::shape_cs::ShapeCS; 
	use crate::bellperson::solver::SatisfyingAssignment; 
	use generic_array::typenum;
	type S = pasta_curves::pallas::Scalar;
	type G = pasta_curves::pallas::Point;
  use crate::traits::PrimeField;
	use rand::rngs::OsRng;

	#[test]
	fn test_le_bits_to_num(){
		let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();
    let mut csprng: OsRng = OsRng;
		let fe = S::random(&mut csprng);
		let num = AllocatedNum::alloc(cs.namespace(|| "input number"), || Ok(fe)).unwrap();
		let _ = num.inputize(&mut cs).unwrap();
		let bits = num.to_bits_le_strict(&mut cs).unwrap();
		let num2 = le_bits_to_num(&mut cs, bits).unwrap();
		assert!(num2.get_value() == num.get_value());
	}

	#[test]
	fn test_poseidon_ro() {
		let mut ro :PoseidonRO<S, typenum::U8> = PoseidonRO::new();
		let mut cs: ShapeCS<G> = ShapeCS::new(); 
		for i in 0..8 {
			let num = AllocatedNum::alloc(cs.namespace(|| format!("data {}", i)), || Ok(S::zero())).unwrap();
			ro.absorb(num)
		}
		assert!(ro.get_challenge(&mut cs).is_ok());
		println!("Number of constraints {}", cs.num_constraints());
	}
}
