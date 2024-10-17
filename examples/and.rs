//! This example executes a batch of 64-bit AND operations.
//! It performs the AND operation by first decomposing the operands into bits and then performing the operation bit-by-bit.
//! We execute a configurable number of AND operations per step of Nova's recursion.
use bellpepper_core::{
  boolean::AllocatedBit, num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError,
};
use core::marker::PhantomData;
use ff::Field;
use ff::{PrimeField, PrimeFieldBits};
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::RelaxedR1CSSNARKTrait,
    Engine, Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use rand::Rng;
use std::time::Instant;

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

#[derive(Clone, Debug)]
struct AndInstance<G: Group> {
  a: u64,
  b: u64,
  _p: PhantomData<G>,
}

impl<G: Group> AndInstance<G> {
  // produces an AND instance
  fn new() -> Self {
    let mut rng = rand::thread_rng();
    let a: u64 = rng.gen();
    let b: u64 = rng.gen();
    Self {
      a,
      b,
      _p: PhantomData,
    }
  }
}

#[derive(Clone, Debug)]
struct AndCircuit<G: Group> {
  batch: Vec<AndInstance<G>>,
}

impl<G: Group> AndCircuit<G> {
  // produces a batch of AND instances
  fn new(num_ops_per_step: usize) -> Self {
    let mut batch = Vec::new();
    for _ in 0..num_ops_per_step {
      batch.push(AndInstance::new());
    }
    Self { batch }
  }
}

pub fn u64_into_bit_vec_le<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  mut cs: CS,
  value: Option<u64>,
) -> Result<Vec<AllocatedBit>, SynthesisError> {
  let values = match value {
    Some(ref value) => {
      let mut tmp = Vec::with_capacity(64);

      for i in 0..64 {
        tmp.push(Some(*value >> i & 1 == 1));
      }

      tmp
    }
    None => vec![None; 64],
  };

  let bits = values
    .into_iter()
    .enumerate()
    .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("bit {}", i)), b))
    .collect::<Result<Vec<_>, SynthesisError>>()?;

  Ok(bits)
}

/// Gets as input the little indian representation of a number and spits out the number
pub fn le_bits_to_num<Scalar, CS>(
  mut cs: CS,
  bits: &[AllocatedBit],
) -> Result<AllocatedNum<Scalar>, SynthesisError>
where
  Scalar: PrimeField + PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  // We loop over the input bits and construct the constraint
  // and the field element that corresponds to the result
  let mut lc = LinearCombination::zero();
  let mut coeff = Scalar::ONE;
  let mut fe = Some(Scalar::ZERO);
  for bit in bits.iter() {
    lc = lc + (coeff, bit.get_variable());
    fe = bit.get_value().map(|val| {
      if val {
        fe.unwrap() + coeff
      } else {
        fe.unwrap()
      }
    });
    coeff = coeff.double();
  }
  let num = AllocatedNum::alloc(cs.namespace(|| "Field element"), || {
    fe.ok_or(SynthesisError::AssignmentMissing)
  })?;
  lc = lc - num.get_variable();
  cs.enforce(|| "compute number from bits", |lc| lc, |lc| lc, |_| lc);
  Ok(num)
}

impl<G: Group> StepCircuit<G::Scalar> for AndCircuit<G> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<G::Scalar>>(
    &self,
    cs: &mut CS,
    z_in: &[AllocatedNum<G::Scalar>],
  ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
    for i in 0..self.batch.len() {
      // allocate a and b as field elements
      let a = AllocatedNum::alloc(cs.namespace(|| format!("a_{}", i)), || {
        Ok(G::Scalar::from(self.batch[i].a))
      })?;
      let b = AllocatedNum::alloc(cs.namespace(|| format!("b_{}", i)), || {
        Ok(G::Scalar::from(self.batch[i].b))
      })?;

      // obtain bit representations of a and b
      let a_bits = u64_into_bit_vec_le(
        cs.namespace(|| format!("a_bits_{}", i)),
        Some(self.batch[i].a),
      )?; // little endian
      let b_bits = u64_into_bit_vec_le(
        cs.namespace(|| format!("b_bits_{}", i)),
        Some(self.batch[i].b),
      )?; // little endian

      // enforce that bits of a and b are correct
      let a_from_bits = le_bits_to_num(cs.namespace(|| format!("a_{}", i)), &a_bits)?;
      let b_from_bits = le_bits_to_num(cs.namespace(|| format!("b_{}", i)), &b_bits)?;

      cs.enforce(
        || format!("a_{} == a_from_bits", i),
        |lc| lc + a.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + a_from_bits.get_variable(),
      );
      cs.enforce(
        || format!("b_{} == b_from_bits", i),
        |lc| lc + b.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + b_from_bits.get_variable(),
      );

      let mut c_bits = Vec::new();

      // perform bitwise AND
      for i in 0..64 {
        let c_bit = AllocatedBit::and(
          cs.namespace(|| format!("and_bit_{}", i)),
          &a_bits[i],
          &b_bits[i],
        )?;
        c_bits.push(c_bit);
      }

      // convert back to an allocated num
      let c_from_bits = le_bits_to_num(cs.namespace(|| format!("c_{}", i)), &c_bits)?;

      let c = AllocatedNum::alloc(cs.namespace(|| format!("c_{}", i)), || {
        Ok(G::Scalar::from(self.batch[i].a & self.batch[i].b))
      })?;

      // enforce that c is correct
      cs.enforce(
        || format!("c_{} == c_from_bits", i),
        |lc| lc + c.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + c_from_bits.get_variable(),
      );
    }

    Ok(z_in.to_vec())
  }
}

/// cargo run --release --example and
fn main() {
  println!("=========================================================");
  println!("Nova-based 64-bit bitwise AND example");
  println!("=========================================================");

  let num_steps = 32;
  for num_ops_per_step in [1024, 2048, 4096, 8192, 16384, 32768, 65536] {
    // number of instances of AND per Nova's recursive step
    let circuit_primary = AndCircuit::new(num_ops_per_step);
    let circuit_secondary = TrivialCircuit::default();

    println!(
      "Proving {} AND ops ({num_ops_per_step} instances per step and {num_steps} steps)",
      num_ops_per_step * num_steps
    );

    // produce public parameters
    let start = Instant::now();
    println!("Producing public parameters...");
    let pp = PublicParams::<
      E1,
      E2,
      AndCircuit<<E1 as Engine>::GE>,
      TrivialCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &circuit_primary,
      &circuit_secondary,
      &*S1::ck_floor(),
      &*S2::ck_floor(),
    )
    .unwrap();
    println!("PublicParams::setup, took {:?} ", start.elapsed());

    println!(
      "Number of constraints per step (primary circuit): {}",
      pp.num_constraints().0
    );
    println!(
      "Number of constraints per step (secondary circuit): {}",
      pp.num_constraints().1
    );

    println!(
      "Number of variables per step (primary circuit): {}",
      pp.num_variables().0
    );
    println!(
      "Number of variables per step (secondary circuit): {}",
      pp.num_variables().1
    );

    // produce non-deterministic advice
    let circuits = (0..num_steps)
      .map(|_| AndCircuit::new(num_ops_per_step))
      .collect::<Vec<_>>();

    type C1 = AndCircuit<<E1 as Engine>::GE>;
    type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;

    // produce a recursive SNARK
    println!("Generating a RecursiveSNARK...");
    let mut recursive_snark: RecursiveSNARK<E1, E2, C1, C2> =
      RecursiveSNARK::<E1, E2, C1, C2>::new(
        &pp,
        &circuits[0],
        &circuit_secondary,
        &[<E1 as Engine>::Scalar::zero()],
        &[<E2 as Engine>::Scalar::zero()],
      )
      .unwrap();

    let start = Instant::now();
    for circuit_primary in circuits.iter() {
      let res = recursive_snark.prove_step(&pp, circuit_primary, &circuit_secondary);
      assert!(res.is_ok());
    }
    println!(
      "RecursiveSNARK::prove {} AND ops: took {:?} ",
      num_ops_per_step * num_steps,
      start.elapsed()
    );

    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    println!("RecursiveSNARK::verify: {:?}", res.is_ok(),);
    assert!(res.is_ok());

    // produce a compressed SNARK
    println!("Generating a CompressedSNARK using Spartan with HyperKZG...");
    let (pk, vk) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

    let start = Instant::now();

    let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
    println!(
      "CompressedSNARK::prove: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
    let compressed_snark_encoded = encoder.finish().unwrap();
    println!(
      "CompressedSNARK::len {:?} bytes",
      compressed_snark_encoded.len()
    );

    // verify the compressed SNARK
    println!("Verifying a CompressedSNARK...");
    let start = Instant::now();
    let res = compressed_snark.verify(
      &vk,
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    println!(
      "CompressedSNARK::verify: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());
    println!("=========================================================");
  }
}
