//! Benchmarks Nova's prover for proving SHA-256 with varying sized messages.
//! We run a single step with the step performing the entire computation.
//! This code invokes a hand-written SHA-256 gadget from bellman/bellperson.
//! It also uses code from bellman/bellperson to compare circuit-generated digest with sha2 crate's output
#![allow(non_snake_case)]
use core::{marker::PhantomData, time::Duration};
use criterion::*;
use ff::{PrimeField, PrimeFieldBits};
use nova_snark::{
  frontend::{
    num::{AllocatedNum, Num},
    sha256, AllocatedBit, Assignment, Boolean, ConstraintSystem, SynthesisError,
  },
  nova::{PublicParams, RecursiveSNARK},
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{circuit::StepCircuit, snark::default_ck_hint, Engine},
};
use sha2::{Digest, Sha256};

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;

#[derive(Clone, Debug)]
struct Sha256Circuit<Scalar: PrimeField> {
  preimage: Vec<u8>,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField + PrimeFieldBits> Sha256Circuit<Scalar> {
  pub fn new(preimage: Vec<u8>) -> Self {
    Self {
      preimage,
      _p: PhantomData,
    }
  }
}

impl<Scalar: PrimeField + PrimeFieldBits> StepCircuit<Scalar> for Sha256Circuit<Scalar> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _z: &[AllocatedNum<Scalar>],
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    let mut z_out: Vec<AllocatedNum<Scalar>> = Vec::new();

    let bit_values: Vec<_> = self
      .preimage
      .clone()
      .into_iter()
      .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
      .map(Some)
      .collect();
    assert_eq!(bit_values.len(), self.preimage.len() * 8);

    let preimage_bits = bit_values
      .into_iter()
      .enumerate()
      .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
      .map(|b| b.map(Boolean::from))
      .collect::<Result<Vec<_>, _>>()?;

    let hash_bits = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    for (i, hash_bits) in hash_bits.chunks(256_usize).enumerate() {
      let mut num = Num::<Scalar>::zero();
      let mut coeff = Scalar::ONE;
      for bit in hash_bits {
        num = num.add_bool_with_coeff(CS::one(), bit, coeff);

        coeff = coeff.double();
      }

      let hash = AllocatedNum::alloc(cs.namespace(|| format!("input {i}")), || {
        Ok(*num.get_value().get()?)
      })?;

      // num * 1 = hash
      cs.enforce(
        || format!("packing constraint {i}"),
        |_| num.lc(Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + hash.get_variable(),
      );
      z_out.push(hash);
    }

    // sanity check with the hasher
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let hash_result = hasher.finalize();

    let mut s = hash_result
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8));

    for b in hash_bits {
      match b {
        Boolean::Is(b) => {
          assert!(s.next().unwrap() == b.get_value().unwrap());
        }
        Boolean::Not(b) => {
          assert!(s.next().unwrap() != b.get_value().unwrap());
        }
        Boolean::Constant(_b) => {
          panic!("Can't reach here")
        }
      }
    }

    Ok(z_out)
  }
}

type C = Sha256Circuit<<E1 as Engine>::Scalar>;

criterion_group! {
name = recursive_snark;
config = Criterion::default().warm_up_time(Duration::from_millis(3000));
targets = bench_recursive_snark
}

criterion_main!(recursive_snark);

fn bench_recursive_snark(c: &mut Criterion) {
  // Test vectors
  let circuits = vec![
    Sha256Circuit::new(vec![0u8; 1 << 6]),
    Sha256Circuit::new(vec![0u8; 1 << 7]),
    Sha256Circuit::new(vec![0u8; 1 << 8]),
    Sha256Circuit::new(vec![0u8; 1 << 9]),
    Sha256Circuit::new(vec![0u8; 1 << 10]),
    Sha256Circuit::new(vec![0u8; 1 << 11]),
    Sha256Circuit::new(vec![0u8; 1 << 12]),
    Sha256Circuit::new(vec![0u8; 1 << 13]),
    Sha256Circuit::new(vec![0u8; 1 << 14]),
    Sha256Circuit::new(vec![0u8; 1 << 15]),
    Sha256Circuit::new(vec![0u8; 1 << 16]),
  ];

  for circuit in circuits {
    let mut group = c.benchmark_group(format!(
      "NovaProve-Sha256-message-len-{}",
      circuit.preimage.len()
    ));
    group.sample_size(10);

    // Produce public parameters
    let pp =
      PublicParams::<E1, E2, C>::setup(&circuit, &*default_ck_hint(), &*default_ck_hint()).unwrap();

    let z0 = vec![<E1 as Engine>::Scalar::from(2u64)];
    group.bench_function("Prove", |b| {
      b.iter(|| {
        let mut recursive_snark =
          RecursiveSNARK::new(black_box(&pp), black_box(&circuit), black_box(&z0)).unwrap();

        // produce a recursive SNARK for a step of the recursion
        assert!(recursive_snark
          .prove_step(black_box(&pp), black_box(&circuit))
          .is_ok());
      })
    });
    group.finish();
  }
}
