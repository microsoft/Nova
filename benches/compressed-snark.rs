#![allow(non_snake_case)]

extern crate flate2;

use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
  traits::{Group, StepCircuit},
  CompressedSNARK, PublicParams, RecursiveSNARK,
};

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

use bellperson::{
  gadgets::{num::AllocatedNum, Assignment},
  ConstraintSystem, SynthesisError,
};
use criterion::*;
use ff::PrimeField;
use rand::rngs::OsRng;
use std::time::Duration;

fn compressed_snark_benchmark(c: &mut Criterion) {
  let num_samples = 10;
  let num_steps = 10;
  let num_cons = 0;
  bench_compressed_snark(c, num_samples, num_steps, num_cons);
}

fn set_duration() -> Criterion {
  Criterion::default().warm_up_time(Duration::from_millis(3000))
}

criterion_group! {
name = compressed_snark;
config = set_duration();
targets = compressed_snark_benchmark
}

criterion_main!(compressed_snark);

fn bench_compressed_snark(
  c: &mut Criterion,
  num_samples: usize,
  num_steps: usize,
  num_cons: usize,
) {
  let name = format!("CompressedSNARK-NumCons-{}", num_cons);
  let mut group = c.benchmark_group(name.clone());
  group.sample_size(num_samples);
  // Produce public parameters
  let pp = PublicParams::<
    G1,
    G2,
    TestCircuit<<G1 as Group>::Scalar>,
    TestCircuit<<G2 as Group>::Scalar>,
  >::setup(TestCircuit::new(num_cons), TestCircuit::new(0));

  // produce a recursive SNARK
  let res = RecursiveSNARK::prove(
    &pp,
    num_steps,
    <G1 as Group>::Scalar::zero(),
    <G2 as Group>::Scalar::zero(),
  );
  assert!(res.is_ok());
  let recursive_snark = res.unwrap();
  // Bench time to produce a compressed SNARK
  group.bench_function("Prove", |b| {
    b.iter(|| {
      assert!(CompressedSNARK::prove(black_box(&pp), black_box(&recursive_snark)).is_ok());
    })
  });
  let res = CompressedSNARK::prove(&pp, &recursive_snark);
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  // Output the proof size
  let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
  bincode::serialize_into(&mut encoder, &compressed_snark.serialize()).unwrap();
  let proof_encoded = encoder.finish().unwrap();
  println!("{}/ProofSize: {} B", name, proof_encoded.len(),);

  // Benchmark the verification time
  let name = "Verify";
  group.bench_function(name, |b| {
    b.iter(|| {
      assert!(black_box(&compressed_snark)
        .verify(
          black_box(&pp),
          black_box(num_steps),
          black_box(<G1 as Group>::Scalar::zero()),
          black_box(<G2 as Group>::Scalar::zero()),
        )
        .is_ok());
    })
  });

  group.finish();
}

// The test circuit has $num_cons constraints. each constraint i is of the form
// (a*x_{i-1} + 1) * b = x_i
// where a and b are random coefficients. x_{-1} = input z and x_n is the output z
#[derive(Clone, Debug)]
struct TestCircuit<F: PrimeField> {
  num_cons: usize,
  coeffs: Vec<(F, F)>,
}

impl<F> TestCircuit<F>
where
  F: PrimeField,
{
  pub fn new(num_cons: usize) -> Self {
    // Generate 2*num_cons random field elements (each constraint has two coefficients
    let coeffs = (0..num_cons)
      .map(|_| (F::random(&mut OsRng), F::random(&mut OsRng)))
      .collect();
    Self {
      num_cons,
      coeffs,
    }
  }
}

impl<F> StepCircuit<F> for TestCircuit<F>
where
  F: PrimeField,
{
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    let mut output = z;
    for i in 0..self.num_cons {
      let a = self.coeffs[i].0;
      let b = self.coeffs[i].1;
      let z_new = AllocatedNum::alloc(cs.namespace(|| format!("alloc x_{}", i)), || {
        Ok((*output.get_value().get()? * a + F::one()) * b)
      })?;
      cs.enforce(
        || format!("Constraint {}", i),
        |lc| lc + (a, output.get_variable()) + CS::one(),
        |lc| lc + (b, CS::one()),
        |lc| lc + z_new.get_variable(),
      );
      output = z_new
    }
    Ok(output)
  }

  fn compute(&self, z: &F) -> F {
    let mut output = *z;
    for i in 0..self.num_cons {
      output = (output * self.coeffs[i].0 + F::one()) * self.coeffs[i].1
    }
    output
  }
}
