#![allow(non_snake_case)]

extern crate flate2;

use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
  traits::{Group, StepCircuit},
  CompressedSNARK, PublicParams, RecursiveSNARK,
};

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::*;
use ff::PrimeField;
use std::time::Duration;

fn compressed_snark_benchmark(c: &mut Criterion) {
  let num_samples = 10;
  let num_steps = 10;
  bench_compressed_snark(c, num_samples, num_steps);
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

fn bench_compressed_snark(c: &mut Criterion, num_samples: usize, num_steps: usize) {
  let mut group = c.benchmark_group("CompressedSNARK");
  group.sample_size(num_samples);
  // Produce public parameters
  let pp = PublicParams::<
    G1,
    G2,
    TrivialTestCircuit<<G1 as Group>::Scalar>,
    TrivialTestCircuit<<G2 as Group>::Scalar>,
  >::setup(
    TrivialTestCircuit {
      _p: Default::default(),
    },
    TrivialTestCircuit {
      _p: Default::default(),
    },
  );

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
  println!("CompressedSNARK/ProofSize: {} B", proof_encoded.len(),);

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

#[derive(Clone, Debug)]
struct TrivialTestCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F> StepCircuit<F> for TrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    Ok(z)
  }

  fn compute(&self, z: &F) -> F {
    *z
  }
}
