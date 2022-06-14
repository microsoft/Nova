#![allow(non_snake_case)]

extern crate flate2;

//use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
  traits::{Group, StepCircuit},
  PublicParams, RecursiveSNARK,
};

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::*;
use ff::PrimeField;
use std::time::Duration;

fn recursive_snark_benchmark(c: &mut Criterion) {
  let num_samples = 10;
  for num_steps in 1..10 {
    bench_recursive_snark(c, num_samples, num_steps);
  }
}

fn set_duration() -> Criterion {
  Criterion::default().warm_up_time(Duration::from_millis(3000))
}

criterion_group! {
name = recursive_snark;
config = set_duration();
targets = recursive_snark_benchmark
}

criterion_main!(recursive_snark);

fn bench_recursive_snark(c: &mut Criterion, num_samples: usize, num_steps: usize) {
  let mut group = c.benchmark_group(format!("RecursiveSNARK-NumSteps-{}", num_steps));
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
  // Bench time to produce a recursive SNARK
  group.bench_function("Prove", |b| {
    b.iter(|| {
      // produce a recursive SNARK
      let _ = RecursiveSNARK::prove(
        black_box(&pp),
        black_box(num_steps),
        black_box(<G1 as Group>::Scalar::zero()),
        black_box(<G2 as Group>::Scalar::zero()),
      );
    })
  });
  let res = RecursiveSNARK::prove(
    &pp,
    num_steps,
    <G1 as Group>::Scalar::zero(),
    <G2 as Group>::Scalar::zero(),
  );
  assert!(res.is_ok());
  let recursive_snark = res.unwrap();

  // TODO: Output the proof size
  // Benchmark the verification time
  let name = "Verify";
  group.bench_function(name, |b| {
    b.iter(|| {
      let _ = black_box(recursive_snark).verify(
        black_box(&pp),
        black_box(num_steps),
        black_box(<G1 as Group>::Scalar::one()),
        black_box(<G2 as Group>::Scalar::zero()),
      );
    });
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
