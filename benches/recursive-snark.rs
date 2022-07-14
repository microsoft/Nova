#![allow(non_snake_case)]

use criterion::*;
use nova_snark::{
  traits::{circuit::TrivialTestCircuit, Group},
  PublicParams, RecursiveSNARK,
};
use std::time::Duration;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type C1 = TrivialTestCircuit<<G1 as Group>::Scalar>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;

fn recursive_snark_benchmark(c: &mut Criterion) {
  let num_samples = 10;
  bench_recursive_snark(c, num_samples);
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

fn bench_recursive_snark(c: &mut Criterion, num_samples: usize) {
  let mut group = c.benchmark_group("RecursiveSNARK".to_string());
  group.sample_size(num_samples);

  // Produce public parameters
  let pp = PublicParams::<G1, G2, C1, C2>::setup(
    TrivialTestCircuit::default(),
    TrivialTestCircuit::default(),
  );

  // Bench time to produce a recursive SNARK;
  // we execute a certain number of warm-up steps since executing
  // the first step is cheaper than other steps owing to the presence of
  // a lot of zeros in the satisfying assignment
  let num_warmup_steps = 10;
  let mut recursive_snark: Option<RecursiveSNARK<G1, G2, C1, C2>> = None;

  for i in 0..num_warmup_steps {
    let res = RecursiveSNARK::prove_step(
      &pp,
      recursive_snark,
      TrivialTestCircuit::default(),
      TrivialTestCircuit::default(),
      <G1 as Group>::Scalar::one(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());
    let recursive_snark_unwrapped = res.unwrap();

    // verify the recursive snark at each step of recursion
    let res = recursive_snark_unwrapped.verify(
      &pp,
      i + 1,
      <G1 as Group>::Scalar::one(),
      <G2 as Group>::Scalar::zero(),
    );
    assert!(res.is_ok());

    // set the running variable for the next iteration
    recursive_snark = Some(recursive_snark_unwrapped);
  }

  group.bench_function("Prove", |b| {
    b.iter(|| {
      // produce a recursive SNARK for a step of the recursion
      assert!(RecursiveSNARK::prove_step(
        black_box(&pp),
        black_box(recursive_snark.clone()),
        black_box(TrivialTestCircuit::default()),
        black_box(TrivialTestCircuit::default()),
        black_box(<G1 as Group>::Scalar::zero()),
        black_box(<G2 as Group>::Scalar::zero()),
      )
      .is_ok());
    })
  });

  let recursive_snark = recursive_snark.unwrap();

  // Benchmark the verification time
  let name = "Verify";
  group.bench_function(name, |b| {
    b.iter(|| {
      assert!(black_box(&recursive_snark)
        .verify(
          black_box(&pp),
          black_box(num_warmup_steps),
          black_box(<G1 as Group>::Scalar::zero()),
          black_box(<G2 as Group>::Scalar::zero()),
        )
        .is_ok());
    });
  });
  group.finish();
}
