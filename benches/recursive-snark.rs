#![allow(non_snake_case)]

use criterion::*;
use nova_snark::{
  nova::{PublicParams, RecursiveSNARK},
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{circuit::NonTrivialCircuit, snark::default_ck_hint, Engine},
};
use std::time::Duration;

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type C = NonTrivialCircuit<<E1 as Engine>::Scalar>;

// To run these benchmarks, first download `criterion` with `cargo install cargo install cargo-criterion`.
// Then `cargo criterion --bench recursive-snark`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// For flamegraphs, run `cargo criterion --bench recursive-snark --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = recursive_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof2::criterion::PProfProfiler::new(100, pprof2::criterion::Output::Flamegraph(None)));
      targets = bench_recursive_snark
    }
  } else {
    criterion_group! {
      name = recursive_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_recursive_snark
    }
  }
}

criterion_main!(recursive_snark);

// This should match the value for the primary in test_recursive_circuit_bn256_grumpkin
const NUM_CONS_VERIFIER_CIRCUIT: usize = 9985;
const NUM_SAMPLES: usize = 10;

fn bench_recursive_snark(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
  ]
  .iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT;

    let mut group = c.benchmark_group(format!("RecursiveSNARK-StepCircuitSize-{num_cons}"));
    group.sample_size(NUM_SAMPLES);

    let c = NonTrivialCircuit::new(num_cons);

    // Produce public parameters
    let pp =
      PublicParams::<E1, E2, C>::setup(&c, &*default_ck_hint(), &*default_ck_hint()).unwrap();

    // Bench time to produce a recursive SNARK;
    // we execute a certain number of warm-up steps since executing
    // the first step is cheaper than other steps owing to the presence of
    // a lot of zeros in the satisfying assignment
    let num_warmup_steps = 10;
    let mut recursive_snark: RecursiveSNARK<E1, E2, C> =
      RecursiveSNARK::new(&pp, &c, &[<E1 as Engine>::Scalar::from(2u64)]).unwrap();

    for i in 0..num_warmup_steps {
      let res = recursive_snark.prove_step(&pp, &c);
      assert!(res.is_ok());

      // verify the recursive snark at each step of recursion
      let res = recursive_snark.verify(&pp, i + 1, &[<E1 as Engine>::Scalar::from(2u64)]);
      assert!(res.is_ok());
    }

    group.bench_function("Prove", |b| {
      b.iter(|| {
        // produce a recursive SNARK for a step of the recursion
        assert!(black_box(&mut recursive_snark.clone())
          .prove_step(black_box(&pp), black_box(&c))
          .is_ok());
      })
    });

    // Benchmark the verification time
    group.bench_function("Verify", |b| {
      b.iter(|| {
        assert!(black_box(&recursive_snark)
          .verify(
            black_box(&pp),
            black_box(num_warmup_steps),
            black_box(&[<E1 as Engine>::Scalar::from(2u64)]),
          )
          .is_ok());
      });
    });
    group.finish();
  }
}
