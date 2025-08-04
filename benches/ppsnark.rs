#![allow(non_snake_case)]

use criterion::*;
use nova_snark::{
  provider::Bn256EngineKZG,
  spartan::direct::DirectSNARK,
  traits::{circuit::NonTrivialCircuit, Engine},
};
use std::time::Duration;

type E = Bn256EngineKZG;
type EE = nova_snark::provider::hyperkzg::EvaluationEngine<E>;
type S = nova_snark::spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

// To run these benchmarks, first download `criterion` with `cargo install cargo install cargo-criterion`.
// Then `cargo criterion --bench ppsnark`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// For flamegraphs, run `cargo criterion --bench ppsnark --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = ppsnark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof2::criterion::PProfProfiler::new(100, pprof2::criterion::Output::Flamegraph(None)));
      targets = bench_ppsnark
    }
  } else {
    criterion_group! {
      name = ppsnark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_ppsnark
    }
  }
}

criterion_main!(ppsnark);

const NUM_SAMPLES: usize = 10;

fn bench_ppsnark(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons in [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].iter() {
    let mut group = c.benchmark_group(format!("ppsnark-CircuitSize-{num_cons}"));
    group.sample_size(NUM_SAMPLES);

    let c = NonTrivialCircuit::new(num_cons);
    let input = vec![<E as Engine>::Scalar::from(42)];

    // produce keys
    let (pk, vk) =
      DirectSNARK::<E, S, NonTrivialCircuit<<E as Engine>::Scalar>>::setup(c.clone()).unwrap();

    // Bench time to produce a  ppSNARK;
    group.bench_function("Prove", |b| {
      b.iter(|| {
        let res = DirectSNARK::prove(
          black_box(&pk),
          black_box(c.clone()),
          black_box(&[<E as Engine>::Scalar::from(42)]),
        );
        assert!(res.is_ok());
      })
    });

    fn output(num_cons: usize, z: &[<E as Engine>::Scalar]) -> Vec<<E as Engine>::Scalar> {
      let mut x = z[0];
      let mut y = x;
      for _ in 0..num_cons {
        y = x * x;
        x = y;
      }
      vec![y]
    }

    let output = output(num_cons, &[<E as Engine>::Scalar::from(42)]);

    let io = input
      .clone()
      .into_iter()
      .chain(output.clone())
      .collect::<Vec<_>>();

    // Benchmark the verification time
    let ppsnark = DirectSNARK::prove(&pk, c.clone(), &input).unwrap();
    group.bench_function("Verify", |b| {
      b.iter(|| {
        assert!(ppsnark.verify(black_box(&vk), black_box(&io),).is_ok());
      });
    });
    group.finish();
  }
}
