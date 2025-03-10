#![allow(non_snake_case)]

use criterion::{measurement::WallTime, *};
use nova_snark::{
  nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait, Engine},
};
use std::time::Duration;

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
// SNARKs without computational commitments
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>;
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;
// SNARKs with computational commitments for the primary curve
type SS1 = nova_snark::spartan::ppsnark::RelaxedR1CSSNARK<E1, EE1>;
type SS2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;
type C = NonTrivialCircuit<<E1 as Engine>::Scalar>;

// To run these benchmarks, first download `criterion` with `cargo install cargo install cargo-criterion`.
// Then `cargo criterion --bench compressed-snark`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// For flamegraphs, run `cargo criterion --bench compressed-snark --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = compressed_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof2::criterion::PProfProfiler::new(100, pprof2::criterion::Output::Flamegraph(None)));
      targets = bench_compressed_snark, bench_compressed_snark_with_computational_commitments
    }
  } else {
    criterion_group! {
      name = compressed_snark;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_compressed_snark, bench_compressed_snark_with_computational_commitments,
    }
  }
}

criterion_main!(compressed_snark);

// This should match the value for the primary in test_recursive_circuit_bn256_grumpkin
const NUM_CONS_VERIFIER_CIRCUIT: usize = 9985;
const NUM_SAMPLES: usize = 10;

/// Benchmarks the compressed SNARK at a provided number of constraints
///
/// Parameters
/// - `group``: the criterion benchmark group
/// - `num_cons`: the number of constraints in the step circuit
fn bench_compressed_snark_internal<S1: RelaxedR1CSSNARKTrait<E1>, S2: RelaxedR1CSSNARKTrait<E2>>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  num_cons: usize,
) {
  let c = NonTrivialCircuit::new(num_cons);

  // Produce public parameters
  let pp = PublicParams::<E1, E2, C>::setup(&c, &*S1::ck_floor(), &*S2::ck_floor()).unwrap();

  // Produce prover and verifier keys for CompressedSNARK
  let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp).unwrap();

  // produce a recursive SNARK
  let num_steps = 3;
  let mut recursive_snark: RecursiveSNARK<E1, E2, C> =
    RecursiveSNARK::new(&pp, &c, &[<E1 as Engine>::Scalar::from(2u64)]).unwrap();

  for i in 0..num_steps {
    let res = recursive_snark.prove_step(&pp, &c);
    assert!(res.is_ok());

    // verify the recursive snark at each step of recursion
    let res = recursive_snark.verify(&pp, i + 1, &[<E1 as Engine>::Scalar::from(2u64)]);
    assert!(res.is_ok());
  }

  // Bench time to produce a compressed SNARK
  group.bench_function("Prove", |b| {
    b.iter(|| {
      assert!(CompressedSNARK::<_, _, _, S1, S2>::prove(
        black_box(&pp),
        black_box(&pk),
        black_box(&recursive_snark),
      )
      .is_ok());
    })
  });
  let res = CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  // Benchmark the verification time
  group.bench_function("Verify", |b| {
    b.iter(|| {
      assert!(black_box(&compressed_snark)
        .verify(
          black_box(&vk),
          black_box(num_steps),
          black_box(&[<E1 as Engine>::Scalar::from(2u64)]),
        )
        .is_ok());
    })
  });
}

fn bench_compressed_snark(c: &mut Criterion) {
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

    let mut group = c.benchmark_group(format!("CompressedSNARK-StepCircuitSize-{num_cons}"));
    group.sample_size(NUM_SAMPLES);

    bench_compressed_snark_internal::<S1, S2>(&mut group, num_cons);

    group.finish();
  }
}

fn bench_compressed_snark_with_computational_commitments(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT,
    16384,
    32768,
    65536,
    131072,
    262144,
  ]
  .iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT;

    let mut group = c.benchmark_group(format!(
      "CompressedSNARK-Commitments-StepCircuitSize-{num_cons}"
    ));
    group
      .sampling_mode(SamplingMode::Flat)
      .sample_size(NUM_SAMPLES);

    bench_compressed_snark_internal::<SS1, SS2>(&mut group, num_cons);

    group.finish();
  }
}
