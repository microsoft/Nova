#![allow(non_snake_case)]

use criterion::*;
use nova_snark::{
  traits::{circuit::TrivialTestCircuit, Group},
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use std::time::Duration;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;
type C1 = TrivialTestCircuit<<G1 as Group>::Scalar>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;

fn compressed_snark_benchmark(c: &mut Criterion) {
  let num_samples = 10;
  bench_compressed_snark(c, num_samples);
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

fn bench_compressed_snark(c: &mut Criterion, num_samples: usize) {
  let mut group = c.benchmark_group("CompressedSNARK");
  group.sample_size(num_samples);

  // Produce public parameters
  let pp = PublicParams::<G1, G2, C1, C2>::setup(
    TrivialTestCircuit::default(),
    TrivialTestCircuit::default(),
  );

  // produce a recursive SNARK
  let num_steps = 3;
  let mut recursive_snark: Option<RecursiveSNARK<G1, G2, C1, C2>> = None;

  for i in 0..num_steps {
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

  // Bench time to produce a compressed SNARK
  let recursive_snark = recursive_snark.unwrap();
  group.bench_function("Prove", |b| {
    b.iter(|| {
      assert!(CompressedSNARK::<_, _, _, _, S1, S2>::prove(
        black_box(&pp),
        black_box(&recursive_snark)
      )
      .is_ok());
    })
  });
  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &recursive_snark);
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

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
