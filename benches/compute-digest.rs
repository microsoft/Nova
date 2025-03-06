use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nova_snark::{
  nova::PublicParams,
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{
    circuit::{NonTrivialCircuit, TrivialCircuit},
    snark::default_ck_hint,
    Engine,
  },
};
use std::time::Duration;

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type C1 = NonTrivialCircuit<<E1 as Engine>::Scalar>;
type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;

criterion_group! {
name = compute_digest;
config = Criterion::default().warm_up_time(Duration::from_millis(3000)).sample_size(10);
targets = bench_compute_digest
}

criterion_main!(compute_digest);

fn bench_compute_digest(c: &mut Criterion) {
  c.bench_function("compute_digest", |b| {
    b.iter(|| {
      PublicParams::<E1, E2, C1, C2>::setup(
        black_box(&C1::new(10)),
        black_box(&C2::default()),
        black_box(&*default_ck_hint()),
        black_box(&*default_ck_hint()),
      )
    })
  });
}
