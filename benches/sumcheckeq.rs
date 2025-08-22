//! Benchmarks Nova's prover for sumcheck involving equality polynomials.
//! The optimization is described in Section 5 of https://eprint.iacr.org/2025/1117.
#![allow(non_snake_case)]
use criterion::*;
use ff::Field;
use nova_snark::{
  provider::Bn256EngineKZG,
  spartan::{
    ppsnark::{MemorySumcheckInstance, OuterSumcheckInstance},
    SumcheckEngine,
  },
  traits::Engine,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::hint::black_box;
use std::time::Duration;

type E = Bn256EngineKZG;

criterion_group! {
name = sumcheckeq;
config = Criterion::default().warm_up_time(Duration::from_millis(3000));
targets = bench_sumcheckeq
}

criterion_main!(sumcheckeq);

fn run_sc<E: Engine, S: SumcheckEngine<E>>(
  rs: &[E::Scalar],
  instance: &mut S,
) -> Vec<Vec<Vec<E::Scalar>>> {
  let mut res = Vec::new();
  for r in rs {
    res.push(instance.evaluation_points());
    instance.bound(r);
  }
  res
}

fn bench_sumcheckeq(c: &mut Criterion) {
  const MAX_NUM_VARS: usize = 26;

  let rs = (0..MAX_NUM_VARS)
    .map(|i| -<E as Engine>::Scalar::from(i as u64))
    .collect::<Vec<_>>();
  let taus = (0..MAX_NUM_VARS)
    .map(|i| -<E as Engine>::Scalar::from(i as u64 * 2))
    .collect::<Vec<_>>();
  let vs_a = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i as u64))
    .collect::<Vec<_>>();
  let vs_b = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i * 2_u64))
    .collect::<Vec<_>>();
  let vs_c = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i * 3_u64))
    .collect::<Vec<_>>();
  let vs_d = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i * 4_u64))
    .collect::<Vec<_>>();
  let vs_e = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i * 5_u64))
    .collect::<Vec<_>>();
  let vs_f = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i * 6_u64))
    .collect::<Vec<_>>();
  let vs_g = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i * 7_u64))
    .collect::<Vec<_>>();
  let vs_h = (0..1 << MAX_NUM_VARS)
    .into_par_iter()
    .map(|i| <E as Engine>::Scalar::from(i * 8_u64))
    .collect::<Vec<_>>();

  for i in 3..MAX_NUM_VARS {
    let mut group = c.benchmark_group(format!("NovaProve-PPSNARK-SumCheckEq-len-{}", i));
    group.sample_size(20);

    group.bench_function("ProveOuter", |b| {
      b.iter(|| {
        let (rs, taus, vs_a, vs_b, vs_c, vs_d) = black_box({
          let len = black_box(1 << i);
          (
            rs[..i].to_vec(),
            taus[..i].to_vec(),
            vs_a[..len].to_vec(),
            vs_b[..len].to_vec(),
            vs_c[..len].to_vec(),
            vs_d[..len].to_vec(),
          )
        });

        black_box({
          let mut instance = OuterSumcheckInstance::<E>::new(
            taus,
            vs_a,
            vs_b,
            vs_c,
            vs_d,
            &<E as Engine>::Scalar::ZERO,
          );
          run_sc(&rs, &mut instance)
        });
      });
    });

    group.finish();

    let mut group = c.benchmark_group(format!("NovaProve-PPSNARK-SumCheckEq-len-{}", i));
    group.sample_size(20);

    group.bench_function("ProveMemory", |b| {
      b.iter(|| {
        let (rs, taus, polys1, polys2, poly3, poly4) = black_box({
          let len = black_box(1 << i);
          (
            rs[..i].to_vec(),
            taus[..i].to_vec(),
            [
              vs_a[..len].to_vec(),
              vs_b[..len].to_vec(),
              vs_c[..len].to_vec(),
              vs_d[..len].to_vec(),
            ],
            [
              vs_e[..len].to_vec(),
              vs_f[..len].to_vec(),
              vs_g[..len].to_vec(),
              vs_h[..len].to_vec(),
            ],
            vs_a[..len].to_vec(),
            vs_b[..len].to_vec(),
          )
        });

        black_box({
          let mut instance = MemorySumcheckInstance::<E>::new(polys1, polys2, taus, poly3, poly4);
          run_sc(&rs, &mut instance)
        });
      });
    });

    group.finish();
  }
}
