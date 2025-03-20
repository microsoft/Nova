//! Benchmarking the commit times for hyperkzg over BN254 field using
//! halo2curves library and the Nova-provided commitment engine, on a range of scalar bit-widths
use core::time::Duration;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use halo2curves::{bn256::Fr as Scalar, ff::Field, msm::msm_best};
use nova_snark::{
  provider::Bn256EngineKZG,
  traits::{commitment::CommitmentEngineTrait, Engine},
};
use rand::Rng;
use rayon::prelude::*;

criterion_group! {
name = commit;
config = Criterion::default().warm_up_time(Duration::from_millis(3000)).sample_size(10);
targets = bench_commit
}

criterion_main!(commit);

fn bench_commit(c: &mut Criterion) {
  type E = Bn256EngineKZG;

  let min = 1 << 20;
  let max = 1 << 24;

  // sample bases for the purpose of testing
  let ck = <E as Engine>::CE::setup(b"test_from_label", max);

  let zero = <E as Engine>::Scalar::zero();

  // random scalars that are in the set {0, 1}
  let scalars_u1 = (0..max)
    .into_par_iter()
    .map(|_| {
      let mut rng = rand::thread_rng();
      rng.gen::<u16>() % 2
    })
    .collect::<Vec<_>>();

  let scalars_u1_field = scalars_u1
    .iter()
    .map(|&x| Scalar::from(x as u64))
    .collect::<Vec<_>>();

  // 10-bit scalars that are in the set {0, ..., 2^10-1}
  let scalars_u10 = (0..max)
    .into_par_iter()
    .map(|_| {
      let mut rng = rand::thread_rng();
      rng.gen::<u16>() % (1 << 10)
    })
    .collect::<Vec<_>>();

  let scalars_u10_field = scalars_u10
    .iter()
    .map(|&x| Scalar::from(x as u64))
    .collect::<Vec<_>>();

  // random scalars that are in the set {0, ..., 2^16-1}
  let scalars_u16 = (0..max)
    .into_par_iter()
    .map(|_| {
      let mut rng = rand::thread_rng();
      rng.gen::<u16>()
    })
    .collect::<Vec<_>>();

  let scalars_u16_field = scalars_u16
    .iter()
    .map(|&x| Scalar::from(x as u64))
    .collect::<Vec<_>>();

  // random scalars that are in the set {0, ..., 2^32-1}
  let scalars_u32 = (0..max)
    .into_par_iter()
    .map(|_| {
      let mut rng = rand::thread_rng();
      rng.gen::<u32>()
    })
    .collect::<Vec<_>>();

  let scalars_u32_field = scalars_u32
    .iter()
    .map(|&x| Scalar::from(x as u64))
    .collect::<Vec<_>>();

  // random scalars that are in the set {0, ..., 2^64-1}
  let scalars_u64 = (0..max)
    .into_par_iter()
    .map(|_| {
      let mut rng = rand::thread_rng();
      rng.gen::<u64>()
    })
    .collect::<Vec<_>>();

  let scalars_u64_field = scalars_u64
    .iter()
    .map(|&x| Scalar::from(x))
    .collect::<Vec<_>>();

  // random scalars in the set {0, ..., p-1}, where p is the modulus for the
  // scalar field of BN254
  let scalars_random_field = (0..max)
    .into_par_iter()
    .map(|_| {
      let mut rng = rand::thread_rng();
      Scalar::random(&mut rng)
    })
    .collect::<Vec<_>>();

  let mut size = min;
  while size <= max {
    c.bench_function(&format!("halo2curves_commit_u1_{size}"), |b| {
      b.iter(|| black_box(msm_best(&scalars_u1_field[..size], &ck.ck()[..size])))
    });

    c.bench_function(&format!("nova_generic_commit_u1_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit(
          &ck,
          &scalars_u1_field[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("nova_specialized_commit_u1_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit_small(
          &ck,
          &scalars_u1[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("halo2curves_commit_u10_{size}"), |b| {
      b.iter(|| black_box(msm_best(&scalars_u10_field[..size], &ck.ck()[..size])))
    });

    c.bench_function(&format!("nova_generic_commit_u10_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit(
          &ck,
          &scalars_u10_field[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("nova_specialized_commit_u10_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit_small(
          &ck,
          &scalars_u10[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("halo2curves_commit_u16_{size}"), |b| {
      b.iter(|| black_box(msm_best(&scalars_u16_field[..size], &ck.ck()[..size])))
    });

    c.bench_function(&format!("nova_generic_commit_u16_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit(
          &ck,
          &scalars_u16_field[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("nova_specialized_commit_u16_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit_small(
          &ck,
          &scalars_u16[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("halo2curves_commit_u32_{size}"), |b| {
      b.iter(|| black_box(msm_best(&scalars_u32_field[..size], &ck.ck()[..size])))
    });

    c.bench_function(&format!("nova_generic_commit_u32_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit(
          &ck,
          &scalars_u32_field[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("nova_specialized_commit_u32_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit_small(
          &ck,
          &scalars_u32[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("halo2curves_commit_u64_{size}"), |b| {
      b.iter(|| black_box(msm_best(&scalars_u64_field[..size], &ck.ck()[..size])))
    });

    c.bench_function(&format!("nova_generic_commit_u64_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit(
          &ck,
          &scalars_u64_field[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("nova_specialized_commit_u64_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit_small(
          &ck,
          &scalars_u64[..size],
          &zero,
        ))
      })
    });

    c.bench_function(&format!("halo2curves_commit_random_{size}"), |b| {
      b.iter(|| black_box(msm_best(&scalars_random_field[..size], &ck.ck()[..size])))
    });

    c.bench_function(&format!("nova_generic_commit_random_{size}"), |b| {
      b.iter(|| {
        black_box(<E as Engine>::CE::commit(
          &ck,
          &scalars_random_field[..size],
          &zero,
        ))
      })
    });

    size *= 4;
  }
}
