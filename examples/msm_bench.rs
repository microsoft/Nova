//! Standalone MSM benchmark using CE::commit — compares CPU, blitzar, and sppark.
//!
//! Usage:
//!   cargo run --release --features test-utils --example msm_bench                    # CPU
//!   cargo run --release --features test-utils,blitzar --example msm_bench            # Blitzar
//!   cargo run --release --features test-utils,sppark --example msm_bench         # sppark
use nova_snark::{
  provider::Bn256EngineKZG,
  traits::{commitment::CommitmentEngineTrait, Engine},
};
use std::time::Instant;

type E = Bn256EngineKZG;
type CE = <E as Engine>::CE;
type Scalar = <E as Engine>::Scalar;

fn backend_name() -> &'static str {
  #[cfg(feature = "blitzar")]
  { "blitzar" }
  #[cfg(feature = "sppark")]
  { "sppark" }
  #[cfg(not(any(feature = "blitzar", feature = "sppark")))]
  { "cpu" }
}

fn bench_commit(ck: &<CE as CommitmentEngineTrait<E>>::CommitmentKey, n: usize, label: &str) {
  use ff::Field;
  let mut rng = rand::thread_rng();
  let scalars: Vec<Scalar> = (0..n).map(|_| Scalar::random(&mut rng)).collect();
  let r = Scalar::ZERO;

  // Warmup
  let _ = CE::commit(ck, &scalars, &r);

  let runs = 5;
  let mut times = Vec::new();
  for _ in 0..runs {
    let t = Instant::now();
    let _ = CE::commit(ck, &scalars, &r);
    times.push(t.elapsed().as_secs_f64() * 1000.0);
  }

  let avg = times.iter().sum::<f64>() / times.len() as f64;
  let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
  println!("  {label:>30}: avg={avg:>8.1}ms  min={min:>8.1}ms");
}

fn bench_commit_pathological(
  ck: &<CE as CommitmentEngineTrait<E>>::CommitmentKey,
  n: usize,
  label: &str,
) {
  use ff::Field;
  let mut rng = rand::thread_rng();
  // Mostly zeros — triggers concentrated bucket distributions in Pippenger
  let mut scalars = vec![Scalar::ZERO; n];
  for i in 0..std::cmp::min(100, n) {
    scalars[i] = Scalar::random(&mut rng);
  }
  let r = Scalar::ZERO;

  let _ = CE::commit(ck, &scalars, &r);

  let runs = 5;
  let mut times = Vec::new();
  for _ in 0..runs {
    let t = Instant::now();
    let _ = CE::commit(ck, &scalars, &r);
    times.push(t.elapsed().as_secs_f64() * 1000.0);
  }

  let avg = times.iter().sum::<f64>() / times.len() as f64;
  let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
  println!("  {label:>30}: avg={avg:>8.1}ms  min={min:>8.1}ms");
}

fn bench_batch_commit(
  ck: &<CE as CommitmentEngineTrait<E>>::CommitmentKey,
  n: usize,
  batch: usize,
) {
  use ff::Field;
  let mut rng = rand::thread_rng();
  let scalars: Vec<Vec<Scalar>> = (0..batch)
    .map(|_| (0..n).map(|_| Scalar::random(&mut rng)).collect())
    .collect();
  let r: Vec<Scalar> = vec![Scalar::ZERO; batch];

  // Warmup
  let _ = CE::batch_commit(ck, &scalars, &r);

  let runs = 3;
  let mut times = Vec::new();
  for _ in 0..runs {
    let t = Instant::now();
    let _ = CE::batch_commit(ck, &scalars, &r);
    times.push(t.elapsed().as_secs_f64() * 1000.0);
  }

  let avg = times.iter().sum::<f64>() / times.len() as f64;
  let per = avg / batch as f64;
  println!("  batch={batch:>2} x n={n:>8}: total={avg:>8.1}ms  per_commit={per:>6.1}ms");
}

fn main() {
  let n = 1 << 20;
  println!(
    "CE::commit Benchmark — backend: {}  n={}\n",
    backend_name(),
    n
  );

  let t = Instant::now();
  let ck = CE::setup(b"bench", n).unwrap();
  println!("Setup: {:.2}s\n", t.elapsed().as_secs_f64());

  println!("=== Single commit (random scalars, n=2^20) ===");
  bench_commit(&ck, n, "full (n=1048576)");
  bench_commit(&ck, n / 4, "quarter (n=262144)");
  bench_commit(&ck, n / 16, "small (n=65536)");

  println!("\n=== Single commit (pathological scalars, n=2^20) ===");
  bench_commit_pathological(&ck, n, "pathological (n=1048576)");

  println!("\n=== Batch commit (random, n=2^20) ===");
  for &batch in &[4, 8, 16] {
    bench_batch_commit(&ck, n, batch);
  }
}

