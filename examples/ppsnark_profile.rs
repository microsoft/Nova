//! Standalone ppsnark benchmark — bypasses IVC, measures raw prove time.
//!
//! Usage:
//!   cargo run --release --features test-utils,sppark --example ppsnark_profile  # GPU
//!   cargo run --release --features test-utils --example ppsnark_profile              # CPU only
use nova_snark::{
  provider::Bn256EngineKZG,
  spartan::direct::DirectSNARK,
  traits::{circuit::NonTrivialCircuit, Engine},
};
use std::time::Instant;

type E = Bn256EngineKZG;
type EE = nova_snark::provider::hyperkzg::EvaluationEngine<E>;
type S = nova_snark::spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

fn main() {
  let num_cons = 1 << 20;
  println!("ppsnark standalone benchmark: num_cons={num_cons}");

  let circuit = NonTrivialCircuit::<<E as Engine>::Scalar>::new(num_cons);
  let input = vec![<E as Engine>::Scalar::from(42u64)];

  let t = Instant::now();
  let (pk, vk) =
    DirectSNARK::<E, S, NonTrivialCircuit<<E as Engine>::Scalar>>::setup(circuit.clone()).unwrap();
  println!("Setup: {:.2}s", t.elapsed().as_secs_f64());

  // Warmup
  println!("\nWarmup prove...");
  let t = Instant::now();
  let _ = DirectSNARK::prove(&pk, circuit.clone(), &input);
  println!("Warmup: {:.2}s", t.elapsed().as_secs_f64());

  // Timed proves
  let num_runs = 3;
  println!("\nTiming {num_runs} ppsnark prove() runs:");
  let mut times = Vec::new();
  for i in 0..num_runs {
    let t = Instant::now();
    let snark = DirectSNARK::prove(&pk, circuit.clone(), &input).unwrap();
    let elapsed = t.elapsed().as_secs_f64();
    times.push(elapsed);
    println!("  Run {}: {:.3}s", i + 1, elapsed);

    if i == num_runs - 1 {
      // Compute expected output for verification
      let mut x = <E as Engine>::Scalar::from(42u64);
      for _ in 0..num_cons {
        x = x * x;
      }
      let io: Vec<_> = vec![<E as Engine>::Scalar::from(42u64), x];
      let t = Instant::now();
      snark.verify(&vk, &io).expect("verification failed");
      println!("  Verify: {:.3}s", t.elapsed().as_secs_f64());
    }
  }

  let avg = times.iter().sum::<f64>() / times.len() as f64;
  let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
  println!("\nProve avg={:.3}s min={:.3}s", avg, min);
}
