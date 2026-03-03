//! Single-run benchmark for ppsnark prove/verify across a range of circuit sizes.
//! Use this for quick A/B comparisons between branches.
//!
//! Run with: `cargo run --release --example ppsnark_bench`
#![allow(non_snake_case)]

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
  let sizes = [8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576];

  println!(
    "{:<12} {:>12} {:>12} {:>12}",
    "num_cons", "setup (ms)", "prove (ms)", "verify (ms)"
  );
  println!("{}", "-".repeat(52));

  for &num_cons in &sizes {
    let circuit = NonTrivialCircuit::<<E as Engine>::Scalar>::new(num_cons);
    let input = vec![<E as Engine>::Scalar::from(42)];

    // Setup
    let start = Instant::now();
    let (pk, vk) =
      DirectSNARK::<E, S, NonTrivialCircuit<<E as Engine>::Scalar>>::setup(circuit.clone())
        .unwrap();
    let setup_ms = start.elapsed().as_millis();

    // Prove
    let start = Instant::now();
    let snark = DirectSNARK::prove(&pk, circuit.clone(), &input).unwrap();
    let prove_ms = start.elapsed().as_millis();

    // Compute expected output for verification
    let mut x = <E as Engine>::Scalar::from(42);
    for _ in 0..num_cons {
      x = x * x;
    }
    let io: Vec<_> = input.iter().copied().chain(std::iter::once(x)).collect();

    // Verify
    let start = Instant::now();
    let res = snark.verify(&vk, &io);
    let verify_ms = start.elapsed().as_millis();
    assert!(res.is_ok());

    println!(
      "{:<12} {:>12} {:>12} {:>12}",
      num_cons, setup_ms, prove_ms, verify_ms
    );
  }
}
