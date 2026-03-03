use nova_snark::{
  provider::Bn256EngineKZG,
  spartan::direct::DirectSNARK,
  traits::{circuit::NonTrivialCircuit, Engine},
};

type E = Bn256EngineKZG;
type EE = nova_snark::provider::mercury::EvaluationEngine<E>;
type S = nova_snark::spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

fn main() {
  let num_cons = 1 << 19; // 524288 constraints -> N ~ 2^21
  eprintln!("Setting up circuit with {} constraints...", num_cons);

  let c = NonTrivialCircuit::new(num_cons);
  let input = vec![<E as Engine>::Scalar::from(42)];

  let (pk, _vk) =
    DirectSNARK::<E, S, NonTrivialCircuit<<E as Engine>::Scalar>>::setup(c.clone()).unwrap();
  eprintln!("Setup done, starting prove...");

  // First prove (cold — allocates GPU buffers)
  let t = std::time::Instant::now();
  let res = DirectSNARK::prove(&pk, c.clone(), &input);
  assert!(res.is_ok());
  eprintln!("\n[1st prove] wall time: {:?}", t.elapsed());

  // Second prove (warm — reuses cached GPU buffers)
  let t = std::time::Instant::now();
  let res = DirectSNARK::prove(&pk, c.clone(), &input);
  assert!(res.is_ok());
  eprintln!("\n[2nd prove] wall time: {:?}", t.elapsed());
}
