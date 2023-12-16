#![allow(non_snake_case)]
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::{measurement::WallTime, *};
use ff::PrimeField;
use nova_snark::{
  provider::{PallasEngine, VestaEngine},
  supernova::NonUniformCircuit,
  supernova::{PublicParams, RecursiveSNARK},
  traits::{
    circuit_supernova::{StepCircuit, TrivialTestCircuit},
    snark::default_ck_hint,
    Engine,
  },
};
use std::time::Duration;

// To run these benchmarks, first download `criterion` with `cargo install cargo-criterion`.
// Then `cargo criterion --bench recursive-snark-supernova`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// For flamegraphs, run `cargo criterion --bench recursive-snark-supernova --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = recursive_snark_supernova;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
      targets = bench_one_augmented_circuit_recursive_snark, bench_two_augmented_circuit_recursive_snark
    }
  } else {
    criterion_group! {
      name = recursive_snark_supernova;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_one_augmented_circuit_recursive_snark, bench_two_augmented_circuit_recursive_snark
    }
  }
}

criterion_main!(recursive_snark_supernova);

// This should match the value in test_supernova_recursive_circuit_pasta
// TODO: This should also be a table matching the num_augmented_circuits in the below
const NUM_CONS_VERIFIER_CIRCUIT_PRIMARY: usize = 9844;
const NUM_SAMPLES: usize = 10;

struct NonUniformBench<E1, E2, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  S: StepCircuit<E2::Scalar> + Default,
{
  num_circuits: usize,
  num_cons: usize,
  _p: PhantomData<(E1, E2, S)>,
}

impl<E1, E2, S> NonUniformBench<E1, E2, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  S: StepCircuit<E2::Scalar> + Default,
{
  fn new(num_circuits: usize, num_cons: usize) -> Self {
    Self {
      num_circuits,
      num_cons,
      _p: Default::default(),
    }
  }
}

impl<E1, E2, S>
  NonUniformCircuit<E1, E2, NonTrivialTestCircuit<E1::Scalar>, TrivialTestCircuit<E2::Scalar>>
  for NonUniformBench<E1, E2, S>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  S: StepCircuit<E2::Scalar> + Default,
{
  fn num_circuits(&self) -> usize {
    self.num_circuits
  }

  fn primary_circuit(&self, circuit_index: usize) -> NonTrivialTestCircuit<E1::Scalar> {
    assert!(circuit_index < self.num_circuits);

    NonTrivialTestCircuit::new(self.num_cons)
  }

  fn secondary_circuit(&self) -> TrivialTestCircuit<E2::Scalar> {
    Default::default()
  }
}

/// Benchmarks the compressed SNARK at a provided number of constraints
///
/// Parameters
/// - `num_augmented_circuits`: the number of augmented circuits in this configuration
/// - `group`: the criterion benchmark group
/// - `num_cons`: the number of constraints in the step circuit
fn bench_recursive_snark_internal_with_arity(
  group: &mut BenchmarkGroup<'_, WallTime>,
  num_augmented_circuits: usize,
  num_cons: usize,
) {
  let bench: NonUniformBench<
    PallasEngine,
    VestaEngine,
    TrivialTestCircuit<<VestaEngine as Engine>::Scalar>,
  > = NonUniformBench::new(2, num_cons);
  let pp = PublicParams::setup(&bench, &*default_ck_hint(), &*default_ck_hint());

  // Bench time to produce a recursive SNARK;
  // we execute a certain number of warm-up steps since executing
  // the first step is cheaper than other steps owing to the presence of
  // a lot of zeros in the satisfying assignment
  let num_warmup_steps = 10;
  let z0_primary = vec![<PallasEngine as Engine>::Scalar::from(2u64)];
  let z0_secondary = vec![<VestaEngine as Engine>::Scalar::from(2u64)];
  let mut recursive_snark_option: Option<RecursiveSNARK<PallasEngine, VestaEngine>> = None;
  let mut selected_augmented_circuit = 0;

  for _ in 0..num_warmup_steps {
    let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
      RecursiveSNARK::new(
        &pp,
        &bench,
        &bench.primary_circuit(0),
        &bench.secondary_circuit(),
        &z0_primary,
        &z0_secondary,
      )
      .unwrap()
    });

    if selected_augmented_circuit == 0 || selected_augmented_circuit == 1 {
      recursive_snark
        .prove_step(
          &pp,
          &bench.primary_circuit(selected_augmented_circuit),
          &bench.secondary_circuit(),
        )
        .expect("Prove step failed");

      recursive_snark
        .verify(&pp, &z0_primary, &z0_secondary)
        .expect("Verify failed");
    } else {
      unimplemented!()
    }

    selected_augmented_circuit = (selected_augmented_circuit + 1) % num_augmented_circuits;
    recursive_snark_option = Some(recursive_snark)
  }

  assert!(recursive_snark_option.is_some());
  let recursive_snark = recursive_snark_option.unwrap();

  // Benchmark the prove time
  group.bench_function("Prove", |b| {
    b.iter(|| {
      // produce a recursive SNARK for a step of the recursion
      assert!(black_box(&mut recursive_snark.clone())
        .prove_step(
          black_box(&pp),
          &bench.primary_circuit(0),
          &bench.secondary_circuit(),
        )
        .is_ok());
    })
  });

  // Benchmark the verification time
  group.bench_function("Verify", |b| {
    b.iter(|| {
      assert!(black_box(&mut recursive_snark.clone())
        .verify(
          black_box(&pp),
          black_box(&[<PallasEngine as Engine>::Scalar::from(2u64)]),
          black_box(&[<VestaEngine as Engine>::Scalar::from(2u64)]),
        )
        .is_ok());
    });
  });
}

fn bench_one_augmented_circuit_recursive_snark(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT_PRIMARY,
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
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT_PRIMARY;

    let mut group = c.benchmark_group(format!(
      "RecursiveSNARKSuperNova-1circuit-StepCircuitSize-{num_cons}"
    ));
    group.sample_size(NUM_SAMPLES);

    bench_recursive_snark_internal_with_arity(&mut group, 1, num_cons);
    group.finish();
  }
}

fn bench_two_augmented_circuit_recursive_snark(c: &mut Criterion) {
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in [
    NUM_CONS_VERIFIER_CIRCUIT_PRIMARY,
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
    let num_cons = num_cons_in_augmented_circuit - NUM_CONS_VERIFIER_CIRCUIT_PRIMARY;

    let mut group = c.benchmark_group(format!(
      "RecursiveSNARKSuperNova-2circuit-StepCircuitSize-{num_cons}"
    ));
    group.sample_size(NUM_SAMPLES);

    bench_recursive_snark_internal_with_arity(&mut group, 2, num_cons);
    group.finish();
  }
}

#[derive(Clone, Debug, Default)]
struct NonTrivialTestCircuit<F: PrimeField> {
  num_cons: usize,
  _p: PhantomData<F>,
}

impl<F> NonTrivialTestCircuit<F>
where
  F: PrimeField,
{
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: Default::default(),
    }
  }
}
impl<F> StepCircuit<F> for NonTrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    // Consider a an equation: `x^2 = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok((pc.cloned(), vec![y]))
  }
}
