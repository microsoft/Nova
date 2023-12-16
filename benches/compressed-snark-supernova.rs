#![allow(non_snake_case)]
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::{measurement::WallTime, *};
use ff::PrimeField;
use nova_snark::{
  supernova::NonUniformCircuit,
  supernova::{snark::CompressedSNARK, PublicParams, RecursiveSNARK},
  traits::{
    circuit_supernova::{StepCircuit, TrivialTestCircuit},
    snark::BatchedRelaxedR1CSSNARKTrait,
    snark::RelaxedR1CSSNARKTrait,
    Engine,
  },
};
use std::time::Duration;

type E1 = nova_snark::provider::PallasEngine;
type E2 = nova_snark::provider::VestaEngine;
type EE1 = nova_snark::provider::ipa_pc::EvaluationEngine<E1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
// SNARKs without computation commitmnets
type S1 = nova_snark::spartan::batched::BatchedRelaxedR1CSSNARK<E1, EE1>;
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;
// SNARKs with computation commitmnets
type SS1 = nova_snark::spartan::batched_ppsnark::BatchedRelaxedR1CSSNARK<E1, EE1>;
type SS2 = nova_snark::spartan::ppsnark::RelaxedR1CSSNARK<E2, EE2>;

// To run these benchmarks, first download `criterion` with `cargo install cargo-criterion`.
// Then `cargo criterion --bench compressed-snark-supernova`. The results are located in `target/criterion/data/<name-of-benchmark>`.
// For flamegraphs, run `cargo criterion --bench compressed-snark-supernova --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
      name = compressed_snark_supernova;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
      targets = bench_one_augmented_circuit_compressed_snark, bench_two_augmented_circuit_compressed_snark, bench_two_augmented_circuit_compressed_snark_with_computational_commitments
    }
  } else {
    criterion_group! {
      name = compressed_snark_supernova;
      config = Criterion::default().warm_up_time(Duration::from_millis(3000));
      targets = bench_one_augmented_circuit_compressed_snark, bench_two_augmented_circuit_compressed_snark, bench_two_augmented_circuit_compressed_snark_with_computational_commitments
    }
  }
}

criterion_main!(compressed_snark_supernova);

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
    assert!(
      circuit_index < self.num_circuits,
      "Circuit index out of bounds: asked for {circuit_index}, but there are only {} circuits.",
      self.num_circuits
    );

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
fn bench_compressed_snark_internal_with_arity<
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
>(
  group: &mut BenchmarkGroup<'_, WallTime>,
  num_augmented_circuits: usize,
  num_cons: usize,
) {
  let bench: NonUniformBench<E1, E2, TrivialTestCircuit<<E2 as Engine>::Scalar>> =
    NonUniformBench::new(num_augmented_circuits, num_cons);
  let pp = PublicParams::setup(&bench, &*S1::ck_floor(), &*S2::ck_floor());

  let num_steps = 3;
  let z0_primary = vec![<E1 as Engine>::Scalar::from(2u64)];
  let z0_secondary = vec![<E2 as Engine>::Scalar::from(2u64)];
  let mut recursive_snark_option: Option<RecursiveSNARK<E1, E2>> = None;
  let mut selected_augmented_circuit = 0;

  for _ in 0..num_steps {
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
      let res = recursive_snark.prove_step(
        &pp,
        &bench.primary_circuit(selected_augmented_circuit),
        &bench.secondary_circuit(),
      );
      res.expect("Prove step failed");

      let res = recursive_snark.verify(&pp, &z0_primary, &z0_secondary);
      res.expect("Verify failed");
    } else {
      unimplemented!()
    }

    selected_augmented_circuit = (selected_augmented_circuit + 1) % num_augmented_circuits;
    recursive_snark_option = Some(recursive_snark)
  }

  assert!(recursive_snark_option.is_some());
  let recursive_snark = recursive_snark_option.unwrap();

  let (prover_key, verifier_key) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

  // Benchmark the prove time
  group.bench_function("Prove", |b| {
    b.iter(|| {
      assert!(CompressedSNARK::<_, _, _, _, S1, S2>::prove(
        black_box(&pp),
        black_box(&prover_key),
        black_box(&recursive_snark)
      )
      .is_ok());
    })
  });

  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &prover_key, &recursive_snark);

  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  // Benchmark the verification time
  group.bench_function("Verify", |b| {
    b.iter(|| {
      assert!(black_box(&compressed_snark)
        .verify(
          black_box(&pp),
          black_box(&verifier_key),
          black_box(&z0_primary),
          black_box(&z0_secondary),
        )
        .is_ok());
    })
  });
}

fn bench_one_augmented_circuit_compressed_snark(c: &mut Criterion) {
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
      "CompressedSNARKSuperNova-1circuit-StepCircuitSize-{num_cons}"
    ));
    group.sample_size(NUM_SAMPLES);

    bench_compressed_snark_internal_with_arity::<S1, S2>(&mut group, 1, num_cons);

    group.finish();
  }
}

fn bench_two_augmented_circuit_compressed_snark(c: &mut Criterion) {
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
      "CompressedSNARKSuperNova-2circuit-StepCircuitSize-{num_cons}"
    ));
    group.sample_size(NUM_SAMPLES);

    bench_compressed_snark_internal_with_arity::<S1, S2>(&mut group, 2, num_cons);

    group.finish();
  }
}

fn bench_two_augmented_circuit_compressed_snark_with_computational_commitments(c: &mut Criterion) {
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
      "CompressedSNARKSuperNova-Commitments-2circuit-StepCircuitSize-{num_cons}"
    ));
    group.sample_size(NUM_SAMPLES);

    bench_compressed_snark_internal_with_arity::<SS1, SS2>(&mut group, 2, num_cons);

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
    // Consider a an equation: `x^{2 * num_cons} = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok((pc.cloned(), vec![y]))
  }
}
