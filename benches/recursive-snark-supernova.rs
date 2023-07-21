#![allow(non_snake_case)]

use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use criterion::*;
use ff::PrimeField;
use nova_snark::{
  supernova::RecursiveSNARK,
  supernova::{gen_commitmentkey_by_r1cs, RunningClaim},
  traits::{
    circuit_supernova::{StepCircuit, TrivialTestCircuit},
    Group,
  },
};
use std::time::Duration;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;

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

fn bench_one_augmented_circuit_recursive_snark(c: &mut Criterion) {
  let num_cons_verifier_circuit_primary = 9819;
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in
    [9819, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - num_cons_verifier_circuit_primary;

    let mut group = c.benchmark_group(format!("RecursiveSNARK-StepCircuitSize-{num_cons}"));
    group.sample_size(10);

    let c_primary = NonTrivialTestCircuit::new(num_cons);
    let c_secondary = TrivialTestCircuit::default();

    // Structuring running claims
    let mut running_claim1 = RunningClaim::<
      G1,
      G2,
      NonTrivialTestCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::new(0, c_primary, c_secondary.clone(), 1);

    let (r1cs_shape_primary, r1cs_shape_secondary) = running_claim1.get_r1cs_shape();
    let ck_primary = gen_commitmentkey_by_r1cs(r1cs_shape_primary);
    let ck_secondary = gen_commitmentkey_by_r1cs(r1cs_shape_secondary);

    // set unified ck_primary, ck_secondary and update digest
    running_claim1.set_commitmentkey(ck_primary.clone(), ck_secondary.clone());

    // Bench time to produce a recursive SNARK;
    // we execute a certain number of warm-up steps since executing
    // the first step is cheaper than other steps owing to the presence of
    // a lot of zeros in the satisfying assignment
    let num_warmup_steps = 10;
    let z0_primary = vec![<G1 as Group>::Scalar::from(2u64)];
    let z0_secondary = vec![<G2 as Group>::Scalar::from(2u64)];
    let initial_program_counter = <G1 as Group>::Scalar::from(0);
    let mut recursive_snark_option: Option<RecursiveSNARK<G1, G2>> = None;

    for _ in 0..num_warmup_steps {
      let program_counter = recursive_snark_option
        .as_ref()
        .map(|recursive_snark| recursive_snark.get_program_counter())
        .unwrap_or_else(|| initial_program_counter);

      let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
        RecursiveSNARK::iter_base_step(
          &running_claim1,
          program_counter,
          0,
          1,
          &z0_primary,
          &z0_secondary,
        )
        .unwrap()
      });

      let res = recursive_snark.prove_step(&running_claim1, &z0_primary, &z0_secondary);
      if let Err(e) = &res {
        println!("res failed {:?}", e);
      }
      assert!(res.is_ok());
      let res = recursive_snark.verify(&running_claim1, &z0_primary, &z0_secondary);
      if let Err(e) = &res {
        println!("res failed {:?}", e);
      }
      assert!(res.is_ok());
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
            black_box(&running_claim1),
            black_box(&vec![<G1 as Group>::Scalar::from(2u64)]),
            black_box(&vec![<G2 as Group>::Scalar::from(2u64)]),
          )
          .is_ok());
      })
    });

    // Benchmark the verification time
    group.bench_function("Verify", |b| {
      b.iter(|| {
        assert!(black_box(&mut recursive_snark.clone())
          .verify(
            black_box(&running_claim1),
            black_box(&vec![<G1 as Group>::Scalar::from(2u64)]),
            black_box(&vec![<G2 as Group>::Scalar::from(2u64)]),
          )
          .is_ok());
      });
    });
    group.finish();
  }
}

fn bench_two_augmented_circuit_recursive_snark(c: &mut Criterion) {
  let num_cons_verifier_circuit_primary = 9819;
  // we vary the number of constraints in the step circuit
  for &num_cons_in_augmented_circuit in
    [9819, 16384, 32768, 65536, 131072, 262144, 524288, 1048576].iter()
  {
    // number of constraints in the step circuit
    let num_cons = num_cons_in_augmented_circuit - num_cons_verifier_circuit_primary;

    let mut group = c.benchmark_group(format!("RecursiveSNARK-StepCircuitSize-{num_cons}"));
    group.sample_size(10);

    let c_primary = NonTrivialTestCircuit::new(num_cons);
    let c_secondary = TrivialTestCircuit::default();

    // Structuring running claims
    let mut running_claim1 = RunningClaim::<
      G1,
      G2,
      NonTrivialTestCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::new(0, c_primary.clone(), c_secondary.clone(), 2);

    // Structuring running claims
    let mut running_claim2 = RunningClaim::<
      G1,
      G2,
      NonTrivialTestCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::new(1, c_primary, c_secondary.clone(), 2);

    let (r1cs_shape_primary, r1cs_shape_secondary) = running_claim1.get_r1cs_shape();
    let ck_primary = gen_commitmentkey_by_r1cs(r1cs_shape_primary);
    let ck_secondary = gen_commitmentkey_by_r1cs(r1cs_shape_secondary);

    // set unified ck_primary, ck_secondary and update digest
    running_claim1.set_commitmentkey(ck_primary.clone(), ck_secondary.clone());
    running_claim2.set_commitmentkey(ck_primary.clone(), ck_secondary.clone());

    // Bench time to produce a recursive SNARK;
    // we execute a certain number of warm-up steps since executing
    // the first step is cheaper than other steps owing to the presence of
    // a lot of zeros in the satisfying assignment
    let num_warmup_steps = 10;
    let z0_primary = vec![<G1 as Group>::Scalar::from(2u64)];
    let z0_secondary = vec![<G2 as Group>::Scalar::from(2u64)];
    let initial_program_counter = <G1 as Group>::Scalar::from(0);
    let mut recursive_snark_option: Option<RecursiveSNARK<G1, G2>> = None;
    let mut selected_augmented_circuit = 0;

    for _ in 0..num_warmup_steps {
      let program_counter = recursive_snark_option
        .as_ref()
        .map(|recursive_snark| recursive_snark.get_program_counter())
        .unwrap_or_else(|| initial_program_counter);

      let mut recursive_snark = recursive_snark_option.unwrap_or_else(|| {
        RecursiveSNARK::iter_base_step(
          &running_claim1,
          program_counter,
          0,
          2,
          &z0_primary,
          &z0_secondary,
        )
        .unwrap()
      });

      if selected_augmented_circuit == 0 {
        let res = recursive_snark.prove_step(&running_claim1, &z0_primary, &z0_secondary);
        if let Err(e) = &res {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
        let res = recursive_snark.verify(&running_claim1, &z0_primary, &z0_secondary);
        if let Err(e) = &res {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
      } else if selected_augmented_circuit == 1 {
        let res = recursive_snark.prove_step(&running_claim2, &z0_primary, &z0_secondary);
        if let Err(e) = &res {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
        let res = recursive_snark.verify(&running_claim2, &z0_primary, &z0_secondary);
        if let Err(e) = &res {
          println!("res failed {:?}", e);
        }
        assert!(res.is_ok());
      } else {
        unimplemented!()
      }
      selected_augmented_circuit = (selected_augmented_circuit + 1) % 2;
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
            black_box(&running_claim1),
            black_box(&vec![<G1 as Group>::Scalar::from(2u64)]),
            black_box(&vec![<G2 as Group>::Scalar::from(2u64)]),
          )
          .is_ok());
      })
    });

    // Benchmark the verification time
    group.bench_function("Verify", |b| {
      b.iter(|| {
        assert!(black_box(&mut recursive_snark.clone())
          .verify(
            black_box(&running_claim1),
            black_box(&vec![<G1 as Group>::Scalar::from(2u64)]),
            black_box(&vec![<G2 as Group>::Scalar::from(2u64)]),
          )
          .is_ok());
      });
    });
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

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: &AllocatedNum<F>,
    z: &[AllocatedNum<F>],
  ) -> Result<(AllocatedNum<F>, Vec<AllocatedNum<F>>), SynthesisError> {
    // Consider a an equation: `x^2 = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok((pc.clone(), vec![y]))
  }

  fn output(&self, pc: F, z: &[F]) -> (F, Vec<F>) {
    let mut x = z[0];
    let mut y = x;
    for _i in 0..self.num_cons {
      y = x * x;
      x = y;
    }
    (pc, vec![y])
  }
}
