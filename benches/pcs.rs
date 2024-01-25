use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion, SamplingMode};
use ff::Field;
use halo2curves::bn256::Bn256;
use nova_snark::provider::{
  hyperkzg::EvaluationEngine as MLEvaluationEngine,
  ipa_pc::EvaluationEngine as IPAEvaluationEngine, non_hiding_zeromorph::ZMPCS, Bn256Engine,
  Bn256EngineKZG, Bn256EngineZM,
};
use nova_snark::spartan::polys::multilinear::MultilinearPolynomial;
use nova_snark::traits::{
  commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Engine,
  TranscriptEngineTrait,
};
use rand::rngs::StdRng;
use rand_core::{CryptoRng, RngCore, SeedableRng};
use std::any::type_name;
use std::time::Duration;

// To run these benchmarks, first download `criterion` with `cargo install cargo-criterion`.
// Then `cargo criterion --bench pcs`.
// For flamegraphs, run `cargo criterion --bench pcs --features flamegraph -- --profile-time <secs>`.
// The results are located in `target/criterion/profile/<name-of-benchmark>`.
cfg_if::cfg_if! {
  if #[cfg(feature = "flamegraph")] {
    criterion_group! {
          name = pcs;
          config = Criterion::default().warm_up_time(Duration::from_millis(3000)).with_profiler(pprof::criterion::PProfProfiler::new(100, pprof::criterion::Output::Flamegraph(None)));
          targets = bench_pcs
    }
  } else {
    criterion_group! {
          name = pcs;
          config = Criterion::default().warm_up_time(Duration::from_millis(3000));
          targets = bench_pcs
    }
  }
}

criterion_main!(pcs);

const NUM_VARS_TEST_VECTOR: [usize; 6] = [10, 12, 14, 16, 18, 20];

struct BenchAssests<E: Engine, EE: EvaluationEngineTrait<E>> {
  poly: MultilinearPolynomial<<E as Engine>::Scalar>,
  point: Vec<<E as Engine>::Scalar>,
  eval: <E as Engine>::Scalar,
  ck: <<E as Engine>::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  commitment: <<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment,
  prover_key: <EE as EvaluationEngineTrait<E>>::ProverKey,
  verifier_key: <EE as EvaluationEngineTrait<E>>::VerifierKey,
  proof: Option<<EE as EvaluationEngineTrait<E>>::EvaluationArgument>,
}

/// Returns a random polynomial, a point and calculate its evaluation.
pub fn random_poly_with_eval<E: Engine, R: RngCore + CryptoRng>(
  num_vars: usize,
  mut rng: &mut R,
) -> (
  MultilinearPolynomial<<E as Engine>::Scalar>,
  Vec<<E as Engine>::Scalar>,
  <E as Engine>::Scalar,
) {
  // Generate random polynomial and point.
  let poly = MultilinearPolynomial::random(num_vars, &mut rng);
  let point = (0..num_vars)
    .map(|_| <E as Engine>::Scalar::random(&mut rng))
    .collect::<Vec<_>>();

  // Calculation evaluation of point over polynomial.
  let eval = MultilinearPolynomial::evaluate_with(poly.evaluations(), &point);

  (poly, point, eval)
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> BenchAssests<E, EE> {
  pub(crate) fn from_num_vars<R: CryptoRng + RngCore>(num_vars: usize, rng: &mut R) -> Self {
    let (poly, point, eval) = random_poly_with_eval::<E, R>(num_vars, rng);

    // Mock commitment key.
    let ck = E::CE::setup(b"test", 1 << num_vars);
    // Commits to the provided vector using the provided generators.
    let commitment = E::CE::commit(&ck, poly.evaluations());

    let (prover_key, verifier_key) = EE::setup(&ck);

    // Generate proof so that we can bench verification.
    let proof = EE::prove(
      &ck,
      &prover_key,
      &mut E::TE::new(b"TestEval"),
      &commitment,
      poly.evaluations(),
      &point,
      &eval,
    )
    .unwrap();

    Self {
      poly,
      point,
      eval,
      ck,
      commitment,
      prover_key,
      verifier_key,
      proof: Some(proof),
    }
  }
}

// Macro to generate benchmark code for multiple evaluation engine types
macro_rules! benchmark_all_engines {
    ($criterion:expr, $test_vector:expr, $proving_fn:expr, $verifying_fn:expr, $( ($assets:ident, $eval_engine:ty) ),*) => {
        for num_vars in $test_vector.iter() {
            let mut rng = rand::rngs::StdRng::seed_from_u64(*num_vars as u64);

            $(
                let $assets: BenchAssests<_, $eval_engine> = BenchAssests::from_num_vars::<StdRng>(*num_vars, &mut rng);
            )*

            // Proving group
            let mut proving_group = $criterion.benchmark_group(format!("PCS-Proving {}", num_vars));
            proving_group
                .sampling_mode(SamplingMode::Auto)
                .sample_size(10);

            $(
                proving_group.bench_with_input(BenchmarkId::new(type_name::<$eval_engine>(), num_vars), &num_vars, |b, _| {
                    $proving_fn(b, &$assets);
                });
            )*

            proving_group.finish();

            // Verifying group
            let mut verifying_group = $criterion.benchmark_group(format!("PCS-Verifying {}", num_vars));
            verifying_group
                .sampling_mode(SamplingMode::Auto)
                .sample_size(10);

            $(
                verifying_group.bench_with_input(BenchmarkId::new(type_name::<$eval_engine>(), num_vars), &num_vars, |b, _| {
                    $verifying_fn(b, &$assets);
                });
            )*

            verifying_group.finish();
        }
    };
}

fn bench_pcs(c: &mut Criterion) {
  benchmark_all_engines!(
    c,
    NUM_VARS_TEST_VECTOR,
    bench_pcs_proving_internal,
    bench_pcs_verifying_internal,
    (ipa_assets, IPAEvaluationEngine<Bn256Engine>),
    (mlkzg_assets, MLEvaluationEngine<Bn256, Bn256EngineKZG>),
    (zm_assets, ZMPCS<Bn256, Bn256EngineZM>)
  );
}

fn bench_pcs_proving_internal<E: Engine, EE: EvaluationEngineTrait<E>>(
  b: &mut Bencher<'_>,
  bench_assets: &BenchAssests<E, EE>,
) {
  // Bench generate proof.
  b.iter(|| {
    EE::prove(
      &bench_assets.ck,
      &bench_assets.prover_key,
      &mut E::TE::new(b"TestEval"),
      &bench_assets.commitment,
      bench_assets.poly.evaluations(),
      &bench_assets.point,
      &bench_assets.eval,
    )
    .unwrap();
  });
}

fn bench_pcs_verifying_internal<E: Engine, EE: EvaluationEngineTrait<E>>(
  b: &mut Bencher<'_>,
  bench_assets: &BenchAssests<E, EE>,
) {
  // Bench verify proof.
  b.iter(|| {
    EE::verify(
      &bench_assets.verifier_key,
      &mut E::TE::new(b"TestEval"),
      &bench_assets.commitment,
      &bench_assets.point,
      &bench_assets.eval,
      bench_assets.proof.as_ref().unwrap(),
    )
    .unwrap();
  });
}
