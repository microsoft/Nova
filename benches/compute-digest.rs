use std::{marker::PhantomData, time::Duration};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ff::PrimeField;
use nova_snark::{
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::default_ck_hint,
    Engine,
  },
  PublicParams,
};

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type C1 = NonTrivialCircuit<<E1 as Engine>::Scalar>;
type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;

criterion_group! {
name = compute_digest;
config = Criterion::default().warm_up_time(Duration::from_millis(3000)).sample_size(10);
targets = bench_compute_digest
}

criterion_main!(compute_digest);

fn bench_compute_digest(c: &mut Criterion) {
  c.bench_function("compute_digest", |b| {
    b.iter(|| {
      PublicParams::<E1, E2, C1, C2>::setup(
        black_box(&C1::new(10)),
        black_box(&C2::default()),
        black_box(&*default_ck_hint()),
        black_box(&*default_ck_hint()),
      )
    })
  });
}

#[derive(Clone, Debug, Default)]
struct NonTrivialCircuit<F: PrimeField> {
  num_cons: usize,
  _p: PhantomData<F>,
}

impl<F: PrimeField> NonTrivialCircuit<F> {
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: PhantomData,
    }
  }
}
impl<F: PrimeField> StepCircuit<F> for NonTrivialCircuit<F> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // Consider a an equation: `x^2 = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok(vec![y])
  }
}
