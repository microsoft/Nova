use std::{marker::PhantomData, time::Duration};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ff::PrimeField;
use nova_snark::{
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::default_ck_hint,
    Group,
  },
  PublicParams,
};

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type C1 = NonTrivialCircuit<<G1 as Group>::Scalar>;
type C2 = TrivialCircuit<<G2 as Group>::Scalar>;

criterion_group! {
name = compute_digest;
config = Criterion::default().warm_up_time(Duration::from_millis(3000)).sample_size(10);
targets = bench_compute_digest
}

criterion_main!(compute_digest);

fn bench_compute_digest(c: &mut Criterion) {
  c.bench_function("compute_digest", |b| {
    b.iter(|| {
      PublicParams::<G1, G2, C1, C2>::setup(
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

impl<F> NonTrivialCircuit<F>
where
  F: PrimeField,
{
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: PhantomData,
    }
  }
}
impl<F> StepCircuit<F> for NonTrivialCircuit<F>
where
  F: PrimeField,
{
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
