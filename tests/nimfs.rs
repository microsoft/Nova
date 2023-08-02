#![cfg(feature = "hypernova")]
use std::marker::PhantomData;

use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::{Field, PrimeField};
use nova_snark::{
  ccs::{CCS, NIMFS},
  traits::{circuit::StepCircuit, Group},
  NovaShape, ShapeCS,
};
use pasta_curves::Ep;

#[derive(Clone, Debug, Default)]
struct CubicCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F> StepCircuit<F> for CubicCircuit<F>
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
    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = &z[0];
    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
    let y = &z[1];

    cs.enforce(
      || "y = x^3 + x + 5",
      |lc| {
        lc + x_cu.get_variable()
          + x.get_variable()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
          + CS::one()
      },
      |lc| lc + CS::one(),
      |lc| lc + y.get_variable(),
    );

    Ok(vec![y.clone()])
  }

  fn output(&self, z: &[F]) -> Vec<F> {
    vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
  }
}

#[test]
fn integration_folding() {
  integration_folding_test::<Ep>()
}

fn integration_folding_test<G: Group>() {
  let circuit = CubicCircuit::<G::Scalar>::default();
  let mut cs: ShapeCS<G> = ShapeCS::new();
  // Generate the inputs:
  // Here we need both the R1CSShape so that we can generate the CCS -> NIMFS and also the witness values.
  let three = AllocatedNum::alloc(&mut cs, || Ok(G::Scalar::from(3u64))).unwrap();
  let thirty_five = AllocatedNum::alloc(&mut cs, || Ok(G::Scalar::from(35u64))).unwrap();
  let _ = circuit.synthesize(&mut cs, &[three, thirty_five]);
  let (r1cs_shape, _) = cs.r1cs_shape();

  let ccs = CCS::<G>::from_r1cs(r1cs_shape);

  // Generate NIMFS object.
  let mut nimfs = NIMFS::init(
    ccs,
    // Note we constructed z on the fly with the previously-used witness.
    vec![
      G::Scalar::ONE,
      G::Scalar::from(3u64),
      G::Scalar::from(35u64),
    ],
    b"test_nimfs",
  );

  // Now, the NIMFS should satisfy correctly as we have inputed valid starting inpuits for the first LCCCS contained instance:
  assert!(nimfs.is_sat().is_ok());

  // Now let's create a valid CCCS instance and fold it:
  let valid_cccs = nimfs.new_cccs(vec![
    G::Scalar::ONE,
    G::Scalar::from(2u64),
    G::Scalar::from(15u64),
  ]);
  nimfs.fold(valid_cccs);

  // Since the instance was correct, the NIMFS should still be satisfied.
  assert!(nimfs.is_sat().is_ok());
}
