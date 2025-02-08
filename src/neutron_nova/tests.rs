use crate::{
  errors::NovaError,
  frontend::{
    num::AllocatedNum,
    r1cs::{NovaShape, NovaWitness},
    solver::SatisfyingAssignment,
    test_shape_cs::TestShapeCS,
    ConstraintSystem, SynthesisError,
  },
  neutron_nova::{
    nifs::NIFS,
    running_instance::{RunningZFInstance, RunningZFWitness},
  },
  provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine},
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  spartan::math::Math,
  traits::{snark::default_ck_hint, Engine, ROTrait},
  CommitmentKey,
};
use ff::{Field, PrimeField};

/// Synthesize a tiny R1CS circuit for testing purposes.
pub fn synthesize_tiny_r1cs_bellpepper<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  cs: &mut CS,
  x_val: Option<Scalar>,
) -> Result<(), SynthesisError> {
  // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are
  // respectively the input and output.

  let x = AllocatedNum::alloc(cs.namespace(|| "x"), || Ok(x_val.unwrap()))?;
  let _ = x.inputize(cs.namespace(|| "x is input"));

  let x_sq = x.square(cs.namespace(|| "x_sq"))?;
  let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
  let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
    Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + Scalar::from(5u64))
  })?;
  let _ = y.inputize(cs.namespace(|| "y is output"));

  cs.enforce(
    || "y = x^3 + x + 5",
    |lc| lc + x_cu.get_variable() + x.get_variable() + (Scalar::from(5u64), CS::one()),
    |lc| lc + CS::one(),
    |lc| lc + y.get_variable(),
  );

  Ok(())
}

fn test_tiny_r1cs_bellpepper_with<E: Engine>() -> Result<(), NovaError> {
  // First create the shape
  let mut cs: TestShapeCS<E> = TestShapeCS::new();
  let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, None);
  let (S, ck): (R1CSShape<E>, _) = cs.r1cs_shape(&*default_ck_hint());

  // Now get the instance and assignment for one instance
  let mut cs = SatisfyingAssignment::<E>::new();
  let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(5)));
  let (U1, W1) = cs.r1cs_instance_and_witness(&S, &ck).unwrap();

  // Make sure that the first instance is satisfiable
  assert!(S.is_sat(&ck, &U1, &W1).is_ok());

  // Now get the instance and assignment for second instance
  let mut cs = SatisfyingAssignment::<E>::new();
  let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(135)));
  let (U2, W2) = cs.r1cs_instance_and_witness(&S, &ck).unwrap();

  // Make sure that the second instance is satisfiable
  assert!(S.is_sat(&ck, &U2, &W2).is_ok());

  // Now get the instance and assignment for third instance
  let mut cs = SatisfyingAssignment::<E>::new();
  let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(1000)));
  let (U3, W3) = cs.r1cs_instance_and_witness(&S, &ck).unwrap();

  // Make sure that the third instance is satisfiable
  assert!(S.is_sat(&ck, &U3, &W3).is_ok());

  // Now get the instance and assignment for fourth instance
  let mut cs = SatisfyingAssignment::<E>::new();
  let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(10000)));
  let (U4, W4) = cs.r1cs_instance_and_witness(&S, &ck).unwrap();

  // Make sure that the fourth instance is satisfiable
  assert!(S.is_sat(&ck, &U4, &W4).is_ok());

  // Pad the shape and instances
  let S = S.pad();
  let W1 = W1.pad(&S);
  let W2 = W2.pad(&S);
  let W3 = W3.pad(&S);
  let W4 = W4.pad(&S);

  // execute a sequence of folds
  execute_sequence(&S, U1, W1, U2, W2, U3, W3, U4, W4, &ck)
}

#[allow(clippy::too_many_arguments)]
fn execute_sequence<E: Engine>(
  S: &R1CSShape<E>,
  U1: R1CSInstance<E>,
  W1: R1CSWitness<E>,
  U2: R1CSInstance<E>,
  W2: R1CSWitness<E>,
  U3: R1CSInstance<E>,
  W3: R1CSWitness<E>,
  U4: R1CSInstance<E>,
  W4: R1CSWitness<E>,
  ck: &CommitmentKey<E>,
) -> Result<(), NovaError> {
  let ro_consts =
    <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default(
    );
  let ell = S.num_cons.log_2();
  // TODO: Create proper default running instances and witnesses withoug cloning a satisiable R1CS instance and witness
  let mut r_W = RunningZFWitness::default(W1.clone(), ell);
  let default_comm_e = r_W.nsc().e().commit::<E>(ck, E::Scalar::ZERO);
  let mut r_U = RunningZFInstance::default(U1.clone(), default_comm_e);

  let (nifs, (_U, W)) = NIFS::prove(S, &ro_consts, E::Scalar::ZERO, ck, &r_U, &r_W, &U1, &W1)?;
  let U = nifs.verify(&ro_consts, E::Scalar::ZERO, &r_U, &U1)?;

  assert_eq!(U, _U);

  r_U = U;
  r_W = W;
  // check first fold
  S.check_running_zerofold_instance(&r_U, &r_W, ck);

  let (nifs, (_U, W)) = NIFS::prove(S, &ro_consts, E::Scalar::ZERO, ck, &r_U, &r_W, &U2, &W2)?;
  let U = nifs.verify(&ro_consts, E::Scalar::ZERO, &r_U, &U2)?;

  assert_eq!(U, _U);

  r_U = U;
  r_W = W;
  // check second fold
  S.check_running_zerofold_instance(&r_U, &r_W, ck);

  let (nifs, (_U, W)) = NIFS::prove(S, &ro_consts, E::Scalar::ZERO, ck, &r_U, &r_W, &U3, &W3)?;
  let U = nifs.verify(&ro_consts, E::Scalar::ZERO, &r_U, &U3)?;

  assert_eq!(U, _U);

  r_U = U;
  r_W = W;
  // check third fold
  S.check_running_zerofold_instance(&r_U, &r_W, ck);

  let (nifs, (_U, W)) = NIFS::prove(S, &ro_consts, E::Scalar::ZERO, ck, &r_U, &r_W, &U4, &W4)?;
  let U = nifs.verify(&ro_consts, E::Scalar::ZERO, &r_U, &U4)?;

  assert_eq!(U, _U);

  r_U = U;
  r_W = W;

  S.check_running_zerofold_instance(&r_U, &r_W, ck);
  Ok(())
}

#[test]
fn test_tiny_r1cs_bellpepper() -> Result<(), NovaError> {
  test_tiny_r1cs_bellpepper_with::<PallasEngine>()?;
  test_tiny_r1cs_bellpepper_with::<Bn256EngineKZG>()?;
  test_tiny_r1cs_bellpepper_with::<Secp256k1Engine>()?;
  Ok(())
}
