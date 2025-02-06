//! This module implements a non-interactive folding scheme from NeutronNova
#![allow(non_snake_case)]
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_CHALLENGE_BITS},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  neutron::relation::{FoldedInstance, FoldedWitness, Structure},
  r1cs::{R1CSInstance, R1CSWitness},
  spartan::{
    polys::{
      power::PowPolynomial,
      univariate::{CompressedUniPoly, UniPoly},
    },
    sumcheck::SumcheckProof,
  },
  traits::{commitment::CommitmentEngineTrait, AbsorbInROTrait, Engine, ROTrait},
  Commitment, CommitmentKey, CE,
};
use ff::Field;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

/// An NIFS message from NeutronNova's folding scheme
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_E: Commitment<E>,
  pub(crate) poly: CompressedUniPoly<E::Scalar>,
}

type ROConstants<E> =
  <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants;

impl<E: Engine> NIFS<E> {
  /// Takes as input a folded instance-witness tuple `(U1, W1)` and
  /// an R1CS instance-witness tuple `(U2, W2)` with a compatible structure `shape`
  /// and defined with respect to the same `ck`, and outputs
  /// a folded instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  ///
  /// TODO: Revisit the following note
  /// Note that this code is tailored for use with Nova's IVC scheme, which enforces
  /// certain requirements between the two instances that are folded.
  /// In particular, it requires that `U1` and `U2` are such that the hash of `U1` is stored in the public IO of `U2`.
  /// In this particular setting, this means that if `U2` is absorbed in the RO, it implicitly absorbs `U1` as well.
  /// So the code below avoids absorbing `U1` in the RO.
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    S: &Structure<E>,
    U1: &FoldedInstance<E>,
    W1: &FoldedWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) -> Result<(NIFS<E>, (FoldedInstance<E>, FoldedWitness<E>)), NovaError> {
    // initialize a new RO
    let mut ro = E::RO::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U1 to transcript
    // TODO: revisit this once we have the full folding scheme
    U1.absorb_in_ro(&mut ro);

    // append U2 to transcript
    U2.absorb_in_ro(&mut ro);

    // generate a challenge for the eq polynomial
    let tau = ro.squeeze(NUM_CHALLENGE_BITS);

    // compute a commitment to the eq polynomial
    let E = PowPolynomial::new(&tau, S.ell).evals();
    let r_E = E::Scalar::random(&mut OsRng);
    let comm_E = CE::<E>::commit(ck, &E, &r_E);

    // absorb tau in bignum format
    // TODO: factor out this code
    let limbs: Vec<E::Scalar> = nat_to_limbs(&f_to_nat(&tau), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
    for limb in limbs {
      ro.absorb(scalar_as_base::<E>(limb));
    }
    comm_E.absorb_in_ro(&mut ro); // absorb the commitment

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS);

    // We now run a single round of the sum-check protocol to establish
    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;

    let z1 = [W1.W.clone(), vec![U1.u], U1.X.clone()].concat();
    let (g1, g2, g3) = S.S.multiply_vec(&z1)?;

    let z2 = [W2.W.clone(), vec![E::Scalar::ONE], U2.X.clone()].concat();
    let (h1, h2, h3) = S.S.multiply_vec(&z2)?;

    // compute the sum-check polynomial's evaluations at 0, 2, 3
    let (eval_point_0, eval_point_2, eval_point_3, eval_point_4) =
      SumcheckProof::<E>::compute_eval_points_quartic_with_additive_term(
        &rho, &W1.E, &g1, &g2, &g3, &E, &h1, &h2, &h3,
      );

    let evals = vec![
      eval_point_0,
      T - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
    ];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInROTrait<E>>::absorb_in_ro(&poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();

    let U = U1.fold(U2, &comm_E, &r_b, &T_out)?;
    let W = W1.fold(W2, &E, &r_E, &r_b)?;

    // return the folded instance and witness
    Ok((
      Self {
        comm_E,
        poly: poly.compress(),
      },
      (U, W),
    ))
  }

  /// Takes as input a relaxed R1CS instance `U1` and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  pub fn verify(
    &self,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    U1: &FoldedInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<FoldedInstance<E>, NovaError> {
    // initialize a new RO
    let mut ro = E::RO::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U1 to transcript
    // TODO: revisit this once we have the full folding scheme
    U1.absorb_in_ro(&mut ro);

    // append U2 to transcript
    U2.absorb_in_ro(&mut ro);

    // generate a challenge for the eq polynomial
    let tau = ro.squeeze(NUM_CHALLENGE_BITS);

    // absorb tau in bignum format
    // TODO: factor out this code
    let limbs: Vec<E::Scalar> = nat_to_limbs(&f_to_nat(&tau), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
    for limb in limbs {
      ro.absorb(scalar_as_base::<E>(limb));
    }
    self.comm_E.absorb_in_ro(&mut ro); // absorb the commitment

    // compute a challenge from the RO
    let rho = ro.squeeze(NUM_CHALLENGE_BITS);

    // T = (1-rho) * T1 + rho * T2, where T1 comes from the running instance and T2 = 0
    let T = (E::Scalar::ONE - rho) * U1.T;

    // decompress the provided polynomial with T as hint
    let poly = self.poly.decompress(&T);

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInROTrait<E>>::absorb_in_ro(&poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS);

    // compute the sum-check polynomial's evaluations at r_b
    let eq_rho_r_b = (E::Scalar::ONE - rho) * (E::Scalar::ONE - r_b) + rho * r_b;
    let T_out = poly.evaluate(&r_b) * eq_rho_r_b.invert().unwrap();

    let U = U1.fold(U2, &self.comm_E, &r_b, &T_out)?;

    // return the folded instance and witness
    Ok(U)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      num::AllocatedNum,
      r1cs::{NovaShape, NovaWitness},
      solver::SatisfyingAssignment,
      test_shape_cs::TestShapeCS,
      ConstraintSystem, SynthesisError,
    },
    provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine},
    r1cs::R1CSShape,
    traits::{snark::default_ck_hint, Engine},
  };
  use ff::{Field, PrimeField};

  fn synthesize_tiny_r1cs_bellpepper<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
    cs: &mut CS,
    x_val: Option<Scalar>,
  ) -> Result<(), SynthesisError> {
    // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
    let x = AllocatedNum::alloc_infallible(cs.namespace(|| "x"), || x_val.unwrap());
    let _ = x.inputize(cs.namespace(|| "x is input"));

    let x_sq = x.square(cs.namespace(|| "x_sq"))?;
    let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), &x)?;
    let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
      Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + Scalar::from(5u64))
    })?;
    let _ = y.inputize(cs.namespace(|| "y is output"));

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

    Ok(())
  }

  fn test_tiny_r1cs_bellpepper_with<E: Engine>() {
    // First create the shape
    let mut cs: TestShapeCS<E> = TestShapeCS::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, None);
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());

    let ro_consts =
      <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default();

    // Now get the instance and assignment for one instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(5)));
    let (U1, W1) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the first instance is satisfiable
    assert!(shape.is_sat(&ck, &U1, &W1).is_ok());

    // Now get the instance and assignment for second instance
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = synthesize_tiny_r1cs_bellpepper(&mut cs, Some(E::Scalar::from(135)));
    let (U2, W2) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that the second instance is satisfiable
    assert!(shape.is_sat(&ck, &U2, &W2).is_ok());

    // pad the shape and witnesses
    let shape = shape.pad();
    let W1 = W1.pad(&shape);
    let W2 = W2.pad(&shape);
    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &shape,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs_bellpepper() {
    test_tiny_r1cs_bellpepper_with::<PallasEngine>();
    test_tiny_r1cs_bellpepper_with::<Bn256EngineKZG>();
    test_tiny_r1cs_bellpepper_with::<Secp256k1Engine>();
  }

  fn execute_sequence<E: Engine>(
    ck: &CommitmentKey<E>,
    ro_consts: &<<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants,
    pp_digest: &<E as Engine>::Scalar,
    shape: &R1CSShape<E>,
    U1: &R1CSInstance<E>,
    W1: &R1CSWitness<E>,
    U2: &R1CSInstance<E>,
    W2: &R1CSWitness<E>,
  ) {
    // produce a default running instance
    let str = Structure::new(shape);
    let mut running_W = FoldedWitness::default(&str);
    let mut running_U = FoldedInstance::default(&str);

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U1, W1,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U1);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());

    // produce a step SNARK with (W2, U2) as the second incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, &str, &running_U, &running_W, U2, W2,
    );
    assert!(res.is_ok());
    let (nifs, (_U, W)) = res.unwrap();

    // verify the step SNARK with U1 as the first incoming instance
    let res = nifs.verify(ro_consts, pp_digest, &running_U, U2);
    assert!(res.is_ok());
    let U = res.unwrap();

    assert_eq!(U, _U);

    // update the running witness and instance
    running_W = W;
    running_U = U;

    // check if the running instance is satisfiable
    let res = str.is_sat(ck, &running_U, &running_W);
    if res != Ok(()) {
      println!("Error: {:?}", res);
    }
    assert!(res.is_ok());
  }
}
