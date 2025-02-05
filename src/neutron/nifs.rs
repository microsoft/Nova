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
      multilinear::MultilinearPolynomial,
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
    let comm_E = CE::<E>::commit(&ck, &E, &r_E);

    // initialize a new RO
    let mut ro = E::RO::new(ro_consts.clone());
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
    // todo: remove the need to wrap
    let poly_E = MultilinearPolynomial::new([W1.E.clone(), E.clone()].concat());
    let poly_A = MultilinearPolynomial::new([g1, h1].concat());
    let poly_B = MultilinearPolynomial::new([g2, h2].concat());
    let poly_C = MultilinearPolynomial::new([g3, h3].concat());

    let comb_func =
      |poly_E_comp: &E::Scalar,
       poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar|
       -> E::Scalar { *poly_E_comp * (*poly_A_comp * *poly_B_comp - *poly_C_comp) };

    let (eval_point_0, eval_point_2, eval_point_3) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &poly_E, &poly_A, &poly_B, &poly_C, &comb_func,
      );

    let evals = vec![eval_point_0, T - eval_point_0, eval_point_2, eval_point_3];
    let poly = UniPoly::<E::Scalar>::from_evals(&evals);

    // absorb poly in the RO
    <UniPoly<E::Scalar> as AbsorbInROTrait<E>>::absorb_in_ro(&poly, &mut ro);

    // squeeze a challenge
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS);

    // compute the sum-check polynomial's evaluations at r_b
    let T_out = poly.evaluate(&r_b);

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

  /*
  /// Takes as input a relaxed R1CS instance `U1` and R1CS instance `U2`
  /// with the same shape and defined with respect to the same parameters,
  /// and outputs a folded instance `U` with the same shape,
  /// with the guarantee that the folded instance `U`
  /// if and only if `U1` and `U2` are satisfiable.
  pub fn verify(
    &self,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    U1: &RelaxedR1CSInstance<E>,
    U2: &R1CSInstance<E>,
  ) -> Result<RelaxedR1CSInstance<E>, NovaError> {
    // initialize a new RO
    let mut ro = E::RO::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U2 to transcript, U1 does not need to absorbed since U2.X[0] = Hash(params, U1, i, z0, zi)
    U2.absorb_in_ro(&mut ro);

    // append `comm_T` to the transcript and obtain a challenge
    self.comm_T.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r = ro.squeeze(NUM_CHALLENGE_BITS);

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &self.comm_T, &r);

    // return the folded instance
    Ok(U)
  }*/
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
    r1cs::SparseMatrix,
    traits::{commitment::CommitmentEngineTrait, snark::default_ck_hint, Engine},
  };
  use ff::{Field, PrimeField};
  use rand::rngs::OsRng;

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
    let mut running_W = RelaxedR1CSWitness::default(shape);
    let mut running_U = RelaxedR1CSInstance::default(ck, shape);

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, shape, &running_U, &running_W, U1, W1,
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

    // produce a step SNARK with (W2, U2) as the second incoming witness-instance pair
    let res = NIFS::prove(
      ck, ro_consts, pp_digest, shape, &running_U, &running_W, U2, W2,
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
    assert!(shape.is_sat_relaxed(ck, &running_U, &running_W).is_ok());
  }

  fn execute_sequence_relaxed<E: Engine>(
    ck: &CommitmentKey<E>,
    ro_consts: &<<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants,
    pp_digest: &<E as Engine>::Scalar,
    shape: &R1CSShape<E>,
    U1: &RelaxedR1CSInstance<E>,
    W1: &RelaxedR1CSWitness<E>,
    U2: &RelaxedR1CSInstance<E>,
    W2: &RelaxedR1CSWitness<E>,
  ) -> (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>) {
    // produce a default running instance
    let mut running_W = RelaxedR1CSWitness::default(shape);
    let mut running_U = RelaxedR1CSInstance::default(ck, shape);

    // produce a step SNARK with (W1, U1) as the first incoming witness-instance pair
    let res = NIFSRelaxed::prove(
      ck, ro_consts, pp_digest, shape, &running_U, &running_W, U1, W1,
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

    // produce a step SNARK with (W2, U2) as the second incoming witness-instance pair
    let res = NIFSRelaxed::prove(
      ck, ro_consts, pp_digest, shape, &running_U, &running_W, U2, W2,
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
    assert!(shape.is_sat_relaxed(ck, &running_U, &running_W).is_ok());

    (running_U, running_W)
  }

  fn test_tiny_r1cs_relaxed_derandomize_with<E: Engine>() {
    let (ck, S, final_U, final_W) = test_tiny_r1cs_relaxed_with::<E>();
    assert!(S.is_sat_relaxed(&ck, &final_U, &final_W).is_ok());

    let dk = E::CE::derand_key(&ck);
    let (derandom_final_W, final_blind_W, final_blind_E) = final_W.derandomize();
    let derandom_final_U = final_U.derandomize(&dk, &final_blind_W, &final_blind_E);

    assert!(S
      .is_sat_relaxed(&ck, &derandom_final_U, &derandom_final_W)
      .is_ok());
  }

  #[test]
  fn test_tiny_r1cs_relaxed_derandomize() {
    test_tiny_r1cs_relaxed_derandomize_with::<PallasEngine>();
    test_tiny_r1cs_relaxed_derandomize_with::<Bn256EngineKZG>();
    test_tiny_r1cs_relaxed_derandomize_with::<Secp256k1Engine>();
  }

  fn test_tiny_r1cs_relaxed_with<E: Engine>() -> (
    CommitmentKey<E>,
    R1CSShape<E>,
    RelaxedR1CSInstance<E>,
    RelaxedR1CSWitness<E>,
  ) {
    let one = <E::Scalar as Field>::ONE;
    let (num_cons, num_vars, num_io, A, B, C) = {
      let num_cons = 4;
      let num_vars = 3;
      let num_io = 2;

      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      // The R1CS for this problem consists of the following constraints:
      // `I0 * I0 - Z0 = 0`
      // `Z0 * I0 - Z1 = 0`
      // `(Z1 + I0) * 1 - Z2 = 0`
      // `(Z2 + 5) * 1 - I1 = 0`

      // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
      // constraint and a column for every entry in z = (vars, u, inputs)
      // An R1CS instance is satisfiable iff:
      // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
      let mut A: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut B: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut C: Vec<(usize, usize, E::Scalar)> = Vec::new();

      // constraint 0 entries in (A,B,C)
      // `I0 * I0 - Z0 = 0`
      A.push((0, num_vars + 1, one));
      B.push((0, num_vars + 1, one));
      C.push((0, 0, one));

      // constraint 1 entries in (A,B,C)
      // `Z0 * I0 - Z1 = 0`
      A.push((1, 0, one));
      B.push((1, num_vars + 1, one));
      C.push((1, 1, one));

      // constraint 2 entries in (A,B,C)
      // `(Z1 + I0) * 1 - Z2 = 0`
      A.push((2, 1, one));
      A.push((2, num_vars + 1, one));
      B.push((2, num_vars, one));
      C.push((2, 2, one));

      // constraint 3 entries in (A,B,C)
      // `(Z2 + 5) * 1 - I1 = 0`
      A.push((3, 2, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 2, one));

      (num_cons, num_vars, num_io, A, B, C)
    };

    // create a shape object
    let rows = num_cons;
    let num_inputs = num_io + 1;
    let cols = num_vars + num_inputs;
    let S = {
      let res = R1CSShape::new(
        num_cons,
        num_vars,
        num_inputs - 1,
        SparseMatrix::new(&A, rows, cols),
        SparseMatrix::new(&B, rows, cols),
        SparseMatrix::new(&C, rows, cols),
      );
      assert!(res.is_ok());
      res.unwrap()
    };

    // generate generators and ro constants
    let ck = S.commitment_key(&*default_ck_hint());
    let ro_consts =
      <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default();

    let rand_inst_witness_generator =
      |ck: &CommitmentKey<E>, I: &E::Scalar| -> (E::Scalar, R1CSInstance<E>, R1CSWitness<E>) {
        let i0 = *I;

        // compute a satisfying (vars, X) tuple
        let (O, vars, X) = {
          let z0 = i0 * i0; // constraint 0
          let z1 = i0 * z0; // constraint 1
          let z2 = z1 + i0; // constraint 2
          let i1 = z2 + one + one + one + one + one; // constraint 3

          // store the witness and IO for the instance
          let W = vec![z0, z1, z2];
          let X = vec![i0, i1];
          (i1, W, X)
        };

        let W = {
          let res = R1CSWitness::new(&S, &vars);
          assert!(res.is_ok());
          res.unwrap()
        };
        let U = {
          let comm_W = W.commit(ck);
          let res = R1CSInstance::new(&S, &comm_W, &X);
          assert!(res.is_ok());
          res.unwrap()
        };

        // check that generated instance is satisfiable
        assert!(S.is_sat(ck, &U, &W).is_ok());

        (O, U, W)
      };

    let mut csprng: OsRng = OsRng;
    let I = E::Scalar::random(&mut csprng); // the first input is picked randomly for the first instance
    let (_O, U1, W1) = rand_inst_witness_generator(&ck, &I);
    let (U2, W2) = S.sample_random_instance_witness(&ck).unwrap(); // random fold

    println!("INSTANCE {:#?}", U1.clone());

    // execute a sequence of folds
    let (final_U, final_W) = execute_sequence_relaxed(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &S,
      &RelaxedR1CSInstance::from_r1cs_instance(&ck, &S, &U1),
      &RelaxedR1CSWitness::from_r1cs_witness(&S, &W1),
      &U2,
      &W2,
    );

    (ck, S, final_U, final_W)
  }

  #[test]
  fn test_tiny_r1cs_relaxed() {
    test_tiny_r1cs_relaxed_with::<PallasEngine>();
    test_tiny_r1cs_relaxed_with::<Bn256EngineKZG>();
    test_tiny_r1cs_relaxed_with::<Secp256k1Engine>();
  }

  fn test_tiny_r1cs_with<E: Engine>() {
    let one = <E::Scalar as Field>::ONE;
    let (num_cons, num_vars, num_io, A, B, C) = {
      let num_cons = 4;
      let num_vars = 3;
      let num_io = 2;

      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      // The R1CS for this problem consists of the following constraints:
      // `I0 * I0 - Z0 = 0`
      // `Z0 * I0 - Z1 = 0`
      // `(Z1 + I0) * 1 - Z2 = 0`
      // `(Z2 + 5) * 1 - I1 = 0`

      // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
      // constraint and a column for every entry in z = (vars, u, inputs)
      // An R1CS instance is satisfiable iff:
      // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
      let mut A: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut B: Vec<(usize, usize, E::Scalar)> = Vec::new();
      let mut C: Vec<(usize, usize, E::Scalar)> = Vec::new();

      // constraint 0 entries in (A,B,C)
      // `I0 * I0 - Z0 = 0`
      A.push((0, num_vars + 1, one));
      B.push((0, num_vars + 1, one));
      C.push((0, 0, one));

      // constraint 1 entries in (A,B,C)
      // `Z0 * I0 - Z1 = 0`
      A.push((1, 0, one));
      B.push((1, num_vars + 1, one));
      C.push((1, 1, one));

      // constraint 2 entries in (A,B,C)
      // `(Z1 + I0) * 1 - Z2 = 0`
      A.push((2, 1, one));
      A.push((2, num_vars + 1, one));
      B.push((2, num_vars, one));
      C.push((2, 2, one));

      // constraint 3 entries in (A,B,C)
      // `(Z2 + 5) * 1 - I1 = 0`
      A.push((3, 2, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 2, one));

      (num_cons, num_vars, num_io, A, B, C)
    };

    // create a shape object
    let rows = num_cons;
    let num_inputs = num_io + 1;
    let cols = num_vars + num_inputs;
    let S = {
      let res = R1CSShape::new(
        num_cons,
        num_vars,
        num_inputs - 1,
        SparseMatrix::new(&A, rows, cols),
        SparseMatrix::new(&B, rows, cols),
        SparseMatrix::new(&C, rows, cols),
      );
      assert!(res.is_ok());
      res.unwrap()
    };

    // generate generators and ro constants
    let ck = S.commitment_key(&*default_ck_hint());
    let ro_consts =
      <<E as Engine>::RO as ROTrait<<E as Engine>::Base, <E as Engine>::Scalar>>::Constants::default();

    let rand_inst_witness_generator =
      |ck: &CommitmentKey<E>, I: &E::Scalar| -> (E::Scalar, R1CSInstance<E>, R1CSWitness<E>) {
        let i0 = *I;

        // compute a satisfying (vars, X) tuple
        let (O, vars, X) = {
          let z0 = i0 * i0; // constraint 0
          let z1 = i0 * z0; // constraint 1
          let z2 = z1 + i0; // constraint 2
          let i1 = z2 + one + one + one + one + one; // constraint 3

          // store the witness and IO for the instance
          let W = vec![z0, z1, z2];
          let X = vec![i0, i1];
          (i1, W, X)
        };

        let W = {
          let res = R1CSWitness::new(&S, &vars);
          assert!(res.is_ok());
          res.unwrap()
        };
        let U = {
          let comm_W = W.commit(ck);
          let res = R1CSInstance::new(&S, &comm_W, &X);
          assert!(res.is_ok());
          res.unwrap()
        };

        // check that generated instance is satisfiable
        assert!(S.is_sat(ck, &U, &W).is_ok());

        (O, U, W)
      };

    let mut csprng: OsRng = OsRng;
    let I = E::Scalar::random(&mut csprng); // the first input is picked randomly for the first instance
    let (O, U1, W1) = rand_inst_witness_generator(&ck, &I);
    let (_O, U2, W2) = rand_inst_witness_generator(&ck, &O);

    // execute a sequence of folds
    execute_sequence(
      &ck,
      &ro_consts,
      &<E as Engine>::Scalar::ZERO,
      &S,
      &U1,
      &W1,
      &U2,
      &W2,
    );
  }

  #[test]
  fn test_tiny_r1cs() {
    test_tiny_r1cs_with::<PallasEngine>();
    test_tiny_r1cs_with::<Bn256EngineKZG>();
    test_tiny_r1cs_with::<Secp256k1Engine>();
  }
}
