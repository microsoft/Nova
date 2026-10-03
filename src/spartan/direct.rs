//! This module provides interfaces to directly prove a step circuit by using Spartan SNARK.
//! In particular, it supports any SNARK that implements `RelaxedR1CSSNARK` trait
//! (e.g., with the SNARKs implemented in ppsnark.rs or snark.rs).
use crate::traits::evm_serde::EvmCompatSerde;
use crate::{
  errors::NovaError,
  frontend::{
    num::AllocatedNum,
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    Circuit, ConstraintSystem, SynthesisError,
  },
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    circuit::StepCircuit,
    commitment::CommitmentEngineTrait,
    snark::{DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine,
  },
  Commitment, CommitmentKey, DerandKey,
};
use core::marker::PhantomData;
use ff::Field;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// A direct circuit that can be synthesized
pub struct DirectCircuit<E: Engine, SC: StepCircuit<E::Scalar>> {
  z_i: Option<Vec<E::Scalar>>, // inputs to the circuit
  sc: SC,                      // step circuit to be executed
}

impl<E: Engine, SC: StepCircuit<E::Scalar>> DirectCircuit<E, SC> {
  /// Create a new direct circuit from a step circuit and optional inputs
  pub fn new(z_i: Option<Vec<E::Scalar>>, sc: SC) -> Self {
    Self { z_i, sc }
  }

  /// Synthesize the direct circuit into `cs`.
  ///
  /// In addition to synthesizing the step circuit, this enforces the contract declared by
  /// [`StepCircuit::arity`]: the supplied inputs (when present) and the outputs returned by
  /// the step circuit must both have exactly `arity()` elements.
  fn synthesize_inner<CS: ConstraintSystem<E::Scalar>>(
    &self,
    cs: &mut CS,
  ) -> Result<(), NovaError> {
    // obtain the arity information
    let arity = self.sc.arity();

    // the supplied inputs, when present, must match the declared arity
    if let Some(z_i) = &self.z_i {
      if z_i.len() != arity {
        return Err(NovaError::InvalidInputLength);
      }
    }

    // Allocate zi. If inputs.zi is not provided, allocate default value 0
    let zero = vec![E::Scalar::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.z_i.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<E::Scalar>>, _>>()?;

    let z_i_plus_one = self.sc.synthesize(&mut cs.namespace(|| "F"), &z_i)?;

    // the outputs must match the declared arity
    if z_i_plus_one.len() != arity {
      return Err(NovaError::InvalidStepOutputLength);
    }

    // inputize both z_i and z_i_plus_one
    for (j, input) in z_i.iter().enumerate().take(arity) {
      input.inputize(cs.namespace(|| format!("input {j}")))?;
    }
    for (j, output) in z_i_plus_one.iter().enumerate().take(arity) {
      output.inputize(cs.namespace(|| format!("output {j}")))?;
    }

    Ok(())
  }
}

impl<E: Engine, SC: StepCircuit<E::Scalar>> Circuit<E::Scalar> for DirectCircuit<E, SC> {
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    self
      .synthesize_inner(cs)
      .map_err(|e| SynthesisError::Unsatisfiable(e.to_string()))
  }
}

/// A type that holds the prover key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E, S>
where
  E: Engine,
  S: RelaxedR1CSSNARKTrait<E>,
{
  S: R1CSShape<E>,
  ck: CommitmentKey<E>,
  pk: S::ProverKey,
}

/// A type that holds the verifier key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E, S>
where
  E: Engine,
  S: RelaxedR1CSSNARKTrait<E>,
{
  dk: DerandKey<E>,
  vk: S::VerifierKey,
}

impl<E: Engine, S: RelaxedR1CSSNARKTrait<E>> VerifierKey<E, S> {
  /// Returns the digest of the verifier's key
  pub fn digest(&self) -> E::Scalar {
    self.vk.digest()
  }
}

/// A direct SNARK proving a step circuit
#[serde_as]
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DirectSNARK<E, S, C>
where
  E: Engine,
  S: RelaxedR1CSSNARKTrait<E>,
  C: StepCircuit<E::Scalar>,
{
  comm_W: Commitment<E>, // commitment to the witness
  #[serde_as(as = "EvmCompatSerde")]
  blind_r_W: E::Scalar,
  snark: S, // snark proving the witness is satisfying
  _p: PhantomData<C>,
}

impl<E: Engine, S: RelaxedR1CSSNARKTrait<E>, C: StepCircuit<E::Scalar>> DirectSNARK<E, S, C> {
  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup(sc: C) -> Result<(ProverKey<E, S>, VerifierKey<E, S>), NovaError> {
    // construct a circuit that can be synthesized
    let circuit: DirectCircuit<E, C> = DirectCircuit { z_i: None, sc };

    let mut cs: ShapeCS<E> = ShapeCS::new();
    circuit.synthesize_inner(&mut cs)?;

    let shape = cs.r1cs_shape()?;
    let ck = R1CSShape::commitment_key(&[&shape], &[&*S::ck_floor()])?;

    let (pk, vk) = S::setup(&ck, &shape)?;

    let dk = E::CE::derand_key(&ck);

    let pk = ProverKey { S: shape, ck, pk };

    let vk = VerifierKey { dk, vk };

    Ok((pk, vk))
  }

  /// Produces a proof of satisfiability of the provided circuit
  pub fn prove(pk: &ProverKey<E, S>, sc: C, z_i: &[E::Scalar]) -> Result<Self, NovaError> {
    let mut cs = SatisfyingAssignment::<E>::new();

    let circuit: DirectCircuit<E, C> = DirectCircuit {
      z_i: Some(z_i.to_vec()),
      sc,
    };

    circuit.synthesize_inner(&mut cs)?;
    let (u, w) = cs
      .r1cs_instance_and_witness(&pk.S, &pk.ck, &mut OsRng)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })?;

    // convert the instance and witness to relaxed form
    let (u_relaxed, w_relaxed) = (
      RelaxedR1CSInstance::from_r1cs_instance_unchecked(&u.comm_W, &u.X),
      RelaxedR1CSWitness::from_r1cs_witness(&pk.S, &w),
    );

    // derandomize/unblind commitments
    let (derandom_w_relaxed, blind_W, blind_E) = w_relaxed.derandomize();
    let derandom_u_relaxed = u_relaxed.derandomize(&E::CE::derand_key(&pk.ck), &blind_W, &blind_E);

    // prove the instance using Spartan
    let snark = S::prove(
      &pk.ck,
      &pk.pk,
      &pk.S,
      &derandom_u_relaxed,
      &derandom_w_relaxed,
    )?;

    Ok(DirectSNARK {
      comm_W: u.comm_W,
      blind_r_W: w_relaxed.r_W,
      snark,
      _p: PhantomData,
    })
  }

  /// Verifies a proof of satisfiability
  pub fn verify(&self, vk: &VerifierKey<E, S>, io: &[E::Scalar]) -> Result<(), NovaError> {
    // derandomize/unblind commitments
    let comm_W = E::CE::derandomize(&vk.dk, &self.comm_W, &self.blind_r_W);

    // construct an instance using the provided commitment to the witness and z_i and z_{i+1}
    let u_relaxed = RelaxedR1CSInstance::from_r1cs_instance_unchecked(&comm_W, io);

    // verify the snark using the constructed instance
    self.snark.verify(&vk.vk, &u_relaxed)?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine},
  };
  use core::marker::PhantomData;
  use ff::PrimeField;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for CubicCircuit<F> {
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
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

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

      Ok(vec![y])
    }
  }

  impl<F: PrimeField> CubicCircuit<F> {
    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }

  /// A step circuit whose synthesis always fails
  #[derive(Clone, Debug, Default)]
  struct FailingCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for FailingCircuit<F> {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      _z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      Err(SynthesisError::Unsatisfiable(
        "step circuit synthesis failed".to_string(),
      ))
    }
  }

  /// A step circuit that declares an arity of 2 but returns a single output
  #[derive(Clone, Debug, Default)]
  struct WrongOutputArityCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F: PrimeField> StepCircuit<F> for WrongOutputArityCircuit<F> {
    fn arity(&self) -> usize {
      2
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      _cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      Ok(vec![z[0].clone()])
    }
  }

  fn test_direct_snark_synthesis_error_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    // setup must surface the error returned by the step circuit
    let res = DirectSNARK::<E, S, FailingCircuit<E::Scalar>>::setup(FailingCircuit::default());
    assert!(matches!(res, Err(NovaError::SynthesisError { .. })));

    // prove must surface the error returned by the step circuit
    let (pk, _vk) = DirectSNARK::<E, S, CubicCircuit<E::Scalar>>::setup(CubicCircuit::default())
      .expect("setup of the cubic circuit should succeed");
    let res = DirectSNARK::<E, S, FailingCircuit<E::Scalar>>::prove(
      &pk,
      FailingCircuit::default(),
      &[E::Scalar::ZERO],
    );
    assert!(matches!(res, Err(NovaError::SynthesisError { .. })));
  }

  fn test_direct_snark_arity_checks_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    // a step circuit that returns fewer outputs than its declared arity is rejected
    let res = DirectSNARK::<E, S, WrongOutputArityCircuit<E::Scalar>>::setup(
      WrongOutputArityCircuit::default(),
    );
    assert_eq!(res.err(), Some(NovaError::InvalidStepOutputLength));

    let (pk, _vk) = DirectSNARK::<E, S, CubicCircuit<E::Scalar>>::setup(CubicCircuit::default())
      .expect("setup of the cubic circuit should succeed");

    // inputs that are longer than the declared arity are rejected ...
    let res = DirectSNARK::<E, S, CubicCircuit<E::Scalar>>::prove(
      &pk,
      CubicCircuit::default(),
      &[E::Scalar::ZERO, E::Scalar::ONE],
    );
    assert_eq!(res.err(), Some(NovaError::InvalidInputLength));

    // ... and so are inputs that are shorter (which previously panicked)
    let res =
      DirectSNARK::<E, S, CubicCircuit<E::Scalar>>::prove(&pk, CubicCircuit::default(), &[]);
    assert_eq!(res.err(), Some(NovaError::InvalidInputLength));
  }

  #[test]
  fn test_direct_snark_error_propagation() {
    type E = PallasEngine;
    type EE = crate::provider::ipa_pc::EvaluationEngine<E>;
    type S = crate::spartan::snark::RelaxedR1CSSNARK<E, EE>;
    test_direct_snark_synthesis_error_with::<E, S>();
  }

  #[test]
  fn test_direct_snark_arity_checks() {
    type E = PallasEngine;
    type EE = crate::provider::ipa_pc::EvaluationEngine<E>;
    type S = crate::spartan::snark::RelaxedR1CSSNARK<E, EE>;
    test_direct_snark_arity_checks_with::<E, S>();
  }

  #[test]
  fn test_direct_snark() {
    type E = PallasEngine;
    type EE = crate::provider::ipa_pc::EvaluationEngine<E>;
    type S = crate::spartan::snark::RelaxedR1CSSNARK<E, EE>;
    test_direct_snark_with::<E, S>();

    type Spp = crate::spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;
    test_direct_snark_with::<E, Spp>();

    type E2 = Bn256EngineKZG;
    type EE2 = crate::provider::hyperkzg::EvaluationEngine<E2>;
    type S2 = crate::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;
    test_direct_snark_with::<E2, S2>();

    type S2pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<E2, EE2>;
    test_direct_snark_with::<E2, S2pp>();

    type E3 = Secp256k1Engine;
    type EE3 = crate::provider::ipa_pc::EvaluationEngine<E3>;
    type S3 = crate::spartan::snark::RelaxedR1CSSNARK<E3, EE3>;
    test_direct_snark_with::<E3, S3>();

    type S3pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<E3, EE3>;
    test_direct_snark_with::<E3, S3pp>();
  }

  fn test_direct_snark_with<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) =
      DirectSNARK::<E, S, CubicCircuit<<E as Engine>::Scalar>>::setup(circuit.clone()).unwrap();

    let num_steps = 3;

    // setup inputs
    let z0 = vec![<E as Engine>::Scalar::ZERO];
    let mut z_i = z0;

    for _i in 0..num_steps {
      // produce a SNARK
      let res = DirectSNARK::prove(&pk, circuit.clone(), &z_i);
      assert!(res.is_ok());

      let z_i_plus_one = circuit.output(&z_i);

      let snark = res.unwrap();

      // verify the SNARK
      let io = z_i
        .clone()
        .into_iter()
        .chain(z_i_plus_one.clone())
        .collect::<Vec<_>>();
      let res = snark.verify(&vk, &io);
      assert!(res.is_ok());

      // set input to the next step
      z_i.clone_from(&z_i_plus_one);
    }

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(z_i, vec![<E as Engine>::Scalar::from(2460515u64)]);
  }
}
