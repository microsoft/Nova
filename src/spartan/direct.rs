//! This module provides interfaces to directly prove a step circuit by using Spartan SNARK.
//! In particular, it supports any SNARK that implements `RelaxedR1CSSNARK` trait
//! (e.g., with the SNARKs implemented in ppsnark.rs or snark.rs).
use crate::{
  bellpepper::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{
    circuit::StepCircuit,
    snark::{DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine,
  },
  Commitment, CommitmentKey,
};
use bellpepper_core::{num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::Field;
use serde::{Deserialize, Serialize};

struct DirectCircuit<E: Engine, SC: StepCircuit<E::Scalar>> {
  z_i: Option<Vec<E::Scalar>>, // inputs to the circuit
  sc: SC,                      // step circuit to be executed
}

impl<E: Engine, SC: StepCircuit<E::Scalar>> Circuit<E::Scalar> for DirectCircuit<E, SC> {
  fn synthesize<CS: ConstraintSystem<E::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // obtain the arity information
    let arity = self.sc.arity();

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

    // inputize both z_i and z_i_plus_one
    for (j, input) in z_i.iter().enumerate().take(arity) {
      let _ = input.inputize(cs.namespace(|| format!("input {j}")));
    }
    for (j, output) in z_i_plus_one.iter().enumerate().take(arity) {
      let _ = output.inputize(cs.namespace(|| format!("output {j}")));
    }

    Ok(())
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
  vk: S::VerifierKey,
}

impl<E: Engine, S: RelaxedR1CSSNARKTrait<E>> VerifierKey<E, S> {
  /// Returns the digest of the verifier's key
  pub fn digest(&self) -> E::Scalar {
    self.vk.digest()
  }
}

/// A direct SNARK proving a step circuit
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DirectSNARK<E, S, C>
where
  E: Engine,
  S: RelaxedR1CSSNARKTrait<E>,
  C: StepCircuit<E::Scalar>,
{
  comm_W: Commitment<E>, // commitment to the witness
  snark: S,              // snark proving the witness is satisfying
  _p: PhantomData<C>,
}

impl<E: Engine, S: RelaxedR1CSSNARKTrait<E>, C: StepCircuit<E::Scalar>> DirectSNARK<E, S, C> {
  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup(sc: C) -> Result<(ProverKey<E, S>, VerifierKey<E, S>), NovaError> {
    // construct a circuit that can be synthesized
    let circuit: DirectCircuit<E, C> = DirectCircuit { z_i: None, sc };

    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);

    let (shape, ck) = cs.r1cs_shape(&*S::ck_floor());

    let (pk, vk) = S::setup(&ck, &shape)?;

    let pk = ProverKey { S: shape, ck, pk };

    let vk = VerifierKey { vk };

    Ok((pk, vk))
  }

  /// Produces a proof of satisfiability of the provided circuit
  pub fn prove(pk: &ProverKey<E, S>, sc: C, z_i: &[E::Scalar]) -> Result<Self, NovaError> {
    let mut cs = SatisfyingAssignment::<E>::new();

    let circuit: DirectCircuit<E, C> = DirectCircuit {
      z_i: Some(z_i.to_vec()),
      sc,
    };

    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs
      .r1cs_instance_and_witness(&pk.S, &pk.ck)
      .map_err(|_e| NovaError::UnSat)?;

    // convert the instance and witness to relaxed form
    let (u_relaxed, w_relaxed) = (
      RelaxedR1CSInstance::from_r1cs_instance_unchecked(&u.comm_W, &u.X),
      RelaxedR1CSWitness::from_r1cs_witness(&pk.S, &w),
    );

    // prove the instance using Spartan
    let snark = S::prove(&pk.ck, &pk.pk, &pk.S, &u_relaxed, &w_relaxed)?;

    Ok(DirectSNARK {
      comm_W: u.comm_W,
      snark,
      _p: PhantomData,
    })
  }

  /// Verifies a proof of satisfiability
  pub fn verify(&self, vk: &VerifierKey<E, S>, io: &[E::Scalar]) -> Result<(), NovaError> {
    // construct an instance using the provided commitment to the witness and z_i and z_{i+1}
    let u_relaxed = RelaxedR1CSInstance::from_r1cs_instance_unchecked(&self.comm_W, io);

    // verify the snark using the constructed instance
    self.snark.verify(&vk.vk, &u_relaxed)?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine};
  use ::bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
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
