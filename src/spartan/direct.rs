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
  traits::{circuit::StepCircuit, snark::RelaxedR1CSSNARKTrait, Group},
  Commitment, CommitmentKey,
};
use bellpepper_core::{num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::Field;
use serde::{Deserialize, Serialize};

struct DirectCircuit<G: Group, SC: StepCircuit<G::Scalar>> {
  z_i: Option<Vec<G::Scalar>>, // inputs to the circuit
  sc: SC,                      // step circuit to be executed
}

impl<G: Group, SC: StepCircuit<G::Scalar>> Circuit<G::Scalar> for DirectCircuit<G, SC> {
  fn synthesize<CS: ConstraintSystem<G::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // obtain the arity information
    let arity = self.sc.arity();

    // Allocate zi. If inputs.zi is not provided, allocate default value 0
    let zero = vec![G::Scalar::ZERO; arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.z_i.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<G::Scalar>>, _>>()?;

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
pub struct ProverKey<G, S>
where
  G: Group,
  S: RelaxedR1CSSNARKTrait<G>,
{
  S: R1CSShape<G>,
  ck: CommitmentKey<G>,
  pk: S::ProverKey,
}

/// A type that holds the verifier key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G, S>
where
  G: Group,
  S: RelaxedR1CSSNARKTrait<G>,
{
  vk: S::VerifierKey,
}

/// A direct SNARK proving a step circuit
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct DirectSNARK<G, S, C>
where
  G: Group,
  S: RelaxedR1CSSNARKTrait<G>,
  C: StepCircuit<G::Scalar>,
{
  comm_W: Commitment<G>, // commitment to the witness
  snark: S,              // snark proving the witness is satisfying
  _p: PhantomData<C>,
}

impl<G: Group, S: RelaxedR1CSSNARKTrait<G>, C: StepCircuit<G::Scalar>> DirectSNARK<G, S, C> {
  /// Produces prover and verifier keys for the direct SNARK
  pub fn setup(sc: C) -> Result<(ProverKey<G, S>, VerifierKey<G, S>), NovaError> {
    // construct a circuit that can be synthesized
    let circuit: DirectCircuit<G, C> = DirectCircuit { z_i: None, sc };

    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (shape, ck) = cs.r1cs_shape();

    let (pk, vk) = S::setup(&ck, &shape)?;

    let pk = ProverKey { S: shape, ck, pk };

    let vk = VerifierKey { vk };

    Ok((pk, vk))
  }

  /// Produces a proof of satisfiability of the provided circuit
  pub fn prove(pk: &ProverKey<G, S>, sc: C, z_i: &[G::Scalar]) -> Result<Self, NovaError> {
    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();

    let circuit: DirectCircuit<G, C> = DirectCircuit {
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
    let snark = S::prove(&pk.ck, &pk.pk, &u_relaxed, &w_relaxed)?;

    Ok(DirectSNARK {
      comm_W: u.comm_W,
      snark,
      _p: PhantomData,
    })
  }

  /// Verifies a proof of satisfiability
  pub fn verify(&self, vk: &VerifierKey<G, S>, io: &[G::Scalar]) -> Result<(), NovaError> {
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
  use crate::provider::{bn256_grumpkin::bn256, secp_secq::secp256k1};
  use ::bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;
  use ff::PrimeField;

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
    type G = pasta_curves::pallas::Point;
    type EE = crate::provider::ipa_pc::EvaluationEngine<G>;
    type S = crate::spartan::snark::RelaxedR1CSSNARK<G, EE>;
    type Spp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G, EE>;
    test_direct_snark_with::<G, S>();
    test_direct_snark_with::<G, Spp>();

    type G2 = bn256::Point;
    type EE2 = crate::provider::ipa_pc::EvaluationEngine<G2>;
    type S2 = crate::spartan::snark::RelaxedR1CSSNARK<G2, EE2>;
    type S2pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G2, EE2>;
    test_direct_snark_with::<G2, S2>();
    test_direct_snark_with::<G2, S2pp>();

    type G3 = secp256k1::Point;
    type EE3 = crate::provider::ipa_pc::EvaluationEngine<G3>;
    type S3 = crate::spartan::snark::RelaxedR1CSSNARK<G3, EE3>;
    type S3pp = crate::spartan::ppsnark::RelaxedR1CSSNARK<G3, EE3>;
    test_direct_snark_with::<G3, S3>();
    test_direct_snark_with::<G3, S3pp>();
  }

  fn test_direct_snark_with<G: Group, S: RelaxedR1CSSNARKTrait<G>>() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) =
      DirectSNARK::<G, S, CubicCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

    let num_steps = 3;

    // setup inputs
    let z0 = vec![<G as Group>::Scalar::ZERO];
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
      z_i = z_i_plus_one.clone();
    }

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(z_i, vec![<G as Group>::Scalar::from(2460515u64)]);
  }
}
