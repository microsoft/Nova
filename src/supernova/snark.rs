//! This module defines a final compressing SNARK for supernova proofs

use super::{error::SuperNovaError, PublicParams, RecursiveSNARK};
use crate::{
  constants::NUM_HASH_BITS,
  r1cs::{R1CSInstance, RelaxedR1CSWitness},
  traits::{
    circuit_supernova::StepCircuit,
    snark::{BatchedRelaxedR1CSSNARKTrait, RelaxedR1CSSNARKTrait},
    AbsorbInROTrait, Engine, ROTrait,
  },
};
use crate::{errors::NovaError, scalar_as_base, RelaxedR1CSInstance, NIFS};
use ff::PrimeField;
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// A type that holds the prover key for `CompressedSNARK`
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  pk_primary: S1::ProverKey,
  pk_secondary: S2::ProverKey,
  _p: PhantomData<(C1, C2)>,
}

/// A type that holds the verifier key for `CompressedSNARK`
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  vk_primary: S1::VerifierKey,
  vk_secondary: S2::VerifierKey,
  _p: PhantomData<(C1, C2)>,
}

/// A SNARK that proves the knowledge of a valid `RecursiveSNARK`
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedSNARK<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  r_U_primary: Vec<RelaxedR1CSInstance<E1>>,
  r_W_snark_primary: S1,

  r_U_secondary: RelaxedR1CSInstance<E2>,
  l_u_secondary: R1CSInstance<E2>,
  nifs_secondary: NIFS<E2>,
  f_W_snark_secondary: S2,

  num_steps: usize,
  program_counter: E1::Scalar,

  zn_primary: Vec<E1::Scalar>,
  zn_secondary: Vec<E2::Scalar>,
  _p: PhantomData<(E1, E2, C1, C2, S1, S2)>,
}

impl<E1, E2, C1, C2, S1, S2> CompressedSNARK<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: BatchedRelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  /// Creates prover and verifier keys for `CompressedSNARK`
  pub fn setup(
    pp: &PublicParams<E1, E2, C1, C2>,
  ) -> Result<
    (
      ProverKey<E1, E2, C1, C2, S1, S2>,
      VerifierKey<E1, E2, C1, C2, S1, S2>,
    ),
    SuperNovaError,
  > {
    let (pk_primary, vk_primary) = S1::setup(&pp.ck_primary, pp.primary_r1cs_shapes())?;

    let (pk_secondary, vk_secondary) =
      S2::setup(&pp.ck_secondary, &pp.circuit_shape_secondary.r1cs_shape)?;

    let prover_key = ProverKey {
      pk_primary,
      pk_secondary,
      _p: PhantomData,
    };
    let verifier_key = VerifierKey {
      vk_primary,
      vk_secondary,
      _p: PhantomData,
    };

    Ok((prover_key, verifier_key))
  }

  /// Create a new `CompressedSNARK`
  pub fn prove(
    pp: &PublicParams<E1, E2, C1, C2>,
    pk: &ProverKey<E1, E2, C1, C2, S1, S2>,
    recursive_snark: &RecursiveSNARK<E1, E2>,
  ) -> Result<Self, SuperNovaError> {
    // fold the secondary circuit's instance
    let res_secondary = NIFS::prove(
      &pp.ck_secondary,
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &pp.circuit_shape_secondary.r1cs_shape,
      &recursive_snark.r_U_secondary,
      &recursive_snark.r_W_secondary,
      &recursive_snark.l_u_secondary,
      &recursive_snark.l_w_secondary,
    );

    let (nifs_secondary, (f_U_secondary, f_W_secondary)) = res_secondary?;

    // Prepare the list of primary Relaxed R1CS instances (a default instance is provided for
    // uninitialized circuits)
    let r_U_primary = recursive_snark
      .r_U_primary
      .iter()
      .enumerate()
      .map(|(idx, r_U)| {
        r_U
          .clone()
          .unwrap_or_else(|| RelaxedR1CSInstance::default(&pp.ck_primary, &pp[idx].r1cs_shape))
      })
      .collect::<Vec<_>>();

    // Prepare the list of primary relaxed R1CS witnesses (a default witness is provided for
    // uninitialized circuits)
    let r_W_primary: Vec<RelaxedR1CSWitness<E1>> = recursive_snark
      .r_W_primary
      .iter()
      .enumerate()
      .map(|(idx, r_W)| {
        r_W
          .clone()
          .unwrap_or_else(|| RelaxedR1CSWitness::default(&pp[idx].r1cs_shape))
      })
      .collect::<Vec<_>>();

    // Generate a primary SNARK proof for the list of primary circuits
    let r_W_snark_primary = S1::prove(
      &pp.ck_primary,
      &pk.pk_primary,
      pp.primary_r1cs_shapes(),
      &r_U_primary,
      &r_W_primary,
    )?;

    // Generate a secondary SNARK proof for the secondary circuit
    let f_W_snark_secondary = S2::prove(
      &pp.ck_secondary,
      &pk.pk_secondary,
      &pp.circuit_shape_secondary.r1cs_shape,
      &f_U_secondary,
      &f_W_secondary,
    )?;

    let compressed_snark = CompressedSNARK {
      r_U_primary,
      r_W_snark_primary,

      r_U_secondary: recursive_snark.r_U_secondary.clone(),
      l_u_secondary: recursive_snark.l_u_secondary.clone(),
      nifs_secondary,
      f_W_snark_secondary,

      num_steps: recursive_snark.i,
      program_counter: recursive_snark.program_counter,

      zn_primary: recursive_snark.zi_primary.clone(),
      zn_secondary: recursive_snark.zi_secondary.clone(),

      _p: PhantomData,
    };

    Ok(compressed_snark)
  }

  /// Verify the correctness of the `CompressedSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<E1, E2, C1, C2>,
    vk: &VerifierKey<E1, E2, C1, C2, S1, S2>,
    z0_primary: &[E1::Scalar],
    z0_secondary: &[E2::Scalar],
  ) -> Result<(Vec<E1::Scalar>, Vec<E2::Scalar>), SuperNovaError> {
    let last_circuit_idx = field_as_usize(self.program_counter);

    let num_field_primary_ro = 3 // params_next, i_new, program_counter_new
    + 2 * pp[last_circuit_idx].F_arity // zo, z1
    + (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // # 1 * (7 + [X0, X1]*#num_limb)

    // secondary circuit
    // NOTE: This count ensure the number of witnesses sent by the prover must equal the number of
    // NIVC circuits
    let num_field_secondary_ro = 2 // params_next, i_new
    + 2 * pp.circuit_shape_secondary.F_arity // zo, z1
    + pp.circuit_shapes.len() * (7 + 2 * pp.augmented_circuit_params_primary.get_n_limbs()); // #num_augment

    // Compute the primary and secondary hashes given the digest, program counter, instances, and
    // witnesses provided by the prover
    let (hash_primary, hash_secondary) = {
      let mut hasher =
        <E2 as Engine>::RO::new(pp.ro_consts_secondary.clone(), num_field_primary_ro);

      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(self.num_steps as u64));
      hasher.absorb(self.program_counter);

      for e in z0_primary {
        hasher.absorb(*e);
      }

      for e in &self.zn_primary {
        hasher.absorb(*e);
      }

      self.r_U_secondary.absorb_in_ro(&mut hasher);

      let mut hasher2 =
        <E1 as Engine>::RO::new(pp.ro_consts_primary.clone(), num_field_secondary_ro);

      hasher2.absorb(scalar_as_base::<E1>(pp.digest()));
      hasher2.absorb(E2::Scalar::from(self.num_steps as u64));

      for e in z0_secondary {
        hasher2.absorb(*e);
      }

      for e in &self.zn_secondary {
        hasher2.absorb(*e);
      }

      self.r_U_primary.iter().for_each(|U| {
        U.absorb_in_ro(&mut hasher2);
      });

      (
        hasher.squeeze(NUM_HASH_BITS),
        hasher2.squeeze(NUM_HASH_BITS),
      )
    };

    // Compare the computed hashes with the public IO of the last invocation of `prove_step`
    if hash_primary != self.l_u_secondary.X[0] {
      return Err(NovaError::ProofVerifyError.into());
    }

    if hash_secondary != scalar_as_base::<E2>(self.l_u_secondary.X[1]) {
      return Err(NovaError::ProofVerifyError.into());
    }

    // Verify the primary SNARK
    let res_primary = self
      .r_W_snark_primary
      .verify(&vk.vk_primary, &self.r_U_primary);

    // Fold the secondary circuit's instance
    let f_U_secondary = self.nifs_secondary.verify(
      &pp.ro_consts_secondary,
      &scalar_as_base::<E1>(pp.digest()),
      &self.r_U_secondary,
      &self.l_u_secondary,
    )?;

    // Verify the secondary SNARK
    let res_secondary = self
      .f_W_snark_secondary
      .verify(&vk.vk_secondary, &f_U_secondary);

    res_primary?;

    res_secondary?;

    Ok((self.zn_primary.clone(), self.zn_secondary.clone()))
  }
}

fn field_as_usize<F: PrimeField>(x: F) -> usize {
  u32::from_le_bytes(x.to_repr().as_ref()[0..4].try_into().unwrap()) as usize
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::{
    provider::{
      ipa_pc, Bn256EngineIPA, GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine,
      VestaEngine,
    },
    spartan::{batched, batched_ppsnark, snark::RelaxedR1CSSNARK},
    supernova::NonUniformCircuit,
    traits::circuit_supernova::TrivialSecondaryCircuit,
  };

  use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::Field;

  type EE<E> = ipa_pc::EvaluationEngine<E>;
  type S1<E> = batched::BatchedRelaxedR1CSSNARK<E, EE<E>>;
  type S1PP<E> = batched_ppsnark::BatchedRelaxedR1CSSNARK<E, EE<E>>;
  type S2<E> = RelaxedR1CSSNARK<E, EE<E>>;

  #[derive(Clone)]
  struct SquareCircuit<E: Engine> {
    _p: PhantomData<E>,
  }

  impl<E: Engine> StepCircuit<E::Scalar> for SquareCircuit<E> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      0
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _pc: Option<&AllocatedNum<E::Scalar>>,
      z: &[AllocatedNum<E::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<E::Scalar>>,
        Vec<AllocatedNum<E::Scalar>>,
      ),
      SynthesisError,
    > {
      let z_i = &z[0];

      let z_next = z_i.square(cs.namespace(|| "z_i^2"))?;

      let next_pc = AllocatedNum::alloc(cs.namespace(|| "next_pc"), || Ok(E::Scalar::from(1u64)))?;

      cs.enforce(
        || "next_pc = 1",
        |lc| lc + CS::one(),
        |lc| lc + next_pc.get_variable(),
        |lc| lc + CS::one(),
      );

      Ok((Some(next_pc), vec![z_next]))
    }
  }

  #[derive(Clone)]
  struct CubeCircuit<E: Engine> {
    _p: PhantomData<E>,
  }

  impl<E: Engine> StepCircuit<E::Scalar> for CubeCircuit<E> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _pc: Option<&AllocatedNum<E::Scalar>>,
      z: &[AllocatedNum<E::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<E::Scalar>>,
        Vec<AllocatedNum<E::Scalar>>,
      ),
      SynthesisError,
    > {
      let z_i = &z[0];

      let z_sq = z_i.square(cs.namespace(|| "z_i^2"))?;
      let z_cu = z_sq.mul(cs.namespace(|| "z_i^3"), z_i)?;

      let next_pc = AllocatedNum::alloc(cs.namespace(|| "next_pc"), || Ok(E::Scalar::from(0u64)))?;

      cs.enforce(
        || "next_pc = 0",
        |lc| lc + CS::one(),
        |lc| lc + next_pc.get_variable(),
        |lc| lc,
      );

      Ok((Some(next_pc), vec![z_cu]))
    }
  }

  #[derive(Clone)]
  enum TestCircuit<E: Engine> {
    Square(SquareCircuit<E>),
    Cube(CubeCircuit<E>),
  }

  impl<E: Engine> TestCircuit<E> {
    fn new(num_steps: usize) -> Vec<Self> {
      let mut circuits = Vec::new();

      for idx in 0..num_steps {
        if idx % 2 == 0 {
          circuits.push(Self::Square(SquareCircuit { _p: PhantomData }))
        } else {
          circuits.push(Self::Cube(CubeCircuit { _p: PhantomData }))
        }
      }

      circuits
    }
  }

  impl<E: Engine> StepCircuit<E::Scalar> for TestCircuit<E> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      match self {
        TestCircuit::Square(c) => c.circuit_index(),
        TestCircuit::Cube(c) => c.circuit_index(),
      }
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      pc: Option<&AllocatedNum<E::Scalar>>,
      z: &[AllocatedNum<E::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<E::Scalar>>,
        Vec<AllocatedNum<E::Scalar>>,
      ),
      SynthesisError,
    > {
      match self {
        TestCircuit::Square(c) => c.synthesize(cs, pc, z),
        TestCircuit::Cube(c) => c.synthesize(cs, pc, z),
      }
    }
  }

  impl<E1, E2> NonUniformCircuit<E1, E2, TestCircuit<E1>, TrivialSecondaryCircuit<E2::Scalar>>
    for TestCircuit<E1>
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    fn num_circuits(&self) -> usize {
      2
    }

    fn primary_circuit(&self, circuit_index: usize) -> TestCircuit<E1> {
      match circuit_index {
        0 => Self::Square(SquareCircuit { _p: PhantomData }),
        1 => Self::Cube(CubeCircuit { _p: PhantomData }),
        _ => panic!("Invalid circuit index"),
      }
    }

    fn secondary_circuit(&self) -> TrivialSecondaryCircuit<E2::Scalar> {
      Default::default()
    }
  }

  fn test_nivc_trivial_with_compression_with<E1, E2, S1, S2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    S1: BatchedRelaxedR1CSSNARKTrait<E1>,
    S2: RelaxedR1CSSNARKTrait<E2>,
  {
    const NUM_STEPS: usize = 6;

    let secondary_circuit = TrivialSecondaryCircuit::default();
    let test_circuits = TestCircuit::new(NUM_STEPS);

    let pp = PublicParams::setup(&test_circuits[0], &*S1::ck_floor(), &*S2::ck_floor());

    let z0_primary = vec![E1::Scalar::from(17u64)];
    let z0_secondary = vec![<E2 as Engine>::Scalar::ZERO];

    let mut recursive_snark = RecursiveSNARK::new(
      &pp,
      &test_circuits[0],
      &test_circuits[0],
      &secondary_circuit,
      &z0_primary,
      &z0_secondary,
    )
    .unwrap();

    for circuit in test_circuits.iter().take(NUM_STEPS) {
      let prove_res = recursive_snark.prove_step(&pp, circuit, &secondary_circuit);

      let verify_res = recursive_snark.verify(&pp, &z0_primary, &z0_secondary);

      assert!(prove_res.is_ok());
      assert!(verify_res.is_ok());
    }

    let (prover_key, verifier_key) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

    let compressed_prove_res = CompressedSNARK::prove(&pp, &prover_key, &recursive_snark);

    assert!(compressed_prove_res.is_ok());

    let compressed_snark = compressed_prove_res.unwrap();

    let compressed_verify_res =
      compressed_snark.verify(&pp, &verifier_key, &z0_primary, &z0_secondary);

    assert!(compressed_verify_res.is_ok());
  }

  #[test]
  fn test_nivc_trivial_with_compression() {
    // ppSNARK
    test_nivc_trivial_with_compression_with::<PallasEngine, VestaEngine, S1PP<_>, S2<_>>();
    test_nivc_trivial_with_compression_with::<Bn256EngineIPA, GrumpkinEngine, S1PP<_>, S2<_>>();
    test_nivc_trivial_with_compression_with::<Secp256k1Engine, Secq256k1Engine, S1PP<_>, S2<_>>();
    // classic SNARK
    test_nivc_trivial_with_compression_with::<PallasEngine, VestaEngine, S1<_>, S2<_>>();
    test_nivc_trivial_with_compression_with::<Bn256EngineIPA, GrumpkinEngine, S1<_>, S2<_>>();
    test_nivc_trivial_with_compression_with::<Secp256k1Engine, Secq256k1Engine, S1<_>, S2<_>>();
  }

  #[derive(Clone)]
  struct BigPowerCircuit<E: Engine> {
    _p: PhantomData<E>,
  }

  impl<E: Engine> StepCircuit<E::Scalar> for BigPowerCircuit<E> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      _pc: Option<&AllocatedNum<E::Scalar>>,
      z: &[AllocatedNum<E::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<E::Scalar>>,
        Vec<AllocatedNum<E::Scalar>>,
      ),
      SynthesisError,
    > {
      let mut x = z[0].clone();
      let mut y = x.clone();
      for i in 0..10_000 {
        y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
        x = y.clone();
      }

      let next_pc = AllocatedNum::alloc(cs.namespace(|| "next_pc"), || Ok(E::Scalar::from(0u64)))?;

      cs.enforce(
        || "next_pc = 0",
        |lc| lc + CS::one(),
        |lc| lc + next_pc.get_variable(),
        |lc| lc,
      );

      Ok((Some(next_pc), vec![y]))
    }
  }

  #[derive(Clone)]
  enum BigTestCircuit<E: Engine> {
    Square(SquareCircuit<E>),
    BigPower(BigPowerCircuit<E>),
  }

  impl<E: Engine> BigTestCircuit<E> {
    fn new(num_steps: usize) -> Vec<Self> {
      let mut circuits = Vec::new();

      for idx in 0..num_steps {
        if idx % 2 == 0 {
          circuits.push(Self::Square(SquareCircuit { _p: PhantomData }))
        } else {
          circuits.push(Self::BigPower(BigPowerCircuit { _p: PhantomData }))
        }
      }

      circuits
    }
  }

  impl<E: Engine> StepCircuit<E::Scalar> for BigTestCircuit<E> {
    fn arity(&self) -> usize {
      1
    }

    fn circuit_index(&self) -> usize {
      match self {
        BigTestCircuit::Square(c) => c.circuit_index(),
        BigTestCircuit::BigPower(c) => c.circuit_index(),
      }
    }

    fn synthesize<CS: ConstraintSystem<E::Scalar>>(
      &self,
      cs: &mut CS,
      pc: Option<&AllocatedNum<E::Scalar>>,
      z: &[AllocatedNum<E::Scalar>],
    ) -> Result<
      (
        Option<AllocatedNum<E::Scalar>>,
        Vec<AllocatedNum<E::Scalar>>,
      ),
      SynthesisError,
    > {
      match self {
        BigTestCircuit::Square(c) => c.synthesize(cs, pc, z),
        BigTestCircuit::BigPower(c) => c.synthesize(cs, pc, z),
      }
    }
  }

  impl<E1, E2> NonUniformCircuit<E1, E2, BigTestCircuit<E1>, TrivialSecondaryCircuit<E2::Scalar>>
    for BigTestCircuit<E1>
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    fn num_circuits(&self) -> usize {
      2
    }

    fn primary_circuit(&self, circuit_index: usize) -> BigTestCircuit<E1> {
      match circuit_index {
        0 => Self::Square(SquareCircuit { _p: PhantomData }),
        1 => Self::BigPower(BigPowerCircuit { _p: PhantomData }),
        _ => panic!("Invalid circuit index"),
      }
    }

    fn secondary_circuit(&self) -> TrivialSecondaryCircuit<E2::Scalar> {
      Default::default()
    }
  }

  fn test_compression_with_circuit_size_difference_with<E1, E2, S1, S2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    S1: BatchedRelaxedR1CSSNARKTrait<E1>,
    S2: RelaxedR1CSSNARKTrait<E2>,
  {
    const NUM_STEPS: usize = 4;

    let secondary_circuit = TrivialSecondaryCircuit::default();
    let test_circuits = BigTestCircuit::new(NUM_STEPS);

    let pp = PublicParams::setup(&test_circuits[0], &*S1::ck_floor(), &*S2::ck_floor());

    let z0_primary = vec![E1::Scalar::from(17u64)];
    let z0_secondary = vec![<E2 as Engine>::Scalar::ZERO];

    let mut recursive_snark = RecursiveSNARK::new(
      &pp,
      &test_circuits[0],
      &test_circuits[0],
      &secondary_circuit,
      &z0_primary,
      &z0_secondary,
    )
    .unwrap();

    for circuit in test_circuits.iter().take(NUM_STEPS) {
      let prove_res = recursive_snark.prove_step(&pp, circuit, &secondary_circuit);

      let verify_res = recursive_snark.verify(&pp, &z0_primary, &z0_secondary);

      assert!(prove_res.is_ok());
      assert!(verify_res.is_ok());
    }

    let (prover_key, verifier_key) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

    let compressed_prove_res = CompressedSNARK::prove(&pp, &prover_key, &recursive_snark);

    assert!(compressed_prove_res.is_ok());

    let compressed_snark = compressed_prove_res.unwrap();

    let compressed_verify_res =
      compressed_snark.verify(&pp, &verifier_key, &z0_primary, &z0_secondary);

    assert!(compressed_verify_res.is_ok());
  }

  #[test]
  fn test_compression_with_circuit_size_difference() {
    // ppSNARK
    test_compression_with_circuit_size_difference_with::<PallasEngine, VestaEngine, S1PP<_>, S2<_>>(
    );
    test_compression_with_circuit_size_difference_with::<
      Bn256EngineIPA,
      GrumpkinEngine,
      S1PP<_>,
      S2<_>,
    >();
    test_compression_with_circuit_size_difference_with::<
      Secp256k1Engine,
      Secq256k1Engine,
      S1PP<_>,
      S2<_>,
    >();
    // classic SNARK
    test_compression_with_circuit_size_difference_with::<PallasEngine, VestaEngine, S1<_>, S2<_>>();
    test_compression_with_circuit_size_difference_with::<
      Bn256EngineIPA,
      GrumpkinEngine,
      S1<_>,
      S2<_>,
    >();
    test_compression_with_circuit_size_difference_with::<
      Secp256k1Engine,
      Secq256k1Engine,
      S1<_>,
      S2<_>,
    >();
  }
}
