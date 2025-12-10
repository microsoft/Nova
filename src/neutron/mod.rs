//! This module implements an IVC scheme based on the NeutronNova folding scheme.
//! This code currently lacks certain checks, so do not use this until the experimental feature is removed.
use crate::{
  constants::NUM_HASH_BITS,
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  frontend::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
    ConstraintSystem, SynthesisError,
  },
  r1cs::{CommitmentKeyHint, R1CSInstance, R1CSWitness},
  traits::{
    circuit::StepCircuit, AbsorbInRO2Trait, Engine, RO2Constants, RO2ConstantsCircuit, ROTrait,
  },
  CommitmentKey,
};
use core::marker::PhantomData;
use ff::Field;
use once_cell::sync::OnceCell;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

mod circuit;
pub mod nifs;
pub mod relation;

use circuit::{NeutronAugmentedCircuit, NeutronAugmentedCircuitInputs};
use nifs::NIFS;
use relation::{FoldedInstance, FoldedWitness, Structure};

/// A type that holds public parameters of Nova
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  F_arity: usize,

  ro_consts: RO2Constants<E1>,
  ro_consts_circuit: RO2ConstantsCircuit<E1>,
  ck: CommitmentKey<E1>,
  structure: Structure<E1>,

  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E1::Scalar>,
  _p: PhantomData<(C, E2)>,
}

impl<E1, E2, C> SimpleDigestible for PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
}

impl<E1, E2, C> PublicParams<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  /// Creates a new `PublicParams` for a circuit `C`.
  ///
  /// # Note
  ///
  /// Public parameters set up a number of bases for the homomorphic commitment scheme of Nova.
  ///
  /// Some final compressing SNARKs, like variants of Spartan, use computation commitments that require
  /// larger sizes for these parameters. These SNARKs provide a hint for these values by
  /// implementing `RelaxedR1CSSNARKTrait::ck_floor()`, which can be passed to this function.
  ///
  /// If you're not using such a SNARK, pass `nova_snark::traits::snark::default_ck_hint()` instead.
  ///
  /// # Arguments
  ///
  /// * `c`: The primary circuit of type `C`.
  /// * `ck_hint1`: A `CommitmentKeyHint` for `S1`, which is a function that provides a hint
  ///   for the number of generators required in the commitment scheme for the primary circuit.
  /// * `ck_hint2`: A `CommitmentKeyHint` for `S2`, similar to `ck_hint1`, but for the secondary circuit.
  ///
  /// # Example
  ///
  /// ```rust
  /// # use nova_snark::spartan::ppsnark::RelaxedR1CSSNARK;
  /// # use nova_snark::provider::ipa_pc::EvaluationEngine;
  /// # use nova_snark::provider::{PallasEngine, VestaEngine};
  /// # use nova_snark::traits::{circuit::TrivialCircuit, Engine, snark::RelaxedR1CSSNARKTrait};
  /// # use nova_snark::nova::PublicParams;
  ///
  /// type E1 = PallasEngine;
  /// type E2 = VestaEngine;
  /// type EE<E> = EvaluationEngine<E>;
  /// type SPrime<E> = RelaxedR1CSSNARK<E, EE<E>>;
  ///
  /// let circuit = TrivialCircuit::<<E1 as Engine>::Scalar>::default();
  /// // Only relevant for a SNARK using computational commitments, pass &(|_| 0)
  /// // or &*nova_snark::traits::snark::default_ck_hint() otherwise.
  /// let ck_hint1 = &*SPrime::<E1>::ck_floor();
  /// let ck_hint2 = &*SPrime::<E2>::ck_floor();
  ///
  /// let pp = PublicParams::setup(&circuit, ck_hint1, ck_hint2)?;
  /// Ok(())
  /// ```
  pub fn setup(
    c: &C,
    ck_hint1: &CommitmentKeyHint<E1>,
    _ck_hint2: &CommitmentKeyHint<E2>,
  ) -> Result<Self, NovaError> {
    let F_arity = c.arity();

    let ro_consts: RO2Constants<E1> = RO2Constants::<E1>::default();
    let ro_consts_circuit: RO2ConstantsCircuit<E1> = RO2ConstantsCircuit::<E1>::default();

    // Initialize ck for the primary
    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(None, c, ro_consts_circuit.clone());
    let mut cs: ShapeCS<E1> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (r1cs_shape, ck) = cs.r1cs_shape(ck_hint1);

    if r1cs_shape.num_io != 1 {
      return Err(NovaError::InvalidStepCircuitIO);
    }

    let structure = Structure::new(&r1cs_shape);

    let pp = PublicParams {
      F_arity,

      ro_consts,
      ro_consts_circuit,
      ck,
      structure,

      digest: OnceCell::new(),
      _p: Default::default(),
    };

    // call pp.digest() so the digest is computed here rather than in RecursiveSNARK methods
    let _ = pp.digest();

    Ok(pp)
  }

  /// Retrieve the digest of the public parameters.
  pub fn digest(&self) -> E1::Scalar {
    self
      .digest
      .get_or_try_init(|| DigestComputer::new(self).digest())
      .cloned()
      .expect("Failure in retrieving digest")
  }
}

/// A SNARK that proves the correct execution of an incremental computation
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RecursiveSNARK<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  z0: Vec<E1::Scalar>,

  r_W: FoldedWitness<E1>,
  r_U: FoldedInstance<E1>,
  ri: E1::Scalar,

  l_w: R1CSWitness<E1>,
  l_u: R1CSInstance<E1>,

  i: usize,

  zi: Vec<E1::Scalar>,

  _p: PhantomData<(C, E2)>,
}

impl<E1, E2, C> RecursiveSNARK<E1, E2, C>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C: StepCircuit<E1::Scalar>,
{
  /// Create new instance of recursive SNARK
  pub fn new(pp: &PublicParams<E1, E2, C>, c: &C, z0: &[E1::Scalar]) -> Result<Self, NovaError> {
    if z0.len() != pp.F_arity {
      return Err(NovaError::InvalidInitialInputLength);
    }

    let ri = E1::Scalar::random(&mut OsRng);

    // base case for the primary
    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      pp.digest(),
      E1::Scalar::ZERO,
      z0.to_vec(),
      None,
      None,
      None,
      ri, // "r next"
      None,
      None,
      None,
      None,
    );

    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone());
    let zi = circuit.synthesize(&mut cs)?;
    let (l_u, l_w) = cs.r1cs_instance_and_witness(&pp.structure.S, &pp.ck)?;

    assert!((zi.len() == pp.F_arity), "Invalid step length");

    let zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    Ok(Self {
      z0: z0.to_vec(),
      r_W: FoldedWitness::default(&pp.structure),
      r_U: FoldedInstance::default(&pp.structure),
      ri,
      l_w,
      l_u,
      i: 0,
      zi,
      _p: Default::default(),
    })
  }

  /// Updates the provided `RecursiveSNARK` by executing a step of the incremental computation
  pub fn prove_step(&mut self, pp: &PublicParams<E1, E2, C>, c: &C) -> Result<(), NovaError> {
    // first step was already done in the constructor
    if self.i == 0 {
      self.i = 1;
      return Ok(());
    }

    // fold the last instance with the running instance
    let (nifs, (r_U, r_W)) = NIFS::prove(
      &pp.ck,
      &pp.ro_consts,
      &pp.digest(),
      &pp.structure,
      &self.r_U,
      &self.r_W,
      &self.l_u,
      &self.l_w,
    )?;

    let r_next = E1::Scalar::random(&mut OsRng);

    let mut cs = SatisfyingAssignment::<E1>::new();
    let inputs: NeutronAugmentedCircuitInputs<E1> = NeutronAugmentedCircuitInputs::new(
      pp.digest(),
      E1::Scalar::from(self.i as u64),
      self.z0.to_vec(),
      Some(self.zi.clone()),
      Some(self.r_U.clone()),
      Some(self.ri),
      r_next,
      Some(self.l_u.clone()),
      Some(nifs),
      Some(r_U.comm_W),
      Some(r_U.comm_E),
    );

    let circuit: NeutronAugmentedCircuit<'_, E1, C> =
      NeutronAugmentedCircuit::new(Some(inputs), c, pp.ro_consts_circuit.clone());
    let zi = circuit.synthesize(&mut cs)?;

    let (l_u, l_w) = cs.r1cs_instance_and_witness(&pp.structure.S, &pp.ck)?;

    // update the running instances and witnesses
    self.zi = zi
      .iter()
      .map(|v| v.get_value().ok_or(SynthesisError::AssignmentMissing))
      .collect::<Result<Vec<<E1 as Engine>::Scalar>, _>>()?;

    self.r_U = r_U;
    self.r_W = r_W;

    self.i += 1;

    self.ri = r_next;

    self.l_u = l_u;
    self.l_w = l_w;

    Ok(())
  }

  /// Verify the correctness of the `RecursiveSNARK`
  pub fn verify(
    &self,
    pp: &PublicParams<E1, E2, C>,
    num_steps: usize,
    z0: &[E1::Scalar],
  ) -> Result<Vec<E1::Scalar>, NovaError> {
    // number of steps cannot be zero
    let is_num_steps_zero = num_steps == 0;

    // check if the provided proof has executed num_steps
    let is_num_steps_not_match = self.i != num_steps;

    // check if the initial inputs match
    let is_inputs_not_match = self.z0 != z0;

    // check if the (relaxed) R1CS instances have two public outputs
    let is_instance_has_two_outputs = self.l_u.X.len() != 1 || self.r_U.X.len() != 1;

    if is_num_steps_zero
      || is_num_steps_not_match
      || is_inputs_not_match
      || is_instance_has_two_outputs
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid number of steps or inputs".to_string(),
      });
    }

    // check if the output hashes in R1CS instances point to the right running instance
    let hash = {
      let mut hasher = E1::RO2::new(pp.ro_consts.clone());
      hasher.absorb(pp.digest());
      hasher.absorb(E1::Scalar::from(num_steps as u64));
      for e in z0 {
        hasher.absorb(*e);
      }
      for e in &self.zi {
        hasher.absorb(*e);
      }
      self.r_U.absorb_in_ro2(&mut hasher);
      hasher.absorb(self.ri);

      hasher.squeeze(NUM_HASH_BITS)
    };

    if hash != self.l_u.X[0] {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid output hash in R1CS instance".to_string(),
      });
    }

    // check the satisfiability of the provided instances
    let (res_r, res_l) = rayon::join(
      || pp.structure.is_sat(&pp.ck, &self.r_U, &self.r_W),
      || pp.structure.S.is_sat(&pp.ck, &self.l_u, &self.l_w),
    );

    // check the returned res objects
    res_r?;
    res_l?;

    Ok(self.zi.clone())
  }

  /// Get the outputs after the last step of computation.
  pub fn outputs(&self) -> &[E1::Scalar] {
    &self.zi
  }

  /// The number of steps which have been executed thus far.
  pub fn num_steps(&self) -> usize {
    self.i
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    provider::{
      pedersen::CommitmentKeyExtTrait, traits::DlogGroup, Bn256EngineIPA, Bn256EngineKZG,
      GrumpkinEngine, PallasEngine, Secp256k1Engine, Secq256k1Engine, VestaEngine,
    },
    traits::{
      circuit::TrivialCircuit,
      snark::{default_ck_hint, RelaxedR1CSSNARKTrait},
    },
    CommitmentEngineTrait,
  };
  use core::{fmt::Write, marker::PhantomData};
  use expect_test::{expect, Expect};
  use ff::PrimeField;

  type EE<E> = crate::provider::ipa_pc::EvaluationEngine<E>;
  type SPrime<E, EE> = crate::spartan::ppsnark::RelaxedR1CSSNARK<E, EE>;

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

  fn test_pp_digest_with<E1, E2, C>(circuit: &C, expected: &Expect)
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
    E1::GE: DlogGroup,
    E2::GE: DlogGroup,
    C: StepCircuit<E1::Scalar>,
    // required to use the IPA in the initialization of the commitment key hints below
    <E1::CE as CommitmentEngineTrait<E1>>::CommitmentKey: CommitmentKeyExtTrait<E1>,
    <E2::CE as CommitmentEngineTrait<E2>>::CommitmentKey: CommitmentKeyExtTrait<E2>,
  {
    // this tests public parameters with a size specifically intended for a spark-compressed SNARK
    let ck_hint1 = &*SPrime::<E1, EE<E1>>::ck_floor();
    let ck_hint2 = &*SPrime::<E2, EE<E2>>::ck_floor();
    let pp = PublicParams::<E1, E2, C>::setup(circuit, ck_hint1, ck_hint2).unwrap();

    let digest_str = pp
      .digest()
      .to_repr()
      .as_ref()
      .iter()
      .fold(String::new(), |mut output, b| {
        let _ = write!(output, "{b:02x}");
        output
      });
    expected.assert_eq(&digest_str);
  }

  #[test]
  fn test_pp_digest() {
    test_pp_digest_with::<PallasEngine, VestaEngine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["b5cc251cc083fd646760c582d688baf03fa1b4d7fc2412952df51f59ee8c1c00"],
    );

    test_pp_digest_with::<Bn256EngineIPA, GrumpkinEngine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["9f842945b19b47c90733fbca7a5749ad120ad96114afdc0b63b2005020f4ea02"],
    );

    test_pp_digest_with::<Secp256k1Engine, Secq256k1Engine, _>(
      &TrivialCircuit::<_>::default(),
      &expect!["956c929feb6c825d072ced92d883caf9fe4aaaddcfefcfc08e61713db1995400"],
    );
  }

  fn test_ivc_trivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit1 = TrivialCircuit::<<E1 as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<E1, E2, TrivialCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit1,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark =
      RecursiveSNARK::new(&pp, &test_circuit1, &[<E1 as Engine>::Scalar::ZERO]).unwrap();

    let res = recursive_snark.prove_step(&pp, &test_circuit1);

    assert!(res.is_ok());

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ZERO]);
    assert!(res.is_ok());
  }

  #[test]
  fn test_ivc_trivial() {
    test_ivc_trivial_with::<PallasEngine, VestaEngine>();
    test_ivc_trivial_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_trivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_nontrivial_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let circuit = CubicCircuit::default();

    // produce public parameters
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 3;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &circuit,
      &[<E1 as Engine>::Scalar::ONE],
    )
    .unwrap();

    for i in 0..num_steps {
      let res = recursive_snark.prove_step(&pp, &circuit);
      assert!(res.is_ok());

      // verify the recursive snark at each step of recursion
      let res = recursive_snark.verify(&pp, i + 1, &[<E1 as Engine>::Scalar::ONE]);
      assert!(res.is_ok());
    }

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ONE]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    // sanity: check the claimed output with a direct computation of the same
    let mut zn_direct = vec![<E1 as Engine>::Scalar::ONE];
    for _i in 0..num_steps {
      zn_direct = circuit.clone().output(&zn_direct);
    }
    assert_eq!(zn, zn_direct);
    assert_eq!(zn, vec![<E1 as Engine>::Scalar::from(0x2aaaaa3u64)]);
  }

  #[test]
  fn test_ivc_nontrivial_neutron() {
    test_ivc_nontrivial_with::<PallasEngine, VestaEngine>();
    test_ivc_nontrivial_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_nontrivial_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_ivc_base_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    let test_circuit1 = CubicCircuit::<<E1 as Engine>::Scalar>::default();

    // produce public parameters
    let pp = PublicParams::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::setup(
      &test_circuit1,
      &*default_ck_hint(),
      &*default_ck_hint(),
    )
    .unwrap();

    let num_steps = 1;

    // produce a recursive SNARK
    let mut recursive_snark = RecursiveSNARK::<E1, E2, CubicCircuit<<E1 as Engine>::Scalar>>::new(
      &pp,
      &test_circuit1,
      &[<E1 as Engine>::Scalar::ONE],
    )
    .unwrap();

    // produce a recursive SNARK
    let res = recursive_snark.prove_step(&pp, &test_circuit1);

    assert!(res.is_ok());

    // verify the recursive SNARK
    let res = recursive_snark.verify(&pp, num_steps, &[<E1 as Engine>::Scalar::ONE]);
    assert!(res.is_ok());

    let zn = res.unwrap();

    assert_eq!(zn, vec![<E1 as Engine>::Scalar::from(7u64)]);
  }

  #[test]
  fn test_ivc_base() {
    test_ivc_base_with::<PallasEngine, VestaEngine>();
    test_ivc_base_with::<Bn256EngineKZG, GrumpkinEngine>();
    test_ivc_base_with::<Secp256k1Engine, Secq256k1Engine>();
  }

  fn test_setup_with<E1, E2>()
  where
    E1: Engine<Base = <E2 as Engine>::Scalar>,
    E2: Engine<Base = <E1 as Engine>::Scalar>,
  {
    #[derive(Clone, Debug, Default)]
    struct CircuitWithInputize<F: PrimeField> {
      _p: PhantomData<F>,
    }

    impl<F: PrimeField> StepCircuit<F> for CircuitWithInputize<F> {
      fn arity(&self) -> usize {
        1
      }

      fn synthesize<CS: ConstraintSystem<F>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<F>],
      ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
        let x = &z[0];
        let y = x.square(cs.namespace(|| "x_sq"))?;
        y.inputize(cs.namespace(|| "y"))?; // inputize y
        Ok(vec![y])
      }
    }

    // produce public parameters with trivial secondary
    let circuit = CircuitWithInputize::<<E1 as Engine>::Scalar>::default();
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));

    // produce public parameters with the trivial primary
    let circuit = CircuitWithInputize::<E1::Scalar>::default();
    let pp = PublicParams::<E1, E2, CircuitWithInputize<E1::Scalar>>::setup(
      &circuit,
      &*default_ck_hint(),
      &*default_ck_hint(),
    );
    assert!(pp.is_err());
    assert_eq!(pp.err(), Some(NovaError::InvalidStepCircuitIO));
  }

  #[test]
  fn test_setup() {
    test_setup_with::<Bn256EngineKZG, GrumpkinEngine>();
  }
}
