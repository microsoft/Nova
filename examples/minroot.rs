//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the MinRoot function, thereby realizing a Nova-based verifiable delay function (VDF).
type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;
use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use generic_array::typenum::U2;
use neptune::{
  circuit::poseidon_hash,
  poseidon::{Poseidon, PoseidonConstants},
  Strength,
};
use nova_snark::{
  traits::{Group, StepCircuit},
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use num_bigint::BigUint;
use std::marker::PhantomData;

// A trivial test circuit that we will use on the secondary curve
#[derive(Clone, Debug)]
struct TrivialTestCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F> StepCircuit<F> for TrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    Ok(z)
  }

  fn compute(&self, z: &F) -> F {
    *z
  }
}

#[derive(Clone, Debug)]
struct MinRootCircuit<F: PrimeField> {
  x_i: F,
  y_i: F,
  x_i_plus_1: F,
  y_i_plus_1: F,
  pc: PoseidonConstants<F, U2>,
}

impl<F: PrimeField> MinRootCircuit<F> {
  fn pow_vartime<S: AsRef<[u64]>>(base: &F, exp: S) -> F {
    let mut res = F::one();
    for e in exp.as_ref().iter().rev() {
      for i in (0..64).rev() {
        res = res.square();

        if ((*e >> i) & 1) == 1 {
          res.mul_assign(base);
        }
      }
    }

    res
  }

  // produces a sample non-deterministic advice
  fn new(num_steps: usize, x_0: &F, y_0: &F, pc: &PoseidonConstants<F, U2>) -> (F, Vec<Self>) {
    // although this code is written generically, it is tailored to Pallas' scalar field
    // (p - 3 / 5)
    let exp = BigUint::parse_bytes(
      b"23158417847463239084714197001737581570690445185553317903743794198714690358477",
      10,
    )
    .unwrap();

    let mut res = Vec::new();
    let mut x_i = *x_0;
    let mut y_i = *y_0;
    for _i in 0..num_steps {
      let x_i_plus_1 = Self::pow_vartime(&(x_i + y_i), exp.to_u64_digits());

      // sanity check
      let sq = x_i_plus_1 * x_i_plus_1;
      let quad = sq * sq;
      let fifth = quad * x_i_plus_1;
      debug_assert_eq!(fifth, x_i + y_i);

      let y_i_plus_1 = x_i;

      res.push(Self {
        x_i,
        y_i,
        x_i_plus_1,
        y_i_plus_1,
        pc: pc.clone(),
      });

      x_i = x_i_plus_1;
      y_i = y_i_plus_1;
    }

    let z0 = Poseidon::<F, U2>::new_with_preimage(&vec![x_0.clone(), y_0.clone()], pc).hash();

    (z0, res)
  }
}

impl<F> StepCircuit<F> for MinRootCircuit<F>
where
  F: PrimeField,
{
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    // Allocate four variables for holding non-deterministic advice: x_i, y_i, x_i_plus_1, y_i_plus_1
    let x_i = AllocatedNum::alloc(cs.namespace(|| "x_i"), || Ok(self.x_i))?;
    let y_i = AllocatedNum::alloc(cs.namespace(|| "y_i"), || Ok(self.y_i))?;
    let x_i_plus_1 = AllocatedNum::alloc(cs.namespace(|| "x_i_plus_1"), || Ok(self.x_i_plus_1))?;
    let y_i_plus_1 = AllocatedNum::alloc(cs.namespace(|| "y_i_plus_1"), || Ok(self.y_i_plus_1))?;

    // check that z = hash(x_i, y_i)
    let z_hash = poseidon_hash(
      cs.namespace(|| "input hash"),
      vec![x_i.clone(), y_i.clone()],
      &self.pc,
    )?;
    cs.enforce(
      || "z =? z_hash",
      |lc| lc + z_hash.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + z.get_variable(),
    );

    // check the following conditions hold:
    // (i) x_i_plus_1 = (x_i + y_i)^{1/5}, which can be more easily checked with x_i_plus_1^5 = x_i + y_i
    // (ii) y_i_plus_1 = x_i

    // (1) constraints for condition (i)
    let x_i_plus_1_sq = x_i_plus_1.square(cs.namespace(|| "x_i_plus_1_sq"))?;
    let x_i_plus_1_quad = x_i_plus_1_sq.square(cs.namespace(|| "x_i_plus_1_quad"))?;
    let x_i_plus_1_pow_5 = x_i_plus_1_quad.mul(cs.namespace(|| "x_i_plus_1_pow_5"), &x_i_plus_1)?;
    cs.enforce(
      || "x_i_plus_1_pow_5 = x_i + y_i",
      |lc| lc + x_i_plus_1_pow_5.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + x_i.get_variable() + y_i.get_variable(),
    );

    // (2) constraints for condition (ii); this constraint could be avoided if we just used x_i wherever y_i_plus_1 is used
    cs.enforce(
      || "y_i_plus_1 = x_i",
      |lc| lc + y_i_plus_1.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + x_i.get_variable(),
    );

    // return hash(x_i_plus_1, y_i_plus_1)
    poseidon_hash(
      cs.namespace(|| "output hash"),
      vec![x_i_plus_1, y_i_plus_1],
      &self.pc,
    )
  }

  fn compute(&self, z: &F) -> F {
    // sanity check
    let z_hash =
      Poseidon::<F, U2>::new_with_preimage(&vec![self.x_i.clone(), self.y_i.clone()], &self.pc)
        .hash();
    debug_assert_eq!(z, &z_hash);

    // compute output hash using advice
    Poseidon::<F, U2>::new_with_preimage(
      &vec![self.x_i_plus_1.clone(), self.y_i_plus_1.clone()],
      &self.pc,
    )
    .hash()
  }
}

fn main() {
  let pc = PoseidonConstants::<<G1 as Group>::Scalar, U2>::new_with_strength(Strength::Standard);

  let circuit_primary = MinRootCircuit {
    x_i: <G1 as Group>::Scalar::zero(),
    y_i: <G1 as Group>::Scalar::zero(),
    x_i_plus_1: <G1 as Group>::Scalar::zero(),
    y_i_plus_1: <G1 as Group>::Scalar::zero(),
    pc: pc.clone(),
  };

  let circuit_secondary = TrivialTestCircuit {
    _p: Default::default(),
  };

  // produce public parameters
  let pp = PublicParams::<
    G1,
    G2,
    MinRootCircuit<<G1 as Group>::Scalar>,
    TrivialTestCircuit<<G2 as Group>::Scalar>,
  >::setup(circuit_primary, circuit_secondary.clone());

  // produce non-deterministic advice
  let num_steps = 3;
  let (z0_primary, minroot_circuits) = MinRootCircuit::new(
    num_steps,
    &<G1 as Group>::Scalar::zero(),
    &<G1 as Group>::Scalar::one(),
    &pc,
  );
  let z0_secondary = <G2 as Group>::Scalar::zero();

  // produce a recursive SNARK
  let mut recursive_snark: Option<
    RecursiveSNARK<
      G1,
      G2,
      MinRootCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >,
  > = None;

  for circuit_primary in minroot_circuits.iter().take(num_steps) {
    let res = RecursiveSNARK::prove_step(
      &pp,
      recursive_snark,
      circuit_primary.clone(),
      circuit_secondary.clone(),
      z0_primary,
      z0_secondary,
    );
    assert!(res.is_ok());
    recursive_snark = Some(res.unwrap());
  }

  assert!(recursive_snark.is_some());
  let recursive_snark = recursive_snark.unwrap();

  // verify the recursive SNARK
  let res = recursive_snark.verify(&pp, num_steps, z0_primary, z0_secondary);
  assert!(res.is_ok());

  // produce a compressed SNARK
  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &recursive_snark);
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  // verify the compressed SNARK
  let res = compressed_snark.verify(&pp, num_steps, z0_primary, z0_secondary);
  assert!(res.is_ok());
}
