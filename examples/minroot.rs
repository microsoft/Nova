//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the MinRoot function, thereby realizing a Nova-based verifiable delay function (VDF).
//! We execute a configurable number of iterations of the MinRoot function per step of Nova's recursion.
type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use generic_array::typenum::U2;
use neptune::{
  circuit::poseidon_hash,
  poseidon::{Poseidon, PoseidonConstants},
  Strength,
};
use nova_snark::{
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use num_bigint::BigUint;
use std::time::Instant;

#[derive(Clone, Debug)]
struct MinRootIteration<F: PrimeField> {
  x_i: F,
  y_i: F,
  x_i_plus_1: F,
  y_i_plus_1: F,
}

impl<F: PrimeField> MinRootIteration<F> {
  // produces a sample non-deterministic advice, executing one invocation of MinRoot per step
  fn new(num_iters: usize, x_0: &F, y_0: &F, pc: &PoseidonConstants<F, U2>) -> (F, Vec<Self>) {
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
    for _i in 0..num_iters {
      let x_i_plus_1 = (x_i + y_i).pow_vartime(exp.to_u64_digits()); // computes the fifth root of x_i + y_i

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
      });

      x_i = x_i_plus_1;
      y_i = y_i_plus_1;
    }

    let z0 = Poseidon::<F, U2>::new_with_preimage(&[*x_0, *y_0], pc).hash();

    (z0, res)
  }
}

#[derive(Clone, Debug)]
struct MinRootCircuit<F: PrimeField> {
  seq: Vec<MinRootIteration<F>>,
  pc: PoseidonConstants<F, U2>,
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
    let mut z_out: Result<AllocatedNum<F>, SynthesisError> = Err(SynthesisError::AssignmentMissing);

    // allocate variables to hold x_0 and y_0
    let x_0 = AllocatedNum::alloc(cs.namespace(|| format!("x_0")), || Ok(self.seq[0].x_i))?;
    let y_0 = AllocatedNum::alloc(cs.namespace(|| format!("y_0")), || Ok(self.seq[0].y_i))?;

    // variables to hold running x_i and y_i
    let mut x_i = x_0;
    let mut y_i = y_0;
    for i in 0..self.seq.len() {
      // non deterministic advice
      let x_i_plus_1 =
        AllocatedNum::alloc(cs.namespace(|| format!("x_i_plus_1_iter_{}", i)), || {
          Ok(self.seq[i].x_i_plus_1)
        })?;

      // check that z = hash(x_i, y_i), where z is an output from the prior step
      if i == 0 {
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
      }

      // check the following conditions hold:
      // (i) x_i_plus_1 = (x_i + y_i)^{1/5}, which can be more easily checked with x_i_plus_1^5 = x_i + y_i
      // (ii) y_i_plus_1 = x_i
      // (1) constraints for condition (i) are below
      // (2) constraints for condition (ii) is avoided because we just used x_i wherever y_i_plus_1 is used
      let x_i_plus_1_sq =
        x_i_plus_1.square(cs.namespace(|| format!("x_i_plus_1_sq_iter_{}", i)))?;
      let x_i_plus_1_quad =
        x_i_plus_1_sq.square(cs.namespace(|| format!("x_i_plus_1_quad_{}", i)))?;
      cs.enforce(
        || format!("x_i_plus_1_quad * x_i_plus_1 = x_i + y_i_iter_{}", i),
        |lc| lc + x_i_plus_1_quad.get_variable(),
        |lc| lc + x_i_plus_1.get_variable(),
        |lc| lc + x_i.get_variable() + y_i.get_variable(),
      );

      // return hash(x_i_plus_1, y_i_plus_1) since Nova circuits expect a single output
      if i == self.seq.len() - 1 {
        z_out = poseidon_hash(
          cs.namespace(|| "output hash"),
          vec![x_i_plus_1.clone(), x_i.clone()],
          &self.pc,
        );
      }

      // update x_i and y_i for the next iteration
      y_i = x_i;
      x_i = x_i_plus_1;
    }

    z_out
  }

  fn output(&self, z: &F) -> F {
    // sanity check
    let z_hash =
      Poseidon::<F, U2>::new_with_preimage(&[self.seq[0].x_i, self.seq[0].y_i], &self.pc).hash();
    debug_assert_eq!(z, &z_hash);

    // compute output hash using advice
    let iters = self.seq.len();
    Poseidon::<F, U2>::new_with_preimage(
      &[
        self.seq[iters - 1].x_i_plus_1,
        self.seq[iters - 1].y_i_plus_1,
      ],
      &self.pc,
    )
    .hash()
  }
}

fn main() {
  println!("Nova-based VDF with MinRoot delay function");
  println!("=========================================================");

  let num_steps = 10;
  for num_iters_per_step in [1024, 2048, 4096, 8192, 16384, 32768, 65535] {
    // number of iterations of MinRoot per Nova's recursive step
    let pc = PoseidonConstants::<<G1 as Group>::Scalar, U2>::new_with_strength(Strength::Standard);
    let circuit_primary = MinRootCircuit {
      seq: vec![
        MinRootIteration {
          x_i: <G1 as Group>::Scalar::zero(),
          y_i: <G1 as Group>::Scalar::zero(),
          x_i_plus_1: <G1 as Group>::Scalar::zero(),
          y_i_plus_1: <G1 as Group>::Scalar::zero(),
        };
        num_iters_per_step
      ],
      pc: pc.clone(),
    };

    let circuit_secondary = TrivialTestCircuit::default();

    println!(
      "Proving {} iterations of MinRoot per step",
      num_iters_per_step
    );

    // produce public parameters
    println!("Producing public parameters...");
    let pp = PublicParams::<
      G1,
      G2,
      MinRootCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(circuit_primary, circuit_secondary.clone());
    println!(
      "Number of constraints per step (primary circuit): {}",
      pp.num_constraints().0
    );
    println!(
      "Number of constraints per step (secondary circuit): {}",
      pp.num_constraints().1
    );

    println!(
      "Number of variables per step (primary circuit): {}",
      pp.num_variables().0
    );
    println!(
      "Number of variables per step (secondary circuit): {}",
      pp.num_variables().1
    );

    // produce non-deterministic advice
    let (z0_primary, minroot_iterations) = MinRootIteration::new(
      num_iters_per_step * num_steps,
      &<G1 as Group>::Scalar::zero(),
      &<G1 as Group>::Scalar::one(),
      &pc,
    );
    let minroot_circuits = (0..num_steps)
      .map(|i| MinRootCircuit {
        seq: (0..num_iters_per_step)
          .map(|j| MinRootIteration {
            x_i: minroot_iterations[i * num_iters_per_step + j].x_i,
            y_i: minroot_iterations[i * num_iters_per_step + j].y_i,
            x_i_plus_1: minroot_iterations[i * num_iters_per_step + j].x_i_plus_1,
            y_i_plus_1: minroot_iterations[i * num_iters_per_step + j].y_i_plus_1,
          })
          .collect::<Vec<_>>(),
        pc: pc.clone(),
      })
      .collect::<Vec<_>>();

    let z0_secondary = <G2 as Group>::Scalar::zero();

    type C1 = MinRootCircuit<<G1 as Group>::Scalar>;
    type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
    // produce a recursive SNARK
    println!("Generating a RecursiveSNARK...");
    let mut recursive_snark: Option<RecursiveSNARK<G1, G2, C1, C2>> = None;

    for (i, circuit_primary) in minroot_circuits.iter().take(num_steps).enumerate() {
      let start = Instant::now();
      let res = RecursiveSNARK::prove_step(
        &pp,
        recursive_snark,
        circuit_primary.clone(),
        circuit_secondary.clone(),
        z0_primary,
        z0_secondary,
      );
      assert!(res.is_ok());
      println!(
        "RecursiveSNARK::prove_step {}: {:?}, took {:?} ",
        i,
        res.is_ok(),
        start.elapsed()
      );
      recursive_snark = Some(res.unwrap());
    }

    assert!(recursive_snark.is_some());
    let recursive_snark = recursive_snark.unwrap();

    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let start = Instant::now();
    let res = recursive_snark.verify(&pp, num_steps, z0_primary, z0_secondary);
    println!(
      "RecursiveSNARK::verify: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());

    // produce a compressed SNARK
    println!("Generating a CompressedSNARK using Spartan with IPA-PC...");
    let start = Instant::now();
    type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
    type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;
    let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &recursive_snark);
    println!(
      "CompressedSNARK::prove: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    // verify the compressed SNARK
    println!("Verifying a CompressedSNARK...");
    let start = Instant::now();
    let res = compressed_snark.verify(&pp, num_steps, z0_primary, z0_secondary);
    println!(
      "CompressedSNARK::verify: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());
    println!("=========================================================");
  }
}
