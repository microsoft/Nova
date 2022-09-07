//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the MinRoot function, thereby realizing a Nova-based verifiable delay function (VDF).
//! We execute a configurable number of iterations of the MinRoot function per step of Nova's recursion.
//! See the description of MinRoot in Section 8.2 in the following link: https://khovratovich.github.io/MinRoot/minroot.pdf
//! We use fifth roots instead of cube roots (as described in the above link) given our implementation targets Pasta curves
type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use nova_snark::{
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group,
  },
  CompressedSNARK, CompressedSNARKParams, RecursiveSNARK, RecursiveSNARKParams,
};
use num_bigint::BigUint;
use std::time::Instant;

#[derive(Clone, Debug)]
struct MinRootIteration<F: PrimeField> {
  i: F,
  x_i: F,
  y_i: F,
  i_plus_1: F,
  x_i_plus_1: F,
  y_i_plus_1: F,
}

impl<F: PrimeField> MinRootIteration<F> {
  // produces a sample non-deterministic advice, executing one invocation of MinRoot per step
  fn new(num_iters: usize, i_0: &F, x_0: &F, y_0: &F) -> (Vec<F>, Vec<Self>) {
    // although this code is written generically, it is tailored to Pallas' scalar field
    // (p - 3 / 5)
    let exp = BigUint::parse_bytes(
      b"23158417847463239084714197001737581570690445185553317903743794198714690358477",
      10,
    )
    .unwrap();

    let mut res = Vec::new();
    let mut i = *i_0;
    let mut x_i = *x_0;
    let mut y_i = *y_0;
    for _ii in 0..num_iters {
      let x_i_plus_1 = (x_i + y_i).pow_vartime(exp.to_u64_digits()); // computes the fifth root of x_i + y_i

      // sanity check
      let sq = x_i_plus_1 * x_i_plus_1;
      let quad = sq * sq;
      let fifth = quad * x_i_plus_1;
      debug_assert_eq!(fifth, x_i + y_i);

      let y_i_plus_1 = x_i + i;
      let i_plus_1 = i + F::one();

      res.push(Self {
        i,
        x_i,
        y_i,
        i_plus_1,
        x_i_plus_1,
        y_i_plus_1,
      });

      i = i_plus_1;
      x_i = x_i_plus_1;
      y_i = y_i_plus_1;
    }

    let z0 = vec![*i_0, *x_0, *y_0];

    (z0, res)
  }
}

#[derive(Clone, Debug)]
struct MinRootCircuit<F: PrimeField> {
  seq: Vec<MinRootIteration<F>>,
}

impl<F> StepCircuit<F> for MinRootCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    3
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let mut z_out: Result<Vec<AllocatedNum<F>>, SynthesisError> =
      Err(SynthesisError::AssignmentMissing);

    // variables to hold running x_i and y_i
    let mut i = z[0].clone();
    let mut x_i = z[1].clone();
    let mut y_i = z[2].clone();
    for ii in 0..self.seq.len() {
      // non deterministic advice
      let i_plus_1 = AllocatedNum::alloc(cs.namespace(|| format!("i_plus_1_iter_{}", ii)), || {
        Ok(self.seq[ii].i_plus_1)
      })?;
      let x_i_plus_1 =
        AllocatedNum::alloc(cs.namespace(|| format!("x_i_plus_1_iter_{}", ii)), || {
          Ok(self.seq[ii].x_i_plus_1)
        })?;
      let y_i_plus_1 =
        AllocatedNum::alloc(cs.namespace(|| format!("y_i_plus_1_iter_{}", ii)), || {
          Ok(self.seq[ii].y_i_plus_1)
        })?;

      // check the following conditions hold:
      // (i) x_i_plus_1 = (x_i + y_i)^{1/5}, which can be more easily checked with x_i_plus_1^5 = x_i + y_i

      let x_i_plus_1_sq =
        x_i_plus_1.square(cs.namespace(|| format!("x_i_plus_1_sq_iter_{}", ii)))?;
      let x_i_plus_1_quad =
        x_i_plus_1_sq.square(cs.namespace(|| format!("x_i_plus_1_quad_{}", ii)))?;
      cs.enforce(
        || format!("x_i_plus_1_quad * x_i_plus_1 = x_i + y_i iter_{}", ii),
        |lc| lc + x_i_plus_1_quad.get_variable(),
        |lc| lc + x_i_plus_1.get_variable(),
        |lc| lc + x_i.get_variable() + y_i.get_variable(),
      );

      // (ii) y_i_plus_1 = x_i + i
      cs.enforce(
        || format!("y_i_plus_1 = x_i + i  iter_{}", ii),
        |lc| lc + y_i_plus_1.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + x_i.get_variable() + i.get_variable(),
      );

      // (ii) i_plus_1 = i + i
      cs.enforce(
        || format!("i_plus_1 = i + i  iter_{}", ii),
        |lc| lc + i_plus_1.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + i.get_variable() + CS::one(),
      );

      // return (i_plus_1, x_i_plus_1, y_i_plus_1)
      if ii == self.seq.len() - 1 {
        z_out = Ok(vec![
          i_plus_1.clone(),
          x_i_plus_1.clone(),
          y_i_plus_1.clone(),
        ]);
      }

      // update i, x_i, and y_i for the next iteration
      i = i_plus_1;
      x_i = x_i_plus_1;
      y_i = y_i_plus_1;
    }

    z_out
  }

  fn output(&self, z: &[F]) -> Vec<F> {
    // sanity check
    debug_assert_eq!(z[0], self.seq[0].i);
    debug_assert_eq!(z[1], self.seq[0].x_i);
    debug_assert_eq!(z[2], self.seq[0].y_i);

    // compute output using advice
    vec![
      self.seq[self.seq.len() - 1].i_plus_1,
      self.seq[self.seq.len() - 1].x_i_plus_1,
      self.seq[self.seq.len() - 1].y_i_plus_1,
    ]
  }
}

fn main() {
  println!("Nova-based VDF with MinRoot delay function");
  println!("=========================================================");

  let num_steps = 10;
  for num_iters_per_step in [65536] {
    // number of iterations of MinRoot per Nova's recursive step
    let circuit_primary = MinRootCircuit {
      seq: vec![
        MinRootIteration {
          i: <G1 as Group>::Scalar::zero(),
          x_i: <G1 as Group>::Scalar::zero(),
          y_i: <G1 as Group>::Scalar::zero(),
          i_plus_1: <G1 as Group>::Scalar::one(),
          x_i_plus_1: <G1 as Group>::Scalar::zero(),
          y_i_plus_1: <G1 as Group>::Scalar::zero(),
        };
        num_iters_per_step
      ],
    };

    let circuit_secondary = TrivialTestCircuit::default();

    println!(
      "Proving {} iterations of MinRoot per step",
      num_iters_per_step
    );

    // produce public parameters
    println!("Producing public parameters...");
    let res_pp = RecursiveSNARKParams::<
      G1,
      G2,
      MinRootCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(circuit_primary, circuit_secondary.clone());
    println!(
      "Number of constraints per step (primary circuit): {}",
      res_pp.num_constraints().0
    );
    println!(
      "Number of constraints per step (secondary circuit): {}",
      res_pp.num_constraints().1
    );

    println!(
      "Number of variables per step (primary circuit): {}",
      res_pp.num_variables().0
    );
    println!(
      "Number of variables per step (secondary circuit): {}",
      res_pp.num_variables().1
    );

    // produce non-deterministic advice
    let (z0_primary, minroot_iterations) = MinRootIteration::new(
      num_iters_per_step * num_steps,
      &<G1 as Group>::Scalar::zero(),
      &<G1 as Group>::Scalar::zero(),
      &<G1 as Group>::Scalar::one(),
    );
    let minroot_circuits = (0..num_steps)
      .map(|i| MinRootCircuit {
        seq: (0..num_iters_per_step)
          .map(|j| MinRootIteration {
            i: minroot_iterations[i * num_iters_per_step + j].i,
            x_i: minroot_iterations[i * num_iters_per_step + j].x_i,
            y_i: minroot_iterations[i * num_iters_per_step + j].y_i,
            i_plus_1: minroot_iterations[i * num_iters_per_step + j].i_plus_1,
            x_i_plus_1: minroot_iterations[i * num_iters_per_step + j].x_i_plus_1,
            y_i_plus_1: minroot_iterations[i * num_iters_per_step + j].y_i_plus_1,
          })
          .collect::<Vec<_>>(),
      })
      .collect::<Vec<_>>();

    let z0_secondary = vec![<G2 as Group>::Scalar::zero()];

    type C1 = MinRootCircuit<<G1 as Group>::Scalar>;
    type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;
    // produce a recursive SNARK
    println!("Generating a RecursiveSNARK...");
    let mut recursive_snark: Option<RecursiveSNARK<G1, G2, C1, C2>> = None;

    for (i, circuit_primary) in minroot_circuits.iter().take(num_steps).enumerate() {
      let start = Instant::now();
      let res = RecursiveSNARK::prove_step(
        &res_pp,
        recursive_snark,
        circuit_primary.clone(),
        circuit_secondary.clone(),
        z0_primary.clone(),
        z0_secondary.clone(),
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
    let res = recursive_snark.verify(&res_pp, num_steps, z0_primary.clone(), z0_secondary.clone());
    println!(
      "RecursiveSNARK::verify: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());

    println!("Producing public parameters for CompressedSNARK");
    type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
    type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;
    let cs_pp = CompressedSNARKParams::<
      G1,
      G2,
      MinRootCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
      S1,
      S2,
    >::setup(&res_pp);

    // produce a compressed SNARK
    println!("Generating a CompressedSNARK using Spartan with IPA-PC...");
    let start = Instant::now();
    let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&res_pp, &cs_pp, &recursive_snark);
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
    let res = compressed_snark.verify(&res_pp, &cs_pp, num_steps, z0_primary, z0_secondary);
    println!(
      "CompressedSNARK::verify: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());
    println!("=========================================================");
  }
}
