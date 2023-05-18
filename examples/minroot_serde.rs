#![allow(non_snake_case)]

//! Demonstrates how to use Nova to produce a recursive proof of the correct execution of
//! iterations of the MinRoot function, thereby realizing a Nova-based verifiable delay function (VDF).
//! We execute a configurable number of iterations of the MinRoot function per step of Nova's recursion.
type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use abomonation::{decode, encode};
use abomonation_derive::Abomonation;
use ff::PrimeField;
use nova_snark::{
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group,
  },
  PublicParams,
};
use std::{io::Write, mem::size_of, time::Instant};

/// Unspeakable horrors
unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
  ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
}

/// this is **incredibly, INCREDIBLY** dangerous
unsafe fn entomb_F<F: PrimeField, W: Write>(f: &F, bytes: &mut W) -> std::io::Result<()> {
  println!("entomb: {}", size_of::<F>());
  // this is **incredibly, INCREDIBLY** dangerous
  bytes.write_all(any_as_u8_slice(&f))?;
  Ok(())
}

/// this is **incredibly, INCREDIBLY** dangerous
unsafe fn exhume_F<'a, 'b, F: PrimeField>(f: &mut F, bytes: &'a mut [u8]) -> Option<&'a mut [u8]> {
  let (mine, rest) = bytes.split_at_mut(size_of::<F>());
  let mine = (mine as *const [u8]) as *const F;
  std::ptr::write(f, std::ptr::read(mine));
  Some(rest)
}

#[derive(Clone, PartialEq, Debug)]
struct MinRootIteration<F: PrimeField> {
  x_i: F,
  y_i: F,
  x_i_plus_1: F,
  y_i_plus_1: F,
}

impl<F: PrimeField> abomonation::Abomonation for MinRootIteration<F> {
  unsafe fn entomb<W: std::io::Write>(&self, bytes: &mut W) -> std::io::Result<()> {
    entomb_F(&self.x_i, bytes)?;
    entomb_F(&self.y_i, bytes)?;
    entomb_F(&self.x_i_plus_1, bytes)?;
    entomb_F(&self.y_i_plus_1, bytes)?;
    Ok(())
  }

  unsafe fn exhume<'a, 'b>(&'a mut self, bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
    let bytes = exhume_F(&mut self.x_i, bytes)?;
    let bytes = exhume_F(&mut self.y_i, bytes)?;
    let bytes = exhume_F(&mut self.x_i_plus_1, bytes)?;
    let bytes = exhume_F(&mut self.y_i_plus_1, bytes)?;
    Some(bytes)
  }

  fn extent(&self) -> usize {
    0
  }
}

#[derive(Clone, PartialEq, Debug, Abomonation)]
struct MinRootCircuit<F: PrimeField> {
  seq: Vec<MinRootIteration<F>>,
}

impl<F> StepCircuit<F> for MinRootCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    2
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    let mut z_out: Result<Vec<AllocatedNum<F>>, SynthesisError> =
      Err(SynthesisError::AssignmentMissing);

    // use the provided inputs
    let x_0 = z[0].clone();
    let y_0 = z[1].clone();

    // variables to hold running x_i and y_i
    let mut x_i = x_0;
    let mut y_i = y_0;
    for i in 0..self.seq.len() {
      // non deterministic advice
      let x_i_plus_1 =
        AllocatedNum::alloc(cs.namespace(|| format!("x_i_plus_1_iter_{i}")), || {
          Ok(self.seq[i].x_i_plus_1)
        })?;

      // check the following conditions hold:
      // (i) x_i_plus_1 = (x_i + y_i)^{1/5}, which can be more easily checked with x_i_plus_1^5 = x_i + y_i
      // (ii) y_i_plus_1 = x_i
      // (1) constraints for condition (i) are below
      // (2) constraints for condition (ii) is avoided because we just used x_i wherever y_i_plus_1 is used
      let x_i_plus_1_sq = x_i_plus_1.square(cs.namespace(|| format!("x_i_plus_1_sq_iter_{i}")))?;
      let x_i_plus_1_quad =
        x_i_plus_1_sq.square(cs.namespace(|| format!("x_i_plus_1_quad_{i}")))?;
      cs.enforce(
        || format!("x_i_plus_1_quad * x_i_plus_1 = x_i + y_i_iter_{i}"),
        |lc| lc + x_i_plus_1_quad.get_variable(),
        |lc| lc + x_i_plus_1.get_variable(),
        |lc| lc + x_i.get_variable() + y_i.get_variable(),
      );

      if i == self.seq.len() - 1 {
        z_out = Ok(vec![x_i_plus_1.clone(), x_i.clone()]);
      }

      // update x_i and y_i for the next iteration
      y_i = x_i;
      x_i = x_i_plus_1;
    }

    z_out
  }

  fn output(&self, z: &[F]) -> Vec<F> {
    // sanity check
    debug_assert_eq!(z[0], self.seq[0].x_i);
    debug_assert_eq!(z[1], self.seq[0].y_i);

    // compute output using advice
    vec![
      self.seq[self.seq.len() - 1].x_i_plus_1,
      self.seq[self.seq.len() - 1].y_i_plus_1,
    ]
  }
}

fn main() {
  println!("Nova-based VDF with MinRoot delay function");
  println!("=========================================================");

  let num_iters_per_step = 1024;
  // number of iterations of MinRoot per Nova's recursive step
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
  };

  let circuit_secondary = TrivialTestCircuit::default();

  println!("Proving {num_iters_per_step} iterations of MinRoot per step");

  let mut bytes = Vec::new();
  unsafe {
    let start = Instant::now();
    println!("Producing public parameters...");
    let pp = PublicParams::<
      G1,
      G2,
      MinRootCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(circuit_primary.clone(), circuit_secondary.clone());
    println!("PublicParams::setup, took {:?} ", start.elapsed());
    encode(&pp, &mut bytes).unwrap()
  };
  println!("Encoded!");
  println!("Read size: {}", bytes.len());

  if let Some((result, remaining)) = unsafe {
    decode::<
      PublicParams<
        G1,
        G2,
        MinRootCircuit<<G1 as Group>::Scalar>,
        TrivialTestCircuit<<G2 as Group>::Scalar>,
      >,
    >(&mut bytes)
  } {
    println!("Producing public parameters...");
    let pp = PublicParams::<
      G1,
      G2,
      MinRootCircuit<<G1 as Group>::Scalar>,
      TrivialTestCircuit<<G2 as Group>::Scalar>,
    >::setup(circuit_primary.clone(), circuit_secondary.clone());
    assert!(result.clone() == pp, "not equal!");
    assert!(remaining.len() == 0);
  } else {
    println!("Something terrible happened");
  }

  // let mut bytes = Vec::new();
  // let zero = pasta_curves::pallas::Scalar::zero();
  // let mut zero_res = pasta_curves::pallas::Scalar::one();
  // println!("equal? {}", zero == zero_res);
  // unsafe {
  //   entomb(zero, &mut bytes).unwrap();
  //   exhume(&mut zero_res, &mut bytes);
  // };
  // println!("equal? {}", zero == zero_res);
  // assert!(zero == zero_res);
}
