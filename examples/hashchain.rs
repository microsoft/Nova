//! This example proves the knowledge of preimage to a hash chain tail, with a configurable number of elements per hash chain node.
//! The output of each step tracks the current tail of the hash chain
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use generic_array::typenum::U24;
use neptune::{
  circuit2::Elt,
  sponge::{
    api::{IOPattern, SpongeAPI, SpongeOp},
    circuit::SpongeCircuit,
    vanilla::{Mode::Simplex, Sponge, SpongeTrait},
  },
  Strength,
};
use nova_snark::{
  provider::{Bn256EngineKZG, GrumpkinEngine},
  traits::{
    circuit::{StepCircuit, TrivialCircuit},
    snark::RelaxedR1CSSNARKTrait,
    Engine, Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use std::time::Instant;

type E1 = Bn256EngineKZG;
type E2 = GrumpkinEngine;
type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>; // non-preprocessing SNARK
type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>; // non-preprocessing SNARK

#[derive(Clone, Debug)]
struct HashChainCircuit<G: Group> {
  num_elts_per_step: usize,
  x_i: Vec<G::Scalar>,
}

impl<G: Group> HashChainCircuit<G> {
  // produces a preimage to be hashed
  fn new(num_elts_per_step: usize) -> Self {
    let mut rng = rand::thread_rng();
    let x_i = (0..num_elts_per_step)
      .map(|_| G::Scalar::random(&mut rng))
      .collect::<Vec<_>>();

    Self {
      num_elts_per_step,
      x_i,
    }
  }
}

impl<G: Group> StepCircuit<G::Scalar> for HashChainCircuit<G> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<G::Scalar>>(
    &self,
    cs: &mut CS,
    z_in: &[AllocatedNum<G::Scalar>],
  ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
    // z_in provides the running digest
    assert_eq!(z_in.len(), 1);

    // allocate x_i
    let x_i = (0..self.num_elts_per_step)
      .map(|i| AllocatedNum::alloc(cs.namespace(|| format!("x_{}", i)), || Ok(self.x_i[i])))
      .collect::<Result<Vec<_>, _>>()?;

    // concatenate z_in and x_i
    let mut m = z_in.to_vec();
    m.extend(x_i);

    let elt = m
      .iter()
      .map(|x| Elt::Allocated(x.clone()))
      .collect::<Vec<_>>();

    let num_absorbs = 1 + self.num_elts_per_step as u32;

    let parameter = IOPattern(vec![SpongeOp::Absorb(num_absorbs), SpongeOp::Squeeze(1u32)]);

    let pc = Sponge::<G::Scalar, U24>::api_constants(Strength::Standard);
    let mut ns = cs.namespace(|| "ns");

    let z_out = {
      let mut sponge = SpongeCircuit::new_with_constants(&pc, Simplex);
      let acc = &mut ns;

      sponge.start(parameter, None, acc);
      neptune::sponge::api::SpongeAPI::absorb(&mut sponge, num_absorbs, &elt, acc);

      let output = neptune::sponge::api::SpongeAPI::squeeze(&mut sponge, 1, acc);
      sponge.finish(acc).unwrap();
      Elt::ensure_allocated(&output[0], &mut ns.namespace(|| "ensure allocated"), true)?
    };

    Ok(vec![z_out])
  }
}

/// cargo run --release --example and
fn main() {
  println!("=========================================================");
  println!("Nova-based hashchain example");
  println!("=========================================================");

  let num_steps = 10;
  for num_elts_per_step in [1024, 2048, 4096] {
    // number of instances of AND per Nova's recursive step
    let circuit_primary = HashChainCircuit::new(num_elts_per_step);
    let circuit_secondary = TrivialCircuit::default();

    // produce public parameters
    let start = Instant::now();
    println!("Producing public parameters...");
    let pp = PublicParams::<
      E1,
      E2,
      HashChainCircuit<<E1 as Engine>::GE>,
      TrivialCircuit<<E2 as Engine>::Scalar>,
    >::setup(
      &circuit_primary,
      &circuit_secondary,
      &*S1::ck_floor(),
      &*S2::ck_floor(),
    )
    .unwrap();
    println!("PublicParams::setup, took {:?} ", start.elapsed());

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
    let circuits = (0..num_steps)
      .map(|_| HashChainCircuit::new(num_elts_per_step))
      .collect::<Vec<_>>();

    type C1 = HashChainCircuit<<E1 as Engine>::GE>;
    type C2 = TrivialCircuit<<E2 as Engine>::Scalar>;

    // produce a recursive SNARK
    println!(
      "Generating a RecursiveSNARK with {num_elts_per_step} field elements per hashchain node..."
    );
    let mut recursive_snark: RecursiveSNARK<E1, E2, C1, C2> =
      RecursiveSNARK::<E1, E2, C1, C2>::new(
        &pp,
        &circuits[0],
        &circuit_secondary,
        &[<E1 as Engine>::Scalar::zero()],
        &[<E2 as Engine>::Scalar::zero()],
      )
      .unwrap();

    for (i, circuit_primary) in circuits.iter().enumerate() {
      let start = Instant::now();
      let res = recursive_snark.prove_step(&pp, circuit_primary, &circuit_secondary);
      assert!(res.is_ok());

      println!("RecursiveSNARK::prove {} : took {:?} ", i, start.elapsed());
    }

    // verify the recursive SNARK
    println!("Verifying a RecursiveSNARK...");
    let res = recursive_snark.verify(
      &pp,
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    println!("RecursiveSNARK::verify: {:?}", res.is_ok(),);
    assert!(res.is_ok());

    // produce a compressed SNARK
    println!("Generating a CompressedSNARK using Spartan with HyperKZG...");
    let (pk, vk) = CompressedSNARK::<_, _, _, _, S1, S2>::setup(&pp).unwrap();

    let start = Instant::now();

    let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark);
    println!(
      "CompressedSNARK::prove: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());
    let compressed_snark = res.unwrap();

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &compressed_snark).unwrap();
    let compressed_snark_encoded = encoder.finish().unwrap();
    println!(
      "CompressedSNARK::len {:?} bytes",
      compressed_snark_encoded.len()
    );

    // verify the compressed SNARK
    println!("Verifying a CompressedSNARK...");
    let start = Instant::now();
    let res = compressed_snark.verify(
      &vk,
      num_steps,
      &[<E1 as Engine>::Scalar::ZERO],
      &[<E2 as Engine>::Scalar::ZERO],
    );
    println!(
      "CompressedSNARK::verify: {:?}, took {:?}",
      res.is_ok(),
      start.elapsed()
    );
    assert!(res.is_ok());
    println!("=========================================================");
  }
}
