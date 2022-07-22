//! Demonstrates how to use Nova to produce a recursive proof of a chain function.
//! ,--->|ID |<--,      |ID |<--,      |ID |<--,      |ID |<--,      |ID |<--,      |ID |
//! |    |---|    \     |---|    \     |---|    \     |---|    \     |---|    \     |---|
//! `---<|PID|     '---<|PID|     '---<|PID|     '---<|PID|     '---<|PID|     '---<|PID|

use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use ff::PrimeField;
use generic_array::typenum::U2;
use neptune::{
  circuit::poseidon_hash,
  poseidon::{HashMode, Poseidon, PoseidonConstants},
  Strength,
};
use nova_snark::{
  gadgets::utils::alloc_num_equals,
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group,
  },
  CompressedSNARK, PublicParams, RecursiveSNARK,
};

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;
type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;

#[derive(Clone, Debug)]
struct Chain<F: PrimeField> {
  id: F,       // Data container ID
  pid: F,      // Parent data container ID
  id_succ: F,  // Successor data container ID
  pid_succ: F, // Successor parent data container ID
}

impl<F: PrimeField> Chain<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  fn new(num_links: usize) -> Vec<Self> {
    let message = "AbcPresKAnFEDIPjpfuFv7vkk66lXKhzYpZfBrAqj2M";
    let mut links = Vec::new();
    for i in 0..num_links {
      let id = format!("{}_{}", message, i);
      let pid = if i == 0 {
        id.clone()
      } else {
        format!("{}_{}", message, i - 1)
      };
      let id_succ = format!("{}_{}", message, i + 1);
      let pid_succ = id.clone();

      links.push(Self {
        id: Self::hash(id.as_bytes()),
        pid: Self::hash(pid.as_bytes()),
        id_succ: Self::hash(id_succ.as_bytes()),
        pid_succ: Self::hash(pid_succ.as_bytes()),
      });
    }

    links
  }

  fn hash(message: &[u8]) -> F {
    let message_len = message.len();
    assert!(message_len <= 64);
    let mut padded_message = message.to_vec();
    padded_message.extend(vec![0u8; (((message_len - 1) / 32) + 1) * 32 - message_len]);
    let s = padded_message
      .chunks(32)
      .map(|b| {
        F::from_repr({
          let mut bytes = [0u8; 32];
          bytes.copy_from_slice(b);
          bytes[bytes.len() - 1] &= 0b0011_1111;
          bytes
        })
        .unwrap()
      })
      .collect::<Vec<_>>();
    let constants: PoseidonConstants<F, U2> =
      PoseidonConstants::new_with_strength(Strength::Standard);
    let mut h: Poseidon<F> = Poseidon::new(&constants);
    h.reset();
    for scalar in s {
      h.input(scalar).unwrap();
    }

    h.hash_in_mode(HashMode::Correct)
  }
}

#[derive(Clone, Debug)]
struct ChainCircuit<F: PrimeField> {
  id: F,
  pid: F,
  id_succ: F,
  pid_succ: F,
  pc: PoseidonConstants<F, U2>,
}

impl<F> ChainCircuit<F>
where
  F: PrimeField<Repr = [u8; 32]>,
{
  pub fn new(num_links: usize, links: &[Chain<F>], pc: &PoseidonConstants<F, U2>) -> (F, Vec<Self>)
  where
    F: PrimeField<Repr = [u8; 32]>,
  {
    let mut z0 = F::zero();
    let mut circuits = Vec::new();
    for (i, link) in links.iter().enumerate().take(num_links) {
      let id = link.id;
      let pid = link.pid;

      let circuit = ChainCircuit {
        id,
        pid,
        id_succ: link.id_succ,
        pid_succ: link.pid_succ,
        pc: pc.clone(),
      };
      circuits.push(circuit);

      if i == 0 {
        z0 = Poseidon::<F, U2>::new_with_preimage(&[id, pid], pc).hash();
      }
    }

    (z0, circuits)
  }
}

impl<F> StepCircuit<F> for ChainCircuit<F>
where
  F: PrimeField,
{
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: AllocatedNum<F>,
  ) -> Result<AllocatedNum<F>, SynthesisError> {
    let id = AllocatedNum::alloc(cs.namespace(|| "id"), || Ok(self.id))?;
    let pid = AllocatedNum::alloc(cs.namespace(|| "pid"), || Ok(self.pid))?;
    let id_succ = AllocatedNum::alloc(cs.namespace(|| "id_succ"), || Ok(self.id_succ))?;
    let pid_succ = AllocatedNum::alloc(cs.namespace(|| "pid_succ"), || Ok(self.pid_succ))?;

    let z_hash = poseidon_hash(
      cs.namespace(|| "input hash"),
      vec![id.clone(), pid],
      &self.pc,
    )?;

    cs.enforce(
      || "z == z1",
      |lc| lc + z.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + z_hash.get_variable(),
    );

    let is_equal = alloc_num_equals(cs.namespace(|| "num_equal"), &id, &pid_succ)?;

    cs.enforce(
      || "is_equal == 1",
      |lc| lc + is_equal.get_variable(),
      |lc| lc + CS::one(),
      |lc| lc + CS::one(),
    );

    poseidon_hash(
      cs.namespace(|| "output hash"),
      vec![id_succ, pid_succ],
      &self.pc,
    )
  }

  fn compute(&self, z: &F) -> F {
    let z_hash = Poseidon::<F, U2>::new_with_preimage(&[self.id, self.pid], &self.pc).hash();
    debug_assert_eq!(z, &z_hash);

    Poseidon::<F, U2>::new_with_preimage(&[self.id_succ, self.pid_succ], &self.pc).hash()
  }
}

fn main() {
  // produce public parameters
  println!("Generating public parameters...");

  let pc = PoseidonConstants::<<G1 as Group>::Scalar, U2>::new_with_strength(Strength::Standard);
  let circuit_primary = ChainCircuit::<<G1 as Group>::Scalar> {
    id: <G1 as Group>::Scalar::zero(),
    pid: <G1 as Group>::Scalar::zero(),
    id_succ: <G1 as Group>::Scalar::zero(),
    pid_succ: <G1 as Group>::Scalar::zero(),
    pc: pc.clone(),
  };

  let circuit_secondary = TrivialTestCircuit::default();

  let pp = PublicParams::<
    G1,
    G2,
    ChainCircuit<<G1 as Group>::Scalar>,
    TrivialTestCircuit<<G2 as Group>::Scalar>,
  >::setup(circuit_primary, circuit_secondary.clone());

  // produce non-deterministic advice
  println!("Generating non-deterministic advice...");

  let num_links = 3;

  let links = Chain::<<G1 as Group>::Scalar>::new(num_links);

  let (z0_primary, circuits_primary) =
    ChainCircuit::<<G1 as Group>::Scalar>::new(num_links, &links, &pc);

  // Secondary circuit
  let z0_secondary = <G2 as Group>::Scalar::zero();

  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");

  type C1 = ChainCircuit<<G1 as Group>::Scalar>;
  type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;

  let mut recursive_snark: Option<RecursiveSNARK<G1, G2, C1, C2>> = None;

  for (i, circuit_primary) in circuits_primary.iter().take(num_links).enumerate() {
    let result = RecursiveSNARK::prove_step(
      &pp,
      recursive_snark,
      circuit_primary.clone(),
      circuit_secondary.clone(),
      z0_primary,
      z0_secondary,
    );
    assert!(result.is_ok());
    println!("RecursiveSNARK::prove_step {}: {:?}", i, result.is_ok());
    recursive_snark = Some(result.unwrap());
  }

  assert!(recursive_snark.is_some());
  let recursive_snark = recursive_snark.unwrap();

  // verify the recursive SNARK
  println!("Verifying the RecursiveSNARK...");
  let res = recursive_snark.verify(&pp, num_links, z0_primary, z0_secondary);
  println!("RecursiveSNARK::verify: {:?}", res.is_ok());
  assert!(res.is_ok());

  // produce a compressed SNARK
  println!("Generating a CompressedSNARK...");
  let res = CompressedSNARK::<_, _, _, _, S1, S2>::prove(&pp, &recursive_snark);
  println!("CompressedSNARK::prove: {:?}", res.is_ok());
  assert!(res.is_ok());
  let compressed_snark = res.unwrap();

  // verify the compressed SNARK
  println!("Verifying a CompressedSNARK...");
  let res = compressed_snark.verify(&pp, num_links, z0_primary, z0_secondary);
  println!("CompressedSNARK::verify: {:?}", res.is_ok());
  assert!(res.is_ok());
}
