//! Benchmarks Nova's prover for proving SHA-256 with varying sized messages.
//! We run a single step with the step performing the entire computation.
//! This code invokes a hand-written SHA-256 gadget from bellman/bellperson.
//! It also uses code from bellman/bellperson to compare circuit-generated digest with sha2 crate's output
#![allow(non_snake_case)]
type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
use ::bellperson::{
  gadgets::{
    boolean::{AllocatedBit, Boolean},
    num::{AllocatedNum, Num},
    sha256::sha256,
    Assignment,
  },
  ConstraintSystem, SynthesisError,
};
use core::time::Duration;
use criterion::*;
use ff::{PrimeField, PrimeFieldBits};
use nova_snark::{
  traits::{
    circuit::{StepCircuit, TrivialTestCircuit},
    Group,
  },
  PublicParams, RecursiveSNARK,
};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
struct Sha256Circuit<Scalar: PrimeField> {
  preimage: Vec<u8>,
  digest: Scalar,
}

impl<Scalar: PrimeField + PrimeFieldBits> StepCircuit<Scalar> for Sha256Circuit<Scalar> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    _z: &[AllocatedNum<Scalar>],
  ) -> Result<Vec<AllocatedNum<Scalar>>, SynthesisError> {
    let mut z_out: Vec<AllocatedNum<Scalar>> = Vec::new();

    let bit_values: Vec<_> = self
      .preimage
      .clone()
      .into_iter()
      .flat_map(|byte| (0..8).map(move |i| (byte >> i) & 1u8 == 1u8))
      .map(Some)
      .collect();
    assert_eq!(bit_values.len(), self.preimage.len() * 8);

    let preimage_bits = bit_values
      .into_iter()
      .enumerate()
      .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("preimage bit {i}")), b))
      .map(|b| b.map(Boolean::from))
      .collect::<Result<Vec<_>, _>>()?;

    let hash_bits = sha256(cs.namespace(|| "sha256"), &preimage_bits)?;

    for (i, hash_bits) in hash_bits.chunks(256_usize).enumerate() {
      let mut num = Num::<Scalar>::zero();
      let mut coeff = Scalar::ONE;
      for bit in hash_bits {
        num = num.add_bool_with_coeff(CS::one(), bit, coeff);

        coeff = coeff.double();
      }

      let hash = AllocatedNum::alloc(cs.namespace(|| format!("input {i}")), || {
        Ok(*num.get_value().get()?)
      })?;

      // num * 1 = hash
      cs.enforce(
        || format!("packing constraint {i}"),
        |_| num.lc(Scalar::ONE),
        |lc| lc + CS::one(),
        |lc| lc + hash.get_variable(),
      );
      z_out.push(hash);
    }

    // sanity check with the hasher
    let mut hasher = Sha256::new();
    hasher.update(&self.preimage);
    let hash_result = hasher.finalize();

    let mut s = hash_result
      .iter()
      .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1u8 == 1u8));

    for b in hash_bits {
      match b {
        Boolean::Is(b) => {
          assert!(s.next().unwrap() == b.get_value().unwrap());
        }
        Boolean::Not(b) => {
          assert!(s.next().unwrap() != b.get_value().unwrap());
        }
        Boolean::Constant(_b) => {
          panic!("Can't reach here")
        }
      }
    }

    Ok(z_out)
  }

  fn output(&self, _z: &[Scalar]) -> Vec<Scalar> {
    vec![self.digest]
  }
}

type C1 = Sha256Circuit<<G1 as Group>::Scalar>;
type C2 = TrivialTestCircuit<<G2 as Group>::Scalar>;

criterion_group! {
name = recursive_snark;
config = Criterion::default().warm_up_time(Duration::from_millis(3000));
targets = bench_recursive_snark
}

criterion_main!(recursive_snark);

fn bench_recursive_snark(c: &mut Criterion) {
  let bytes_to_scalar = |bytes: [u8; 32]| -> <G1 as Group>::Scalar {
    let mut bytes_le = bytes;
    bytes_le.reverse();
    <G1 as Group>::Scalar::from_repr(bytes_le).unwrap()
  };

  let decode_hex = |s: &str| -> <G1 as Group>::Scalar {
    let bytes = (0..s.len())
      .step_by(2)
      .map(|i| u8::from_str_radix(&s[i..i + 2], 16))
      .collect::<Result<Vec<u8>, _>>()
      .unwrap();
    let bytes_arr: [u8; 32] = bytes.try_into().unwrap();
    bytes_to_scalar(bytes_arr)
  };

  // Test vectors
  let circuits = vec![
    Sha256Circuit {
      preimage: vec![0u8; 64],
      digest: decode_hex("12df9ae4958c1957170f9b04c4bc00c27315c5d75a391f4b672f952842bfa5ac"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 128],
      digest: decode_hex("13abfac9782cb9c13c4508bde596f1914fe2f744f6a661c0c9a16659745c4e1b"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 256],
      digest: decode_hex("0f5a007b5aef126a58f9bbd937842967c44253e7f97d98b5cd10bfe44d6782c8"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 512],
      digest: decode_hex("06a6cfaad91d49366f18443cd4e11576ff27c174bb9fe2bc54735a79e3e456e0"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 1024],
      digest: decode_hex("3763c73508f5fbb36daae8257d6c5c07db08ec5df0549ccf692b9fa218fd0ef7"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 2048],
      digest: decode_hex("35c18d6c3cf49e42b3ffcb54ea04bdc16617efba0e673abc8c858257955005a5"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 4096],
      digest: decode_hex("25349112d1bd5ba15e3e2d3effa01af1da02c097ce6208cdf28f34b74d35feb2"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 8192],
      digest: decode_hex("22bc891155c7d423039a2206ed4a5342755948baeb13a54b61dbead7c3d3b8f6"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 16384],
      digest: decode_hex("3fda713dc72ddcd42ce625c75f7e41d526d30647278a3dfcda95904e59ade7f1"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 32768],
      digest: decode_hex("1e2091bd3e3cedffebb7316b52414fff82511cbd232561874a4ae11ae2040ac1"),
    },
    Sha256Circuit {
      preimage: vec![0u8; 65536],
      digest: decode_hex("0c33953975c438ce357912f27b0fbcf98bae6eb68a1a913386672ee406a4f479"),
    },
  ];

  for circuit_primary in circuits {
    let mut group = c.benchmark_group(format!(
      "NovaProve-Sha256-message-len-{}",
      circuit_primary.preimage.len()
    ));
    group.sample_size(10);

    // Produce public parameters
    let pp =
      PublicParams::<G1, G2, C1, C2>::setup(circuit_primary.clone(), TrivialTestCircuit::default());

    let circuit_secondary = TrivialTestCircuit::default();
    let z0_primary = vec![<G1 as Group>::Scalar::from(2u64)];
    let z0_secondary = vec![<G2 as Group>::Scalar::from(2u64)];

    group.bench_function("Prove", |b| {
      b.iter(|| {
        let mut recursive_snark = RecursiveSNARK::new(
          black_box(&pp),
          black_box(&circuit_primary),
          black_box(&circuit_secondary),
          black_box(z0_primary.clone()),
          black_box(z0_secondary.clone()),
        );
        // produce a recursive SNARK for a step of the recursion
        assert!(recursive_snark
          .prove_step(
            black_box(&pp),
            black_box(&circuit_primary),
            black_box(&circuit_secondary),
            black_box(z0_primary.clone()),
            black_box(z0_secondary.clone()),
          )
          .is_ok());
      })
    });
    group.finish();
  }
}
