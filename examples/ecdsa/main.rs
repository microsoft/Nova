//! Demonstrates how to use Nova to produce a recursive proof of an ECDSA signature.
//! This example proves the knowledge of a sequence of ECDSA signatures with different public keys on different messages,
//! but the example can be adapted to other settings (e.g., proving the validity of the certificate chain with a well-known root public key)
//! Scheme borrowed from https://github.com/filecoin-project/bellperson-gadgets/blob/main/src/eddsa.rs
//! Sign using G1 curve, and prove using G2 curve.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg};
use ff::{
  derive::byteorder::{ByteOrder, LittleEndian},
  Field, PrimeField, PrimeFieldBits,
};
use nova_snark::{
  traits::{circuit::TrivialTestCircuit, Group as Nova_Group},
  CompressedSNARK, PublicParams, RecursiveSNARK,
};
use num_bigint::BigUint;
use pasta_curves::{
  arithmetic::CurveAffine,
  group::{Curve, Group},
};
use rand::{rngs::OsRng, RngCore};
use sha3::{Digest, Sha3_512};
use subtle::Choice;

mod circuit;
mod utils;

use crate::circuit::{Coordinate, EcdsaCircuit, EcdsaSignature};
use crate::utils::BitIterator;

type G1 = pasta_curves::pallas::Point;
type G2 = pasta_curves::vesta::Point;
type S1 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G2>;
type S2 = nova_snark::spartan_with_ipa_pc::RelaxedR1CSSNARK<G1>;

#[derive(Debug, Clone, Copy)]
pub struct SecretKey(pub <G1 as Group>::Scalar);

impl SecretKey {
  pub fn random(mut rng: impl RngCore) -> Self {
    let secret = <G1 as Group>::Scalar::random(&mut rng);
    Self(secret)
  }
}

#[derive(Debug, Clone, Copy)]
pub struct PublicKey(pub G1);

impl PublicKey {
  pub fn from_secret_key(s: &SecretKey) -> Self {
    let point = G1::generator() * s.0;
    Self(point)
  }
}

#[derive(Clone)]
pub struct Signature {
  pub r: G1,
  pub s: <G1 as Group>::Scalar,
}

impl SecretKey {
  pub fn sign(self, c: <G1 as Group>::Scalar, mut rng: impl RngCore) -> Signature {
    // T
    let mut t = [0u8; 80];
    rng.fill_bytes(&mut t[..]);

    // h = H*(T || M)
    let h = Self::hash_to_scalar(b"Nova_Ecdsa_Hash", &t[..], &c.to_repr());

    // R = [h]G
    let r = G1::generator().mul(h);

    // s = h + c * sk
    let mut s = c;

    s.mul_assign(&self.0);
    s.add_assign(&h);

    Signature { r, s }
  }

  fn mul_bits<B: AsRef<[u64]>>(
    s: &<G1 as Group>::Scalar,
    bits: BitIterator<B>,
  ) -> <G1 as Group>::Scalar {
    let mut x = <G1 as Group>::Scalar::zero();
    for bit in bits {
      x.double();

      if bit {
        x.add_assign(s)
      }
    }
    x
  }

  fn to_uniform(digest: &[u8]) -> <G1 as Group>::Scalar {
    assert_eq!(digest.len(), 64);
    let mut bits: [u64; 8] = [0; 8];
    LittleEndian::read_u64_into(digest, &mut bits);
    Self::mul_bits(&<G1 as Group>::Scalar::one(), BitIterator::new(bits))
  }

  pub fn to_uniform_32(digest: &[u8]) -> <G1 as Group>::Scalar {
    assert_eq!(digest.len(), 32);
    let mut bits: [u64; 4] = [0; 4];
    LittleEndian::read_u64_into(digest, &mut bits);
    Self::mul_bits(&<G1 as Group>::Scalar::one(), BitIterator::new(bits))
  }

  pub fn hash_to_scalar(persona: &[u8], a: &[u8], b: &[u8]) -> <G1 as Group>::Scalar {
    let mut hasher = Sha3_512::new();
    hasher.input(persona);
    hasher.input(a);
    hasher.input(b);
    let digest = hasher.result();
    Self::to_uniform(digest.as_ref())
  }
}

impl PublicKey {
  pub fn verify(&self, c: <G1 as Group>::Scalar, signature: &Signature) -> bool {
    let modulus = Self::modulus_as_scalar();
    let order_check_pk = self.0.mul(modulus);
    if !order_check_pk.eq(&G1::identity()) {
      return false;
    }

    let order_check_r = signature.r.mul(modulus);
    if !order_check_r.eq(&G1::identity()) {
      return false;
    }

    // 0 = [-s]G + R + [c]PK
    self
      .0
      .mul(c)
      .add(&signature.r)
      .add(G1::generator().mul(signature.s).neg())
      .eq(&G1::identity())
  }

  fn modulus_as_scalar() -> <G1 as Group>::Scalar {
    let mut bits = <G1 as Group>::Scalar::char_le_bits().to_bitvec();
    let mut acc = BigUint::new(Vec::<u32>::new());
    while let Some(b) = bits.pop() {
      acc <<= 1_i32;
      acc += b as u8;
    }
    let modulus = acc.to_str_radix(10);
    <G1 as Group>::Scalar::from_str_vartime(&modulus).unwrap()
  }
}

fn main() {
  // In a VERY LIMITED case of messages known to be unique due to application level
  // and being less than the group order when interpreted as integer, one can sign
  // the message directly without hashing
  pub const MAX_MESSAGE_LEN: usize = 16;
  assert!(MAX_MESSAGE_LEN * 8 <= <G1 as Group>::Scalar::CAPACITY as usize);

  // produce public parameters
  println!("Generating public parameters...");

  let circuit_primary = EcdsaCircuit::<<G2 as Nova_Group>::Scalar> {
    r: Coordinate::new(
      <G2 as Nova_Group>::Scalar::zero(),
      <G2 as Nova_Group>::Scalar::zero(),
    ),
    g: Coordinate::new(
      <G2 as Nova_Group>::Scalar::zero(),
      <G2 as Nova_Group>::Scalar::zero(),
    ),
    pk: Coordinate::new(
      <G2 as Nova_Group>::Scalar::zero(),
      <G2 as Nova_Group>::Scalar::zero(),
    ),
    c: <G2 as Nova_Group>::Scalar::zero(),
    s: <G2 as Nova_Group>::Scalar::zero(),
    c_bits: vec![Choice::from(0u8); 256],
    s_bits: vec![Choice::from(0u8); 256],
  };

  let circuit_secondary = TrivialTestCircuit::default();

  let pp = PublicParams::<
    G2,
    G1,
    EcdsaCircuit<<G2 as Group>::Scalar>,
    TrivialTestCircuit<<G1 as Group>::Scalar>,
  >::setup(circuit_primary, circuit_secondary.clone());

  // produce non-deterministic advice
  println!("Generating non-deterministic advice...");

  let num_steps = 3;

  let signatures = || {
    let mut signatures = Vec::new();
    for i in 0..num_steps {
      let sk = SecretKey::random(&mut OsRng);
      let pk = PublicKey::from_secret_key(&sk);

      let message = format!("MESSAGE{}", i).as_bytes().to_owned();
      assert!(message.len() <= MAX_MESSAGE_LEN);

      let mut digest: Vec<u8> = message.to_vec();
      for _ in 0..(32 - message.len() as u32) {
        digest.extend(&[0u8; 1]);
      }

      let c = SecretKey::to_uniform_32(digest.as_ref());

      let signature_primary = sk.sign(c, &mut OsRng);
      let result = pk.verify(c, &signature_primary);
      assert!(result);

      // Affine coordinates guaranteed to be on the curve
      let rxy = signature_primary.r.to_affine().coordinates().unwrap();
      let gxy = G1::generator().to_affine().coordinates().unwrap();
      let pkxy = pk.0.to_affine().coordinates().unwrap();

      let s = signature_primary.s;

      signatures.push(EcdsaSignature::<
        <G1 as Nova_Group>::Base,
        <G1 as Nova_Group>::Scalar,
      >::new(
        Coordinate::<<G1 as Nova_Group>::Base>::new(*pkxy.x(), *pkxy.y()),
        Coordinate::<<G1 as Nova_Group>::Base>::new(*rxy.x(), *rxy.y()),
        s,
        c,
        Coordinate::<<G1 as Nova_Group>::Base>::new(*gxy.x(), *gxy.y()),
      ));
    }
    signatures
  };

  let (z0_primary, circuits_primary) = EcdsaCircuit::<<G2 as Nova_Group>::Scalar>::new::<
    <G1 as Nova_Group>::Base,
    <G1 as Nova_Group>::Scalar,
  >(num_steps, &signatures());

  // Secondary circuit
  let z0_secondary = vec![<G1 as Group>::Scalar::zero()];

  // produce a recursive SNARK
  println!("Generating a RecursiveSNARK...");

  type C1 = EcdsaCircuit<<G2 as Nova_Group>::Scalar>;
  type C2 = TrivialTestCircuit<<G1 as Nova_Group>::Scalar>;

  let mut recursive_snark: Option<RecursiveSNARK<G2, G1, C1, C2>> = None;

  for (i, circuit_primary) in circuits_primary.iter().take(num_steps).enumerate() {
    let result = RecursiveSNARK::prove_step(
      &pp,
      recursive_snark,
      circuit_primary.clone(),
      circuit_secondary.clone(),
      z0_primary.clone(),
      z0_secondary.clone(),
    );
    assert!(result.is_ok());
    println!("RecursiveSNARK::prove_step {}: {:?}", i, result.is_ok());
    recursive_snark = Some(result.unwrap());
  }

  assert!(recursive_snark.is_some());
  let recursive_snark = recursive_snark.unwrap();

  // verify the recursive SNARK
  println!("Verifying the RecursiveSNARK...");
  let res = recursive_snark.verify(&pp, num_steps, z0_primary.clone(), z0_secondary.clone());
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
  let res = compressed_snark.verify(&pp, num_steps, z0_primary, z0_secondary);
  println!("CompressedSNARK::verify: {:?}", res.is_ok());
  assert!(res.is_ok());
}
