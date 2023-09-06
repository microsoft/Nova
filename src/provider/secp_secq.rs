//! This module implements the Nova traits for secp::Point, secp::Scalar, secq::Point, secq::Scalar.
use crate::{
  impl_traits,
  provider::{
    cpu_best_multiexp,
    keccak::Keccak256Transcript,
    pedersen::CommitmentEngine,
    poseidon::{PoseidonRO, PoseidonROCircuit},
  },
  traits::{CompressedGroup, Group, PrimeFieldExt, TranscriptReprTrait},
};
use digest::{ExtendableOutput, Update};
use ff::{FromUniformBytes, PrimeField};
use group::{cofactor::CofactorCurveAffine, Curve, Group as AnotherGroup, GroupEncoding};
use num_bigint::BigInt;
use num_traits::Num;
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

use halo2curves::secp256k1::{Secp256k1, Secp256k1Affine, Secp256k1Compressed};
use halo2curves::secq256k1::{Secq256k1, Secq256k1Affine, Secq256k1Compressed};

/// Re-exports that give access to the standard aliases used in the code base, for secp
pub mod secp256k1 {
  pub use halo2curves::secp256k1::{
    Fp as Base, Fq as Scalar, Secp256k1 as Point, Secp256k1Affine as Affine,
    Secp256k1Compressed as Compressed,
  };
}

/// Re-exports that give access to the standard aliases used in the code base, for secq
pub mod secq256k1 {
  pub use halo2curves::secq256k1::{
    Fp as Base, Fq as Scalar, Secq256k1 as Point, Secq256k1Affine as Affine,
    Secq256k1Compressed as Compressed,
  };
}

impl<G: Group> TranscriptReprTrait<G> for secp256k1::Base {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().to_vec()
  }
}

impl<G: Group> TranscriptReprTrait<G> for secp256k1::Scalar {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().to_vec()
  }
}

impl_traits!(
  secp256k1,
  Secp256k1Compressed,
  Secp256k1,
  Secp256k1Affine,
  "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
);

impl_traits!(
  secq256k1,
  Secq256k1Compressed,
  Secq256k1,
  Secq256k1Affine,
  "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
);

#[cfg(test)]
mod tests {
  use super::*;
  type G = secp256k1::Point;

  fn from_label_serial(label: &'static [u8], n: usize) -> Vec<Secp256k1Affine> {
    let mut shake = Shake256::default();
    shake.update(label);
    let mut reader = shake.finalize_xof();
    let mut ck = Vec::new();
    for _ in 0..n {
      let mut uniform_bytes = [0u8; 32];
      reader.read_exact(&mut uniform_bytes).unwrap();
      let hash = secp256k1::Point::hash_to_curve("from_uniform_bytes");
      ck.push(hash(&uniform_bytes).to_affine());
    }
    ck
  }

  #[test]
  fn test_from_label() {
    let label = b"test_from_label";
    for n in [
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1021,
    ] {
      let ck_par = <G as Group>::from_label(label, n);
      let ck_ser = from_label_serial(label, n);
      assert_eq!(ck_par.len(), n);
      assert_eq!(ck_ser.len(), n);
      assert_eq!(ck_par, ck_ser);
    }
  }
}
