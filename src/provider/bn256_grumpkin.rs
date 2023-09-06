//! This module implements the Nova traits for `bn256::Point`, `bn256::Scalar`, `grumpkin::Point`, `grumpkin::Scalar`.
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
// Remove this when https://github.com/zcash/pasta_curves/issues/41 resolves
use pasta_curves::arithmetic::{CurveAffine, CurveExt};
use rayon::prelude::*;
use sha3::Shake256;
use std::io::Read;

use halo2curves::bn256::{
  G1Affine as Bn256Affine, G1Compressed as Bn256Compressed, G1 as Bn256Point,
};
use halo2curves::grumpkin::{
  G1Affine as GrumpkinAffine, G1Compressed as GrumpkinCompressed, G1 as GrumpkinPoint,
};

/// Re-exports that give access to the standard aliases used in the code base, for bn256
pub mod bn256 {
  pub use halo2curves::bn256::{
    Fq as Base, Fr as Scalar, G1Affine as Affine, G1Compressed as Compressed, G1 as Point,
  };
}

/// Re-exports that give access to the standard aliases used in the code base, for grumpkin
pub mod grumpkin {
  pub use halo2curves::grumpkin::{
    Fq as Base, Fr as Scalar, G1Affine as Affine, G1Compressed as Compressed, G1 as Point,
  };
}

impl<G: Group> TranscriptReprTrait<G> for grumpkin::Base {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().to_vec()
  }
}

impl<G: Group> TranscriptReprTrait<G> for grumpkin::Scalar {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.to_repr().to_vec()
  }
}

impl_traits!(
  bn256,
  Bn256Compressed,
  Bn256Point,
  Bn256Affine,
  "30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001"
);

impl_traits!(
  grumpkin,
  GrumpkinCompressed,
  GrumpkinPoint,
  GrumpkinAffine,
  "30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47"
);

#[cfg(test)]
mod tests {
  use super::*;
  type G = bn256::Point;

  fn from_label_serial(label: &'static [u8], n: usize) -> Vec<Bn256Affine> {
    let mut shake = Shake256::default();
    shake.update(label);
    let mut reader = shake.finalize_xof();
    let mut ck = Vec::new();
    for _ in 0..n {
      let mut uniform_bytes = [0u8; 32];
      reader.read_exact(&mut uniform_bytes).unwrap();
      let hash = bn256::Point::hash_to_curve("from_uniform_bytes");
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
