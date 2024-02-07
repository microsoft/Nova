//! This module implements Nova's traits using the following several different combinations

// public modules to be used as an evaluation engine with Spartan
pub mod hyperkzg;
pub mod ipa_pc;

// crate-public modules, made crate-public mostly for tests
pub(crate) mod bn256_grumpkin;
pub(crate) mod pasta;
pub(crate) mod pedersen;
pub(crate) mod poseidon;
pub(crate) mod secp_secq;
pub(crate) mod traits;

// crate-private modules
mod keccak;

use crate::{
  provider::{
    bn256_grumpkin::{bn256, grumpkin},
    hyperkzg::CommitmentEngine as HyperKZGCommitmentEngine,
    keccak::Keccak256Transcript,
    pedersen::CommitmentEngine as PedersenCommitmentEngine,
    poseidon::{PoseidonRO, PoseidonROCircuit},
    secp_secq::{secp256k1, secq256k1},
  },
  traits::Engine,
};
use pasta_curves::{pallas, vesta};
use serde::{Deserialize, Serialize};

/// An implementation of Nova traits with HyperKZG over the BN256 curve
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Bn256EngineKZG;

/// An implementation of the Nova `Engine` trait with Grumpkin curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct GrumpkinEngine;

/// An implementation of the Nova `Engine` trait with BN254 curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Bn256EngineIPA;

impl Engine for Bn256EngineKZG {
  type Base = bn256::Base;
  type Scalar = bn256::Scalar;
  type GE = bn256::Point;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = HyperKZGCommitmentEngine<Self>;
}

impl Engine for Bn256EngineIPA {
  type Base = bn256::Base;
  type Scalar = bn256::Scalar;
  type GE = bn256::Point;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
}

impl Engine for GrumpkinEngine {
  type Base = grumpkin::Base;
  type Scalar = grumpkin::Scalar;
  type GE = grumpkin::Point;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
}

/// An implementation of the Nova `Engine` trait with Secp256k1 curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Secp256k1Engine;

/// An implementation of the Nova `Engine` trait with Secp256k1 curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Secq256k1Engine;

impl Engine for Secp256k1Engine {
  type Base = secp256k1::Base;
  type Scalar = secp256k1::Scalar;
  type GE = secp256k1::Point;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
}

impl Engine for Secq256k1Engine {
  type Base = secq256k1::Base;
  type Scalar = secq256k1::Scalar;
  type GE = secq256k1::Point;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
}

/// An implementation of the Nova `Engine` trait with Pallas curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PallasEngine;

/// An implementation of the Nova `Engine` trait with Vesta curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct VestaEngine;

impl Engine for PallasEngine {
  type Base = pallas::Base;
  type Scalar = pallas::Scalar;
  type GE = pallas::Point;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
}

impl Engine for VestaEngine {
  type Base = vesta::Base;
  type Scalar = vesta::Scalar;
  type GE = vesta::Point;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = PedersenCommitmentEngine<Self>;
}

#[cfg(test)]
mod tests {
  use crate::provider::{bn256_grumpkin::bn256, secp_secq::secp256k1, traits::DlogGroup};
  use digest::{ExtendableOutput, Update};
  use group::Curve;
  use halo2curves::CurveExt;
  use pasta_curves::pallas;
  use sha3::Shake256;
  use std::io::Read;

  macro_rules! impl_cycle_pair_test {
    ($curve:ident) => {
      fn from_label_serial(label: &'static [u8], n: usize) -> Vec<$curve::Affine> {
        let mut shake = Shake256::default();
        shake.update(label);
        let mut reader = shake.finalize_xof();
        (0..n)
          .map(|_| {
            let mut uniform_bytes = [0u8; 32];
            reader.read_exact(&mut uniform_bytes).unwrap();
            let hash = $curve::Point::hash_to_curve("from_uniform_bytes");
            hash(&uniform_bytes).to_affine()
          })
          .collect()
      }

      let label = b"test_from_label";
      for n in [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 1021,
      ] {
        let ck_par = <$curve::Point as DlogGroup>::from_label(label, n);
        let ck_ser = from_label_serial(label, n);
        assert_eq!(ck_par.len(), n);
        assert_eq!(ck_ser.len(), n);
        assert_eq!(ck_par, ck_ser);
      }
    };
  }

  #[test]
  fn test_bn256_from_label() {
    impl_cycle_pair_test!(bn256);
  }

  #[test]
  fn test_pallas_from_label() {
    impl_cycle_pair_test!(pallas);
  }

  #[test]
  fn test_secp256k1_from_label() {
    impl_cycle_pair_test!(secp256k1);
  }
}
