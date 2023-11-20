//! This module implements the Nova traits for `secp::Point`, `secp::Scalar`, `secq::Point`, `secq::Scalar`.
use crate::{
  impl_engine, impl_traits,
  provider::{
    cpu_best_multiexp,
    keccak::Keccak256Transcript,
    pedersen::CommitmentEngine,
    poseidon::{PoseidonRO, PoseidonROCircuit},
    CompressedGroup, DlogGroup,
  },
  traits::{Engine, Group, PrimeFieldExt, TranscriptReprTrait},
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

/// An implementation of the Nova `Engine` trait with Secp256k1 curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Secp256k1Engine;

/// An implementation of the Nova `Engine` trait with Secp256k1 curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Secq256k1Engine;

impl_traits!(
  Secp256k1Engine,
  secp256k1,
  Secp256k1Compressed,
  Secp256k1,
  Secp256k1Affine,
  "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141",
  "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f"
);

impl_traits!(
  Secq256k1Engine,
  secq256k1,
  Secq256k1Compressed,
  Secq256k1,
  Secq256k1Affine,
  "fffffffffffffffffffffffffffffffffffffffffffffffffffffffefffffc2f",
  "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141"
);
