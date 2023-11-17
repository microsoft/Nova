//! This module implements Nova's traits using the following several different combinations

// public modules to be used as an evaluation engine with Spartan
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
mod msm;

use crate::{
  provider::{
    bn256_grumpkin::{bn256, grumpkin},
    keccak::Keccak256Transcript,
    pedersen::CommitmentEngine as PedersenCommitmentEngine,
    poseidon::{PoseidonRO, PoseidonROCircuit},
    secp_secq::{secp256k1, secq256k1},
  },
  traits::Engine,
};
use pasta_curves::{pallas, vesta};

/// An implementation of the Nova `Engine` trait with BN254 curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bn256Engine;

/// An implementation of the Nova `Engine` trait with Grumpkin curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GrumpkinEngine;

impl Engine for Bn256Engine {
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Secp256k1Engine;

/// An implementation of the Nova `Engine` trait with Secp256k1 curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PallasEngine;

/// An implementation of the Nova `Engine` trait with Vesta curve and Pedersen commitment scheme
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
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
