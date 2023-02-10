//! This module implements Nova's traits using the following configuration:
//! `CommitmentEngine` with Pedersen's commitments
//! `Group` with pasta curves
//! `RO` traits with Poseidon
//! `EvaluationEngine` with an IPA-based polynomial evaluation argument

pub mod ipa_pc;
pub mod keccak;
pub mod pasta;
pub mod pedersen;
pub mod poseidon;
