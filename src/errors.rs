//! This module defines errors returned by the library.
use core::fmt::Debug;
use thiserror::Error;

/// Errors returned by Nova
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum NovaError {
  /// returned if the supplied row or col in (row,col,val) tuple is out of range
  #[error("InvalidIndex")]
  InvalidIndex,
  /// returned if the step circuit calls inputize or alloc_io in its synthesize method
  /// instead of passing output with the return value
  #[error("InvalidStepCircuitIO")]
  InvalidStepCircuitIO,
  /// returned if the supplied input is not of the right length
  #[error("InvalidInputLength")]
  InvalidInputLength,
  /// returned if the supplied witness is not of the right length
  #[error("InvalidWitnessLength")]
  InvalidWitnessLength,
  /// returned if the supplied witness is not a satisfying witness to a given shape and instance
  #[error("UnSat")]
  UnSat,
  /// returned when the supplied compressed commitment cannot be decompressed
  #[error("DecompressionError")]
  DecompressionError,
  /// returned if proof verification fails
  #[error("ProofVerifyError")]
  ProofVerifyError,
  /// returned if the provided commitment key is not of sufficient length
  #[error("InvalidCommitmentKeyLength")]
  InvalidCommitmentKeyLength,
  /// returned if the provided number of steps is zero
  #[error("InvalidNumSteps")]
  InvalidNumSteps,
  /// returned when an invalid PCS evaluation argument is provided
  #[error("InvalidPCS")]
  InvalidPCS,
  /// returned when an invalid sum-check proof is provided
  #[error("InvalidSumcheckProof")]
  InvalidSumcheckProof,
  /// returned when the initial input to an incremental computation differs from a previously declared arity
  #[error("InvalidInitialInputLength")]
  InvalidInitialInputLength,
  /// returned when the step execution produces an output whose length differs from a previously declared arity
  #[error("InvalidStepOutputLength")]
  InvalidStepOutputLength,
  /// returned when the transcript engine encounters an overflow of the round number
  #[error("InternalTranscriptError")]
  InternalTranscriptError,
  /// returned when the multiset check fails
  #[error("InvalidMultisetProof")]
  InvalidMultisetProof,
  /// returned when the product proof check fails
  #[error("InvalidProductProof")]
  InvalidProductProof,
  /// returned when the consistency with public IO and assignment used fails
  #[error("IncorrectWitness")]
  IncorrectWitness,
  /// return when error during synthesis
  #[error("SynthesisError: {reason}")]
  SynthesisError {
    /// The reason for circuit synthesis failure
    reason: String,
  },
  /// returned when there is an error creating a digest
  #[error("DigestError")]
  DigestError,
  /// returned when the prover cannot prove the provided statement due to completeness error
  #[error("InternalError")]
  InternalError,
}

impl From<bellpepper_core::SynthesisError> for NovaError {
  fn from(err: bellpepper_core::SynthesisError) -> Self {
    Self::SynthesisError {
      reason: err.to_string(),
    }
  }
}
