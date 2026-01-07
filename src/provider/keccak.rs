//! This module provides an implementation of `TranscriptEngineTrait` using keccak256
use crate::{
  errors::NovaError,
  traits::{Engine, PrimeFieldExt, TranscriptEngineTrait, TranscriptReprTrait},
};
use core::marker::PhantomData;
use sha3::{Digest, Keccak256};

const PERSONA_TAG: &[u8] = b"NoTR";
const DOM_SEP_TAG: &[u8] = b"NoDS";
const KECCAK256_STATE_SIZE: usize = 64;
const KECCAK256_PREFIX_CHALLENGE_LO: u8 = 0;
const KECCAK256_PREFIX_CHALLENGE_HI: u8 = 1;

/// Provides an implementation of `TranscriptEngine`
#[derive(Debug, Clone)]
pub struct Keccak256Transcript<E: Engine> {
  round: u16,
  state: [u8; KECCAK256_STATE_SIZE],
  transcript: Keccak256,
  _p: PhantomData<E>,
}

fn compute_updated_state(keccak_instance: Keccak256, input: &[u8]) -> [u8; KECCAK256_STATE_SIZE] {
  let mut updated_instance = keccak_instance;
  updated_instance.update(input);

  let input_lo = &[KECCAK256_PREFIX_CHALLENGE_LO];
  let input_hi = &[KECCAK256_PREFIX_CHALLENGE_HI];

  let mut hasher_lo = updated_instance.clone();
  let mut hasher_hi = updated_instance;

  hasher_lo.update(input_lo);
  hasher_hi.update(input_hi);

  let output_lo = hasher_lo.finalize();
  let output_hi = hasher_hi.finalize();

  // EVM-compatible: big-endian order (hi, lo)
  [output_hi, output_lo]
    .concat()
    .as_slice()
    .try_into()
    .unwrap()
}

impl<E: Engine> TranscriptEngineTrait<E> for Keccak256Transcript<E> {
  fn new(label: &'static [u8]) -> Self {
    let keccak_instance = Keccak256::new();
    let input = [PERSONA_TAG, label].concat();
    let output = compute_updated_state(keccak_instance.clone(), &input);

    Self {
      round: 0u16,
      state: output,
      transcript: keccak_instance,
      _p: PhantomData,
    }
  }

  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, NovaError> {
    // we gather the full input from the round, preceded by the current state of the transcript
    // EVM-compatible: big-endian byte order for round
    let input = [
      DOM_SEP_TAG,
      self.round.to_be_bytes().as_ref(),
      self.state.as_ref(),
      label,
    ]
    .concat();
    let mut output = compute_updated_state(self.transcript.clone(), &input);

    // update state
    self.round = {
      if let Some(v) = self.round.checked_add(1) {
        v
      } else {
        return Err(NovaError::InternalTranscriptError);
      }
    };
    self.state.copy_from_slice(&output);
    self.transcript = Keccak256::new();

    // squeeze out a challenge (reverse for EVM compatibility)
    output.reverse();
    Ok(E::Scalar::from_uniform(&output))
  }

  fn absorb<T: TranscriptReprTrait<E::GE>>(&mut self, label: &'static [u8], o: &T) {
    self.transcript.update(label);
    self.transcript.update(o.to_transcript_bytes());
  }

  fn dom_sep(&mut self, bytes: &'static [u8]) {
    self.transcript.update(DOM_SEP_TAG);
    self.transcript.update(bytes);
  }
}

#[cfg(test)]
mod tests {
  use sha3::{Digest, Keccak256};

  #[test]
  fn test_keccak_example() {
    let mut hasher = Keccak256::new();
    hasher.update(0xffffffff_u32.to_le_bytes());
    let output: [u8; 32] = hasher.finalize().into();
    assert_eq!(
      hex::encode(output),
      "29045a592007d0c246ef02c2223570da9522d0cf0f73282c79a1bc8f0bb2c238"
    );
  }
}
