//! This module provides an implementation of TranscriptEngineTrait using keccak256
use crate::traits::PrimeFieldExt;
use crate::{
  errors::NovaError,
  traits::{Group, TranscriptEngineTrait, TranscriptReprTrait},
};
use core::marker::PhantomData;
use sha3::{Digest, Keccak256};

const PERSONA_TAG: &[u8] = b"NoTR";
const DOM_SEP_TAG: &[u8] = b"NoDS";
const KECCAK256_STATE_SIZE: usize = 64;
const KECCAK256_PREFIX_CHALLENGE_LO: u8 = 0;
const KECCAK256_PREFIX_CHALLENGE_HI: u8 = 1;

/// Provides an implementation of TranscriptEngine
#[derive(Debug, Clone)]
pub struct Keccak256Transcript<G: Group> {
  round: u16,
  state: [u8; KECCAK256_STATE_SIZE],
  transcript: Vec<u8>,
  _p: PhantomData<G>,
}

fn compute_updated_state(input: &[u8]) -> [u8; KECCAK256_STATE_SIZE] {
  let input_lo = [input, &[KECCAK256_PREFIX_CHALLENGE_LO]].concat();
  let input_hi = [input, &[KECCAK256_PREFIX_CHALLENGE_HI]].concat();

  let mut hasher_lo = Keccak256::new();
  let mut hasher_hi = Keccak256::new();

  hasher_lo.input(&input_lo);
  hasher_hi.input(&input_hi);

  let output_lo = hasher_lo.result();
  let output_hi = hasher_hi.result();

  [output_lo, output_hi]
    .concat()
    .as_slice()
    .try_into()
    .unwrap()
}

impl<G: Group> TranscriptEngineTrait<G> for Keccak256Transcript<G> {
  fn new(label: &'static [u8]) -> Self {
    let input = [PERSONA_TAG, label].concat();
    let output = compute_updated_state(&input);

    Self {
      round: 0u16,
      state: output,
      transcript: vec![],
      _p: Default::default(),
    }
  }

  fn squeeze(&mut self, label: &'static [u8]) -> Result<G::Scalar, NovaError> {
    let input = [
      DOM_SEP_TAG,
      self.round.to_le_bytes().as_ref(),
      self.state.as_ref(),
      self.transcript.as_ref(),
      label,
    ]
    .concat();
    let output = compute_updated_state(&input);

    // update state
    self.round = {
      if let Some(v) = self.round.checked_add(1) {
        v
      } else {
        return Err(NovaError::InternalTranscriptError);
      }
    };
    self.state.copy_from_slice(&output);
    self.transcript = vec![];

    // squeeze out a challenge
    Ok(G::Scalar::from_uniform(&output))
  }

  fn absorb<T: TranscriptReprTrait<G>>(&mut self, label: &'static [u8], o: &T) {
    self.transcript.extend_from_slice(label);
    self.transcript.extend_from_slice(&o.to_transcript_bytes());
  }

  fn dom_sep(&mut self, bytes: &'static [u8]) {
    self.transcript.extend_from_slice(DOM_SEP_TAG);
    self.transcript.extend_from_slice(bytes);
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    provider::keccak::Keccak256Transcript,
    traits::{Group, TranscriptEngineTrait},
  };
  use ff::PrimeField;
  use sha3::{Digest, Keccak256};

  type G = pasta_curves::pallas::Point;

  #[test]
  fn test_keccak_transcript() {
    let mut transcript: Keccak256Transcript<G> = Keccak256Transcript::new(b"test");

    // two scalars
    let s1 = <G as Group>::Scalar::from(2u64);
    let s2 = <G as Group>::Scalar::from(5u64);

    // add the scalars to the transcript
    transcript.absorb(b"s1", &s1);
    transcript.absorb(b"s2", &s2);

    // make a challenge
    let c1: <G as Group>::Scalar = transcript.squeeze(b"c1").unwrap();
    assert_eq!(
      hex::encode(c1.to_repr().as_ref()),
      "432d5811c8be3d44d47f52108a8749ae18482efd1a37b830f966456b5d75340c"
    );

    // a scalar
    let s3 = <G as Group>::Scalar::from(128u64);

    // add the scalar to the transcript
    transcript.absorb(b"s3", &s3);

    // make a challenge
    let c2: <G as Group>::Scalar = transcript.squeeze(b"c2").unwrap();
    assert_eq!(
      hex::encode(c2.to_repr().as_ref()),
      "65f7908d53abcd18f3b1d767456ef9009b91c7566a635e9ca7be26e21d4d7a10"
    );
  }

  #[test]
  fn test_keccak_example() {
    let mut hasher = Keccak256::new();
    hasher.input(0xffffffff_u32.to_le_bytes());
    let output: [u8; 32] = hasher.result().try_into().unwrap();
    assert_eq!(
      hex::encode(output),
      "29045a592007d0c246ef02c2223570da9522d0cf0f73282c79a1bc8f0bb2c238"
    );
  }
}
