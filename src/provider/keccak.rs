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
  transcript: Keccak256,
  _p: PhantomData<G>,
}

fn compute_updated_state(keccak_instance: Keccak256, input: &[u8]) -> [u8; KECCAK256_STATE_SIZE] {
  let input_lo = [input, &[KECCAK256_PREFIX_CHALLENGE_LO]].concat();
  let input_hi = [input, &[KECCAK256_PREFIX_CHALLENGE_HI]].concat();

  let mut hasher_lo = keccak_instance.clone();
  let mut hasher_hi = keccak_instance;

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
    let keccak_instance = Keccak256::new();
    let input = [PERSONA_TAG, label].concat();
    let output = compute_updated_state(keccak_instance.clone(), &input);

    Self {
      round: 0u16,
      state: output,
      transcript: keccak_instance,
      _p: Default::default(),
    }
  }

  fn squeeze(&mut self, label: &'static [u8]) -> Result<G::Scalar, NovaError> {
    // we gather the full input from the round, preceded by the current state of the transcript
    let input = [
      DOM_SEP_TAG,
      self.round.to_le_bytes().as_ref(),
      self.state.as_ref(),
      label,
    ]
    .concat();
    let output = compute_updated_state(self.transcript.clone(), &input);

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

    // squeeze out a challenge
    Ok(G::Scalar::from_uniform(&output))
  }

  fn absorb<T: TranscriptReprTrait<G>>(&mut self, label: &'static [u8], o: &T) {
    self.transcript.input(label);
    self.transcript.input(&o.to_transcript_bytes());
  }

  fn dom_sep(&mut self, bytes: &'static [u8]) {
    self.transcript.input(DOM_SEP_TAG);
    self.transcript.input(bytes);
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    provider::bn256_grumpkin::bn256,
    provider::keccak::Keccak256Transcript,
    traits::{Group, TranscriptEngineTrait},
  };
  use ff::PrimeField;
  use sha3::{Digest, Keccak256};

  fn test_keccak_transcript_with<G: Group>(expected_h1: &'static str, expected_h2: &'static str) {
    let mut transcript: Keccak256Transcript<G> = Keccak256Transcript::new(b"test");

    // two scalars
    let s1 = <G as Group>::Scalar::from(2u64);
    let s2 = <G as Group>::Scalar::from(5u64);

    // add the scalars to the transcript
    transcript.absorb(b"s1", &s1);
    transcript.absorb(b"s2", &s2);

    // make a challenge
    let c1: <G as Group>::Scalar = transcript.squeeze(b"c1").unwrap();
    assert_eq!(hex::encode(c1.to_repr().as_ref()), expected_h1);

    // a scalar
    let s3 = <G as Group>::Scalar::from(128u64);

    // add the scalar to the transcript
    transcript.absorb(b"s3", &s3);

    // make a challenge
    let c2: <G as Group>::Scalar = transcript.squeeze(b"c2").unwrap();
    assert_eq!(hex::encode(c2.to_repr().as_ref()), expected_h2);
  }

  #[test]
  fn test_keccak_transcript() {
    test_keccak_transcript_with::<pasta_curves::pallas::Point>(
      "5ddffa8dc091862132788b8976af88b9a2c70594727e611c7217ba4c30c8c70a",
      "4d4bf42c065870395749fa1c4fb641df1e0d53f05309b03d5b1db7f0be3aa13d",
    );

    test_keccak_transcript_with::<bn256::Point>(
      "9fb71e3b74bfd0b60d97349849b895595779a240b92a6fae86bd2812692b6b0e",
      "bfd4c50b7d6317e9267d5d65c985eb455a3561129c0b3beef79bfc8461a84f18",
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
