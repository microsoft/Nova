//! This module provides an implementation of `TranscriptEngineTrait` using keccak256
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

/// Provides an implementation of `TranscriptEngine`
#[derive(Debug, Clone)]
pub struct Keccak256Transcript<G: Group> {
  round: u16,
  state: [u8; KECCAK256_STATE_SIZE],
  transcript: Keccak256,
  _p: PhantomData<G>,
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
      _p: PhantomData,
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
    self.transcript.update(label);
    self.transcript.update(&o.to_transcript_bytes());
  }

  fn dom_sep(&mut self, bytes: &'static [u8]) {
    self.transcript.update(DOM_SEP_TAG);
    self.transcript.update(bytes);
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    provider::bn256_grumpkin::bn256,
    provider::{self, keccak::Keccak256Transcript, secp_secq},
    traits::{Group, PrimeFieldExt, TranscriptEngineTrait, TranscriptReprTrait},
  };
  use ff::PrimeField;
  use rand::Rng;
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

    test_keccak_transcript_with::<provider::secp_secq::secp256k1::Point>(
      "9723aafb69ec8f0e9c7de756df0993247d98cf2b2f72fa353e3de654a177e310",
      "a6a90fcb6e1b1a2a2f84c950ef1510d369aea8e42085f5c629bfa66d00255f25",
    );
  }

  #[test]
  fn test_keccak_example() {
    let mut hasher = Keccak256::new();
    hasher.update(0xffffffff_u32.to_le_bytes());
    let output: [u8; 32] = hasher.finalize().try_into().unwrap();
    assert_eq!(
      hex::encode(output),
      "29045a592007d0c246ef02c2223570da9522d0cf0f73282c79a1bc8f0bb2c238"
    );
  }

  use super::{
    DOM_SEP_TAG, KECCAK256_PREFIX_CHALLENGE_HI, KECCAK256_PREFIX_CHALLENGE_LO,
    KECCAK256_STATE_SIZE, PERSONA_TAG,
  };

  fn compute_updated_state_for_testing(input: &[u8]) -> [u8; KECCAK256_STATE_SIZE] {
    let input_lo = [input, &[KECCAK256_PREFIX_CHALLENGE_LO]].concat();
    let input_hi = [input, &[KECCAK256_PREFIX_CHALLENGE_HI]].concat();

    let mut hasher_lo = Keccak256::new();
    let mut hasher_hi = Keccak256::new();

    hasher_lo.update(&input_lo);
    hasher_hi.update(&input_hi);

    let output_lo = hasher_lo.finalize();
    let output_hi = hasher_hi.finalize();

    [output_lo, output_hi]
      .concat()
      .as_slice()
      .try_into()
      .unwrap()
  }

  fn squeeze_for_testing(
    transcript: &[u8],
    round: u16,
    state: [u8; KECCAK256_STATE_SIZE],
    label: &'static [u8],
  ) -> [u8; 64] {
    let input = [
      transcript,
      DOM_SEP_TAG,
      round.to_le_bytes().as_ref(),
      state.as_ref(),
      label,
    ]
    .concat();
    compute_updated_state_for_testing(&input)
  }

  // This test is meant to ensure compatibility between the incremental way of computing the transcript above, and
  // the former, which materialized the entirety of the input vector before calling Keccak256 on it.
  fn test_keccak_transcript_incremental_vs_explicit_with<G: Group>() {
    let test_label = b"test";
    let mut transcript: Keccak256Transcript<G> = Keccak256Transcript::new(test_label);
    let mut rng = rand::thread_rng();

    // ten scalars
    let scalars = std::iter::from_fn(|| Some(<G as Group>::Scalar::from(rng.gen::<u64>())))
      .take(10)
      .collect::<Vec<_>>();

    // add the scalars to the transcripts,
    let mut manual_transcript: Vec<u8> = vec![];
    let labels = [
      b"s1", b"s2", b"s3", b"s4", b"s5", b"s6", b"s7", b"s8", b"s9", b"s0",
    ];

    for i in 0..10 {
      transcript.absorb(&labels[i][..], &scalars[i]);
      manual_transcript.extend(labels[i]);
      manual_transcript.extend(scalars[i].to_transcript_bytes());
    }

    // compute the initial state
    let input = [PERSONA_TAG, test_label].concat();
    let initial_state = compute_updated_state_for_testing(&input);

    // make a challenge
    let c1: <G as Group>::Scalar = transcript.squeeze(b"c1").unwrap();

    let c1_bytes = squeeze_for_testing(&manual_transcript[..], 0u16, initial_state, b"c1");
    let to_hex = |g: G::Scalar| hex::encode(g.to_repr().as_ref());
    assert_eq!(to_hex(c1), to_hex(G::Scalar::from_uniform(&c1_bytes)));
  }

  #[test]
  fn test_keccak_transcript_incremental_vs_explicit() {
    test_keccak_transcript_incremental_vs_explicit_with::<pasta_curves::pallas::Point>();
    test_keccak_transcript_incremental_vs_explicit_with::<bn256::Point>();
    test_keccak_transcript_incremental_vs_explicit_with::<secp_secq::secp256k1::Point>();
    test_keccak_transcript_incremental_vs_explicit_with::<secp_secq::secq256k1::Point>();
  }
}
