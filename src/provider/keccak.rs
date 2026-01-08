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

  #[cfg(not(feature = "evm"))]
  return [output_lo, output_hi]
    .concat()
    .as_slice()
    .try_into()
    .unwrap();
  #[cfg(feature = "evm")]
  return [output_hi, output_lo]
    .concat()
    .as_slice()
    .try_into()
    .unwrap();
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

  #[cfg(not(feature = "evm"))]
  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, NovaError> {
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
    Ok(E::Scalar::from_uniform(&output))
  }

  #[cfg(feature = "evm")]
  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, NovaError> {
    // we gather the full input from the round, preceded by the current state of the transcript
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

    // squeeze out a challenge
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
  use crate::{
    provider::{
      keccak::Keccak256Transcript, Bn256EngineKZG, GrumpkinEngine, PallasEngine, Secp256k1Engine,
      Secq256k1Engine, VestaEngine,
    },
    traits::{Engine, PrimeFieldExt, TranscriptEngineTrait, TranscriptReprTrait},
  };
  use ff::PrimeField;
  use rand::Rng;
  use sha3::{Digest, Keccak256};

  fn test_keccak_transcript_with<E: Engine>(expected_h1: &'static str, expected_h2: &'static str) {
    let mut transcript: Keccak256Transcript<E> = Keccak256Transcript::new(b"test");

    // two scalars
    let s1 = <E as Engine>::Scalar::from(2u64);
    let s2 = <E as Engine>::Scalar::from(5u64);

    // add the scalars to the transcript
    transcript.absorb(b"s1", &s1);
    transcript.absorb(b"s2", &s2);

    // make a challenge
    let c1: <E as Engine>::Scalar = transcript.squeeze(b"c1").unwrap();
    assert_eq!(hex::encode(c1.to_repr().as_ref()), expected_h1);

    // a scalar
    let s3 = <E as Engine>::Scalar::from(128u64);

    // add the scalar to the transcript
    transcript.absorb(b"s3", &s3);

    // make a challenge
    let c2: <E as Engine>::Scalar = transcript.squeeze(b"c2").unwrap();
    assert_eq!(hex::encode(c2.to_repr().as_ref()), expected_h2);
  }

  #[test]
  fn test_keccak_transcript() {
    test_keccak_transcript_with::<PallasEngine>(
      "b67339da79ce5f6dc72ad23c8c3b4179f49655cadf92d47e79c3e7788f00f125",
      "b7f033d47b3519dd6efe320b995eaad1dc11712cb9b655d2e7006ed5f86bd321",
    );

    test_keccak_transcript_with::<Bn256EngineKZG>(
      "b387ba3a8b9a22b3b7544a3dbbd26a048a1d354d8dc582c64d1513335e66a205",
      "73ad65097d947fe45de5241bb340bbd97b198b52cc559a9657f73c361bf8700b",
    );

    test_keccak_transcript_with::<Secp256k1Engine>(
      "f15ddd8fa1675a9e273e0ef441711005d77a5fd485f4e6cdee59760ca01493fa",
      "3c019f0e557abaecc99790382974cb27132bfe038af9c4d43a33ec9c426e19f5",
    );
  }

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
  fn test_keccak_transcript_incremental_vs_explicit_with<E: Engine>() {
    let test_label = b"test";
    let mut transcript: Keccak256Transcript<E> = Keccak256Transcript::new(test_label);
    let mut rng = rand::thread_rng();

    // ten scalars
    let scalars = std::iter::from_fn(|| Some(<E as Engine>::Scalar::from(rng.gen::<u64>())))
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
    let c1: <E as Engine>::Scalar = transcript.squeeze(b"c1").unwrap();

    let c1_bytes = squeeze_for_testing(&manual_transcript[..], 0u16, initial_state, b"c1");
    let to_hex = |g: E::Scalar| hex::encode(g.to_repr().as_ref());
    assert_eq!(to_hex(c1), to_hex(E::Scalar::from_uniform(&c1_bytes)));
  }

  #[test]
  fn test_keccak_transcript_incremental_vs_explicit() {
    test_keccak_transcript_incremental_vs_explicit_with::<PallasEngine>();
    test_keccak_transcript_incremental_vs_explicit_with::<VestaEngine>();
    test_keccak_transcript_incremental_vs_explicit_with::<Bn256EngineKZG>();
    test_keccak_transcript_incremental_vs_explicit_with::<GrumpkinEngine>();
    test_keccak_transcript_incremental_vs_explicit_with::<Secp256k1Engine>();
    test_keccak_transcript_incremental_vs_explicit_with::<Secq256k1Engine>();
  }
}
