//! This module provides an implementation of `TranscriptEngineTrait` using keccak256
use crate::{
  errors::NovaError,
  traits::{Engine, PrimeFieldExt, TranscriptEngineTrait, TranscriptReprTrait},
};
use core::marker::PhantomData;
use ff::PrimeField;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha3::{Digest, Keccak256};

const PERSONA_TAG: &[u8] = b"NoTR";
const DOM_SEP_TAG: &[u8] = b"NoDS";
const KECCAK256_STATE_SIZE: usize = 64;
const KECCAK256_PREFIX_CHALLENGE_LO: u8 = 0;
const KECCAK256_PREFIX_CHALLENGE_HI: u8 = 1;

/// Provides an implementation of `TranscriptEngine`
#[derive(Debug, Clone)]
pub struct Keccak256Transcript<E: Engine> {
  round: u64,
  state: [u8; KECCAK256_STATE_SIZE],
  transcript: Keccak256,
  /// Bytes absorbed since the last squeeze (used to reconstruct `transcript` on deserialize)
  transcript_buffer: Vec<u8>,
  _p: PhantomData<E>,
}

impl<E: Engine> Serialize for Keccak256Transcript<E> {
  fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeStruct;
    let mut s = serializer.serialize_struct("Keccak256Transcript", 3)?;
    s.serialize_field("round", &self.round)?;
    s.serialize_field("state", &self.state.as_slice())?;
    s.serialize_field("transcript_buffer", &self.transcript_buffer)?;
    s.end()
  }
}

impl<'de, E: Engine> Deserialize<'de> for Keccak256Transcript<E> {
  fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    #[derive(Deserialize)]
    struct Helper {
      round: u64,
      state: Vec<u8>,
      #[serde(default)]
      transcript_buffer: Vec<u8>,
    }
    let h = Helper::deserialize(deserializer)?;
    let state: [u8; KECCAK256_STATE_SIZE] = h
      .state
      .try_into()
      .map_err(|_| serde::de::Error::custom("invalid state length"))?;

    // Replay buffered absorbs into a fresh Keccak256 to restore the hasher state
    let mut transcript = Keccak256::new();
    transcript.update(&h.transcript_buffer);

    Ok(Keccak256Transcript {
      round: h.round,
      state,
      transcript,
      transcript_buffer: h.transcript_buffer,
      _p: PhantomData,
    })
  }
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

impl<E: Engine> Keccak256Transcript<E> {
  /// Hash the transcript, update internal state, and return the raw 64-byte output.
  ///
  /// The only EVM/non-EVM differences (round byte order, output reversal) are
  /// handled here so that `squeeze` and `squeeze_bits` stay cfg-free.
  fn squeeze_raw(&mut self, label: &'static [u8]) -> Result<[u8; KECCAK256_STATE_SIZE], NovaError> {
    #[cfg(not(feature = "evm"))]
    let round_bytes = self.round.to_le_bytes();
    #[cfg(feature = "evm")]
    let round_bytes = self.round.to_be_bytes();

    let input = [
      DOM_SEP_TAG,
      round_bytes.as_ref(),
      self.state.as_ref(),
      label,
    ]
    .concat();
    #[allow(unused_mut)]
    let mut output = compute_updated_state(self.transcript.clone(), &input);

    self.round = self
      .round
      .checked_add(1)
      .ok_or(NovaError::InternalTranscriptError)?;
    self.state.copy_from_slice(&output);
    self.transcript = Keccak256::new();
    self.transcript_buffer.clear();

    #[cfg(feature = "evm")]
    output.reverse();

    Ok(output)
  }
}

impl<E: Engine> TranscriptEngineTrait<E> for Keccak256Transcript<E> {
  fn new(label: &'static [u8]) -> Self {
    let keccak_instance = Keccak256::new();
    let input = [PERSONA_TAG, label].concat();
    let output = compute_updated_state(keccak_instance.clone(), &input);

    Self {
      round: 0u64,
      state: output,
      transcript: keccak_instance,
      transcript_buffer: Vec::new(),
      _p: PhantomData,
    }
  }

  fn squeeze(&mut self, label: &'static [u8]) -> Result<E::Scalar, NovaError> {
    let output = self.squeeze_raw(label)?;
    Ok(E::Scalar::from_uniform(&output))
  }

  fn absorb<T: TranscriptReprTrait<E::GE>>(&mut self, label: &'static [u8], o: &T) {
    self.transcript.update(label);
    self.transcript_buffer.extend_from_slice(label);
    let repr = o.to_transcript_bytes();
    self.transcript.update(&repr);
    self.transcript_buffer.extend_from_slice(&repr);
  }

  fn dom_sep(&mut self, bytes: &'static [u8]) {
    self.transcript.update(DOM_SEP_TAG);
    self.transcript_buffer.extend_from_slice(DOM_SEP_TAG);
    self.transcript.update(bytes);
    self.transcript_buffer.extend_from_slice(bytes);
  }

  fn squeeze_bits(
    &mut self,
    label: &'static [u8],
    num_bits: usize,
    start_with_one: bool,
  ) -> Result<E::Scalar, NovaError> {
    assert!(num_bits >= 2);
    assert!(
      num_bits <= (E::Scalar::NUM_BITS - 1) as usize,
      "num_bits must be < field bit-width to avoid overflow when setting MSB"
    );

    let output = self.squeeze_raw(label)?;

    // Build scalar directly from raw hash bytes (little-endian)
    let mut repr = <E::Scalar as PrimeField>::Repr::default();
    let repr_bytes = repr.as_mut();
    let num_full_bytes = num_bits / 8;
    let remaining_bits = num_bits % 8;
    repr_bytes[..num_full_bytes].copy_from_slice(&output[..num_full_bytes]);
    if remaining_bits > 0 {
      repr_bytes[num_full_bytes] = output[num_full_bytes] & ((1u8 << remaining_bits) - 1);
    }
    if start_with_one {
      let msb_byte = (num_bits - 1) / 8;
      let msb_bit = (num_bits - 1) % 8;
      repr_bytes[msb_byte] |= 1u8 << msb_bit;
    }
    // Safe: num_bits < NUM_BITS, so value < 2^(NUM_BITS-1) < field modulus
    Ok(E::Scalar::from_repr(repr).unwrap())
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
  #[cfg(not(feature = "evm"))]
  fn test_keccak_transcript() {
    test_keccak_transcript_with::<PallasEngine>(
      "60dba8657186ff1abbeb237854707faf6ea79361546f8aae65a8fbb722c9ca0c",
      "8bb5dcd9f95115fbc178a1e76d04955423610f5788c7ef2ed200611fecfdf60b",
    );

    test_keccak_transcript_with::<Bn256EngineKZG>(
      "0f8d4f359394760435374d3d603ce0e970ea12f7a05e88eccd52d845f4ac542a",
      "6b32523d63dedd6fb51d5dfc127b9d133cad433ea0b38c4627abadd0e4404c10",
    );

    test_keccak_transcript_with::<Secp256k1Engine>(
      "6dbabc32c27f3512d7592ca08e50e2ded102959bd4bb01245f2ea8dcbae74ec4",
      "c4a806654016a01dd6a0c80e2a5484cba5af27ec4a0fd838ecca11eb1b4437bd",
    );
  }

  #[test]
  #[cfg(feature = "evm")]
  fn test_keccak_transcript() {
    test_keccak_transcript_with::<PallasEngine>(
      "78cce45b5f6cdc2021d9bba6c69c8c78c80c9a6ed65604db82d12166b28d212c",
      "7de5b755566a6a0423117770a9f3427f64fc0133dd6fc38a5e1f0790d3c6b20a",
    );

    test_keccak_transcript_with::<Bn256EngineKZG>(
      "59b12afc64ee9e2e1740bcd6d881ca1fab187a6261366b48aaeb5e23d949cf20",
      "b17d158ee602f2434af680597b09b9770022408c98276f0f46cbbf13bd86e020",
    );

    test_keccak_transcript_with::<Secp256k1Engine>(
      "f7ce678fa4de4f3bdbf1deaa5fc68e567f65e23ea2639585b01dc5127887721b",
      "a7fc93173c05e007ef1b30631400ed112463958e80a3af4d2508e4ac0e9a7409",
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

  #[cfg(not(feature = "evm"))]
  fn squeeze_for_testing(
    transcript: &[u8],
    round: u64,
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

  #[cfg(feature = "evm")]
  fn squeeze_for_testing(
    transcript: &[u8],
    round: u64,
    state: [u8; KECCAK256_STATE_SIZE],
    label: &'static [u8],
  ) -> [u8; 64] {
    let input = [
      transcript,
      DOM_SEP_TAG,
      round.to_be_bytes().as_ref(),
      state.as_ref(),
      label,
    ]
    .concat();
    let mut output = compute_updated_state_for_testing(&input);
    output.reverse();
    output
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

    let c1_bytes = squeeze_for_testing(&manual_transcript[..], 0u64, initial_state, b"c1");
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

  /// Test that a fresh transcript round-trips through serde (JSON).
  fn test_keccak_transcript_serde_fresh_with<E: Engine>() {
    let transcript: Keccak256Transcript<E> = Keccak256Transcript::new(b"serde_test");
    let json = serde_json::to_string(&transcript).unwrap();
    let restored: Keccak256Transcript<E> = serde_json::from_str(&json).unwrap();

    assert_eq!(transcript.round, restored.round);
    assert_eq!(transcript.state, restored.state);
  }

  /// Test that round and state survive serialization after absorb + squeeze.
  fn test_keccak_transcript_serde_after_ops_with<E: Engine>() {
    let mut transcript: Keccak256Transcript<E> = Keccak256Transcript::new(b"serde_test");

    let s1 = <E as Engine>::Scalar::from(42u64);
    let s2 = <E as Engine>::Scalar::from(99u64);
    transcript.absorb(b"s1", &s1);
    transcript.absorb(b"s2", &s2);
    let _c1: <E as Engine>::Scalar = transcript.squeeze(b"c1").unwrap();

    // Capture state after operations
    let round_before = transcript.round;
    let state_before = transcript.state;

    let json = serde_json::to_string(&transcript).unwrap();
    let restored: Keccak256Transcript<E> = serde_json::from_str(&json).unwrap();

    assert_eq!(round_before, restored.round);
    assert_eq!(state_before, restored.state);
  }

  /// Test that pending absorbs survive serialization (squeeze after deserialize
  /// must produce the same challenge as without the round-trip).
  fn test_keccak_transcript_serde_mid_absorb_with<E: Engine>() {
    let mut t1: Keccak256Transcript<E> = Keccak256Transcript::new(b"mid_absorb");
    let s1 = <E as Engine>::Scalar::from(11u64);
    let s2 = <E as Engine>::Scalar::from(22u64);
    t1.absorb(b"a", &s1);
    t1.absorb(b"b", &s2);
    // Do NOT squeeze yet — there are pending absorbs in the hasher.

    let json = serde_json::to_string(&t1).unwrap();
    let mut t2: Keccak256Transcript<E> = serde_json::from_str(&json).unwrap();

    // Now squeeze both and verify they produce the same challenge
    let c1: <E as Engine>::Scalar = t1.squeeze(b"ch").unwrap();
    let c2: <E as Engine>::Scalar = t2.squeeze(b"ch").unwrap();
    assert_eq!(c1, c2);
  }

  /// Test that a transcript deserialized from JSON with a wrong-length state
  /// produces an error instead of panicking.
  fn test_keccak_transcript_serde_bad_state_with<E: Engine>() {
    // Craft JSON with a state array of wrong length (32 instead of 64)
    let bad_json = r#"{"round":0,"state":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],"transcript_buffer":[]}"#;
    let result = serde_json::from_str::<Keccak256Transcript<E>>(bad_json);
    assert!(result.is_err());
  }

  /// Test bincode round-trip (used by Nova for proof serialization).
  fn test_keccak_transcript_serde_bincode_with<E: Engine>() {
    let mut transcript: Keccak256Transcript<E> = Keccak256Transcript::new(b"bincode_test");

    let s = <E as Engine>::Scalar::from(7u64);
    transcript.absorb(b"val", &s);
    let _c: <E as Engine>::Scalar = transcript.squeeze(b"ch").unwrap();

    let encoded = bincode::serde::encode_to_vec(&transcript, bincode::config::standard()).unwrap();
    let (restored, _): (Keccak256Transcript<E>, _) =
      bincode::serde::decode_from_slice(&encoded, bincode::config::standard()).unwrap();

    assert_eq!(transcript.round, restored.round);
    assert_eq!(transcript.state, restored.state);
  }

  #[test]
  fn test_keccak_transcript_serde_fresh() {
    test_keccak_transcript_serde_fresh_with::<PallasEngine>();
    test_keccak_transcript_serde_fresh_with::<VestaEngine>();
    test_keccak_transcript_serde_fresh_with::<Bn256EngineKZG>();
    test_keccak_transcript_serde_fresh_with::<GrumpkinEngine>();
    test_keccak_transcript_serde_fresh_with::<Secp256k1Engine>();
    test_keccak_transcript_serde_fresh_with::<Secq256k1Engine>();
  }

  #[test]
  fn test_keccak_transcript_serde_after_ops() {
    test_keccak_transcript_serde_after_ops_with::<PallasEngine>();
    test_keccak_transcript_serde_after_ops_with::<VestaEngine>();
    test_keccak_transcript_serde_after_ops_with::<Bn256EngineKZG>();
    test_keccak_transcript_serde_after_ops_with::<GrumpkinEngine>();
    test_keccak_transcript_serde_after_ops_with::<Secp256k1Engine>();
    test_keccak_transcript_serde_after_ops_with::<Secq256k1Engine>();
  }

  #[test]
  fn test_keccak_transcript_serde_mid_absorb() {
    test_keccak_transcript_serde_mid_absorb_with::<PallasEngine>();
    test_keccak_transcript_serde_mid_absorb_with::<VestaEngine>();
    test_keccak_transcript_serde_mid_absorb_with::<Bn256EngineKZG>();
    test_keccak_transcript_serde_mid_absorb_with::<GrumpkinEngine>();
    test_keccak_transcript_serde_mid_absorb_with::<Secp256k1Engine>();
    test_keccak_transcript_serde_mid_absorb_with::<Secq256k1Engine>();
  }

  #[test]
  fn test_keccak_transcript_serde_bad_state() {
    test_keccak_transcript_serde_bad_state_with::<PallasEngine>();
    test_keccak_transcript_serde_bad_state_with::<Bn256EngineKZG>();
    test_keccak_transcript_serde_bad_state_with::<Secp256k1Engine>();
  }

  #[test]
  fn test_keccak_transcript_serde_bincode() {
    test_keccak_transcript_serde_bincode_with::<PallasEngine>();
    test_keccak_transcript_serde_bincode_with::<VestaEngine>();
    test_keccak_transcript_serde_bincode_with::<Bn256EngineKZG>();
    test_keccak_transcript_serde_bincode_with::<GrumpkinEngine>();
    test_keccak_transcript_serde_bincode_with::<Secp256k1Engine>();
    test_keccak_transcript_serde_bincode_with::<Secq256k1Engine>();
  }
}
