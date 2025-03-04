#[cfg(not(feature = "std"))]
use crate::prelude::*;
use crate::{constants::NUM_HASH_BITS, errors::NovaError};
use bincode::config::legacy;
use ff::PrimeField;
use serde::Serialize;
use sha3::{Digest, Sha3_256};
#[cfg(feature = "std")]
use std::marker::PhantomData;

/// Trait for components with potentially discrete digests to be included in their container's digest.
pub trait Digestible {
  /// Write the byte representation of Self in a byte buffer
  fn write_bytes(&self) -> Result<Vec<u8>, NovaError>;
}

/// Marker trait to be implemented for types that implement `Digestible` and `Serialize`.
/// Their instances will be serialized to bytes then digested.
pub trait SimpleDigestible: Serialize {}

impl<T: SimpleDigestible> Digestible for T {
  fn write_bytes(&self) -> Result<Vec<u8>, NovaError> {
    bincode::serde::encode_to_vec(self, legacy()).map_err(|e| NovaError::DigestError {
      reason: e.to_string(),
    })
  }
}

pub struct DigestComputer<'a, F: PrimeField, T> {
  inner: &'a T,
  _phantom: PhantomData<F>,
}

impl<'a, F: PrimeField, T: Digestible> DigestComputer<'a, F, T> {
  fn hasher() -> Sha3_256 {
    Sha3_256::new()
  }

  fn map_to_field(digest: &[u8]) -> F {
    let bv = (0..NUM_HASH_BITS).map(|i| {
      let (byte_pos, bit_pos) = (i / 8, i % 8);
      let bit = (digest[byte_pos] >> bit_pos) & 1;
      bit == 1
    });

    // turn the bit vector into a scalar
    let mut result = F::ZERO;
    let mut coeff = F::ONE;
    for bit in bv {
      if bit {
        result += coeff;
      }
      coeff += coeff;
    }
    result
  }

  /// Create a new `DigestComputer`
  pub fn new(inner: &'a T) -> Self {
    DigestComputer {
      inner,
      _phantom: PhantomData,
    }
  }

  /// Compute the digest of a `Digestible` instance.
  pub fn digest(&self) -> Result<F, core::fmt::Error> {
    let bytes = self.inner.write_bytes().expect("Serialization error");

    let mut hasher = Self::hasher();
    hasher.update(&bytes);
    let final_bytes = hasher.finalize();
    let bytes: Vec<u8> = final_bytes.to_vec();

    // Now map to the field or handle it as necessary
    Ok(Self::map_to_field(&bytes))
  }
}

#[cfg(test)]
mod tests {
  use super::{DigestComputer, SimpleDigestible};
  use crate::{provider::PallasEngine, traits::Engine};
  use bincode::config::legacy;
  use ff::Field;
  use serde::{Deserialize, Serialize};

  type E = PallasEngine;

  #[derive(Serialize, Deserialize)]
  struct S<E: Engine> {
    i: usize,
    #[serde(skip)]
    digest: Option<E::Scalar>,
  }

  impl<E: Engine> SimpleDigestible for S<E> {}

  impl<E: Engine> S<E> {
    fn new(i: usize) -> Self {
      S { i, digest: None }
    }

    fn digest(&mut self) -> E::Scalar {
      let digest: E::Scalar = DigestComputer::new(self).digest().unwrap();
      *self.digest.get_or_insert(digest)
    }
  }

  #[test]
  fn test_digest_field_not_ingested_in_computation() {
    let s1 = S::<E>::new(42);

    let mut s2 = S::<E>::new(42);
    s2.digest = Some(<E as Engine>::Scalar::ONE); // Set manually for test

    assert_eq!(
      DigestComputer::<<E as Engine>::Scalar, _>::new(&s1)
        .digest()
        .unwrap(),
      DigestComputer::<<E as Engine>::Scalar, _>::new(&s2)
        .digest()
        .unwrap()
    );

    // note: because of the semantics of `OnceCell::get_or_try_init`, the above
    // equality will not result in `s1.digest() == s2.digest`
    assert_ne!(
      s2.digest(),
      DigestComputer::<<E as Engine>::Scalar, _>::new(&s2)
        .digest()
        .unwrap()
    );
  }

  #[test]
  fn test_digest_impervious_to_serialization() {
    let mut good_s = S::<E>::new(42);
    let mut bad_s = S::<E>::new(42);
    bad_s.digest = Some(<E as Engine>::Scalar::ONE); // Set manually for test

    // this justifies the adjective "bad"
    assert_ne!(good_s.digest(), bad_s.digest());

    let naughty_bytes = bincode::serde::encode_to_vec(&bad_s, legacy()).unwrap();
    let mut retrieved_s: S<E> = bincode::serde::decode_from_slice(&naughty_bytes, legacy())
      .unwrap()
      .0;

    assert_eq!(good_s.digest(), retrieved_s.digest())
  }
}
