#[cfg(not(feature = "std"))]
use crate::prelude::*;
use crate::{constants::NUM_HASH_BITS, errors::NovaError};
use bincode::config::legacy;
use core::marker::PhantomData;
use ff::PrimeField;
use serde::Serialize;
use sha3::{Digest, Sha3_256};

/// Trait for components with potentially discrete digests to be included in their container's digest.
pub trait Digestible {
  /// Write the byte representation of Self. Returns a byte vector.
  fn to_bytes(&self) -> Result<Vec<u8>, NovaError>;
}

/// Marker trait to be implemented for types that implement `Digestible` and `Serialize`.
/// Their instances will be serialized to bytes then digested.
pub trait SimpleDigestible: Serialize {}

impl<T: SimpleDigestible> Digestible for T {
  fn to_bytes(&self) -> Result<Vec<u8>, NovaError> {
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
    let mut digest = F::ZERO;
    let mut coeff = F::ONE;
    for bit in bv {
      if bit {
        digest += coeff;
      }
      coeff += coeff;
    }
    digest
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
    let bytes = self.inner.to_bytes().expect("Serialization error");

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
  #[cfg(feature = "std")]
  use once_cell::sync::OnceCell;

  use super::{DigestComputer, SimpleDigestible};
  use crate::{provider::PallasEngine, traits::Engine};
  use bincode::config::legacy;
  use ff::Field;
  use serde::{Deserialize, Serialize};

  type E = PallasEngine;

  #[derive(Serialize, Deserialize)]
  struct S<E: Engine> {
    i: usize,
    #[serde(skip, default = "OnceCell::new")]
    digest: OnceCell<E::Scalar>,
  }

  impl<E: Engine> SimpleDigestible for S<E> {}

  impl<E: Engine> S<E> {
    fn new(i: usize) -> Self {
      S {
        i,
        digest: OnceCell::new(),
      }
    }

    fn digest(&mut self) -> E::Scalar {
      #[cfg(feature = "std")]
      let res = self
        .digest
        .get_or_try_init(|| DigestComputer::new(self).digest())
        .cloned()
        .expect("Failure in retrieving digest");
      #[cfg(not(feature = "std"))]
      let res = *self.digest.get_or_init(|| {
        DigestComputer::new(self)
          .digest()
          .expect("Failure in retrieving digest")
      });

      res
    }
  }

  #[test]
  fn test_digest_field_not_ingested_in_computation() {
    let s1 = S::<E>::new(42);

    // let's set up a struct with a weird digest field to make sure the digest computation does not depend of it
    let oc = OnceCell::new();
    oc.set(<E as Engine>::Scalar::ONE).unwrap();

    let mut s2: S<E> = S { i: 42, digest: oc };

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
    // let's set up a struct with a weird digest field to confuse deserializers
    let oc = OnceCell::new();
    oc.set(<E as Engine>::Scalar::ONE).unwrap();

    let mut good_s = S::<E>::new(42);
    let mut bad_s: S<E> = S { i: 42, digest: oc };

    // this justifies the adjective "bad"
    assert_ne!(good_s.digest(), bad_s.digest());

    let naughty_bytes = bincode::serde::encode_to_vec(&bad_s, legacy()).unwrap();
    let mut retrieved_s: S<E> = bincode::serde::decode_from_slice(&naughty_bytes, legacy())
      .unwrap()
      .0;

    assert_eq!(good_s.digest(), retrieved_s.digest())
  }
}
