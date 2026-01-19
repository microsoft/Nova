//! This module defines a trait that with serde_with::serde_as crate,
//! defines the behavior of a customized (de)serializer
//! to override the default ones from serde crate.

use serde::{Deserializer, Serializer};
use serde_with::{DeserializeAs, SerializeAs};

/// A helper trait to implement serde with custom behavior
pub trait CustomSerdeTrait: Sized + serde::Serialize + for<'de> serde::Deserialize<'de> {
  /// customized serializer, default to original serializer from serde
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serde::Serialize::serialize(&self, serializer)
  }
  /// customized deserializer, default to original deserializer from serde
  fn deserialize<'de, D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    <Self as serde::Deserialize>::deserialize(deserializer)
  }
}

/// A struct as helper for serde_as to use customized (de)serializer defined by CustomSerdeTrait
pub struct EvmCompatSerde;

impl<T: CustomSerdeTrait> SerializeAs<T> for EvmCompatSerde {
  fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error>
  where
    S: Serializer,
  {
    <T as CustomSerdeTrait>::serialize(source, serializer)
  }
}

impl<'de, T: CustomSerdeTrait> DeserializeAs<'de, T> for EvmCompatSerde {
  fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error>
  where
    D: Deserializer<'de>,
  {
    <T as CustomSerdeTrait>::deserialize(deserializer)
  }
}
