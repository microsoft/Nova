//! Projective fraction arithmetic for the Logup-GKR fractional-sum tree.
//!
//! A fraction `num/den` is kept in projective form so the GKR circuit never
//! performs a field inversion: the fraction-add gate combines two children
//! `(n_l, d_l)` and `(n_r, d_r)` into `(n_l·d_r + n_r·d_l, d_l·d_r)`. This is the
//! root of why Logup-GKR avoids the inverse-polynomial commitments that the
//! current ppSNARK memory-check pays for.

use crate::traits::evm_serde::{CustomSerdeTrait, EvmCompatSerde};
use core::iter::Sum;
use core::ops::Add;
use ff::Field;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// A projective fraction `numerator / denominator` over the engine scalar field.
///
/// Equality of the represented rationals is cross-multiplicative
/// (`a/b == c/d` iff `a·d == c·b`); this type does not normalize.
#[serde_as]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "F: CustomSerdeTrait")]
pub struct Fraction<F: Field> {
  /// Numerator.
  #[serde_as(as = "EvmCompatSerde")]
  pub num: F,
  /// Denominator.
  #[serde_as(as = "EvmCompatSerde")]
  pub den: F,
}

impl<F: Field> Fraction<F> {
  /// Creates a new projective fraction `num/den`.
  pub fn new(num: F, den: F) -> Self {
    Self { num, den }
  }

  /// The additive identity `0/1` for projective fraction addition.
  pub fn zero() -> Self {
    Self {
      num: F::ZERO,
      den: F::ONE,
    }
  }
}

/// Projective fraction addition (the 2-to-1 gate): `a/b + c/d = (a·d + c·b)/(b·d)`.
///
/// `Fraction` is `Copy`, so `+` takes values with no cost.
impl<F: Field> Add for Fraction<F> {
  type Output = Self;

  fn add(self, rhs: Self) -> Self {
    Self {
      num: self.num * rhs.den + rhs.num * self.den,
      den: self.den * rhs.den,
    }
  }
}

/// Sum of a sequence of fractions, folding from the identity `0/1`. Lets a whole
/// tree level be reduced with `.sum()`.
impl<F: Field> Sum for Fraction<F> {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
    iter.fold(Self::zero(), |a, b| a + b)
  }
}

#[cfg(all(test, feature = "evm"))]
mod evm_serde_tests {
  use super::Fraction;
  use crate::traits::Engine;

  type E = crate::provider::Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  #[test]
  fn fraction_has_big_endian_scalar_golden_encoding() {
    let fraction = Fraction::new(Fr::from(1), Fr::from(2));
    let config = bincode::config::legacy()
      .with_big_endian()
      .with_fixed_int_encoding();
    let bytes = bincode::serde::encode_to_vec(fraction, config).expect("serialize fraction");

    let mut expected = [0u8; 64];
    expected[31] = 1;
    expected[63] = 2;
    assert_eq!(bytes, expected);

    let (decoded, consumed): (Fraction<Fr>, usize) =
      bincode::serde::decode_from_slice(&bytes, config).expect("deserialize fraction");
    assert_eq!(decoded, fraction);
    assert_eq!(consumed, expected.len());
  }
}
