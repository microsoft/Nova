//! Circuit representation of a [`u32`], with helpers for the sha256
//! gadgets.

use ff::PrimeField;

use crate::frontend::{ConstraintSystem, LinearCombination, SynthesisError};

use super::{
  boolean::{AllocatedBit, Boolean},
  multieq::MultiEq,
};

/// Represents an interpretation of 32 `Boolean` objects as an
/// unsigned integer.
#[derive(Clone, Debug)]
pub struct UInt32 {
  // Least significant bit first
  bits: Vec<Boolean>,
  value: Option<u32>,
}

impl UInt32 {
  /// Construct a constant `UInt32` from a `u32`
  pub fn constant(value: u32) -> Self {
    let mut bits = Vec::with_capacity(32);

    let mut tmp = value;
    for _ in 0..32 {
      if tmp & 1 == 1 {
        bits.push(Boolean::constant(true))
      } else {
        bits.push(Boolean::constant(false))
      }

      tmp >>= 1;
    }

    UInt32 {
      bits,
      value: Some(value),
    }
  }

  /// Returns the bits of this `UInt32` in big-endian order
  pub fn into_bits_be(self) -> Vec<Boolean> {
    let mut ret = self.bits;
    ret.reverse();
    ret
  }

  /// Constructs a `UInt32` from a slice of 32 `Boolean` bits in big-endian order.
  pub fn from_bits_be(bits: &[Boolean]) -> Self {
    assert_eq!(bits.len(), 32);

    let mut value = Some(0u32);
    for b in bits {
      if let Some(v) = value.as_mut() {
        *v <<= 1;
      }

      match b.get_value() {
        Some(true) => {
          if let Some(v) = value.as_mut() {
            *v |= 1;
          }
        }
        Some(false) => {}
        None => {
          value = None;
        }
      }
    }

    UInt32 {
      value,
      bits: bits.iter().rev().cloned().collect(),
    }
  }

  /// Returns a new `UInt32` with its bits rotated right by `by` positions.
  pub fn rotr(&self, by: usize) -> Self {
    let by = by % 32;

    let new_bits = self
      .bits
      .iter()
      .skip(by)
      .chain(self.bits.iter())
      .take(32)
      .cloned()
      .collect();

    UInt32 {
      bits: new_bits,
      value: self.value.map(|v| v.rotate_right(by as u32)),
    }
  }

  /// Returns a new `UInt32` with its bits shifted right by `by` positions, filling with zeros.
  pub fn shr(&self, by: usize) -> Self {
    let by = by % 32;

    let fill = Boolean::constant(false);

    let new_bits = self
            .bits
            .iter() // The bits are least significant first
            .skip(by) // Skip the bits that will be lost during the shift
            .chain(Some(&fill).into_iter().cycle()) // Rest will be zeros
            .take(32) // Only 32 bits needed!
            .cloned()
            .collect();

    UInt32 {
      bits: new_bits,
      value: self.value.map(|v| v >> by as u32),
    }
  }

  fn triop<Scalar, CS, F, U>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    c: &Self,
    tri_fn: F,
    circuit_fn: U,
  ) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
    F: Fn(u32, u32, u32) -> u32,
    U: Fn(&mut CS, usize, &Boolean, &Boolean, &Boolean) -> Result<Boolean, SynthesisError>,
  {
    let new_value = match (a.value, b.value, c.value) {
      (Some(a), Some(b), Some(c)) => Some(tri_fn(a, b, c)),
      _ => None,
    };

    let bits = a
      .bits
      .iter()
      .zip(b.bits.iter())
      .zip(c.bits.iter())
      .enumerate()
      .map(|(i, ((a, b), c))| circuit_fn(&mut cs, i, a, b, c))
      .collect::<Result<_, _>>()?;

    Ok(UInt32 {
      bits,
      value: new_value,
    })
  }

  /// Compute the `maj` value (a and b) xor (a and c) xor (b and c)
  /// during SHA256.
  pub fn sha256_maj<Scalar, CS>(
    cs: CS,
    a: &Self,
    b: &Self,
    c: &Self,
  ) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    Self::triop(
      cs,
      a,
      b,
      c,
      |a, b, c| (a & b) ^ (a & c) ^ (b & c),
      |cs, i, a, b, c| Boolean::sha256_maj(cs.namespace(|| format!("maj {i}")), a, b, c),
    )
  }

  /// Compute the `ch` value `(a and b) xor ((not a) and c)`
  /// during SHA256.
  pub fn sha256_ch<Scalar, CS>(cs: CS, a: &Self, b: &Self, c: &Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    Self::triop(
      cs,
      a,
      b,
      c,
      |a, b, c| (a & b) ^ ((!a) & c),
      |cs, i, a, b, c| Boolean::sha256_ch(cs.namespace(|| format!("ch {i}")), a, b, c),
    )
  }

  /// XOR this `UInt32` with another `UInt32`
  pub fn xor<Scalar, CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let new_value = match (self.value, other.value) {
      (Some(a), Some(b)) => Some(a ^ b),
      _ => None,
    };

    let bits = self
      .bits
      .iter()
      .zip(other.bits.iter())
      .enumerate()
      .map(|(i, (a, b))| Boolean::xor(cs.namespace(|| format!("xor of bit {i}")), a, b))
      .collect::<Result<_, _>>()?;

    Ok(UInt32 {
      bits,
      value: new_value,
    })
  }

  /// Perform modular addition of several `UInt32` objects.
  #[allow(clippy::unnecessary_unwrap)]
  pub fn addmany<Scalar, CS, M>(mut cs: M, operands: &[Self]) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
    M: ConstraintSystem<Scalar, Root = MultiEq<Scalar, CS>>,
  {
    // Make some arbitrary bounds for ourselves to avoid overflows
    // in the scalar field
    assert!(Scalar::NUM_BITS >= 64);
    assert!(operands.len() >= 2); // Weird trivial cases that should never happen
    assert!(operands.len() <= 10);

    // Compute the maximum value of the sum so we allocate enough bits for
    // the result
    let mut max_value = (operands.len() as u64) * (u64::from(u32::MAX));

    // Keep track of the resulting value
    let mut result_value = Some(0u64);

    // This is a linear combination that we will enforce to equal the
    // output
    let mut lc = LinearCombination::zero();

    let mut all_constants = true;

    // Iterate over the operands
    for op in operands {
      // Accumulate the value
      match op.value {
        Some(val) => {
          if let Some(v) = result_value.as_mut() {
            *v += u64::from(val);
          }
        }
        None => {
          // If any of our operands have unknown value, we won't
          // know the value of the result
          result_value = None;
        }
      }

      // Iterate over each bit of the operand and add the operand to
      // the linear combination
      let mut coeff = Scalar::ONE;
      for bit in &op.bits {
        lc = lc + &bit.lc(CS::one(), coeff);

        all_constants &= bit.is_constant();

        coeff = coeff.double();
      }
    }

    // The value of the actual result is modulo 2^32
    let modular_value = result_value.map(|v| v as u32);

    if all_constants && modular_value.is_some() {
      // We can just return a constant, rather than
      // unpacking the result into allocated bits.

      return Ok(UInt32::constant(modular_value.unwrap()));
    }

    // Storage area for the resulting bits
    let mut result_bits = vec![];

    // Linear combination representing the output,
    // for comparison with the sum of the operands
    let mut result_lc = LinearCombination::zero();

    // Allocate each bit of the result
    let mut coeff = Scalar::ONE;
    let mut i = 0;
    while max_value != 0 {
      // Allocate the bit
      let b = AllocatedBit::alloc(
        cs.namespace(|| format!("result bit {i}")),
        result_value.map(|v| (v >> i) & 1 == 1),
      )?;

      // Add this bit to the result combination
      result_lc = result_lc + (coeff, b.get_variable());

      result_bits.push(b.into());

      max_value >>= 1;
      i += 1;
      coeff = coeff.double();
    }

    // Enforce equality between the sum and result
    cs.get_root().enforce_equal(i, &lc, &result_lc);

    // Discard carry bits that we don't care about
    result_bits.truncate(32);

    Ok(UInt32 {
      bits: result_bits,
      value: modular_value,
    })
  }
}
