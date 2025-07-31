//! Gadgets representing numbers in the scalar field of the underlying curve.

use ff::{PrimeField, PrimeFieldBits};
use serde::{Deserialize, Serialize};

use crate::frontend::{ConstraintSystem, LinearCombination, SynthesisError, Variable};

use crate::frontend::gadgets::boolean::{self, AllocatedBit, Boolean};

/// Represents an allocated number in the circuit.
#[derive(Debug, Serialize, Deserialize)]
pub struct AllocatedNum<Scalar: PrimeField> {
  value: Option<Scalar>,
  variable: Variable,
}

impl<Scalar: PrimeField> Clone for AllocatedNum<Scalar> {
  fn clone(&self) -> Self {
    AllocatedNum {
      value: self.value,
      variable: self.variable,
    }
  }
}

impl<Scalar: PrimeField> AllocatedNum<Scalar> {
  /// Allocate a `Variable(Aux)` in a `ConstraintSystem`.
  pub fn alloc<CS, F>(mut cs: CS, value: F) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    F: FnOnce() -> Result<Scalar, SynthesisError>,
  {
    let mut new_value = None;
    let var = cs.alloc(
      || "num",
      || {
        let tmp = value()?;

        new_value = Some(tmp);

        Ok(tmp)
      },
    )?;

    Ok(AllocatedNum {
      value: new_value,
      variable: var,
    })
  }

  /// Allocate a `Variable(Aux)` in a `ConstraintSystem`. Requires an
  /// infallible getter for the value.
  pub fn alloc_infallible<CS, F>(cs: CS, value: F) -> Self
  where
    CS: ConstraintSystem<Scalar>,
    F: FnOnce() -> Scalar,
  {
    Self::alloc(cs, || Ok(value())).unwrap()
  }

  /// Allocate a `Variable(Input)` in a `ConstraintSystem`.
  pub fn alloc_input<CS, F>(mut cs: CS, value: F) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    F: FnOnce() -> Result<Scalar, SynthesisError>,
  {
    let mut new_value = None;
    let var = cs.alloc_input(
      || "input num",
      || {
        let tmp = value()?;

        new_value = Some(tmp);

        Ok(tmp)
      },
    )?;

    Ok(AllocatedNum {
      value: new_value,
      variable: var,
    })
  }

  /// Allocate a `Variable` of either `Aux` or `Input` in a
  /// `ConstraintSystem`. The `Variable` is an `Input` if `is_input` is
  /// true. This allows uniform creation of circuits containing components
  /// which may or may not be public inputs.
  pub fn alloc_maybe_input<CS, F>(cs: CS, is_input: bool, value: F) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    F: FnOnce() -> Result<Scalar, SynthesisError>,
  {
    if is_input {
      Self::alloc_input(cs, value)
    } else {
      Self::alloc(cs, value)
    }
  }

  /// Make [`AllocatedNum`] a public input.
  pub fn inputize<CS>(&self, mut cs: CS) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let input = cs.alloc_input(
      || "input variable",
      || self.value.ok_or(SynthesisError::AssignmentMissing),
    )?;

    cs.enforce(
      || "enforce input is correct",
      |lc| lc + input,
      |lc| lc + CS::one(),
      |lc| lc + self.variable,
    );

    Ok(())
  }

  /// Deconstructs this allocated number into its
  /// boolean representation in little-endian bit
  /// order, requiring that the representation
  /// strictly exists "in the field" (i.e., a
  /// congruency is not allowed.)
  pub fn to_bits_le_strict<CS>(&self, mut cs: CS) -> Result<Vec<Boolean>, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    Scalar: PrimeFieldBits,
  {
    pub fn kary_and<Scalar, CS>(
      mut cs: CS,
      v: &[AllocatedBit],
    ) -> Result<AllocatedBit, SynthesisError>
    where
      Scalar: PrimeField,
      CS: ConstraintSystem<Scalar>,
    {
      assert!(!v.is_empty());

      // Let's keep this simple for now and just AND them all
      // manually
      let mut cur = None;

      for (i, v) in v.iter().enumerate() {
        if cur.is_none() {
          cur = Some(v.clone());
        } else {
          cur = Some(AllocatedBit::and(
            cs.namespace(|| format!("and {i}")),
            cur.as_ref().unwrap(),
            v,
          )?);
        }
      }

      Ok(cur.expect("v.len() > 0"))
    }

    // We want to ensure that the bit representation of a is
    // less than or equal to r - 1.
    let a = self.value.map(|e| e.to_le_bits());
    let b = (-Scalar::ONE).to_le_bits();

    // Get the bits of `a` in big-endian order.
    let mut a = a.as_ref().map(|e| e.into_iter().rev());

    let mut result = vec![];

    // Runs of ones in r
    let mut last_run = None;
    let mut current_run = vec![];

    let mut found_one = false;
    let mut i = 0;
    for b in b.into_iter().rev() {
      let a_bit: Option<bool> = a.as_mut().map(|e| *e.next().unwrap());

      // Skip over unset bits at the beginning
      found_one |= b;
      if !found_one {
        // a_bit should also be false
        if let Some(a_bit) = a_bit {
          assert!(!a_bit);
        }
        continue;
      }

      if b {
        // This is part of a run of ones. Let's just
        // allocate the boolean with the expected value.
        let a_bit = AllocatedBit::alloc(cs.namespace(|| format!("bit {i}")), a_bit)?;
        // ... and add it to the current run of ones.
        current_run.push(a_bit.clone());
        result.push(a_bit);
      } else {
        if !current_run.is_empty() {
          // This is the start of a run of zeros, but we need
          // to k-ary AND against `last_run` first.

          if last_run.is_some() {
            current_run.push(last_run.clone().unwrap());
          }
          last_run = Some(kary_and(
            cs.namespace(|| format!("run ending at {i}")),
            &current_run,
          )?);
          current_run.truncate(0);
        }

        // If `last_run` is true, `a` must be false, or it would
        // not be in the field.
        //
        // If `last_run` is false, `a` can be true or false.

        let a_bit = AllocatedBit::alloc_conditionally(
          cs.namespace(|| format!("bit {i}")),
          a_bit,
          last_run.as_ref().expect("char always starts with a one"),
        )?;
        result.push(a_bit);
      }

      i += 1;
    }

    // char is prime, so we'll always end on
    // a run of zeros.
    assert_eq!(current_run.len(), 0);

    // Now, we have `result` in big-endian order.
    // However, now we have to unpack self!

    let mut lc = LinearCombination::zero();
    let mut coeff = Scalar::ONE;

    for bit in result.iter().rev() {
      lc = lc + (coeff, bit.get_variable());

      coeff = coeff.double();
    }

    lc = lc - self.variable;

    cs.enforce(|| "unpacking constraint", |lc| lc, |lc| lc, |_| lc);

    // Convert into booleans, and reverse for little-endian bit order
    Ok(result.into_iter().map(Boolean::from).rev().collect())
  }

  /// Convert the allocated number into its little-endian representation.
  /// Note that this does not strongly enforce that the commitment is
  /// "in the field."
  pub fn to_bits_le<CS>(&self, mut cs: CS) -> Result<Vec<Boolean>, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
    Scalar: PrimeFieldBits,
  {
    let bits = boolean::field_into_allocated_bits_le(&mut cs, self.value)?;

    let mut lc = LinearCombination::zero();
    let mut coeff = Scalar::ONE;

    for bit in bits.iter() {
      lc = lc + (coeff, bit.get_variable());

      coeff = coeff.double();
    }

    lc = lc - self.variable;

    cs.enforce(|| "unpacking constraint", |lc| lc, |lc| lc, |_| lc);

    Ok(bits.into_iter().map(Boolean::from).collect())
  }

  /// Adds two allocated numbers together, returning a new allocated number.
  pub fn add<CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let mut value = None;

    let var = cs.alloc(
      || "sum num",
      || {
        let mut tmp = self.value.ok_or(SynthesisError::AssignmentMissing)?;
        tmp.add_assign(other.value.ok_or(SynthesisError::AssignmentMissing)?);

        value = Some(tmp);

        Ok(tmp)
      },
    )?;

    // Constrain: (a + b) * 1 = a + b
    cs.enforce(
      || "addition constraint",
      |lc| lc + self.variable + other.variable,
      |lc| lc + CS::one(),
      |lc| lc + var,
    );

    Ok(AllocatedNum {
      value,
      variable: var,
    })
  }

  /// Multiplies two allocated numbers together, returning a new allocated number.
  pub fn mul<CS>(&self, mut cs: CS, other: &Self) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let mut value = None;

    let var = cs.alloc(
      || "product num",
      || {
        let mut tmp = self.value.ok_or(SynthesisError::AssignmentMissing)?;
        tmp.mul_assign(other.value.ok_or(SynthesisError::AssignmentMissing)?);

        value = Some(tmp);

        Ok(tmp)
      },
    )?;

    // Constrain: a * b = ab
    cs.enforce(
      || "multiplication constraint",
      |lc| lc + self.variable,
      |lc| lc + other.variable,
      |lc| lc + var,
    );

    Ok(AllocatedNum {
      value,
      variable: var,
    })
  }

  /// Squares an allocated number, returning a new allocated number.
  pub fn square<CS>(&self, mut cs: CS) -> Result<Self, SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let mut value = None;

    let var = cs.alloc(
      || "squared num",
      || {
        let mut tmp = self.value.ok_or(SynthesisError::AssignmentMissing)?;
        tmp = tmp.square();

        value = Some(tmp);

        Ok(tmp)
      },
    )?;

    // Constrain: a * a = aa
    cs.enforce(
      || "squaring constraint",
      |lc| lc + self.variable,
      |lc| lc + self.variable,
      |lc| lc + var,
    );

    Ok(AllocatedNum {
      value,
      variable: var,
    })
  }

  /// Asserts that the allocated number is not zero.
  pub fn assert_nonzero<CS>(&self, mut cs: CS) -> Result<(), SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let inv = cs.alloc(
      || "ephemeral inverse",
      || {
        let tmp = self.value.ok_or(SynthesisError::AssignmentMissing)?;

        if tmp.is_zero().into() {
          Err(SynthesisError::DivisionByZero)
        } else {
          Ok(tmp.invert().unwrap())
        }
      },
    )?;

    // Constrain a * inv = 1, which is only valid
    // iff a has a multiplicative inverse, untrue
    // for zero.
    cs.enforce(
      || "nonzero assertion constraint",
      |lc| lc + self.variable,
      |lc| lc + inv,
      |lc| lc + CS::one(),
    );

    Ok(())
  }

  /// Takes two allocated numbers (a, b) and returns
  /// (b, a) if the condition is true, and (a, b)
  /// otherwise.
  pub fn conditionally_reverse<CS>(
    mut cs: CS,
    a: &Self,
    b: &Self,
    condition: &Boolean,
  ) -> Result<(Self, Self), SynthesisError>
  where
    CS: ConstraintSystem<Scalar>,
  {
    let c = Self::alloc(cs.namespace(|| "conditional reversal result 1"), || {
      if condition
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?
      {
        Ok(b.value.ok_or(SynthesisError::AssignmentMissing)?)
      } else {
        Ok(a.value.ok_or(SynthesisError::AssignmentMissing)?)
      }
    })?;

    cs.enforce(
      || "first conditional reversal",
      |lc| lc + a.variable - b.variable,
      |_| condition.lc(CS::one(), Scalar::ONE),
      |lc| lc + a.variable - c.variable,
    );

    let d = Self::alloc(cs.namespace(|| "conditional reversal result 2"), || {
      if condition
        .get_value()
        .ok_or(SynthesisError::AssignmentMissing)?
      {
        Ok(a.value.ok_or(SynthesisError::AssignmentMissing)?)
      } else {
        Ok(b.value.ok_or(SynthesisError::AssignmentMissing)?)
      }
    })?;

    cs.enforce(
      || "second conditional reversal",
      |lc| lc + b.variable - a.variable,
      |_| condition.lc(CS::one(), Scalar::ONE),
      |lc| lc + b.variable - d.variable,
    );

    Ok((c, d))
  }

  /// Get scalar value of the [`AllocatedNum`].
  pub fn get_value(&self) -> Option<Scalar> {
    self.value
  }

  /// Get the inner [`Variable`] of the [`AllocatedNum`].
  pub fn get_variable(&self) -> Variable {
    self.variable
  }
}

/// Represents a number in the circuit using a linear combination.
#[derive(Debug, Clone)]
pub struct Num<Scalar: PrimeField> {
  value: Option<Scalar>,
  lc: LinearCombination<Scalar>,
}

impl<Scalar: PrimeField> From<AllocatedNum<Scalar>> for Num<Scalar> {
  fn from(num: AllocatedNum<Scalar>) -> Num<Scalar> {
    Num {
      value: num.value,
      lc: LinearCombination::<Scalar>::from_variable(num.variable),
    }
  }
}

impl<Scalar: PrimeField> Num<Scalar> {
  /// Create a zero [`Num`].
  pub fn zero() -> Self {
    Num {
      value: Some(Scalar::ZERO),
      lc: LinearCombination::zero(),
    }
  }

  /// Get the scalar value of the [`Num`].
  pub fn get_value(&self) -> Option<Scalar> {
    self.value
  }

  /// Get the inner [`LinearCombination`] of the [`Num`].
  pub fn lc(&self, coeff: Scalar) -> LinearCombination<Scalar> {
    LinearCombination::zero() + (coeff, &self.lc)
  }

  /// Add a boolean to the Num with a given coefficient.
  pub fn add_bool_with_coeff(self, one: Variable, bit: &Boolean, coeff: Scalar) -> Self {
    let newval = match (self.value, bit.get_value()) {
      (Some(mut curval), Some(bval)) => {
        if bval {
          curval.add_assign(&coeff);
        }

        Some(curval)
      }
      _ => None,
    };

    Num {
      value: newval,
      lc: self.lc + &bit.lc(one, coeff),
    }
  }

  /// Add self to another Num, returning a new Num.
  #[allow(clippy::should_implement_trait)]
  pub fn add(self, other: &Self) -> Self {
    let lc = self.lc + &other.lc;
    let value = match (self.value, other.value) {
      (Some(v1), Some(v2)) => {
        let mut tmp = v1;
        tmp.add_assign(&v2);
        Some(tmp)
      }
      (Some(v), None) | (None, Some(v)) => Some(v),
      (None, None) => None,
    };

    Num { value, lc }
  }

  /// Scale the [`Num`] by a scalar.
  pub fn scale(mut self, scalar: Scalar) -> Self {
    for (_variable, fr) in self.lc.iter_mut() {
      fr.mul_assign(&scalar);
    }

    if let Some(ref mut v) = self.value {
      v.mul_assign(&scalar);
    }

    self
  }
}
