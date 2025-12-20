//! Gadgets for allocating bits in the circuit and performing boolean logic.

use ff::{PrimeField, PrimeFieldBits};

use crate::frontend::{ConstraintSystem, LinearCombination, SynthesisError, Variable};

/// Represents a variable in the constraint system which is guaranteed
/// to be either zero or one.
#[derive(Debug, Clone)]
pub struct AllocatedBit {
  variable: Variable,
  value: Option<bool>,
}

impl AllocatedBit {
  /// Get inner value of [`AllocatedBit`]
  pub fn get_value(&self) -> Option<bool> {
    self.value
  }

  /// Get inner [`Variable`] of [`AllocatedBit`]
  pub fn get_variable(&self) -> Variable {
    self.variable
  }

  /// Allocate a variable in the constraint system which can only be a
  /// boolean value. Further, constrain that the boolean is false
  /// unless the condition is false.
  pub fn alloc_conditionally<Scalar, CS>(
    mut cs: CS,
    value: Option<bool>,
    must_be_false: &AllocatedBit,
  ) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let var = cs.alloc(
      || "boolean",
      || {
        if value.ok_or(SynthesisError::AssignmentMissing)? {
          Ok(Scalar::ONE)
        } else {
          Ok(Scalar::ZERO)
        }
      },
    )?;

    // Constrain: (1 - must_be_false - a) * a = 0
    // if must_be_false is true, the equation
    // reduces to -a * a = 0, which implies a = 0.
    // if must_be_false is false, the equation
    // reduces to (1 - a) * a = 0, which is a
    // traditional boolean constraint.
    cs.enforce(
      || "boolean constraint",
      |lc| lc + CS::one() - must_be_false.variable - var,
      |lc| lc + var,
      |lc| lc,
    );

    Ok(AllocatedBit {
      variable: var,
      value,
    })
  }

  /// Allocate a variable in the constraint system which can only be a
  /// boolean value.
  pub fn alloc<Scalar, CS>(mut cs: CS, value: Option<bool>) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let var = cs.alloc(
      || "boolean",
      || {
        if value.ok_or(SynthesisError::AssignmentMissing)? {
          Ok(Scalar::ONE)
        } else {
          Ok(Scalar::ZERO)
        }
      },
    )?;

    // Constrain: (1 - a) * a = 0
    // This constrains a to be either 0 or 1.
    cs.enforce(
      || "boolean constraint",
      |lc| lc + CS::one() - var,
      |lc| lc + var,
      |lc| lc,
    );

    Ok(AllocatedBit {
      variable: var,
      value,
    })
  }

  /// Performs an XOR operation over the two operands, returning
  /// an `AllocatedBit`.
  pub fn xor<Scalar, CS>(mut cs: CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let mut result_value = None;

    let result_var = cs.alloc(
      || "xor result",
      || {
        if a.value.ok_or(SynthesisError::AssignmentMissing)?
          ^ b.value.ok_or(SynthesisError::AssignmentMissing)?
        {
          result_value = Some(true);

          Ok(Scalar::ONE)
        } else {
          result_value = Some(false);

          Ok(Scalar::ZERO)
        }
      },
    )?;

    // Constrain (a + a) * (b) = (a + b - c)
    // Given that a and b are boolean constrained, if they
    // are equal, the only solution for c is 0, and if they
    // are different, the only solution for c is 1.
    //
    // ¬(a ∧ b) ∧ ¬(¬a ∧ ¬b) = c
    // (1 - (a * b)) * (1 - ((1 - a) * (1 - b))) = c
    // (1 - ab) * (1 - (1 - a - b + ab)) = c
    // (1 - ab) * (a + b - ab) = c
    // a + b - ab - (a^2)b - (b^2)a + (a^2)(b^2) = c
    // a + b - ab - ab - ab + ab = c
    // a + b - 2ab = c
    // -2a * b = c - a - b
    // 2a * b = a + b - c
    // (a + a) * b = a + b - c
    cs.enforce(
      || "xor constraint",
      |lc| lc + a.variable + a.variable,
      |lc| lc + b.variable,
      |lc| lc + a.variable + b.variable - result_var,
    );

    Ok(AllocatedBit {
      variable: result_var,
      value: result_value,
    })
  }

  /// Performs an AND operation over the two operands, returning
  /// an `AllocatedBit`.
  pub fn and<Scalar, CS>(mut cs: CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let mut result_value = None;

    let result_var = cs.alloc(
      || "and result",
      || {
        if a.value.ok_or(SynthesisError::AssignmentMissing)?
          & b.value.ok_or(SynthesisError::AssignmentMissing)?
        {
          result_value = Some(true);

          Ok(Scalar::ONE)
        } else {
          result_value = Some(false);

          Ok(Scalar::ZERO)
        }
      },
    )?;

    // Constrain (a) * (b) = (c), ensuring c is 1 iff
    // a AND b are both 1.
    cs.enforce(
      || "and constraint",
      |lc| lc + a.variable,
      |lc| lc + b.variable,
      |lc| lc + result_var,
    );

    Ok(AllocatedBit {
      variable: result_var,
      value: result_value,
    })
  }

  /// Calculates `a AND (NOT b)`.
  pub fn and_not<Scalar, CS>(mut cs: CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let mut result_value = None;

    let result_var = cs.alloc(
      || "and not result",
      || {
        if a.value.ok_or(SynthesisError::AssignmentMissing)?
          & !b.value.ok_or(SynthesisError::AssignmentMissing)?
        {
          result_value = Some(true);

          Ok(Scalar::ONE)
        } else {
          result_value = Some(false);

          Ok(Scalar::ZERO)
        }
      },
    )?;

    // Constrain (a) * (1 - b) = (c), ensuring c is 1 iff
    // a is true and b is false, and otherwise c is 0.
    cs.enforce(
      || "and not constraint",
      |lc| lc + a.variable,
      |lc| lc + CS::one() - b.variable,
      |lc| lc + result_var,
    );

    Ok(AllocatedBit {
      variable: result_var,
      value: result_value,
    })
  }

  /// Calculates `(NOT a) AND (NOT b)`.
  pub fn nor<Scalar, CS>(mut cs: CS, a: &Self, b: &Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let mut result_value = None;

    let result_var = cs.alloc(
      || "nor result",
      || {
        if !a.value.ok_or(SynthesisError::AssignmentMissing)?
          & !b.value.ok_or(SynthesisError::AssignmentMissing)?
        {
          result_value = Some(true);

          Ok(Scalar::ONE)
        } else {
          result_value = Some(false);

          Ok(Scalar::ZERO)
        }
      },
    )?;

    // Constrain (1 - a) * (1 - b) = (c), ensuring c is 1 iff
    // a and b are both false, and otherwise c is 0.
    cs.enforce(
      || "nor constraint",
      |lc| lc + CS::one() - a.variable,
      |lc| lc + CS::one() - b.variable,
      |lc| lc + result_var,
    );

    Ok(AllocatedBit {
      variable: result_var,
      value: result_value,
    })
  }
}

/// Convert a field element into a vector of [`AllocatedBit`]'s representing its bits.
pub fn field_into_allocated_bits_le<Scalar, CS>(
  mut cs: CS,
  value: Option<Scalar>,
) -> Result<Vec<AllocatedBit>, SynthesisError>
where
  Scalar: PrimeField,
  Scalar: PrimeFieldBits,
  CS: ConstraintSystem<Scalar>,
{
  // Deconstruct in big-endian bit order
  let values = match value {
    Some(ref value) => {
      let field_char = Scalar::char_le_bits();
      let mut field_char = field_char.into_iter().rev();

      let mut tmp = Vec::with_capacity(Scalar::NUM_BITS as usize);

      let mut found_one = false;
      for b in value.to_le_bits().into_iter().rev() {
        // Skip leading bits
        found_one |= field_char.next().unwrap();
        if !found_one {
          continue;
        }

        tmp.push(Some(b));
      }

      assert_eq!(tmp.len(), Scalar::NUM_BITS as usize);

      tmp
    }
    None => vec![None; Scalar::NUM_BITS as usize],
  };

  // Allocate in little-endian order
  let bits = values
    .into_iter()
    .rev()
    .enumerate()
    .map(|(i, b)| AllocatedBit::alloc(cs.namespace(|| format!("bit {i}")), b))
    .collect::<Result<Vec<_>, SynthesisError>>()?;

  Ok(bits)
}

/// This is a boolean value which may be either a constant or
/// an interpretation of an `AllocatedBit`.
#[derive(Clone, Debug)]
pub enum Boolean {
  /// Existential view of the boolean variable
  Is(AllocatedBit),
  /// Negated view of the boolean variable
  Not(AllocatedBit),
  /// Constant (not an allocated variable)
  Constant(bool),
}

impl Boolean {
  /// Check if the boolean is a constant
  pub fn is_constant(&self) -> bool {
    matches!(*self, Boolean::Constant(_))
  }

  /// Constrain two booleans to be equal.
  pub fn enforce_equal<Scalar, CS>(mut cs: CS, a: &Self, b: &Self) -> Result<(), SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    match (a, b) {
      (&Boolean::Constant(a), &Boolean::Constant(b)) => {
        if a == b {
          Ok(())
        } else {
          Err(SynthesisError::Unsatisfiable(
            "Booleans are not equal".to_string(),
          ))
        }
      }
      (&Boolean::Constant(true), a) | (a, &Boolean::Constant(true)) => {
        cs.enforce(
          || "enforce equal to one",
          |lc| lc,
          |lc| lc,
          |lc| lc + CS::one() - &a.lc(CS::one(), Scalar::ONE),
        );

        Ok(())
      }
      (&Boolean::Constant(false), a) | (a, &Boolean::Constant(false)) => {
        cs.enforce(
          || "enforce equal to zero",
          |lc| lc,
          |lc| lc,
          |_| a.lc(CS::one(), Scalar::ONE),
        );

        Ok(())
      }
      (a, b) => {
        cs.enforce(
          || "enforce equal",
          |lc| lc,
          |lc| lc,
          |_| a.lc(CS::one(), Scalar::ONE) - &b.lc(CS::one(), Scalar::ONE),
        );

        Ok(())
      }
    }
  }

  /// Get the inner value of the boolean.
  pub fn get_value(&self) -> Option<bool> {
    match *self {
      Boolean::Constant(c) => Some(c),
      Boolean::Is(ref v) => v.get_value(),
      Boolean::Not(ref v) => v.get_value().map(|b| !b),
    }
  }

  /// Return a linear combination representing the boolean.
  pub fn lc<Scalar: PrimeField>(&self, one: Variable, coeff: Scalar) -> LinearCombination<Scalar> {
    match *self {
      Boolean::Constant(c) => {
        if c {
          LinearCombination::<Scalar>::zero() + (coeff, one)
        } else {
          LinearCombination::<Scalar>::zero()
        }
      }
      Boolean::Is(ref v) => LinearCombination::<Scalar>::zero() + (coeff, v.get_variable()),
      Boolean::Not(ref v) => {
        LinearCombination::<Scalar>::zero() + (coeff, one) - (coeff, v.get_variable())
      }
    }
  }

  /// Construct a boolean from a known constant
  pub fn constant(b: bool) -> Self {
    Boolean::Constant(b)
  }

  /// Return a negated interpretation of this boolean.
  pub fn not(&self) -> Self {
    match *self {
      Boolean::Constant(c) => Boolean::Constant(!c),
      Boolean::Is(ref v) => Boolean::Not(v.clone()),
      Boolean::Not(ref v) => Boolean::Is(v.clone()),
    }
  }

  /// Perform XOR over two boolean operands
  pub fn xor<'a, Scalar, CS>(cs: CS, a: &'a Self, b: &'a Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    match (a, b) {
      (&Boolean::Constant(false), x) | (x, &Boolean::Constant(false)) => Ok(x.clone()),
      (&Boolean::Constant(true), x) | (x, &Boolean::Constant(true)) => Ok(x.not()),
      // a XOR (NOT b) = NOT(a XOR b)
      (is @ &Boolean::Is(_), not @ &Boolean::Not(_))
      | (not @ &Boolean::Not(_), is @ &Boolean::Is(_)) => {
        Ok(Boolean::xor(cs, is, &not.not())?.not())
      }
      // a XOR b = (NOT a) XOR (NOT b)
      (&Boolean::Is(ref a), &Boolean::Is(ref b)) | (&Boolean::Not(ref a), &Boolean::Not(ref b)) => {
        Ok(Boolean::Is(AllocatedBit::xor(cs, a, b)?))
      }
    }
  }

  /// Perform AND over two boolean operands
  pub fn and<'a, Scalar, CS>(cs: CS, a: &'a Self, b: &'a Self) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    match (a, b) {
      // false AND x is always false
      (&Boolean::Constant(false), _) | (_, &Boolean::Constant(false)) => {
        Ok(Boolean::Constant(false))
      }
      // true AND x is always x
      (&Boolean::Constant(true), x) | (x, &Boolean::Constant(true)) => Ok(x.clone()),
      // a AND (NOT b)
      (&Boolean::Is(ref is), &Boolean::Not(ref not))
      | (&Boolean::Not(ref not), &Boolean::Is(ref is)) => {
        Ok(Boolean::Is(AllocatedBit::and_not(cs, is, not)?))
      }
      // (NOT a) AND (NOT b) = a NOR b
      (Boolean::Not(a), Boolean::Not(b)) => Ok(Boolean::Is(AllocatedBit::nor(cs, a, b)?)),
      // a AND b
      (Boolean::Is(a), Boolean::Is(b)) => Ok(Boolean::Is(AllocatedBit::and(cs, a, b)?)),
    }
  }

  /// Perform OR over two boolean operands
  pub fn or<'a, Scalar, CS>(
    mut cs: CS,
    a: &'a Boolean,
    b: &'a Boolean,
  ) -> Result<Boolean, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    Ok(Boolean::not(&Boolean::and(
      cs.namespace(|| "not and (not a) (not b)"),
      &Boolean::not(a),
      &Boolean::not(b),
    )?))
  }

  /// Computes (a and b) xor ((not a) and c)
  pub fn sha256_ch<'a, Scalar, CS>(
    mut cs: CS,
    a: &'a Self,
    b: &'a Self,
    c: &'a Self,
  ) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let ch_value = match (a.get_value(), b.get_value(), c.get_value()) {
      (Some(a), Some(b), Some(c)) => {
        // (a and b) xor ((not a) and c)
        Some((a & b) ^ ((!a) & c))
      }
      _ => None,
    };

    match (a, b, c) {
      (&Boolean::Constant(_), &Boolean::Constant(_), &Boolean::Constant(_)) => {
        // They're all constants, so we can just compute the value.

        return Ok(Boolean::Constant(ch_value.expect("they're all constants")));
      }
      (&Boolean::Constant(false), _, c) => {
        // If a is false
        // (a and b) xor ((not a) and c)
        // equals
        // (false) xor (c)
        // equals
        // c
        return Ok(c.clone());
      }
      (a, &Boolean::Constant(false), c) => {
        // If b is false
        // (a and b) xor ((not a) and c)
        // equals
        // ((not a) and c)
        return Boolean::and(cs, &a.not(), c);
      }
      (a, b, &Boolean::Constant(false)) => {
        // If c is false
        // (a and b) xor ((not a) and c)
        // equals
        // (a and b)
        return Boolean::and(cs, a, b);
      }
      (a, b, &Boolean::Constant(true)) => {
        // If c is true
        // (a and b) xor ((not a) and c)
        // equals
        // (a and b) xor (not a)
        // equals
        // not (a and (not b))
        return Ok(Boolean::and(cs, a, &b.not())?.not());
      }
      (a, &Boolean::Constant(true), c) => {
        // If b is true
        // (a and b) xor ((not a) and c)
        // equals
        // a xor ((not a) and c)
        // equals
        // not ((not a) and (not c))
        return Ok(Boolean::and(cs, &a.not(), &c.not())?.not());
      }
      (&Boolean::Constant(true), _, _) => {
        // If a is true
        // (a and b) xor ((not a) and c)
        // equals
        // b xor ((not a) and c)
        // So we just continue!
      }
      (
        &Boolean::Is(_) | &Boolean::Not(_),
        &Boolean::Is(_) | &Boolean::Not(_),
        &Boolean::Is(_) | &Boolean::Not(_),
      ) => {}
    }

    let ch = cs.alloc(
      || "ch",
      || {
        ch_value.ok_or(SynthesisError::AssignmentMissing).map(|v| {
          if v {
            Scalar::ONE
          } else {
            Scalar::ZERO
          }
        })
      },
    )?;

    // a(b - c) = ch - c
    cs.enforce(
      || "ch computation",
      |_| b.lc(CS::one(), Scalar::ONE) - &c.lc(CS::one(), Scalar::ONE),
      |_| a.lc(CS::one(), Scalar::ONE),
      |lc| lc + ch - &c.lc(CS::one(), Scalar::ONE),
    );

    Ok(
      AllocatedBit {
        value: ch_value,
        variable: ch,
      }
      .into(),
    )
  }

  /// Computes (a and b) xor (a and c) xor (b and c)
  pub fn sha256_maj<'a, Scalar, CS>(
    mut cs: CS,
    a: &'a Self,
    b: &'a Self,
    c: &'a Self,
  ) -> Result<Self, SynthesisError>
  where
    Scalar: PrimeField,
    CS: ConstraintSystem<Scalar>,
  {
    let maj_value = match (a.get_value(), b.get_value(), c.get_value()) {
      (Some(a), Some(b), Some(c)) => {
        // (a and b) xor (a and c) xor (b and c)
        Some((a & b) ^ (a & c) ^ (b & c))
      }
      _ => None,
    };

    match (a, b, c) {
      (&Boolean::Constant(_), &Boolean::Constant(_), &Boolean::Constant(_)) => {
        // They're all constants, so we can just compute the value.

        return Ok(Boolean::Constant(maj_value.expect("they're all constants")));
      }
      (&Boolean::Constant(false), b, c) => {
        // If a is false,
        // (a and b) xor (a and c) xor (b and c)
        // equals
        // (b and c)
        return Boolean::and(cs, b, c);
      }
      (a, &Boolean::Constant(false), c) => {
        // If b is false,
        // (a and b) xor (a and c) xor (b and c)
        // equals
        // (a and c)
        return Boolean::and(cs, a, c);
      }
      (a, b, &Boolean::Constant(false)) => {
        // If c is false,
        // (a and b) xor (a and c) xor (b and c)
        // equals
        // (a and b)
        return Boolean::and(cs, a, b);
      }
      (a, b, &Boolean::Constant(true)) => {
        // If c is true,
        // (a and b) xor (a and c) xor (b and c)
        // equals
        // (a and b) xor (a) xor (b)
        // equals
        // not ((not a) and (not b))
        return Ok(Boolean::and(cs, &a.not(), &b.not())?.not());
      }
      (a, &Boolean::Constant(true), c) => {
        // If b is true,
        // (a and b) xor (a and c) xor (b and c)
        // equals
        // (a) xor (a and c) xor (c)
        return Ok(Boolean::and(cs, &a.not(), &c.not())?.not());
      }
      (&Boolean::Constant(true), b, c) => {
        // If a is true,
        // (a and b) xor (a and c) xor (b and c)
        // equals
        // (b) xor (c) xor (b and c)
        return Ok(Boolean::and(cs, &b.not(), &c.not())?.not());
      }
      (
        &Boolean::Is(_) | &Boolean::Not(_),
        &Boolean::Is(_) | &Boolean::Not(_),
        &Boolean::Is(_) | &Boolean::Not(_),
      ) => {}
    }

    let maj = cs.alloc(
      || "maj",
      || {
        maj_value.ok_or(SynthesisError::AssignmentMissing).map(|v| {
          if v {
            Scalar::ONE
          } else {
            Scalar::ZERO
          }
        })
      },
    )?;

    // ¬(¬a ∧ ¬b) ∧ ¬(¬a ∧ ¬c) ∧ ¬(¬b ∧ ¬c)
    // (1 - ((1 - a) * (1 - b))) * (1 - ((1 - a) * (1 - c))) * (1 - ((1 - b) * (1 - c)))
    // (a + b - ab) * (a + c - ac) * (b + c - bc)
    // -2abc + ab + ac + bc
    // a (-2bc + b + c) + bc
    //
    // (b) * (c) = (bc)
    // (2bc - b - c) * (a) = bc - maj

    let bc = Self::and(cs.namespace(|| "b and c"), b, c)?;

    cs.enforce(
      || "maj computation",
      |_| {
        bc.lc(CS::one(), Scalar::ONE) + &bc.lc(CS::one(), Scalar::ONE)
          - &b.lc(CS::one(), Scalar::ONE)
          - &c.lc(CS::one(), Scalar::ONE)
      },
      |_| a.lc(CS::one(), Scalar::ONE),
      |_| bc.lc(CS::one(), Scalar::ONE) - maj,
    );

    Ok(
      AllocatedBit {
        value: maj_value,
        variable: maj,
      }
      .into(),
    )
  }
}

impl From<AllocatedBit> for Boolean {
  fn from(b: AllocatedBit) -> Boolean {
    Boolean::Is(b)
  }
}
