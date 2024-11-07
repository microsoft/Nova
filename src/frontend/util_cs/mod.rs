//! The `util_cs` module provides a set of utilities for working with constraint system
use crate::frontend::LinearCombination;
use ff::PrimeField;

#[cfg(test)]
pub mod test_cs;
pub mod witness_cs;

/// Alias for a constraint in a constraint system
pub type Constraint<Scalar> = (
  LinearCombination<Scalar>,
  LinearCombination<Scalar>,
  LinearCombination<Scalar>,
  String,
);

/// The `Comparable` trait allows comparison of two constraint systems which
/// implement the trait. The only non-trivial method, `delta`, has a default
/// implementation which supplies the desired behavior.
///
/// Use `delta` to compare constraint systems. If they are not identical, the
/// returned `Delta` enum contains fine-grained information about how they
/// differ. This can be especially useful when debugging the situation in which
/// a constraint system is satisfied, but the corresponding Groth16 proof does
/// not verify.
///
/// If `ignore_counts` is  true, count mismatches will be ignored, and any constraint
/// mismatch will be returned. This is useful in pinpointing the source of a mismatch.
///
/// Example usage:
///
/// ```ignore
/// let delta = cs.delta(&cs_blank, false);
/// assert!(delta == Delta::Equal);
/// ```
pub trait Comparable<Scalar: PrimeField> {
  /// Returns the number of inputs in the constraint system
  fn num_inputs(&self) -> usize;
  /// Returns the number of constraints in the constraint system
  fn num_constraints(&self) -> usize;
  /// Returns the inputs in the constraint system
  fn inputs(&self) -> Vec<String>;
  /// Returns the auxiliary variables in the constraint system
  fn aux(&self) -> Vec<String>;
  /// Returns the constraints in the constraint system
  fn constraints(&self) -> &[Constraint<Scalar>];

  /// Compare two constraint systems and return a `Delta` enum
  fn delta<C: Comparable<Scalar>>(&self, other: &C, ignore_counts: bool) -> Delta<Scalar>
  where
    Scalar: PrimeField,
  {
    let input_count_matches = self.num_inputs() == other.num_inputs();
    let constraint_count_matches = self.num_constraints() == other.num_constraints();

    let inputs_match = self.inputs() == other.inputs();
    let constraints_match = self.constraints() == other.constraints();

    let equal =
      input_count_matches && constraint_count_matches && inputs_match && constraints_match;

    if !ignore_counts && !input_count_matches {
      Delta::InputCountMismatch(self.num_inputs(), other.num_inputs())
    } else if !ignore_counts && !constraint_count_matches {
      Delta::ConstraintCountMismatch(self.num_constraints(), other.num_constraints())
    } else if !constraints_match {
      let c = self.constraints();
      let o = other.constraints();

      let mismatch = c
        .iter()
        .zip(o)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| (i, a, b))
        .next();

      let m = mismatch.unwrap();

      Delta::ConstraintMismatch(m.0, m.1.clone(), m.2.clone())
    } else if equal {
      Delta::Equal
    } else {
      Delta::Different
    }
  }
}

/// The `Delta` enum is used to compare two constraint systems which implement
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq)]
pub enum Delta<Scalar: PrimeField> {
  /// The two constraint systems are equal
  Equal,
  /// The two constraint systems are different
  Different,
  /// The two constraint systems have a mismatch in the number of inputs
  InputCountMismatch(usize, usize),
  /// The two constraint systems have a mismatch in the number of constraints
  ConstraintCountMismatch(usize, usize),
  /// The two constraint systems have a mismatch in a constraint
  ConstraintMismatch(usize, Constraint<Scalar>, Constraint<Scalar>),
}
