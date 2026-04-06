//! Support for efficiently generating R1CS witness using bellperson.

use ff::PrimeField;

use crate::frontend::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

/// A [`ConstraintSystem`] trait
pub trait SizedWitness<Scalar: PrimeField> {
  /// Returns the number of constraints in the constraint system
  fn num_constraints(&self) -> usize;
  /// Returns the number of inputs in the constraint system
  fn num_inputs(&self) -> usize;
  /// Returns the number of auxiliary variables in the constraint system
  fn num_aux(&self) -> usize;

  /// Generate a witness for the constraint system
  fn generate_witness_into(&mut self, aux: &mut [Scalar], inputs: &mut [Scalar]) -> Scalar;

  /// Generate a witness for the constraint system
  fn generate_witness_into_cs<CS: ConstraintSystem<Scalar>>(&mut self, cs: &mut CS) -> Scalar {
    assert!(cs.is_witness_generator());

    let aux_count = self.num_aux();
    let inputs_count = self.num_inputs();

    let (aux, inputs) = cs.allocate_empty(aux_count, inputs_count);

    assert_eq!(aux.len(), aux_count);
    assert_eq!(inputs.len(), inputs_count);

    self.generate_witness_into(aux, inputs)
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
pub struct WitnessCS<Scalar>
where
  Scalar: PrimeField,
{
  // Assignments of variables
  pub(crate) input_assignment: Vec<Scalar>,
  pub(crate) aux_assignment: Vec<Scalar>,
}

impl<Scalar> WitnessCS<Scalar>
where
  Scalar: PrimeField,
{
  /// Create a new WitnessCS with pre-allocated capacity for aux and input variables.
  pub fn with_capacity(aux_capacity: usize, input_capacity: usize) -> Self {
    let mut input_assignment = Vec::with_capacity(input_capacity + 2);
    input_assignment.push(Scalar::ONE);
    input_assignment.push(Scalar::ZERO);
    Self {
      input_assignment,
      aux_assignment: Vec::with_capacity(aux_capacity),
    }
  }

  /// Clear the assignments while retaining allocated capacity.
  pub fn clear(&mut self) {
    self.input_assignment.clear();
    self.input_assignment.push(Scalar::ONE);
    self.input_assignment.push(Scalar::ZERO);
    self.aux_assignment.clear();
  }

  /// Get input assignment
  pub fn input_assignment(&self) -> &[Scalar] {
    &self.input_assignment
  }

  /// Get aux assignment
  pub fn aux_assignment(&self) -> &[Scalar] {
    &self.aux_assignment
  }
}

impl<Scalar> ConstraintSystem<Scalar> for WitnessCS<Scalar>
where
  Scalar: PrimeField,
{
  type Root = Self;

  fn new() -> Self {
    let input_assignment = vec![Scalar::ONE, Scalar::ZERO];

    Self {
      input_assignment,
      aux_assignment: vec![],
    }
  }

  fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.aux_assignment.push(f()?);

    Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.input_assignment.push(f()?);

    Ok(Variable(Index::Input(self.input_assignment.len() - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, _a: LA, _b: LB, _c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
  {
    // Do nothing: we don't care about linear-combination evaluations in this context.
  }

  fn push_namespace<NR, N>(&mut self, _: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    // Do nothing; we don't care about namespaces in this context.
  }

  fn pop_namespace(&mut self) {
    // Do nothing; we don't care about namespaces in this context.
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Extensible
  fn is_extensible() -> bool {
    true
  }

  fn extend(&mut self, other: &Self) {
    self.input_assignment
            // Skip built-in Input(0) = ONE and Input(1) = ZERO.
            .extend(&other.input_assignment[2..]);
    self.aux_assignment.extend(&other.aux_assignment);
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Witness generator
  fn is_witness_generator(&self) -> bool {
    true
  }

  fn extend_inputs(&mut self, new_inputs: &[Scalar]) {
    self.input_assignment.extend(new_inputs);
  }

  fn extend_aux(&mut self, new_aux: &[Scalar]) {
    self.aux_assignment.extend(new_aux);
  }

  fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
    let allocated_aux = {
      let i = self.aux_assignment.len();
      self.aux_assignment.resize(aux_n + i, Scalar::ZERO);
      &mut self.aux_assignment[i..]
    };

    let allocated_inputs = {
      let i = self.input_assignment.len();
      self.input_assignment.resize(inputs_n + i, Scalar::ZERO);
      &mut self.input_assignment[i..]
    };

    (allocated_aux, allocated_inputs)
  }

  fn inputs_slice(&self) -> &[Scalar] {
    &self.input_assignment
  }

  fn aux_slice(&self) -> &[Scalar] {
    &self.aux_assignment
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::frontend::ConstraintSystem;
  use ff::Field;
  use halo2curves::pasta::Fq as Scalar;

  #[test]
  fn test_extend_skips_builtin_inputs() {
    // `other` has the 2 built-in inputs (ONE, ZERO) plus one user input
    let mut other = WitnessCS::<Scalar>::new();
    other
      .alloc_input(|| "user_input", || Ok(Scalar::from(42u64)))
      .unwrap();
    assert_eq!(other.input_assignment.len(), 3); // ONE, ZERO, user

    // `self` starts with just the 2 built-in inputs
    let mut base = WitnessCS::<Scalar>::new();
    assert_eq!(base.input_assignment.len(), 2); // ONE, ZERO

    base.extend(&other);

    // Only the user input should be copied, not the built-in ONE or ZERO
    assert_eq!(
      base.input_assignment.len(),
      3,
      "extend should copy only user inputs, not built-in ONE/ZERO"
    );
    assert_eq!(base.input_assignment[0], Scalar::ONE); // ONE
    assert_eq!(base.input_assignment[1], Scalar::ZERO); // ZERO
    assert_eq!(base.input_assignment[2], Scalar::from(42u64)); // user input
  }
}
