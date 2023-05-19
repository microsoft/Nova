//! Support for generating R1CS witness using bellperson.

use crate::traits::Group;
use ff::Field;

use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
pub struct SatisfyingAssignment<G: Group>
{
  // Assignments of variables
  pub(crate) input_assignment: Vec<G::Scalar>,
  pub(crate) aux_assignment: Vec<G::Scalar>,
}
use std::fmt;

impl<G: Group> fmt::Debug for SatisfyingAssignment<G> {
  fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
    fmt
      .debug_struct("SatisfyingAssignment")
      .field("input_assignment", &self.input_assignment)
      .field("aux_assignment", &self.aux_assignment)
      .finish()
  }
}

impl<G: Group> PartialEq for SatisfyingAssignment<G>
{
  fn eq(&self, other: &SatisfyingAssignment<G>) -> bool {
    self.input_assignment == other.input_assignment && self.aux_assignment == other.aux_assignment
  }
}

impl<G: Group> ConstraintSystem<G::Scalar> for SatisfyingAssignment<G>
{
  type Root = Self;

  fn new() -> Self {
    let input_assignment = vec![G::Scalar::ONE];

    Self {
      input_assignment,
      aux_assignment: vec![],
    }
  }

  fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<G::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.aux_assignment.push(f()?);

    Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<G::Scalar, SynthesisError>,
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
    LA: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
    LB: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
    LC: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
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

  fn is_extensible() -> bool {
    true
  }

  fn extend(&mut self, other: &Self) {
    self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
    self.aux_assignment.extend(&other.aux_assignment);
  }

  fn is_witness_generator(&self) -> bool {
    true
  }

  fn extend_inputs(&mut self, new_inputs: &[G::Scalar]) {
    self.input_assignment.extend(new_inputs);
  }

  fn extend_aux(&mut self, new_aux: &[G::Scalar]) {
    self.aux_assignment.extend(new_aux);
  }

  fn allocate_empty(
    &mut self,
    aux_n: usize,
    inputs_n: usize,
  ) -> (&mut [G::Scalar], &mut [G::Scalar]) {
    let allocated_aux = {
      let i = self.aux_assignment.len();
      self.aux_assignment.resize(aux_n + i, G::Scalar::ZERO);
      &mut self.aux_assignment[i..]
    };

    let allocated_inputs = {
      let i = self.input_assignment.len();
      self.input_assignment.resize(inputs_n + i, G::Scalar::ZERO);
      &mut self.input_assignment[i..]
    };

    (allocated_aux, allocated_inputs)
  }
}

#[allow(dead_code)]
impl<G: Group> SatisfyingAssignment<G>
{
  pub fn scalar_inputs(&self) -> Vec<G::Scalar> {
    self.input_assignment.clone()
  }

  pub fn scalar_aux(&self) -> Vec<G::Scalar> {
    self.aux_assignment.clone()
  }
}
