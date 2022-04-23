//! Support for generating R1CS witness using bellperson.

use crate::traits::Group;
use ff::{Field, PrimeField};

use bellperson::{
  multiexp::DensityTracker, ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};

/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
pub struct SatisfyingAssignment<G: Group>
where
  G::Scalar: PrimeField,
{
  // Density of queries
  a_aux_density: DensityTracker,
  b_input_density: DensityTracker,
  b_aux_density: DensityTracker,

  // Evaluations of A, B, C polynomials
  a: Vec<G::Scalar>,
  b: Vec<G::Scalar>,
  c: Vec<G::Scalar>,

  // Assignments of variables
  pub(crate) input_assignment: Vec<G::Scalar>,
  pub(crate) aux_assignment: Vec<G::Scalar>,
}
use std::fmt;

impl<G: Group> fmt::Debug for SatisfyingAssignment<G>
where
  G::Scalar: PrimeField,
{
  fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
    fmt
      .debug_struct("SatisfyingAssignment")
      .field("a_aux_density", &self.a_aux_density)
      .field("b_input_density", &self.b_input_density)
      .field("b_aux_density", &self.b_aux_density)
      .field(
        "a",
        &self
          .a
          .iter()
          .map(|v| format!("Fr({:?})", v))
          .collect::<Vec<_>>(),
      )
      .field(
        "b",
        &self
          .b
          .iter()
          .map(|v| format!("Fr({:?})", v))
          .collect::<Vec<_>>(),
      )
      .field(
        "c",
        &self
          .c
          .iter()
          .map(|v| format!("Fr({:?})", v))
          .collect::<Vec<_>>(),
      )
      .field("input_assignment", &self.input_assignment)
      .field("aux_assignment", &self.aux_assignment)
      .finish()
  }
}

impl<G: Group> PartialEq for SatisfyingAssignment<G>
where
  G::Scalar: PrimeField,
{
  fn eq(&self, other: &SatisfyingAssignment<G>) -> bool {
    self.a_aux_density == other.a_aux_density
      && self.b_input_density == other.b_input_density
      && self.b_aux_density == other.b_aux_density
      && self.a == other.a
      && self.b == other.b
      && self.c == other.c
      && self.input_assignment == other.input_assignment
      && self.aux_assignment == other.aux_assignment
  }
}

impl<G: Group> ConstraintSystem<G::Scalar> for SatisfyingAssignment<G>
where
  G::Scalar: PrimeField,
{
  type Root = Self;

  fn new() -> Self {
    let input_assignment = vec![G::Scalar::one()];
    let mut d = DensityTracker::new();
    d.add_element();

    Self {
      a_aux_density: DensityTracker::new(),
      b_input_density: d,
      b_aux_density: DensityTracker::new(),
      a: vec![],
      b: vec![],
      c: vec![],
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
    self.a_aux_density.add_element();
    self.b_aux_density.add_element();

    Ok(Variable(Index::Aux(self.aux_assignment.len() - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<G::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.input_assignment.push(f()?);
    self.b_input_density.add_element();

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

  fn extend(&mut self, other: Self) {
    self.a_aux_density.extend(other.a_aux_density, false);
    self.b_input_density.extend(other.b_input_density, true);
    self.b_aux_density.extend(other.b_aux_density, false);

    self.a.extend(other.a);
    self.b.extend(other.b);
    self.c.extend(other.c);

    self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
    self.aux_assignment.extend(other.aux_assignment);
  }
}
