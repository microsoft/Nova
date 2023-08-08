//! Support for generating R1CS shape using bellperson.

use crate::traits::Group;
use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use ff::PrimeField;

#[allow(clippy::upper_case_acronyms)]
/// `ShapeCS` is a `ConstraintSystem` for creating `R1CSShape`s for a circuit.
pub struct ShapeCS<G: Group>
where
  G::Scalar: PrimeField,
{
  #[allow(clippy::type_complexity)]
  /// All constraints added to the `ShapeCS`.
  pub constraints: Vec<(
    LinearCombination<G::Scalar>,
    LinearCombination<G::Scalar>,
    LinearCombination<G::Scalar>,
  )>,
  inputs: usize,
  aux: usize,
}

impl<G: Group> ShapeCS<G> {
  /// Create a new, default `ShapeCS`,
  pub fn new() -> Self {
    ShapeCS::default()
  }

  /// Returns the number of constraints defined for this `ShapeCS`.
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Returns the number of inputs defined for this `ShapeCS`.
  pub fn num_inputs(&self) -> usize {
    self.inputs
  }

  /// Returns the number of aux inputs defined for this `ShapeCS`.
  pub fn num_aux(&self) -> usize {
    self.aux
  }
}

impl<G: Group> Default for ShapeCS<G> {
  fn default() -> Self {
    ShapeCS {
      constraints: vec![],
      inputs: 1,
      aux: 0,
    }
  }
}

impl<G: Group> ConstraintSystem<G::Scalar> for ShapeCS<G> {
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<G::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.aux += 1;

    Ok(Variable::new_unchecked(Index::Aux(self.aux - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<G::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.inputs += 1;

    Ok(Variable::new_unchecked(Index::Input(self.inputs - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
    LB: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
    LC: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
  {
    let a = a(LinearCombination::zero());
    let b = b(LinearCombination::zero());
    let c = c(LinearCombination::zero());

    self.constraints.push((a, b, c));
  }

  fn push_namespace<NR, N>(&mut self, _name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
  }

  fn pop_namespace(&mut self) {}

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }
}
