//! Allocation-bounded R1CS shape synthesis.
//!
//! Setup first counts the exact matrix dimensions and then repeats shape
//! synthesis while writing each constraint directly into canonical CSR
//! storage. The circuit must therefore produce the same shape on both passes.

use super::{
  r1cs::{add_constraint, nonzero_terms, sparse_matrix_with_capacity},
  ConstraintSystem, Index, LinearCombination, SynthesisError, Variable,
};
use crate::{
  errors::NovaError,
  r1cs::{R1CSShape, SparseMatrix},
  traits::Engine,
};
use core::marker::PhantomData;

/// Exact allocation plan produced by the counting synthesis pass.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ShapePlan {
  inputs: usize,
  aux: usize,
  constraints: usize,
  a_terms: usize,
  b_terms: usize,
  c_terms: usize,
}

impl ShapePlan {
  fn empty() -> Self {
    Self {
      inputs: 1,
      aux: 0,
      constraints: 0,
      a_terms: 0,
      b_terms: 0,
      c_terms: 0,
    }
  }
}

/// Constraint system that retains only exact shape allocation counts.
pub(crate) struct ShapePlanCS<E: Engine> {
  plan: ShapePlan,
  _p: PhantomData<E>,
}

impl<E: Engine> ShapePlanCS<E> {
  pub(crate) fn new() -> Self {
    Self {
      plan: ShapePlan::empty(),
      _p: PhantomData,
    }
  }

  pub(crate) fn into_plan(self) -> ShapePlan {
    self.plan
  }
}

impl<E: Engine> ConstraintSystem<E::Scalar> for ShapePlanCS<E> {
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let index = self.plan.aux;
    self.plan.aux = self.plan.aux.checked_add(1).expect("shape auxiliary count");
    Ok(Variable::new_unchecked(Index::Aux(index)))
  }

  fn alloc_input<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let index = self.plan.inputs;
    self.plan.inputs = self.plan.inputs.checked_add(1).expect("shape input count");
    Ok(Variable::new_unchecked(Index::Input(index)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LB: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LC: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
  {
    let a = a(LinearCombination::zero());
    let b = b(LinearCombination::zero());
    let c = c(LinearCombination::zero());

    self.plan.constraints = self
      .plan
      .constraints
      .checked_add(1)
      .expect("shape constraint count");
    self.plan.a_terms = self
      .plan
      .a_terms
      .checked_add(nonzero_terms(&a))
      .expect("shape A term count");
    self.plan.b_terms = self
      .plan
      .b_terms
      .checked_add(nonzero_terms(&b))
      .expect("shape B term count");
    self.plan.c_terms = self
      .plan
      .c_terms
      .checked_add(nonzero_terms(&c))
      .expect("shape C term count");
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

/// Second-pass constraint system that writes directly into exact CSR storage.
pub(crate) struct StreamingShapeCS<E: Engine> {
  plan: ShapePlan,
  inputs: usize,
  aux: usize,
  constraints: usize,
  a: SparseMatrix<E::Scalar>,
  b: SparseMatrix<E::Scalar>,
  c: SparseMatrix<E::Scalar>,
}

impl<E: Engine> StreamingShapeCS<E> {
  pub(crate) fn new(plan: ShapePlan) -> Self {
    assert!(plan.inputs > 0, "shape plan omits the one input");
    Self {
      plan,
      inputs: 1,
      aux: 0,
      constraints: 0,
      a: sparse_matrix_with_capacity(plan.constraints, plan.a_terms),
      b: sparse_matrix_with_capacity(plan.constraints, plan.b_terms),
      c: sparse_matrix_with_capacity(plan.constraints, plan.c_terms),
    }
  }

  pub(crate) fn into_r1cs_shape(mut self) -> Result<R1CSShape<E>, NovaError> {
    assert_eq!(self.inputs, self.plan.inputs, "streamed input count");
    assert_eq!(self.aux, self.plan.aux, "streamed auxiliary count");
    assert_eq!(
      self.constraints, self.plan.constraints,
      "streamed constraint count"
    );
    assert_eq!(self.a.data.len(), self.plan.a_terms, "streamed A terms");
    assert_eq!(self.b.data.len(), self.plan.b_terms, "streamed B terms");
    assert_eq!(self.c.data.len(), self.plan.c_terms, "streamed C terms");

    let columns = self.aux.checked_add(self.inputs).expect("shape columns");
    self.a.cols = columns;
    self.b.cols = columns;
    self.c.cols = columns;

    R1CSShape::new(
      self.constraints,
      self.aux,
      self.inputs - 1,
      self.a,
      self.b,
      self.c,
    )
  }
}

impl<E: Engine> ConstraintSystem<E::Scalar> for StreamingShapeCS<E> {
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    assert!(self.aux < self.plan.aux, "streamed auxiliary overflow");
    let index = self.aux;
    self.aux += 1;
    Ok(Variable::new_unchecked(Index::Aux(index)))
  }

  fn alloc_input<F, A, AR>(&mut self, _annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    assert!(self.inputs < self.plan.inputs, "streamed input overflow");
    let index = self.inputs;
    self.inputs += 1;
    Ok(Variable::new_unchecked(Index::Input(index)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, _annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LB: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LC: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
  {
    assert!(
      self.constraints < self.plan.constraints,
      "streamed constraint overflow"
    );
    let a = a(LinearCombination::zero());
    let b = b(LinearCombination::zero());
    let c = c(LinearCombination::zero());
    let mut matrices = (&mut self.a, &mut self.b, &mut self.c, &mut self.constraints);
    add_constraint(&mut matrices, self.plan.aux, &a, &b, &c);
    assert!(
      self.a.data.len() <= self.plan.a_terms,
      "streamed A overflow"
    );
    assert!(
      self.b.data.len() <= self.plan.b_terms,
      "streamed B overflow"
    );
    assert!(
      self.c.data.len() <= self.plan.c_terms,
      "streamed C overflow"
    );
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

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{r1cs::NovaShape, shape_cs::ShapeCS},
    provider::PallasEngine,
  };
  use ff::Field;

  type Scalar = <PallasEngine as Engine>::Scalar;

  fn synthesize<CS: ConstraintSystem<Scalar>>(cs: &mut CS) {
    let input = cs.alloc_input(|| "input", || Ok(Scalar::from(2))).unwrap();
    let left = cs.alloc(|| "left", || Ok(Scalar::from(3))).unwrap();
    cs.enforce(
      || "coalesced and zero terms",
      |linear| linear + left + left + (Scalar::ZERO, input) + (Scalar::from(7), input),
      |linear| linear + CS::one(),
      |linear| linear + input,
    );
    let right = cs.alloc(|| "right", || Ok(Scalar::from(5))).unwrap();
    cs.enforce(
      || "late auxiliary",
      |linear| linear + right,
      |linear| linear + left,
      |linear| linear + input,
    );
  }

  #[test]
  fn streamed_shape_matches_retained_shape_bytes() {
    let mut retained = ShapeCS::<PallasEngine>::new();
    synthesize(&mut retained);
    let retained = retained.r1cs_shape().unwrap();

    let mut counter = ShapePlanCS::<PallasEngine>::new();
    synthesize(&mut counter);
    let plan = counter.into_plan();
    assert_eq!(plan.inputs, 2);
    assert_eq!(plan.aux, 2);
    assert_eq!(plan.constraints, 2);

    let mut streamed = StreamingShapeCS::<PallasEngine>::new(plan);
    synthesize(&mut streamed);
    let streamed = streamed.into_r1cs_shape().unwrap();

    assert_eq!(streamed, retained);
    assert_eq!(streamed.digest(), retained.digest());
    let config = bincode::config::legacy()
      .with_little_endian()
      .with_fixed_int_encoding();
    assert_eq!(
      bincode::serde::encode_to_vec(&streamed, config).unwrap(),
      bincode::serde::encode_to_vec(&retained, config).unwrap()
    );
  }
}
