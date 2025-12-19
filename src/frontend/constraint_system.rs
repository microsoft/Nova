use std::marker::PhantomData;

use ff::PrimeField;

use super::lc::{Index, LinearCombination, Variable};

/// Computations are expressed in terms of arithmetic circuits, in particular
/// rank-1 quadratic constraint systems. The `Circuit` trait represents a
/// circuit that can be synthesized. The `synthesize` method is called during
/// CRS generation and during proving.
pub trait Circuit<Scalar: PrimeField> {
  /// Synthesize the circuit into a rank-1 quadratic constraint system.
  fn synthesize<CS: ConstraintSystem<Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError>;
}

/// This is an error that could occur during circuit synthesis contexts,
/// such as CRS generation, proving or verification.
#[allow(clippy::upper_case_acronyms)]
#[derive(thiserror::Error, Debug, Clone)]
pub enum SynthesisError {
  /// During synthesis, we lacked knowledge of a variable assignment.
  #[error("an assignment for a variable could not be computed")]
  AssignmentMissing,
  /// During synthesis, we divided by zero.
  #[error("division by zero")]
  DivisionByZero,
  /// During synthesis, we constructed an unsatisfiable constraint system.
  #[error("unsatisfiable constraint system: {0}")]
  Unsatisfiable(String),
  /// During synthesis, our polynomials ended up being too high of degree
  #[error("polynomial degree is too large")]
  PolynomialDegreeTooLarge,
  /// During proof generation, we encountered an identity in the CRS
  #[error("encountered an identity element in the CRS")]
  UnexpectedIdentity,
  /// During verification, our verifying key was malformed.
  #[error("malformed verifying key")]
  MalformedVerifyingKey,
  /// During CRS generation, we observed an unconstrained auxiliary variable
  #[error("auxiliary variable was unconstrained")]
  UnconstrainedVariable,
  /// During GPU multiexp/fft, some GPU related error happened
  #[error("attempted to aggregate malformed proofs: {0}")]
  MalformedProofs(String),
  /// Non power of two proofs given for aggregation
  #[error("non power of two proofs given for aggregation")]
  NonPowerOfTwo,
  /// Incompatible vector length
  #[error("incompatible vector length: {0}")]
  IncompatibleLengthVector(String),
}

/// Represents a constraint system which can have new variables
/// allocated and constrains between them formed.
pub trait ConstraintSystem<Scalar: PrimeField>: Sized + Send {
  /// Represents the type of the "root" of this constraint system
  /// so that nested namespaces can minimize indirection.
  type Root: ConstraintSystem<Scalar>;

  /// Create a new empty constraint system
  fn new() -> Self {
    unimplemented!(
            "ConstraintSystem::new must be implemented for extensible types implementing ConstraintSystem"
        );
  }

  /// Return the "one" input variable
  fn one() -> Variable {
    Variable::new_unchecked(Index::Input(0))
  }

  /// Allocate a private variable in the constraint system. The provided function is used to
  /// determine the assignment of the variable. The given `annotation` function is invoked
  /// in testing contexts in order to derive a unique name for this variable in the current
  /// namespace.
  fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>;

  /// Allocate a public variable in the constraint system. The provided function is used to
  /// determine the assignment of the variable.
  fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>;

  /// Enforce that `A` * `B` = `C`. The `annotation` function is invoked in testing contexts
  /// in order to derive a unique name for the constraint in the current namespace.
  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>;

  /// Create a new (sub)namespace and enter into it. Not intended
  /// for downstream use; use `namespace` instead.
  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR;

  /// Exit out of the existing namespace. Not intended for
  /// downstream use; use `namespace` instead.
  fn pop_namespace(&mut self);

  /// Gets the "root" constraint system, bypassing the namespacing.
  /// Not intended for downstream use; use `namespace` instead.
  fn get_root(&mut self) -> &mut Self::Root;

  /// Begin a namespace for this constraint system.
  fn namespace<NR, N>(&mut self, name_fn: N) -> Namespace<'_, Scalar, Self::Root>
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    self.get_root().push_namespace(name_fn);

    Namespace(self.get_root(), Default::default())
  }

  /// Most implementations of ConstraintSystem are not 'extensible': they won't implement a specialized
  /// version of `extend` and should therefore also keep the default implementation of `is_extensible`
  /// so callers which optionally make use of `extend` can know to avoid relying on it when unimplemented.
  fn is_extensible() -> bool {
    false
  }

  /// Extend concatenates thew  `other` constraint systems to the receiver, modifying the receiver, whose
  /// inputs, allocated variables, and constraints will precede those of the `other` constraint system.
  /// The primary use case for this is parallel synthesis of circuits which can be decomposed into
  /// entirely independent sub-circuits. Each can be synthesized in its own thread, then the
  /// original `ConstraintSystem` can be extended with each, in the same order they would have
  /// been synthesized sequentially.
  fn extend(&mut self, _other: &Self) {
    unimplemented!(
      "ConstraintSystem::extend must be implemented for types implementing ConstraintSystem"
    );
  }

  /// Determines if the current `ConstraintSystem` instance is a witness generator.
  /// ConstraintSystems that are witness generators need not assemble the actual constraints. Rather, they exist only
  /// to efficiently create a witness.
  ///
  /// # Returns
  ///
  /// * `false` - By default, a `ConstraintSystem` is not a witness generator.
  fn is_witness_generator(&self) -> bool {
    false
  }

  /// Extend the inputs of the `ConstraintSystem`.
  ///
  /// # Panics
  ///
  /// Panics if called on a `ConstraintSystem` that is not a witness generator.
  fn extend_inputs(&mut self, _new_inputs: &[Scalar]) {
    assert!(self.is_witness_generator());
    unimplemented!("ConstraintSystem::extend_inputs must be implemented for witness generators implementing ConstraintSystem")
  }

  /// Extend the auxiliary inputs of the `ConstraintSystem`.
  ///
  /// # Panics
  ///
  /// Panics if called on a `ConstraintSystem` that is not a witness generator.
  fn extend_aux(&mut self, _new_aux: &[Scalar]) {
    assert!(self.is_witness_generator());
    unimplemented!("ConstraintSystem::extend_aux must be implemented for witness generators implementing ConstraintSystem")
  }

  /// Allocate empty space for the auxiliary inputs and the main inputs of the `ConstraintSystem`.
  ///
  /// # Panics
  ///
  /// Panics if called on a `ConstraintSystem` that is not a witness generator.
  fn allocate_empty(&mut self, _aux_n: usize, _inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
    // This method should only ever be called on witness generators.
    assert!(self.is_witness_generator());
    unimplemented!("ConstraintSystem::allocate_empty must be implemented for witness generators implementing ConstraintSystem")
  }

  /// Allocate empty space for the main inputs of the `ConstraintSystem`.
  ///
  /// # Panics
  ///
  /// Panics if called on a `ConstraintSystem` that is not a witness generator.
  fn allocate_empty_inputs(&mut self, _n: usize) -> &mut [Scalar] {
    // This method should only ever be called on witness generators.
    assert!(self.is_witness_generator());
    unimplemented!("ConstraintSystem::allocate_empty_inputs must be implemented for witness generators implementing ConstraintSystem")
  }

  /// Allocate empty space for the auxiliary inputs of the `ConstraintSystem`.
  ///
  /// # Panics
  ///
  /// Panics if called on a `ConstraintSystem` that is not a witness generator.
  fn allocate_empty_aux(&mut self, _n: usize) -> &mut [Scalar] {
    // This method should only ever be called on witness generators.
    assert!(self.is_witness_generator());
    unimplemented!("ConstraintSystem::allocate_empty_aux must be implemented for witness generators implementing ConstraintSystem")
  }

  /// Returns the constraint system's inputs as a slice of `Scalar`s.
  ///
  /// # Panics
  ///
  /// Panics if called on a `ConstraintSystem` that is not a witness generator.
  fn inputs_slice(&self) -> &[Scalar] {
    assert!(self.is_witness_generator());
    unimplemented!("ConstraintSystem::inputs_slice must be implemented for witness generators implementing ConstraintSystem")
  }

  /// Returns the constraint system's aux witness as a slice of `Scalar`s.
  ///
  /// # Panics
  ///
  /// Panics if called on a `ConstraintSystem` that is not a witness generator.
  fn aux_slice(&self) -> &[Scalar] {
    assert!(self.is_witness_generator());
    unimplemented!("ConstraintSystem::aux_slice must be implemented for witness generators implementing ConstraintSystem")
  }
}

/// This is a "namespaced" constraint system which borrows a constraint system (pushing
/// a namespace context) and, when dropped, pops out of the namespace context.
#[derive(Debug)]
pub struct Namespace<'a, Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  &'a mut CS,
  PhantomData<Scalar>,
);

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>> ConstraintSystem<Scalar>
  for Namespace<'_, Scalar, CS>
{
  type Root = CS::Root;

  fn one() -> Variable {
    CS::one()
  }

  fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.0.alloc(annotation, f)
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    self.0.alloc_input(annotation, f)
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
  {
    self.0.enforce(annotation, a, b, c)
  }

  // Downstream users who use `namespace` will never interact with these
  // functions and they will never be invoked because the namespace is
  // never a root constraint system.

  fn push_namespace<NR, N>(&mut self, _: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    panic!("only the root's push_namespace should be called");
  }

  fn pop_namespace(&mut self) {
    panic!("only the root's pop_namespace should be called");
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self.0.get_root()
  }

  fn is_witness_generator(&self) -> bool {
    self.0.is_witness_generator()
  }

  fn extend_inputs(&mut self, new_inputs: &[Scalar]) {
    self.0.extend_inputs(new_inputs)
  }

  fn extend_aux(&mut self, new_aux: &[Scalar]) {
    self.0.extend_aux(new_aux)
  }

  fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
    self.0.allocate_empty(aux_n, inputs_n)
  }

  fn inputs_slice(&self) -> &[Scalar] {
    self.0.inputs_slice()
  }
  fn aux_slice(&self) -> &[Scalar] {
    self.0.aux_slice()
  }
}

impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>> Drop for Namespace<'_, Scalar, CS> {
  fn drop(&mut self) {
    self.get_root().pop_namespace()
  }
}

/// Convenience implementation of `ConstraintSystem<Scalar>` for mutable references to
/// constraint systems.
impl<Scalar: PrimeField, CS: ConstraintSystem<Scalar>> ConstraintSystem<Scalar> for &'_ mut CS {
  type Root = CS::Root;

  fn one() -> Variable {
    CS::one()
  }

  fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    (**self).alloc(annotation, f)
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    (**self).alloc_input(annotation, f)
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
  {
    (**self).enforce(annotation, a, b, c)
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    (**self).push_namespace(name_fn)
  }

  fn pop_namespace(&mut self) {
    (**self).pop_namespace()
  }

  fn get_root(&mut self) -> &mut Self::Root {
    (**self).get_root()
  }

  fn namespace<NR, N>(&mut self, name_fn: N) -> Namespace<'_, Scalar, Self::Root>
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    (**self).namespace(name_fn)
  }

  fn is_extensible() -> bool {
    CS::is_extensible()
  }

  fn extend(&mut self, other: &Self) {
    (**self).extend(other)
  }

  fn is_witness_generator(&self) -> bool {
    (**self).is_witness_generator()
  }

  fn extend_inputs(&mut self, new_inputs: &[Scalar]) {
    (**self).extend_inputs(new_inputs)
  }

  fn extend_aux(&mut self, new_aux: &[Scalar]) {
    (**self).extend_aux(new_aux)
  }

  fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
    (**self).allocate_empty(aux_n, inputs_n)
  }

  fn allocate_empty_inputs(&mut self, n: usize) -> &mut [Scalar] {
    (**self).allocate_empty_inputs(n)
  }

  fn allocate_empty_aux(&mut self, n: usize) -> &mut [Scalar] {
    (**self).allocate_empty_aux(n)
  }

  fn inputs_slice(&self) -> &[Scalar] {
    (**self).inputs_slice()
  }

  fn aux_slice(&self) -> &[Scalar] {
    (**self).aux_slice()
  }
}
