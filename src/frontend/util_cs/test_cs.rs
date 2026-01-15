//! Test constraint system for use in tests.

use std::collections::HashMap;

use crate::frontend::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

use ff::PrimeField;

#[derive(Debug)]
enum NamedObject {
  Constraint,
  Var,
  Namespace,
}

/// Constraint system for testing purposes.
#[derive(Debug)]
pub struct TestConstraintSystem<Scalar: PrimeField> {
  named_objects: HashMap<String, NamedObject>,
  current_namespace: Vec<String>,
  #[allow(clippy::type_complexity)]
  constraints: Vec<(
    LinearCombination<Scalar>,
    LinearCombination<Scalar>,
    LinearCombination<Scalar>,
    String,
  )>,
  inputs: Vec<(Scalar, String)>,
  aux: Vec<(Scalar, String)>,
}

fn _eval_lc2<Scalar: PrimeField>(
  terms: &LinearCombination<Scalar>,
  inputs: &[Scalar],
  aux: &[Scalar],
) -> Scalar {
  let mut acc = Scalar::ZERO;

  for (var, coeff) in terms.iter() {
    let mut tmp = match var.get_unchecked() {
      Index::Input(index) => inputs[index],
      Index::Aux(index) => aux[index],
    };

    tmp.mul_assign(coeff);
    acc.add_assign(&tmp);
  }

  acc
}

fn eval_lc<Scalar: PrimeField>(
  terms: &LinearCombination<Scalar>,
  inputs: &[(Scalar, String)],
  aux: &[(Scalar, String)],
) -> Scalar {
  let mut acc = Scalar::ZERO;

  for (var, coeff) in terms.iter() {
    let mut tmp = match var.get_unchecked() {
      Index::Input(index) => inputs[index].0,
      Index::Aux(index) => aux[index].0,
    };

    tmp.mul_assign(coeff);
    acc.add_assign(&tmp);
  }

  acc
}

impl<Scalar: PrimeField> Default for TestConstraintSystem<Scalar> {
  fn default() -> Self {
    let mut map = HashMap::new();
    map.insert("ONE".into(), NamedObject::Var);

    TestConstraintSystem {
      named_objects: map,
      current_namespace: vec![],
      constraints: vec![],
      inputs: vec![(Scalar::ONE, "ONE".into())],
      aux: vec![],
    }
  }
}

impl<Scalar: PrimeField> TestConstraintSystem<Scalar> {
  /// Create a new test constraint system.
  pub fn new() -> Self {
    Default::default()
  }

  /// Get the number of constraints
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Get path which is unsatisfied
  pub fn which_is_unsatisfied(&self) -> Option<&str> {
    for (a, b, c, path) in &self.constraints {
      let mut a = eval_lc::<Scalar>(a, &self.inputs, &self.aux);
      let b = eval_lc::<Scalar>(b, &self.inputs, &self.aux);
      let c = eval_lc::<Scalar>(c, &self.inputs, &self.aux);

      a.mul_assign(&b);

      if a != c {
        return Some(path);
      }
    }

    None
  }

  /// Check if the constraint system is satisfied.
  pub fn is_satisfied(&self) -> bool {
    match self.which_is_unsatisfied() {
      Some(b) => {
        #[allow(clippy::print_stdout)]
        {
          println!("fail: {b:?}");
        }
        false
      }
      None => true,
    }
  }

  fn set_named_obj(&mut self, path: String, to: NamedObject) {
    assert!(
      !self.named_objects.contains_key(&path),
      "tried to create object at existing path: {path}"
    );

    self.named_objects.insert(path, to);
  }
}

fn compute_path(ns: &[String], this: &str) -> String {
  assert!(
    !this.chars().any(|a| a == '/'),
    "'/' is not allowed in names"
  );

  if ns.is_empty() {
    return this.to_string();
  }

  let name = ns.join("/");
  format!("{name}/{this}")
}

impl<Scalar: PrimeField> ConstraintSystem<Scalar> for TestConstraintSystem<Scalar> {
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let index = self.aux.len();
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.aux.push((f()?, path.clone()));
    let var = Variable::new_unchecked(Index::Aux(index));
    self.set_named_obj(path, NamedObject::Var);

    Ok(var)
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let index = self.inputs.len();
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.inputs.push((f()?, path.clone()));
    let var = Variable::new_unchecked(Index::Input(index));
    self.set_named_obj(path, NamedObject::Var);

    Ok(var)
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
  {
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.set_named_obj(path.clone(), NamedObject::Constraint);

    let a = a(LinearCombination::zero());
    let b = b(LinearCombination::zero());
    let c = c(LinearCombination::zero());

    self.constraints.push((a, b, c, path));
  }

  fn push_namespace<NR, N>(&mut self, name_fn: N)
  where
    NR: Into<String>,
    N: FnOnce() -> NR,
  {
    let name = name_fn().into();
    let path = compute_path(&self.current_namespace, &name);
    self.set_named_obj(path, NamedObject::Namespace);
    self.current_namespace.push(name);
  }

  fn pop_namespace(&mut self) {
    assert!(self.current_namespace.pop().is_some());
  }

  fn get_root(&mut self) -> &mut Self::Root {
    self
  }
}
