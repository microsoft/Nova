//! Test constraint system for use in tests.

#![allow(dead_code)]
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::collections::HashMap;

use super::Comparable;
use crate::frontend::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

use ff::PrimeField;

#[derive(Debug)]
enum NamedObject {
  Constraint(usize),
  Var(Variable),
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
  precommitted: Vec<(Scalar, String)>,
}

#[derive(Clone, Copy)]
struct OrderedVariable(Variable);

impl Eq for OrderedVariable {}
impl PartialEq for OrderedVariable {
  fn eq(&self, other: &OrderedVariable) -> bool {
    match (self.0.get_unchecked(), other.0.get_unchecked()) {
      (Index::Input(ref a), Index::Input(ref b)) => a == b,
      (Index::Aux(ref a), Index::Aux(ref b)) => a == b,
      _ => false,
    }
  }
}
impl PartialOrd for OrderedVariable {
  fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
    Some(self.cmp(other))
  }
}
impl Ord for OrderedVariable {
  fn cmp(&self, other: &Self) -> Ordering {
    match (self.0.get_unchecked(), other.0.get_unchecked()) {
      (Index::Input(ref a), Index::Input(ref b)) => a.cmp(b),
      (Index::Aux(ref a), Index::Aux(ref b)) => a.cmp(b),
      (Index::Precommitted(ref a), Index::Precommitted(ref b)) => a.cmp(b),
      (Index::Input(_), Index::Aux(_)) => Ordering::Less,
      (Index::Aux(_), Index::Input(_)) => Ordering::Greater,
      (Index::Precommitted(_), Index::Aux(_)) => Ordering::Less,
      (Index::Aux(_), Index::Precommitted(_)) => Ordering::Greater,
      (Index::Input(_), Index::Precommitted(_)) => Ordering::Less,
      (Index::Precommitted(_), Index::Input(_)) => Ordering::Greater,
    }
  }
}

fn proc_lc<Scalar: PrimeField>(
  terms: &LinearCombination<Scalar>,
) -> BTreeMap<OrderedVariable, Scalar> {
  let mut map = BTreeMap::new();
  for (var, &coeff) in terms.iter() {
    map
      .entry(OrderedVariable(var))
      .or_insert_with(|| Scalar::ZERO)
      .add_assign(&coeff);
  }

  // Remove terms that have a zero coefficient to normalize
  let mut to_remove = vec![];
  for (var, coeff) in map.iter() {
    if coeff.is_zero().into() {
      to_remove.push(*var)
    }
  }

  for var in to_remove {
    map.remove(&var);
  }

  map
}

fn _eval_lc2<Scalar: PrimeField>(
  terms: &LinearCombination<Scalar>,
  inputs: &[Scalar],
  aux: &[Scalar],
  precommitted: &[Scalar],
) -> Scalar {
  let mut acc = Scalar::ZERO;

  for (var, coeff) in terms.iter() {
    let mut tmp = match var.get_unchecked() {
      Index::Input(index) => inputs[index],
      Index::Aux(index) => aux[index],
      Index::Precommitted(index) => precommitted[index],
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
  precommitted: &[(Scalar, String)],
) -> Scalar {
  let mut acc = Scalar::ZERO;

  for (var, coeff) in terms.iter() {
    let mut tmp = match var.get_unchecked() {
      Index::Input(index) => inputs[index].0,
      Index::Aux(index) => aux[index].0,
      Index::Precommitted(index) => precommitted[index].0,
    };

    tmp.mul_assign(coeff);
    acc.add_assign(&tmp);
  }

  acc
}

impl<Scalar: PrimeField> Default for TestConstraintSystem<Scalar> {
  fn default() -> Self {
    let mut map = HashMap::new();
    map.insert(
      "ONE".into(),
      NamedObject::Var(TestConstraintSystem::<Scalar>::one()),
    );

    TestConstraintSystem {
      named_objects: map,
      current_namespace: vec![],
      constraints: vec![],
      inputs: vec![(Scalar::ONE, "ONE".into())],
      aux: vec![],
      precommitted: vec![],
    }
  }
}

impl<Scalar: PrimeField> TestConstraintSystem<Scalar> {
  /// Create a new test constraint system.
  pub fn new() -> Self {
    Default::default()
  }

  /// Get scalar inputs
  pub fn scalar_inputs(&self) -> Vec<Scalar> {
    self
      .inputs
      .iter()
      .map(|(scalar, _string)| *scalar)
      .collect()
  }

  /// Get scalar aux
  pub fn scalar_aux(&self) -> Vec<Scalar> {
    self.aux.iter().map(|(scalar, _string)| *scalar).collect()
  }

  /// Pretty print
  pub fn pretty_print_list(&self) -> Vec<String> {
    let mut result = Vec::new();

    for input in &self.inputs {
      result.push(format!("INPUT {}", input.1));
    }
    for aux in &self.aux {
      result.push(format!("AUX {}", aux.1));
    }

    for (_a, _b, _c, name) in &self.constraints {
      result.push(name.to_string());
    }

    result
  }

  /// Pretty print
  pub fn pretty_print(&self) -> String {
    let res = self.pretty_print_list();

    res.join("\n")
  }

  /// Get path which is unsatisfied
  pub fn which_is_unsatisfied(&self) -> Option<&str> {
    for (a, b, c, path) in &self.constraints {
      let mut a = eval_lc::<Scalar>(a, &self.inputs, &self.aux, &self.precommitted);
      let b = eval_lc::<Scalar>(b, &self.inputs, &self.aux, &self.precommitted);
      let c = eval_lc::<Scalar>(c, &self.inputs, &self.aux, &self.precommitted);

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
        println!("fail: {:?}", b);
        false
      }
      None => true,
    }
    // self.which_is_unsatisfied().is_none()
  }

  /// Return the number of constraints in the constraint system.
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Create a new variable in the constraint system.
  pub fn set(&mut self, path: &str, to: Scalar) {
    match self.named_objects.get(path) {
      Some(NamedObject::Var(v)) => match v.get_unchecked() {
        Index::Input(index) => self.inputs[index].0 = to,
        Index::Aux(index) => self.aux[index].0 = to,
        Index::Precommitted(index) => self.precommitted[index].0 = to,
      },
      Some(e) => panic!(
        "tried to set path `{}` to value, but `{:?}` already exists there.",
        path, e
      ),
      _ => panic!("no variable exists at path: {}", path),
    }
  }

  /// Verify expected vec == self.inputs
  pub fn verify(&self, expected: &[Scalar]) -> bool {
    assert_eq!(expected.len() + 1, self.inputs.len());
    for (a, b) in self.inputs.iter().skip(1).zip(expected.iter()) {
      if &a.0 != b {
        return false;
      }
    }

    true
  }

  /// Return number of inputs in the constraint system.
  pub fn num_inputs(&self) -> usize {
    self.inputs.len()
  }

  /// Get an input variable.
  pub fn get_input(&mut self, index: usize, path: &str) -> Scalar {
    let (assignment, name) = self.inputs[index].clone();

    assert_eq!(path, name);

    assignment
  }

  /// Get inputs
  pub fn get_inputs(&self) -> &[(Scalar, String)] {
    &self.inputs[..]
  }

  /// Get Scalar from path
  pub fn get(&mut self, path: &str) -> Scalar {
    match self.named_objects.get(path) {
      Some(NamedObject::Var(v)) => match v.get_unchecked() {
        Index::Input(index) => self.inputs[index].0,
        Index::Aux(index) => self.aux[index].0,
        Index::Precommitted(index) => self.precommitted[index].0,
      },
      Some(e) => panic!(
        "tried to get value of path `{}`, but `{:?}` exists there (not a variable)",
        path, e
      ),
      _ => panic!("no variable exists at path: {}", path),
    }
  }

  fn set_named_obj(&mut self, path: String, to: NamedObject) {
    assert!(
      !self.named_objects.contains_key(&path),
      "tried to create object at existing path: {}",
      path
    );

    self.named_objects.insert(path, to);
  }
}

impl<Scalar: PrimeField> Comparable<Scalar> for TestConstraintSystem<Scalar> {
  fn num_inputs(&self) -> usize {
    self.num_inputs()
  }
  fn num_constraints(&self) -> usize {
    self.num_constraints()
  }

  fn aux(&self) -> Vec<String> {
    self
      .aux
      .iter()
      .map(|(_scalar, string)| string.to_string())
      .collect()
  }

  fn inputs(&self) -> Vec<String> {
    self
      .inputs
      .iter()
      .map(|(_scalar, string)| string.to_string())
      .collect()
  }

  fn constraints(&self) -> &[crate::frontend::util_cs::Constraint<Scalar>] {
    &self.constraints
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
  format!("{}/{}", name, this)
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
    self.set_named_obj(path, NamedObject::Var(var));

    Ok(var)
  }

  fn alloc_precommitted<F, A, AR>(
    &mut self,
    annotation: A,
    f: F,
  ) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let index = self.precommitted.len();
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.precommitted.push((f()?, path.clone()));
    let var = Variable::new_unchecked(Index::Precommitted(index));
    self.set_named_obj(path, NamedObject::Var(var));

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
    self.set_named_obj(path, NamedObject::Var(var));

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
    let index = self.constraints.len();
    self.set_named_obj(path.clone(), NamedObject::Constraint(index));

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
