//! Support for generating R1CS shape using bellpepper.
//! `TestShapeCS` implements a superset of `ShapeCS`, adding non-trivial namespace support for use in testing.

use std::{
  cmp::Ordering,
  collections::{BTreeMap, HashMap},
};

use crate::traits::Engine;
use bellpepper_core::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use core::fmt::Write;
use ff::{Field, PrimeField};

#[derive(Clone, Copy)]
struct OrderedVariable(Variable);

#[allow(unused)]
#[derive(Debug)]
enum NamedObject {
  Constraint(usize),
  Var(Variable),
  Namespace,
}

impl Eq for OrderedVariable {}
impl PartialEq for OrderedVariable {
  fn eq(&self, other: &OrderedVariable) -> bool {
    match (self.0.get_unchecked(), other.0.get_unchecked()) {
      (Index::Input(ref a), Index::Input(ref b)) | (Index::Aux(ref a), Index::Aux(ref b)) => a == b,
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
      (Index::Input(ref a), Index::Input(ref b)) | (Index::Aux(ref a), Index::Aux(ref b)) => {
        a.cmp(b)
      }
      (Index::Input(_), Index::Aux(_)) => Ordering::Less,
      (Index::Aux(_), Index::Input(_)) => Ordering::Greater,
    }
  }
}

/// `TestShapeCS` is a `ConstraintSystem` for creating `R1CSShape`s for a circuit.
pub struct TestShapeCS<E: Engine> {
  named_objects: HashMap<String, NamedObject>,
  current_namespace: Vec<String>,
  /// All constraints added to the `TestShapeCS`.
  pub constraints: Vec<(
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
    LinearCombination<E::Scalar>,
    String,
  )>,
  inputs: Vec<String>,
  aux: Vec<String>,
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

impl<E: Engine> TestShapeCS<E>
where
  E::Scalar: PrimeField,
{
  #[allow(unused)]
  /// Create a new, default `TestShapeCS`,
  pub fn new() -> Self {
    TestShapeCS::default()
  }

  /// Returns the number of constraints defined for this `TestShapeCS`.
  pub fn num_constraints(&self) -> usize {
    self.constraints.len()
  }

  /// Returns the number of inputs defined for this `TestShapeCS`.
  pub fn num_inputs(&self) -> usize {
    self.inputs.len()
  }

  /// Returns the number of aux inputs defined for this `TestShapeCS`.
  pub fn num_aux(&self) -> usize {
    self.aux.len()
  }

  /// Print all public inputs, aux inputs, and constraint names.
  #[allow(dead_code)]
  pub fn pretty_print_list(&self) -> Vec<String> {
    let mut result = Vec::new();

    for input in &self.inputs {
      result.push(format!("INPUT {input}"));
    }
    for aux in &self.aux {
      result.push(format!("AUX {aux}"));
    }

    for (_a, _b, _c, name) in &self.constraints {
      result.push(name.to_string());
    }

    result
  }

  /// Print all iputs and a detailed representation of each constraint.
  #[allow(dead_code)]
  pub fn pretty_print(&self) -> String {
    let mut s = String::new();

    for input in &self.inputs {
      writeln!(s, "INPUT {}", &input).unwrap()
    }

    let negone = -<E::Scalar>::ONE;

    let powers_of_two = (0..E::Scalar::NUM_BITS)
      .map(|i| E::Scalar::from(2u64).pow_vartime([u64::from(i)]))
      .collect::<Vec<_>>();

    let pp = |s: &mut String, lc: &LinearCombination<E::Scalar>| {
      s.push('(');
      let mut is_first = true;
      for (var, coeff) in proc_lc::<E::Scalar>(lc) {
        if coeff == negone {
          s.push_str(" - ")
        } else if !is_first {
          s.push_str(" + ")
        }
        is_first = false;

        if coeff != <E::Scalar>::ONE && coeff != negone {
          for (i, x) in powers_of_two.iter().enumerate() {
            if x == &coeff {
              write!(s, "2^{i} . ").unwrap();
              break;
            }
          }

          write!(s, "{coeff:?} . ").unwrap()
        }

        match var.0.get_unchecked() {
          Index::Input(i) => {
            write!(s, "`I{}`", &self.inputs[i]).unwrap();
          }
          Index::Aux(i) => {
            write!(s, "`A{}`", &self.aux[i]).unwrap();
          }
        }
      }
      if is_first {
        // Nothing was visited, print 0.
        s.push('0');
      }
      s.push(')');
    };

    for (a, b, c, name) in &self.constraints {
      s.push('\n');

      write!(s, "{name}: ").unwrap();
      pp(&mut s, a);
      write!(s, " * ").unwrap();
      pp(&mut s, b);
      s.push_str(" = ");
      pp(&mut s, c);
    }

    s.push('\n');

    s
  }

  /// Associate `NamedObject` with `path`.
  /// `path` must not already have an associated object.
  fn set_named_obj(&mut self, path: String, to: NamedObject) {
    assert!(
      !self.named_objects.contains_key(&path),
      "tried to create object at existing path: {path}"
    );

    self.named_objects.insert(path, to);
  }
}

impl<E: Engine> Default for TestShapeCS<E> {
  fn default() -> Self {
    let mut map = HashMap::new();
    map.insert("ONE".into(), NamedObject::Var(TestShapeCS::<E>::one()));
    TestShapeCS {
      named_objects: map,
      current_namespace: vec![],
      constraints: vec![],
      inputs: vec![String::from("ONE")],
      aux: vec![],
    }
  }
}

impl<E: Engine> ConstraintSystem<E::Scalar> for TestShapeCS<E>
where
  E::Scalar: PrimeField,
{
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.aux.push(path);

    Ok(Variable::new_unchecked(Index::Aux(self.aux.len() - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<E::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.inputs.push(path);

    Ok(Variable::new_unchecked(Index::Input(self.inputs.len() - 1)))
  }

  fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
  where
    A: FnOnce() -> AR,
    AR: Into<String>,
    LA: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LB: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
    LC: FnOnce(LinearCombination<E::Scalar>) -> LinearCombination<E::Scalar>,
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

fn compute_path(ns: &[String], this: &str) -> String {
  assert!(
    !this.chars().any(|a| a == '/'),
    "'/' is not allowed in names"
  );

  let mut name = String::new();

  let mut needs_separation = false;
  for ns in ns.iter().chain(Some(this.to_string()).iter()) {
    if needs_separation {
      name += "/";
    }

    name += ns;
    needs_separation = true;
  }

  name
}
