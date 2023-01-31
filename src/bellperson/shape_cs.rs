//! Support for generating R1CS shape using bellperson.

use std::{
  cmp::Ordering,
  collections::{BTreeMap, HashMap},
};

use crate::traits::Group;
use bellperson::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};
use core::fmt::Write;
use ff::{Field, PrimeField};

#[derive(Clone, Copy)]
struct OrderedVariable(Variable);

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
      (Index::Input(_), Index::Aux(_)) => Ordering::Less,
      (Index::Aux(_), Index::Input(_)) => Ordering::Greater,
    }
  }
}

#[allow(clippy::upper_case_acronyms)]
/// `ShapeCS` is a `ConstraintSystem` for creating `R1CSShape`s for a circuit.
pub struct ShapeCS<G: Group>
where
  G::Scalar: PrimeField + Field,
{
  named_objects: HashMap<String, NamedObject>,
  current_namespace: Vec<String>,
  #[allow(clippy::type_complexity)]
  /// All constraints added to the `ShapeCS`.
  pub constraints: Vec<(
    LinearCombination<G::Scalar>,
    LinearCombination<G::Scalar>,
    LinearCombination<G::Scalar>,
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
      .or_insert_with(Scalar::zero)
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

impl<G: Group> ShapeCS<G>
where
  G::Scalar: PrimeField,
{
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
    self.inputs.len()
  }

  /// Returns the number of aux inputs defined for this `ShapeCS`.
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

    let negone = -<G::Scalar>::one();

    let powers_of_two = (0..G::Scalar::NUM_BITS)
      .map(|i| G::Scalar::from(2u64).pow_vartime([u64::from(i)]))
      .collect::<Vec<_>>();

    let pp = |s: &mut String, lc: &LinearCombination<G::Scalar>| {
      s.push('(');
      let mut is_first = true;
      for (var, coeff) in proc_lc::<G::Scalar>(lc) {
        if coeff == negone {
          s.push_str(" - ")
        } else if !is_first {
          s.push_str(" + ")
        }
        is_first = false;

        if coeff != <G::Scalar>::one() && coeff != negone {
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

impl<G: Group> Default for ShapeCS<G>
where
  G::Scalar: PrimeField,
{
  fn default() -> Self {
    let mut map = HashMap::new();
    map.insert("ONE".into(), NamedObject::Var(ShapeCS::<G>::one()));
    ShapeCS {
      named_objects: map,
      current_namespace: vec![],
      constraints: vec![],
      inputs: vec![String::from("ONE")],
      aux: vec![],
    }
  }
}

impl<G: Group> ConstraintSystem<G::Scalar> for ShapeCS<G>
where
  G::Scalar: PrimeField,
{
  type Root = Self;

  fn alloc<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<G::Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let path = compute_path(&self.current_namespace, &annotation().into());
    self.aux.push(path);

    Ok(Variable::new_unchecked(Index::Aux(self.aux.len() - 1)))
  }

  fn alloc_input<F, A, AR>(&mut self, annotation: A, _f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<G::Scalar, SynthesisError>,
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
    LA: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
    LB: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
    LC: FnOnce(LinearCombination<G::Scalar>) -> LinearCombination<G::Scalar>,
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
