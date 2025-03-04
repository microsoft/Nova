#[cfg(not(feature = "std"))]
use crate::prelude::*;
use ff::PrimeField;
use serde::{Deserialize, Serialize};
#[cfg(feature = "std")]
use std::ops::{Add, Sub};

/// Represents a variable in our constraint system.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Variable(pub Index);

impl Variable {
  /// This constructs a variable with an arbitrary index.
  /// Circuit implementations are not recommended to use this.
  pub fn new_unchecked(idx: Index) -> Variable {
    Variable(idx)
  }

  /// This returns the index underlying the variable.
  /// Circuit implementations are not recommended to use this.
  pub fn get_unchecked(&self) -> Index {
    self.0
  }
}

/// Represents the index of either an input variable or
/// auxiliary variable.
#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash, Serialize, Deserialize)]
pub enum Index {
  /// Public input variable
  Input(usize),
  /// Private auxiliary variable
  Aux(usize),
}

/// This represents a linear combination of some variables, with coefficients
/// in the scalar field of a pairing-friendly elliptic curve group.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCombination<Scalar: PrimeField> {
  inputs: Indexer<Scalar>,
  aux: Indexer<Scalar>,
}

#[derive(Clone, Debug, PartialEq)]
struct Indexer<T> {
  /// Stores a list of `T` indexed by the number in the first slot of the tuple.
  values: Vec<(usize, T)>,
  /// `(index, key)` of the last insertion operation. Used to optimize consecutive operations
  last_inserted: Option<(usize, usize)>,
}

impl<T> Default for Indexer<T> {
  fn default() -> Self {
    Indexer {
      values: Vec::new(),
      last_inserted: None,
    }
  }
}

impl<T> Indexer<T> {
  pub fn from_value(index: usize, value: T) -> Self {
    Indexer {
      values: vec![(index, value)],
      last_inserted: Some((0, index)),
    }
  }

  pub fn iter(&self) -> impl Iterator<Item = (&usize, &T)> + '_ {
    #[allow(clippy::map_identity)]
    self.values.iter().map(|(key, value)| (key, value))
  }

  pub fn iter_mut(&mut self) -> impl Iterator<Item = (&usize, &mut T)> + '_ {
    self.values.iter_mut().map(|(key, value)| (&*key, value))
  }

  pub fn insert_or_update<F, G>(&mut self, key: usize, insert: F, update: G)
  where
    F: FnOnce() -> T,
    G: FnOnce(&mut T),
  {
    if let Some((last_index, last_key)) = self.last_inserted {
      // Optimization to avoid doing binary search on inserts & updates that are linear, meaning
      // they are adding a consecutive values.
      if last_key == key {
        // update the same key again
        update(&mut self.values[last_index].1);
        return;
      } else if last_key + 1 == key {
        // optimization for follow on updates
        let i = last_index + 1;
        if i >= self.values.len() {
          // insert at the end
          self.values.push((key, insert()));
          self.last_inserted = Some((i, key));
        } else if self.values[i].0 == key {
          // update
          update(&mut self.values[i].1);
        } else {
          // insert
          self.values.insert(i, (key, insert()));
          self.last_inserted = Some((i, key));
        }
        return;
      }
    }
    match self.values.binary_search_by_key(&key, |(k, _)| *k) {
      Ok(i) => {
        update(&mut self.values[i].1);
      }
      Err(i) => {
        self.values.insert(i, (key, insert()));
        self.last_inserted = Some((i, key));
      }
    }
  }

  pub fn len(&self) -> usize {
    self.values.len()
  }

  pub fn is_empty(&self) -> bool {
    self.values.is_empty()
  }
}

impl<Scalar: PrimeField> Default for LinearCombination<Scalar> {
  fn default() -> Self {
    Self::zero()
  }
}

impl<Scalar: PrimeField> LinearCombination<Scalar> {
  /// This returns a zero [`LinearCombination`].
  pub fn zero() -> LinearCombination<Scalar> {
    LinearCombination {
      inputs: Default::default(),
      aux: Default::default(),
    }
  }

  /// Create a [`LinearCombination`] from a variable and coefficient.
  pub fn from_coeff(var: Variable, coeff: Scalar) -> Self {
    match var {
      Variable(Index::Input(i)) => Self {
        inputs: Indexer::from_value(i, coeff),
        aux: Default::default(),
      },
      Variable(Index::Aux(i)) => Self {
        inputs: Default::default(),
        aux: Indexer::from_value(i, coeff),
      },
    }
  }

  /// Create a [`LinearCombination`] from a variable.
  pub fn from_variable(var: Variable) -> Self {
    Self::from_coeff(var, Scalar::ONE)
  }

  /// Iter for the [`LinearCombination`].
  pub fn iter(&self) -> impl Iterator<Item = (Variable, &Scalar)> + '_ {
    self
      .inputs
      .iter()
      .map(|(k, v)| (Variable(Index::Input(*k)), v))
      .chain(self.aux.iter().map(|(k, v)| (Variable(Index::Aux(*k)), v)))
  }

  /// Iter inputs for the [`LinearCombination`]
  #[inline]
  pub fn iter_inputs(&self) -> impl Iterator<Item = (&usize, &Scalar)> + '_ {
    self.inputs.iter()
  }

  /// Iter aux for the [`LinearCombination`]
  #[inline]
  pub fn iter_aux(&self) -> impl Iterator<Item = (&usize, &Scalar)> + '_ {
    self.aux.iter()
  }

  /// Iter mut for the [`LinearCombination`]
  pub fn iter_mut(&mut self) -> impl Iterator<Item = (Variable, &mut Scalar)> + '_ {
    self
      .inputs
      .iter_mut()
      .map(|(k, v)| (Variable(Index::Input(*k)), v))
      .chain(
        self
          .aux
          .iter_mut()
          .map(|(k, v)| (Variable(Index::Aux(*k)), v)),
      )
  }

  #[inline]
  fn add_assign_unsimplified_input(&mut self, new_var: usize, coeff: Scalar) {
    self
      .inputs
      .insert_or_update(new_var, || coeff, |val| *val += coeff);
  }

  #[inline]
  fn add_assign_unsimplified_aux(&mut self, new_var: usize, coeff: Scalar) {
    self
      .aux
      .insert_or_update(new_var, || coeff, |val| *val += coeff);
  }

  /// Add unsimplified
  pub fn add_unsimplified(mut self, (coeff, var): (Scalar, Variable)) -> LinearCombination<Scalar> {
    match var.0 {
      Index::Input(new_var) => {
        self.add_assign_unsimplified_input(new_var, coeff);
      }
      Index::Aux(new_var) => {
        self.add_assign_unsimplified_aux(new_var, coeff);
      }
    }

    self
  }

  #[inline]
  fn sub_assign_unsimplified_input(&mut self, new_var: usize, coeff: Scalar) {
    self.add_assign_unsimplified_input(new_var, -coeff);
  }

  #[inline]
  fn sub_assign_unsimplified_aux(&mut self, new_var: usize, coeff: Scalar) {
    self.add_assign_unsimplified_aux(new_var, -coeff);
  }

  /// Sub unsimplified
  pub fn sub_unsimplified(mut self, (coeff, var): (Scalar, Variable)) -> LinearCombination<Scalar> {
    match var.0 {
      Index::Input(new_var) => {
        self.sub_assign_unsimplified_input(new_var, coeff);
      }
      Index::Aux(new_var) => {
        self.sub_assign_unsimplified_aux(new_var, coeff);
      }
    }

    self
  }

  /// len of the [`LinearCombination`]
  pub fn len(&self) -> usize {
    self.inputs.len() + self.aux.len()
  }

  /// Check if the [`LinearCombination`] is empty
  pub fn is_empty(&self) -> bool {
    self.inputs.is_empty() && self.aux.is_empty()
  }

  /// Evaluate the [`LinearCombination`] with the given input and aux assignments.
  pub fn eval(&self, input_assignment: &[Scalar], aux_assignment: &[Scalar]) -> Scalar {
    let mut acc = Scalar::ZERO;

    let one = Scalar::ONE;

    for (index, coeff) in self.iter_inputs() {
      let mut tmp = input_assignment[*index];
      if coeff != &one {
        tmp *= coeff;
      }
      acc += tmp;
    }

    for (index, coeff) in self.iter_aux() {
      let mut tmp = aux_assignment[*index];
      if coeff != &one {
        tmp *= coeff;
      }
      acc += tmp;
    }

    acc
  }
}

impl<Scalar: PrimeField> Add<(Scalar, Variable)> for LinearCombination<Scalar> {
  type Output = LinearCombination<Scalar>;

  fn add(self, (coeff, var): (Scalar, Variable)) -> LinearCombination<Scalar> {
    self.add_unsimplified((coeff, var))
  }
}

impl<Scalar: PrimeField> Sub<(Scalar, Variable)> for LinearCombination<Scalar> {
  type Output = LinearCombination<Scalar>;

  #[allow(clippy::suspicious_arithmetic_impl)]
  fn sub(self, (coeff, var): (Scalar, Variable)) -> LinearCombination<Scalar> {
    self.sub_unsimplified((coeff, var))
  }
}

impl<Scalar: PrimeField> Add<Variable> for LinearCombination<Scalar> {
  type Output = LinearCombination<Scalar>;

  fn add(self, other: Variable) -> LinearCombination<Scalar> {
    self + (Scalar::ONE, other)
  }
}

impl<Scalar: PrimeField> Sub<Variable> for LinearCombination<Scalar> {
  type Output = LinearCombination<Scalar>;

  fn sub(self, other: Variable) -> LinearCombination<Scalar> {
    self - (Scalar::ONE, other)
  }
}

impl<'a, Scalar: PrimeField> Add<&'a LinearCombination<Scalar>> for LinearCombination<Scalar> {
  type Output = LinearCombination<Scalar>;

  fn add(mut self, other: &'a LinearCombination<Scalar>) -> LinearCombination<Scalar> {
    for (var, val) in other.inputs.iter() {
      self.add_assign_unsimplified_input(*var, *val);
    }

    for (var, val) in other.aux.iter() {
      self.add_assign_unsimplified_aux(*var, *val);
    }

    self
  }
}

impl<'a, Scalar: PrimeField> Sub<&'a LinearCombination<Scalar>> for LinearCombination<Scalar> {
  type Output = LinearCombination<Scalar>;

  fn sub(mut self, other: &'a LinearCombination<Scalar>) -> LinearCombination<Scalar> {
    for (var, val) in other.inputs.iter() {
      self.sub_assign_unsimplified_input(*var, *val);
    }

    for (var, val) in other.aux.iter() {
      self.sub_assign_unsimplified_aux(*var, *val);
    }

    self
  }
}

impl<'a, Scalar: PrimeField> Add<(Scalar, &'a LinearCombination<Scalar>)>
  for LinearCombination<Scalar>
{
  type Output = LinearCombination<Scalar>;

  fn add(
    mut self,
    (coeff, other): (Scalar, &'a LinearCombination<Scalar>),
  ) -> LinearCombination<Scalar> {
    for (var, val) in other.inputs.iter() {
      self.add_assign_unsimplified_input(*var, *val * coeff);
    }

    for (var, val) in other.aux.iter() {
      self.add_assign_unsimplified_aux(*var, *val * coeff);
    }

    self
  }
}

impl<'a, Scalar: PrimeField> Sub<(Scalar, &'a LinearCombination<Scalar>)>
  for LinearCombination<Scalar>
{
  type Output = LinearCombination<Scalar>;

  fn sub(
    mut self,
    (coeff, other): (Scalar, &'a LinearCombination<Scalar>),
  ) -> LinearCombination<Scalar> {
    for (var, val) in other.inputs.iter() {
      self.sub_assign_unsimplified_input(*var, *val * coeff);
    }

    for (var, val) in other.aux.iter() {
      self.sub_assign_unsimplified_aux(*var, *val * coeff);
    }

    self
  }
}
