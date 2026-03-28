//! Support for efficiently generating R1CS witness using bellperson.

use ff::PrimeField;

use crate::frontend::{ConstraintSystem, Index, LinearCombination, SynthesisError, Variable};

/// A [`ConstraintSystem`] trait
pub trait SizedWitness<Scalar: PrimeField> {
  /// Returns the number of constraints in the constraint system
  fn num_constraints(&self) -> usize;
  /// Returns the number of inputs in the constraint system
  fn num_inputs(&self) -> usize;
  /// Returns the number of auxiliary variables in the constraint system
  fn num_aux(&self) -> usize;

  /// Generate a witness for the constraint system
  fn generate_witness_into(&mut self, aux: &mut [Scalar], inputs: &mut [Scalar]) -> Scalar;

  /// Generate a witness for the constraint system
  fn generate_witness_into_cs<CS: ConstraintSystem<Scalar>>(&mut self, cs: &mut CS) -> Scalar {
    assert!(cs.is_witness_generator());

    let aux_count = self.num_aux();
    let inputs_count = self.num_inputs();

    let (aux, inputs) = cs.allocate_empty(aux_count, inputs_count);

    assert_eq!(aux.len(), aux_count);
    assert_eq!(inputs.len(), inputs_count);

    self.generate_witness_into(aux, inputs)
  }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
pub struct WitnessCS<Scalar>
where
  Scalar: PrimeField,
{
  // Assignments of variables
  pub(crate) input_assignment: Vec<Scalar>,
  pub(crate) aux_assignment: Vec<Scalar>,
  // Inline boolean tracking: avoids expensive post-hoc scan of witness data.
  // aux_is_bool[i] bit j = 1 iff aux_assignment[32*i + j] is 0 or 1.
  // aux_bool_val[i] bit j = 1 iff aux_assignment[32*i + j] == 1.
  pub(crate) aux_is_bool: Vec<u32>,
  pub(crate) aux_bool_val: Vec<u32>,
  // Ranges of aux entries added via allocate_empty (need deferred classification)
  unclassified_ranges: Vec<(usize, usize)>,
}

impl<Scalar> WitnessCS<Scalar>
where
  Scalar: PrimeField,
{
  /// Create a new WitnessCS with pre-allocated capacity for aux and input variables.
  /// This avoids repeated reallocations during synthesis for large circuits.
  pub fn with_capacity(aux_capacity: usize, input_capacity: usize) -> Self {
    let mut input_assignment = Vec::with_capacity(input_capacity + 1);
    input_assignment.push(Scalar::ONE);
    let words = (aux_capacity + 31) / 32;
    Self {
      input_assignment,
      aux_assignment: Vec::with_capacity(aux_capacity),
      aux_is_bool: Vec::with_capacity(words),
      aux_bool_val: Vec::with_capacity(words),
      unclassified_ranges: Vec::new(),
    }
  }

  /// Clear the assignments while retaining allocated capacity.
  /// This allows reusing the same WitnessCS across multiple synthesis calls
  /// without reallocating.
  pub fn clear(&mut self) {
    self.input_assignment.clear();
    self.input_assignment.push(Scalar::ONE);
    self.aux_assignment.clear();
    self.aux_is_bool.clear();
    self.aux_bool_val.clear();
    self.unclassified_ranges.clear();
  }

  /// Take the aux_assignment vector and replace it with the provided buffer.
  /// The provided buffer is cleared before being installed.
  /// This enables zero-copy witness creation by swapping buffers.
  pub fn swap_aux(&mut self, mut buf: Vec<Scalar>) -> Vec<Scalar> {
    buf.clear();
    std::mem::swap(&mut self.aux_assignment, &mut buf);
    buf // returns the old aux_assignment data
  }

  /// Take the inline boolean tracking bitfields.
  /// Finalizes any entries added via allocate_empty before returning.
  /// Returns (is_bool, bool_val) where:
  /// - is_bool[i] bit j = 1 iff aux[32*i+j] is boolean (0 or 1)
  /// - bool_val[i] bit j = 1 iff aux[32*i+j] == 1
  pub fn take_bool_bitfields(&mut self) -> (Vec<u32>, Vec<u32>) {
    for &(start, end) in &self.unclassified_ranges {
      for idx in start..end {
        let word = idx >> 5;
        let bit = idx & 31;
        let val = &self.aux_assignment[idx];
        self.aux_is_bool[word] &= !(1u32 << bit);
        self.aux_bool_val[word] &= !(1u32 << bit);
        if *val == Scalar::ZERO {
          self.aux_is_bool[word] |= 1u32 << bit;
        } else if *val == Scalar::ONE {
          self.aux_is_bool[word] |= 1u32 << bit;
          self.aux_bool_val[word] |= 1u32 << bit;
        }
      }
    }
    self.unclassified_ranges.clear();
    (
      std::mem::take(&mut self.aux_is_bool),
      std::mem::take(&mut self.aux_bool_val),
    )
  }

  /// Get input assignment
  pub fn input_assignment(&self) -> &[Scalar] {
    &self.input_assignment
  }

  /// Get aux assignment
  pub fn aux_assignment(&self) -> &[Scalar] {
    &self.aux_assignment
  }
}

impl<Scalar> ConstraintSystem<Scalar> for WitnessCS<Scalar>
where
  Scalar: PrimeField,
{
  type Root = Self;

  fn new() -> Self {
    let input_assignment = vec![Scalar::ONE];

    Self {
      input_assignment,
      aux_assignment: vec![],
      aux_is_bool: vec![],
      aux_bool_val: vec![],
      unclassified_ranges: vec![],
    }
  }

  fn alloc<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
    A: FnOnce() -> AR,
    AR: Into<String>,
  {
    let val = f()?;
    let idx = self.aux_assignment.len();
    self.aux_assignment.push(val);

    let word = idx >> 5;
    let bit = idx & 31;
    if word >= self.aux_is_bool.len() {
      self.aux_is_bool.push(0u32);
      self.aux_bool_val.push(0u32);
    }
    if val == Scalar::ZERO {
      self.aux_is_bool[word] |= 1u32 << bit;
    } else if val == Scalar::ONE {
      self.aux_is_bool[word] |= 1u32 << bit;
      self.aux_bool_val[word] |= 1u32 << bit;
    }

    Ok(Variable(Index::Aux(idx)))
  }

  fn alloc_input<F, A, AR>(&mut self, _: A, f: F) -> Result<Variable, SynthesisError>
  where
    F: FnOnce() -> Result<Scalar, SynthesisError>,
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
    LA: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LB: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
    LC: FnOnce(LinearCombination<Scalar>) -> LinearCombination<Scalar>,
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

  ////////////////////////////////////////////////////////////////////////////////
  // Extensible
  fn is_extensible() -> bool {
    true
  }

  fn extend(&mut self, other: &Self) {
    self.input_assignment
            // Skip first input, which must have been a temporarily allocated one variable.
            .extend(&other.input_assignment[1..]);
    for val in &other.aux_assignment {
      let idx = self.aux_assignment.len();
      self.aux_assignment.push(*val);
      let word = idx >> 5;
      let bit = idx & 31;
      if word >= self.aux_is_bool.len() {
        self.aux_is_bool.push(0u32);
        self.aux_bool_val.push(0u32);
      }
      if *val == Scalar::ZERO {
        self.aux_is_bool[word] |= 1u32 << bit;
      } else if *val == Scalar::ONE {
        self.aux_is_bool[word] |= 1u32 << bit;
        self.aux_bool_val[word] |= 1u32 << bit;
      }
    }
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Witness generator
  fn is_witness_generator(&self) -> bool {
    true
  }

  fn extend_inputs(&mut self, new_inputs: &[Scalar]) {
    self.input_assignment.extend(new_inputs);
  }

  fn extend_aux(&mut self, new_aux: &[Scalar]) {
    for val in new_aux {
      let idx = self.aux_assignment.len();
      self.aux_assignment.push(*val);
      let word = idx >> 5;
      let bit = idx & 31;
      if word >= self.aux_is_bool.len() {
        self.aux_is_bool.push(0u32);
        self.aux_bool_val.push(0u32);
      }
      if *val == Scalar::ZERO {
        self.aux_is_bool[word] |= 1u32 << bit;
      } else if *val == Scalar::ONE {
        self.aux_is_bool[word] |= 1u32 << bit;
        self.aux_bool_val[word] |= 1u32 << bit;
      }
    }
  }

  fn allocate_empty(&mut self, aux_n: usize, inputs_n: usize) -> (&mut [Scalar], &mut [Scalar]) {
    let allocated_aux = {
      let i = self.aux_assignment.len();
      self.aux_assignment.resize(aux_n + i, Scalar::ZERO);
      // Resize bitfield arrays; record range for deferred classification
      let new_words_needed = (aux_n + i + 31) / 32;
      self.aux_is_bool.resize(new_words_needed, 0u32);
      self.aux_bool_val.resize(new_words_needed, 0u32);
      self.unclassified_ranges.push((i, i + aux_n));
      &mut self.aux_assignment[i..]
    };

    let allocated_inputs = {
      let i = self.input_assignment.len();
      self.input_assignment.resize(inputs_n + i, Scalar::ZERO);
      &mut self.input_assignment[i..]
    };

    (allocated_aux, allocated_inputs)
  }

  fn inputs_slice(&self) -> &[Scalar] {
    &self.input_assignment
  }

  fn aux_slice(&self) -> &[Scalar] {
    &self.aux_assignment
  }
}
