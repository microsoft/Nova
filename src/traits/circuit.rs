//! This module defines traits that a step function must implement
use crate::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
#[cfg(not(feature = "std"))]
use crate::prelude::*;
use core::marker::PhantomData;
use ff::PrimeField;

/// A helper trait for a step of the incremental computation (i.e., circuit for F)
pub trait StepCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Return the number of inputs or outputs of each step
  /// (this method is called only at circuit synthesis time)
  /// `synthesize` and `output` methods are expected to take as
  /// input a vector of size equal to arity and output a vector of size equal to arity
  fn arity(&self) -> usize;

  /// Synthesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step `z_{i+1}`
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError>;
}

/// A trivial step circuit that simply returns the input
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TrivialCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F: PrimeField> StepCircuit<F> for TrivialCircuit<F> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    Ok(z.to_vec())
  }
}

/// A non-trivial step circuit that repeats the squaring operation `num_cons` times
#[derive(Clone, Debug, Default)]
pub struct NonTrivialCircuit<F: PrimeField> {
  num_cons: usize,
  _p: PhantomData<F>,
}

impl<F: PrimeField> NonTrivialCircuit<F> {
  /// Create a new non-trivial circuit that repeats the squaring operation `num_cons` times
  pub fn new(num_cons: usize) -> Self {
    Self {
      num_cons,
      _p: PhantomData,
    }
  }
}
impl<F: PrimeField> StepCircuit<F> for NonTrivialCircuit<F> {
  fn arity(&self) -> usize {
    1
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    z: &[AllocatedNum<F>],
  ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
    // Consider an equation: `x^2 = y`, where `x` and `y` are respectively the input and output.
    let mut x = z[0].clone();
    let mut y = x.clone();
    for i in 0..self.num_cons {
      y = x.square(cs.namespace(|| format!("x_sq_{i}")))?;
      x = y.clone();
    }
    Ok(vec![y])
  }
}
