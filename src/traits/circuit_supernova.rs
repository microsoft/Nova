//! This module defines traits that a supernova step function must implement
use bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;

#[cfg(feature = "supernova")]
/// A helper trait for a step of the incremental computation for SuperNova (i.e., circuit for F)
pub trait StepCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Return the the number of inputs or outputs of each step
  /// (this method is called only at circuit synthesis time)
  /// `synthesize` and `output` methods are expected to take as
  /// input a vector of size equal to arity and output a vector of size equal to arity
  fn arity(&self) -> usize;

  /// Sythesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step pc_{i+1} and z_{i+1}
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: &AllocatedNum<F>,
    z: &[AllocatedNum<F>],
  ) -> Result<(AllocatedNum<F>, Vec<AllocatedNum<F>>), SynthesisError>;

  /// return the output and next program counter of the step when provided with the step's input
  fn output(&self, z: &[F]) -> (F, Vec<F>);
}

/// A trivial step circuit that simply returns the input
#[derive(Clone, Debug, Default)]
pub struct TrivialTestCircuit<F: PrimeField> {
  _p: PhantomData<F>,
  rom_size: usize,
}

impl<F> TrivialTestCircuit<F>
where
  F: PrimeField,
{
  /// new TrivialTestCircuit
  pub fn new(rom_size: usize) -> Self {
    TrivialTestCircuit {
      rom_size,
      _p: PhantomData,
    }
  }
}

impl<F> StepCircuit<F> for TrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1 + 1 + self.rom_size // value + pc + rom[].len()
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    _pc_counter: &AllocatedNum<F>,
    z: &[AllocatedNum<F>],
  ) -> Result<(AllocatedNum<F>, Vec<AllocatedNum<F>>), SynthesisError> {
    Ok((z[1].clone(), z.to_vec()))
  }

  fn output(&self, z: &[F]) -> (F, Vec<F>) {
    (z[1], z.to_vec())
  }
}
