//! This module defines traits that a supernova step function must implement
use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use core::marker::PhantomData;
use ff::PrimeField;

/// A helper trait for a step of the incremental computation for `SuperNova` (i.e., circuit for F) -- to be implemented by
/// applications.
pub trait StepCircuit<F: PrimeField>: Send + Sync + Clone {
  /// Return the the number of inputs or outputs of each step
  /// (this method is called only at circuit synthesis time)
  /// `synthesize` and `output` methods are expected to take as
  /// input a vector of size equal to arity and output a vector of size equal to arity
  fn arity(&self) -> usize;

  /// Return this `StepCircuit`'s assigned index, for use when enforcing the program counter.
  fn circuit_index(&self) -> usize;

  /// Synthesize the circuit for a computation step and return variable
  /// that corresponds to the output of the step `pc_{i+1}` and `z_{i+1}`
  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError>;
}

/// A helper trait for a step of the incremental computation for `SuperNova` (i.e., circuit for F) -- automatically
/// implemented for `StepCircuit` and used internally to enforce that the circuit selected by the program counter is used
/// at each step.
pub trait EnforcingStepCircuit<F: PrimeField>: Send + Sync + Clone + StepCircuit<F> {
  /// Delegate synthesis to `StepCircuit::synthesize`, and additionally, enforce the constraint that program counter
  /// `pc`, if supplied, is equal to the circuit's assigned index.
  fn enforcing_synthesize<CS: ConstraintSystem<F>>(
    &self,
    cs: &mut CS,
    pc: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    if let Some(pc) = pc {
      let circuit_index = F::from(self.circuit_index() as u64);

      // pc * 1 = circuit_index
      cs.enforce(
        || "pc matches circuit index",
        |lc| lc + pc.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (circuit_index, CS::one()),
      );
    }
    self.synthesize(cs, pc, z)
  }
}

impl<F: PrimeField, S: StepCircuit<F>> EnforcingStepCircuit<F> for S {}

/// A trivial step circuit that simply returns the input
#[derive(Clone, Debug, Default)]
pub struct TrivialTestCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F> StepCircuit<F> for TrivialTestCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    program_counter: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    Ok((program_counter.cloned(), z.to_vec()))
  }
}

/// A trivial step circuit that simply returns the input, for use on the secondary circuit when implementing NIVC.
/// NOTE: This should not be needed. The secondary circuit doesn't need the program counter at all.
/// Ideally, the need this fills could be met by `traits::circuit::TrivialTestCircuit` (or equivalent).
#[derive(Clone, Debug, Default)]
pub struct TrivialSecondaryCircuit<F: PrimeField> {
  _p: PhantomData<F>,
}

impl<F> StepCircuit<F> for TrivialSecondaryCircuit<F>
where
  F: PrimeField,
{
  fn arity(&self) -> usize {
    1
  }

  fn circuit_index(&self) -> usize {
    0
  }

  fn synthesize<CS: ConstraintSystem<F>>(
    &self,
    _cs: &mut CS,
    program_counter: Option<&AllocatedNum<F>>,
    z: &[AllocatedNum<F>],
  ) -> Result<(Option<AllocatedNum<F>>, Vec<AllocatedNum<F>>), SynthesisError> {
    assert!(program_counter.is_none());
    assert_eq!(z.len(), 1, "Arity of trivial step circuit should be 1");
    Ok((None, z.to_vec()))
  }
}
