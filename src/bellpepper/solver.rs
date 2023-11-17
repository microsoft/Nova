//! Support for generating R1CS witness using bellpepper.

use crate::traits::Engine;

use bellpepper::util_cs::witness_cs::WitnessCS;

/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
pub type SatisfyingAssignment<E> = WitnessCS<<E as Engine>::Scalar>;
