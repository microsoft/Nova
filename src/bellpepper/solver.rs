//! Support for generating R1CS witness using bellpepper.

use crate::traits::Group;

use bellpepper::util_cs::witness_cs::WitnessCS;

/// A `ConstraintSystem` which calculates witness values for a concrete instance of an R1CS circuit.
pub type SatisfyingAssignment<G> = WitnessCS<<G as Group>::Scalar>;
