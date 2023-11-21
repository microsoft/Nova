//! A type that holds the prover key for `CompressedSNARK`
use crate::traits::{circuit::StepCircuit, snark::RelaxedR1CSSNARKTrait, Engine};
use core::marker::PhantomData;
use serde::{Deserialize, Serialize};

/// A type that holds the prover key for `CompressedSNARK`
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E1, E2, C1, C2, S1, S2>
where
  E1: Engine<Base = <E2 as Engine>::Scalar>,
  E2: Engine<Base = <E1 as Engine>::Scalar>,
  C1: StepCircuit<E1::Scalar>,
  C2: StepCircuit<E2::Scalar>,
  S1: RelaxedR1CSSNARKTrait<E1>,
  S2: RelaxedR1CSSNARKTrait<E2>,
{
  pub(crate) pk_primary: S1::ProverKey,
  pub(crate) pk_secondary: S2::ProverKey,
  pub(crate) _p: PhantomData<(C1, C2)>,
}
