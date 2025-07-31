//! Support for generating R1CS
//!
//! Most of the code is ported from <https://github.com/argumentcomputer/bellpepper>.

pub mod gadgets;

mod constraint_system;
mod lc;

pub use constraint_system::{Circuit, ConstraintSystem, Namespace, SynthesisError};
pub use gadgets::{
  boolean::{AllocatedBit, Boolean},
  num,
  poseidon::{Elt, PoseidonConstants, SpongeCircuit},
  sha256::sha256,
  Assignment,
};
pub use lc::{Index, LinearCombination, Variable};

pub mod r1cs;
pub mod shape_cs;
pub mod solver;
pub mod test_shape_cs;
pub mod util_cs;

#[cfg(test)]
pub use util_cs::test_cs;

#[cfg(test)]
mod tests {
  use crate::{
    frontend::{
      num::AllocatedNum,
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      ConstraintSystem,
    },
    provider::{Bn256EngineKZG, PallasEngine, Secp256k1Engine},
    traits::{snark::default_ck_hint, Engine},
  };
  use ff::PrimeField;

  fn synthesize_alloc_bit<Fr: PrimeField, CS: ConstraintSystem<Fr>>(cs: &mut CS) {
    // get two bits as input and check that they are indeed bits
    let a = AllocatedNum::alloc_infallible(cs.namespace(|| "a"), || Fr::ONE);
    let _ = a.inputize(cs.namespace(|| "a is input"));
    cs.enforce(
      || "check a is 0 or 1",
      |lc| lc + CS::one() - a.get_variable(),
      |lc| lc + a.get_variable(),
      |lc| lc,
    );
    let b = AllocatedNum::alloc_infallible(cs.namespace(|| "b"), || Fr::ONE);
    let _ = b.inputize(cs.namespace(|| "b is input"));
    cs.enforce(
      || "check b is 0 or 1",
      |lc| lc + CS::one() - b.get_variable(),
      |lc| lc + b.get_variable(),
      |lc| lc,
    );
  }

  fn test_alloc_bit_with<E: Engine>() {
    // First create the shape
    let mut cs: ShapeCS<E> = ShapeCS::new();
    synthesize_alloc_bit(&mut cs);
    let (shape, ck) = cs.r1cs_shape(&*default_ck_hint());

    // Now get the assignment
    let mut cs = SatisfyingAssignment::<E>::new();
    synthesize_alloc_bit(&mut cs);
    let (inst, witness) = cs.r1cs_instance_and_witness(&shape, &ck).unwrap();

    // Make sure that this is satisfiable
    assert!(shape.is_sat(&ck, &inst, &witness).is_ok());
  }

  #[test]
  fn test_alloc_bit() {
    test_alloc_bit_with::<PallasEngine>();
    test_alloc_bit_with::<Bn256EngineKZG>();
    test_alloc_bit_with::<Secp256k1Engine>();
  }
}
