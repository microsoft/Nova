//! Support for generating R1CS using bellperson.

#![allow(non_snake_case)]

use super::{shape_cs::ShapeCS, solver::SatisfyingAssignment};
use bellperson::{Index, LinearCombination};

use ff::PrimeField;

use crate::{
  errors::NovaError,
  r1cs::{R1CSGens, R1CSInstance, R1CSShape, R1CSWitness},
  traits::Group,
};

/// `NovaWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait NovaWitness<G: Group> {
  /// Return an instance and witness, given a shape and gens.
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    gens: &R1CSGens<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), NovaError>;
}

/// `NovaShape` provides methods for acquiring `R1CSShape` and `R1CSGens` from implementers.
pub trait NovaShape<G: Group> {
  /// Return an appropriate `R1CSShape` struct.
  fn r1cs_shape(&self) -> R1CSShape<G>;
  /// Return an appropriate `R1CSGens` struct.
  fn r1cs_gens(&self) -> R1CSGens<G>;
}

impl<G: Group> NovaWitness<G> for SatisfyingAssignment<G>
where
  G::Scalar: PrimeField,
{
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    gens: &R1CSGens<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), NovaError> {
    let W = R1CSWitness::<G>::new(shape, &self.aux_assignment)?;
    let X = &self.input_assignment[1..];

    let comm_W = W.commit(gens);

    let instance = R1CSInstance::<G>::new(shape, &comm_W, X)?;

    Ok((instance, W))
  }
}

impl<G: Group> NovaShape<G> for ShapeCS<G>
where
  G::Scalar: PrimeField,
{
  fn r1cs_shape(&self) -> R1CSShape<G> {
    let mut A: Vec<(usize, usize, G::Scalar)> = Vec::new();
    let mut B: Vec<(usize, usize, G::Scalar)> = Vec::new();
    let mut C: Vec<(usize, usize, G::Scalar)> = Vec::new();

    let mut num_cons_added = 0;
    let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);

    let num_inputs = self.num_inputs();
    let num_constraints = self.num_constraints();
    let num_vars = self.num_aux();

    for constraint in self.constraints.iter() {
      add_constraint(
        &mut X,
        num_vars,
        &constraint.0,
        &constraint.1,
        &constraint.2,
      );
    }

    assert_eq!(num_cons_added, num_constraints);

    let S: R1CSShape<G> = {
      // Don't count One as an input for shape's purposes.
      let res = R1CSShape::new(num_constraints, num_vars, num_inputs - 1, &A, &B, &C);
      res.unwrap()
    };

    S
  }

  fn r1cs_gens(&self) -> R1CSGens<G> {
    R1CSGens::<G>::new(self.num_constraints(), self.num_aux())
  }
}

fn add_constraint<S: PrimeField>(
  X: &mut (
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut Vec<(usize, usize, S)>,
    &mut usize,
  ),
  num_vars: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
) {
  let (A, B, C, nn) = X;
  let n = **nn;
  let one = S::one();

  let add_constraint_component = |index: Index, coeff, V: &mut Vec<_>| {
    match index {
      Index::Input(idx) => {
        // Inputs come last, with input 0, reprsenting 'one',
        // at position num_vars within the witness vector.
        let i = idx + num_vars;
        V.push((n, i, one * coeff))
      }
      Index::Aux(idx) => V.push((n, idx, one * coeff)),
    }
  };

  for (index, coeff) in a_lc.iter() {
    add_constraint_component(index.0, coeff, A);
  }
  for (index, coeff) in b_lc.iter() {
    add_constraint_component(index.0, coeff, B)
  }
  for (index, coeff) in c_lc.iter() {
    add_constraint_component(index.0, coeff, C)
  }

  **nn += 1;
}
