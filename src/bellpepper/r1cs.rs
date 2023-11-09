//! Support for generating R1CS using bellpepper.

#![allow(non_snake_case)]

use super::{shape_cs::ShapeCS, solver::SatisfyingAssignment, test_shape_cs::TestShapeCS};
use crate::{
  errors::NovaError,
  r1cs::{CommitmentKeyHint, R1CSInstance, R1CSShape, R1CSWitness, SparseMatrix, R1CS},
  traits::Group,
  CommitmentKey,
};
use bellpepper_core::{Index, LinearCombination};
use ff::PrimeField;

/// `NovaWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait NovaWitness<G: Group> {
  /// Return an instance and witness, given a shape and ck.
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    ck: &CommitmentKey<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), NovaError>;
}

/// `NovaShape` provides methods for acquiring `R1CSShape` and `CommitmentKey` from implementers.
pub trait NovaShape<G: Group> {
  /// Return an appropriate `R1CSShape` and `CommitmentKey` structs.
  /// A `CommitmentKeyHint` should be provided to help guide the construction of the `CommitmentKey`.
  /// This parameter is documented in `r1cs::R1CS::commitment_key`.
  fn r1cs_shape(&self, ck_hint: &CommitmentKeyHint<G>) -> (R1CSShape<G>, CommitmentKey<G>);
}

impl<G: Group> NovaWitness<G> for SatisfyingAssignment<G> {
  fn r1cs_instance_and_witness(
    &self,
    shape: &R1CSShape<G>,
    ck: &CommitmentKey<G>,
  ) -> Result<(R1CSInstance<G>, R1CSWitness<G>), NovaError> {
    let W = R1CSWitness::<G>::new(shape, &self.aux_assignment)?;
    let X = &self.input_assignment[1..];

    let comm_W = W.commit(ck);

    let instance = R1CSInstance::<G>::new(shape, &comm_W, X)?;

    Ok((instance, W))
  }
}

macro_rules! impl_nova_shape {
  ( $name:ident) => {
    impl<G: Group> NovaShape<G> for $name<G>
    where
      G::Scalar: PrimeField,
    {
      fn r1cs_shape(&self, ck_hint: &CommitmentKeyHint<G>) -> (R1CSShape<G>, CommitmentKey<G>) {
        let mut A = SparseMatrix::<G::Scalar>::empty();
        let mut B = SparseMatrix::<G::Scalar>::empty();
        let mut C = SparseMatrix::<G::Scalar>::empty();

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

        A.cols = num_vars + num_inputs;
        B.cols = num_vars + num_inputs;
        C.cols = num_vars + num_inputs;

        // Don't count One as an input for shape's purposes.
        let S = R1CSShape::new(num_constraints, num_vars, num_inputs - 1, A, B, C).unwrap();
        let ck = R1CS::<G>::commitment_key(&S, ck_hint);

        (S, ck)
      }
    }
  };
}

impl_nova_shape!(ShapeCS);
impl_nova_shape!(TestShapeCS);

fn add_constraint<S: PrimeField>(
  X: &mut (
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut SparseMatrix<S>,
    &mut usize,
  ),
  num_vars: usize,
  a_lc: &LinearCombination<S>,
  b_lc: &LinearCombination<S>,
  c_lc: &LinearCombination<S>,
) {
  let (A, B, C, nn) = X;
  let n = **nn;
  assert_eq!(n + 1, A.indptr.len(), "A: invalid shape");
  assert_eq!(n + 1, B.indptr.len(), "B: invalid shape");
  assert_eq!(n + 1, C.indptr.len(), "C: invalid shape");

  let add_constraint_component = |index: Index, coeff: &S, M: &mut SparseMatrix<S>| {
    match index {
      Index::Input(idx) => {
        // Inputs come last, with input 0, reprsenting 'one',
        // at position num_vars within the witness vector.
        let idx = idx + num_vars;
        M.data.push(*coeff);
        M.indices.push(idx);
      }
      Index::Aux(idx) => {
        M.data.push(*coeff);
        M.indices.push(idx);
      }
    }
  };

  for (index, coeff) in a_lc.iter() {
    add_constraint_component(index.0, coeff, A);
  }
  A.indptr.push(A.indices.len());

  for (index, coeff) in b_lc.iter() {
    add_constraint_component(index.0, coeff, B)
  }
  B.indptr.push(B.indices.len());

  for (index, coeff) in c_lc.iter() {
    add_constraint_component(index.0, coeff, C)
  }
  C.indptr.push(C.indices.len());

  **nn += 1;
}
