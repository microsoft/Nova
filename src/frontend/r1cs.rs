//! Support for generating R1CS using bellpepper.
use super::{shape_cs::ShapeCS, solver::SatisfyingAssignment, test_shape_cs::TestShapeCS};
use crate::{
  errors::NovaError,
  frontend::{Index, LinearCombination},
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, SparseMatrix},
  traits::Engine,
  CommitmentKey,
};
use ff::PrimeField;
use rand_core::{CryptoRng, RngCore};

/// `NovaWitness` provide a method for acquiring an `R1CSInstance` and `R1CSWitness` from implementers.
pub trait NovaWitness<E: Engine> {
  /// Returns an instance and witness using a fresh witness blind sampled from `rng`.
  /// The sampled blind is available through [`R1CSWitness::r_W`].
  fn r1cs_instance_and_witness<R: CryptoRng + RngCore>(
    &self,
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    rng: &mut R,
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), NovaError>;
}

/// `NovaShape` provides methods for acquiring `R1CSShape` from implementers.
pub trait NovaShape<E: Engine> {
  /// Return an appropriate `R1CSShape` struct.
  fn r1cs_shape(&self) -> Result<R1CSShape<E>, NovaError>;
}

impl<E: Engine> NovaWitness<E> for SatisfyingAssignment<E> {
  fn r1cs_instance_and_witness<R: CryptoRng + RngCore>(
    &self,
    shape: &R1CSShape<E>,
    ck: &CommitmentKey<E>,
    rng: &mut R,
  ) -> Result<(R1CSInstance<E>, R1CSWitness<E>), NovaError> {
    let W = R1CSWitness::<E>::new(shape, self.aux_assignment(), rng)?;
    let X = &self.input_assignment()[1..];
    let comm_W = W.commit(ck);
    let instance = R1CSInstance::<E>::new(shape, &comm_W, X)?;
    Ok((instance, W))
  }
}

impl<E: Engine> ShapeCS<E>
where
  E::Scalar: PrimeField,
{
  /// Consume this constraint system and return its `R1CSShape`.
  ///
  /// This conversion reserves each sparse matrix exactly and releases each
  /// source constraint after lowering it. Use [`NovaShape::r1cs_shape`] when
  /// the source constraint system must remain available.
  pub fn into_r1cs_shape(self) -> Result<R1CSShape<E>, NovaError> {
    let num_inputs = self.num_inputs();
    let num_constraints = self.num_constraints();
    let num_vars = self.num_aux();
    let (a_capacity, b_capacity, c_capacity) = self.constraints.iter().fold(
      (0, 0, 0),
      |(a_capacity, b_capacity, c_capacity), (a, b, c)| {
        (
          a_capacity + nonzero_terms(a),
          b_capacity + nonzero_terms(b),
          c_capacity + nonzero_terms(c),
        )
      },
    );

    let mut A = sparse_matrix_with_capacity(num_constraints, a_capacity);
    let mut B = sparse_matrix_with_capacity(num_constraints, b_capacity);
    let mut C = sparse_matrix_with_capacity(num_constraints, c_capacity);
    let mut num_cons_added = 0;
    let mut X = (&mut A, &mut B, &mut C, &mut num_cons_added);

    for (a, b, c) in self.constraints {
      add_constraint(&mut X, num_vars, &a, &b, &c);
    }
    assert_eq!(num_cons_added, num_constraints);

    A.cols = num_vars + num_inputs;
    B.cols = num_vars + num_inputs;
    C.cols = num_vars + num_inputs;

    // Don't count One as an input for shape's purposes.
    R1CSShape::new(num_constraints, num_vars, num_inputs - 1, A, B, C)
  }
}

macro_rules! impl_nova_shape {
  ( $name:ident) => {
    impl<E: Engine> NovaShape<E> for $name<E>
    where
      E::Scalar: PrimeField,
    {
      fn r1cs_shape(&self) -> Result<R1CSShape<E>, NovaError> {
        let mut A = SparseMatrix::<E::Scalar>::empty();
        let mut B = SparseMatrix::<E::Scalar>::empty();
        let mut C = SparseMatrix::<E::Scalar>::empty();

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
        R1CSShape::new(num_constraints, num_vars, num_inputs - 1, A, B, C)
      }
    }
  };
}

impl_nova_shape!(ShapeCS);
impl_nova_shape!(TestShapeCS);

pub(super) fn nonzero_terms<S: PrimeField>(lc: &LinearCombination<S>) -> usize {
  lc.iter()
    .filter(|(_, coefficient)| **coefficient != S::ZERO)
    .count()
}

pub(super) fn sparse_matrix_with_capacity<S: PrimeField>(
  rows: usize,
  terms: usize,
) -> SparseMatrix<S> {
  let mut indptr = Vec::with_capacity(rows + 1);
  indptr.push(0);

  SparseMatrix {
    data: crate::r1cs::SparseCoefficients::with_capacity(terms),
    indices: Vec::with_capacity(terms),
    indptr,
    cols: 0,
  }
}

pub(super) fn add_constraint<S: PrimeField>(
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
    // we add constraints to the matrix only if the associated coefficient is non-zero
    if *coeff != S::ZERO {
      match index {
        Index::Input(idx) => {
          // Inputs come last, with input 0, representing 'one',
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

#[cfg(test)]
mod tests {
  use super::NovaShape;
  use crate::{
    frontend::{shape_cs::ShapeCS, ConstraintSystem},
    provider::PallasEngine,
    traits::Engine,
  };
  use ff::Field;

  type Scalar = <PallasEngine as Engine>::Scalar;

  fn test_shape_cs() -> ShapeCS<PallasEngine> {
    let mut cs = ShapeCS::new();
    let input = cs.alloc_input(|| "input", || Ok(Scalar::from(2))).unwrap();
    let left = cs.alloc(|| "left", || Ok(Scalar::from(3))).unwrap();
    let right = cs.alloc(|| "right", || Ok(Scalar::from(5))).unwrap();

    cs.enforce(
      || "multiply",
      |lc| lc + left + (Scalar::ZERO, input),
      |lc| lc + right,
      |lc| lc + input,
    );
    cs.enforce(
      || "linear",
      |lc| lc + input + (Scalar::from(7), left),
      |lc| lc + ShapeCS::<PallasEngine>::one(),
      |lc| lc + right,
    );
    cs
  }

  #[test]
  fn consumed_shape_matches_borrowed_shape() {
    let borrowed_cs = test_shape_cs();
    let borrowed_shape = borrowed_cs.r1cs_shape().unwrap();
    let consumed_shape = test_shape_cs().into_r1cs_shape().unwrap();

    assert_eq!(consumed_shape, borrowed_shape);
  }

  #[test]
  fn consumed_shape_preserves_digest() {
    let borrowed_shape = test_shape_cs().r1cs_shape().unwrap();
    let consumed_shape = test_shape_cs().into_r1cs_shape().unwrap();

    assert_eq!(consumed_shape.digest(), borrowed_shape.digest());
  }
}
