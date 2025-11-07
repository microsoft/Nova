//! The `circuit2` module implements the optimal Poseidon hash circuit.

use super::{
  hash_type::HashType,
  matrix::Matrix,
  mds::SparseMatrix,
  poseidon_inner::{Arity, PoseidonConstants},
};
use crate::frontend::{
  num::{self, AllocatedNum},
  Boolean, ConstraintSystem, LinearCombination, SynthesisError,
};
use ff::PrimeField;
use std::marker::PhantomData;

/// Similar to `num::Num`, we use `Elt` to accumulate both values and linear combinations, then eventually
/// extract into a `num::AllocatedNum`, enforcing that the linear combination corresponds to the result.
#[derive(Clone)]
pub enum Elt<Scalar: PrimeField> {
  /// [`AllocatedNum`] variant
  Allocated(AllocatedNum<Scalar>),
  /// [`num::Num`] variant
  Num(num::Num<Scalar>),
}

impl<Scalar: PrimeField> From<AllocatedNum<Scalar>> for Elt<Scalar> {
  fn from(allocated: AllocatedNum<Scalar>) -> Self {
    Self::Allocated(allocated)
  }
}

impl<Scalar: PrimeField> Elt<Scalar> {
  /// Create an Elt from a scalar value.
  pub fn num_from_fr<CS: ConstraintSystem<Scalar>>(fr: Scalar) -> Self {
    let num = num::Num::<Scalar>::zero();
    Self::Num(num.add_bool_with_coeff(CS::one(), &Boolean::Constant(true), fr))
  }

  /// Ensure Elt is allocated.
  pub fn ensure_allocated<CS: ConstraintSystem<Scalar>>(
    &self,
    cs: &mut CS,
    enforce: bool,
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    match self {
      Self::Allocated(v) => Ok(v.clone()),
      Self::Num(num) => {
        let v = AllocatedNum::alloc(cs.namespace(|| "allocate for Elt::Num"), || {
          num.get_value().ok_or(SynthesisError::AssignmentMissing)
        })?;

        if enforce {
          cs.enforce(
            || "enforce num allocation preserves lc".to_string(),
            |_| num.lc(Scalar::ONE),
            |lc| lc + CS::one(),
            |lc| lc + v.get_variable(),
          );
        }
        Ok(v)
      }
    }
  }

  /// Get the value of the Elt.
  pub fn val(&self) -> Option<Scalar> {
    match self {
      Self::Allocated(v) => v.get_value(),
      Self::Num(num) => num.get_value(),
    }
  }

  /// Get the [`LinearCombination`]  of the Elt.
  pub fn lc(&self) -> LinearCombination<Scalar> {
    match self {
      Self::Num(num) => num.lc(Scalar::ONE),
      Self::Allocated(v) => LinearCombination::<Scalar>::zero() + v.get_variable(),
    }
  }

  /// Add two Elts and return Elt::Num tracking the calculation.
  #[allow(clippy::should_implement_trait)]
  pub fn add(self, other: Elt<Scalar>) -> Result<Elt<Scalar>, SynthesisError> {
    match (self, other) {
      (Elt::Num(a), Elt::Num(b)) => Ok(Elt::Num(a.add(&b))),
      (a, b) => Ok(Elt::Num(a.num().add(&b.num()))),
    }
  }

  /// Add two Elts and return Elt::Num tracking the calculation.
  pub fn add_ref(self, other: &Elt<Scalar>) -> Result<Elt<Scalar>, SynthesisError> {
    match (self, other) {
      (Elt::Num(a), Elt::Num(b)) => Ok(Elt::Num(a.add(b))),
      (a, b) => Ok(Elt::Num(a.num().add(&b.num()))),
    }
  }

  /// Scale
  pub fn scale<CS: ConstraintSystem<Scalar>>(
    self,
    scalar: Scalar,
  ) -> Result<Elt<Scalar>, SynthesisError> {
    match self {
      Elt::Num(num) => Ok(Elt::Num(num.scale(scalar))),
      Elt::Allocated(a) => Elt::Num(a.into()).scale::<CS>(scalar),
    }
  }

  /// Square
  pub fn square<CS: ConstraintSystem<Scalar>>(
    &self,
    mut cs: CS,
  ) -> Result<AllocatedNum<Scalar>, SynthesisError> {
    match self {
      Elt::Num(num) => {
        let allocated = AllocatedNum::alloc(&mut cs.namespace(|| "squared num"), || {
          num
            .get_value()
            .ok_or(SynthesisError::AssignmentMissing)
            .map(|tmp| tmp * tmp)
        })?;
        cs.enforce(
          || "squaring constraint",
          |_| num.lc(Scalar::ONE),
          |_| num.lc(Scalar::ONE),
          |lc| lc + allocated.get_variable(),
        );
        Ok(allocated)
      }
      Elt::Allocated(a) => a.square(cs),
    }
  }

  /// Return inner Num.
  pub fn num(&self) -> num::Num<Scalar> {
    match self {
      Elt::Num(num) => num.clone(),
      Elt::Allocated(a) => a.clone().into(),
    }
  }
}

/// Circuit for Poseidon hash.
pub struct PoseidonCircuit2<'a, Scalar, A>
where
  Scalar: PrimeField,
  A: Arity<Scalar>,
{
  constants_offset: usize,
  width: usize,
  pub(crate) elements: Vec<Elt<Scalar>>,
  pub(crate) pos: usize,
  current_round: usize,
  constants: &'a PoseidonConstants<Scalar, A>,
  _w: PhantomData<A>,
}

/// PoseidonCircuit2 implementation.
impl<'a, Scalar, A> PoseidonCircuit2<'a, Scalar, A>
where
  Scalar: PrimeField,
  A: Arity<Scalar>,
{
  /// Create a new Poseidon hasher for `preimage`.
  pub fn new(elements: Vec<Elt<Scalar>>, constants: &'a PoseidonConstants<Scalar, A>) -> Self {
    let width = constants.width();

    PoseidonCircuit2 {
      constants_offset: 0,
      width,
      elements,
      pos: 1,
      current_round: 0,
      constants,
      _w: PhantomData::<A>,
    }
  }

  pub fn new_empty<CS: ConstraintSystem<Scalar>>(
    constants: &'a PoseidonConstants<Scalar, A>,
  ) -> Self {
    let elements = Self::initial_elements::<CS>();
    Self::new(elements, constants)
  }

  pub fn hash<CS: ConstraintSystem<Scalar>>(
    &mut self,
    cs: &mut CS,
  ) -> Result<Elt<Scalar>, SynthesisError> {
    self.full_round(cs.namespace(|| "first round"), true, false)?;

    for i in 1..self.constants.full_rounds / 2 {
      self.full_round(
        cs.namespace(|| format!("initial full round {i}")),
        false,
        false,
      )?;
    }

    for i in 0..self.constants.partial_rounds {
      self.partial_round(cs.namespace(|| format!("partial round {i}")))?;
    }

    for i in 0..(self.constants.full_rounds / 2) - 1 {
      self.full_round(
        cs.namespace(|| format!("final full round {i}")),
        false,
        false,
      )?;
    }
    self.full_round(cs.namespace(|| "terminal full round"), false, true)?;

    let elt = self.elements[1].clone();
    self.reset_offsets();

    Ok(elt)
  }

  pub fn apply_padding<CS: ConstraintSystem<Scalar>>(&mut self) {
    if let HashType::ConstantLength(l) = self.constants.hash_type {
      let final_pos = 1 + (l % self.constants.arity());

      assert_eq!(
        self.pos, final_pos,
        "preimage length does not match constant length required for hash"
      );
    };
    match self.constants.hash_type {
      HashType::ConstantLength(_) | HashType::Encryption => {
        for elt in self.elements[self.pos..].iter_mut() {
          *elt = Elt::num_from_fr::<CS>(Scalar::ZERO);
        }
        self.pos = self.elements.len();
      }
      HashType::VariableLength => todo!(),
      _ => (), // incl HashType::Sponge
    }
  }

  fn full_round<CS: ConstraintSystem<Scalar>>(
    &mut self,
    mut cs: CS,
    first_round: bool,
    last_round: bool,
  ) -> Result<(), SynthesisError> {
    let mut constants_offset = self.constants_offset;

    let pre_round_keys = if first_round {
      (0..self.width)
        .map(|i| self.constants.compressed_round_constants[constants_offset + i])
        .collect::<Vec<_>>()
    } else {
      Vec::new()
    };
    constants_offset += pre_round_keys.len();

    let post_round_keys = if first_round || !last_round {
      (0..self.width)
        .map(|i| self.constants.compressed_round_constants[constants_offset + i])
        .collect::<Vec<_>>()
    } else {
      Vec::new()
    };
    constants_offset += post_round_keys.len();

    // Apply the quintic S-Box to all elements
    for i in 0..self.elements.len() {
      let pre_round_key = if first_round {
        let rk = pre_round_keys[i];
        Some(rk)
      } else {
        None
      };

      let post_round_key = if first_round || !last_round {
        let rk = post_round_keys[i];
        Some(rk)
      } else {
        None
      };

      if first_round {
        {
          self.elements[i] = quintic_s_box_pre_add(
            cs.namespace(|| format!("quintic s-box {i}")),
            &self.elements[i],
            pre_round_key,
            post_round_key,
          )?;
        }
      } else {
        self.elements[i] = quintic_s_box(
          cs.namespace(|| format!("quintic s-box {i}")),
          &self.elements[i],
          post_round_key,
        )?;
      }
    }
    self.constants_offset = constants_offset;

    // Multiply the elements by the constant MDS matrix
    self.product_mds::<CS>()?;
    Ok(())
  }

  fn partial_round<CS: ConstraintSystem<Scalar>>(
    &mut self,
    mut cs: CS,
  ) -> Result<(), SynthesisError> {
    let round_key = self.constants.compressed_round_constants[self.constants_offset];
    self.constants_offset += 1;
    // Apply the quintic S-Box to the first element.
    self.elements[0] = quintic_s_box(
      cs.namespace(|| "solitary quintic s-box"),
      &self.elements[0],
      Some(round_key),
    )?;

    // Multiply the elements by the constant MDS matrix
    self.product_mds::<CS>()?;
    Ok(())
  }

  fn product_mds_m<CS: ConstraintSystem<Scalar>>(&mut self) -> Result<(), SynthesisError> {
    self.product_mds_with_matrix::<CS>(&self.constants.mds_matrices.m)
  }

  /// Set the provided elements with the result of the product between the elements and the appropriate
  /// MDS matrix.
  #[allow(clippy::collapsible_else_if)]
  fn product_mds<CS: ConstraintSystem<Scalar>>(&mut self) -> Result<(), SynthesisError> {
    let full_half = self.constants.half_full_rounds;
    let sparse_offset = full_half - 1;
    if self.current_round == sparse_offset {
      self.product_mds_with_matrix::<CS>(&self.constants.pre_sparse_matrix)?;
    } else {
      if (self.current_round > sparse_offset)
        && (self.current_round < full_half + self.constants.partial_rounds)
      {
        let index = self.current_round - sparse_offset - 1;
        let sparse_matrix = &self.constants.sparse_matrixes[index];

        self.product_mds_with_sparse_matrix::<CS>(sparse_matrix)?;
      } else {
        self.product_mds_m::<CS>()?;
      }
    };

    self.current_round += 1;
    Ok(())
  }

  #[allow(clippy::ptr_arg, clippy::needless_range_loop)]
  fn product_mds_with_matrix<CS: ConstraintSystem<Scalar>>(
    &mut self,
    matrix: &Matrix<Scalar>,
  ) -> Result<(), SynthesisError> {
    let mut result: Vec<Elt<Scalar>> = Vec::with_capacity(self.constants.width());

    for j in 0..self.constants.width() {
      let column = (0..self.constants.width())
        .map(|i| matrix[i][j])
        .collect::<Vec<_>>();

      let product = scalar_product::<Scalar, CS>(self.elements.as_slice(), &column)?;

      result.push(product);
    }

    self.elements = result;

    Ok(())
  }

  // Sparse matrix in this context means one of the form, M''.
  fn product_mds_with_sparse_matrix<CS: ConstraintSystem<Scalar>>(
    &mut self,
    matrix: &SparseMatrix<Scalar>,
  ) -> Result<(), SynthesisError> {
    let mut result: Vec<Elt<Scalar>> = Vec::with_capacity(self.constants.width());

    result.push(scalar_product::<Scalar, CS>(
      self.elements.as_slice(),
      &matrix.w_hat,
    )?);

    for j in 1..self.width {
      result.push(self.elements[j].clone().add(
        self.elements[0]
                        .clone() // First row is dense.
                        .scale::<CS>(matrix.v_rest[j - 1])?, // Except for first row/column, diagonals are one.
      )?);
    }

    self.elements = result;

    Ok(())
  }

  fn initial_elements<CS: ConstraintSystem<Scalar>>() -> Vec<Elt<Scalar>> {
    std::iter::repeat(Elt::num_from_fr::<CS>(Scalar::ZERO))
      .take(A::to_usize() + 1)
      .collect()
  }

  pub fn reset_offsets(&mut self) {
    self.constants_offset = 0;
    self.current_round = 0;
    self.pos = 1;
  }
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to result.
fn quintic_s_box<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
  mut cs: CS,
  l: &Elt<Scalar>,
  post_round_key: Option<Scalar>,
) -> Result<Elt<Scalar>, SynthesisError> {
  // If round_key was supplied, add it after all exponentiation.
  let l2 = l.square(cs.namespace(|| "l^2"))?;
  let l4 = l2.square(cs.namespace(|| "l^4"))?;
  let l5 = mul_sum(
    cs.namespace(|| "(l4 * l) + rk)"),
    &l4,
    l,
    None,
    post_round_key,
    true,
  );

  Ok(Elt::Allocated(l5?))
}

/// Compute l^5 and enforce constraint. If round_key is supplied, add it to l first.
fn quintic_s_box_pre_add<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
  mut cs: CS,
  l: &Elt<Scalar>,
  pre_round_key: Option<Scalar>,
  post_round_key: Option<Scalar>,
) -> Result<Elt<Scalar>, SynthesisError> {
  if let (Some(pre_round_key), Some(post_round_key)) = (pre_round_key, post_round_key) {
    // If round_key was supplied, add it to l before squaring.
    let l2 = square_sum(cs.namespace(|| "(l+rk)^2"), pre_round_key, l, true)?;
    let l4 = l2.square(cs.namespace(|| "l^4"))?;
    let l5 = mul_sum(
      cs.namespace(|| "l4 * (l + rk)"),
      &l4,
      l,
      Some(pre_round_key),
      Some(post_round_key),
      true,
    );

    Ok(Elt::Allocated(l5?))
  } else {
    panic!("pre_round_key and post_round_key must both be provided.");
  }
}

/// Calculates square of sum and enforces that constraint.
pub fn square_sum<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
  mut cs: CS,
  to_add: Scalar,
  elt: &Elt<Scalar>,
  enforce: bool,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
  let res = AllocatedNum::alloc(cs.namespace(|| "squared sum"), || {
    let mut tmp = elt.val().ok_or(SynthesisError::AssignmentMissing)?;
    tmp.add_assign(&to_add);
    tmp = tmp.square();
    Ok(tmp)
  })?;

  if enforce {
    cs.enforce(
      || "squared sum constraint",
      |_| elt.lc() + (to_add, CS::one()),
      |_| elt.lc() + (to_add, CS::one()),
      |lc| lc + res.get_variable(),
    );
  }
  Ok(res)
}

/// Calculates (a * (pre_add + b)) + post_add — and enforces that constraint.
#[allow(clippy::collapsible_else_if)]
pub fn mul_sum<CS: ConstraintSystem<Scalar>, Scalar: PrimeField>(
  mut cs: CS,
  a: &AllocatedNum<Scalar>,
  b: &Elt<Scalar>,
  pre_add: Option<Scalar>,
  post_add: Option<Scalar>,
  enforce: bool,
) -> Result<AllocatedNum<Scalar>, SynthesisError> {
  let res = AllocatedNum::alloc(cs.namespace(|| "mul_sum"), || {
    let mut tmp = b.val().ok_or(SynthesisError::AssignmentMissing)?;
    if let Some(x) = pre_add {
      tmp.add_assign(&x);
    }
    tmp.mul_assign(&a.get_value().ok_or(SynthesisError::AssignmentMissing)?);
    if let Some(x) = post_add {
      tmp.add_assign(&x);
    }

    Ok(tmp)
  })?;

  if enforce {
    if let Some(x) = post_add {
      let neg = -x;

      if let Some(pre) = pre_add {
        cs.enforce(
          || "mul sum constraint pre-post-add",
          |_| b.lc() + (pre, CS::one()),
          |lc| lc + a.get_variable(),
          |lc| lc + res.get_variable() + (neg, CS::one()),
        );
      } else {
        cs.enforce(
          || "mul sum constraint post-add",
          |_| b.lc(),
          |lc| lc + a.get_variable(),
          |lc| lc + res.get_variable() + (neg, CS::one()),
        );
      }
    } else {
      if let Some(pre) = pre_add {
        cs.enforce(
          || "mul sum constraint pre-add",
          |_| b.lc() + (pre, CS::one()),
          |lc| lc + a.get_variable(),
          |lc| lc + res.get_variable(),
        );
      } else {
        cs.enforce(
          || "mul sum constraint",
          |_| b.lc(),
          |lc| lc + a.get_variable(),
          |lc| lc + res.get_variable(),
        );
      }
    }
  }
  Ok(res)
}

fn scalar_product<Scalar: PrimeField, CS: ConstraintSystem<Scalar>>(
  elts: &[Elt<Scalar>],
  scalars: &[Scalar],
) -> Result<Elt<Scalar>, SynthesisError> {
  elts
    .iter()
    .zip(scalars)
    .try_fold(Elt::Num(num::Num::zero()), |acc, (elt, &scalar)| {
      acc.add(elt.clone().scale::<CS>(scalar)?)
    })
}
