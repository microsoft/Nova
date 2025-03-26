#[cfg(not(feature = "std"))]
use crate::prelude::*;
use ff::PrimeField;
use generic_array::{sequence::GenericSequence, typenum, ArrayLength, GenericArray};
#[cfg(feature = "std")]
use std::{marker::PhantomData, mem};
use typenum::*;

use super::{
  matrix::transpose,
  mds::{derive_mds_matrices, factor_to_sparse_matrixes, generate_mds},
  preprocessing::compress_round_constants,
};

use super::{
  hash_type::HashType,
  matrix::{left_apply_matrix, Matrix},
  mds::{MdsMatrices, SparseMatrix},
  quintic_s_box, round_constants, round_numbers, Strength, DEFAULT_STRENGTH,
};

/// Available arities for the Poseidon hasher.
pub trait Arity<T>: ArrayLength {
  /// Must be Arity + 1.
  type ConstantsSize: ArrayLength;

  fn tag() -> T;
}

macro_rules! impl_arity {
  ($($a:ty),*) => {
      $(
          impl<F: PrimeField> Arity<F> for $a {
              type ConstantsSize = Add1<$a>;

              fn tag() -> F {
                  F::from((1 << <$a as Unsigned>::to_usize()) - 1)
              }
          }
      )*
  };
}

// Dummy implementation to allow for an "optional" argument.
impl<F: PrimeField> Arity<F> for U0 {
  type ConstantsSize = U0;

  fn tag() -> F {
    unreachable!("dummy implementation for U0, should not be called")
  }
}

impl_arity!(
  U1, U2, U3, U4, U5, U6, U7, U8, U9, U10, U11, U12, U13, U14, U15, U16, U17, U18, U19, U20, U21,
  U22, U23, U24, U25, U26, U27, U28, U29, U30, U31, U32, U33, U34, U35, U36
);

/// Holds preimage, some utility offsets and counters along with the reference
/// to [`PoseidonConstants`] required for hashing. [`Poseidon`] is parameterized
/// by [`ff::PrimeField`] and [`Arity`], which should be similar to [`PoseidonConstants`].
///
/// [`Poseidon`] accepts input `elements` set with length equal or less than [`Arity`].
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Poseidon<'a, F, A = U2>
where
  F: PrimeField,
  A: Arity<F>,
{
  pub(crate) constants_offset: usize,
  pub(crate) current_round: usize, // Used in static optimization only for now.
  /// the elements to permute
  pub elements: GenericArray<F, A::ConstantsSize>,
  /// index of the next element of state to be absorbed
  pub(crate) pos: usize,
  pub(crate) constants: &'a PoseidonConstants<F, A>,
  _f: PhantomData<F>,
}

/// Holds constant values required for further [`Poseidon`] hashing. It contains MDS matrices,
/// round constants and numbers, parameters that specify security level ([`Strength`]) and
/// domain separation ([`HashType`]). Additional constants related to optimizations are also included.
///
/// For correct operation, [`PoseidonConstants`] instance should be parameterized with the same [`ff::PrimeField`]
/// and [`Arity`] as [`Poseidon`] instance that consumes it.
///
/// See original [Poseidon paper](https://eprint.iacr.org/2019/458.pdf) for more details.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoseidonConstants<F: PrimeField, A: Arity<F>> {
  pub(crate) mds_matrices: MdsMatrices<F>,
  pub(crate) round_constants: Option<Vec<F>>, // TODO: figure out how to automatically allocate `None`
  pub(crate) compressed_round_constants: Vec<F>,
  pub(crate) pre_sparse_matrix: Matrix<F>,
  pub(crate) sparse_matrixes: Vec<SparseMatrix<F>>,
  pub(crate) strength: Strength,
  /// The domain tag is the first element of a Poseidon permutation.
  /// This extra element is necessary for 128-bit security.
  pub(crate) domain_tag: F,
  pub(crate) full_rounds: usize,
  pub(crate) half_full_rounds: usize,
  pub(crate) partial_rounds: usize,
  pub(crate) hash_type: HashType<F, A>,
  pub(crate) _a: PhantomData<A>,
}

impl<F, A> Default for PoseidonConstants<F, A>
where
  F: PrimeField,
  A: Arity<F>,
{
  fn default() -> Self {
    Self::new()
  }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum HashMode {
  // Consumes statically pre-processed constants for simplest operation.
  OptimizedStatic,
}
use HashMode::OptimizedStatic;

pub const DEFAULT_HASH_MODE: HashMode = OptimizedStatic;

impl<F, A> PoseidonConstants<F, A>
where
  F: PrimeField,
  A: Arity<F>,
{
  /// Generates new instance of [`PoseidonConstants`] suitable for both optimized / non-optimized hashing
  /// with following default parameters:
  /// - 128 bit of security;
  /// - Merkle Tree (where all leaves are presented) domain separation ([`HashType`]).
  pub fn new() -> Self {
    Self::new_with_strength(DEFAULT_STRENGTH)
  }

  /// Generates new instance of [`PoseidonConstants`] suitable for both optimized / non-optimized hashing
  /// with Merkle Tree (where all leaves are presented) domain separation ([`HashType`]) custom security level ([`Strength`]).
  pub fn new_with_strength(strength: Strength) -> Self {
    Self::new_with_strength_and_type(strength, HashType::MerkleTree)
  }

  /// Generates new instance of [`PoseidonConstants`] suitable for both optimized / non-optimized hashing
  /// with custom domain separation ([`HashType`]) and custom security level ([`Strength`]).
  pub fn new_with_strength_and_type(strength: Strength, hash_type: HashType<F, A>) -> Self {
    assert!(hash_type.is_supported());
    let arity = A::to_usize();
    let width = arity + 1;
    let mds = generate_mds(width);
    let (full_rounds, partial_rounds) = round_numbers(arity, &strength);
    let round_constants = round_constants(arity, &strength);

    // Now call new_from_parameters with all the necessary parameters.
    Self::new_from_parameters(
      width,
      mds,
      round_constants,
      full_rounds,
      partial_rounds,
      hash_type,
      strength,
    )
  }

  /// Generates new instance of [`PoseidonConstants`] with matrix, constants and number of rounds.
  /// The matrix does not have to be symmetric.
  pub fn new_from_parameters(
    width: usize,
    m: Matrix<F>,
    round_constants: Vec<F>,
    full_rounds: usize,
    partial_rounds: usize,
    hash_type: HashType<F, A>,
    strength: Strength,
  ) -> Self {
    let mds_matrices = derive_mds_matrices(m);
    let half_full_rounds = full_rounds / 2;
    let compressed_round_constants = compress_round_constants(
      width,
      full_rounds,
      partial_rounds,
      &round_constants,
      &mds_matrices,
      partial_rounds,
    );

    let (pre_sparse_matrix, sparse_matrixes) =
      factor_to_sparse_matrixes(&transpose(&mds_matrices.m), partial_rounds);

    // Ensure we have enough constants for the sbox rounds
    assert!(
      width * (full_rounds + partial_rounds) <= round_constants.len(),
      "Not enough round constants"
    );

    assert_eq!(
      full_rounds * width + partial_rounds,
      compressed_round_constants.len()
    );

    Self {
      mds_matrices,
      round_constants: Some(round_constants),
      compressed_round_constants,
      pre_sparse_matrix,
      sparse_matrixes,
      strength,
      domain_tag: hash_type.domain_tag(),
      full_rounds,
      half_full_rounds,
      partial_rounds,
      hash_type,
      _a: PhantomData::<A>,
    }
  }

  /// Returns the [`Arity`] value represented as `usize`.
  #[inline]
  pub fn arity(&self) -> usize {
    A::to_usize()
  }

  /// Returns `width` value represented as `usize`. It equals to [`Arity`] + 1.
  #[inline]
  pub fn width(&self) -> usize {
    A::ConstantsSize::to_usize()
  }
}

impl<'a, F, A> Poseidon<'a, F, A>
where
  F: PrimeField,
  A: Arity<F>,
{
  /// Creates [`Poseidon`] instance using provided [`PoseidonConstants`] as input. Underlying set of
  /// elements are initialized and `domain_tag` from [`PoseidonConstants`] is used as zero element in the set.
  /// Therefore, hashing is eventually performed over [`Arity`] + 1 elements in fact, while [`Arity`] elements
  /// are occupied by preimage data.
  pub fn new(constants: &'a PoseidonConstants<F, A>) -> Self {
    let elements = GenericArray::generate(|i| {
      if i == 0 {
        constants.domain_tag
      } else {
        F::ZERO
      }
    });
    Poseidon {
      constants_offset: 0,
      current_round: 0,
      elements,
      pos: 1,
      constants,
      _f: PhantomData::<F>,
    }
  }

  pub(crate) fn reset_offsets(&mut self) {
    self.constants_offset = 0;
    self.current_round = 0;
    self.pos = 1;
  }

  /// Performs hashing using underlying [`Poseidon`] buffer of the preimage' field elements
  /// using provided [`HashMode`]. Always outputs digest expressed as a single field element
  /// of concrete type specified upon [`PoseidonConstants`] and [`Poseidon`] instantiations.
  pub fn hash_in_mode(&mut self, mode: HashMode) -> F {
    let res = match mode {
      OptimizedStatic => self.hash_optimized_static(),
    };
    self.reset_offsets();
    res
  }

  /// Performs hashing using underlying [`Poseidon`] buffer of the preimage' field elements
  /// in default (optimized) mode. Always outputs digest expressed as a single field element
  /// of concrete type specified upon [`PoseidonConstants`] and [`Poseidon`] instantiations.
  pub fn hash(&mut self) -> F {
    self.hash_in_mode(DEFAULT_HASH_MODE)
  }

  pub(crate) fn apply_padding(&mut self) {
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
          *elt = F::ZERO;
        }
        self.pos = self.elements.len();
      }
      HashType::VariableLength => todo!(),
      _ => (), // incl. HashType::Sponge
    }
  }

  /// Returns 1-th element from underlying [`Poseidon`] buffer. This function is important, since
  /// according to [`Poseidon`] design, after performing hashing, output digest will be stored at
  /// 1-st place of underlying buffer.
  #[inline]
  pub fn extract_output(&self) -> F {
    self.elements[1]
  }

  /// Performs hashing using underlying [`Poseidon`] buffer of the preimage' field elements
  /// using [`HashMode::OptimizedStatic`] mode. Always outputs digest expressed as a single field element
  /// of concrete type specified upon [`PoseidonConstants`] and [`Poseidon`] instantiations.
  pub fn hash_optimized_static(&mut self) -> F {
    // The first full round should use the initial constants.
    self.add_round_constants();

    for _ in 0..self.constants.half_full_rounds {
      self.full_round(false);
    }

    for _ in 0..self.constants.partial_rounds {
      self.partial_round();
    }

    // All but last full round.
    for _ in 1..self.constants.half_full_rounds {
      self.full_round(false);
    }
    self.full_round(true);

    assert_eq!(
      self.constants_offset,
      self.constants.compressed_round_constants.len(),
      "Constants consumed ({}) must equal preprocessed constants provided ({}).",
      self.constants_offset,
      self.constants.compressed_round_constants.len()
    );

    self.extract_output()
  }

  fn full_round(&mut self, last_round: bool) {
    let to_take = self.elements.len();
    let post_round_keys = self
      .constants
      .compressed_round_constants
      .iter()
      .skip(self.constants_offset)
      .take(to_take);

    if !last_round {
      let needed = self.constants_offset + to_take;
      assert!(
        needed <= self.constants.compressed_round_constants.len(),
        "Not enough preprocessed round constants ({}), need {}.",
        self.constants.compressed_round_constants.len(),
        needed
      );
    }
    self
      .elements
      .iter_mut()
      .zip(post_round_keys)
      .for_each(|(l, post)| {
        // Be explicit that no round key is added after last round of S-boxes.
        let post_key = if last_round {
          panic!(
            "Trying to skip last full round, but there is a key here! ({:?})",
            post
          );
        } else {
          Some(post)
        };
        quintic_s_box(l, None, post_key);
      });
    // We need this because post_round_keys will have been empty, so it didn't happen in the for_each. :(
    if last_round {
      self
        .elements
        .iter_mut()
        .for_each(|l| quintic_s_box(l, None, None));
    } else {
      self.constants_offset += self.elements.len();
    }
    self.round_product_mds();
  }

  /// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first (arity tag) poseidon leaf.
  fn partial_round(&mut self) {
    let post_round_key = self.constants.compressed_round_constants[self.constants_offset];

    // Apply the quintic S-Box to the first element
    quintic_s_box(&mut self.elements[0], None, Some(&post_round_key));
    self.constants_offset += 1;

    self.round_product_mds();
  }

  fn add_round_constants(&mut self) {
    for (element, round_constant) in self.elements.iter_mut().zip(
      self
        .constants
        .compressed_round_constants
        .iter()
        .skip(self.constants_offset),
    ) {
      element.add_assign(round_constant);
    }
    self.constants_offset += self.elements.len();
  }

  /// Set the provided elements with the result of the product between the elements and the appropriate
  /// MDS matrix.
  #[allow(clippy::collapsible_else_if)]
  fn round_product_mds(&mut self) {
    let full_half = self.constants.half_full_rounds;
    let sparse_offset = full_half - 1;
    if self.current_round == sparse_offset {
      self.product_mds_with_matrix(&self.constants.pre_sparse_matrix);
    } else {
      if (self.current_round > sparse_offset)
        && (self.current_round < full_half + self.constants.partial_rounds)
      {
        let index = self.current_round - sparse_offset - 1;
        let sparse_matrix = &self.constants.sparse_matrixes[index];

        self.product_mds_with_sparse_matrix(sparse_matrix);
      } else {
        self.product_mds();
      }
    };

    self.current_round += 1;
  }

  /// Set the provided elements with the result of the product between the elements and the constant
  /// MDS matrix.
  pub(crate) fn product_mds(&mut self) {
    self.product_mds_with_matrix_left(&self.constants.mds_matrices.m);
  }

  /// NOTE: This calculates a vector-matrix product (`elements * matrix`) rather than the
  /// expected matrix-vector `(matrix * elements)`. This is a performance optimization which
  /// exploits the fact that our MDS matrices are symmetric by construction.
  #[allow(clippy::ptr_arg)]
  pub(crate) fn product_mds_with_matrix(&mut self, matrix: &Matrix<F>) {
    let mut result = GenericArray::<F, A::ConstantsSize>::generate(|_| F::ZERO);

    for (j, val) in result.iter_mut().enumerate() {
      for (i, row) in matrix.iter().enumerate() {
        let mut tmp = row[j];
        tmp.mul_assign(&self.elements[i]);
        val.add_assign(&tmp);
      }
    }

    let _ = mem::replace(&mut self.elements, result);
  }

  pub(crate) fn product_mds_with_matrix_left(&mut self, matrix: &Matrix<F>) {
    let result = left_apply_matrix(matrix, &self.elements);
    let _ = mem::replace(
      &mut self.elements,
      GenericArray::<F, A::ConstantsSize>::generate(|i| result[i]),
    );
  }

  // Sparse matrix in this context means one of the form, M''.
  fn product_mds_with_sparse_matrix(&mut self, sparse_matrix: &SparseMatrix<F>) {
    let mut result = GenericArray::<F, A::ConstantsSize>::generate(|_| F::ZERO);

    // First column is dense.
    for (i, val) in sparse_matrix.w_hat.iter().enumerate() {
      let mut tmp = *val;
      tmp.mul_assign(&self.elements[i]);
      result[0].add_assign(&tmp);
    }

    for (j, val) in result.iter_mut().enumerate().skip(1) {
      // Except for first row/column, diagonals are one.
      val.add_assign(&self.elements[j]);

      // First row is dense.
      let mut tmp = sparse_matrix.v_rest[j - 1];
      tmp.mul_assign(&self.elements[0]);
      val.add_assign(&tmp);
    }

    let _ = mem::replace(&mut self.elements, result);
  }
}
