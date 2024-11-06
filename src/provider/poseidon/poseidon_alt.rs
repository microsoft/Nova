//! This module contains the 'correct' and 'dynamic' versions of Poseidon hashing.
//! These are tested (in `poseidon::test`) to be equivalent to the 'static optimized' version
//! used for actual hashing by the neptune library.
use super::poseidon_inner::{Arity, Poseidon};
use super::{matrix, quintic_s_box};
use ff::PrimeField;

////////////////////////////////////////////////////////////////////////////////
/// Correct
///
/// This code path implements a naive and evidently correct poseidon hash.
///
/// The returned element is the second poseidon element, the first is the arity tag.
pub(crate) fn hash_correct<F, A>(p: &mut Poseidon<'_, F, A>) -> F
where
  F: PrimeField,
  A: Arity<F>,
{
  // This counter is incremented when a round constants is read. Therefore, the round constants never repeat.
  // The first full round should use the initial constants.
  full_round(p);

  for _ in 1..p.constants.half_full_rounds {
    full_round(p);
  }

  partial_round(p);

  for _ in 1..p.constants.partial_rounds {
    partial_round(p);
  }

  for _ in 0..p.constants.half_full_rounds {
    full_round(p);
  }

  p.elements[1]
}

pub(crate) fn full_round<F, A>(p: &mut Poseidon<'_, F, A>)
where
  F: PrimeField,
  A: Arity<F>,
{
  // Apply the quintic S-Box to all elements, after adding the round key.
  // Round keys are added in the S-box to match circuits (where the addition is free)
  // and in preparation for the shift to adding round keys after (rather than before) applying the S-box.

  let pre_round_keys = p
    .constants
    .round_constants
    .as_ref()
    .unwrap()
    .iter()
    .skip(p.constants_offset)
    .map(Some);

  p.elements
    .iter_mut()
    .zip(pre_round_keys)
    .for_each(|(l, pre)| {
      quintic_s_box(l, pre, None);
    });

  p.constants_offset += p.elements.len();

  // M(B)
  // Multiply the elements by the constant MDS matrix
  p.product_mds();
}

/// The partial round is the same as the full round, with the difference that we apply the S-Box only to the first bitflags poseidon leaf.
pub(crate) fn partial_round<F, A>(p: &mut Poseidon<'_, F, A>)
where
  F: PrimeField,
  A: Arity<F>,
{
  // Every element of the hash buffer is incremented by the round constants
  add_round_constants(p);

  // Apply the quintic S-Box to the first element
  quintic_s_box(&mut p.elements[0], None, None);

  // Multiply the elements by the constant MDS matrix
  p.product_mds();
}

////////////////////////////////////////////////////////////////////////////////
/// Dynamic
///
/// This code path implements a code path which dynamically calculates compressed round constants one-deep.
/// It serves as a bridge between the 'correct' and fully, statically optimized implementations.
/// Comments reference notation also expanded in matrix.rs and help clarify the relationship between
/// our optimizations and those described in the paper.
pub(crate) fn hash_optimized_dynamic<F, A>(p: &mut Poseidon<'_, F, A>) -> F
where
  F: PrimeField,
  A: Arity<F>,
{
  // The first full round should use the initial constants.
  full_round_dynamic(p, true, true);

  for _ in 1..(p.constants.half_full_rounds) {
    full_round_dynamic(p, false, true);
  }

  partial_round_dynamic(p);

  for _ in 1..p.constants.partial_rounds {
    partial_round(p);
  }

  for _ in 0..p.constants.half_full_rounds {
    full_round_dynamic(p, true, false);
  }

  p.elements[1]
}

pub(crate) fn full_round_dynamic<F, A>(
  p: &mut Poseidon<'_, F, A>,
  add_current_round_keys: bool,
  absorb_next_round_keys: bool,
) where
  F: PrimeField,
  A: Arity<F>,
{
  // NOTE: decrease in performance is expected when using this pathway.
  // We seek to preserve correctness while transforming the algorithm to an eventually more performant one.

  // Round keys are added in the S-box to match circuits (where the addition is free).
  // If requested, add round keys synthesized from following round after (rather than before) applying the S-box.
  let pre_round_keys = p
    .constants
    .round_constants
    .as_ref()
    .unwrap()
    .iter()
    .skip(p.constants_offset)
    .map(|x| {
      if add_current_round_keys {
        Some(x)
      } else {
        None
      }
    });

  if absorb_next_round_keys {
    // Using the notation from `test_inverse` in matrix.rs:
    // S
    let post_vec = p
      .constants
      .round_constants
      .as_ref()
      .unwrap()
      .iter()
      .skip(
        p.constants_offset
          + if add_current_round_keys {
            p.elements.len()
          } else {
            0
          },
      )
      .take(p.elements.len())
      .copied()
      .collect::<Vec<_>>();

    // Compute the constants which should be added *before* the next `product_mds`.
    // in order to have the same effect as adding the given constants *after* the next `product_mds`.

    // M^-1(S)
    let inverted_vec = matrix::left_apply_matrix(&p.constants.mds_matrices.m_inv, &post_vec);

    // M(M^-1(S))
    let original = matrix::left_apply_matrix(&p.constants.mds_matrices.m, &inverted_vec);

    // S = M(M^-1(S))
    assert_eq!(&post_vec, &original, "Oh no, the inversion trick failed.");

    let post_round_keys = inverted_vec.iter();

    // S-Box Output = B.
    // With post-add, result is B + M^-1(S).
    p.elements
      .iter_mut()
      .zip(pre_round_keys.zip(post_round_keys))
      .for_each(|(l, (pre, post))| {
        quintic_s_box(l, pre, Some(post));
      });
  } else {
    p.elements
      .iter_mut()
      .zip(pre_round_keys)
      .for_each(|(l, pre)| {
        quintic_s_box(l, pre, None);
      });
  }
  let mut consumed = 0;
  if add_current_round_keys {
    consumed += p.elements.len()
  };
  if absorb_next_round_keys {
    consumed += p.elements.len()
  };
  p.constants_offset += consumed;

  // If absorb_next_round_keys
  //   M(B + M^-1(S)
  // else
  //   M(B)
  // Multiply the elements by the constant MDS matrix
  p.product_mds();
}

pub(crate) fn partial_round_dynamic<F, A>(p: &mut Poseidon<'_, F, A>)
where
  F: PrimeField,
  A: Arity<F>,
{
  // Apply the quintic S-Box to the first element
  quintic_s_box(&mut p.elements[0], None, None);

  // Multiply the elements by the constant MDS matrix
  p.product_mds();
}

/// For every leaf, add the round constants with index defined by the constants offset, and increment the
/// offset.
fn add_round_constants<F, A>(p: &mut Poseidon<'_, F, A>)
where
  F: PrimeField,
  A: Arity<F>,
{
  for (element, round_constant) in p.elements.iter_mut().zip(
    p.constants
      .round_constants
      .as_ref()
      .unwrap()
      .iter()
      .skip(p.constants_offset),
  ) {
    element.add_assign(round_constant);
  }

  p.constants_offset += p.elements.len();
}
