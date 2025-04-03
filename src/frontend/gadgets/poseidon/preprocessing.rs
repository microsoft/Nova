use super::{
  matrix::{left_apply_matrix, vec_add},
  mds::MdsMatrices,
  quintic_s_box,
};
#[cfg(not(feature = "std"))]
use crate::prelude::*;
use ff::PrimeField;

// - Compress constants by pushing them back through linear layers and through the identity components of partial layers.
// - As a result, constants need only be added after each S-box.
#[allow(clippy::ptr_arg)]
pub(crate) fn compress_round_constants<F: PrimeField>(
  width: usize,
  full_rounds: usize,
  partial_rounds: usize,
  round_constants: &Vec<F>,
  mds_matrices: &MdsMatrices<F>,
  partial_preprocessed: usize,
) -> Vec<F> {
  let mds_matrix = &mds_matrices.m;
  let inverse_matrix = &mds_matrices.m_inv;

  let mut res = Vec::new();

  let round_keys = |r: usize| &round_constants[r * width..(r + 1) * width];

  let half_full_rounds = full_rounds / 2; // Not half-full rounds; half full-rounds.

  // First round constants are unchanged.
  res.extend(round_keys(0));

  let unpreprocessed = partial_rounds - partial_preprocessed;

  // Post S-box adds for the first set of full rounds should be 'inverted' from next round.
  // The final round is skipped when fully preprocessing because that value must be obtained from the result of preprocessing the partial rounds.
  let end = if unpreprocessed > 0 {
    half_full_rounds
  } else {
    half_full_rounds - 1
  };
  for i in 0..end {
    let next_round = round_keys(i + 1); // First round was added before any S-boxes.
    let inverted = left_apply_matrix(inverse_matrix, next_round);
    res.extend(inverted);
  }

  // The plan:
  // - Work backwards from last row in this group
  // - Invert the row.
  // - Save first constant (corresponding to the one S-box performed).
  // - Add inverted result to previous row.
  // - Repeat until all partial round key rows have been consumed.
  // - Extend the preprocessed result by the final resultant row.
  // - Move the accumulated list of single round keys to the preprocessed result.
  //   - (Last produced should be first applied, so either pop until empty, or reverse and extend, etc.

  // `partial_keys` will accumulate the single post-S-box constant for each partial-round, in reverse order.
  let mut partial_keys: Vec<F> = Vec::new();

  let final_round = half_full_rounds + partial_rounds;
  let final_round_key = round_keys(final_round).to_vec();

  // `round_acc` holds the accumulated result of inverting and adding subsequent round constants (in reverse).
  let round_acc = (0..partial_preprocessed)
    .map(|i| round_keys(final_round - i - 1))
    .fold(final_round_key, |acc, previous_round_keys| {
      let mut inverted = left_apply_matrix(inverse_matrix, &acc);

      partial_keys.push(inverted[0]);
      inverted[0] = F::ZERO;

      vec_add(previous_round_keys, &inverted)
    });

  // Everything in here is dev-driven testing.
  // Dev test case only checks one deep.
  if partial_preprocessed == 1 {
    // Check assumptions about how the fold calculating round_acc  manifested.

    // The last round containing unpreprocessed constants which should be compressed.
    let terminal_constants_round = half_full_rounds + partial_rounds;

    // Constants from the last round (of two) which should be compressed.
    // T
    let terminal_round_keys = round_keys(terminal_constants_round);

    // Constants from the first round (of two) which should be compressed.
    // I
    let initial_round_keys = round_keys(terminal_constants_round - 1);

    // M^-1(T)
    let mut inv = left_apply_matrix(inverse_matrix, terminal_round_keys);

    // M^-1(T)[0]
    let pk = inv[0];

    // M^-1(T) - pk (kinda)
    inv[0] = F::ZERO;

    // (M^-1(T) - pk) - I
    let result_key = vec_add(initial_round_keys, &inv);

    assert_eq!(&result_key, &round_acc, "Acc assumption failed.");
    assert_eq!(pk, partial_keys[0], "Partial-key assumption failed.");
    assert_eq!(
      1,
      partial_keys.len(),
      "Partial-keys length assumption failed."
    );

    ////////////////////////////////////////////////////////////////////////////////
    // Shared between branches, arbitrary initial state representing the output of a previous round's S-Box layer.
    // X
    let initial_state = vec![F::ONE; width];

    ////////////////////////////////////////////////////////////////////////////////
    // Compute one step with the given (unpreprocessed) constants.

    // ARK
    // I + X
    let mut q_state = vec_add(initial_round_keys, &initial_state);

    // S-Box (partial layer)
    // S((I + X)[0]) = S(I[0] + X[0])
    quintic_s_box(&mut q_state[0], None, None);

    // Mix with mds_matrix
    let mixed = left_apply_matrix(mds_matrix, &q_state);

    // Ark
    let plain_result = vec_add(terminal_round_keys, &mixed);

    ////////////////////////////////////////////////////////////////////////////////
    // Compute the same step using the preprocessed constants.
    // M'(initial_state) + (inverted_id - initial_state) = inverted_id
    //let initial_state1 = apply_matrix::<E>(&m_prime, &initial_state);
    let mut p_state = vec_add(&result_key, &initial_state);

    // In order for the S-box result to be correct, it must have the same input as in the plain path.
    // That means its input (the first component of the state) must have been constructed by
    // adding the same single round constant in that position.
    // NOTE: this assertion uncovered a bug which was causing failure.
    assert_eq!(
      &result_key[0], &initial_round_keys[0],
      "S-box inputs did not match."
    );

    quintic_s_box(&mut p_state[0], None, Some(&pk));

    let preprocessed_result = left_apply_matrix(mds_matrix, &p_state);

    assert_eq!(
      plain_result, preprocessed_result,
      "Single preprocessing step couldn't be verified."
    );
  }

  for i in 1..unpreprocessed {
    res.extend(round_keys(half_full_rounds + i));
  }
  res.extend(left_apply_matrix(inverse_matrix, &round_acc));

  while let Some(x) = partial_keys.pop() {
    res.push(x)
  }

  // Post S-box adds for the first set of full rounds should be 'inverted' from next round.
  for i in 1..(half_full_rounds) {
    let start = half_full_rounds + partial_rounds;
    let next_round = round_keys(i + start);
    let inverted = left_apply_matrix(inverse_matrix, next_round);
    res.extend(inverted);
  }

  res
}
