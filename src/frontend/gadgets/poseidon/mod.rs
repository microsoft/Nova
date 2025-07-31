//! The underlying Poseidon sponge code is ported from <https://github.com/argumentcomputer/neptune>.
use ff::PrimeField;
use serde::{Deserialize, Serialize};

mod circuit2;
mod circuit2_witness;
mod hash_type;
mod matrix;
mod mds;
mod poseidon_inner;
mod preprocessing;
mod round_constants;
mod round_numbers;
mod serde_impl;
mod sponge;

pub use circuit2::Elt;
pub use poseidon_inner::PoseidonConstants;
use round_constants::generate_constants;
use round_numbers::{round_numbers_base, round_numbers_strengthened};
pub use sponge::{
  api::{IOPattern, SpongeAPI, SpongeOp},
  circuit::SpongeCircuit,
  vanilla::{Mode::Simplex, Sponge, SpongeTrait},
};

/// The strength of the Poseidon hash function
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Strength {
  /// Standard strength
  Standard,
  /// Strengthened strength
  Strengthened,
}

const DEFAULT_STRENGTH: Strength = Strength::Standard;

fn round_numbers(arity: usize, strength: &Strength) -> (usize, usize) {
  match strength {
    Strength::Standard => round_numbers_base(arity),
    Strength::Strengthened => round_numbers_strengthened(arity),
  }
}

const SBOX: u8 = 1; // x^5
const FIELD: u8 = 1; // Gf(p)

fn round_constants<F: PrimeField>(arity: usize, strength: &Strength) -> Vec<F> {
  let t = arity + 1;

  let (full_rounds, partial_rounds) = round_numbers(arity, strength);

  let r_f = full_rounds as u16;
  let r_p = partial_rounds as u16;

  let fr_num_bits = F::NUM_BITS;
  let field_size = {
    assert!(fr_num_bits <= u32::from(u16::MAX));
    // It's safe to convert to u16 for compatibility with other types.
    fr_num_bits as u16
  };

  generate_constants::<F>(FIELD, SBOX, field_size, t as u16, r_f, r_p)
}

/// Apply the quintic S-Box (s^5) to a given item
pub(crate) fn quintic_s_box<F: PrimeField>(l: &mut F, pre_add: Option<&F>, post_add: Option<&F>) {
  if let Some(x) = pre_add {
    l.add_assign(x);
  }
  let mut tmp = *l;
  tmp = tmp.square(); // l^2
  tmp = tmp.square(); // l^4
  l.mul_assign(&tmp); // l^5
  if let Some(x) = post_add {
    l.add_assign(x);
  }
}

#[derive(Debug, Clone)]
/// Possible error states for the hashing.
pub enum PoseidonError {}
