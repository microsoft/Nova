/// `HashType` provides support for domain separation tags.
/// For 128-bit security, we need to reserve one (~256-bit) field element per Poseidon permutation.
/// This element cannot be used for hash preimage data — but can be assigned a constant value designating
/// the hash function built on top of the underlying permutation.
///
/// `neptune` implements a variation of the domain separation tag scheme suggested in the updated Poseidon paper. This
/// allows for a variety of modes. This ensures that digest values produced using one hash function cannot be reused
/// where another is required.
///
/// Because `neptune` also supports a first-class notion of `Strength`, we include a mechanism for composing
/// `Strength` with `HashType` so that hashes with `Strength` other than `Standard` (currently only `Strengthened`)
/// may still express the full range of hash function types.
use ff::PrimeField;
use serde::{Deserialize, Serialize};

use super::poseidon_inner::Arity;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: Deserialize<'de>"))]
pub enum HashType<F: PrimeField, A: Arity<F>> {
  MerkleTree,
  MerkleTreeSparse(u64),
  VariableLength,
  ConstantLength(usize),
  Encryption,
  Custom(CType<F, A>),
  Sponge,
}

impl<F: PrimeField, A: Arity<F>> HashType<F, A> {
  /// Implements domain separation defined in original [Poseidon paper](https://eprint.iacr.org/2019/458.pdf).
  /// Calculates field element used as a zero element in the Poseidon buffer that holds preimage.
  pub fn domain_tag(&self) -> F {
    match self {
      // 2^arity - 1
      HashType::MerkleTree => A::tag(),
      // bitmask
      HashType::MerkleTreeSparse(bitmask) => F::from(*bitmask),
      // 2^64
      HashType::VariableLength => pow2::<F>(64),
      // length * 2^64
      // length of 0 denotes a duplex sponge
      HashType::ConstantLength(length) => x_pow2::<F>(*length as u64, 64),
      // 2^32 or (2^32 + 2^32 = 2^33) with strength tag
      HashType::Encryption => pow2::<F>(32),
      // identifier * 2^40
      // identifier must be in range [1..=256]
      // If identifier == 0 then the strengthened version collides with Encryption with standard strength.
      // NOTE: in order to leave room for future `Strength` tags,
      // we make identifier a multiple of 2^40 rather than 2^32.
      HashType::Custom(ref ctype) => ctype.domain_tag(),
      HashType::Sponge => F::ZERO,
    }
  }

  /// Some HashTypes require more testing so are not yet supported, since they are not yet needed.
  /// As and when needed, support can be added, along with tests to ensure the initial implementation
  /// is sound.
  pub const fn is_supported(&self) -> bool {
    match self {
      HashType::MerkleTreeSparse(_) | HashType::VariableLength => false,
      HashType::MerkleTree
      | HashType::ConstantLength(_)
      | HashType::Encryption
      | HashType::Custom(_)
      | HashType::Sponge => true,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CType<F: PrimeField, A: Arity<F>> {
  Arbitrary(u64),
  // See: https://github.com/bincode-org/bincode/issues/424
  // This is a bit of a hack, but since `serde(skip)` tags the last variant arm,
  // the generated code ends up being correct. But, in the future, do not
  // carelessly add new variants to this enum.
  #[serde(skip)]
  _Phantom((F, A)),
}

impl<F: PrimeField, A: Arity<F>> CType<F, A> {
  const fn identifier(&self) -> u64 {
    match self {
      CType::Arbitrary(id) => *id,
      CType::_Phantom(_) => panic!("_Phantom is not a real custom tag type."),
    }
  }

  fn domain_tag(&self) -> F {
    let id = self.identifier();
    assert!(id > 0, "custom domain tag id out of range");
    assert!(id <= 256, "custom domain tag id out of range");

    x_pow2::<F>(id, 40)
  }
}

/// pow2(n) = 2^n
fn pow2<F: PrimeField>(n: u64) -> F {
  F::from(2).pow_vartime([n])
}

/// x_pow2(x, n) = x * 2^n
fn x_pow2<F: PrimeField>(coeff: u64, n: u64) -> F {
  let mut tmp = pow2::<F>(n);
  tmp.mul_assign(F::from(coeff));
  tmp
}
