/// This module implements a variant of the 'Secure Sponge API for Field Elements':  https://hackmd.io/bHgsH6mMStCVibM_wYvb2w
///
/// The API is defined by the `SpongeAPI` trait, which is implemented in terms of the `InnerSpongeAPI` trait.
/// `Neptune` provides implementations of `InnerSpongeAPI` for both `sponge::Sponge` and `sponge_circuit::SpongeCircuit`.
use crate::frontend::gadgets::poseidon::poseidon_inner::Arity;
#[cfg(not(feature = "std"))]
use crate::prelude::*;
use ff::PrimeField;
#[derive(Debug)]
pub enum Error {
  ParameterUsageMismatch,
}

/// Sponge operations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpongeOp {
  /// Absorb
  Absorb(u32),
  /// Squeeze
  Squeeze(u32),
}

/// A sequence of sponge operations
#[derive(Clone, Debug)]
pub struct IOPattern(pub Vec<SpongeOp>);

impl IOPattern {
  /// Compute the value of the pattern given a domain separator
  pub fn value(&self, domain_separator: u32) -> u128 {
    let mut hasher = Hasher::new();

    for op in self.0.iter() {
      hasher.update_op(*op);
    }
    hasher.finalize(domain_separator)
  }

  /// Get the operation at a given index
  pub fn op_at(&self, i: usize) -> Option<&SpongeOp> {
    self.0.get(i)
  }
}

// A large 128-bit prime, per https://primes.utm.edu/lists/2small/100bit.html.
const HASHER_BASE: u128 = (0 - 159) as u128;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Hasher {
  x: u128,
  x_i: u128,
  state: u128,
  current_op: SpongeOp,
}

impl Default for Hasher {
  fn default() -> Self {
    Self {
      x: HASHER_BASE,
      x_i: 1,
      state: 0,
      current_op: SpongeOp::Absorb(0),
    }
  }
}

impl Hasher {
  pub(crate) fn new() -> Self {
    Default::default()
  }

  /// Update hasher's current op to coalesce absorb/squeeze runs.
  pub(crate) fn update_op(&mut self, op: SpongeOp) {
    if self.current_op.matches(op) {
      self.current_op = self.current_op.combine(op)
    } else {
      self.finish_op();
      self.current_op = op;
    }
  }

  fn finish_op(&mut self) {
    if self.current_op.count() == 0 {
      return;
    };
    let op_value = self.current_op.value();

    self.update(op_value);
  }

  pub(crate) fn update(&mut self, a: u32) {
    self.x_i = self.x_i.overflowing_mul(self.x).0;
    self.state = self
      .state
      .overflowing_add(self.x_i.overflowing_mul(u128::from(a)).0)
      .0;
  }

  pub(crate) fn finalize(&mut self, domain_separator: u32) -> u128 {
    self.finish_op();
    self.update(domain_separator);
    self.state
  }
}

impl SpongeOp {
  /// Return the count of the SpongeOp
  pub const fn count(&self) -> u32 {
    match self {
      Self::Absorb(n) | Self::Squeeze(n) => *n,
    }
  }

  /// Return true if the SpongeOp is absorb
  pub const fn is_absorb(&self) -> bool {
    matches!(self, Self::Absorb(_))
  }

  /// Combine two SpongeOps
  pub fn combine(&self, other: Self) -> Self {
    assert!(self.matches(other));

    match self {
      Self::Absorb(n) => Self::Absorb(n + other.count()),
      Self::Squeeze(n) => Self::Squeeze(n + other.count()),
    }
  }

  /// Check if two SpongeOps match
  pub const fn matches(&self, other: Self) -> bool {
    self.is_absorb() == other.is_absorb()
  }

  /// Return the value of the SpongeOp
  pub fn value(&self) -> u32 {
    match self {
      Self::Absorb(n) => {
        assert_eq!(0, n >> 31);
        n + (1 << 31)
      }
      Self::Squeeze(n) => {
        assert_eq!(0, n >> 31);
        *n
      }
    }
  }
}

/// Sponge API trait
pub trait SpongeAPI<F: PrimeField, A: Arity<F>> {
  /// Accumulator type
  type Acc;
  /// Value type
  type Value;

  /// Optional `domain_separator` defaults to 0
  fn start(&mut self, p: IOPattern, domain_separator: Option<u32>, _: &mut Self::Acc);
  /// Perform an absorb operation
  fn absorb(&mut self, length: u32, elements: &[Self::Value], acc: &mut Self::Acc);
  /// Perform a squeeze operation
  fn squeeze(&mut self, length: u32, acc: &mut Self::Acc) -> Vec<Self::Value>;
  /// Finish the sponge operation
  fn finish(&mut self, _: &mut Self::Acc) -> Result<(), Error>;
}

pub trait InnerSpongeAPI<F: PrimeField, A: Arity<F>> {
  type Acc;
  type Value;

  fn initialize_capacity(&mut self, tag: u128, acc: &mut Self::Acc);
  fn read_rate_element(&mut self, offset: usize) -> Self::Value;
  fn add_rate_element(&mut self, offset: usize, x: &Self::Value);
  fn permute(&mut self, acc: &mut Self::Acc);

  // Supplemental methods needed for a generic implementation.
  fn rate(&self) -> usize;
  fn absorb_pos(&self) -> usize;
  fn squeeze_pos(&self) -> usize;
  fn set_absorb_pos(&mut self, pos: usize);
  fn set_squeeze_pos(&mut self, pos: usize);

  fn add(a: Self::Value, b: &Self::Value) -> Self::Value;

  fn initialize_state(&mut self, p_value: u128, acc: &mut Self::Acc) {
    self.initialize_capacity(p_value, acc);

    for i in 0..self.rate() {
      self.add_rate_element(i, &Self::zero());
    }
  }

  fn pattern(&self) -> &IOPattern;
  fn set_pattern(&mut self, pattern: IOPattern);

  fn increment_io_count(&mut self) -> usize;

  fn zero() -> Self::Value;
}

impl<F: PrimeField, A: Arity<F>, S: InnerSpongeAPI<F, A>> SpongeAPI<F, A> for S {
  type Acc = <S as InnerSpongeAPI<F, A>>::Acc;
  type Value = <S as InnerSpongeAPI<F, A>>::Value;

  fn start(&mut self, p: IOPattern, domain_separator: Option<u32>, acc: &mut Self::Acc) {
    let p_value = p.value(domain_separator.unwrap_or(0));

    self.set_pattern(p);
    self.initialize_state(p_value, acc);

    self.set_absorb_pos(0);
    self.set_squeeze_pos(0);
  }

  fn absorb(&mut self, length: u32, elements: &[Self::Value], acc: &mut Self::Acc) {
    assert_eq!(length as usize, elements.len());
    let rate = self.rate();

    for element in elements.iter() {
      if self.absorb_pos() == rate {
        self.permute(acc);
        self.set_absorb_pos(0);
      }
      let old = self.read_rate_element(self.absorb_pos());
      self.add_rate_element(self.absorb_pos(), &S::add(old, element));
      self.set_absorb_pos(self.absorb_pos() + 1);
    }
    let op = SpongeOp::Absorb(length);
    let old_count = self.increment_io_count();
    assert_eq!(Some(&op), self.pattern().op_at(old_count));

    self.set_squeeze_pos(rate);
  }

  fn squeeze(&mut self, length: u32, acc: &mut Self::Acc) -> Vec<Self::Value> {
    let rate = self.rate();

    let mut out = Vec::with_capacity(length as usize);

    for _ in 0..length {
      if self.squeeze_pos() == rate {
        self.permute(acc);
        self.set_squeeze_pos(0);
        self.set_absorb_pos(0);
      }
      out.push(self.read_rate_element(self.squeeze_pos()));
      self.set_squeeze_pos(self.squeeze_pos() + 1);
    }
    let op = SpongeOp::Squeeze(length);
    let old_count = self.increment_io_count();
    assert_eq!(Some(&op), self.pattern().op_at(old_count));

    out
  }

  fn finish(&mut self, acc: &mut Self::Acc) -> Result<(), Error> {
    // Clear state.
    self.initialize_state(0, acc);
    let final_io_count = self.increment_io_count();

    if final_io_count == self.pattern().0.len() {
      Ok(())
    } else {
      Err(Error::ParameterUsageMismatch)
    }
  }
}

#[cfg(test)]
mod test {
  use super::*;

  #[test]
  fn test_tag_values() {
    let test = |expected_value: u128, pattern: IOPattern, domain_separator: u32| {
      assert_eq!(expected_value, pattern.value(domain_separator));
    };

    test(0, IOPattern(vec![]), 0);
    test(
      340282366920938463463374607431768191899,
      IOPattern(vec![]),
      123,
    );
    test(
      340282366920938463463374607090318361668,
      IOPattern(vec![SpongeOp::Absorb(2), SpongeOp::Squeeze(2)]),
      0,
    );
    test(
      340282366920938463463374607090314341989,
      IOPattern(vec![SpongeOp::Absorb(2), SpongeOp::Squeeze(2)]),
      1,
    );
    test(
      340282366920938463463374607090318361668,
      IOPattern(vec![SpongeOp::Absorb(2), SpongeOp::Squeeze(2)]),
      0,
    );
    test(
      340282366920938463463374607090318361668,
      IOPattern(vec![
        SpongeOp::Absorb(1),
        SpongeOp::Absorb(1),
        SpongeOp::Squeeze(2),
      ]),
      0,
    );
    test(
      340282366920938463463374607090318361668,
      IOPattern(vec![
        SpongeOp::Absorb(1),
        SpongeOp::Absorb(1),
        SpongeOp::Squeeze(1),
        SpongeOp::Squeeze(1),
      ]),
      0,
    );
  }
}
