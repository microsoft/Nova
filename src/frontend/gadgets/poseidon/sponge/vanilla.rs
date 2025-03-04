use crate::frontend::gadgets::poseidon::{
  hash_type::HashType,
  poseidon_inner::{Arity, Poseidon, PoseidonConstants},
  sponge::api::{IOPattern, InnerSpongeAPI},
  PoseidonError, Strength,
};
#[cfg(not(feature = "std"))]
use crate::prelude::*;
use ff::PrimeField;
#[cfg(feature = "std")]
use std::collections::VecDeque;

// General information on sponge construction: https://keccak.team/files/CSF-0.1.pdf

/*
A sponge can be instantiated in either simplex or duplex mode. Once instantiated, a sponge's mode never changes.

At any time, a sponge is operating in one of two directions: squeezing or absorbing. All sponges are initialized in the
absorbing direction. The number of absorbed field elements is incremented each time an element is absorbed and
decremented each time an element is squeezed. In duplex mode, the count of currently absorbed elements can never
decrease below zero, so only as many elements as have been absorbed can be squeezed at any time. In simplex mode, there
is no limit on the number of elements that can be squeezed, once absorption is complete.

In simplex mode, absorbing and squeezing cannot be interleaved. First all elements are absorbed, then all needed
elements are squeezed. At most the number of elements which were absorbed can be squeezed. Elements must be absorbed in
chunks of R (rate). After every R chunks have been absorbed, the state is permuted. After the final element has been
absorbed, any needed padding is added, and the final permutation (or two -- if required by padding) is performed. Then
groups of R field elements are squeezed, and the state is permuted after each group of R elements has been squeezed.
After squeezing is complete, a simplex sponge is exhausted, and no further absorption is possible.

In duplex mode, absorbing and squeezing can be interleaved. The state is permuted after every R elements have been
absorbed. This makes R elements available to be squeezed. If elements remain to be squeezed when the state is permuted,
remaining unsqueezed elements are queued. Otherwise they would be lost when permuting.

*/

/// Mode of the sponge
#[derive(Clone, Copy)]
pub enum Mode {
  /// Simplex mode
  Simplex,
}

#[derive(Clone, Copy)]
pub enum Direction {
  Absorbing,
  Squeezing,
}

/// Poseidon sponge
pub struct Sponge<'a, F: PrimeField, A: Arity<F>> {
  absorbed: usize,
  squeezed: usize,
  /// Poseidon state
  pub state: Poseidon<'a, F, A>,
  mode: Mode,
  direction: Direction,
  squeeze_pos: usize,
  queue: VecDeque<F>,
  pattern: IOPattern,
  io_count: usize,
}

/// Sponge trait
pub trait SpongeTrait<'a, F: PrimeField, A: Arity<F>>
where
  Self: Sized,
{
  /// Accumulator type
  type Acc;
  /// Value type
  type Elt;
  /// Error type
  type Error;

  /// Create a new sponge with the given constants and mode
  fn new_with_constants(constants: &'a PoseidonConstants<F, A>, mode: Mode) -> Self;

  /// Return API constants
  fn api_constants(strength: Strength) -> PoseidonConstants<F, A> {
    PoseidonConstants::new_with_strength_and_type(strength, HashType::Sponge)
  }

  /// Return the mode of the sponge
  fn mode(&self) -> Mode;
  /// Return the direction of the sponge
  fn direction(&self) -> Direction;
  /// Set the direction of the sponge
  fn set_direction(&mut self, direction: Direction);
  /// Return the number of absorbed elements
  fn absorbed(&self) -> usize;
  /// Return the number of squeezed elements
  fn squeezed(&self) -> usize;
  /// Set the number of squeezed elements
  fn set_squeezed(&mut self, squeezed: usize);
  /// Return the squeeze position
  fn squeeze_pos(&self) -> usize;
  /// Set the squeeze position
  fn set_squeeze_pos(&mut self, squeeze_pos: usize);
  /// Return the absorb position
  fn absorb_pos(&self) -> usize;
  /// Set the absorb position
  fn set_absorb_pos(&mut self, pos: usize);

  /// Return the element at the given index
  fn element(&self, index: usize) -> Self::Elt;
  /// Set the element at the given index
  fn set_element(&mut self, index: usize, elt: Self::Elt);

  /// Return whether the sponge is in simplex mode
  fn is_simplex(&self) -> bool {
    match self.mode() {
      Mode::Simplex => true,
    }
  }
  /// Return whether the sponge is in duplex mode
  fn is_duplex(&self) -> bool {
    match self.mode() {
      Mode::Simplex => false,
    }
  }

  /// Return whether the sponge is absorbing
  fn is_absorbing(&self) -> bool {
    match self.direction() {
      Direction::Absorbing => true,
      Direction::Squeezing => false,
    }
  }

  /// Return the number of available elements
  fn available(&self) -> usize {
    self.absorbed() - self.squeezed()
  }

  /// Return the rate of the sponge
  fn rate(&self) -> usize;

  /// Return the capacity of the sponge
  fn capacity(&self) -> usize;

  /// Return the size of the sponge
  fn size(&self) -> usize;

  /// Return the total size of the sponge
  fn total_size(&self) -> usize {
    assert!(self.is_simplex());
    match self.constants().hash_type {
      HashType::ConstantLength(l) => l,
      HashType::VariableLength => unimplemented!(),
      _ => A::to_usize(),
    }
  }

  /// Return the constants of the sponge
  fn constants(&self) -> &PoseidonConstants<F, A>;

  /// Return whether the sponge can squeeze without permuting
  fn can_squeeze_without_permuting(&self) -> bool {
    self.squeeze_pos() < self.size() - self.capacity()
  }

  /// Permute the sponge
  fn permute(&mut self, acc: &mut Self::Acc) -> Result<(), Self::Error> {
    // NOTE: this will apply any needed padding in the partially-absorbed case.
    // However, padding should only be applied when no more elements will be absorbed.
    // A duplex sponge should never apply padding implicitly, and a simplex sponge should only do so when it is
    // about to apply its final permutation.
    let unpermuted = self.absorb_pos();
    let needs_padding = self.is_absorbing() && unpermuted < self.rate();

    if needs_padding {
      match self.mode() {
        Mode::Simplex => {
          let final_permutation = self.squeezed() % self.total_size() <= self.rate();
          assert!(
            final_permutation,
            "Simplex sponge may only pad before final permutation"
          );
          self.pad();
        }
      }
    }

    self.permute_state(acc)?;
    self.set_absorb_pos(0);
    self.set_squeeze_pos(0);
    Ok(())
  }

  /// permutate the sponge state
  fn pad(&mut self);

  /// Permute the sponge state
  fn permute_state(&mut self, acc: &mut Self::Acc) -> Result<(), Self::Error>;

  /// Ensure the sponge is squeezing
  fn ensure_squeezing(&mut self, acc: &mut Self::Acc) -> Result<(), Self::Error> {
    match self.direction() {
      Direction::Squeezing => (),
      Direction::Absorbing => {
        match self.mode() {
          Mode::Simplex => {
            let done_squeezing_previous = self.squeeze_pos() >= self.rate();
            let partially_absorbed = self.absorb_pos() > 0;

            if done_squeezing_previous || partially_absorbed {
              self.permute(acc)?;
            }
          }
        }
        self.set_direction(Direction::Squeezing);
      }
    }
    Ok(())
  }

  /// Squeeze Aux
  fn squeeze_aux(&mut self) -> Self::Elt;

  /// Perform a squeeze operation
  fn squeeze(&mut self, acc: &mut Self::Acc) -> Result<Option<Self::Elt>, Self::Error> {
    self.ensure_squeezing(acc)?;

    if self.is_duplex() && self.available() == 0 {
      // What has not yet been absorbed cannot be squeezed.
      return Ok(None);
    };

    self.set_squeezed(self.squeezed() + 1);

    if let Some(queued) = self.dequeue() {
      return Ok(Some(queued));
    }

    if !self.can_squeeze_without_permuting() && self.is_simplex() {
      self.permute(acc)?;
    }

    let squeezed = self.squeeze_aux();

    Ok(Some(squeezed))
  }

  /// Dequeue an element
  fn dequeue(&mut self) -> Option<Self::Elt>;
}

impl<'a, F: PrimeField, A: Arity<F>> SpongeTrait<'a, F, A> for Sponge<'a, F, A> {
  type Acc = ();
  type Elt = F;
  type Error = PoseidonError;

  fn new_with_constants(constants: &'a PoseidonConstants<F, A>, mode: Mode) -> Self {
    let poseidon = Poseidon::new(constants);

    Self {
      mode,
      direction: Direction::Absorbing,
      state: poseidon,
      absorbed: 0,
      squeezed: 0,
      squeeze_pos: 0,
      queue: VecDeque::with_capacity(A::to_usize()),
      pattern: IOPattern(Vec::new()),
      io_count: 0,
    }
  }

  fn mode(&self) -> Mode {
    self.mode
  }
  fn direction(&self) -> Direction {
    self.direction
  }
  fn set_direction(&mut self, direction: Direction) {
    self.direction = direction;
  }
  fn absorbed(&self) -> usize {
    self.absorbed
  }

  fn squeezed(&self) -> usize {
    self.squeezed
  }
  fn set_squeezed(&mut self, squeezed: usize) {
    self.squeezed = squeezed;
  }
  fn squeeze_pos(&self) -> usize {
    self.squeeze_pos
  }
  fn set_squeeze_pos(&mut self, squeeze_pos: usize) {
    self.squeeze_pos = squeeze_pos;
  }
  fn absorb_pos(&self) -> usize {
    self.state.pos - 1
  }
  fn set_absorb_pos(&mut self, pos: usize) {
    self.state.pos = pos + 1;
  }

  fn element(&self, index: usize) -> Self::Elt {
    self.state.elements[index]
  }
  fn set_element(&mut self, index: usize, elt: Self::Elt) {
    self.state.elements[index] = elt;
  }

  fn rate(&self) -> usize {
    A::to_usize()
  }

  fn capacity(&self) -> usize {
    1
  }

  fn size(&self) -> usize {
    self.state.constants.width()
  }

  fn constants(&self) -> &PoseidonConstants<F, A> {
    self.state.constants
  }

  fn pad(&mut self) {
    self.state.apply_padding();
  }

  fn permute_state(&mut self, _acc: &mut Self::Acc) -> Result<(), Self::Error> {
    self.state.hash();
    Ok(())
  }

  fn dequeue(&mut self) -> Option<Self::Elt> {
    self.queue.pop_front()
  }

  fn squeeze_aux(&mut self) -> Self::Elt {
    let squeezed = self.element(SpongeTrait::squeeze_pos(self) + SpongeTrait::capacity(self));
    SpongeTrait::set_squeeze_pos(self, SpongeTrait::squeeze_pos(self) + 1);

    squeezed
  }
}

impl<F: PrimeField, A: Arity<F>> Iterator for Sponge<'_, F, A> {
  type Item = F;

  fn next(&mut self) -> Option<F> {
    self.squeeze(&mut ()).unwrap_or(None)
  }

  fn size_hint(&self) -> (usize, Option<usize>) {
    match self.mode {
      Mode::Simplex => (0, None),
    }
  }
}

impl<F: PrimeField, A: Arity<F>> InnerSpongeAPI<F, A> for Sponge<'_, F, A> {
  type Acc = ();
  type Value = F;

  fn initialize_capacity(&mut self, tag: u128, _: &mut ()) {
    let mut repr = F::Repr::default();
    repr.as_mut()[..16].copy_from_slice(&tag.to_le_bytes());

    let f = F::from_repr(repr).unwrap();
    self.set_element(0, f);
  }

  fn read_rate_element(&mut self, offset: usize) -> F {
    self.element(offset + SpongeTrait::capacity(self))
  }
  fn add_rate_element(&mut self, offset: usize, x: &F) {
    self.set_element(offset + SpongeTrait::capacity(self), *x);
  }
  fn permute(&mut self, acc: &mut ()) {
    SpongeTrait::permute(self, acc).unwrap();
  }

  // Supplemental methods needed for a generic implementation.

  fn zero() -> F {
    F::ZERO
  }

  fn rate(&self) -> usize {
    SpongeTrait::rate(self)
  }
  fn absorb_pos(&self) -> usize {
    SpongeTrait::absorb_pos(self)
  }
  fn squeeze_pos(&self) -> usize {
    SpongeTrait::squeeze_pos(self)
  }
  fn set_absorb_pos(&mut self, pos: usize) {
    SpongeTrait::set_absorb_pos(self, pos);
  }
  fn set_squeeze_pos(&mut self, pos: usize) {
    SpongeTrait::set_squeeze_pos(self, pos);
  }
  fn add(a: F, b: &F) -> F {
    a + b
  }

  fn pattern(&self) -> &IOPattern {
    &self.pattern
  }

  fn set_pattern(&mut self, pattern: IOPattern) {
    self.pattern = pattern
  }

  fn increment_io_count(&mut self) -> usize {
    let old_count = self.io_count;
    self.io_count += 1;
    old_count
  }
}
