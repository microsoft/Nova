use crate::frontend::{
  gadgets::poseidon::{
    circuit2::{Elt, PoseidonCircuit2},
    poseidon_inner::{Arity, Poseidon, PoseidonConstants},
    sponge::{
      api::{IOPattern, InnerSpongeAPI},
      vanilla::{Direction, Mode, SpongeTrait},
    },
  },
  util_cs::witness_cs::SizedWitness,
  ConstraintSystem, Namespace, SynthesisError,
};
#[cfg(not(feature = "std"))]
use crate::prelude::*;

use ff::PrimeField;
#[cfg(feature = "std")]
use std::{collections::VecDeque, marker::PhantomData};

/// The Poseidon sponge circuit
pub struct SpongeCircuit<'a, F, A, C>
where
  F: PrimeField,
  A: Arity<F>,
  C: ConstraintSystem<F>,
{
  constants: &'a PoseidonConstants<F, A>,
  mode: Mode,
  direction: Direction,
  absorbed: usize,
  squeezed: usize,
  squeeze_pos: usize,
  permutation_count: usize,
  state: PoseidonCircuit2<'a, F, A>,
  queue: VecDeque<Elt<F>>,
  pattern: IOPattern,
  io_count: usize,
  poseidon: Poseidon<'a, F, A>,
  _c: PhantomData<C>,
}

impl<'a, F: PrimeField, A: Arity<F>, CS: 'a + ConstraintSystem<F>> SpongeTrait<'a, F, A>
  for SpongeCircuit<'a, F, A, CS>
{
  type Acc = Namespace<'a, F, CS>;
  type Elt = Elt<F>;
  type Error = SynthesisError;

  fn new_with_constants(constants: &'a PoseidonConstants<F, A>, mode: Mode) -> Self {
    Self {
      mode,
      direction: Direction::Absorbing,
      constants,
      absorbed: 0,
      squeezed: 0,
      squeeze_pos: 0,
      permutation_count: 0,
      state: PoseidonCircuit2::new_empty::<CS>(constants),
      queue: VecDeque::with_capacity(A::to_usize()),
      pattern: IOPattern(Vec::new()),
      poseidon: Poseidon::new(constants),
      io_count: 0,
      _c: Default::default(),
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
    self.state.elements[index].clone()
  }

  fn set_element(&mut self, index: usize, elt: Self::Elt) {
    self.poseidon.elements[index] = elt.val().unwrap();
    self.state.elements[index] = elt;
  }

  fn rate(&self) -> usize {
    A::to_usize()
  }

  fn capacity(&self) -> usize {
    1
  }

  fn size(&self) -> usize {
    self.constants.width()
  }

  fn constants(&self) -> &PoseidonConstants<F, A> {
    self.constants
  }

  fn pad(&mut self) {
    self.state.apply_padding::<CS>();
  }

  fn permute_state(&mut self, ns: &mut Self::Acc) -> Result<(), Self::Error> {
    self.permutation_count += 1;

    if ns.is_witness_generator() {
      self.poseidon.generate_witness_into_cs(ns);

      for (elt, scalar) in self
        .state
        .elements
        .iter_mut()
        .zip(self.poseidon.elements.iter())
      {
        *elt = Elt::num_from_fr::<CS>(*scalar);
      }
    } else {
      self
        .state
        .hash(&mut ns.namespace(|| format!("permutation {}", self.permutation_count)))?;
    };

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

impl<'a, F: PrimeField, A: Arity<F>, CS: 'a + ConstraintSystem<F>> InnerSpongeAPI<F, A>
  for SpongeCircuit<'a, F, A, CS>
{
  type Acc = Namespace<'a, F, CS>;
  type Value = Elt<F>;

  fn initialize_capacity(&mut self, tag: u128, _acc: &mut Self::Acc) {
    let mut repr = F::Repr::default();
    repr.as_mut()[..16].copy_from_slice(&tag.to_le_bytes());

    let f = F::from_repr(repr).unwrap();
    let elt = Elt::num_from_fr::<Self::Acc>(f);
    self.set_element(0, elt);
  }

  fn read_rate_element(&mut self, offset: usize) -> Self::Value {
    self.element(offset + SpongeTrait::capacity(self))
  }
  fn add_rate_element(&mut self, offset: usize, x: &Self::Value) {
    self.set_element(offset + SpongeTrait::capacity(self), x.clone());
  }
  fn permute(&mut self, acc: &mut Self::Acc) {
    SpongeTrait::permute(self, acc).unwrap();
  }

  // Supplemental methods needed for a generic implementation.

  fn zero() -> Elt<F> {
    Elt::num_from_fr::<CS>(F::ZERO)
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
  fn add(a: Elt<F>, b: &Elt<F>) -> Elt<F> {
    a.add_ref(b).unwrap()
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
