use super::{
  errors::NovaError,
  traits::{CompressedGroup, Group},
};
use core::{
  fmt::Debug,
  ops::{Add, AddAssign, Mul, MulAssign},
};
use digest::{ExtendableOutput, Input};
use merlin::Transcript;
use sha3::Shake256;
use std::io::Read;

#[derive(Debug)]
pub struct CommitGens<G: Group> {
  gens: Vec<G>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Commitment<G: Group> {
  pub(crate) comm: G,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedCommitment<C: CompressedGroup> {
  comm: C,
}

impl<G: Group> CommitGens<G> {
  pub fn new(label: &[u8], n: usize) -> Self {
    let mut shake = Shake256::default();
    shake.input(label);
    let mut reader = shake.xof_result();
    let mut gens: Vec<G> = Vec::new();
    let mut uniform_bytes = [0u8; 64];
    for _ in 0..n {
      reader.read_exact(&mut uniform_bytes).unwrap();
      gens.push(G::from_uniform_bytes(&uniform_bytes).unwrap());
    }

    CommitGens { gens }
  }
}

impl<G: Group> Commitment<G> {
  pub fn compress(&self) -> CompressedCommitment<G::CompressedGroupElement> {
    CompressedCommitment {
      comm: self.comm.compress(),
    }
  }
}

impl<C: CompressedGroup> CompressedCommitment<C> {
  pub fn decompress(&self) -> Result<Commitment<C::GroupElement>, NovaError> {
    let comm = self.comm.decompress();
    if comm.is_none() {
      return Err(NovaError::DecompressionError);
    }
    Ok(Commitment {
      comm: comm.unwrap(),
    })
  }
}

pub trait CommitTrait<G: Group> {
  fn commit(&self, gens: &CommitGens<G>) -> Commitment<G>;
}

impl<G: Group> CommitTrait<G> for [G::Scalar] {
  fn commit(&self, gens: &CommitGens<G>) -> Commitment<G> {
    assert!(gens.gens.len() >= self.len());
    Commitment {
      comm: G::vartime_multiscalar_mul(self, &gens.gens[..self.len()]),
    }
  }
}

pub trait AppendToTranscriptTrait {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript);
}

impl<G: Group> AppendToTranscriptTrait for Commitment<G> {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_message(label, self.comm.compress().as_bytes());
  }
}

impl<C: CompressedGroup> AppendToTranscriptTrait for CompressedCommitment<C> {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_message(label, self.comm.as_bytes());
  }
}

impl<'b, G: Group> MulAssign<&'b G::Scalar> for Commitment<G> {
  fn mul_assign(&mut self, scalar: &'b G::Scalar) {
    let result = (self as &Commitment<G>).comm * scalar;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b, G: Group> Mul<&'b G::Scalar> for &'a Commitment<G> {
  type Output = Commitment<G>;
  fn mul(self, scalar: &'b G::Scalar) -> Commitment<G> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<G: Group> Mul<G::Scalar> for Commitment<G> {
  type Output = Commitment<G>;

  fn mul(self, scalar: G::Scalar) -> Commitment<G> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<'b, G: Group> AddAssign<&'b Commitment<G>> for Commitment<G> {
  fn add_assign(&mut self, other: &'b Commitment<G>) {
    let result = (self as &Commitment<G>).comm + other.comm;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b, G: Group> Add<&'b Commitment<G>> for &'a Commitment<G> {
  type Output = Commitment<G>;
  fn add(self, other: &'b Commitment<G>) -> Commitment<G> {
    Commitment {
      comm: self.comm + other.comm,
    }
  }
}

macro_rules! define_add_variants {
  (G = $g:path, LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
    impl<'b, G: $g> Add<&'b $rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: &'b $rhs) -> $out {
        &self + rhs
      }
    }

    impl<'a, G: $g> Add<$rhs> for &'a $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        self + &rhs
      }
    }

    impl<G: $g> Add<$rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        &self + &rhs
      }
    }
  };
}

macro_rules! define_add_assign_variants {
  (G = $g:path, LHS = $lhs:ty, RHS = $rhs:ty) => {
    impl<G: $g> AddAssign<$rhs> for $lhs {
      fn add_assign(&mut self, rhs: $rhs) {
        *self += &rhs;
      }
    }
  };
}

define_add_assign_variants!(G = Group, LHS = Commitment<G>, RHS = Commitment<G>);
define_add_variants!(G = Group, LHS = Commitment<G>, RHS = Commitment<G>, Output = Commitment<G>);
