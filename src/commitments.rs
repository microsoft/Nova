use super::{
  errors::NovaError,
  traits::{AbsorbInROTrait, AppendToTranscriptTrait, CompressedGroup, Group, HashFuncTrait},
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, AddAssign, Mul, MulAssign},
};
use ff::Field;
use merlin::Transcript;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
pub struct CommitGens<G: Group> {
  gens: Vec<G::PreprocessedGroupElement>,
  _p: PhantomData<G>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment<G: Group> {
  pub(crate) comm: G,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedCommitment<C: CompressedGroup> {
  pub(crate) comm: C,
}

impl<G: Group> CommitGens<G> {
  pub fn new(label: &'static [u8], n: usize) -> Self {
    CommitGens {
      gens: G::from_label(label, n.next_power_of_two()),
      _p: Default::default(),
    }
  }

  pub fn len(&self) -> usize {
    self.gens.len()
  }

  pub fn split_at(&self, n: usize) -> (CommitGens<G>, CommitGens<G>) {
    (
      CommitGens {
        gens: self.gens[0..n].to_vec(),
        _p: Default::default(),
      },
      CommitGens {
        gens: self.gens[n..].to_vec(),
        _p: Default::default(),
      },
    )
  }

  pub fn combine(&self, other: &CommitGens<G>) -> CommitGens<G> {
    let gens = {
      let mut c = self.gens.clone();
      c.extend(other.gens.clone());
      c
    };
    CommitGens {
      gens,
      _p: Default::default(),
    }
  }

  // combines the left and right halves of `self` using `w1` and `w2` as the weights
  pub fn fold(&mut self, w1: &G::Scalar, w2: &G::Scalar) {
    let w = vec![*w1, *w2];
    let (L, R) = self.split_at(self.len() / 2);

    let gens = (0..self.len() / 2)
      .into_par_iter()
      .map(|i| {
        let gens = CommitGens::<G> {
          gens: [L.gens[i].clone(), R.gens[i].clone()].to_vec(),
          _p: Default::default(),
        };
        w.commit(&gens).comm.preprocessed()
      })
      .collect();

    self.gens = gens;
  }

  /// returns a singleton vector of generators where the entry is r * G, where G is the generator of the group
  pub fn from_scalar(r: &G::Scalar) -> Self {
    CommitGens {
      gens: vec![G::gen().mul(r).preprocessed()],
      _p: Default::default(),
    }
  }

  /// reinterprets a vector of commitments as a set of generators
  pub fn reinterpret_commitments_as_gens(
    c: &[CompressedCommitment<G::CompressedGroupElement>],
  ) -> Result<Self, NovaError> {
    let d = (0..c.len())
      .into_par_iter()
      .map(|i| c[i].decompress())
      .collect::<Result<Vec<Commitment<G>>, NovaError>>()?;
    let gens = (0..d.len())
      .into_par_iter()
      .map(|i| d[i].comm.preprocessed())
      .collect();
    Ok(CommitGens {
      gens,
      _p: Default::default(),
    })
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

impl<G: Group> AppendToTranscriptTrait for Commitment<G> {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_message(label, self.comm.compress().as_bytes());
  }
}

impl<G: Group> AbsorbInROTrait<G> for Commitment<G> {
  fn absorb_in_ro(&self, ro: &mut G::HashFunc) {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    ro.absorb(x);
    ro.absorb(y);
    ro.absorb(if is_infinity {
      G::Base::one()
    } else {
      G::Base::zero()
    });
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
