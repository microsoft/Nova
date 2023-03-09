//! This module provides an implementation of a commitment engine
use crate::{
  errors::NovaError,
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    AbsorbInROTrait, CompressedGroup, Group, ROTrait, TranscriptReprTrait,
  },
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, AddAssign, Mul, MulAssign},
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<G: Group> {
  ck: Vec<G::PreprocessedGroupElement>,
  _p: PhantomData<G>,
}

/// A type that holds a commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<G: Group> {
  pub(crate) comm: G,
}

/// A type that holds a compressed commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedCommitment<G: Group> {
  comm: G::CompressedGroupElement,
}

impl<G: Group> CommitmentTrait<G> for Commitment<G> {
  type CompressedCommitment = CompressedCommitment<G>;

  fn compress(&self) -> Self::CompressedCommitment {
    CompressedCommitment {
      comm: self.comm.compress(),
    }
  }

  fn to_coordinates(&self) -> (G::Base, G::Base, bool) {
    self.comm.to_coordinates()
  }

  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError> {
    let comm = c.comm.decompress();
    if comm.is_none() {
      return Err(NovaError::DecompressionError);
    }
    Ok(Commitment {
      comm: comm.unwrap(),
    })
  }
}

impl<G: Group> Default for Commitment<G> {
  fn default() -> Self {
    Commitment { comm: G::zero() }
  }
}

impl<G: Group> TranscriptReprTrait<G> for Commitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    let is_infinity_byte = if is_infinity { 0u8 } else { 1u8 };
    [
      x.to_transcript_bytes(),
      y.to_transcript_bytes(),
      [is_infinity_byte].to_vec(),
    ]
    .concat()
  }
}

impl<G: Group> AbsorbInROTrait<G> for Commitment<G> {
  fn absorb_in_ro(&self, ro: &mut G::RO) {
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

impl<G: Group> TranscriptReprTrait<G> for CompressedCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

impl<G: Group> MulAssign<G::Scalar> for Commitment<G> {
  fn mul_assign(&mut self, scalar: G::Scalar) {
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

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentEngine<G: Group> {
  _p: PhantomData<G>,
}

impl<G: Group> CommitmentEngineTrait<G> for CommitmentEngine<G> {
  type CommitmentKey = CommitmentKey<G>;
  type Commitment = Commitment<G>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    Self::CommitmentKey {
      ck: G::from_label(label, n.next_power_of_two()),
      _p: Default::default(),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[G::Scalar]) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());
    Commitment {
      comm: G::vartime_multiscalar_mul(v, &ck.ck[..v.len()]),
    }
  }
}

pub(crate) trait CommitmentKeyExtTrait<G: Group> {
  type CE: CommitmentEngineTrait<G>;

  /// Splits the commitment key into two pieces at a specified point
  fn split_at(&self, n: usize) -> (Self, Self)
  where
    Self: Sized;

  /// Combines two commitment keys into one
  fn combine(&self, other: &Self) -> Self;

  /// Folds the two commitment keys into one using the provided weights
  fn fold(&self, w1: &G::Scalar, w2: &G::Scalar) -> Self;

  /// Scales the commitment key using the provided scalar
  fn scale(&self, r: &G::Scalar) -> Self;

  /// Reinterprets commitments as commitment keys
  fn reinterpret_commitments_as_ck(
    c: &[<<<Self as CommitmentKeyExtTrait<G>>::CE as CommitmentEngineTrait<G>>::Commitment as CommitmentTrait<G>>::CompressedCommitment],
  ) -> Result<Self, NovaError>
  where
    Self: Sized;
}

impl<G: Group> CommitmentKeyExtTrait<G> for CommitmentKey<G> {
  type CE = CommitmentEngine<G>;

  fn split_at(&self, n: usize) -> (CommitmentKey<G>, CommitmentKey<G>) {
    (
      CommitmentKey {
        ck: self.ck[0..n].to_vec(),
        _p: Default::default(),
      },
      CommitmentKey {
        ck: self.ck[n..].to_vec(),
        _p: Default::default(),
      },
    )
  }

  fn combine(&self, other: &CommitmentKey<G>) -> CommitmentKey<G> {
    let ck = {
      let mut c = self.ck.clone();
      c.extend(other.ck.clone());
      c
    };
    CommitmentKey {
      ck,
      _p: Default::default(),
    }
  }

  // combines the left and right halves of `self` using `w1` and `w2` as the weights
  fn fold(&self, w1: &G::Scalar, w2: &G::Scalar) -> CommitmentKey<G> {
    let w = vec![*w1, *w2];
    let (L, R) = self.split_at(self.ck.len() / 2);

    let ck = (0..self.ck.len() / 2)
      .into_par_iter()
      .map(|i| {
        let bases = [L.ck[i].clone(), R.ck[i].clone()].to_vec();
        G::vartime_multiscalar_mul(&w, &bases).preprocessed()
      })
      .collect();

    CommitmentKey {
      ck,
      _p: Default::default(),
    }
  }

  /// Scales each element in `self` by `r`
  fn scale(&self, r: &G::Scalar) -> Self {
    let ck_scaled = self
      .ck
      .clone()
      .into_par_iter()
      .map(|g| G::vartime_multiscalar_mul(&[*r], &[g]).preprocessed())
      .collect();

    CommitmentKey {
      ck: ck_scaled,
      _p: Default::default(),
    }
  }

  /// reinterprets a vector of commitments as a set of generators
  fn reinterpret_commitments_as_ck(c: &[CompressedCommitment<G>]) -> Result<Self, NovaError> {
    let d = (0..c.len())
      .into_par_iter()
      .map(|i| Commitment::<G>::decompress(&c[i]))
      .collect::<Result<Vec<Commitment<G>>, NovaError>>()?;
    let ck = (0..d.len())
      .into_par_iter()
      .map(|i| d[i].comm.preprocessed())
      .collect();
    Ok(CommitmentKey {
      ck,
      _p: Default::default(),
    })
  }
}
