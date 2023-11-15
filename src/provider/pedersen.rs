//! This module provides an implementation of a commitment engine
use crate::{
  errors::NovaError,
  provider::{CompressedGroup, EngineExt},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    AbsorbInROTrait, Engine, ROTrait, TranscriptReprTrait,
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
pub struct CommitmentKey<E: EngineExt> {
  ck: Vec<E::PreprocessedGroupElement>,
}

impl<E: EngineExt> Len for CommitmentKey<E> {
  fn length(&self) -> usize {
    self.ck.len()
  }
}

/// A type that holds a commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: EngineExt> {
  pub(crate) comm: E,
}

/// A type that holds a compressed commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedCommitment<E: EngineExt> {
  comm: E::CompressedGroupElement,
}

impl<E: EngineExt> CommitmentTrait<E> for Commitment<E> {
  type CompressedCommitment = CompressedCommitment<E>;

  fn compress(&self) -> Self::CompressedCommitment {
    CompressedCommitment {
      comm: self.comm.compress(),
    }
  }

  fn to_coordinates(&self) -> (E::Base, E::Base, bool) {
    self.comm.to_coordinates()
  }

  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError> {
    let comm = <E as EngineExt>::CompressedGroupElement::decompress(&c.comm);
    if comm.is_none() {
      return Err(NovaError::DecompressionError);
    }
    Ok(Commitment {
      comm: comm.unwrap(),
    })
  }
}

impl<E: EngineExt> Default for Commitment<E> {
  fn default() -> Self {
    Commitment { comm: E::zero() }
  }
}

impl<E: EngineExt> TranscriptReprTrait<E> for Commitment<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    let is_infinity_byte = (!is_infinity).into();
    [
      x.to_transcript_bytes(),
      y.to_transcript_bytes(),
      [is_infinity_byte].to_vec(),
    ]
    .concat()
  }
}

impl<E: EngineExt> AbsorbInROTrait<E> for Commitment<E> {
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    ro.absorb(x);
    ro.absorb(y);
    ro.absorb(if is_infinity {
      E::Base::ONE
    } else {
      E::Base::ZERO
    });
  }
}

impl<E: EngineExt> TranscriptReprTrait<E> for CompressedCommitment<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

impl<E: EngineExt> MulAssign<E::Scalar> for Commitment<E> {
  fn mul_assign(&mut self, scalar: E::Scalar) {
    let result = (self as &Commitment<E>).comm * scalar;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b, E: EngineExt> Mul<&'b E::Scalar> for &'a Commitment<E> {
  type Output = Commitment<E>;
  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: EngineExt> Mul<E::Scalar> for Commitment<E> {
  type Output = Commitment<E>;

  fn mul(self, scalar: E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<'b, E: EngineExt> AddAssign<&'b Commitment<E>> for Commitment<E> {
  fn add_assign(&mut self, other: &'b Commitment<E>) {
    let result = (self as &Commitment<E>).comm + other.comm;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b, E: EngineExt> Add<&'b Commitment<E>> for &'a Commitment<E> {
  type Output = Commitment<E>;
  fn add(self, other: &'b Commitment<E>) -> Commitment<E> {
    Commitment {
      comm: self.comm + other.comm,
    }
  }
}

macro_rules! define_add_variants {
  (E = $g:path, LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
    impl<'b, E: $g> Add<&'b $rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: &'b $rhs) -> $out {
        &self + rhs
      }
    }

    impl<'a, E: $g> Add<$rhs> for &'a $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        self + &rhs
      }
    }

    impl<E: $g> Add<$rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        &self + &rhs
      }
    }
  };
}

macro_rules! define_add_assign_variants {
  (E = $g:path, LHS = $lhs:ty, RHS = $rhs:ty) => {
    impl<E: $g> AddAssign<$rhs> for $lhs {
      fn add_assign(&mut self, rhs: $rhs) {
        *self += &rhs;
      }
    }
  };
}

define_add_assign_variants!(E = EngineExt, LHS = Commitment<E>, RHS = Commitment<E>);
define_add_variants!(E = EngineExt, LHS = Commitment<E>, RHS = Commitment<E>, Output = Commitment<E>);

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentEngine<E: EngineExt> {
  _p: PhantomData<E>,
}

impl<E: EngineExt> CommitmentEngineTrait<E> for CommitmentEngine<E> {
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    Self::CommitmentKey {
      ck: E::from_label(label, n.next_power_of_two()),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar]) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());
    Commitment {
      comm: E::vartime_multiscalar_mul(v, &ck.ck[..v.len()]),
    }
  }
}

/// A trait listing properties of a commitment key that can be managed in a divide-and-conquer fashion
pub trait CommitmentKeyExtTrait<E: EngineExt> {
  /// Splits the commitment key into two pieces at a specified point
  fn split_at(&self, n: usize) -> (Self, Self)
  where
    Self: Sized;

  /// Combines two commitment keys into one
  fn combine(&self, other: &Self) -> Self;

  /// Folds the two commitment keys into one using the provided weights
  fn fold(&self, w1: &E::Scalar, w2: &E::Scalar) -> Self;

  /// Scales the commitment key using the provided scalar
  fn scale(&self, r: &E::Scalar) -> Self;

  /// Reinterprets commitments as commitment keys
  fn reinterpret_commitments_as_ck(
    c: &[<<<E as Engine>::CE as CommitmentEngineTrait<E>>::Commitment as CommitmentTrait<E>>::CompressedCommitment],
  ) -> Result<Self, NovaError>
  where
    Self: Sized;
}

impl<E: Engine<CE = CommitmentEngine<E>> + EngineExt> CommitmentKeyExtTrait<E>
  for CommitmentKey<E>
{
  fn split_at(&self, n: usize) -> (CommitmentKey<E>, CommitmentKey<E>) {
    (
      CommitmentKey {
        ck: self.ck[0..n].to_vec(),
      },
      CommitmentKey {
        ck: self.ck[n..].to_vec(),
      },
    )
  }

  fn combine(&self, other: &CommitmentKey<E>) -> CommitmentKey<E> {
    let ck = {
      let mut c = self.ck.clone();
      c.extend(other.ck.clone());
      c
    };
    CommitmentKey { ck }
  }

  // combines the left and right halves of `self` using `w1` and `w2` as the weights
  fn fold(&self, w1: &E::Scalar, w2: &E::Scalar) -> CommitmentKey<E> {
    let w = vec![*w1, *w2];
    let (L, R) = self.split_at(self.ck.len() / 2);

    let ck = (0..self.ck.len() / 2)
      .into_par_iter()
      .map(|i| {
        let bases = [L.ck[i].clone(), R.ck[i].clone()].to_vec();
        E::vartime_multiscalar_mul(&w, &bases).preprocessed()
      })
      .collect();

    CommitmentKey { ck }
  }

  /// Scales each element in `self` by `r`
  fn scale(&self, r: &E::Scalar) -> Self {
    let ck_scaled = self
      .ck
      .clone()
      .into_par_iter()
      .map(|g| E::vartime_multiscalar_mul(&[*r], &[g]).preprocessed())
      .collect();

    CommitmentKey { ck: ck_scaled }
  }

  /// reinterprets a vector of commitments as a set of generators
  fn reinterpret_commitments_as_ck(c: &[CompressedCommitment<E>]) -> Result<Self, NovaError> {
    let d = (0..c.len())
      .into_par_iter()
      .map(|i| Commitment::<E>::decompress(&c[i]))
      .collect::<Result<Vec<Commitment<E>>, NovaError>>()?;
    let ck = (0..d.len())
      .into_par_iter()
      .map(|i| d[i].comm.preprocessed())
      .collect();
    Ok(CommitmentKey { ck })
  }
}
