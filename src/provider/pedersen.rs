//! This module provides an implementation of a commitment engine
use crate::{
  errors::NovaError,
  provider::traits::{CompressedGroup, DlogGroup},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    AbsorbInROTrait, Engine, ROTrait, TranscriptReprTrait,
  },
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, Mul, MulAssign},
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
}

impl<E> Len for CommitmentKey<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn length(&self) -> usize {
    self.ck.len()
  }
}

/// A type that holds a commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: Engine> {
  pub(crate) comm: E::GE,
}

/// A type that holds a compressed commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CompressedCommitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  comm: <E::GE as DlogGroup>::CompressedGroupElement,
}

impl<E> CommitmentTrait<E> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
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
    let comm = <<E as Engine>::GE as DlogGroup>::CompressedGroupElement::decompress(&c.comm)?;
    Ok(Commitment { comm })
  }
}

impl<E: Engine> Default for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn default() -> Self {
    Commitment {
      comm: E::GE::zero(),
    }
  }
}

impl<E> TranscriptReprTrait<E::GE> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
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

impl<E> AbsorbInROTrait<E> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
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

impl<E> TranscriptReprTrait<E::GE> for CompressedCommitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.comm.to_transcript_bytes()
  }
}

impl<E> MulAssign<E::Scalar> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    *self = Commitment {
      comm: self.comm * scalar,
    };
  }
}

impl<'a, 'b, E> Mul<&'b E::Scalar> for &'a Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;
  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E> Mul<E::Scalar> for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E> Add for Commitment<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;

  fn add(self, other: Commitment<E>) -> Commitment<E> {
    Commitment {
      comm: self.comm + other.comm,
    }
  }
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E> CommitmentEngineTrait<E> for CommitmentEngine<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    Self::CommitmentKey {
      ck: E::GE::from_label(label, n.next_power_of_two()),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar]) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());
    Commitment {
      comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()]),
    }
  }
}

/// A trait listing properties of a commitment key that can be managed in a divide-and-conquer fashion
pub trait CommitmentKeyExtTrait<E>
where
  E: Engine,
  E::GE: DlogGroup,
{
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

impl<E> CommitmentKeyExtTrait<E> for CommitmentKey<E>
where
  E: Engine<CE = CommitmentEngine<E>>,
  E::GE: DlogGroup,
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
        E::GE::vartime_multiscalar_mul(&w, &bases).affine()
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
      .map(|g| E::GE::vartime_multiscalar_mul(&[*r], &[g]).affine())
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
      .map(|i| d[i].comm.affine())
      .collect();
    Ok(CommitmentKey { ck })
  }
}
