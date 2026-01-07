//! This module provides an implementation of a commitment engine
#[cfg(feature = "io")]
use crate::provider::ptau::{read_points, write_points, PtauFileError};
use crate::{
  errors::NovaError,
  gadgets::utils::to_bignat_repr,
  provider::traits::{DlogGroup, DlogGroupExt},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    AbsorbInRO2Trait, AbsorbInROTrait, Engine, ROTrait, TranscriptReprTrait,
  },
};
use core::{
  fmt::Debug,
  marker::PhantomData,
  ops::{Add, Mul, MulAssign, Range},
};
use ff::Field;
use num_integer::Integer;
use num_traits::ToPrimitive;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "io")]
const KEY_FILE_HEAD: [u8; 12] = *b"PEDERSEN_KEY";

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<E: Engine>
where
  E::GE: DlogGroup,
{
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

impl<E: Engine> Len for CommitmentKey<E>
where
  E::GE: DlogGroup,
{
  fn length(&self) -> usize {
    self.ck.len()
  }
}

/// A type that holds blinding generator
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerandKey<E: Engine>
where
  E::GE: DlogGroup,
{
  h: <E::GE as DlogGroup>::AffineGroupElement,
}

/// A type that holds a commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: Engine> {
  pub(crate) comm: E::GE,
}

impl<E: Engine> CommitmentTrait<E> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn to_coordinates(&self) -> (E::Base, E::Base, bool) {
    self.comm.to_coordinates()
  }
}

impl<E: Engine> Default for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn default() -> Self {
    Commitment {
      comm: E::GE::zero(),
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for Commitment<E>
where
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

impl<E: Engine> AbsorbInROTrait<E> for Commitment<E>
where
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

impl<E: Engine> AbsorbInRO2Trait<E> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn absorb_in_ro2(&self, ro: &mut E::RO2) {
    let (x, y, is_infinity) = self.comm.to_coordinates();

    // we have to absorb x and y in big num format
    let limbs_x = to_bignat_repr(&x);
    let limbs_y = to_bignat_repr(&y);

    for limb in limbs_x.iter().chain(limbs_y.iter()) {
      ro.absorb(*limb);
    }
    ro.absorb(if is_infinity {
      E::Scalar::ONE
    } else {
      E::Scalar::ZERO
    });
  }
}

impl<E: Engine> MulAssign<E::Scalar> for Commitment<E>
where
  E::GE: DlogGroup,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    *self = Commitment {
      comm: self.comm * scalar,
    };
  }
}

impl<'b, E: Engine> Mul<&'b E::Scalar> for &'_ Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;
  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Mul<E::Scalar> for Commitment<E>
where
  E::GE: DlogGroup,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<E: Engine> Add for Commitment<E>
where
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

impl<E: Engine> CommitmentEngineTrait<E> for CommitmentEngine<E>
where
  E::GE: DlogGroupExt,
{
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;
  type DerandKey = DerandKey<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let gens = E::GE::from_label(label, n.next_power_of_two() + 1);

    let (h, ck) = gens.split_first().unwrap();

    Self::CommitmentKey {
      ck: ck.to_vec(),
      h: *h,
    }
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    Self::DerandKey { h: ck.h }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &E::Scalar) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    Commitment {
      comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()])
        + <E::GE as DlogGroup>::group(&ck.h) * r,
    }
  }

  fn commit_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    ck: &Self::CommitmentKey,
    v: &[T],
    r: &E::Scalar,
  ) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    Commitment {
      comm: E::GE::vartime_multiscalar_mul_small(v, &ck.ck[..v.len()])
        + <E::GE as DlogGroup>::group(&ck.h) * r,
    }
  }

  fn commit_small_range<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    ck: &Self::CommitmentKey,
    v: &[T],
    r: &<E as Engine>::Scalar,
    range: Range<usize>,
    max_num_bits: usize,
  ) -> Self::Commitment {
    let bases = &ck.ck[range.clone()];
    let scalars = &v[range];

    assert!(bases.len() == scalars.len());

    let mut res =
      E::GE::vartime_multiscalar_mul_small_with_max_num_bits(scalars, bases, max_num_bits);

    if r != &E::Scalar::ZERO {
      res += <E::GE as DlogGroup>::group(&ck.h) * r;
    }

    Commitment { comm: res }
  }

  fn derandomize(
    dk: &Self::DerandKey,
    commit: &Self::Commitment,
    r: &E::Scalar,
  ) -> Self::Commitment {
    Commitment {
      comm: commit.comm - <E::GE as DlogGroup>::group(&dk.h) * r,
    }
  }

  #[cfg(feature = "io")]
  fn load_setup(
    reader: &mut (impl std::io::Read + std::io::Seek),
    _label: &'static [u8],
    n: usize,
  ) -> Result<Self::CommitmentKey, PtauFileError> {
    let num = n.next_power_of_two();
    {
      let mut head = [0u8; 12];
      reader.read_exact(&mut head)?;
      if head != KEY_FILE_HEAD {
        return Err(PtauFileError::InvalidHead);
      }
    }

    let points = read_points(reader, num + 1)?;

    let (first, second) = points.split_at(1);

    Ok(Self::CommitmentKey {
      ck: second.to_vec(),
      h: first[0],
    })
  }

  #[cfg(feature = "io")]
  fn save_setup(
    ck: &Self::CommitmentKey,
    writer: &mut impl std::io::Write,
  ) -> Result<(), PtauFileError> {
    writer.write_all(&KEY_FILE_HEAD)?;
    let mut points = Vec::with_capacity(ck.ck.len() + 1);
    points.push(ck.h);
    points.extend(ck.ck.iter().cloned());
    write_points(writer, points)
  }
}

/// A trait listing properties of a commitment key that can be managed in a divide-and-conquer fashion
pub trait CommitmentKeyExtTrait<E: Engine>
where
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
    c: &[<E::CE as CommitmentEngineTrait<E>>::Commitment],
  ) -> Result<Self, NovaError>
  where
    Self: Sized;
}

impl<E: Engine<CE = CommitmentEngine<E>>> CommitmentKeyExtTrait<E> for CommitmentKey<E>
where
  E::GE: DlogGroupExt,
{
  fn split_at(&self, n: usize) -> (CommitmentKey<E>, CommitmentKey<E>) {
    (
      CommitmentKey {
        ck: self.ck[0..n].to_vec(),
        h: self.h,
      },
      CommitmentKey {
        ck: self.ck[n..].to_vec(),
        h: self.h,
      },
    )
  }

  fn combine(&self, other: &CommitmentKey<E>) -> CommitmentKey<E> {
    let ck = {
      let mut c = self.ck.clone();
      c.extend(other.ck.clone());
      c
    };
    CommitmentKey { ck, h: self.h }
  }

  // combines the left and right halves of `self` using `w1` and `w2` as the weights
  fn fold(&self, w1: &E::Scalar, w2: &E::Scalar) -> CommitmentKey<E> {
    let w = vec![*w1, *w2];
    let (L, R) = self.split_at(self.ck.len() / 2);

    let ck = (0..self.ck.len() / 2)
      .into_par_iter()
      .map(|i| {
        let bases = [L.ck[i], R.ck[i]].to_vec();
        E::GE::vartime_multiscalar_mul(&w, &bases).affine()
      })
      .collect();

    CommitmentKey { ck, h: self.h }
  }

  /// Scales each element in `self` by `r`
  fn scale(&self, r: &E::Scalar) -> Self {
    let ck_scaled = self
      .ck
      .clone()
      .into_par_iter()
      .map(|g| E::GE::vartime_multiscalar_mul(&[*r], &[g]).affine())
      .collect();

    CommitmentKey {
      ck: ck_scaled,
      h: self.h,
    }
  }

  /// reinterprets a vector of commitments as a set of generators
  fn reinterpret_commitments_as_ck(c: &[Commitment<E>]) -> Result<Self, NovaError> {
    let ck = (0..c.len())
      .into_par_iter()
      .map(|i| c[i].comm.affine())
      .collect();

    // cmt is derandomized by the point that this is called
    Ok(CommitmentKey {
      ck,
      h: E::GE::zero().affine(), // this is okay, since this method is used in IPA only,
                                 // and we only use non-blinding commits afterwards
                                 // bc we don't use ZK IPA
    })
  }
}

#[cfg(feature = "io")]
#[cfg(test)]
mod tests {
  use super::*;

  use crate::{provider::GrumpkinEngine, CommitmentKey};
  use std::{fs::File, io::BufWriter};

  type E = GrumpkinEngine;

  #[test]
  fn test_key_save_load() {
    let path = "/tmp/pedersen_test.keys";

    const LABEL: &[u8; 4] = b"test";

    let keys = CommitmentEngine::<E>::setup(LABEL, 100);

    CommitmentEngine::save_setup(&keys, &mut BufWriter::new(File::create(path).unwrap())).unwrap();

    let keys_read = CommitmentEngine::load_setup(&mut File::open(path).unwrap(), LABEL, 100);

    assert!(keys_read.is_ok());
    let keys_read: CommitmentKey<E> = keys_read.unwrap();
    assert_eq!(keys_read.ck.len(), keys.ck.len());
    assert_eq!(keys_read.h, keys.h);
    assert_eq!(keys_read.ck, keys.ck);
  }
}
