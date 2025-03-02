//! This module provides an implementation of a commitment engine
use crate::{
  errors::NovaError,
  provider::{commitment_key_io::load_ck_vec, traits::DlogGroup},
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
use std::{fs::File, io::Read};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::commitment_key_io::{id_of, write_ck_vec, CommitmentKeyIO};

/// A type that holds commitment generators
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommitmentKey<E: Engine>
where
  E::GE: DlogGroup,
{
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  h: Option<<E::GE as DlogGroup>::AffineGroupElement>,
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
  E::GE: DlogGroup,
{
  type CommitmentKey = CommitmentKey<E>;
  type Commitment = Commitment<E>;
  type DerandKey = DerandKey<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    let gens = E::GE::from_label(label, n.next_power_of_two() + 1);

    let (h, ck) = gens.split_first().unwrap();

    Self::CommitmentKey {
      ck: ck.to_vec(),
      h: Some(h.clone()),
    }
  }

  fn derand_key(ck: &Self::CommitmentKey) -> Self::DerandKey {
    assert!(ck.h.is_some());
    Self::DerandKey {
      h: ck.h.as_ref().unwrap().clone(),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar], r: &E::Scalar) -> Self::Commitment {
    assert!(ck.ck.len() >= v.len());

    if ck.h.is_some() {
      Commitment {
        comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()])
          + <E::GE as DlogGroup>::group(ck.h.as_ref().unwrap()) * r,
      }
    } else {
      assert_eq!(*r, E::Scalar::ZERO);

      Commitment {
        comm: E::GE::vartime_multiscalar_mul(v, &ck.ck[..v.len()]),
      }
    }
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

impl<E> CommitmentKeyExtTrait<E> for CommitmentKey<E>
where
  E: Engine<CE = CommitmentEngine<E>>,
  E::GE: DlogGroup,
{
  fn split_at(&self, n: usize) -> (CommitmentKey<E>, CommitmentKey<E>) {
    (
      CommitmentKey {
        ck: self.ck[0..n].to_vec(),
        h: self.h.clone(),
      },
      CommitmentKey {
        ck: self.ck[n..].to_vec(),
        h: self.h.clone(),
      },
    )
  }

  fn combine(&self, other: &CommitmentKey<E>) -> CommitmentKey<E> {
    let ck = {
      let mut c = self.ck.clone();
      c.extend(other.ck.clone());
      c
    };
    CommitmentKey {
      ck,
      h: self.h.clone(),
    }
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

    CommitmentKey {
      ck,
      h: self.h.clone(),
    }
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
      h: self.h.clone(),
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
      h: None, // this is okay, since this method is used in IPA only,
               // and we only use non-blinding commits afterwards
               // bc we don't use ZK IPA
    })
  }
}

// ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
//   h: Option<<E::GE as DlogGroup>::AffineGroupElement>,

impl<E: Engine> CommitmentKeyIO for CommitmentKey<E>
where
  E::GE: DlogGroup,
{
  fn save_to(&self, writer: &mut impl std::io::Write) -> Result<(), std::io::Error> {
    // Write the Group ID
    let group_id = id_of::<E::GE>().unwrap();
    writer.write_all(&[group_id])?;

    // Write the ck length
    let ck_len = self.ck.len() as u64;
    writer.write_all(&ck_len.to_le_bytes())?;

    // Write h
    let bin = bincode::serialize(&self.h).unwrap();
    writer.write_all(&bin)?;

    // Write ck
    write_ck_vec::<E>(writer, &self.ck)?;

    writer.flush()
  }

  fn load_from(
    mut reader: &mut impl std::io::Read,
    needed_num_ck: usize,
  ) -> Result<Self, std::io::Error> {
    // Read the group ID
    let mut group_id = [0u8; 1];
    reader.read_exact(&mut group_id)?;

    // Read the ck length
    let mut ck_len = [0u8; 8];
    reader.read_exact(&mut ck_len)?;
    let ck_len = u64::from_le_bytes(ck_len) as usize;

    assert!(needed_num_ck <= ck_len);

    // Read h
    let h = bincode::deserialize_from(&mut reader).unwrap();

    // Read ck list
    let ck = load_ck_vec::<E>(reader, needed_num_ck);

    Ok(Self { ck, h })
  }

  fn check_sanity_of_file(path: impl AsRef<std::path::Path>, required_len_ck_list: usize) -> bool {
    let file = File::open(path);
    if file.is_err() {
      return false;
    }
    let mut reader = file.unwrap();

    // Read the group ID
    let expected_id = id_of::<E::GE>();
    if expected_id.is_none() {
      return false;
    }

    let expected_id = expected_id.unwrap();

    let mut group_id = [0u8; 1];
    if reader.read_exact(&mut group_id).is_err() {
      return false;
    }

    if group_id[0] != expected_id {
      return false;
    }
    // Read the ck length
    let mut ck_len = [0u8; 8];
    if reader.read_exact(&mut ck_len).is_err() {
      return false;
    }
    let ck_len = u64::from_le_bytes(ck_len) as usize;

    if required_len_ck_list > ck_len {
      return false;
    }

    if let Some(point) = <E::GE as DlogGroup>::decode_from(&mut reader) {
      let bin = <E::GE as DlogGroup>::encode(&point);
      let size_point = bin.len();

      let expcted_size = size_point as u64 * (ck_len + 2) as u64
        + bincode::serialized_size(&E::GE::zero()).unwrap()
        + 1;

      expcted_size == reader.metadata().unwrap().len()
    } else {
      false
    }
  }
}
