use std::{
  io::{self, Read, Seek, SeekFrom, Write},
  path::Path,
  str::{from_utf8, Utf8Error},
};

use bincode::de::read;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ff::PrimeField;
use halo2curves::{ff_ext::jacobi::LInt, CurveAffine};
use num_bigint::BigUint;
use num_traits::One;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::{errors::NovaError, traits::Engine};

use super::{
  bn256_grumpkin::{bn256, grumpkin},
  pasta::pallas,
  secp_secq::{secp256k1, secq256k1},
  traits::DlogGroup,
};

#[derive(thiserror::Error, Debug)]
pub enum PtauFileError {
  #[error("Invalid magic string")]
  InvalidHead,

  #[error("Unsupported version")]
  UnsupportedVersion(u32),

  #[error("Invalid number of sections")]
  InvalidNumSections(u32),

  #[error("Invalid base prime")]
  InvalidPrime(BigUint),

  #[error("Insufficient power for G1")]
  InsufficientPowerForG1 { power: u32, required: usize },

  #[error("Insufficient power for G2")]
  InsufficientPowerForG2 { power: u32, required: usize },

  #[error(transparent)]
  IoError(#[from] io::Error),
  #[error(transparent)]
  Utf8Error(#[from] Utf8Error),
}

#[derive(Debug)]
struct MetaData {
  pos_header: u64,
  pos_tau_g1: u64,
  pos_tau_g2: u64,
}

const PTAU_VERSION: u32 = 1;
const NUM_SECTIONS: u32 = 11;

fn write_header<Base: PrimeField>(
  writer: &mut impl Write,
  power: u32,
) -> Result<(), PtauFileError> {
  const N8: usize = 32;

  writer.write_all(b"ptau")?;

  writer.write_u32::<LittleEndian>(PTAU_VERSION)?;
  writer.write_u32::<LittleEndian>(NUM_SECTIONS)?;

  // * header
  writer.write_u32::<LittleEndian>(1)?;
  writer.write_i64::<LittleEndian>(4 + N8 as i64 + 4)?;

  writer.write_u32::<LittleEndian>(N8 as u32)?;

  let modulus = BigUint::parse_bytes(Base::MODULUS[2..].as_bytes(), 16).unwrap();
  let mut bytes = [0u8; N8];
  bytes.copy_from_slice(&modulus.to_bytes_le());
  writer.write_all(&bytes)?;

  writer.write_u32::<LittleEndian>(power)?;

  Ok(())
}

pub(crate) fn write_points<G>(
  mut writer: &mut impl Write,
  points: Vec<G>,
  // section_id: u32,
) -> Result<(), PtauFileError>
where
  G: halo2curves::serde::SerdeObject + CurveAffine,
{
  for point in points {
    point.write_raw(&mut writer)?;
  }
  Ok(())
}

pub fn write_ptau<G1, G2>(
  mut writer: &mut (impl Write + Seek),
  g1_points: Vec<G1>,
  g2_points: Vec<G2>,
  power: u32,
) -> Result<(), PtauFileError>
where
  G1: halo2curves::serde::SerdeObject + CurveAffine,
  G2: halo2curves::serde::SerdeObject + CurveAffine,
{
  write_header::<G1::Base>(&mut writer, power)?;

  for id in 4..NUM_SECTIONS {
    writer.write_u32::<LittleEndian>(id)?;
    writer.write_i64::<LittleEndian>(0)?;
  }

  {
    writer.write_u32::<LittleEndian>(2)?;
    let pos = writer.stream_position()?;

    writer.write_i64::<LittleEndian>(0)?;
    let start = writer.stream_position()?;

    write_points(writer, g1_points)?;

    let size = writer.stream_position()? - start;

    writer.seek(SeekFrom::Start(pos))?;
    writer.write_i64::<LittleEndian>(size as i64)?;
    
    writer.seek(SeekFrom::Current(size as i64))?;
  }

  {
    writer.write_u32::<LittleEndian>(3)?;
    let pos = writer.stream_position()?;

    writer.write_i64::<LittleEndian>(0)?;
    let start = writer.stream_position()?;

    write_points(writer, g2_points)?;

    let size = writer.stream_position()? - start;

    writer.seek(SeekFrom::Start(pos))?;
    writer.write_i64::<LittleEndian>(size as i64)?;
  }
  Ok(())
}

fn read_meta_data(reader: &mut (impl Read + Seek)) -> Result<MetaData, PtauFileError> {
  {
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    if from_utf8(&buf)? != "ptau" {
      return Err(PtauFileError::InvalidHead);
    }
  }
  {
    let version = reader.read_u32::<LittleEndian>()?;
    if version != PTAU_VERSION {
      return Err(PtauFileError::UnsupportedVersion(version));
    }
  }
  {
    let num_sections = reader.read_u32::<LittleEndian>()?;
    if num_sections != NUM_SECTIONS {
      return Err(PtauFileError::InvalidNumSections(num_sections));
    }
  }
  let mut pos_header = 0;
  let mut pos_tau_g1 = 0;
  let mut pos_tau_g2 = 0;

  for _ in 0..NUM_SECTIONS {
    let id = reader.read_u32::<LittleEndian>()?;
    let size = reader.read_i64::<LittleEndian>()?;

    let pos = reader.stream_position()?;

    match id {
      1 => {
        pos_header = pos;
      }
      2 => {
        pos_tau_g1 = pos;
      }
      3 => {
        pos_tau_g2 = pos;
      }
      _ => {}
    };
    reader.seek(SeekFrom::Current(size))?;
  }

  assert_ne!(pos_header, 0);
  assert_ne!(pos_tau_g1, 0);
  assert_ne!(pos_tau_g2, 0);

  Ok(MetaData {
    pos_header,
    pos_tau_g1,
    pos_tau_g2,
  })
}

fn read_header<Base: PrimeField>(
  reader: &mut impl Read,
  num_g1: usize,
  num_g2: usize,
) -> Result<(), PtauFileError> {
  // * n8
  let n8 = reader.read_u32::<LittleEndian>()?;

  // * prime
  {
    let mut buf = vec![0u8; n8 as usize];
    reader.read_exact(&mut buf)?;

    let modulus = BigUint::from_bytes_le(&buf);

    let modulus_expected = BigUint::parse_bytes(Base::MODULUS[2..].as_bytes(), 16).unwrap();

    if modulus != modulus_expected {
      return Err(PtauFileError::InvalidPrime(modulus));
    }
  }

  // * power
  let power = reader.read_u32::<LittleEndian>()?;

  let max_num_g2 = 1 << power;
  let max_num_g1 = max_num_g2 * 2 - 1;
  if num_g1 > max_num_g1 {
    return Err(PtauFileError::InsufficientPowerForG1 {
      power,
      required: max_num_g1,
    });
  }
  if num_g2 > max_num_g2 {
    return Err(PtauFileError::InsufficientPowerForG2 {
      power,
      required: max_num_g2,
    });
  }

  Ok(())
}

pub(crate) fn read_points<G>(mut reader: &mut impl Read, num: usize) -> Result<Vec<G>, PtauFileError>
where
  G: halo2curves::serde::SerdeObject + CurveAffine,
{
  let mut res = Vec::with_capacity(num);
  for _ in 0..num {
    res.push(G::read_raw(&mut reader)?);
  }
  Ok(res)
}

pub fn read_ptau<G1, G2>(
  mut reader: &mut (impl Read + Seek),
  num_g1: usize,
  num_g2: usize,
) -> Result<(Vec<G1>, Vec<G2>), PtauFileError>
where
  G1: halo2curves::serde::SerdeObject + CurveAffine,
  G2: halo2curves::serde::SerdeObject + CurveAffine,
{
  let metadata = read_meta_data(&mut reader)?;

  reader.seek(SeekFrom::Start(metadata.pos_header))?;
  read_header::<G1::Base>(reader, num_g1, num_g2)?;

  reader.seek(SeekFrom::Start(metadata.pos_tau_g1))?;
  let g1_points = read_points::<G1>(&mut reader, num_g1)?;

  reader.seek(SeekFrom::Start(metadata.pos_tau_g2))?;
  let g2_points = read_points::<G2>(&mut reader, num_g2)?;

  Ok((g1_points, g2_points))
}

fn encode<G: DlogGroup>() -> Vec<u8> {
  let one = G::gen().affine();
  let zero = G::zero().affine();
  let mut encoded = G::encode(&one);
  encoded.extend(G::encode(&zero));
  encoded
}

/// Returns the ID of the DLog group
pub fn id_of<G: DlogGroup>() -> Option<u8> {
  let bn256 = encode::<bn256::Point>();
  let grumpkin = encode::<grumpkin::Point>();
  let palas = encode::<pallas::Point>();
  let vesta = encode::<pallas::Point>();
  let secp = encode::<secp256k1::Point>();
  let secq = encode::<secq256k1::Point>();

  let self_encode = encode::<G>();

  if bn256 == self_encode {
    Some(1)
  } else if grumpkin == self_encode {
    Some(2)
  } else if palas == self_encode {
    Some(3)
  } else if vesta == self_encode {
    Some(4)
  } else if secp == self_encode {
    Some(5)
  } else if secq == self_encode {
    Some(6)
  } else {
    None
  }
}

/// A trait for saving and loading commitment keys
pub trait CommitmentKeyIO: Sized {
  /// Saves the commitment key to the provided writer
  fn save_to(&self, writer: &mut impl Write) -> Result<(), io::Error>;
  /// Loads the commitment key from the provided reader
  fn load_from(reader: &mut impl Read, needed_num_ck: usize) -> Result<Self, io::Error>;
  /// Checks the sanity of the key file
  fn check_sanity_of_file(path: impl AsRef<Path>, required_len_ck_list: usize) -> bool;
}

pub fn load_ck_vec<E: Engine>(
  reader: &mut impl Read,
  ck_len: usize,
) -> Vec<<E::GE as DlogGroup>::AffineGroupElement>
where
  E::GE: DlogGroup,
{
  let mut ck = Vec::with_capacity(ck_len);
  for _ in 0..ck_len {
    let p = <E::GE as DlogGroup>::decode_from(reader).unwrap();
    ck.push(p);
  }
  ck
}

pub fn write_ck_vec<E: Engine>(
  writer: &mut impl Write,
  ck: &[<E::GE as DlogGroup>::AffineGroupElement],
) -> Result<(), io::Error>
where
  E::GE: DlogGroup,
{
  let chunk_size = rayon::current_num_threads() * 1000;
  ck.chunks(chunk_size).for_each(|cks| {
    let bins = cks
      .par_iter()
      .map(<E::GE as DlogGroup>::encode)
      .flatten()
      .collect::<Vec<_>>();
    writer.write_all(&bins).unwrap();
  });
  Ok(())
}
