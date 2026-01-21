//! Powers of Tau (PTAU) file handling for trusted setup.
//!
//! This module provides functionality for reading and writing PTAU files,
//! which contain the structured reference string (SRS) needed for KZG-based
//! polynomial commitment schemes like HyperKZG and Mercury.
//!
//! # Obtaining PTAU Files
//!
//! There are two ways to obtain PTAU files:
//!
//! ## Option 1: Use Ethereum Powers of Tau Ceremony Files (Recommended for Production)
//!
//! For production use, you should use files from the Ethereum [Perpetual Powers of Tau](https://github.com/privacy-ethereum/perpetualpowersoftau)
//! ceremony. This ceremony has 80+ participants, providing strong security guarantees
//! (only one honest participant needed).
//!
//! ### Downloading Original PPOT Files
//!
//! Original PPOT files can be downloaded directly from the PSE S3 bucket:
//! <https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080/>
//!
//! Files are named:
//! - `ppot_0080_15.ptau` through `ppot_0080_27.ptau` (powers 15-27)
//! - `ppot_0080_final.ptau` (power 28, ~288GB)
//!
//! ### Pruning Files (Optional, Reduces Size ~18x)
//!
//! To reduce file size, you can prune files using the `ppot_prune` example:
//!
//! ```bash
//! # Prune power 24 (16M constraints, ~1GB output vs ~18GB original)
//! cargo run --example ppot_prune --features io -- --power 24
//!
//! # Prune power 20 with custom output directory
//! cargo run --example ppot_prune --features io -- --power 20 --output ./my_ptau
//! ```
//!
//! Pruned files are named `ppot_pruned_XX.ptau`.
//!
//! ### Using PTAU Files in Your Application
//!
//! Place your PTAU files (either original or pruned) in a directory and use:
//!
//! ```ignore
//! use nova_snark::nova::PublicParams;
//! use std::path::Path;
//!
//! // Load commitment key from ptau directory
//! // Accepts both ppot_pruned_XX.ptau and ppot_0080_XX.ptau naming conventions
//! let pp = PublicParams::setup_with_ptau_dir(
//!     &circuit,
//!     &*S1::ck_floor(),
//!     &*S2::ck_floor(),
//!     Path::new("path/to/ptau_files"),
//! )?;
//! ```
//!
//! The library automatically selects the smallest available file that provides
//! enough generators for your circuit.
//!
//! ## Option 2: Generate Test PTAU Files (Testing Only)
//!
//! For testing purposes only, you can generate PTAU files with a random tau.
//! **These are insecure and must not be used in production.**
//!
//! ```bash
//! # Requires the test-utils feature
//! cargo run --example ptau_test_setup --features test-utils,io
//! ```
//!
//! # Supported File Naming Conventions
//!
//! The library accepts files with these naming patterns:
//! - `ppot_pruned_XX.ptau` - Pruned files (preferred, smaller)
//! - `ppot_0080_XX.ptau` - Original PPOT files from PSE
//! - `ppot_0080_final.ptau` - Power 28 original file
//!
//! # File Format
//!
//! PTAU files use a binary format with the following structure:
//!
//! - **Header**: Magic ("ptau"), version (1), number of sections
//! - **Section 1**: n8, prime modulus, power
//! - **Section 2**: TauG1 - N × G1 points (64 bytes each on BN254)
//! - **Section 3**: TauG2 - M × G2 points (128 bytes each on BN254)
//!
//! Both full (11 sections) and pruned (3 sections) formats are supported.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use ff::PrimeField;
use halo2curves::CurveAffine;
use num_bigint::BigUint;
use std::{
  fs::File,
  io::{self, Read, Seek, SeekFrom, Write},
  path::Path,
  str::{from_utf8, Utf8Error},
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
const NUM_SECTIONS_FULL: u32 = 11; // Original PPOT files
const NUM_SECTIONS_PRUNED: u32 = 3; // Pruned files (header, tau_g1, tau_g2 only)

fn write_header<Base: PrimeField>(
  writer: &mut impl Write,
  power: u32,
) -> Result<(), PtauFileError> {
  const N8: usize = 32;

  writer.write_all(b"ptau")?;

  writer.write_u32::<LittleEndian>(PTAU_VERSION)?;
  writer.write_u32::<LittleEndian>(NUM_SECTIONS_FULL)?;

  // * header
  writer.write_u32::<LittleEndian>(1)?;
  writer.write_i64::<LittleEndian>(4 + N8 as i64 + 4)?;

  writer.write_u32::<LittleEndian>(N8 as u32)?;

  let modulus = BigUint::parse_bytes(&Base::MODULUS.as_bytes()[2..], 16).unwrap();
  let mut bytes = [0u8; N8];
  bytes.copy_from_slice(&modulus.to_bytes_le());
  writer.write_all(&bytes)?;

  writer.write_u32::<LittleEndian>(power)?;

  Ok(())
}

pub(crate) fn write_points<G>(
  mut writer: &mut impl Write,
  points: Vec<G>,
) -> Result<(), PtauFileError>
where
  G: halo2curves::serde::SerdeObject + CurveAffine,
{
  for point in points {
    point.write_raw(&mut writer)?;
  }
  Ok(())
}

/// Save Ptau File
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

  writer.write_u32::<LittleEndian>(0)?;
  writer.write_i64::<LittleEndian>(0)?;

  for id in 4..NUM_SECTIONS_FULL {
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
  let num_sections = {
    let num_sections = reader.read_u32::<LittleEndian>()?;
    // Accept both full (11 sections) and pruned (3 sections) ptau files
    if num_sections != NUM_SECTIONS_FULL && num_sections != NUM_SECTIONS_PRUNED {
      return Err(PtauFileError::InvalidNumSections(num_sections));
    }
    num_sections
  };
  let mut pos_header = 0;
  let mut pos_tau_g1 = 0;
  let mut pos_tau_g2 = 0;

  for _ in 0..num_sections {
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

    let modulus_expected = BigUint::parse_bytes(&Base::MODULUS.as_bytes()[2..], 16).unwrap();

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

pub(crate) fn read_points<G>(
  mut reader: &mut impl Read,
  num: usize,
) -> Result<Vec<G>, PtauFileError>
where
  G: halo2curves::serde::SerdeObject + CurveAffine,
{
  let mut res = Vec::with_capacity(num);
  for _ in 0..num {
    res.push(G::read_raw(&mut reader)?);
  }
  Ok(res)
}

/// Load Ptau File
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

/// Check the sanity of the ptau file
pub fn check_sanity_of_ptau_file<G1>(
  path: impl AsRef<Path>,
  num_g1: usize,
  num_g2: usize,
) -> Result<(), PtauFileError>
where
  G1: halo2curves::serde::SerdeObject + CurveAffine,
{
  let mut reader = File::open(path)?;

  let metadata = read_meta_data(&mut reader)?;

  reader.seek(SeekFrom::Start(metadata.pos_header))?;
  read_header::<G1::Base>(&mut reader, num_g1, num_g2)
}
