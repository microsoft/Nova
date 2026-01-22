//! # PPOT Pruner - Powers of Tau File Pruning for HyperKZG/Mercury
//!
//! This example downloads and prunes Powers of Tau (PPOT) files from the Ethereum
//! Privacy & Scaling Explorations (PSE) trusted setup ceremony for use with Nova's
//! HyperKZG and Mercury polynomial commitment schemes.
//!
//! ## Background
//!
//! The Ethereum PPOT ceremony produces large `.ptau` files containing structured reference
//! strings (SRS) for pairing-based cryptographic schemes. These files contain many different
//! types of curve points, but HyperKZG and Mercury only need a small subset:
//!
//! - **N G1 points**: τ^0·G1, τ^1·G1, ..., τ^(N-1)·G1
//! - **2 G2 points**: G2 and τ·G2
//!
//! This allows significant size reduction. For example:
//! - Power 20 (2^20 = 1M constraints): 100MB → 64MB (1.6x reduction)
//! - Power 23 (2^23 = 8M constraints): 800MB → 512MB (1.6x reduction)
//! - Power 28 (2^28 = 256M constraints): 288GB → ~16GB (18x reduction)
//!
//! ## PTAU File Format
//!
//! The `.ptau` files use a binary format with:
//! - 4-byte magic: "ptau"
//! - 4-byte version: 1
//! - 4-byte section count: 11 (original) or 3 (pruned)
//! - Section table: pairs of (section_id: u32, size: i64) followed by data
//!
//! Sections used:
//! - Section 1: Header (n8, prime modulus, power)
//! - Section 2: TauG1 (G1 points, 64 bytes each on BN254)
//! - Section 3: TauG2 (G2 points, 128 bytes each on BN254)
//!
//! ## Usage
//!
//! ```bash
//! # Prune power 20 (1M constraints, ~64MB output)
//! cargo run --example ppot_prune --features io -- --power 20
//!
//! # Prune power 23 with custom output directory
//! cargo run --example ppot_prune --features io -- --power 23 --output ./my_ptau
//!
//! # Prune keeping only specific number of G1 points
//! cargo run --example ppot_prune --features io -- --power 20 -n 100000
//! ```
//!
//! ## Source
//!
//! Files are downloaded from the PSE trusted setup S3 bucket:
//! <https://github.com/privacy-ethereum/perpetualpowersoftau>
//!
//! The ceremony was conducted with 80+ participants, providing strong security guarantees
//! (only one honest participant needed for security).

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use halo2curves::bn256::{G1Affine, G2Affine};
use halo2curves::serde::SerdeObject;
use num_bigint::BigUint;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

/// PTAU file format version (always 1)
const PTAU_VERSION: u32 = 1;

/// Number of sections in original PPOT files
const NUM_SECTIONS: u32 = 11;

/// Number of sections we write in pruned files (header, tau_g1, tau_g2)
const PRUNED_NUM_SECTIONS: u32 = 3;

/// Base URL for downloading PPOT files from PSE S3 bucket
const PPOT_BASE_URL: &str =
  "https://pse-trusted-setup-ppot.s3.eu-central-1.amazonaws.com/pot28_0080";

/// Error types for the PPOT pruner
#[derive(Debug)]
pub enum PrunerError {
  /// Invalid magic bytes (expected "ptau")
  InvalidMagic,
  /// Unsupported PTAU version
  UnsupportedVersion(u32),
  /// Invalid number of sections in file
  InvalidNumSections(u32),
  /// Requested power is too small for number of G1 points needed
  InsufficientPower(u32, usize),
  /// IO error during read/write
  Io(io::Error),
  /// HTTP error during download
  Http(String),
  /// Required section not found in file
  SectionNotFound(u32),
}

impl std::fmt::Display for PrunerError {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      PrunerError::InvalidMagic => write!(f, "Invalid ptau magic string"),
      PrunerError::UnsupportedVersion(v) => write!(f, "Unsupported version: {}", v),
      PrunerError::InvalidNumSections(n) => write!(f, "Invalid number of sections: {}", n),
      PrunerError::InsufficientPower(p, n) => {
        write!(f, "Power {} insufficient for {} G1 points", p, n)
      }
      PrunerError::Io(e) => write!(f, "IO error: {}", e),
      PrunerError::Http(e) => write!(f, "HTTP error: {}", e),
      PrunerError::SectionNotFound(s) => write!(f, "Section {} not found", s),
    }
  }
}

impl std::error::Error for PrunerError {}

impl From<io::Error> for PrunerError {
  fn from(e: io::Error) -> Self {
    PrunerError::Io(e)
  }
}

/// Metadata extracted from a PTAU file header
#[derive(Debug)]
struct PtauMetadata {
  /// Power of 2 (e.g., 20 means 2^20 points)
  power: u32,
  /// File position of TauG1 section data
  pos_tau_g1: u64,
  /// File position of TauG2 section data
  pos_tau_g2: u64,
  /// Size of TauG1 section in bytes
  size_tau_g1: u64,
  /// Size of TauG2 section in bytes
  size_tau_g2: u64,
}

/// Get the filename for a PPOT file of a given power.
///
/// Power 28 uses a special "final" filename, others use the standard format.
fn get_ptau_filename(power: u32) -> String {
  if power == 28 {
    "ppot_0080_final.ptau".to_string()
  } else {
    format!("ppot_0080_{:02}.ptau", power)
  }
}

/// Download a PPOT file from the PSE S3 bucket.
///
/// Shows progress during download. The file is written to `dest`.
fn download_ptau(power: u32, dest: &PathBuf) -> Result<(), PrunerError> {
  let filename = get_ptau_filename(power);
  let url = format!("{}/{}", PPOT_BASE_URL, filename);

  println!("Downloading {} from {}", filename, url);

  let client = reqwest::blocking::Client::new();
  let mut response = client
    .get(&url)
    .send()
    .map_err(|e| PrunerError::Http(e.to_string()))?;

  if !response.status().is_success() {
    return Err(PrunerError::Http(format!(
      "HTTP {} for {}",
      response.status(),
      url
    )));
  }

  let total_size = response.content_length().unwrap_or(0);
  println!("Total size: {:.2} MB", total_size as f64 / 1_000_000.0);

  let mut file = File::create(dest)?;
  let mut downloaded: u64 = 0;
  let mut buffer = [0u8; 8192];
  let mut last_percent = 0;

  loop {
    let bytes_read = response
      .read(&mut buffer)
      .map_err(|e| PrunerError::Http(e.to_string()))?;
    if bytes_read == 0 {
      break;
    }
    file.write_all(&buffer[..bytes_read])?;
    downloaded += bytes_read as u64;

    // Print progress every 10%
    if total_size > 0 {
      let percent = (downloaded * 100 / total_size) as u32;
      if percent >= last_percent + 10 {
        println!(
          "  {}% downloaded ({:.2} MB)",
          percent,
          downloaded as f64 / 1_000_000.0
        );
        last_percent = percent / 10 * 10;
      }
    }
  }

  println!("Download complete: {:?}", dest);
  Ok(())
}

/// Read and parse PTAU file metadata (header and section positions).
///
/// This reads the file header and section table to locate the TauG1 and TauG2
/// sections without reading the actual point data.
fn read_ptau_metadata(reader: &mut (impl Read + Seek)) -> Result<PtauMetadata, PrunerError> {
  // Read magic bytes
  let mut magic = [0u8; 4];
  reader.read_exact(&mut magic)?;
  if &magic != b"ptau" {
    return Err(PrunerError::InvalidMagic);
  }

  // Read version
  let version = reader.read_u32::<LittleEndian>()?;
  if version != PTAU_VERSION {
    return Err(PrunerError::UnsupportedVersion(version));
  }

  // Read number of sections
  let num_sections = reader.read_u32::<LittleEndian>()?;
  if num_sections != NUM_SECTIONS && num_sections != PRUNED_NUM_SECTIONS {
    return Err(PrunerError::InvalidNumSections(num_sections));
  }

  let mut pos_header = 0u64;
  let mut pos_tau_g1 = 0u64;
  let mut pos_tau_g2 = 0u64;
  let mut size_tau_g1 = 0u64;
  let mut size_tau_g2 = 0u64;

  // Read section table entries
  for _ in 0..num_sections {
    let section_id = reader.read_u32::<LittleEndian>()?;
    let section_size = reader.read_i64::<LittleEndian>()? as u64;
    let section_pos = reader.stream_position()?;

    match section_id {
      1 => pos_header = section_pos,
      2 => {
        pos_tau_g1 = section_pos;
        size_tau_g1 = section_size;
      }
      3 => {
        pos_tau_g2 = section_pos;
        size_tau_g2 = section_size;
      }
      _ => {}
    }

    reader.seek(SeekFrom::Current(section_size as i64))?;
  }

  if pos_header == 0 {
    return Err(PrunerError::SectionNotFound(1));
  }
  if pos_tau_g1 == 0 {
    return Err(PrunerError::SectionNotFound(2));
  }
  if pos_tau_g2 == 0 {
    return Err(PrunerError::SectionNotFound(3));
  }

  // Read power from header section
  reader.seek(SeekFrom::Start(pos_header))?;
  let n8 = reader.read_u32::<LittleEndian>()?;

  // Skip prime modulus
  reader.seek(SeekFrom::Current(n8 as i64))?;

  let power = reader.read_u32::<LittleEndian>()?;

  Ok(PtauMetadata {
    power,
    pos_tau_g1,
    pos_tau_g2,
    size_tau_g1,
    size_tau_g2,
  })
}

/// Read G1 affine points from a reader.
///
/// Each G1 point is 64 bytes on BN254 (32 bytes for x, 32 bytes for y).
fn read_g1_points(reader: &mut impl Read, num: usize) -> Result<Vec<G1Affine>, PrunerError> {
  println!("Reading {} G1 points...", num);

  let mut points = Vec::with_capacity(num);
  for i in 0..num {
    let point = G1Affine::read_raw(reader)
      .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    points.push(point);

    // Progress every 10%
    if num > 10 && i % (num / 10) == 0 {
      println!("  {}% read", i * 100 / num);
    }
  }

  println!("Done reading G1 points");
  Ok(points)
}

/// Read G2 affine points from a reader.
///
/// Each G2 point is 128 bytes on BN254 (64 bytes for x, 64 bytes for y,
/// where x and y are each elements of Fq2).
fn read_g2_points(reader: &mut impl Read, num: usize) -> Result<Vec<G2Affine>, PrunerError> {
  println!("Reading {} G2 points...", num);

  let mut points = Vec::with_capacity(num);
  for _ in 0..num {
    let point = G2Affine::read_raw(reader)
      .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    points.push(point);
  }

  println!("Done reading G2 points");
  Ok(points)
}

/// Write a pruned PTAU file containing only the necessary points.
///
/// The output file contains:
/// - Header section with power metadata
/// - TauG1 section with N G1 points
/// - TauG2 section with 2 G2 points
fn write_pruned_ptau(
  writer: &mut (impl Write + Seek),
  g1_points: &[G1Affine],
  g2_points: &[G2Affine],
  power: u32,
) -> Result<(), PrunerError> {
  const N8: usize = 32; // Field element size in bytes for BN254

  // Write magic
  writer.write_all(b"ptau")?;

  // Write version
  writer.write_u32::<LittleEndian>(PTAU_VERSION)?;

  // Write number of sections (we only write 3: header, tau_g1, tau_g2)
  writer.write_u32::<LittleEndian>(PRUNED_NUM_SECTIONS)?;

  // Write header section entry
  let header_size = 4 + N8 + 4; // n8 + prime + power
  writer.write_u32::<LittleEndian>(1)?; // section id
  writer.write_i64::<LittleEndian>(header_size as i64)?;

  // Write header content
  writer.write_u32::<LittleEndian>(N8 as u32)?;

  // Write BN254 prime modulus (little-endian)
  let modulus = BigUint::parse_bytes(
    b"30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47",
    16,
  )
  .unwrap();
  let mut modulus_bytes = [0u8; N8];
  let bytes = modulus.to_bytes_le();
  modulus_bytes[..bytes.len()].copy_from_slice(&bytes);
  writer.write_all(&modulus_bytes)?;

  writer.write_u32::<LittleEndian>(power)?;

  // Write TauG1 section
  let g1_size = g1_points.len() * 64; // 64 bytes per G1 point
  writer.write_u32::<LittleEndian>(2)?;
  writer.write_i64::<LittleEndian>(g1_size as i64)?;

  println!("Writing {} G1 points...", g1_points.len());
  for (i, point) in g1_points.iter().enumerate() {
    point
      .write_raw(writer)
      .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;

    // Progress every 10%
    let num = g1_points.len();
    if num > 10 && i % (num / 10) == 0 {
      println!("  {}% written", i * 100 / num);
    }
  }
  println!("Done writing G1 points");

  // Write TauG2 section
  let g2_size = g2_points.len() * 128; // 128 bytes per G2 point
  writer.write_u32::<LittleEndian>(3)?;
  writer.write_i64::<LittleEndian>(g2_size as i64)?;

  println!("Writing {} G2 points...", g2_points.len());
  for point in g2_points {
    point
      .write_raw(writer)
      .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
  }
  println!("Done writing G2 points");

  Ok(())
}

/// Prune a PTAU file to contain only the points needed for HyperKZG/Mercury.
///
/// # Arguments
///
/// * `input_path` - Path to the original PPOT file
/// * `output_path` - Path for the pruned output file
/// * `num_g1` - Number of G1 points to keep
/// * `num_g2` - Number of G2 points to keep (should be 2)
fn prune_ptau(
  input_path: &PathBuf,
  output_path: &PathBuf,
  num_g1: usize,
  num_g2: usize,
) -> Result<(), PrunerError> {
  println!("Reading metadata from {:?}", input_path);

  let mut reader = BufReader::new(File::open(input_path)?);
  let metadata = read_ptau_metadata(&mut reader)?;

  println!("PTAU file power: {}", metadata.power);
  println!("TauG1 section size: {} bytes", metadata.size_tau_g1);
  println!("TauG2 section size: {} bytes", metadata.size_tau_g2);

  let max_g1 = metadata.size_tau_g1 as usize / 64;
  let max_g2 = metadata.size_tau_g2 as usize / 128;

  println!("Max G1 points available: {}", max_g1);
  println!("Max G2 points available: {}", max_g2);

  if num_g1 > max_g1 {
    return Err(PrunerError::InsufficientPower(metadata.power, num_g1));
  }

  // Read G1 points
  reader.seek(SeekFrom::Start(metadata.pos_tau_g1))?;
  let g1_points = read_g1_points(&mut reader, num_g1)?;

  // Read G2 points (only need first 2)
  reader.seek(SeekFrom::Start(metadata.pos_tau_g2))?;
  let g2_points = read_g2_points(&mut reader, num_g2)?;

  // Write pruned file
  println!("Writing pruned ptau to {:?}", output_path);
  let mut writer = BufWriter::new(File::create(output_path)?);
  write_pruned_ptau(&mut writer, &g1_points, &g2_points, metadata.power)?;

  let input_size = std::fs::metadata(input_path)?.len();
  let output_size = std::fs::metadata(output_path)?.len();

  println!("\n=== Summary ===");
  println!(
    "Input size:  {} bytes ({:.2} MB)",
    input_size,
    input_size as f64 / 1_000_000.0
  );
  println!(
    "Output size: {} bytes ({:.2} MB)",
    output_size,
    output_size as f64 / 1_000_000.0
  );
  println!(
    "Reduction:   {:.1}x smaller",
    input_size as f64 / output_size as f64
  );
  println!("G1 points:   {}", num_g1);
  println!("G2 points:   {}", num_g2);

  Ok(())
}

/// Print usage information
fn print_usage() {
  println!("PPOT Pruner - Prune Powers of Tau files for HyperKZG/Mercury");
  println!();
  println!("Usage: cargo run --example ppot_prune --features io -- [OPTIONS]");
  println!();
  println!("Options:");
  println!("  --power <N>       Power of 2 for the ptau file (e.g., 20 for 2^20 constraints)");
  println!("  --output <DIR>    Output directory for pruned files (default: pruned)");
  println!("  --download <DIR>  Download directory for original files (default: temp)");
  println!("  --num-g1 <N>      Number of G1 points to keep (default: 2^power)");
  println!("  --skip-download   Skip download if file already exists (default: true)");
  println!();
  println!("Examples:");
  println!("  # Prune power 20 (1M constraints, ~64MB output)");
  println!("  cargo run --example ppot_prune --features io -- --power 20");
  println!();
  println!("  # Prune power 23 with custom output");
  println!("  cargo run --example ppot_prune --features io -- --power 23 --output ./my_ptau");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = std::env::args().collect();

  // Parse arguments manually (avoiding clap dependency for example)
  let mut power: Option<u32> = None;
  let mut output = PathBuf::from("pruned");
  let mut download_dir: Option<PathBuf> = None;
  let mut num_g1_override: Option<usize> = None;
  let mut skip_existing = true;

  let mut i = 1;
  while i < args.len() {
    match args[i].as_str() {
      "--power" | "-p" => {
        if i + 1 >= args.len() {
          eprintln!("Missing value for --power");
          print_usage();
          return Ok(());
        }
        i += 1;
        power = Some(args[i].parse()?);
      }
      "--output" | "-o" => {
        if i + 1 >= args.len() {
          eprintln!("Missing value for --output");
          print_usage();
          return Ok(());
        }
        i += 1;
        output = PathBuf::from(&args[i]);
      }
      "--download" | "-d" => {
        if i + 1 >= args.len() {
          eprintln!("Missing value for --download");
          print_usage();
          return Ok(());
        }
        i += 1;
        download_dir = Some(PathBuf::from(&args[i]));
      }
      "--num-g1" | "-n" => {
        if i + 1 >= args.len() {
          eprintln!("Missing value for --num-g1");
          print_usage();
          return Ok(());
        }
        i += 1;
        num_g1_override = Some(args[i].parse()?);
      }
      "--skip-download" => {
        skip_existing = true;
      }
      "--force-download" => {
        skip_existing = false;
      }
      "--help" | "-h" => {
        print_usage();
        return Ok(());
      }
      _ => {
        eprintln!("Unknown argument: {}", args[i]);
        print_usage();
        return Ok(());
      }
    }
    i += 1;
  }

  let power = match power {
    Some(p) => p,
    None => {
      print_usage();
      return Ok(());
    }
  };

  // Create output directory
  std::fs::create_dir_all(&output)?;

  // Determine download path
  let download_dir = download_dir.unwrap_or_else(std::env::temp_dir);
  std::fs::create_dir_all(&download_dir)?;

  let ptau_filename = get_ptau_filename(power);
  let input_path = download_dir.join(&ptau_filename);

  // Download if needed
  if !input_path.exists() || !skip_existing {
    download_ptau(power, &input_path)?;
  } else {
    println!("Using existing file: {:?}", input_path);
  }

  // Determine number of points
  let num_g1 = num_g1_override.unwrap_or(1 << power);
  let num_g2 = 2; // Always just need G2 and τ*G2

  // Output filename
  let output_filename = format!("ppot_pruned_{:02}.ptau", power);
  let output_path = output.join(output_filename);

  // Prune
  prune_ptau(&input_path, &output_path, num_g1, num_g2)?;

  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_filename_generation() {
    assert_eq!(get_ptau_filename(15), "ppot_0080_15.ptau");
    assert_eq!(get_ptau_filename(20), "ppot_0080_20.ptau");
    assert_eq!(get_ptau_filename(28), "ppot_0080_final.ptau");
  }
}
