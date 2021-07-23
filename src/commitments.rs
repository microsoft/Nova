use super::errors::NovaError;
use core::ops::{Add, AddAssign, Mul, MulAssign};
use curve25519_dalek::traits::VartimeMultiscalarMul;
use digest::{ExtendableOutput, Input};
use merlin::Transcript;
use sha3::Shake256;
use std::io::Read;

pub type Scalar = curve25519_dalek::scalar::Scalar;
type GroupElement = curve25519_dalek::ristretto::RistrettoPoint;
type CompressedGroup = curve25519_dalek::ristretto::CompressedRistretto;

#[derive(Debug)]
pub struct CommitGens {
  gens: Vec<GroupElement>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Commitment {
  comm: GroupElement,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompressedCommitment {
  comm: CompressedGroup,
}

impl CommitGens {
  pub fn new(label: &[u8], n: usize) -> Self {
    let mut shake = Shake256::default();
    shake.input(label);
    let mut reader = shake.xof_result();
    let mut gens: Vec<GroupElement> = Vec::new();
    let mut uniform_bytes = [0u8; 64];
    for _ in 0..n {
      reader.read_exact(&mut uniform_bytes).unwrap();
      gens.push(GroupElement::from_uniform_bytes(&uniform_bytes));
    }

    CommitGens { gens }
  }
}

impl Commitment {
  pub fn compress(&self) -> CompressedCommitment {
    CompressedCommitment {
      comm: self.comm.compress(),
    }
  }
}

impl CompressedCommitment {
  pub fn decompress(&self) -> Result<Commitment, NovaError> {
    let comm = self.comm.decompress();
    if comm.is_none() {
      return Err(NovaError::DecompressionError);
    }
    Ok(Commitment {
      comm: comm.unwrap(),
    })
  }
}

pub trait CommitTrait {
  fn commit(&self, gens: &CommitGens) -> Commitment;
}

impl CommitTrait for [Scalar] {
  fn commit(&self, gens: &CommitGens) -> Commitment {
    assert_eq!(gens.gens.len(), self.len());
    Commitment {
      comm: GroupElement::vartime_multiscalar_mul(self, &gens.gens),
    }
  }
}

pub trait ProofTranscriptTrait {
  fn append_protocol_name(&mut self, protocol_name: &'static [u8]);
  fn challenge_scalar(&mut self, label: &'static [u8]) -> Scalar;
}

impl ProofTranscriptTrait for Transcript {
  fn append_protocol_name(&mut self, protocol_name: &'static [u8]) {
    self.append_message(b"protocol-name", protocol_name);
  }

  fn challenge_scalar(&mut self, label: &'static [u8]) -> Scalar {
    let mut buf = [0u8; 64];
    self.challenge_bytes(label, &mut buf);
    Scalar::from_bytes_mod_order_wide(&buf)
  }
}

pub trait AppendToTranscriptTrait {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript);
}

impl AppendToTranscriptTrait for CompressedCommitment {
  fn append_to_transcript(&self, label: &'static [u8], transcript: &mut Transcript) {
    transcript.append_message(label, self.comm.as_bytes());
  }
}

impl<'b> MulAssign<&'b Scalar> for Commitment {
  fn mul_assign(&mut self, scalar: &'b Scalar) {
    let result = (self as &Commitment).comm * scalar;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b> Mul<&'b Scalar> for &'a Commitment {
  type Output = Commitment;
  fn mul(self, scalar: &'b Scalar) -> Commitment {
    Commitment {
      comm: self.comm * scalar,
    }
  }
}

impl<'a, 'b> Mul<&'b Commitment> for &'a Scalar {
  type Output = Commitment;

  fn mul(self, comm: &'b Commitment) -> Commitment {
    Commitment {
      comm: self * comm.comm,
    }
  }
}

macro_rules! define_mul_variants {
  (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
    impl<'b> Mul<&'b $rhs> for $lhs {
      type Output = $out;
      fn mul(self, rhs: &'b $rhs) -> $out {
        &self * rhs
      }
    }

    impl<'a> Mul<$rhs> for &'a $lhs {
      type Output = $out;
      fn mul(self, rhs: $rhs) -> $out {
        self * &rhs
      }
    }

    impl Mul<$rhs> for $lhs {
      type Output = $out;
      fn mul(self, rhs: $rhs) -> $out {
        &self * &rhs
      }
    }
  };
}

macro_rules! define_mul_assign_variants {
  (LHS = $lhs:ty, RHS = $rhs:ty) => {
    impl MulAssign<$rhs> for $lhs {
      fn mul_assign(&mut self, rhs: $rhs) {
        *self *= &rhs;
      }
    }
  };
}

define_mul_assign_variants!(LHS = Commitment, RHS = Scalar);
define_mul_variants!(LHS = Commitment, RHS = Scalar, Output = Commitment);
define_mul_variants!(LHS = Scalar, RHS = Commitment, Output = Commitment);

impl<'b> AddAssign<&'b Commitment> for Commitment {
  fn add_assign(&mut self, other: &'b Commitment) {
    let result = (self as &Commitment).comm + other.comm;
    *self = Commitment { comm: result };
  }
}

impl<'a, 'b> Add<&'b Commitment> for &'a Commitment {
  type Output = Commitment;
  fn add(self, other: &'b Commitment) -> Commitment {
    Commitment {
      comm: self.comm + other.comm,
    }
  }
}

macro_rules! define_add_variants {
  (LHS = $lhs:ty, RHS = $rhs:ty, Output = $out:ty) => {
    impl<'b> Add<&'b $rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: &'b $rhs) -> $out {
        &self + rhs
      }
    }

    impl<'a> Add<$rhs> for &'a $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        self + &rhs
      }
    }

    impl Add<$rhs> for $lhs {
      type Output = $out;
      fn add(self, rhs: $rhs) -> $out {
        &self + &rhs
      }
    }
  };
}

macro_rules! define_add_assign_variants {
  (LHS = $lhs:ty, RHS = $rhs:ty) => {
    impl AddAssign<$rhs> for $lhs {
      fn add_assign(&mut self, rhs: $rhs) {
        *self += &rhs;
      }
    }
  };
}

define_add_assign_variants!(LHS = Commitment, RHS = Commitment);
define_add_variants!(LHS = Commitment, RHS = Commitment, Output = Commitment);
