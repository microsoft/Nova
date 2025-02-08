//! This module implements the types that form the R1CS NeutronNova folding schemes running instance (i.e Running zero-fold instance)
//!
//! The running instance is of NSC × NSC_pc × ZC_pc

use crate::{
  r1cs::{R1CSInstance, R1CSWitness},
  spartan::polys::power::PowPoly,
  traits::Engine,
  Commitment,
};
use ff::Field;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
/// Running ZeroFold instance i.e. the running instance of the NeutronNova folding scheme
pub struct RunningZFInstance<E>
where
  E: Engine,
{
  pub(crate) nsc: NSCInstance<E>,
  pub(crate) nsc_pc: NSCPCInstance<E>,
  pub(crate) zc_pc: ZCPCInstance<E>,
}

impl<E> RunningZFInstance<E>
where
  E: Engine,
{
  /// Fold this [`RunningZFInstance`] with another [`RunningZFInstance`]
  pub(crate) fn fold(
    &self,
    nsc_U2: &NSCInstance<E>,
    r_b: E::Scalar,
    T: E::Scalar,
    // pc inputs
    nsc_U2_pc: &NSCPCInstance<E>,
    T_pc: E::Scalar,
    // zc-pc inputs
    nsc_U2_zc_pc: ZCPCInstance<E>,
  ) -> Self {
    let nsc_U = self.nsc().fold(nsc_U2, r_b, T);
    let nsc_pc_U = self.nsc_pc().fold(nsc_U2_pc, r_b, T_pc);
    Self {
      nsc: nsc_U,
      nsc_pc: nsc_pc_U,
      zc_pc: nsc_U2_zc_pc,
    }
  }

  /// Get the [`NSCInstance`]
  pub(crate) fn nsc(&self) -> &NSCInstance<E> {
    &self.nsc
  }

  /// Get the [`NSCPCInstance`]
  pub(crate) fn nsc_pc(&self) -> &NSCPCInstance<E> {
    &self.nsc_pc
  }

  /// Get the [`ZCPCInstance`]
  pub(crate) fn zc_pc(&self) -> &ZCPCInstance<E> {
    &self.zc_pc
  }

  /// Create a default instance of [`RunningZFInstance`]
  pub fn default(U: R1CSInstance<E>, comm_e: Commitment<E>) -> Self {
    // TODO: Create proper default running instances and witnesses withoug cloning a satisiable R1CS instance
    Self {
      nsc: NSCInstance::default(U, comm_e),
      nsc_pc: NSCPCInstance::default(comm_e),
      zc_pc: ZCPCInstance::default(comm_e),
    }
  }
}

/// Nested sumcheck instance created from a zero-check instance
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NSCInstance<E>
where
  E: Engine,
{
  T: E::Scalar,
  U: R1CSInstance<E>,
  comm_e: Commitment<E>,
}

impl<E> NSCInstance<E>
where
  E: Engine,
{
  /// Fold this [`NSCInstance`] with another [`NSCInstance`]
  pub(crate) fn fold(&self, other: &Self, r_b: E::Scalar, T: E::Scalar) -> Self {
    let (U1, U2) = (self.U(), other.U());
    let U = U1.fold(U2, r_b);
    let folded_comm_e = self.comm_e() * (E::Scalar::ONE - r_b) + other.comm_e() * r_b;
    NSCInstance::new(T, U, folded_comm_e)
  }

  /// Create a new instance of [`NSCInstance`]
  pub(crate) fn new(T: E::Scalar, U: R1CSInstance<E>, comm_e: Commitment<E>) -> Self {
    NSCInstance { T, U, comm_e }
  }

  /// Get the [`R1CSInstance`]
  pub(crate) fn U(&self) -> &R1CSInstance<E> {
    &self.U
  }

  /// Get the claim T
  pub(crate) fn T(&self) -> E::Scalar {
    self.T
  }

  /// Get the commitment to e
  pub(crate) fn comm_e(&self) -> Commitment<E> {
    self.comm_e
  }

  fn default(U: R1CSInstance<E>, comm_e: Commitment<E>) -> Self {
    Self {
      T: E::Scalar::ZERO,
      U,
      comm_e,
    }
  }
}

/// Nested sumcheck power-check instance
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NSCPCInstance<E>
where
  E: Engine,
{
  T: E::Scalar,
  comm_e: Commitment<E>,
  tau: E::Scalar,
  new_comm_e: Commitment<E>,
}

impl<E> NSCPCInstance<E>
where
  E: Engine,
{
  /// Create a new instance of [`NSCPCInstance`]
  pub(crate) fn new(
    T: E::Scalar,
    comm_e: Commitment<E>,
    tau: E::Scalar,
    new_comm_e: Commitment<E>,
  ) -> Self {
    NSCPCInstance {
      T,
      comm_e,
      tau,
      new_comm_e,
    }
  }

  /// Fold this [`NSCPCInstance`] with another [`NSCPCInstance`]
  pub(crate) fn fold(&self, other: &Self, r_b: E::Scalar, T: E::Scalar) -> Self {
    let folded_comm_e = self.comm_e() * (E::Scalar::ONE - r_b) + other.comm_e() * r_b;
    let folded_new_comm_e = self.new_comm_e() * (E::Scalar::ONE - r_b) + other.new_comm_e() * r_b;
    let tau = self.tau() * (E::Scalar::ONE - r_b) + other.tau() * r_b;
    NSCPCInstance::new(T, folded_comm_e, tau, folded_new_comm_e)
  }

  /// Get the claim T
  pub(crate) fn T(&self) -> E::Scalar {
    self.T
  }

  /// Get the commitment to e
  pub(crate) fn comm_e(&self) -> Commitment<E> {
    self.comm_e
  }

  /// Get the commitment to new e
  pub(crate) fn new_comm_e(&self) -> Commitment<E> {
    self.new_comm_e
  }

  /// Get tau
  pub(crate) fn tau(&self) -> E::Scalar {
    self.tau
  }

  /// Default instance of [`NSCPCInstance`]
  pub(crate) fn default(comm_e: Commitment<E>) -> Self {
    Self {
      T: E::Scalar::ZERO,
      tau: E::Scalar::ZERO,
      comm_e,
      new_comm_e: comm_e,
    }
  }
}

/// zero-check instance for power-check
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ZCPCInstance<E>
where
  E: Engine,
{
  comm_e: Commitment<E>,
  tau: E::Scalar,
}

impl<E> ZCPCInstance<E>
where
  E: Engine,
{
  /// Create a new instance of [`ZCPCInstance`]
  pub(crate) fn new(comm_e: Commitment<E>, tau: E::Scalar) -> Self {
    ZCPCInstance { comm_e, tau }
  }

  /// Get the commitment
  pub(crate) fn comm_e(&self) -> Commitment<E> {
    self.comm_e
  }

  /// Default instance of [`ZCPCInstance`]
  pub(crate) fn default(comm_e: Commitment<E>) -> Self {
    Self {
      comm_e,
      tau: E::Scalar::ZERO,
    }
  }
}

/// Running ZeroFold witness
#[derive(Debug, Clone)]
pub struct RunningZFWitness<E>
where
  E: Engine,
{
  pub(crate) nsc: NSCWitness<E>,
  pub(crate) nsc_pc: NSCPCWitness<E>,
  pub(crate) zc_pc: ZCPCWitness<E>,
}

impl<E> RunningZFWitness<E>
where
  E: Engine,
{
  /// Fold this [`RunningZFWitness`] with another [`RunningZFWitness`]
  pub(crate) fn fold(
    &self,
    nsc_W2: &NSCWitness<E>,
    r_b: E::Scalar,
    // nsc pc inputs
    nsc_pc_W2: &NSCPCWitness<E>,
    // zc-pc inputs
    nsc_W2_zc_pc: ZCPCWitness<E>,
  ) -> Self {
    let nsc_W = self.nsc().fold(nsc_W2, r_b);
    let nsc_pc_W = self.nsc_pc().fold(nsc_pc_W2, r_b);
    Self {
      nsc: nsc_W,
      nsc_pc: nsc_pc_W,
      zc_pc: nsc_W2_zc_pc,
    }
  }

  /// Get the [`NSCWitness`]
  pub(crate) fn nsc(&self) -> &NSCWitness<E> {
    &self.nsc
  }

  /// Get the [`NSCPCWitness`]
  pub(crate) fn nsc_pc(&self) -> &NSCPCWitness<E> {
    &self.nsc_pc
  }

  /// Get the [`ZCPCWitness`]
  pub(crate) fn zc_pc(&self) -> &ZCPCWitness<E> {
    &self.zc_pc
  }

  /// Create a default instance of [`RunningZFWitness`]
  pub fn default(W: R1CSWitness<E>, ell: usize) -> Self {
    // TODO: Create proper default running instances and witnesses withoug cloning a satisiable R1CS witness
    Self {
      nsc: NSCWitness::default(W, ell),
      nsc_pc: NSCPCWitness::default(ell),
      zc_pc: ZCPCWitness::default(ell),
    }
  }
}

/// Nested sumcheck witness
#[derive(Debug, Clone)]
pub struct NSCWitness<E>
where
  E: Engine,
{
  W: R1CSWitness<E>,
  e: PowPoly<E::Scalar>,
  r_e: E::Scalar,
}

impl<E> NSCWitness<E>
where
  E: Engine,
{
  /// Create a new instance of [`NSCWitness`]
  pub(crate) fn new(W: R1CSWitness<E>, e: PowPoly<E::Scalar>, r_e: E::Scalar) -> Self {
    NSCWitness { W, e, r_e }
  }

  /// Fold this [`NSCWitness`] with another [`NSCWitness`]
  pub(crate) fn fold(&self, other: &Self, r_b: E::Scalar) -> Self {
    let (W1, W2) = (self.W(), other.W());
    let W = W1.fold(W2, r_b);
    let folded_e = self.e().fold(other.e(), r_b);
    let r_e = self.r_e() * (E::Scalar::ONE - r_b) + other.r_e() * r_b;
    NSCWitness::new(W, folded_e, r_e)
  }

  /// Get the [`R1CSWitness`]
  pub(crate) fn W(&self) -> &R1CSWitness<E> {
    &self.W
  }

  /// Get the [`PowPoly`]
  pub(crate) fn e(&self) -> &PowPoly<E::Scalar> {
    &self.e
  }

  /// Get the r_e
  pub(crate) fn r_e(&self) -> E::Scalar {
    self.r_e
  }

  /// Create a default instance of [`NSCWitness`]
  pub fn default(W: R1CSWitness<E>, ell: usize) -> Self {
    Self {
      W,
      e: PowPoly::new(E::Scalar::ZERO, ell),
      r_e: E::Scalar::ZERO,
    }
  }
}

/// Running NSC power-check witness
#[derive(Debug, Clone)]
pub struct NSCPCWitness<E>
where
  E: Engine,
{
  e: PowPoly<E::Scalar>,
  new_e: PowPoly<E::Scalar>,
  r_e: E::Scalar,
  new_r_e: E::Scalar,
}

impl<E> NSCPCWitness<E>
where
  E: Engine,
{
  /// Create a new instance of [`NSCPCWitness`]
  pub(crate) fn new(
    e: PowPoly<E::Scalar>,
    new_e: PowPoly<E::Scalar>,
    r_e: E::Scalar,
    new_r_e: E::Scalar,
  ) -> Self {
    NSCPCWitness {
      e,
      new_e,
      r_e,
      new_r_e,
    }
  }

  /// Fold this [`NSCPCWitness`] with another [`NSCPCWitness`]
  pub(crate) fn fold(&self, other: &Self, r_b: E::Scalar) -> Self {
    let folded_e = self.e().fold(other.e(), r_b);
    let new_folded_e = self.new_e().fold(other.new_e(), r_b);
    let r_e = self.r_e() * (E::Scalar::ONE - r_b) + other.r_e() * r_b;
    let new_r_e = self.new_r_e() * (E::Scalar::ONE - r_b) + other.new_r_e() * r_b;

    NSCPCWitness::new(folded_e, new_folded_e, r_e, new_r_e)
  }

  /// Get the [`PowPoly`] e
  pub(crate) fn e(&self) -> &PowPoly<E::Scalar> {
    &self.e
  }

  /// Get the [`PowPoly`] new_e
  pub(crate) fn new_e(&self) -> &PowPoly<E::Scalar> {
    &self.new_e
  }

  /// Get the blinding factor r_e
  pub(crate) fn r_e(&self) -> E::Scalar {
    self.r_e
  }

  /// Get the blinding factor for new_e
  pub(crate) fn new_r_e(&self) -> E::Scalar {
    self.new_r_e
  }

  /// Create a default instance of [`NSCPCWitness`]
  pub fn default(ell: usize) -> Self {
    Self {
      e: PowPoly::new(E::Scalar::ZERO, ell),
      new_e: PowPoly::new(E::Scalar::ZERO, ell),
      r_e: E::Scalar::ZERO,
      new_r_e: E::Scalar::ZERO,
    }
  }
}

/// Running zero-check power-check witness
#[derive(Debug, Clone)]
pub struct ZCPCWitness<E>
where
  E: Engine,
{
  e: PowPoly<E::Scalar>,
  r_e: E::Scalar,
}

impl<E> ZCPCWitness<E>
where
  E: Engine,
{
  /// Create a new instance of [`ZCPCWitness`]
  pub(crate) fn new(e: PowPoly<E::Scalar>, r_e: E::Scalar) -> Self {
    ZCPCWitness { e, r_e }
  }

  /// Get the [`PowPoly`]
  pub(crate) fn e(&self) -> &PowPoly<E::Scalar> {
    &self.e
  }

  /// Get the blinding factor r_e
  pub(crate) fn r_e(&self) -> E::Scalar {
    self.r_e
  }

  /// Create a default instance of [`ZCPCWitness`]
  pub fn default(ell: usize) -> Self {
    Self {
      e: PowPoly::new(E::Scalar::ZERO, ell),
      r_e: E::Scalar::ZERO,
    }
  }
}
