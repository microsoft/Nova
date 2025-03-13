//! This module implements Nova's evaluation engine using `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in <https://eprint.iacr.org/2022/420.pdf>).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).
#![allow(non_snake_case)]
#[cfg(not(feature = "std"))]
use crate::prelude::*;
#[cfg(feature = "std")]
use crate::provider::{ptau::PtauFileError, read_ptau, write_ptau};
use crate::{
  errors::NovaError,
  gadgets::utils::to_bignat_repr,
  provider::traits::{DlogGroup, PairingGroup},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
    AbsorbInRO2Trait, AbsorbInROTrait, Engine, ROTrait, TranscriptEngineTrait, TranscriptReprTrait,
  },
};
use core::{
  array, iter,
  marker::PhantomData,
  ops::{Add, Mul, MulAssign},
  slice,
};
use ff::{Field, PrimeFieldBits};
#[cfg(not(feature = "std"))]
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "std")]
use rand_core::OsRng;
#[cfg(not(feature = "std"))]
use rand_core::SeedableRng;
#[cfg(feature = "std")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Alias to points on G1 that are in preprocessed form
type G1Affine<E> = <<E as Engine>::GE as DlogGroup>::AffineGroupElement;

/// Alias to points on G1 that are in preprocessed form
type G2Affine<E> = <<<E as Engine>::GE as PairingGroup>::G2 as DlogGroup>::AffineGroupElement;

/// KZG commitment key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentKey<E: Engine>
where
  E::GE: PairingGroup,
{
  ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
  h: <E::GE as DlogGroup>::AffineGroupElement,
  tau_H: <<E::GE as PairingGroup>::G2 as DlogGroup>::AffineGroupElement, // needed only for the verifier key
}

impl<E: Engine> CommitmentKey<E>
where
  E::GE: PairingGroup,
{
  /// Create a new commitment key
  pub fn new(
    ck: Vec<<E::GE as DlogGroup>::AffineGroupElement>,
    h: <E::GE as DlogGroup>::AffineGroupElement,
    tau_H: <<E::GE as PairingGroup>::G2 as DlogGroup>::AffineGroupElement,
  ) -> Self {
    Self { ck, h, tau_H }
  }

  /// Returns a reference to the ck field
  pub fn ck(&self) -> &[<E::GE as DlogGroup>::AffineGroupElement] {
    &self.ck
  }

  /// Returns a reference to the h field
  pub fn h(&self) -> &<E::GE as DlogGroup>::AffineGroupElement {
    &self.h
  }

  /// Returns a reference to the tau_H field
  pub fn tau_H(&self) -> &<<E::GE as PairingGroup>::G2 as DlogGroup>::AffineGroupElement {
    &self.tau_H
  }
}

impl<E: Engine> Len for CommitmentKey<E>
where
  E::GE: PairingGroup,
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

/// A KZG commitment
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct Commitment<E: Engine>
where
  E::GE: PairingGroup,
{
  comm: <E as Engine>::GE,
}

impl<E: Engine> Commitment<E>
where
  E::GE: PairingGroup,
{
  /// Creates a new commitment from the underlying group element
  pub fn new(comm: <E as Engine>::GE) -> Self {
    Commitment { comm }
  }
  /// Returns the commitment as a group element
  pub fn into_inner(self) -> <E as Engine>::GE {
    self.comm
  }
}

impl<E: Engine> CommitmentTrait<E> for Commitment<E>
where
  E::GE: PairingGroup,
{
  fn to_coordinates(&self) -> (E::Base, E::Base, bool) {
    self.comm.to_coordinates()
  }
}

impl<E: Engine> CommitmentKey<E>
where
  E::GE: PairingGroup,
{
  #[cfg(feature = "std")]
  /// Save keys
  pub fn save_to(
    &self,
    mut writer: &mut (impl std::io::Write + std::io::Seek),
  ) -> Result<(), PtauFileError> {
    let mut g1_points = Vec::with_capacity(self.ck.len() + 1);
    g1_points.push(self.h);
    g1_points.extend(self.ck.iter().cloned());

    let g2_points = vec![self.tau_H];
    let power = g1_points.len().next_power_of_two().trailing_zeros() + 1;

    write_ptau(&mut writer, g1_points, g2_points, power)
  }
}

impl<E: Engine> Default for Commitment<E>
where
  E::GE: PairingGroup,
{
  fn default() -> Self {
    Commitment {
      comm: E::GE::zero(),
    }
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for Commitment<E>
where
  E::GE: PairingGroup,
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
  E::GE: PairingGroup,
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
  E::GE: PairingGroup,
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
  E::GE: PairingGroup,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    let result = (self as &Commitment<E>).comm * scalar;
    *self = Commitment { comm: result };
  }
}

impl<'b, E: Engine> Mul<&'b E::Scalar> for &'_ Commitment<E>
where
  E::GE: PairingGroup,
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
  E::GE: PairingGroup,
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
  E::GE: PairingGroup,
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

impl<E: Engine> CommitmentKey<E>
where
  E::GE: PairingGroup,
{
  /// NOTE: this is for testing purposes and should not be used in production
  /// This can be used instead of `setup` to generate a reproducible commitment key
  pub fn setup_from_rng(label: &'static [u8], n: usize, rng: impl rand_core::RngCore) -> Self {
    const T1: usize = 1 << 16;
    const T2: usize = 100_000;

    let num_gens = n.next_power_of_two();

    let tau = E::Scalar::random(rng);

    let powers_of_tau = if num_gens < T1 {
      Self::compute_powers_serial(tau, num_gens)
    } else {
      Self::compute_powers_par(tau, num_gens)
    };

    if num_gens < T2 {
      Self::setup_from_tau_direct(label, &powers_of_tau)
    } else {
      Self::setup_from_tau_fixed_base_exp(label, &powers_of_tau)
    }
  }

  fn setup_from_tau_fixed_base_exp(label: &'static [u8], powers_of_tau: &[E::Scalar]) -> Self {
    let tau = powers_of_tau[1];

    let gen = <E::GE as DlogGroup>::gen();

    let ck = fixed_base_exp_comb_batch::<4, 16, 64, 2, 32, _>(gen, powers_of_tau);
    #[cfg(feature = "std")]
    let ck = ck.par_iter().map(|p| p.affine()).collect();
    #[cfg(not(feature = "std"))]
    let ck = ck.iter().map(|p| p.affine()).collect();

    let h = *E::GE::from_label(label, 1).first().unwrap();

    let tau_H = (<<E::GE as PairingGroup>::G2 as DlogGroup>::gen() * tau).affine();

    Self { ck, h, tau_H }
  }

  fn setup_from_tau_direct(label: &'static [u8], powers_of_tau: &[E::Scalar]) -> Self {
    let num_gens = powers_of_tau.len();
    let tau = powers_of_tau[1];

    #[cfg(feature = "std")]
    let ck: Vec<G1Affine<E>> = (0..num_gens)
      .into_par_iter()
      .map(|i| (<E::GE as DlogGroup>::gen() * powers_of_tau[i]).affine())
      .collect();
    #[cfg(not(feature = "std"))]
    let ck: Vec<G1Affine<E>> = (0..num_gens)
      .into_iter()
      .map(|i| (<E::GE as DlogGroup>::gen() * powers_of_tau[i]).affine())
      .collect();

    let h = *E::GE::from_label(label, 1).first().unwrap();

    let tau_H = (<<E::GE as PairingGroup>::G2 as DlogGroup>::gen() * tau).affine();

    Self { ck, h, tau_H }
  }

  fn compute_powers_serial(tau: E::Scalar, n: usize) -> Vec<E::Scalar> {
    let mut powers_of_tau = Vec::with_capacity(n);
    powers_of_tau.insert(0, E::Scalar::ONE);
    for i in 1..n {
      powers_of_tau.insert(i, powers_of_tau[i - 1] * tau);
    }
    powers_of_tau
  }

  fn compute_powers_par(tau: E::Scalar, n: usize) -> Vec<E::Scalar> {
    #[cfg(feature = "std")]
    let num_threads = rayon::current_num_threads();
    #[cfg(not(feature = "std"))]
    let num_threads = 1;

    #[cfg(feature = "std")]
    let res = (0..n)
      .collect::<Vec<_>>()
      .par_chunks(std::cmp::max(n / num_threads, 1))
      .into_par_iter()
      .map(|sub_list| {
        let mut res = Vec::with_capacity(sub_list.len());
        res.push(tau.pow([sub_list[0] as u64]));
        for i in 1..sub_list.len() {
          res.push(res[i - 1] * tau);
        }
        res
      })
      .flatten()
      .collect::<Vec<_>>();
    #[cfg(not(feature = "std"))]
    let res = (0..n)
      .collect::<Vec<_>>()
      .chunks(max(n / num_threads, 1))
      .into_iter()
      .map(|sub_list| {
        let mut res = Vec::with_capacity(sub_list.len());
        res.push(tau.pow([sub_list[0] as u64]));
        for i in 1..sub_list.len() {
          res.push(res[i - 1] * tau);
        }
        res
      })
      .flatten()
      .collect::<Vec<_>>();

    res
  }
}

// * Implementation of https://www.weimerskirch.org/files/Weimerskirch_FixedBase.pdf
fn fixed_base_exp_comb_batch<
  const H: usize,
  const POW_2_H: usize,
  const A: usize,
  const B: usize,
  const V: usize,
  G: DlogGroup,
>(
  gen: G,
  scalars: &[G::Scalar],
) -> Vec<G> {
  assert_eq!(1 << H, POW_2_H);
  assert_eq!(A, V * B);
  assert!(A <= 64);

  let zero = G::zero();
  let one = gen;

  let gi = {
    let mut res = [one; H];
    for i in 1..H {
      let prod = (0..A).fold(res[i - 1], |acc, _| acc + acc);
      res[i] = prod;
    }
    res
  };

  #[cfg(feature = "std")]
  let mut precompute_res = (1..POW_2_H)
    .into_par_iter()
    .map(|i| {
      let mut res = [zero; V];

      // * G[0][i]
      let mut g_0_i = zero;
      for (j, item) in gi.iter().enumerate().take(H) {
        if (1 << j) & i > 0 {
          g_0_i += item;
        }
      }

      res[0] = g_0_i;

      // * G[j][i]
      for j in 1..V {
        res[j] = (0..B).fold(res[j - 1], |acc, _| acc + acc);
      }

      res
    })
    .collect::<Vec<_>>();
  #[cfg(not(feature = "std"))]
  let mut precompute_res = (1..POW_2_H)
    .into_iter()
    .map(|i| {
      let mut res = [zero; V];

      // * G[0][i]
      let mut g_0_i = zero;
      for (j, item) in gi.iter().enumerate().take(H) {
        if (1 << j) & i > 0 {
          g_0_i += item;
        }
      }

      res[0] = g_0_i;

      // * G[j][i]
      for j in 1..V {
        res[j] = (0..B).fold(res[j - 1], |acc, _| acc + acc);
      }

      res
    })
    .collect::<Vec<_>>();

  precompute_res.insert(0, [zero; V]);

  let precomputed_g: [_; POW_2_H] = array::from_fn(|j| precompute_res[j]);

  let zero = G::zero();
  #[cfg(feature = "std")]
  let res = scalars
    .par_iter()
    .map(|e| {
      let mut a = zero;
      let mut bits = e.to_le_bits().into_iter().collect::<Vec<_>>();

      while bits.len() % A != 0 {
        bits.push(false);
      }

      for k in (0..B).rev() {
        a += a;
        for j in (0..V).rev() {
          let i_j_k = (0..H)
            .map(|h| {
              let b = bits[h * A + j * B + k];
              (1 << h) * b as usize
            })
            .sum::<usize>();

          if i_j_k > 0 {
            a += precomputed_g[i_j_k][j];
          }
        }
      }

      a
    })
    .collect::<Vec<_>>();
  #[cfg(not(feature = "std"))]
  let res = scalars
    .iter()
    .map(|e| {
      let mut a = zero;
      let mut bits = e.to_le_bits().into_iter().collect::<Vec<_>>();

      while bits.len() % A != 0 {
        bits.push(false);
      }

      for k in (0..B).rev() {
        a += a;
        for j in (0..V).rev() {
          let i_j_k = (0..H)
            .map(|h| {
              let b = bits[h * A + j * B + k];
              (1 << h) * b as usize
            })
            .sum::<usize>();

          if i_j_k > 0 {
            a += precomputed_g[i_j_k][j];
          }
        }
      }

      a
    })
    .collect::<Vec<_>>();

  res
}

impl<E: Engine> CommitmentEngineTrait<E> for CommitmentEngine<E>
where
  E::GE: PairingGroup,
{
  type Commitment = Commitment<E>;
  type CommitmentKey = CommitmentKey<E>;
  type DerandKey = DerandKey<E>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    // NOTE: this is for testing purposes and should not be used in production
    #[cfg(feature = "std")]
    let res = Self::CommitmentKey::setup_from_rng(label, n, OsRng);
    #[cfg(not(feature = "std"))]
    let res = Self::CommitmentKey::setup_from_rng(label, n, ChaCha20Rng::seed_from_u64(0xDEADBEEF));

    res
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

  fn batch_commit(
    ck: &Self::CommitmentKey,
    v: &[Vec<<E as Engine>::Scalar>],
    r: &[<E as Engine>::Scalar],
  ) -> Vec<Self::Commitment> {
    assert!(v.len() == r.len());

    let max = v.iter().map(|v| v.len()).max().unwrap_or(0);
    assert!(ck.ck.len() >= max);

    let h = <E::GE as DlogGroup>::group(&ck.h);

    E::GE::batch_vartime_multiscalar_mul(v, &ck.ck[..max])
      .iter()
      .zip(r.iter())
      .map(|(commit, r_i)| Commitment {
        comm: *commit + (h * r_i),
      })
      .collect()
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

  #[cfg(feature = "std")]
  fn load_setup(
    reader: &mut (impl std::io::Read + std::io::Seek),
    n: usize,
  ) -> Result<Self::CommitmentKey, PtauFileError> {
    let num = n.next_power_of_two();

    let (g1_points, g2_points) = read_ptau(reader, num + 1, 1)?;

    let (h, ck) = g1_points.split_at(1);
    let h = h[0];
    let ck = ck.to_vec();

    Ok(CommitmentKey {
      ck,
      h,
      tau_H: g2_points[0],
    })
  }
}

/// Provides an implementation of generators for proving evaluations
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine> {
  _p: PhantomData<E>,
}

/// A verifier key
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine>
where
  E::GE: PairingGroup,
{
  G: G1Affine<E>,
  H: G2Affine<E>,
  tau_H: G2Affine<E>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EvaluationArgument<E: Engine>
where
  E::GE: PairingGroup,
{
  com: Vec<G1Affine<E>>,
  w: [G1Affine<E>; 3],
  v: [Vec<E::Scalar>; 3],
}

impl<E: Engine> EvaluationArgument<E>
where
  E::GE: PairingGroup,
{
  /// The KZG commitments to intermediate polynomials
  pub fn com(&self) -> &[G1Affine<E>] {
    &self.com
  }
  /// The KZG witnesses for batch openings
  pub fn w(&self) -> &[G1Affine<E>] {
    &self.w
  }
  /// The evaluations of the polynomials at challenge points
  pub fn v(&self) -> &[Vec<E::Scalar>] {
    &self.v
  }
}

/// Provides an implementation of a polynomial evaluation engine using KZG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine> EvaluationEngine<E>
where
  E::GE: PairingGroup,
{
  // This impl block defines helper functions that are not a part of
  // EvaluationEngineTrait, but that we will use to implement the trait methods.
  fn compute_challenge(com: &[G1Affine<E>], transcript: &mut <E as Engine>::TE) -> E::Scalar {
    transcript.absorb(b"c", &com.to_vec().as_slice());

    transcript.squeeze(b"c").unwrap()
  }

  // Compute challenge q = Hash(vk, C0, ..., C_{k-1}, u0, ...., u_{t-1},
  // (f_i(u_j))_{i=0..k-1,j=0..t-1})
  fn get_batch_challenge(v: &[Vec<E::Scalar>], transcript: &mut <E as Engine>::TE) -> E::Scalar {
    transcript.absorb(
      b"v",
      &v.iter()
        .flatten()
        .cloned()
        .collect::<Vec<E::Scalar>>()
        .as_slice(),
    );

    transcript.squeeze(b"r").unwrap()
  }

  fn batch_challenge_powers(q: E::Scalar, k: usize) -> Vec<E::Scalar> {
    // Compute powers of q : (1, q, q^2, ..., q^(k-1))
    let mut q_powers = vec![E::Scalar::ONE; k];
    for i in 1..k {
      q_powers[i] = q_powers[i - 1] * q;
    }
    q_powers
  }

  fn verifier_second_challenge(W: &[G1Affine<E>], transcript: &mut <E as Engine>::TE) -> E::Scalar {
    transcript.absorb(b"W", &W.to_vec().as_slice());

    transcript.squeeze(b"d").unwrap()
  }
}

impl<E> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine<CE = CommitmentEngine<E>>,
  E::GE: PairingGroup,
{
  type EvaluationArgument = EvaluationArgument<E>;
  type ProverKey = ProverKey<E>;
  type VerifierKey = VerifierKey<E>;

  fn setup(
    ck: &<E::CE as CommitmentEngineTrait<E>>::CommitmentKey,
  ) -> (Self::ProverKey, Self::VerifierKey) {
    let pk = ProverKey {
      _p: Default::default(),
    };

    let vk = VerifierKey {
      G: E::GE::gen().affine(),
      H: <<E::GE as PairingGroup>::G2 as DlogGroup>::gen().affine(),
      tau_H: ck.tau_H,
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<E>,
    _pk: &Self::ProverKey,
    transcript: &mut <E as Engine>::TE,
    _C: &Commitment<E>,
    hat_P: &[E::Scalar],
    point: &[E::Scalar],
    _eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let x: Vec<E::Scalar> = point.to_vec();

    //////////////// begin helper closures //////////
    let kzg_open = |f: &[E::Scalar], u: E::Scalar| -> G1Affine<E> {
      // On input f(x) and u compute the witness polynomial used to prove
      // that f(u) = v. The main part of this is to compute the
      // division (f(x) - f(u)) / (x - u), but we don't use a general
      // division algorithm, we make use of the fact that the division
      // never has a remainder, and that the denominator is always a linear
      // polynomial. The cost is (d-1) mults + (d-1) adds in E::Scalar, where
      // d is the degree of f.
      //
      // We use the fact that if we compute the quotient of f(x)/(x-u),
      // there will be a remainder, but it'll be v = f(u).  Put another way
      // the quotient of f(x)/(x-u) and (f(x) - f(v))/(x-u) is the
      // same.  One advantage is that computing f(u) could be decoupled
      // from kzg_open, it could be done later or separate from computing W.

      let compute_witness_polynomial = |f: &[E::Scalar], u: E::Scalar| -> Vec<E::Scalar> {
        let d = f.len();

        // Compute h(x) = f(x)/(x - u)
        let mut h = vec![E::Scalar::ZERO; d];
        for i in (1..d).rev() {
          h[i - 1] = f[i] + h[i] * u;
        }

        h
      };

      let h = compute_witness_polynomial(f, u);

      E::CE::commit(ck, &h, &E::Scalar::ZERO).comm.affine()
    };

    let kzg_open_batch = |f: &[Vec<E::Scalar>],
                          u: &[E::Scalar],
                          transcript: &mut <E as Engine>::TE|
     -> (Vec<G1Affine<E>>, Vec<Vec<E::Scalar>>) {
      let poly_eval = |f: &[E::Scalar], u: E::Scalar| -> E::Scalar {
        let mut v = f[0];
        let mut u_power = E::Scalar::ONE;

        for fi in f.iter().skip(1) {
          u_power *= u;
          v += u_power * fi;
        }

        v
      };

      let scalar_vector_muladd = |a: &mut Vec<E::Scalar>, v: &Vec<E::Scalar>, s: E::Scalar| {
        assert!(a.len() >= v.len());
        for i in 0..v.len() {
          a[i] += s * v[i];
        }
      };

      let kzg_compute_batch_polynomial = |f: &[Vec<E::Scalar>], q: E::Scalar| -> Vec<E::Scalar> {
        let k = f.len(); // Number of polynomials we're batching

        let q_powers = Self::batch_challenge_powers(q, k);

        // Compute B(x) = f[0] + q*f[1] + q^2 * f[2] + ... q^(k-1) * f[k-1]
        let mut B = f[0].clone();
        for i in 1..k {
          scalar_vector_muladd(&mut B, &f[i], q_powers[i]); // B += q_powers[i] * f[i]
        }

        B
      };
      ///////// END kzg_open_batch closure helpers

      let k = f.len();
      let t = u.len();

      // The verifier needs f_i(u_j), so we compute them here
      // (V will compute B(u_j) itself)
      let mut v = vec![vec!(E::Scalar::ZERO; k); t];
      #[cfg(feature = "std")]
      v.par_iter_mut().enumerate().for_each(|(i, v_i)| {
        // for each point u
        v_i.par_iter_mut().zip_eq(f).for_each(|(v_ij, f)| {
          // for each poly f
          // for each poly f (except the last one - since it is constant)
          *v_ij = poly_eval(f, u[i]);
        });
      });
      #[cfg(not(feature = "std"))]
      v.iter_mut().enumerate().for_each(|(i, v_i)| {
        // for each point u
        v_i.iter_mut().zip_eq(f).for_each(|(v_ij, f)| {
          // for each poly f
          // for each poly f (except the last one - since it is constant)
          *v_ij = poly_eval(f, u[i]);
        });
      });

      let q = Self::get_batch_challenge(&v, transcript);
      let B = kzg_compute_batch_polynomial(f, q);

      // Now open B at u0, ..., u_{t-1}
      #[cfg(feature = "std")]
      let w = u
        .into_par_iter()
        .map(|ui| kzg_open(&B, *ui))
        .collect::<Vec<G1Affine<E>>>();
      #[cfg(not(feature = "std"))]
      let w = u
        .into_iter()
        .map(|ui| kzg_open(&B, *ui))
        .collect::<Vec<G1Affine<E>>>();

      // The prover computes the challenge to keep the transcript in the same
      // state as that of the verifier
      let _d_0 = Self::verifier_second_challenge(&w, transcript);

      (w, v)
    };

    ///// END helper closures //////////

    let ell = x.len();
    let n = hat_P.len();
    assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

    // Phase 1  -- create commitments com_1, ..., com_\ell
    // We do not compute final Pi (and its commitment) as it is constant and equals to 'eval'
    // also known to verifier, so can be derived on its side as well
    let mut polys: Vec<Vec<E::Scalar>> = Vec::new();
    polys.push(hat_P.to_vec());
    for i in 0..ell - 1 {
      let Pi_len = polys[i].len() / 2;
      let mut Pi = vec![E::Scalar::ZERO; Pi_len];

      #[allow(clippy::needless_range_loop)]
      #[cfg(feature = "std")]
      Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
        *Pi_j = x[ell - i - 1] * (polys[i][2 * j + 1] - polys[i][2 * j]) + polys[i][2 * j];
      });
      #[allow(clippy::needless_range_loop)]
      #[cfg(not(feature = "std"))]
      Pi.iter_mut().enumerate().for_each(|(j, Pi_j)| {
        *Pi_j = x[ell - i - 1] * (polys[i][2 * j + 1] - polys[i][2 * j]) + polys[i][2 * j];
      });

      polys.push(Pi);
    }

    // We do not need to commit to the first polynomial as it is already committed.
    // Compute commitments in parallel
    let r = vec![E::Scalar::ZERO; ell - 1];
    let com: Vec<G1Affine<E>> = E::CE::batch_commit(ck, &polys[1..], r.as_slice())
      .iter()
      .map(|i| i.comm.affine())
      .collect();

    // Phase 2
    // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
    // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
    let r = Self::compute_challenge(&com, transcript);
    let u = vec![r, -r, r * r];

    // Phase 3 -- create response
    let (w, v) = kzg_open_batch(&polys, &u, transcript);

    Ok(EvaluationArgument {
      com,
      w: w.try_into().expect("w should have length 3"),
      v: v.try_into().expect("v should have length 3"),
    })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut <E as Engine>::TE,
    C: &Commitment<E>,
    x: &[E::Scalar],
    y: &E::Scalar,
    pi: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let ell = x.len();

    // we do not need to add x to the transcript, because in our context x was
    // obtained from the transcript
    let r = Self::compute_challenge(&pi.com, transcript);

    if r == E::Scalar::ZERO || C.comm == E::GE::zero() {
      return Err(NovaError::ProofVerifyError {
        reason: "r is zero or commitment is zero".to_string(),
      });
    }

    let u = [r, -r, r * r];

    // Setup vectors (Y, ypos, yneg) from pi.v
    if pi.v[0].len() != ell
      || pi.v[1].len() != ell
      || pi.v[2].len() != ell
      || pi.com.len() != ell - 1
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid lengths of pi.v".to_string(),
      });
    }
    let ypos = &pi.v[0];
    let yneg = &pi.v[1];
    let Y = &pi.v[2];

    // Check consistency of (Y, ypos, yneg)
    for i in 0..ell {
      if r.double() * Y.get(i + 1).unwrap_or(y)
        != r * (E::Scalar::ONE - x[ell - i - 1]) * (ypos[i] + yneg[i])
          + x[ell - i - 1] * (ypos[i] - yneg[i])
      {
        return Err(NovaError::ProofVerifyError {
          reason: "Inconsistent (Y, ypos, yneg)".to_string(),
        });
      }
      // Note that we don't make any checks about Y[0] here, but our batching
      // check below requires it
    }

    // Check commitments to (Y, ypos, yneg) are valid

    // vk is hashed in transcript already, so we do not add it here

    let q = Self::get_batch_challenge(&pi.v, transcript);

    let d_0 = Self::verifier_second_challenge(&pi.w, transcript);
    let d_1 = d_0.square();

    // We write a special case for t=3, since this what is required for
    // hyperkzg. Following the paper directly, we must compute:
    // let L0 = C_B - vk.G * B_u[0] + W[0] * u[0];
    // let L1 = C_B - vk.G * B_u[1] + W[1] * u[1];
    // let L2 = C_B - vk.G * B_u[2] + W[2] * u[2];
    // let R0 = -W[0];
    // let R1 = -W[1];
    // let R2 = -W[2];
    // let L = L0 + L1*d_0 + L2*d_1;
    // let R = R0 + R1*d_0 + R2*d_1;
    //
    // We group terms to reduce the number of scalar mults (to seven):
    // In Rust, we could use MSMs for these, and speed up verification.
    //
    // Note, that while computing L, the intermediate computation of C_B together with computing
    // L0, L1, L2 can be replaced by single MSM of C with the powers of q multiplied by (1 + d_0 + d_1)
    // with additionally concatenated inputs for scalars/bases.

    let q_power_multiplier = E::Scalar::ONE + d_0 + d_1;

    let q_powers_multiplied: Vec<E::Scalar> =
      iter::successors(Some(q_power_multiplier), |qi| Some(*qi * q))
        .take(ell)
        .collect();

    // Compute the batched openings
    // compute B(u_i) = v[i][0] + q*v[i][1] + ... + q^(t-1) * v[i][t-1]
    #[cfg(feature = "std")]
    let B_u = pi
      .v
      .par_iter()
      .map(|v_i| {
        v_i
          .iter()
          .rev()
          .fold(E::Scalar::ZERO, |acc, v_ij| acc * q + v_ij)
      })
      .collect::<Vec<E::Scalar>>();
    #[cfg(not(feature = "std"))]
    let B_u = pi
      .v
      .iter()
      .map(|v_i| {
        v_i
          .iter()
          .rev()
          .fold(E::Scalar::ZERO, |acc, v_ij| acc * q + v_ij)
      })
      .collect::<Vec<E::Scalar>>();

    let L = E::GE::vartime_multiscalar_mul(
      &[
        &q_powers_multiplied[..],
        &[
          u[0],
          (u[1] * d_0),
          (u[2] * d_1),
          -(B_u[0] + d_0 * B_u[1] + d_1 * B_u[2]),
        ],
      ]
      .concat(),
      &[
        &[C.comm.affine()][..],
        &pi.com,
        &pi.w,
        slice::from_ref(&vk.G),
      ]
      .concat(),
    );

    let R0 = E::GE::group(&pi.w[0]);
    let R1 = E::GE::group(&pi.w[1]);
    let R2 = E::GE::group(&pi.w[2]);
    let R = R0 + R1 * d_0 + R2 * d_1;

    // Check that e(L, vk.H) == e(R, vk.tau_H)
    if (E::GE::pairing(&L, &DlogGroup::group(&vk.H)))
      != (E::GE::pairing(&R, &DlogGroup::group(&vk.tau_H)))
    {
      return Err(NovaError::ProofVerifyError {
        reason: "Pairing check failed".to_string(),
      });
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use std::{
    fs::OpenOptions,
    io::{BufReader, BufWriter},
  };

  use super::*;
  use crate::{
    provider::{hyperkzg, keccak::Keccak256Transcript, Bn256EngineKZG},
    spartan::polys::multilinear::MultilinearPolynomial,
  };
  use bincode::config::legacy;
  use rand::SeedableRng;

  type E = Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  #[test]
  fn test_hyperkzg_eval() {
    // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
    let n = 4;
    let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
    let (pk, vk): (ProverKey<E>, VerifierKey<E>) = EvaluationEngine::setup(&ck);

    // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
    let poly = vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)];

    let C = CommitmentEngine::commit(&ck, &poly, &<E as Engine>::Scalar::ZERO);

    let test_inner = |point: Vec<Fr>, eval: Fr| -> Result<(), NovaError> {
      let mut tr = Keccak256Transcript::new(b"TestEval");
      let proof = EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).unwrap();
      let mut tr = Keccak256Transcript::new(b"TestEval");
      EvaluationEngine::verify(&vk, &mut tr, &C, &point, &eval, &proof)
    };

    // Call the prover with a (point, eval) pair.
    // The prover does not recompute so it may produce a proof, but it should not verify
    let point = vec![Fr::from(0), Fr::from(0)];
    let eval = Fr::ONE;
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(1)];
    let eval = Fr::from(2);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(1), Fr::from(1)];
    let eval = Fr::from(4);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(3);
    assert!(test_inner(point, eval).is_ok());

    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(9);
    assert!(test_inner(point, eval).is_ok());

    // Try a couple incorrect evaluations and expect failure
    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(50);
    assert!(test_inner(point, eval).is_err());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(4);
    assert!(test_inner(point, eval).is_err());
  }

  #[test]
  fn test_hyperkzg_small() {
    let n = 4;

    // poly = [1, 2, 1, 4]
    let poly = vec![Fr::ONE, Fr::from(2), Fr::from(1), Fr::from(4)];

    // point = [4,3]
    let point = vec![Fr::from(4), Fr::from(3)];

    // eval = 28
    let eval = Fr::from(28);

    let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
    let (pk, vk) = EvaluationEngine::setup(&ck);

    // make a commitment
    let C = CommitmentEngine::commit(&ck, &poly, &<E as Engine>::Scalar::ZERO);

    // prove an evaluation
    let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
    let proof =
      EvaluationEngine::<E>::prove(&ck, &pk, &mut prover_transcript, &C, &poly, &point, &eval)
        .unwrap();
    let post_c_p = prover_transcript.squeeze(b"c").unwrap();

    // verify the evaluation
    let mut verifier_transcript = Keccak256Transcript::new(b"TestEval");
    assert!(
      EvaluationEngine::verify(&vk, &mut verifier_transcript, &C, &point, &eval, &proof).is_ok()
    );
    let post_c_v = verifier_transcript.squeeze(b"c").unwrap();

    // check if the prover transcript and verifier transcript are kept in the same state
    assert_eq!(post_c_p, post_c_v);

    let proof_bytes = bincode::serde::encode_to_vec(&proof, legacy()).unwrap();

    // let proof_bytes = bincode::DefaultOptions::new()
    //   .with_big_endian()
    //   .with_fixint_encoding()
    //   .serialize(&proof)
    //   .unwrap();
    assert_eq!(proof_bytes.len(), 352);

    // Change the proof and expect verification to fail
    let mut bad_proof = proof.clone();
    let v1 = bad_proof.v[1].clone();
    bad_proof.v[0].clone_from(&v1);
    let mut verifier_transcript2 = Keccak256Transcript::new(b"TestEval");
    assert!(EvaluationEngine::verify(
      &vk,
      &mut verifier_transcript2,
      &C,
      &point,
      &eval,
      &bad_proof
    )
    .is_err());
  }

  #[test]
  fn test_hyperkzg_large() {
    // test the hyperkzg prover and verifier with random instances (derived from a seed)
    for ell in [4, 5, 6] {
      let mut rng = rand::rngs::StdRng::seed_from_u64(ell as u64);

      let n = 1 << ell; // n = 2^ell

      let poly = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let point = (0..ell).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let eval = MultilinearPolynomial::evaluate_with(&poly, &point);

      let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
      let (pk, vk) = EvaluationEngine::setup(&ck);

      // make a commitment
      let C = CommitmentEngine::commit(&ck, &poly, &<E as Engine>::Scalar::ZERO);

      // prove an evaluation
      let mut prover_transcript = Keccak256Transcript::new(b"TestEval");
      let proof: EvaluationArgument<E> =
        EvaluationEngine::prove(&ck, &pk, &mut prover_transcript, &C, &poly, &point, &eval)
          .unwrap();

      // verify the evaluation
      let mut verifier_tr = Keccak256Transcript::new(b"TestEval");
      assert!(EvaluationEngine::verify(&vk, &mut verifier_tr, &C, &point, &eval, &proof).is_ok());

      // Change the proof and expect verification to fail
      let mut bad_proof = proof.clone();
      let v1 = bad_proof.v[1].clone();
      bad_proof.v[0].clone_from(&v1);
      let mut verifier_tr2 = Keccak256Transcript::new(b"TestEval");
      assert!(
        EvaluationEngine::verify(&vk, &mut verifier_tr2, &C, &point, &eval, &bad_proof).is_err()
      );
    }
  }

  #[test]
  fn test_key_gen() {
    let n = 100;
    let tau = Fr::random(OsRng);
    let powers_of_tau = CommitmentKey::<E>::compute_powers_serial(tau, n);
    let label = b"test";
    let res1 = CommitmentKey::<E>::setup_from_tau_direct(label, &powers_of_tau);
    let res2 = CommitmentKey::<E>::setup_from_tau_fixed_base_exp(label, &powers_of_tau);

    assert_eq!(res1.ck.len(), res2.ck.len());
    assert_eq!(res1.h, res2.h);
    assert_eq!(res1.tau_H, res2.tau_H);
    for i in 0..res1.ck.len() {
      assert_eq!(res1.ck[i], res2.ck[i]);
    }
  }

  #[test]
  fn test_save_load_ck() {
    let n = 4;
    let filename = "/tmp/kzg_test.ptau";
    const BUFFER_SIZE: usize = 64 * 1024;
    let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);

    let file = OpenOptions::new()
      .write(true)
      .create(true)
      .truncate(true)
      .open(filename)
      .unwrap();
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, file);

    ck.save_to(&mut writer).unwrap();

    let file = OpenOptions::new().read(true).open(filename).unwrap();

    let mut reader = BufReader::new(file);

    let read_ck = hyperkzg::CommitmentEngine::<E>::load_setup(&mut reader, ck.ck.len()).unwrap();

    assert_eq!(ck.ck.len(), read_ck.ck.len());
    assert_eq!(ck.h, read_ck.h);
    assert_eq!(ck.tau_H, read_ck.tau_H);
    for i in 0..ck.ck.len() {
      assert_eq!(ck.ck[i], read_ck.ck[i]);
    }
  }
}
