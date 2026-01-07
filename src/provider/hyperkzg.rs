//! This module implements Nova's evaluation engine using `HyperKZG`, a KZG-based polynomial commitment for multilinear polynomials
//! HyperKZG is based on the transformation from univariate PCS to multilinear PCS in the Gemini paper (section 2.4.2 in <https://eprint.iacr.org/2022/420.pdf>).
//! However, there are some key differences:
//! (1) HyperKZG works with multilinear polynomials represented in evaluation form (rather than in coefficient form in Gemini's transformation).
//! This means that Spartan's polynomial IOP can use commit to its polynomials as-is without incurring any interpolations or FFTs.
//! (2) HyperKZG is specialized to use KZG as the univariate commitment scheme, so it includes several optimizations (both during the transformation of multilinear-to-univariate claims
//! and within the KZG commitment scheme implementation itself).
#![allow(non_snake_case)]
#[cfg(feature = "io")]
use crate::provider::{ptau::PtauFileError, read_ptau, write_ptau};
use crate::{
  errors::NovaError,
  gadgets::utils::to_bignat_repr,
  provider::traits::{DlogGroup, DlogGroupExt, PairingGroup},
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
    AbsorbInRO2Trait, AbsorbInROTrait, Engine, ROTrait, TranscriptEngineTrait, TranscriptReprTrait,
  },
};
use core::{
  iter,
  marker::PhantomData,
  ops::Range,
  ops::{Add, Mul, MulAssign},
  slice,
};
use ff::{Field, PrimeFieldBits};
use num_integer::Integer;
use num_traits::ToPrimitive;
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Alias to points on G1 that are in preprocessed form
type G1Affine<E> = <<E as Engine>::GE as DlogGroup>::AffineGroupElement;

/// Alias to points on G1 that are in preprocessed form
type G2Affine<E> = <<<E as Engine>::GE as PairingGroup>::G2 as DlogGroup>::AffineGroupElement;

/// Default number of target chunks used in splitting up polynomial division in the kzg_open closure
const DEFAULT_TARGET_CHUNKS: usize = 1 << 10;

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
      Self::setup_from_tau_direct(label, &powers_of_tau, tau)
    } else {
      Self::setup_from_tau_fixed_base_exp(label, &powers_of_tau)
    }
  }

  fn setup_from_tau_fixed_base_exp(label: &'static [u8], powers_of_tau: &[E::Scalar]) -> Self {
    let tau = powers_of_tau[1];

    let gen = <E::GE as DlogGroup>::gen();

    let ck = fixed_base_exp_comb_batch::<4, 16, 64, 2, 32, _>(gen, powers_of_tau);
    let ck = ck.par_iter().map(|p| p.affine()).collect();

    let h = *E::GE::from_label(label, 1).first().unwrap();

    let tau_H = (<<E::GE as PairingGroup>::G2 as DlogGroup>::gen() * tau).affine();

    Self { ck, h, tau_H }
  }

  fn setup_from_tau_direct(
    label: &'static [u8],
    powers_of_tau: &[E::Scalar],
    tau: E::Scalar,
  ) -> Self {
    let num_gens = powers_of_tau.len();

    let ck: Vec<G1Affine<E>> = (0..num_gens)
      .into_par_iter()
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
    let num_threads = rayon::current_num_threads();
    (0..n)
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
      .collect::<Vec<_>>()
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

  precompute_res.insert(0, [zero; V]);

  let precomputed_g: [_; POW_2_H] = std::array::from_fn(|j| precompute_res[j]);

  let zero = G::zero();

  scalars
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
    .collect::<Vec<_>>()
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
    Self::CommitmentKey::setup_from_rng(label, n, OsRng)
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
      .par_iter()
      .zip(r.par_iter())
      .map(|(commit, r_i)| Commitment {
        comm: *commit + (h * r_i),
      })
      .collect()
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

  fn batch_commit_small<T: Integer + Into<u64> + Copy + Sync + ToPrimitive>(
    ck: &Self::CommitmentKey,
    v: &[Vec<T>],
    r: &[E::Scalar],
  ) -> Vec<Self::Commitment> {
    assert!(v.len() == r.len());

    let max = v.iter().map(|v| v.len()).max().unwrap_or(0);
    assert!(ck.ck.len() >= max);

    let h = <E::GE as DlogGroup>::group(&ck.h);

    E::GE::batch_vartime_multiscalar_mul_small(v, &ck.ck[..max])
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

  #[cfg(feature = "io")]
  fn load_setup(
    reader: &mut (impl std::io::Read + std::io::Seek),
    label: &'static [u8],
    n: usize,
  ) -> Result<Self::CommitmentKey, PtauFileError> {
    let num = n.next_power_of_two();

    // read points as well as check sanity of ptau file
    let (g1_points, g2_points) = read_ptau(reader, num, 2)?;

    let ck = g1_points.to_vec();

    let tau_H = *g2_points.last().unwrap();

    let h = *E::GE::from_label(label, 1).first().unwrap();

    Ok(CommitmentKey { ck, h, tau_H })
  }

  /// Save keys
  #[cfg(feature = "io")]
  fn save_setup(
    ck: &Self::CommitmentKey,
    mut writer: &mut (impl std::io::Write + std::io::Seek),
  ) -> Result<(), PtauFileError> {
    let g1_points = ck.ck.clone();

    let g2_points = vec![ck.tau_H, ck.tau_H];
    let power = g1_points.len().next_power_of_two().trailing_zeros() + 1;

    write_ptau(&mut writer, g1_points, g2_points, power)
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
  pub(crate) G: G1Affine<E>,
  pub(crate) H: G2Affine<E>,
  pub(crate) tau_H: G2Affine<E>,
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
  v: Vec<[E::Scalar; 3]>,
}

impl<E: Engine> EvaluationArgument<E>
where
  E::GE: PairingGroup,
{
  /// Create a new evaluation argument
  pub fn new(com: Vec<G1Affine<E>>, w: [G1Affine<E>; 3], v: Vec<[E::Scalar; 3]>) -> Self {
    Self { com, w, v }
  }
  /// The KZG commitments to intermediate polynomials
  pub fn com(&self) -> &[G1Affine<E>] {
    &self.com
  }
  /// The KZG witnesses for batch openings
  pub fn w(&self) -> &[G1Affine<E>] {
    &self.w
  }
  /// The evaluations of the polynomials at challenge points
  pub fn v(&self) -> &[[E::Scalar; 3]] {
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
  fn get_batch_challenge(v: &[[E::Scalar; 3]], transcript: &mut <E as Engine>::TE) -> E::Scalar {
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
      // Divides polynomial f(x) by (x - u) to obtain the witness polynomial h(x) = f(x)/(x - u)
      // for KZG opening.
      //
      // This implementation uses a chunking strategy to enable parallelization:
      // - Divides the polynomial into chunks
      // - Processes chunks in parallel using Rayon's par_chunks_exact_mut
      // - Within each chunk, maintains a "running" partial result
      // - Combines results across chunk boundaries
      //
      // While this adds more total computation than the standard Horner's method,
      // the parallel execution provides significant speedup for large polynomials.
      //
      // Original sequential algorithm using Horner's method:
      // ```
      // let mut h = vec![E::Scalar::ZERO; d];
      // for i in (1..d).rev() {
      //   h[i - 1] = f[i] + h[i] * u;
      // }
      // ```
      //
      // The resulting polynomial h(x) satisfies: f(x) = h(x) * (x - u)
      // The degree of h(x) is one less than the degree of f(x).
      let div_by_monomial =
        |f: &[E::Scalar], u: E::Scalar, target_chunks: usize| -> Vec<E::Scalar> {
          assert!(!f.is_empty());
          let target_chunk_size = f.len() / target_chunks;
          let log2_chunk_size = target_chunk_size.max(1).ilog2();
          let chunk_size = 1 << log2_chunk_size;

          let u_to_the_chunk_size = (0..log2_chunk_size).fold(u, |u_pow, _| u_pow.square());
          let mut result = f.to_vec();
          result
            .par_chunks_mut(chunk_size)
            .zip(f.par_chunks(chunk_size))
            .for_each(|(chunk, f_chunk)| {
              for i in (0..chunk.len() - 1).rev() {
                chunk[i] = f_chunk[i] + u * chunk[i + 1];
              }
            });

          let mut iter = result.chunks_mut(chunk_size).rev();
          if let Some(last_chunk) = iter.next() {
            let mut prev_partial = last_chunk[0];
            for chunk in iter {
              prev_partial = chunk[0] + u_to_the_chunk_size * prev_partial;
              chunk[0] = prev_partial;
            }
          }

          result[1..]
            .par_chunks_exact_mut(chunk_size)
            .rev()
            .for_each(|chunk| {
              let mut prev_partial = chunk[chunk_size - 1];
              for e in chunk.iter_mut().rev().skip(1) {
                prev_partial *= u;
                *e += prev_partial;
              }
            });
          result[1..].to_vec()
        };

      let target_chunks = DEFAULT_TARGET_CHUNKS;
      let h = &div_by_monomial(f, u, target_chunks);

      E::CE::commit(ck, h, &E::Scalar::ZERO).comm.affine()
    };

    let kzg_open_batch = |f: &[Vec<E::Scalar>],
                          u: &[E::Scalar; 3],
                          transcript: &mut <E as Engine>::TE|
     -> (Vec<G1Affine<E>>, Vec<[E::Scalar; 3]>) {
      let poly_eval = |f: &[E::Scalar], u: E::Scalar| -> E::Scalar {
        // Horner's method
        let mut acc = E::Scalar::ZERO;
        for &fi in f.iter().rev() {
          acc = acc * u + fi;
        }

        acc
      };

      let scalar_vector_muladd = |a: &mut Vec<E::Scalar>, v: &Vec<E::Scalar>, s: E::Scalar| {
        assert!(a.len() >= v.len());
        a.par_iter_mut().zip(v.par_iter()).for_each(|(a_i, v_i)| {
          *a_i += s * *v_i;
        });
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
      // Note: u.len() is always 3.

      // The verifier needs f_i(u_j), so we compute them here
      // (V will compute B(u_j) itself)
      let mut v = vec![[E::Scalar::ZERO; 3]; k];
      v.par_iter_mut().zip_eq(f).for_each(|(v_j, f)| {
        // for each poly f
        // for each poly f (except the last one - since it is constant)
        v_j.par_iter_mut().enumerate().for_each(|(i, v_ij)| {
          // for each point u
          *v_ij = poly_eval(f, u[i]);
        });
      });

      let q = Self::get_batch_challenge(&v, transcript);
      let B = kzg_compute_batch_polynomial(f, q);

      // Now open B at u0, ..., u_{t-1}
      let w = u
        .into_par_iter()
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
      Pi.par_iter_mut().enumerate().for_each(|(j, Pi_j)| {
        *Pi_j = x[ell - i - 1] * (polys[i][2 * j + 1] - polys[i][2 * j]) + polys[i][2 * j];
      });

      polys.push(Pi);
    }

    // We do not need to commit to the first polynomial as it is already committed.
    // Compute commitments in parallel
    let r = vec![E::Scalar::ZERO; ell - 1];
    let com: Vec<G1Affine<E>> = E::CE::batch_commit(ck, &polys[1..], r.as_slice())
      .par_iter()
      .map(|i| i.comm.affine())
      .collect();

    // Phase 2
    // We do not need to add x to the transcript, because in our context x was obtained from the transcript.
    // We also do not need to absorb `C` and `eval` as they are already absorbed by the transcript by the caller
    let r = Self::compute_challenge(&com, transcript);
    let u = [r, -r, r * r];

    // Phase 3 -- create response
    let (w, v) = kzg_open_batch(&polys, &u, transcript);

    Ok(EvaluationArgument {
      com,
      w: w.try_into().expect("w should have length 3"),
      v,
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

    let u = [r, -r, r * r];

    // Setup vectors (Y, ypos, yneg) from pi.v
    if pi.v.len() != ell || pi.com.len() != ell - 1 {
      return Err(NovaError::ProofVerifyError {
        reason: "Invalid lengths of pi.v".to_string(),
      });
    }

    // Check consistency of (Y, ypos, yneg)
    for i in 0..ell {
      let ypos = pi.v[i][0];
      let yneg = pi.v[i][1];
      let Y = pi.v.get(i + 1).map_or(*y, |v| v[2]);
      if r.double() * Y
        != r * (E::Scalar::ONE - x[ell - i - 1]) * (ypos + yneg) + x[ell - i - 1] * (ypos - yneg)
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
    let B_u = (0..3)
      .into_par_iter()
      .map(|i| {
        pi.v
          .iter()
          .rev()
          .fold(E::Scalar::ZERO, |acc, v_j| acc * q + v_j[i])
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
  use super::*;
  #[cfg(feature = "io")]
  use crate::provider::hyperkzg;
  use crate::{
    provider::{keccak::Keccak256Transcript, Bn256EngineKZG},
    spartan::polys::multilinear::MultilinearPolynomial,
  };
  use rand::SeedableRng;
  #[cfg(feature = "io")]
  use std::{
    fs::OpenOptions,
    io::{BufReader, BufWriter},
  };

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

    let config = bincode::config::legacy()
      .with_big_endian()
      .with_fixed_int_encoding();
    let proof_bytes =
      bincode::serde::encode_to_vec(&proof, config).expect("Failed to serialize proof");
    assert_eq!(proof_bytes.len(), 336);

    // Change the proof and expect verification to fail
    let mut bad_proof = proof.clone();
    let v1 = bad_proof.v[1];
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
      let v1 = bad_proof.v[1];
      bad_proof.v[0].clone_from(&v1);
      let mut verifier_tr2 = Keccak256Transcript::new(b"TestEval");
      assert!(
        EvaluationEngine::verify(&vk, &mut verifier_tr2, &C, &point, &eval, &bad_proof).is_err()
      );
    }
  }

  #[cfg(feature = "io")]
  #[ignore = "only available with external ptau files"]
  #[test]
  fn test_hyperkzg_large_from_file() {
    // test the hyperkzg prover and verifier with random instances (derived from a seed)
    for ell in [4, 5, 6] {
      let mut rng = rand::rngs::StdRng::seed_from_u64(ell as u64);

      let n = 1 << ell; // n = 2^ell

      let poly = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let point = (0..ell).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let eval = MultilinearPolynomial::evaluate_with(&poly, &point);

      let mut reader = BufReader::new(std::fs::File::open("/tmp/ppot_0080_13.ptau").unwrap());

      let ck: CommitmentKey<E> = CommitmentEngine::load_setup(&mut reader, b"test", n).unwrap();
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
      let v1 = bad_proof.v[1];
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
    let res1 = CommitmentKey::<E>::setup_from_tau_direct(label, &powers_of_tau, tau);
    let res2 = CommitmentKey::<E>::setup_from_tau_fixed_base_exp(label, &powers_of_tau);

    assert_eq!(res1.ck.len(), res2.ck.len());
    assert_eq!(res1.h, res2.h);
    assert_eq!(res1.tau_H, res2.tau_H);
    for i in 0..res1.ck.len() {
      assert_eq!(res1.ck[i], res2.ck[i]);
    }
  }

  #[cfg(feature = "io")]
  #[test]
  fn test_save_load_ck() {
    const BUFFER_SIZE: usize = 64 * 1024;
    const LABEL: &[u8] = b"test";

    let n = 4;
    let filename = "/tmp/kzg_test.ptau";

    let ck: CommitmentKey<E> = CommitmentEngine::setup(LABEL, n);

    let file = OpenOptions::new()
      .write(true)
      .create(true)
      .truncate(true)
      .open(filename)
      .unwrap();
    let mut writer = BufWriter::with_capacity(BUFFER_SIZE, file);

    CommitmentEngine::save_setup(&ck, &mut writer).unwrap();

    let file = OpenOptions::new().read(true).open(filename).unwrap();

    let mut reader = BufReader::new(file);

    let read_ck =
      hyperkzg::CommitmentEngine::<E>::load_setup(&mut reader, LABEL, ck.ck.len()).unwrap();

    assert_eq!(ck.ck.len(), read_ck.ck.len());
    assert_eq!(ck.h, read_ck.h);
    assert_eq!(ck.tau_H, read_ck.tau_H);
    for i in 0..ck.ck.len() {
      assert_eq!(ck.ck[i], read_ck.ck[i]);
    }
  }

  #[cfg(feature = "io")]
  #[ignore = "only available with external ptau files"]
  #[test]
  fn test_load_ptau() {
    let filename = "/tmp/ppot_0080_13.ptau";
    let file = OpenOptions::new().read(true).open(filename).unwrap();

    let mut reader = BufReader::new(file);

    let ck = hyperkzg::CommitmentEngine::<E>::load_setup(&mut reader, b"test", 1).unwrap();

    let mut rng = rand::thread_rng();

    let gen_g1 = ck.ck[0];
    let t_g2 = ck.tau_H;

    for _ in 0..1000 {
      let x = Fr::from(<rand::rngs::ThreadRng as rand::Rng>::gen::<u64>(&mut rng));
      let x = x * x * x * x;

      let left = halo2curves::bn256::G1::pairing(&(gen_g1 * x), &t_g2.into());
      let right = halo2curves::bn256::G1::pairing(&gen_g1.into(), &t_g2.into()) * x;

      assert_eq!(left, right);
    }
  }
}
