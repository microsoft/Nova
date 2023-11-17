//! This module implements Nova's evaluation engine using multilinear KZG:
#![allow(non_snake_case)]
use crate::{
  errors::NovaError,
  provider::{
    cpu_best_multiexp,
    keccak::Keccak256Transcript,
    poseidon::{PoseidonRO, PoseidonROCircuit},
    DlogGroup,
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
    AbsorbInROTrait, Engine, Group, ROTrait, TranscriptEngineTrait, TranscriptReprTrait,
  },
};
use core::{
  marker::PhantomData,
  ops::{Add, Mul, MulAssign},
};
use ff::{Field, PrimeField};
use halo2curves::{
  bn256::{Fq, Fr, G1Affine, G1Compressed, G2Affine, G1},
  group::{prime::PrimeCurveAffine, Curve},
  CurveAffine,
};
use rand_core::OsRng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// KZG commitment key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommitmentKey<E: Engine> {
  Gi: Vec<G1Affine>,

  // needed only for verification
  G: G1Affine,
  H: G2Affine,
  tauH: G2Affine,
  _p: PhantomData<E>,
}

impl<E: Engine> Len for CommitmentKey<E> {
  fn length(&self) -> usize {
    self.Gi.len()
  }
}

/// A KZG commitment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment<E: Engine> {
  comm: G1,
  _p: PhantomData<E>,
}

/// A compressed commitment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompressedCommitment<E>
where
  E: Engine,
{
  comm: G1Affine,
  _p: PhantomData<E>,
}

impl<E> CommitmentTrait<E> for Commitment<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  type CompressedCommitment = CompressedCommitment<E>;

  fn compress(&self) -> Self::CompressedCommitment {
    CompressedCommitment {
      comm: self.comm.to_affine(),
      _p: Default::default(),
    }
  }

  fn to_coordinates(&self) -> (E::Base, E::Base, bool) {
    self.comm.to_coordinates()
  }

  fn decompress(c: &Self::CompressedCommitment) -> Result<Self, NovaError> {
    let comm = G1::from(c.comm);
    Ok(Self {
      comm,
      _p: Default::default(),
    })
  }
}

impl<E> Default for Commitment<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  fn default() -> Self {
    Commitment {
      comm: G1::zero(),
      _p: Default::default(),
    }
  }
}

impl<G> TranscriptReprTrait<G> for G1Affine
where
  G: Group<Scalar = Fr, Base = Fq>,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let coords = self.coordinates().unwrap();

    [coords.x().to_repr(), coords.y().to_repr()].concat()
  }
}

impl<E> TranscriptReprTrait<E::GE> for Commitment<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let affine = self.comm.to_affine();
    let coords = affine.coordinates().unwrap();

    [coords.x().to_repr(), coords.y().to_repr()].concat()
  }
}

impl<E> AbsorbInROTrait<E> for Commitment<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    let (x, y, is_infinity) = self.comm.to_coordinates();
    ro.absorb(x);
    ro.absorb(y);
    ro.absorb(if is_infinity {
      E::Base::one()
    } else {
      E::Base::zero()
    });
  }
}

impl<E: Engine> TranscriptReprTrait<E::GE> for CompressedCommitment<E> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    let affine = &self.comm;
    let coords = affine.coordinates().unwrap();

    [coords.x().to_repr(), coords.y().to_repr()].concat()
  }
}

impl<E> MulAssign<E::Scalar> for Commitment<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  fn mul_assign(&mut self, scalar: E::Scalar) {
    let result = (self as &Commitment<E>).comm * scalar;
    *self = Commitment {
      comm: result,
      _p: Default::default(),
    };
  }
}

impl<'a, 'b, E> Mul<&'b E::Scalar> for &'a Commitment<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: &'b E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
      _p: Default::default(),
    }
  }
}

impl<E> Mul<E::Scalar> for Commitment<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  type Output = Commitment<E>;

  fn mul(self, scalar: E::Scalar) -> Commitment<E> {
    Commitment {
      comm: self.comm * scalar,
      _p: Default::default(),
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
      _p: Default::default(),
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
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  type Commitment = Commitment<E>;
  type CommitmentKey = CommitmentKey<E>;

  fn setup(_label: &'static [u8], n: usize) -> Self::CommitmentKey {
    // this is for testing purposes
    let tau = Fr::random(OsRng);
    let num_gens = n.next_power_of_two();

    // compute powers of tau
    let mut powers_of_tau: Vec<Fr> = Vec::with_capacity(num_gens);
    powers_of_tau.insert(0, Fr::one());
    for i in 1..num_gens {
      powers_of_tau.insert(i, powers_of_tau[i - 1] * tau);
    }

    let Gi: Vec<G1Affine> = (0..num_gens)
      .into_par_iter()
      .map(|i| (G1Affine::generator() * powers_of_tau[i]).to_affine())
      .collect();

    let H = G2Affine::generator();
    let tauH = (H * tau).to_affine();

    let G = Gi[0];

    Self::CommitmentKey {
      Gi,
      G,
      H,
      tauH,
      _p: Default::default(),
    }
  }

  fn commit(ck: &Self::CommitmentKey, v: &[E::Scalar]) -> Self::Commitment {
    assert!(ck.Gi.len() >= v.len());
    Commitment {
      comm: cpu_best_multiexp(v, &ck.Gi[..v.len()]),
      _p: Default::default(),
    }
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
pub struct VerifierKey<E: Engine> {
  G: G1Affine,
  H: G2Affine,
  tauH: G2Affine,
  _p: PhantomData<E>,
}

/// Provides an implementation of a polynomial evaluation argument
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct EvaluationArgument<E: Engine> {
  com: Vec<G1Affine>,
  w: Vec<G1Affine>,
  v: Vec<Vec<Fr>>,
  _p: PhantomData<E>,
}

/// Provides an implementation of a polynomial evaluation engine using KZG
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvaluationEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E> EvaluationEngine<E>
where
  E: Engine<Scalar = Fr, Base = Fq>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
{
  // This impl block defines helper functions that are not a part of
  // EvaluationEngineTrait, but that we will use to implement the trait
  // functions.
  fn compute_challenge(
    C: &G1Affine,
    y: &Fr,
    com: &[G1Affine],
    transcript: &mut <E as Engine>::TE,
  ) -> Fr {
    transcript.absorb(b"C", C);
    transcript.absorb(b"y", y);
    transcript.absorb(b"c", &com.to_vec().as_slice());

    transcript.squeeze(b"c").unwrap()
  }

  // Compute challenge q = Hash(vk, C0, ..., C_{k-1}, u0, ...., u_{t-1},
  // (f_i(u_j))_{i=0..k-1,j=0..t-1})
  // TODO: Including f_i(u_j) in the challenge may be optional?
  // Both prover and verifier have these values, but if f_i(u_j) can be changed
  // then the soundness of batching has failed.  Still, both parties have the data it seems prudent to hash.
  fn get_batch_challenge(
    C: &[G1Affine],
    u: &[Fr],
    v: &[Vec<Fr>],
    transcript: &mut <E as Engine>::TE,
  ) -> Fr {
    transcript.absorb(b"C", &C.to_vec().as_slice());
    transcript.absorb(b"u", &u.to_vec().as_slice());
    transcript.absorb(
      b"v",
      &v.iter().flatten().cloned().collect::<Vec<Fr>>().as_slice(),
    );

    transcript.squeeze(b"r").unwrap()
  }

  fn batch_challenge_powers(q: Fr, k: usize) -> Vec<Fr> {
    // Compute powers of q : (1, q, q^2, ..., q^(k-1))
    let mut q_powers = vec![Fr::one(); k];
    for i in 1..k {
      q_powers[i] = q_powers[i - 1] * q;
    }
    q_powers
  }

  fn verifier_second_challenge(
    C_B: G1Affine,
    W: &[G1Affine],
    transcript: &mut <E as Engine>::TE,
  ) -> Fr {
    transcript.absorb(b"C_b", &C_B);
    transcript.absorb(b"W", &W.to_vec().as_slice());

    transcript.squeeze(b"d").unwrap()
  }
}

impl<E> EvaluationEngineTrait<E> for EvaluationEngine<E>
where
  E: Engine<Scalar = Fr, Base = Fq, CE = CommitmentEngine<E>>,
  E::GE: DlogGroup<CompressedGroupElement = G1Compressed, PreprocessedGroupElement = G1Affine>,
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
      G: ck.G,
      H: ck.H,
      tauH: ck.tauH,
      _p: Default::default(),
    };

    (pk, vk)
  }

  fn prove(
    ck: &CommitmentKey<E>,
    _pk: &Self::ProverKey,
    transcript: &mut <E as Engine>::TE,
    C: &Commitment<E>,
    hat_P: &[E::Scalar],
    point: &[E::Scalar],
    eval: &E::Scalar,
  ) -> Result<Self::EvaluationArgument, NovaError> {
    let x: Vec<E::Scalar> = point.to_vec();

    //////////////// begin helper closures //////////
    let kzg_open = |f: &[Fr], u: Fr| -> G1Affine {
      //   On input f(x) and u compute the witness polynomial used to prove
      // that f(u) = v.   The main part of this is to compute the
      // division (f(x) - f(u)) / (x - u), but we   don't use a general
      // division algorithm, we make use of the fact that the division
      //   never has a remainder, and that the denominator is always a linear
      // polynomial.   The cost is (d-1) mults + (d-1) adds in Fr, where
      // d is the degree of f.

      //   We use the fact that if we compute the quotient of f(x)/(x-u),
      //   there will be a remainder, but it'll be v = f(u).  Put another way
      // the quotient of   f(x)/(x-u) and (f(x) - f(v))/(x-u) is the
      // same.   One advantage is that computing f(u) could be decoupled
      // from kzg_open,   it could be done later or separate from
      // computing W.

      let compute_witness_polynomial = |f: &[Fr], u: Fr| -> Vec<Fr> {
        let d = f.len();

        // Compute h(x) = f(x)/(x - u)
        let mut h = vec![Fr::zero(); d];
        for i in (1..d).rev() {
          h[i - 1] = f[i] + h[i] * u;
        }

        h
      };

      let h = compute_witness_polynomial(f, u);

      E::CE::commit(ck, &h).comm.to_affine()
    };

    let kzg_open_batch = |C: &Vec<G1Affine>,
                          f: &Vec<Vec<Fr>>,
                          u: &Vec<Fr>,
                          transcript: &mut <E as Engine>::TE|
     -> (Vec<G1Affine>, Vec<Vec<Fr>>) {
      let poly_eval = |f: &[Fr], u: Fr| -> Fr {
        let mut v = f[0];
        let mut u_power = Fr::one();

        for fi in f.iter().skip(1) {
          u_power *= u;
          v += u_power * fi;
        }

        v
      };

      let scalar_vector_muladd = |a: &mut Vec<Fr>, v: &Vec<Fr>, s: Fr| {
        assert!(a.len() >= v.len());
        for i in 0..v.len() {
          a[i] += s * v[i];
        }
      };

      let kzg_compute_batch_polynomial = |f: &Vec<Vec<Fr>>, q: Fr| -> Vec<Fr> {
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
      assert!(C.len() == k);

      // The verifier needs f_i(u_j), so we compute them here (V will compute
      // B(u_j) itself)
      let mut v = vec![vec!(Fr::zero(); k); t];
      for i in 0..t {
        // for each point u
        #[allow(clippy::needless_range_loop)]
        for j in 0..k {
          // for each poly f
          v[i][j] = poly_eval(&f[j], u[i]); // = f_j(u_i)
        }
      }

      let q = Self::get_batch_challenge(C, u, &v, transcript);
      let B = kzg_compute_batch_polynomial(f, q);

      // Now open B at u0, ..., u_{t-1}
      let mut w = Vec::with_capacity(t);
      for ui in u {
        let wi = kzg_open(&B, *ui);
        w.push(wi);
      }

      // Compute the commitment to the batched polynomial B(X)
      let q_powers = Self::batch_challenge_powers(q, k);
      let C_B = (C[0] + cpu_best_multiexp(&q_powers[1..k], &C[1..k])).to_affine();

      // the prover computes the challenge to keep the transcript in the same
      // state as that of the verifier
      let _d_0 = Self::verifier_second_challenge(C_B, &w, transcript);

      (w, v)
    };

    ///// END helper closures //////////

    let ell = x.len();
    let n = hat_P.len();
    assert_eq!(n, 1 << ell); // Below we assume that n is a power of two

    // Phase 1  -- create commitments com_1, ..., com_\ell
    let mut polys: Vec<Vec<Fr>> = Vec::new();
    polys.push(hat_P.to_vec());
    for i in 0..ell {
      let Pi_len = polys[i].len() / 2;
      let mut Pi = vec![Fr::zero(); Pi_len];

      #[allow(clippy::needless_range_loop)]
      for j in 0..Pi_len {
        Pi[j] = x[ell-i-1] * polys[i][2*j + 1]            // Odd part of P^(i-1)
                      + (Fr::one() - x[ell-i-1]) * polys[i][2*j]; // Even part of P^(i-1)
      }

      if i == ell - 1 && *eval != Pi[0] {
        return Err(NovaError::UnSat);
      }

      polys.push(Pi);
    }

    // we do not need to commit to the first polynomial as it is already
    // committed compute commitments in parallel
    let com: Vec<G1Affine> = (1..polys.len())
      .into_par_iter()
      .map(|i| E::CE::commit(ck, &polys[i]).comm.to_affine())
      .collect();

    // Phase 2
    // we do not need to add x to the transcript, because in our context x was
    // obtained from the transcript
    let r = Self::compute_challenge(&C.comm.to_affine(), eval, &com, transcript);
    let u = vec![r, -r, r * r];

    // Phase 3 -- create response
    let mut com_all = com.clone();
    com_all.insert(0, C.comm.to_affine());
    let (w, v) = kzg_open_batch(&com_all, &polys, &u, transcript);

    Ok(EvaluationArgument {
      com,
      w,
      v,
      _p: Default::default(),
    })
  }

  /// A method to verify purported evaluations of a batch of polynomials
  fn verify(
    vk: &Self::VerifierKey,
    transcript: &mut <E as Engine>::TE,
    C: &Commitment<E>,
    point: &[E::Scalar],
    P_of_x: &E::Scalar,
    pi: &Self::EvaluationArgument,
  ) -> Result<(), NovaError> {
    let x = point.to_vec();
    let y = P_of_x;

    // vk is hashed in transcript already, so we do not add it here

    let kzg_verify_batch = |vk: &VerifierKey<E>,
                            C: &Vec<G1Affine>,
                            W: &Vec<G1Affine>,
                            u: &Vec<Fr>,
                            v: &Vec<Vec<Fr>>,
                            transcript: &mut <E as Engine>::TE|
     -> bool {
      let k = C.len();
      let t = u.len();

      let q = Self::get_batch_challenge(C, u, v, transcript);
      let q_powers = Self::batch_challenge_powers(q, k); // 1, q, q^2, ..., q^(k-1)

      // Compute the commitment to the batched polynomial B(X)
      let C_B = (C[0] + cpu_best_multiexp(&q_powers[1..k], &C[1..k])).to_affine();

      // Compute the batched openings
      // compute B(u_i) = v[i][0] + q*v[i][1] + ... + q^(t-1) * v[i][t-1]
      let B_u = (0..t)
        .map(|i| {
          assert_eq!(q_powers.len(), v[i].len());
          q_powers.iter().zip(v[i].iter()).map(|(a, b)| a * b).sum()
        })
        .collect::<Vec<Fr>>();

      let d_0 = Self::verifier_second_challenge(C_B, W, transcript);
      // TODO: (perf) Since we derive d by hashing, can we then have the prover
      // compute & send R? Saves two SMs in verify

      let d = [d_0, d_0 * d_0];

      assert!(t == 3);
      // We write a special case for t=3, since this what is required for
      // mlkzg. Following the paper directly, we must compute:
      // let L0 = C_B - vk.G * B_u[0] + W[0] * u[0];
      // let L1 = C_B - vk.G * B_u[1] + W[1] * u[1];
      // let L2 = C_B - vk.G * B_u[2] + W[2] * u[2];
      // let R0 = -W[0];
      // let R1 = -W[1];
      // let R2 = -W[2];
      // let L = L0 + L1*d[0] + L2*d[1];
      // let R = R0 + R1*d[0] + R2*d[1];
      //
      // We group terms to reduce the number of scalar mults (to seven):
      // In Rust we could use MSMs for these, and speed up verification, but we
      // use individual scalar mults to match what will be implemented in
      // Solidity.
      let L = C_B * (Fr::one() + d[0] + d[1]) - vk.G * (B_u[0] + d[0] * B_u[1] + d[1] * B_u[2])
        + W[0] * u[0]
        + W[1] * (u[1] * d[0])
        + W[2] * (u[2] * d[1]);

      let R0 = -W[0];
      let R1 = -W[1];
      let R2 = -W[2];
      let R = R0 + R1 * d[0] + R2 * d[1];

      let pairing1 = halo2curves::bn256::pairing(&L.to_affine(), &vk.H);
      let pairing2 = halo2curves::bn256::pairing(&R.to_affine(), &vk.tauH);

      pairing1 == -pairing2
    };
    ////// END verify() closure helpers

    let ell = x.len();

    let mut com = pi.com.clone();

    // we do not need to add x to the transcript, because in our context x was
    // obtained from the transcript
    let r = Self::compute_challenge(&C.comm.to_affine(), y, &com, transcript);

    if r.is_zero_vartime() || C.comm.to_affine().is_identity().unwrap_u8() == 1 {
      return Err(NovaError::ProofVerifyError);
    }
    com.insert(0, C.comm.to_affine()); // set com_0 = C, shifts other commitments to the right

    let u = vec![r, -r, r * r];

    // Setup vectors (Y, ypos, yneg) from pi.v
    let v = &pi.v;
    if v.len() != 3 {
      return Err(NovaError::ProofVerifyError);
    }
    if v[0].len() != ell + 1 || v[1].len() != ell + 1 || v[2].len() != ell + 1 {
      return Err(NovaError::ProofVerifyError);
    }
    let ypos = &v[0];
    let yneg = &v[1];
    let Y = &v[2];

    // Check consistency of (Y, ypos, yneg)
    if Y[ell] != *y {
      return Err(NovaError::ProofVerifyError);
    }

    let two = Fr::from(2u64);
    for i in 0..ell {
      if two * r * Y[i + 1]
        != r * (Fr::one() - x[ell - i - 1]) * (ypos[i] + yneg[i])
          + x[ell - i - 1] * (ypos[i] - yneg[i])
      {
        return Err(NovaError::ProofVerifyError);
      }
      // Note that we don't make any checks about Y[0] here, but our batching
      // check below requires it
    }

    // Check commitments to (Y, ypos, yneg) are valid
    if !kzg_verify_batch(vk, &com, &pi.w, &u, &pi.v, transcript) {
      return Err(NovaError::ProofVerifyError);
    }

    Ok(())
  }
}

/// An implementation of Nova traits with multilinear KZG over the BN256 curve
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Bn256EngineKZG;

impl Engine for Bn256EngineKZG {
  type Base = Fq;
  type Scalar = Fr;
  type GE = G1;
  type RO = PoseidonRO<Self::Base, Self::Scalar>;
  type ROCircuit = PoseidonROCircuit<Self::Base>;
  type TE = Keccak256Transcript<Self>;
  type CE = CommitmentEngine<Self>;
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    provider::keccak::Keccak256Transcript, spartan::polys::multilinear::MultilinearPolynomial,
  };
  use bincode::Options;
  use rand::SeedableRng;

  type E = Bn256EngineKZG;

  #[test]
  fn test_mlkzg_eval() {
    // Test with poly(X1, X2) = 1 + X1 + X2 + X1*X2
    let n = 4;
    let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
    let (pk, _vk): (ProverKey<E>, VerifierKey<E>) = EvaluationEngine::setup(&ck);

    // poly is in eval. representation; evaluated at [(0,0), (0,1), (1,0), (1,1)]
    let poly = vec![Fr::from(1), Fr::from(2), Fr::from(2), Fr::from(4)];

    let C = CommitmentEngine::commit(&ck, &poly);
    let mut tr = Keccak256Transcript::new(b"TestEval");

    // Call the prover with a (point, eval) pair. The prover recomputes
    // poly(point) = eval', and fails if eval' != eval
    let point = vec![Fr::from(0), Fr::from(0)];
    let eval = Fr::one();
    assert!(EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(1)];
    let eval = Fr::from(2);
    assert!(EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).is_ok());

    let point = vec![Fr::from(1), Fr::from(1)];
    let eval = Fr::from(4);
    assert!(EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).is_ok());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(3);
    assert!(EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).is_ok());

    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(9);
    assert!(EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).is_ok());

    // Try a couple incorrect evaluations and expect failure
    let point = vec![Fr::from(2), Fr::from(2)];
    let eval = Fr::from(50);
    assert!(EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).is_err());

    let point = vec![Fr::from(0), Fr::from(2)];
    let eval = Fr::from(4);
    assert!(EvaluationEngine::prove(&ck, &pk, &mut tr, &C, &poly, &point, &eval).is_err());
  }

  #[test]
  fn test_mlkzg() {
    let n = 4;

    // poly = [1, 2, 1, 4]
    let poly = vec![Fr::one(), Fr::from(2), Fr::from(1), Fr::from(4)];

    // point = [4,3]
    let point = vec![Fr::from(4), Fr::from(3)];

    // eval = 28
    let eval = Fr::from(28);

    let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
    let (pk, vk) = EvaluationEngine::setup(&ck);

    // make a commitment
    let C = CommitmentEngine::commit(&ck, &poly);

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

    // check if the prover transcript and verifier transcript are kept in the
    // same state
    assert_eq!(post_c_p, post_c_v);

    let my_options = bincode::DefaultOptions::new()
      .with_big_endian()
      .with_fixint_encoding();
    let mut output_bytes = my_options.serialize(&vk).unwrap();
    output_bytes.append(&mut my_options.serialize(&C.compress()).unwrap());
    output_bytes.append(&mut my_options.serialize(&point).unwrap());
    output_bytes.append(&mut my_options.serialize(&eval).unwrap());
    output_bytes.append(&mut my_options.serialize(&proof).unwrap());
    println!("total output = {} bytes", output_bytes.len());

    // Change the proof and expect verification to fail
    let mut bad_proof = proof.clone();
    bad_proof.com[0] = (bad_proof.com[0] + bad_proof.com[1]).to_affine();
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
  fn test_mlkzg_more() {
    // test the mlkzg prover and verifier with random instances (derived from a seed)
    for ell in [4, 5, 6] {
      let mut rng = rand::rngs::StdRng::seed_from_u64(ell as u64);

      let n = 1 << ell; // n = 2^ell

      let poly = (0..n).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let point = (0..ell).map(|_| Fr::random(&mut rng)).collect::<Vec<_>>();
      let eval = MultilinearPolynomial::evaluate_with(&poly, &point);

      let ck: CommitmentKey<E> = CommitmentEngine::setup(b"test", n);
      let (pk, vk) = EvaluationEngine::setup(&ck);

      // make a commitment
      let C = CommitmentEngine::commit(&ck, &poly);

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
      bad_proof.com[0] = (bad_proof.com[0] + bad_proof.com[1]).to_affine();
      let mut verifier_tr2 = Keccak256Transcript::new(b"TestEval");
      assert!(
        EvaluationEngine::verify(&vk, &mut verifier_tr2, &C, &point, &eval, &bad_proof).is_err()
      );
    }
  }
}
