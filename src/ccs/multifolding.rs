use super::cccs::{self, CCCS};
use super::lcccs::LCCCS;
use super::util::{compute_sum_Mz, VirtualPolynomial};
use super::{CCSWitness, CCS};
use crate::ccs::util::compute_all_sum_Mz_evals;
use crate::hypercube::BooleanHypercube;
use crate::spartan::math::Math;
use crate::spartan::polynomial::{EqPolynomial, MultilinearPolynomial};
use crate::traits::{TranscriptEngineTrait, TranscriptReprTrait};
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_FOR_RO, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, R1CS},
  traits::{
    commitment::CommitmentEngineTrait, commitment::CommitmentTrait, AbsorbInROTrait, Group, ROTrait,
  },
  utils::*,
  Commitment, CommitmentKey, CE,
};
use bitvec::vec;
use core::{cmp::max, marker::PhantomData};
use ff::{Field, PrimeField};
use flate2::{write::ZlibEncoder, Compression};
use itertools::concat;
use rand_core::RngCore;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};
use std::sync::Arc;

/// The NIMFS (Non-Interactive MultiFolding Scheme) structure is the center of operations of the folding scheme.
/// Once generated, it allows us to fold any upcomming CCCS instances within it without needing to do much.
// XXX: Pending to add doc examples.
#[derive(Debug)]
pub struct NIMFS<G: Group> {
  ccs: CCS<G>,
  ccs_mle: Vec<MultilinearPolynomial<G::Scalar>>,
  ck: CommitmentKey<G>,
  lcccs: LCCCS<G>,
  transcript: G::TE,
}

impl<G: Group> NIMFS<G> {
  /// Generates a new NIMFS instance based on the given CCS instance, it's matrix mle's, an existing LCCCS instance and a commitment key to the CCS.
  pub fn new(
    ccs: CCS<G>,
    ccs_mle: Vec<MultilinearPolynomial<G::Scalar>>,
    lcccs: LCCCS<G>,
    ck: CommitmentKey<G>,
  ) -> Self {
    Self {
      ccs,
      ccs_mle,
      ck,
      lcccs,
      transcript: TranscriptEngineTrait::new(b"NIMFS"),
    }
  }

  /// Initializes a NIMFS instance given the CCS of it and a first witness vector that satifies it.
  // XXX: This should probably return an error as we should check whether is satisfied or not.
  pub fn init(ccs: CCS<G>, z: Vec<G::Scalar>, label: &'static [u8]) -> Self {
    let mut transcript: G::TE = TranscriptEngineTrait::new(label);
    let ccs_mle: Vec<MultilinearPolynomial<G::Scalar>> =
      ccs.M.iter().map(|matrix| matrix.to_mle()).collect();

    // Add the first round of witness to the transcript.
    let w: Vec<G::Scalar> = z[(1 + ccs.l)..].to_vec();
    TranscriptEngineTrait::<G>::absorb(&mut transcript, b"og_w", &w);

    let ck = ccs.commitment_key();
    let w_comm = <G as Group>::CE::commit(&ck, &w);

    // Query challenge to get initial `r_x`.
    let r_x: Vec<G::Scalar> = vec![
      TranscriptEngineTrait::<G>::squeeze(&mut transcript, b"r_x")
        .expect("This should never fail");
      ccs.s
    ];

    // Gen LCCCS initial instance.
    let lcccs: LCCCS<G> = LCCCS::new(&ccs, &ccs_mle, &ck, z, r_x);

    Self {
      ccs,
      ccs_mle,
      lcccs,
      ck,
      transcript,
    }
  }

  /// Generates a new [`CCCS`] instance ready to be folded.
  pub fn new_cccs(&self, z: Vec<G::Scalar>) -> CCCS<G> {
    CCCS::new(&self.ccs, &self.ccs_mle, z, &self.ck)
  }

  /// Generates a new `r_x` vector using the NIMFS challenge query method.
  pub(crate) fn gen_r_x(&mut self) -> Vec<G::Scalar> {
    vec![
      TranscriptEngineTrait::<G>::squeeze(&mut self.transcript, b"r_x")
        .expect("This should never fail");
      self.ccs.s
    ]
  }

  /// This function checks whether the current IVC after the last fold performed is satisfied and returns an error if it isn't.
  pub fn is_sat(&self) -> Result<(), NovaError> {
    self.lcccs.is_sat(&self.ccs, &self.ccs_mle, &self.ck)
  }

  /// Compute sigma_i and theta_i from step 4.
  pub fn compute_sigmas_and_thetas(
    &self,
    // Note `z2` represents the input of the incoming CCCS instance.
    // As the current IVC accumulated input is holded inside of the NIMFS(`self`).
    z: &[G::Scalar],
    r_x_prime: &[G::Scalar],
  ) -> (Vec<G::Scalar>, Vec<G::Scalar>) {
    (
      // sigmas
      compute_all_sum_Mz_evals::<G>(
        &self.ccs_mle,
        self.lcccs.z.as_slice(),
        r_x_prime,
        self.ccs.s_prime,
      ),
      // thetas
      compute_all_sum_Mz_evals::<G>(&self.ccs_mle, z, r_x_prime, self.ccs.s_prime),
    )
  }

  /// Compute the right-hand-side of step 5 of the NIMFS scheme
  pub fn compute_c_from_sigmas_and_thetas(
    &self,
    sigmas: &[G::Scalar],
    thetas: &[G::Scalar],
    gamma: G::Scalar,
    beta: &[G::Scalar],
    r_x_prime: &[G::Scalar],
  ) -> G::Scalar {
    let mut c = G::Scalar::ZERO;

    let e1 = EqPolynomial::new(self.lcccs.r_x.to_vec()).evaluate(r_x_prime);
    let e2 = EqPolynomial::new(beta.to_vec()).evaluate(r_x_prime);

    // (sum gamma^j * e1 * sigma_j)
    for (j, sigma_j) in sigmas.iter().enumerate() {
      let gamma_j = gamma.pow([j as u64]);
      c += gamma_j * e1 * sigma_j;
    }

    // + gamma^{t+1} * e2 * sum c_i * prod theta_j
    let mut lhs = G::Scalar::ZERO;
    for i in 0..self.ccs.q {
      let mut prod = G::Scalar::ONE;
      for j in self.ccs.S[i].clone() {
        prod *= thetas[j];
      }
      lhs += self.ccs.c[i] * prod;
    }
    let gamma_t1 = gamma.pow([(self.ccs.t + 1) as u64]);
    c += gamma_t1 * e2 * lhs;
    c
  }

  /// Compute g(x) polynomial for the given inputs.
  pub fn compute_g(
    &self,
    cccs: &CCCS<G>,
    gamma: G::Scalar,
    beta: &[G::Scalar],
  ) -> VirtualPolynomial<G::Scalar> {
    let mut vec_L = self.lcccs.compute_Ls(&self.ccs, &self.ccs_mle);

    let mut Q = cccs
      .compute_Q(&self.ccs, &self.ccs_mle, beta)
      .expect("Q comp should not fail");

    let mut g = vec_L[0].clone();

    for (j, L_j) in vec_L.iter_mut().enumerate().skip(1) {
      let gamma_j = gamma.pow([j as u64]);
      L_j.scalar_mul(&gamma_j);
      g = g.add(L_j);
    }

    let gamma_t1 = gamma.pow([(self.ccs.t + 1) as u64]);
    Q.scalar_mul(&gamma_t1);
    g = g.add(&Q);
    g
  }

  /// This folds an upcoming CCCS instance into the running LCCCS instance contained within the NIMFS object.
  pub fn fold(&mut self, cccs: CCCS<G>) {
    // Compute r_x_prime and rho from challenging the transcript.
    let r_x_prime = self.gen_r_x();
    // Challenge the transcript once more to obtain `rho`
    let rho = TranscriptEngineTrait::<G>::squeeze(&mut self.transcript, b"rho")
      .expect("This should not fail");

    // Compute sigmas an thetas to fold `v`s.
    let (sigmas, thetas) = self.compute_sigmas_and_thetas(&cccs.z, &r_x_prime);

    // Compute new v from sigmas and thetas.
    let folded_v: Vec<G::Scalar> = sigmas
      .iter()
      .zip(
        thetas
          .iter()
          .map(|x_i| *x_i * rho)
          .collect::<Vec<G::Scalar>>(),
      )
      .map(|(a_i, b_i)| *a_i + b_i)
      .collect();

    // Here we perform steps 7 & 8 of the section 5 of the paper. Were we actually fold LCCCS & CCCS instances.
    self.lcccs.w_comm += cccs.w_comm.mul(rho);
    self.lcccs.v = folded_v;
    self.lcccs.r_x = r_x_prime;
    self.fold_z(cccs, rho);
  }

  /// Folds the current `z` vector of the upcomming CCCS instance together with the LCCCS instance that is contained inside of the NIMFS object.
  fn fold_z(&mut self, cccs: CCCS<G>, rho: G::Scalar) {
    // Update u first.
    self.lcccs.z[0] += rho;
    self.lcccs.z[1..]
      .iter_mut()
      .zip(cccs.z[1..].iter().map(|x_i| *x_i * rho))
      .for_each(|(a_i, b_i)| *a_i += b_i);

    // XXX: There's no handling of r_w atm. So we will ingore until all folding is implemented,
    // let r_w = w1.r_w + rho * w2.r_w;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ccs::test;
  use pasta_curves::{Ep, Fq};
  use rand_core::OsRng;

  fn test_compute_g_with<G: Group>() {
    let z1 = CCS::<G>::get_test_z(3);
    let z2 = CCS::<G>::get_test_z(4);

    let (_, ccs_witness_1, ccs_instance_1, mles) = CCS::<G>::gen_test_ccs(&z2);
    let (ccs, ccs_witness_2, ccs_instance_2, _) = CCS::<G>::gen_test_ccs(&z1);
    let ck = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let gamma: G::Scalar = G::Scalar::random(&mut rng);
    let beta: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();
    let r_x: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut OsRng)).collect();

    let lcccs = LCCCS::new(&ccs, &mles, &ck, z1, r_x);
    let cccs = CCCS::new(&ccs, &mles, z2, &ck);

    let mut sum_v_j_gamma = G::Scalar::ZERO;
    for j in 0..lcccs.v.len() {
      let gamma_j = gamma.pow([j as u64]);
      sum_v_j_gamma += lcccs.v[j] * gamma_j;
    }

    let nimfs = NIMFS::<G>::new(ccs.clone(), mles.clone(), lcccs.clone(), ck.clone());

    // Compute g(x) with that r_x
    let g = nimfs.compute_g(&cccs, gamma, &beta);

    // evaluate g(x) over x \in {0,1}^s
    let mut g_on_bhc = G::Scalar::ZERO;
    for x in BooleanHypercube::new(ccs.s) {
      g_on_bhc += g.evaluate(&x).unwrap();
    }

    // evaluate sum_{j \in [t]} (gamma^j * Lj(x)) over x \in {0,1}^s
    let mut sum_Lj_on_bhc = G::Scalar::ZERO;
    let vec_L = lcccs.compute_Ls(&ccs, &mles);
    for x in BooleanHypercube::new(ccs.s) {
      for (j, coeff) in vec_L.iter().enumerate() {
        let gamma_j = gamma.pow([j as u64]);
        sum_Lj_on_bhc += coeff.evaluate(&x).unwrap() * gamma_j;
      }
    }

    // Q(x) over bhc is assumed to be zero, as checked in the test 'test_compute_Q'
    assert_ne!(g_on_bhc, G::Scalar::ZERO);

    // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
    // sum of gamma^j * Lj(x) over the boolean hypercube
    assert_eq!(g_on_bhc, sum_Lj_on_bhc);

    // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
    // sum of gamma^j * v_j over j \in [t]
    assert_eq!(g_on_bhc, sum_v_j_gamma);
  }

  fn test_compute_sigmas_and_thetas_with<G: Group>() {
    let z1 = CCS::<G>::get_test_z(3);
    let z2 = CCS::<G>::get_test_z(4);

    let (_, ccs_witness_1, ccs_instance_1, mles) = CCS::<G>::gen_test_ccs(&z2);
    let (ccs, ccs_witness_2, ccs_instance_2, _) = CCS::<G>::gen_test_ccs(&z1);
    let ck: CommitmentKey<G> = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let gamma: G::Scalar = G::Scalar::random(&mut rng);
    let beta: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();
    let r_x: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut OsRng)).collect();

    let lcccs = LCCCS::new(&ccs, &mles, &ck, z1, r_x.clone());
    let cccs = CCCS::new(&ccs, &mles, z2, &ck);

    // Generate a new NIMFS instance
    let nimfs = NIMFS::<G>::new(ccs.clone(), mles.clone(), lcccs, ck.clone());

    let (sigmas, thetas) = nimfs.compute_sigmas_and_thetas(&cccs.z, r_x.as_slice());

    let g = nimfs.compute_g(&cccs, gamma, &beta);
    // Assert `g` is correctly computed here.
    {
      // evaluate g(x) over x \in {0,1}^s
      let mut g_on_bhc = G::Scalar::ZERO;
      for x in BooleanHypercube::new(ccs.s) {
        g_on_bhc += g.evaluate(&x).unwrap();
      }
      // evaluate sum_{j \in [t]} (gamma^j * Lj(x)) over x \in {0,1}^s
      let mut sum_Lj_on_bhc = G::Scalar::ZERO;
      let vec_L = nimfs.lcccs.compute_Ls(&ccs, &mles);
      for x in BooleanHypercube::new(ccs.s) {
        for (j, coeff) in vec_L.iter().enumerate() {
          let gamma_j = gamma.pow([j as u64]);
          sum_Lj_on_bhc += coeff.evaluate(&x).unwrap() * gamma_j;
        }
      }

      // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
      // sum of gamma^j * Lj(x) over the boolean hypercube
      assert_eq!(g_on_bhc, sum_Lj_on_bhc);
    };

    // XXX: We need a better way to do this. Sum_Mz has also the same issue.
    // reverse the `r` given to evaluate to match Spartan/Nova endianness.
    let mut reversed = r_x.clone();
    reversed.reverse();

    // we expect g(r_x_prime) to be equal to:
    // c = (sum gamma^j * e1 * sigma_j) + gamma^{t+1} * e2 * sum c_i * prod theta_j
    // from `compute_c_from_sigmas_and_thetas`
    let expected_c = g.evaluate(&reversed).unwrap();

    let c = nimfs.compute_c_from_sigmas_and_thetas(&sigmas, &thetas, gamma, &beta, &r_x);
    assert_eq!(c, expected_c);
  }

  fn test_compute_g() {
    test_compute_g_with::<Ep>();
  }

  fn test_lccs_fold_with<G: Group>() {
    let z1 = CCS::<G>::get_test_z(3);
    let z2 = CCS::<G>::get_test_z(4);

    // ccs stays the same regardless of z1 or z2
    let (ccs, ccs_witness_1, ccs_instance_1, mles) = CCS::<G>::gen_test_ccs(&z1);
    let (_, ccs_witness_2, ccs_instance_2, _) = CCS::gen_test_ccs(&z2);
    let ck: CommitmentKey<G> = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    // Generate a new NIMFS instance
    let mut nimfs = NIMFS::init(ccs.clone(), z1, b"test_NIMFS");
    assert!(nimfs.is_sat().is_ok());

    // check folding correct stuff still alows the NIMFS to be satisfied correctly.
    let cccs = nimfs.new_cccs(z2);
    assert!(cccs.is_sat(&ccs, &mles, &ck).is_ok());
    nimfs.fold(cccs);
    assert!(nimfs.is_sat().is_ok());

    // // Folding garbage should cause a failure
    // let cccs = nimfs.new_cccs(vec![Fq::ONE, Fq::ONE, Fq::ONE]);
    // nimfs.fold(&mut rng, cccs);
    // assert!(nimfs.is_sat().is_err());
    // XXX: Should this indeed pass as it does now?
  }

  #[test]
  fn test_compute_sigmas_and_thetas() {
    test_compute_sigmas_and_thetas_with::<Ep>()
  }

  #[test]
  fn test_lcccs_fold() {
    test_lccs_fold_with::<Ep>()
  }
}
