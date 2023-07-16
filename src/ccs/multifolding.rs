use super::cccs::{CCCSInstance, CCCSShape};
use super::lcccs::LCCCS;
use super::util::{compute_sum_Mz, VirtualPolynomial};
use super::{CCSShape, CCSWitness};
use crate::ccs::util::compute_all_sum_Mz_evals;
use crate::hypercube::BooleanHypercube;
use crate::spartan::math::Math;
use crate::spartan::polynomial::MultilinearPolynomial;
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
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};
use std::sync::Arc;

// XXX: THe idea is to have Multifolding as IVC instance in the future, holding the main CCS
// instances. Then the rest of CCS, CCCS, LCCCS hold references to it.
// Is our single source of data.
#[derive(Debug)]
pub struct Multifolding<G: Group> {
  ccs: CCSShape<G>,
  ccs_mle: Vec<MultilinearPolynomial<G::Scalar>>,
}

impl<G: Group> Multifolding<G> {
  /// Generates a new Multifolding instance based on the given CCS.
  pub fn new(ccs: CCSShape<G>) -> Self {
    let ccs_mle = ccs.M.iter().map(|matrix| matrix.to_mle()).collect();
    Self { ccs, ccs_mle }
  }
}

impl<G: Group> Multifolding<G> {
  /// Compute sigma_i and theta_i from step 4
  pub fn compute_sigmas_and_thetas(
    &self,
    z1: &Vec<G::Scalar>,
    z2: &Vec<G::Scalar>,
    r_x_prime: &[G::Scalar],
  ) -> (Vec<G::Scalar>, Vec<G::Scalar>) {
    (
      // sigmas
      compute_all_sum_Mz_evals::<G>(&self.ccs_mle, z1, r_x_prime, self.ccs.s_prime),
      // thetas
      compute_all_sum_Mz_evals::<G>(&self.ccs_mle, z2, r_x_prime, self.ccs.s_prime),
    )
  }

  /// Compute the right-hand-side of step 5 of the multifolding scheme
  pub fn compute_c_from_sigmas_and_thetas(
    &self,
    sigmas: &[G::Scalar],
    thetas: &[G::Scalar],
    gamma: G::Scalar,
    beta: &[G::Scalar],
    r_x: &[G::Scalar],
    r_x_prime: &[G::Scalar],
  ) -> G::Scalar {
    let mut c = G::Scalar::ZERO;

    let e1 = eq_eval(r_x, r_x_prime);
    let e2 = eq_eval(beta, r_x_prime);

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
    running_instance: &LCCCS<G>,
    cccs_instance: &CCCSShape<G>,
    z1: &Vec<G::Scalar>,
    z2: &Vec<G::Scalar>,
    gamma: G::Scalar,
    beta: &[G::Scalar],
  ) -> VirtualPolynomial<G::Scalar> {
    let mut vec_L = running_instance.compute_Ls(z1);
    let mut Q = cccs_instance
      .compute_Q(z2, beta)
      .expect("TQ comp should not fail");
    let mut g = vec_L[0].clone();
    for (j, L_j) in vec_L.iter_mut().enumerate().skip(1) {
      let gamma_j = gamma.pow([j as u64]);
      L_j.scalar_mul(&gamma_j);
      g = g.add(L_j);
    }
    let gamma_t1 = gamma.pow([(cccs_instance.ccs.t + 1) as u64]);
    Q.scalar_mul(&gamma_t1);
    g = g.add(&Q);
    g
  }

  // XXX: This might need to be mutable if we want to hold an LCCCS instance as the IVC inside the
  // NIMFS object.
  pub fn fold(
    &self,
    lcccs1: &LCCCS<G>,
    cccs2: &CCCSInstance<G>,
    sigmas: &[G::Scalar],
    thetas: &[G::Scalar],
    r_x_prime: Vec<G::Scalar>,
    rho: G::Scalar,
  ) -> LCCCS<G> {
    let C = lcccs1.C + cccs2.C.mul(rho);
    let u = lcccs1.u + rho;
    let x: Vec<G::Scalar> = lcccs1
      .x
      .iter()
      .zip(
        cccs2
          .x
          .iter()
          .map(|x_i| *x_i * rho)
          .collect::<Vec<G::Scalar>>(),
      )
      .map(|(a_i, b_i)| *a_i + b_i)
      .collect();
    let v: Vec<G::Scalar> = sigmas
      .iter()
      .zip(
        thetas
          .iter()
          .map(|x_i| *x_i * rho)
          .collect::<Vec<G::Scalar>>(),
      )
      .map(|(a_i, b_i)| *a_i + b_i)
      .collect();

    LCCCS {
      matrix_mles: lcccs1.matrix_mles.clone(),
      C,
      ccs: lcccs1.ccs.clone(),
      u,
      x,
      r_x: r_x_prime,
      v,
    }
  }

  pub fn fold_witness(w1: &CCSWitness<G>, w2: &CCSWitness<G>, rho: G::Scalar) -> CCSWitness<G> {
    let w = w1
      .w
      .iter()
      .zip(
        w2.w
          .iter()
          .map(|x_i| *x_i * rho)
          .collect::<Vec<G::Scalar>>(),
      )
      .map(|(a_i, b_i)| *a_i + b_i)
      .collect();

    // XXX: There's no handling of r_w atm. So we will ingore until all folding is implemented,
    // let r_w = w1.r_w + rho * w2.r_w;
    CCSWitness { w }
  }
}

/// Evaluate eq polynomial.
pub fn eq_eval<F: PrimeField>(x: &[F], y: &[F]) -> F {
  assert_eq!(x.len(), y.len());

  let mut res = F::ONE;
  for (&xi, &yi) in x.iter().zip(y.iter()) {
    let xi_yi = xi * yi;
    res *= xi_yi + xi_yi - xi - yi + F::ONE;
  }
  res
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::ccs::util::virtual_poly::build_eq_x_r;
  use pasta_curves::{Ep, Fq};
  use rand_core::OsRng;
  // NIMFS: Non Interactive Multifolding Scheme
  type NIMFS = Multifolding<Ep>;

  #[test]
  fn test_compute_g() {
    let z1 = CCSShape::<Ep>::get_test_z(3);
    let z2 = CCSShape::<Ep>::get_test_z(4);

    let (_, ccs_witness_1, ccs_instance_1) = CCSShape::gen_test_ccs(&z2);
    let (ccs, ccs_witness_2, ccs_instance_2) = CCSShape::gen_test_ccs(&z1);
    let ck = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let gamma: Fq = Fq::random(&mut rng);
    let beta: Vec<Fq> = (0..ccs.s).map(|_| Fq::random(&mut rng)).collect();

    let (lcccs_instance, _) = ccs.to_lcccs(&mut rng, &ck, &z1);
    let cccs_instance = ccs.to_cccs_shape();

    let mut sum_v_j_gamma = Fq::zero();
    for j in 0..lcccs_instance.v.len() {
      let gamma_j = gamma.pow([j as u64]);
      sum_v_j_gamma += lcccs_instance.v[j] * gamma_j;
    }

    // Compute g(x) with that r_x
    let g = NIMFS::compute_g(&lcccs_instance, &cccs_instance, &z1, &z2, gamma, &beta);

    // evaluate g(x) over x \in {0,1}^s
    let mut g_on_bhc = Fq::zero();
    for x in BooleanHypercube::new(ccs.s) {
      g_on_bhc += g.evaluate(&x).unwrap();
    }

    // evaluate sum_{j \in [t]} (gamma^j * Lj(x)) over x \in {0,1}^s
    let mut sum_Lj_on_bhc = Fq::zero();
    let vec_L = lcccs_instance.compute_Ls(&z1);
    for x in BooleanHypercube::new(ccs.s) {
      for (j, coeff) in vec_L.iter().enumerate() {
        let gamma_j = gamma.pow([j as u64]);
        sum_Lj_on_bhc += coeff.evaluate(&x).unwrap() * gamma_j;
      }
    }

    // Q(x) over bhc is assumed to be zero, as checked in the test 'test_compute_Q'
    assert_ne!(g_on_bhc, Fq::zero());

    // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
    // sum of gamma^j * Lj(x) over the boolean hypercube
    assert_eq!(g_on_bhc, sum_Lj_on_bhc);

    // evaluating g(x) over the boolean hypercube should give the same result as evaluating the
    // sum of gamma^j * v_j over j \in [t]
    assert_eq!(g_on_bhc, sum_v_j_gamma);
  }

  #[test]
  fn test_compute_sigmas_and_thetas() {
    let z1 = CCSShape::<Ep>::get_test_z(3);
    let z2 = CCSShape::<Ep>::get_test_z(4);

    let (_, ccs_witness_1, ccs_instance_1) = CCSShape::gen_test_ccs(&z2);
    let (ccs, ccs_witness_2, ccs_instance_2) = CCSShape::gen_test_ccs(&z1);
    let ck = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let gamma: Fq = Fq::random(&mut rng);
    let beta: Vec<Fq> = (0..ccs.s).map(|_| Fq::random(&mut rng)).collect();
    let r_x_prime: Vec<Fq> = (0..ccs.s).map(|_| Fq::random(&mut rng)).collect();

    // Initialize a multifolding object
    let (lcccs_instance, _) = ccs.to_lcccs(&mut rng, &ck, &z1);
    let (cccs_instance) = ccs.to_cccs_shape();

    // Generate a new multifolding instance
    let nimfs = NIMFS::new(ccs.clone());

    let (sigmas, thetas) = nimfs.compute_sigmas_and_thetas(&z1, &z2, &r_x_prime);

    let g = NIMFS::compute_g(&lcccs_instance, &cccs_instance, &z1, &z2, gamma, &beta);

    // Assert `g` is correctly computed here.
    {
      // evaluate g(x) over x \in {0,1}^s
      let mut g_on_bhc = Fq::zero();
      for x in BooleanHypercube::new(ccs.s) {
        g_on_bhc += g.evaluate(&x).unwrap();
      }
      // evaluate sum_{j \in [t]} (gamma^j * Lj(x)) over x \in {0,1}^s
      let mut sum_Lj_on_bhc = Fq::zero();
      let vec_L = lcccs_instance.compute_Ls(&z1);
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
    let mut revsersed = r_x_prime.clone();
    revsersed.reverse();

    // we expect g(r_x_prime) to be equal to:
    // c = (sum gamma^j * e1 * sigma_j) + gamma^{t+1} * e2 * sum c_i * prod theta_j
    // from `compute_c_from_sigmas_and_thetas`
    let expected_c = g.evaluate(&revsersed).unwrap();

    let c = nimfs.compute_c_from_sigmas_and_thetas(
      &sigmas,
      &thetas,
      gamma,
      &beta,
      &lcccs_instance.r_x,
      &r_x_prime,
    );
    assert_eq!(c, expected_c);
  }

  #[test]
  fn test_lcccs_fold() {
    let z1 = CCSShape::<Ep>::get_test_z(3);
    let z2 = CCSShape::<Ep>::get_test_z(4);

    // ccs stays the same regardless of z1 or z2
    let (ccs, ccs_witness_1, ccs_instance_1) = CCSShape::gen_test_ccs(&z1);
    let (_, ccs_witness_2, ccs_instance_2) = CCSShape::gen_test_ccs(&z2);
    let ck = ccs.commitment_key();

    assert!(ccs.is_sat(&ck, &ccs_instance_1, &ccs_witness_1).is_ok());
    assert!(ccs.is_sat(&ck, &ccs_instance_2, &ccs_witness_2).is_ok());

    let mut rng = OsRng;
    let r_x_prime: Vec<Fq> = (0..ccs.s).map(|_| Fq::random(&mut rng)).collect();

    // Generate a new multifolding instance
    let mut nimfs = NIMFS::new(ccs.clone());

    let (sigmas, thetas) = nimfs.compute_sigmas_and_thetas(&z1, &z2, &r_x_prime);

    // Initialize a multifolding object
    let (lcccs_instance, lcccs_witness) = ccs.to_lcccs(&mut rng, &ck, &z1);

    let (cccs_instance, cccs_witness, cccs_shape) = ccs.to_cccs(&mut rng, &ck, &z2);

    assert!(lcccs_instance.is_sat(&ck, &lcccs_witness).is_ok());

    assert!(cccs_shape
      .is_sat(&ck, &ccs_witness_2, &cccs_instance)
      .is_ok());

    let rho = Fq::random(&mut rng);

    let folded = nimfs.fold(
      &lcccs_instance,
      &cccs_instance,
      &sigmas,
      &thetas,
      r_x_prime,
      rho,
    );

    let w_folded = NIMFS::fold_witness(&lcccs_witness, &cccs_witness, rho);

    // check lcccs relation
    assert!(folded.is_sat(&ck, &w_folded).is_ok());
  }
}
