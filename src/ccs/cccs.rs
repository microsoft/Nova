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

use super::util::compute_sum_Mz;
use super::util::virtual_poly::VirtualPolynomial;
use super::CCS;

/// A type that holds the shape of a Committed CCS (CCCS) instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCCS<G: Group> {
  // The `z` vector used as input for this instance.
  pub(crate) z: Vec<G::Scalar>,
  // Commitment to the witness of `z`.
  pub(crate) w_comm: Commitment<G>,
}

impl<G: Group> CCCS<G> {
  /// Generates a new CCCS given a reference to it's original CCS repr and it's public and private inputs.
  pub(crate) fn new(
    ccs: &CCS<G>,
    ccs_matrix_mle: &[MultilinearPolynomial<G::Scalar>],
    z: Vec<G::Scalar>,
    ck: &CommitmentKey<G>,
  ) -> Self {
    let w_comm = CE::<G>::commit(ck, &z[(1 + ccs.l)..]);

    Self {
      z: z.to_vec(),
      w_comm,
    }
  }

  /// Computes q(x) = \sum^q c_i * \prod_{j \in S_i} ( \sum_{y \in {0,1}^s'} M_j(x, y) * z(y) )
  /// polynomial over x
  pub(crate) fn compute_q(
    &self,
    ccs: &CCS<G>,
    ccs_mles: &[MultilinearPolynomial<G::Scalar>],
  ) -> Result<VirtualPolynomial<G::Scalar>, NovaError> {
    let z_mle = dense_vec_to_mle::<G::Scalar>(ccs.s_prime, &self.z);
    if z_mle.get_num_vars() != ccs.s_prime {
      // this check if redundant if dense_vec_to_mle is correct
      return Err(NovaError::VpArith);
    }

    // Using `fold` requires to not have results inside. So we unwrap for now but
    // a better approach is needed (we ca just keep the for loop otherwise.)
    Ok(
      (0..ccs.q).fold(VirtualPolynomial::<G::Scalar>::new(ccs.s), |q, idx| {
        let mut prod = VirtualPolynomial::<G::Scalar>::new(ccs.s);

        for &j in &ccs.S[idx] {
          let sum_Mz = compute_sum_Mz::<G>(&ccs_mles[j], &z_mle);

          // Fold this sum into the running product
          if prod.products.is_empty() {
            // If this is the first time we are adding something to this virtual polynomial, we need to
            // explicitly add the products using add_mle_list()
            // XXX is this true? improve API
            prod
              .add_mle_list([Arc::new(sum_Mz)], G::Scalar::ONE)
              .unwrap();
          } else {
            prod.mul_by_mle(Arc::new(sum_Mz), G::Scalar::ONE).unwrap();
          }
        }
        // Multiply by the product by the coefficient c_i
        prod.scalar_mul(&ccs.c[idx]);
        // Add it to the running sum
        q.add(&prod)
      }),
    )
  }

  /// Computes Q(x) = eq(beta, x) * q(x)
  ///               = eq(beta, x) * \sum^q c_i * \prod_{j \in S_i} ( \sum_{y \in {0,1}^s'} M_j(x, y) * z(y) )
  /// polynomial over x
  pub fn compute_Q(
    &self,
    ccs: &CCS<G>,
    ccs_mles: &[MultilinearPolynomial<G::Scalar>],
    beta: &[G::Scalar],
  ) -> Result<VirtualPolynomial<G::Scalar>, NovaError> {
    let q = self.compute_q(ccs, ccs_mles)?;
    q.build_f_hat(beta)
  }

  /// Perform the check of the CCCS instance described at section 4.1
  pub fn is_sat(
    &self,
    ccs: &CCS<G>,
    ccs_mles: &[MultilinearPolynomial<G::Scalar>],
    ck: &CommitmentKey<G>,
  ) -> Result<(), NovaError> {
    // check that C is the commitment of w. Notice that this is not verifying a Pedersen
    // opening, but checking that the Commmitment comes from committing to the witness.
    assert_eq!(self.w_comm, CE::<G>::commit(ck, &self.z[(1 + ccs.l)..]));

    // A CCCS relation is satisfied if the q(x) multivariate polynomial evaluates to zero in the hypercube
    let q_x = self.compute_q(ccs, ccs_mles).unwrap();
    for x in BooleanHypercube::new(ccs.s) {
      if !q_x.evaluate(&x).unwrap().is_zero().unwrap_u8() == 0 {
        return Err(NovaError::UnSat);
      }
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {

  use crate::ccs::CCSInstance;
  use crate::ccs::CCSWitness;

  use super::*;
  use ff::PrimeField;
  use pasta_curves::pallas::Scalar;
  use pasta_curves::Ep;
  use pasta_curves::Fp;
  use pasta_curves::Fq;
  use rand_core::OsRng;
  use rand_core::RngCore;

  // Deduplicate this
  fn to_F_matrix<F: PrimeField>(m: Vec<Vec<u64>>) -> Vec<Vec<F>> {
    m.iter().map(|x| to_F_vec(x.clone())).collect()
  }

  // Deduplicate this
  fn to_F_vec<F: PrimeField>(v: Vec<u64>) -> Vec<F> {
    v.iter().map(|x| F::from(*x)).collect()
  }

  fn vecs_to_slices<T>(vecs: &[Vec<T>]) -> Vec<&[T]> {
    vecs.iter().map(Vec::as_slice).collect()
  }

  fn test_compute_q_with<G: Group>() {
    let mut rng = OsRng;

    let z = CCS::<G>::get_test_z(3);
    let (ccs, ccs_witness, ccs_instance, mles) = CCS::<G>::gen_test_ccs(&z);

    // generate ck
    let ck = CCS::<G>::commitment_key(&ccs);
    // ensure CCS is satisfied
    ccs.is_sat(&ck, &ccs_instance, &ccs_witness).unwrap();

    // Generate CCCS artifacts
    let cccs = CCCS::new(&ccs, &mles, z, &ck);
    let q = cccs.compute_q(&ccs, &mles).unwrap();

    // Evaluate inside the hypercube
    BooleanHypercube::new(ccs.s).for_each(|x| {
      assert_eq!(G::Scalar::ZERO, q.evaluate(&x).unwrap());
    });

    // Evaluate outside the hypercube
    let beta: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();
    assert_ne!(G::Scalar::ZERO, q.evaluate(&beta).unwrap());
  }

  fn test_compute_Q_with<G: Group>() {
    let mut rng = OsRng;

    let z = CCS::<G>::get_test_z(3);
    let (ccs, ccs_witness, ccs_instance, mles) = CCS::<G>::gen_test_ccs(&z);

    // generate ck
    let ck = CCS::<G>::commitment_key(&ccs);
    // ensure CCS is satisfied
    ccs.is_sat(&ck, &ccs_instance, &ccs_witness).unwrap();

    // Generate CCCS artifacts
    let cccs = CCCS::new(&ccs, &mles, z, &ck);
    let beta: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();
    // Compute Q(x) = eq(beta, x) * q(x).
    let Q = cccs
      .compute_Q(&ccs, &mles, &beta)
      .expect("Computation of Q should not fail");

    // Let's consider the multilinear polynomial G(x) = \sum_{y \in {0, 1}^s} eq(x, y) q(y)
    // which interpolates the multivariate polynomial q(x) inside the hypercube.
    //
    // Observe that summing Q(x) inside the hypercube, directly computes G(\beta).
    //
    // Now, G(x) is multilinear and agrees with q(x) inside the hypercube. Since q(x) vanishes inside the
    // hypercube, this means that G(x) also vanishes in the hypercube. Since G(x) is multilinear and vanishes
    // inside the hypercube, this makes it the zero polynomial.
    //
    // Hence, evaluating G(x) at a random beta should give zero.

    // Now sum Q(x) evaluations in the hypercube and expect it to be 0
    let r = BooleanHypercube::new(ccs.s)
      .map(|x| Q.evaluate(&x).unwrap())
      .fold(G::Scalar::ZERO, |acc, result| acc + result);
    assert_eq!(r, G::Scalar::ZERO);
  }

  fn test_Q_against_q_with<G: Group>() {
    let mut rng = OsRng;

    let z = CCS::<G>::get_test_z(3);
    let (ccs, ccs_witness, ccs_instance, mles) = CCS::<G>::gen_test_ccs(&z);

    // generate ck
    let ck = CCS::<G>::commitment_key(&ccs);
    // ensure CCS is satisfied
    ccs.is_sat(&ck, &ccs_instance, &ccs_witness).unwrap();

    // Generate CCCS artifacts
    let cccs = CCCS::new(&ccs, &mles, z, &ck);
    // Now test that if we create Q(x) with eq(d,y) where d is inside the hypercube, \sum Q(x) should be G(d) which
    // should be equal to q(d), since G(x) interpolates q(x) inside the hypercube
    let q = cccs
      .compute_q(&ccs, &mles)
      .expect("Computing q shoud not fail");

    for d in BooleanHypercube::new(ccs.s) {
      let Q_at_d = cccs
        .compute_Q(&ccs, &mles, &d)
        .expect("Computing Q_at_d shouldn't fail");

      // Get G(d) by summing over Q_d(x) over the hypercube
      let G_at_d = BooleanHypercube::new(ccs.s)
        .map(|x| Q_at_d.evaluate(&x).unwrap())
        .fold(G::Scalar::ZERO, |acc, result| acc + result);
      assert_eq!(G_at_d, q.evaluate(&d).unwrap());
    }

    // Now test that they should disagree outside of the hypercube
    let r: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();
    let Q_at_r = cccs
      .compute_Q(&ccs, &mles, &r)
      .expect("Computing Q_at_r shouldn't fail");

    // Get G(d) by summing over Q_d(x) over the hypercube
    let G_at_r = BooleanHypercube::new(ccs.s)
      .map(|x| Q_at_r.evaluate(&x).unwrap())
      .fold(G::Scalar::ZERO, |acc, result| acc + result);
    assert_ne!(G_at_r, q.evaluate(&r).unwrap());
  }

  /// Do some sanity checks on q(x). It's a multivariable polynomial and it should evaluate to zero inside the
  /// hypercube, but to not-zero outside the hypercube.
  #[test]
  fn test_compute_q() {
    test_compute_q_with::<Ep>();
  }

  #[test]
  fn test_compute_Q() {
    test_compute_Q_with::<Ep>();
  }

  /// The polynomial G(x) (see above) interpolates q(x) inside the hypercube.
  /// Summing Q(x) over the hypercube is equivalent to evaluating G(x) at some point.
  /// This test makes sure that G(x) agrees with q(x) inside the hypercube, but not outside
  #[test]
  fn test_Q_against_q() {
    test_Q_against_q_with::<Ep>();
  }
}
