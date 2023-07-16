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
use super::{CCSShape, CCSWitness};

/// A type that holds the shape of a Committed CCS (CCCS) instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCCSShape<G: Group> {
  // Sequence of sparse MLE polynomials in s+s' variables M_MLE1, ..., M_MLEt
  pub(crate) M_MLE: Vec<MultilinearPolynomial<G::Scalar>>,

  pub(crate) ccs: CCSShape<G>,
}

/// A type that holds a CCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCCSInstance<G: Group> {
  // Commitment to a multilinear polynomial in s' - 1 variables
  pub(crate) C: Commitment<G>,

  // $x in F^l$
  pub(crate) x: Vec<G::Scalar>,
}

impl<G: Group> CCCSShape<G> {
  // Computes q(x) = \sum^q c_i * \prod_{j \in S_i} ( \sum_{y \in {0,1}^s'} M_j(x, y) * z(y) )
  // polynomial over x
  pub fn compute_q(&self, z: &Vec<G::Scalar>) -> Result<VirtualPolynomial<G::Scalar>, NovaError> {
    let z_mle = dense_vec_to_mle::<G::Scalar>(self.ccs.s_prime, z);
    if z_mle.get_num_vars() != self.ccs.s_prime {
      // this check if redundant if dense_vec_to_mle is correct
      return Err(NovaError::VpArith);
    }

    // Using `fold` requires to not have results inside. So we unwrap for now but
    // a better approach is needed (we ca just keep the for loop otherwise.)
    Ok(
      (0..self.ccs.q).fold(VirtualPolynomial::<G::Scalar>::new(self.ccs.s), |q, idx| {
        let mut prod = VirtualPolynomial::<G::Scalar>::new(self.ccs.s);

        for &j in &self.ccs.S[idx] {
          let sum_Mz = compute_sum_Mz::<G>(&self.M_MLE[j], &z_mle);

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
        prod.scalar_mul(&self.ccs.c[idx]);
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
    z: &Vec<G::Scalar>,
    beta: &[G::Scalar],
  ) -> Result<VirtualPolynomial<G::Scalar>, NovaError> {
    let q = self.compute_q(z)?;
    q.build_f_hat(beta)
  }

  /// Perform the check of the CCCS instance described at section 4.1
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<G>,
    w: &CCSWitness<G>,
    x: &CCCSInstance<G>,
  ) -> Result<(), NovaError> {
    // check that C is the commitment of w. Notice that this is not verifying a Pedersen
    // opening, but checking that the Commmitment comes from committing to the witness.
    assert_eq!(x.C, CE::<G>::commit(ck, &w.w));

    // check CCCS relation
    let z: Vec<G::Scalar> = [vec![G::Scalar::ONE], x.x.clone(), w.w.to_vec()].concat();

    // A CCCS relation is satisfied if the q(x) multivariate polynomial evaluates to zero in the hypercube
    let q_x = self.compute_q(&z).unwrap();
    for x in BooleanHypercube::new(self.ccs.s) {
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

  /// Do some sanity checks on q(x). It's a multivariable polynomial and it should evaluate to zero inside the
  /// hypercube, but to not-zero outside the hypercube.
  #[test]
  fn test_compute_q() {
    let mut rng = OsRng;

    let z = CCSShape::<Ep>::get_test_z(3);
    let (ccs_shape, ccs_witness, ccs_instance) = CCSShape::<Ep>::gen_test_ccs(&z);

    // generate ck
    let ck = CCSShape::<Ep>::commitment_key(&ccs_shape);
    // ensure CCS is satisfied
    ccs_shape.is_sat(&ck, &ccs_instance, &ccs_witness).unwrap();

    // Generate CCCS artifacts
    let cccs_shape = ccs_shape.to_cccs_shape();

    let q = cccs_shape.compute_q(&z).unwrap();

    // Evaluate inside the hypercube
    BooleanHypercube::new(ccs_shape.s).for_each(|x| {
      assert_eq!(Fq::zero(), q.evaluate(&x).unwrap());
    });

    // Evaluate outside the hypercube
    let beta: Vec<Fq> = (0..ccs_shape.s).map(|_| Fq::random(&mut rng)).collect();
    assert_ne!(Fq::zero(), q.evaluate(&beta).unwrap());
  }

  #[test]
  fn test_compute_Q() {
    let mut rng = OsRng;

    let z = CCSShape::<Ep>::get_test_z(3);
    let (ccs_shape, ccs_witness, ccs_instance) = CCSShape::<Ep>::gen_test_ccs(&z);

    // generate ck
    let ck = CCSShape::<Ep>::commitment_key(&ccs_shape);
    // ensure CCS is satisfied
    ccs_shape.is_sat(&ck, &ccs_instance, &ccs_witness).unwrap();

    // Generate CCCS artifacts
    let cccs_shape = ccs_shape.to_cccs_shape();

    let beta: Vec<Fq> = (0..ccs_shape.s).map(|_| Fq::random(&mut rng)).collect();

    // Compute Q(x) = eq(beta, x) * q(x).
    let Q = cccs_shape
      .compute_Q(&z, &beta)
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
    let r = BooleanHypercube::new(ccs_shape.s)
      .map(|x| Q.evaluate(&x).unwrap())
      .fold(Fq::zero(), |acc, result| acc + result);
    assert_eq!(r, Fq::zero());
  }

  /// The polynomial G(x) (see above) interpolates q(x) inside the hypercube.
  /// Summing Q(x) over the hypercube is equivalent to evaluating G(x) at some point.
  /// This test makes sure that G(x) agrees with q(x) inside the hypercube, but not outside
  #[test]
  fn test_Q_against_q() {
    let mut rng = OsRng;

    let z = CCSShape::<Ep>::get_test_z(3);
    let (ccs_shape, ccs_witness, ccs_instance) = CCSShape::<Ep>::gen_test_ccs(&z);

    // generate ck
    let ck = CCSShape::<Ep>::commitment_key(&ccs_shape);
    // ensure CCS is satisfied
    ccs_shape.is_sat(&ck, &ccs_instance, &ccs_witness).unwrap();

    // Generate CCCS artifacts
    let cccs_shape = ccs_shape.to_cccs_shape();

    // Now test that if we create Q(x) with eq(d,y) where d is inside the hypercube, \sum Q(x) should be G(d) which
    // should be equal to q(d), since G(x) interpolates q(x) inside the hypercube
    let q = cccs_shape
      .compute_q(&z)
      .expect("Computing q shoud not fail");
    for d in BooleanHypercube::new(ccs_shape.s) {
      let Q_at_d = cccs_shape
        .compute_Q(&z, &d)
        .expect("Computing Q_at_d shouldn't fail");

      // Get G(d) by summing over Q_d(x) over the hypercube
      let G_at_d = BooleanHypercube::new(ccs_shape.s)
        .map(|x| Q_at_d.evaluate(&x).unwrap())
        .fold(Fq::zero(), |acc, result| acc + result);
      assert_eq!(G_at_d, q.evaluate(&d).unwrap());
    }

    // Now test that they should disagree outside of the hypercube
    let r: Vec<Fq> = (0..ccs_shape.s).map(|_| Fq::random(&mut rng)).collect();
    let Q_at_r = cccs_shape
      .compute_Q(&z, &r)
      .expect("Computing Q_at_r shouldn't fail");

    // Get G(d) by summing over Q_d(x) over the hypercube
    let G_at_r = BooleanHypercube::new(ccs_shape.s)
      .map(|x| Q_at_r.evaluate(&x).unwrap())
      .fold(Fq::zero(), |acc, result| acc + result);
    assert_ne!(G_at_r, q.evaluate(&r).unwrap());
  }
}
