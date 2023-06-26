use crate::hypercube::BooleanHypercube;
use crate::spartan::math::Math;
use crate::spartan::polynomial::MultilinearPolynomial;
use crate::utils::bit_decompose;
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

use super::virtual_poly::VirtualPolynomial;
use super::CCSShape;

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

// NOTE: We deal with `r` parameter later in `nimfs.rs` when running `execute_sequence` with `ro_consts`
/// A type that holds a witness for a given CCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CCCSWitness<G: Group> {
  // Multilinear polynomial w_mle in s' - 1 variables
  pub(crate) w_mle: Vec<G::Scalar>,
}

impl<G: Group> CCCSShape<G> {
  /// Computes the MLE of the CCS's Matrix at index `j` and executes the reduction of it summing over the given z.
  pub fn compute_sum_Mz(
    &self,
    m_idx: usize,
    z: &MultilinearPolynomial<G::Scalar>,
  ) -> MultilinearPolynomial<G::Scalar> {
    let s_prime = self.ccs.s_prime;
    let M_j_mle = self.ccs.M[m_idx].to_mle();
    assert_eq!(z.get_num_vars(), s_prime);
    //dbg!(M_j_mle.clone());
    let mut sum_Mz = MultilinearPolynomial::new(vec![
      G::Scalar::ZERO;
      1 << (M_j_mle.get_num_vars() - s_prime)
    ]);

    let bhc = BooleanHypercube::<G::Scalar>::new(s_prime);
    bhc.into_iter().for_each(|bit_vec| {
      // Perform the reduction
      dbg!(bit_vec.clone());
      let mut M_j_y: MultilinearPolynomial<<G as Group>::Scalar> = M_j_mle.clone();

      for bit in bit_vec.iter() {
        M_j_y.bound_poly_var_top(bit);
        dbg!(M_j_y.clone());
      }

      let z_y = z.evaluate(&bit_vec);
      // dbg!(z_y.clone());
      let M_j_z = M_j_y.scalar_mul(&z_y);
      // dbg!(M_j_z.clone());
      // XXX: It's crazy to have results in the ops impls. Remove them!
      sum_Mz = sum_Mz.clone().add(M_j_z).expect("This should not fail");
      // dbg!(sum_Mz.clone());
    });

    sum_Mz
  }

  // Computes q(x) = \sum^q c_i * \prod_{j \in S_i} ( \sum_{y \in {0,1}^s'} M_j(x, y) * z(y) )
  // polynomial over x
  pub fn compute_q(&self, z: &Vec<G::Scalar>) -> Result<VirtualPolynomial<G::Scalar>, NovaError> {
    let z_mle = dense_vec_to_mle::<G::Scalar>(6, z);
    if z_mle.get_num_vars() != self.ccs.s_prime {
      return Err(NovaError::VpArith);
    }

    // Using `fold` requires to not have results inside. So we unwrap for now but
    // a better approach is needed (we ca just keep the for loop otherwise.)
    Ok((0..self.ccs.q).into_iter().fold(
      VirtualPolynomial::<G::Scalar>::new(self.ccs.s),
      |q, idx| {
        let mut prod = VirtualPolynomial::<G::Scalar>::new(self.ccs.s);

        for &j in &self.ccs.S[idx] {
          let sum_Mz = self.compute_sum_Mz(j, &z_mle);

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
      },
    ))
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
}

pub fn fix_variables<F: PrimeField>(
  poly: &MultilinearPolynomial<F>,
  partial_point: &[F],
) -> MultilinearPolynomial<F> {
  assert!(
    partial_point.len() <= poly.get_num_vars(),
    "invalid size of partial point"
  );
  let nv = poly.get_num_vars();
  let mut poly = poly.Z.to_vec();
  let dim = partial_point.len();
  // evaluate single variable of partial point from left to right
  for (i, point) in partial_point.iter().enumerate() {
    poly = fix_one_variable_helper(&poly, nv - i, point);
  }

  MultilinearPolynomial::<F>::new(poly[..(1 << (nv - dim))].to_vec())
}

fn fix_one_variable_helper<F: PrimeField>(data: &[F], nv: usize, point: &F) -> Vec<F> {
  let mut res = vec![F::ZERO; 1 << (nv - 1)];

  for i in 0..(1 << (nv - 1)) {
    res[i] = data[i << 1] + (data[(i << 1) + 1] - data[i << 1]) * point;
  }

  res
}

#[cfg(test)]
mod tests {

  use crate::ccs::CCSInstance;
  use crate::ccs::CCSWitness;
  use crate::ccs::CCS;

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

  fn gen_test_ccs<G: Group, R: RngCore>(
    z: &Vec<G::Scalar>,
    rng: &mut R,
  ) -> (CCSShape<G>, CCSWitness<G>, CCSInstance<G>) {
    let one = G::Scalar::ONE;
    let A = vec![
      (0, 1, one),
      (1, 3, one),
      (2, 1, one),
      (2, 4, one),
      (3, 0, G::Scalar::from(5u64)),
      (3, 5, one),
    ];

    let B = vec![(0, 1, one), (1, 1, one), (2, 0, one), (3, 0, one)];
    let C = vec![(0, 3, one), (1, 4, one), (2, 5, one), (3, 2, one)];

    // 2. Take R1CS and convert to CCS
    let ccs = CCSShape::from_r1cs(R1CSShape::new(4, 6, 1, &A, &B, &C).unwrap());
    // Generate other artifacts
    let ck = CCSShape::<G>::commitment_key(&ccs);
    let ccs_w = CCSWitness::new(z[2..].to_vec());
    let ccs_instance = CCSInstance::new(&ccs, &ccs_w.commit(&ck), vec![z[1]]).unwrap();

    ccs
      .is_sat(&ck, &ccs_instance, &ccs_w)
      .expect("This does not fail");
    (ccs, ccs_w, ccs_instance)
  }

  /// Computes the z vector for the given input for Vitalik's equation.
  pub fn get_test_z<G: Group>(input: u64) -> Vec<G::Scalar> {
    // z = (1, io, w)
    to_F_vec(vec![
      1,
      input,
      input * input * input + input + 5, // x^3 + x + 5
      input * input,                     // x^2
      input * input * input,             // x^2 * x
      input * input * input + input,     // x^3 + x
    ])
  }

  #[test]
  fn test_compute_sum_Mz_over_boolean_hypercube() -> () {
    let z = get_test_z::<Ep>(3);
    let (ccs, _, _) = gen_test_ccs::<Ep, _>(&z, &mut OsRng);

    // Generate other artifacts
    let ck = CCSShape::<Ep>::commitment_key(&ccs);
    let (_, _, cccs) = ccs.to_cccs_artifacts(&mut OsRng, &ck, &z);

    let z_mle = dense_vec_to_mle(ccs.s_prime, &z);
    // dbg!(z_mle.clone());

    // check that evaluating over all the values x over the boolean hypercube, the result of
    // the next for loop is equal to 0
    let mut r = Fq::zero();
    let bch = BooleanHypercube::new(ccs.s);
    bch.into_iter().for_each(|x| {
      // dbg!(x.clone());
      for i in 0..ccs.q {
        let mut Sj_prod = Fq::one();
        for j in ccs.S[i].clone() {
          let sum_Mz = cccs.compute_sum_Mz(j, &z_mle);
          dbg!(sum_Mz.clone());
          let sum_Mz_x = sum_Mz.evaluate(&x);
          dbg!(sum_Mz_x.clone());
          Sj_prod *= sum_Mz_x;
          dbg!(Sj_prod.clone());
        }
        r += (Sj_prod * ccs.c[i]);
      }
      // dbg!(r.clone());
      assert_eq!(r, Fq::ZERO);
    });
  }

  /// Do some sanity checks on q(x). It's a multivariable polynomial and it should evaluate to zero inside the
  /// hypercube, but to not-zero outside the hypercube.
  #[test]
  fn test_compute_q() {
    let mut rng = OsRng;

    let z = get_test_z::<Ep>(3);
    let (ccs_shape, ccs_witness, ccs_instance) = gen_test_ccs(&z, &mut rng);

    // generate ck
    let ck = CCSShape::<Ep>::commitment_key(&ccs_shape);
    // ensure CCS is satisfied
    ccs_shape.is_sat(&ck, &ccs_instance, &ccs_witness).unwrap();

    // Generate CCCS artifacts
    let cccs_shape = ccs_shape.to_cccs_shape();

    let q = cccs_shape.compute_q(&z).unwrap();

    // Evaluate inside the hypercube
    for x in BooleanHypercube::new(ccs_shape.s).into_iter() {
      assert_eq!(Fq::zero(), q.evaluate(&x).unwrap());
    }

    // Evaluate outside the hypercube
    let beta: Vec<Fq> = (0..ccs_shape.s).map(|_| Fq::random(&mut rng)).collect();
    assert_ne!(Fq::zero(), q.evaluate(&beta).unwrap());
  }

  #[test]
  fn test_compute_Q() {
    let mut rng = OsRng;

    let z = get_test_z::<Ep>(3);
    let (ccs_shape, ccs_witness, ccs_instance) = gen_test_ccs(&z, &mut rng);

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
      .into_iter()
      .map(|x| Q.evaluate(&x).unwrap())
      .fold(Fq::zero(), |acc, result| acc + result);
    assert_eq!(r, Fq::zero());
  }

  /// The polynomial G(x) (see above) interpolates q(x) inside the hypercube.
  /// Summing Q(x) over the hypercube is equivalent to evaluating G(x) at some point.
  /// This test makes sure that G(x) agrees with q(x) inside the hypercube, but not outside
  #[test]
  fn test_Q_against_q() -> () {
    let mut rng = OsRng;

    let z = get_test_z::<Ep>(3);
    let (ccs_shape, ccs_witness, ccs_instance) = gen_test_ccs(&z, &mut rng);

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
        .into_iter()
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
      .into_iter()
      .map(|x| Q_at_r.evaluate(&x).unwrap())
      .fold(Fq::zero(), |acc, result| acc + result);
    assert_ne!(G_at_r, q.evaluate(&r).unwrap());
  }

  #[test]
  fn test_fix_variables() {
    let A = SparseMatrix::<Ep>::with_coeffs(
      4,
      4,
      vec![
        (0, 0, Fq::from(2u64)),
        (0, 1, Fq::from(3u64)),
        (0, 2, Fq::from(4u64)),
        (0, 3, Fq::from(4u64)),
        (1, 0, Fq::from(4u64)),
        (1, 1, Fq::from(11u64)),
        (1, 2, Fq::from(14u64)),
        (1, 3, Fq::from(14u64)),
        (2, 0, Fq::from(2u64)),
        (2, 1, Fq::from(8u64)),
        (2, 2, Fq::from(17u64)),
        (2, 3, Fq::from(17u64)),
        (3, 0, Fq::from(420u64)),
        (3, 1, Fq::from(4u64)),
        (3, 2, Fq::from(2u64)),
        (3, 3, Fq::ZERO),
      ],
    );

    let A_mle = A.to_mle();
    let bhc = BooleanHypercube::<Fq>::new(2);
    for (i, y) in bhc.enumerate() {
      let A_mle_op = fix_variables(&A_mle, &y);

      // Check that fixing first variables pins down a column
      // i.e. fixing x to 0 will return the first column
      //      fixing x to 1 will return the second column etc.
      let column_i: Vec<Fq> = A
        .clone()
        .coeffs()
        .iter()
        .copied()
        .filter_map(|(_, col, coeff)| if col == i { Some(coeff) } else { None })
        .collect();

      assert_eq!(A_mle_op.Z, column_i);

      // // Now check that fixing last variables pins down a row
      // // i.e. fixing y to 0 will return the first row
      // //      fixing y to 1 will return the second row etc.
      let row_i: Vec<Fq> = A
        .clone()
        .coeffs()
        .iter()
        .copied()
        .filter_map(|(row, _, coeff)| if row == i { Some(coeff) } else { None })
        .collect();

      let mut last_vars_fixed = A_mle.clone();
      // this is equivalent to Espresso/hyperplonk's 'fix_last_variables' mehthod
      for bit in y.clone().iter().rev() {
        last_vars_fixed.bound_poly_var_top(&bit)
      }

      assert_eq!(last_vars_fixed.Z, row_i);
    }
  }
}
