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
use itertools::concat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};
use std::sync::Arc;

use super::CCS;
pub(crate) mod virtual_poly;
pub(crate) use virtual_poly::VirtualPolynomial;

/// Computes the MLE of the CCS's Matrix at index `j` and executes the reduction of it summing over the given z.
pub fn compute_sum_Mz<G: Group>(
  M_mle: &MultilinearPolynomial<G::Scalar>,
  z: &MultilinearPolynomial<G::Scalar>,
) -> MultilinearPolynomial<G::Scalar> {
  let mut sum_Mz = MultilinearPolynomial::new(vec![
    G::Scalar::ZERO;
    1 << (M_mle.get_num_vars() - z.get_num_vars())
  ]);

  let bhc = BooleanHypercube::<G::Scalar>::new(z.get_num_vars());
  for y in bhc.into_iter() {
    let M_y = fix_variables(M_mle, &y);

    // reverse y to match spartan/polynomial evaluate
    let y_rev: Vec<G::Scalar> = y.into_iter().rev().collect();
    let z_y = z.evaluate(&y_rev);
    let M_z = M_y.scalar_mul(&z_y);
    // XXX: It's crazy to have results in the ops impls. Remove them!
    sum_Mz = sum_Mz.clone().add(M_z).expect("This should not fail");
  }

  sum_Mz
}

pub(crate) fn fix_variables<F: PrimeField>(
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

/// Return a vector of evaluations p_j(r) = \sum_{y \in {0,1}^s'} M_j(r, y) * z(y)
/// for all j values in 0..self.t
pub fn compute_all_sum_Mz_evals<G: Group>(
  M_x_y_mle: &[MultilinearPolynomial<G::Scalar>],
  z: &[G::Scalar],
  r: &[G::Scalar],
  s_prime: usize,
) -> Vec<G::Scalar> {
  // Convert z to MLE
  let z_y_mle = dense_vec_to_mle(s_prime, z);

  let mut v = Vec::with_capacity(M_x_y_mle.len());
  for M_i in M_x_y_mle {
    let sum_Mz = compute_sum_Mz::<G>(M_i, &z_y_mle);

    // XXX: We need a better way to do this. Sum_Mz has also the same issue.
    // reverse the `r` given to evaluate to match Spartan/Nova endianness.
    let mut r = r.to_vec();
    r.reverse();

    let v_i = sum_Mz.evaluate(&r);
    v.push(v_i);
  }
  v
}

#[cfg(test)]
mod tests {
  use crate::ccs::cccs::CCCS;

  use super::*;
  use pasta_curves::{Ep, Fq};
  use rand_core::OsRng;

  fn test_fix_variables_with<F: PrimeField>() {
    let A = SparseMatrix::<F>::with_coeffs(
      4,
      4,
      vec![
        (0, 0, F::from(2u64)),
        (0, 1, F::from(3u64)),
        (0, 2, F::from(4u64)),
        (0, 3, F::from(4u64)),
        (1, 0, F::from(4u64)),
        (1, 1, F::from(11u64)),
        (1, 2, F::from(14u64)),
        (1, 3, F::from(14u64)),
        (2, 0, F::from(2u64)),
        (2, 1, F::from(8u64)),
        (2, 2, F::from(17u64)),
        (2, 3, F::from(17u64)),
        (3, 0, F::from(420u64)),
        (3, 1, F::from(4u64)),
        (3, 2, F::from(2u64)),
        (3, 3, F::ZERO),
      ],
    );

    let A_mle = A.to_mle();
    let bhc = BooleanHypercube::<F>::new(2);
    for (i, y) in bhc.enumerate() {
      let A_mle_op = fix_variables(&A_mle, &y);

      // Check that fixing first variables pins down a column
      // i.e. fixing x to 0 will return the first column
      //      fixing x to 1 will return the second column etc.
      let column_i: Vec<F> = A
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
      let row_i: Vec<F> = A
        .clone()
        .coeffs()
        .iter()
        .copied()
        .filter_map(|(row, _, coeff)| if row == i { Some(coeff) } else { None })
        .collect();

      let mut last_vars_fixed = A_mle.clone();
      // this is equivalent to Espresso/hyperplonk's 'fix_last_variables' mehthod
      for bit in y.clone().iter().rev() {
        last_vars_fixed.bound_poly_var_top(bit)
      }

      assert_eq!(last_vars_fixed.Z, row_i);
    }
  }

  fn test_compute_sum_Mz_over_boolean_hypercube_with<G: Group>() {
    let z = CCS::<G>::get_test_z(3);
    let (ccs, _, _, mles) = CCS::<G>::gen_test_ccs(&z);

    // Generate other artifacts
    let ck = CCS::<G>::commitment_key(&ccs);
    let z_mle = dense_vec_to_mle(ccs.s_prime, &z);
    let cccs = CCCS::new(&ccs, &mles, z, &ck);

    // check that evaluating over all the values x over the boolean hypercube, the result of
    // the next for loop is equal to 0
    let mut r = G::Scalar::ZERO;
    let bch = BooleanHypercube::new(ccs.s);
    for x in bch.into_iter() {
      for i in 0..ccs.q {
        let mut Sj_prod = G::Scalar::ONE;
        for j in ccs.S[i].clone() {
          let sum_Mz: MultilinearPolynomial<G::Scalar> = compute_sum_Mz::<G>(&mles[j], &z_mle);
          let sum_Mz_x = sum_Mz.evaluate(&x);
          Sj_prod *= sum_Mz_x;
        }
        r += Sj_prod * ccs.c[i];
      }
      assert_eq!(r, G::Scalar::ZERO);
    }
  }

  fn test_compute_all_sum_Mz_evals_with<G: Group>() {
    let z = CCS::<G>::get_test_z(3);
    let (ccs, _, _, mles) = CCS::<G>::gen_test_ccs(&z);

    let mut r = vec![G::Scalar::ONE, G::Scalar::ZERO];
    let res = compute_all_sum_Mz_evals::<G>(&mles, z.as_slice(), &r, ccs.s_prime);
    assert_eq!(
      res,
      vec![
        G::Scalar::from(9u64),
        G::Scalar::from(3u64),
        G::Scalar::from(27u64)
      ]
    );

    r.reverse();
    let res = compute_all_sum_Mz_evals::<G>(&mles, z.as_slice(), &r, ccs.s_prime);
    assert_eq!(
      res,
      vec![
        G::Scalar::from(30u64),
        G::Scalar::from(1u64),
        G::Scalar::from(30u64)
      ]
    )
  }

  #[test]
  fn test_fix_variables() {
    test_fix_variables_with::<Fq>();
  }

  #[test]
  fn test_compute_sum_Mz_over_boolean_hypercube() {
    test_compute_sum_Mz_over_boolean_hypercube_with::<Ep>();
  }

  #[test]
  fn test_compute_all_sum_Mz_evals() {
    test_compute_all_sum_Mz_evals_with::<Ep>();
  }
}
