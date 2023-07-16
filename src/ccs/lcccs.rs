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

/// A type that holds a LCCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LCCCS<G: Group> {
  pub(crate) C: Commitment<G>,
  pub(crate) x: Vec<G::Scalar>,
  pub(crate) u: G::Scalar,
  pub(crate) v: Vec<G::Scalar>,
  // Random evaluation point for the v_i
  pub r_x: Vec<G::Scalar>,
  // This should not need to be here. Should be a reference only.
  pub(crate) matrix_mles: Vec<MultilinearPolynomial<G::Scalar>>,
  pub(crate) ccs: CCSShape<G>,
}

impl<G: Group> LCCCS<G> {
  // XXX: Double check that this is indeed correct.
  /// Samples public parameters for the specified number of constraints and variables in an CCS
  pub fn commitment_key(&self) -> CommitmentKey<G> {
    let total_nz = self.ccs.M.iter().fold(0, |acc, m| acc + m.coeffs().len());

    G::CE::setup(b"ck", max(max(self.ccs.m, self.ccs.t), total_nz))
  }

  /// Checks if the CCS instance is satisfiable given a witness and its shape
  pub fn is_sat(&self, ck: &CommitmentKey<G>, W: &CCSWitness<G>) -> Result<(), NovaError> {
    // check that C is the commitment of w. Notice that this is not verifying a Pedersen
    // opening, but checking that the Commmitment comes from committing to the witness.
    let comm_eq = self.C == CE::<G>::commit(ck, &W.w);

    // check CCS relation
    let z: Vec<G::Scalar> = [vec![self.u], self.x.clone(), W.w.to_vec()].concat();
    let computed_v =
      compute_all_sum_Mz_evals::<G>(&self.matrix_mles, &z, &self.r_x, self.ccs.s_prime);
    let vs_eq = computed_v == self.v;

    dbg!(vs_eq);
    dbg!(comm_eq);

    if vs_eq && comm_eq {
      Ok(())
    } else {
      Err(NovaError::UnSat)
    }
  }

  /// Compute all L_j(x) polynomials
  // Can we recieve the MLE of z directy?
  pub fn compute_Ls(&self, z: &Vec<G::Scalar>) -> Vec<VirtualPolynomial<G::Scalar>> {
    let z_mle = dense_vec_to_mle(self.ccs.s_prime, z);

    let mut vec_L_j_x = Vec::with_capacity(self.ccs.t);
    for M_j in self.matrix_mles.iter() {
      // Sanity check
      assert_eq!(z_mle.get_num_vars(), self.ccs.s_prime);

      let sum_Mz = compute_sum_Mz::<G>(M_j, &z_mle);
      let sum_Mz_virtual = VirtualPolynomial::new_from_mle(&Arc::new(sum_Mz), G::Scalar::ONE);
      let L_j_x = sum_Mz_virtual.build_f_hat(&self.r_x).unwrap();
      vec_L_j_x.push(L_j_x);
    }

    vec_L_j_x
  }
}

#[cfg(test)]
mod tests {
  use pasta_curves::{Ep, Fq};
  use rand_core::OsRng;

  use super::*;

  #[test]
  fn satisfied_ccs_is_satisfied_lcccs() {
    // Gen test vectors & artifacts
    let z = CCSShape::<Ep>::get_test_z(3);
    let (ccs, witness, instance) = CCSShape::<Ep>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();
    assert!(ccs.is_sat(&ck, &instance, &witness).is_ok());

    // Wrong z so that the relation does not hold
    let mut bad_z = z.clone();
    bad_z[3] = Fq::ZERO;

    // LCCCS with the correct z should pass
    let (lcccs, _) = ccs.to_lcccs(&mut OsRng, &ck, &z);
    assert!(lcccs.is_sat(&ck, &witness).is_ok());

    // LCCCS with the wrong z should not pass `is_sat`.
    // LCCCS with the correct z should pass
    let (lcccs, _) = ccs.to_lcccs(&mut OsRng, &ck, &bad_z);
    assert!(lcccs.is_sat(&ck, &witness).is_err());
  }

  #[test]
  /// Test linearized CCCS v_j against the L_j(x)
  fn test_lcccs_v_j() {
    let mut rng = OsRng;

    // Gen test vectors & artifacts
    let z = CCSShape::<Ep>::get_test_z(3);
    let (ccs, _, _) = CCSShape::<Ep>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();

    // Get LCCCS
    let (lcccs, _) = ccs.to_lcccs(&mut rng, &ck, &z);

    let vec_L_j_x = lcccs.compute_Ls(&z);
    assert_eq!(vec_L_j_x.len(), lcccs.v.len());

    for (v_i, L_j_x) in lcccs.v.into_iter().zip(vec_L_j_x) {
      let sum_L_j_x = BooleanHypercube::new(ccs.s)
        .map(|y| L_j_x.evaluate(&y).unwrap())
        .fold(Fq::ZERO, |acc, result| acc + result);
      assert_eq!(v_i, sum_L_j_x);
    }
  }

  /// Given a bad z, check that the v_j should not match with the L_j(x)
  #[test]
  fn test_bad_v_j() {
    let mut rng = OsRng;

    // Gen test vectors & artifacts
    let z = CCSShape::<Ep>::get_test_z(3);
    let (ccs, witness, instance) = CCSShape::<Ep>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();

    // Mutate z so that the relation does not hold
    let mut bad_z = z.clone();
    bad_z[3] = Fq::ZERO;

    // Compute v_j with the right z
    let (lcccs, _) = ccs.to_lcccs(&mut rng, &ck, &z);
    // Assert LCCCS is satisfied with the original Z
    assert!(lcccs.is_sat(&ck, &witness).is_ok());

    // Bad compute L_j(x) with the bad z
    let vec_L_j_x = lcccs.compute_Ls(&bad_z);
    assert_eq!(vec_L_j_x.len(), lcccs.v.len());
    // Assert LCCCS is not satisfied with the bad Z
    assert!(lcccs.is_sat(&ck, &CCSWitness { w: bad_z }).is_err());

    // Make sure that the LCCCS is not satisfied given these L_j(x)
    // i.e. summing L_j(x) over the hypercube should not give v_j for all j
    let mut satisfied = true;
    for (v_i, L_j_x) in lcccs.v.into_iter().zip(vec_L_j_x) {
      let sum_L_j_x = BooleanHypercube::new(ccs.s)
        .map(|y| L_j_x.evaluate(&y).unwrap())
        .fold(Fq::ZERO, |acc, result| acc + result);
      if v_i != sum_L_j_x {
        satisfied = false;
      }
    }

    assert!(!satisfied);
  }
}
