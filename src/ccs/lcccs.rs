use super::util::{compute_sum_Mz, VirtualPolynomial};
use super::{CCSWitness, CCCS, CCS};
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
use rand_core::RngCore;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};
use std::sync::Arc;

/// A type that holds a LCCCS instance
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LCCCS<G: Group> {
  pub(crate) w_comm: Commitment<G>,
  pub(crate) v: Vec<G::Scalar>,
  // Random evaluation point for the v_i
  pub(crate) r_x: Vec<G::Scalar>,
  pub(crate) z: Vec<G::Scalar>,
}

impl<G: Group> LCCCS<G> {
  /// Generates a new LCCCS instance from a given randomness, CommitmentKey & witness input vector.
  /// This should only be used to probably test or setup the initial NIMFS instance.
  pub(crate) fn new<R: RngCore>(
    ccs: &CCS<G>,
    ccs_m_mle: &[MultilinearPolynomial<G::Scalar>],
    ck: &CommitmentKey<G>,
    z: Vec<G::Scalar>,
    mut rng: &mut R,
  ) -> Self {
    // XXX: API doesn't offer a way to handle this??
    let _r_w = G::Scalar::random(&mut rng);
    let w_comm = <<G as Group>::CE as CommitmentEngineTrait<G>>::commit(ck, &z[(1 + ccs.l)..]);

    // Evaluation points for `v`
    let r_x: Vec<G::Scalar> = (0..ccs.s).map(|_| G::Scalar::random(&mut rng)).collect();
    let v = ccs.compute_v_j(&z, &r_x, ccs_m_mle);

    Self { w_comm, v, r_x, z }
  }

  /// Checks if the CCS instance is satisfiable given a witness and its shape
  pub fn is_sat(
    &self,
    ccs: &CCS<G>,
    ccs_m_mle: &[MultilinearPolynomial<G::Scalar>],
    ck: &CommitmentKey<G>,
  ) -> Result<(), NovaError> {
    dbg!(self.z.clone());
    let w = &self.z[(1 + ccs.l)..];
    // check that C is the commitment of w. Notice that this is not verifying a Pedersen
    // opening, but checking that the Commmitment comes from committing to the witness.
    let comm_eq = self.w_comm == CE::<G>::commit(ck, w);

    let computed_v = compute_all_sum_Mz_evals::<G>(ccs_m_mle, &self.z, &self.r_x, ccs.s_prime);
    dbg!(self.v.clone());
    dbg!(computed_v.clone());
    let vs_eq = computed_v == self.v;

    if vs_eq && comm_eq {
      Ok(())
    } else {
      Err(NovaError::UnSat)
    }
  }

  /// Compute all L_j(x) polynomials.
  pub fn compute_Ls(
    &self,
    ccs: &CCS<G>,
    ccs_m_mle: &[MultilinearPolynomial<G::Scalar>],
  ) -> Vec<VirtualPolynomial<G::Scalar>> {
    let z_mle = dense_vec_to_mle(ccs.s_prime, self.z.as_slice());

    let mut vec_L_j_x = Vec::with_capacity(ccs.t);
    for M_j in ccs_m_mle.iter() {
      // Sanity check
      assert_eq!(z_mle.get_num_vars(), ccs.s_prime);

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

  fn satisfied_ccs_is_satisfied_lcccs_with<G: Group>() {
    // Gen test vectors & artifacts
    let z = CCS::<G>::get_test_z(3);
    let (ccs, witness, instance, mles) = CCS::<G>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();
    assert!(ccs.is_sat(&ck, &instance, &witness).is_ok());

    // LCCCS with the correct z should pass
    let mut lcccs = LCCCS::new(&ccs, &mles, &ck, z.clone(), &mut OsRng);
    assert!(lcccs.is_sat(&ccs, &mles, &ck).is_ok());

    // Wrong z so that the relation does not hold
    let mut bad_z = z;
    bad_z[3] = G::Scalar::ZERO;

    // LCCCS with the wrong z should not pass `is_sat`.
    lcccs.z = bad_z;
    assert!(lcccs.is_sat(&ccs, &mles, &ck).is_err());
  }

  fn test_lcccs_v_j_with<G: Group>() {
    let mut rng = OsRng;

    // Gen test vectors & artifacts
    let z = CCS::<G>::get_test_z(3);
    let (ccs, _, _, mles) = CCS::<G>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();

    // Get LCCCS
    let lcccs = LCCCS::new(&ccs, &mles, &ck, z, &mut OsRng);

    let vec_L_j_x = lcccs.compute_Ls(&ccs, &mles);
    assert_eq!(vec_L_j_x.len(), lcccs.v.len());

    for (v_i, L_j_x) in lcccs.v.into_iter().zip(vec_L_j_x) {
      let sum_L_j_x = BooleanHypercube::new(ccs.s)
        .map(|y| L_j_x.evaluate(&y).unwrap())
        .fold(G::Scalar::ZERO, |acc, result| acc + result);
      assert_eq!(v_i, sum_L_j_x);
    }
  }

  fn test_bad_v_j_with<G: Group>() {
    let mut rng = OsRng;

    // Gen test vectors & artifacts
    let z = CCS::<G>::get_test_z(3);
    let (ccs, witness, instance, mles) = CCS::<G>::gen_test_ccs(&z);
    let ck = ccs.commitment_key();

    // Mutate z so that the relation does not hold
    let mut bad_z = z.clone();
    bad_z[3] = G::Scalar::ZERO;

    // Compute v_j with the right z
    let mut lcccs = LCCCS::new(&ccs, &mles, &ck, z, &mut OsRng);
    // Assert LCCCS is satisfied with the original Z
    assert!(lcccs.is_sat(&ccs, &mles, &ck).is_ok());

    // Compute L_j(x) with the bad z
    lcccs.z = bad_z;
    let vec_L_j_x = lcccs.compute_Ls(&ccs, &mles);
    assert_eq!(vec_L_j_x.len(), lcccs.v.len());
    // Assert LCCCS is not satisfied with the bad Z
    assert!(lcccs.is_sat(&ccs, &mles, &ck).is_err());

    // Make sure that the LCCCS is not satisfied given these L_j(x)
    // i.e. summing L_j(x) over the hypercube should not give v_j for all j
    let mut satisfied = true;
    for (v_i, L_j_x) in lcccs.v.into_iter().zip(vec_L_j_x) {
      let sum_L_j_x = BooleanHypercube::new(ccs.s)
        .map(|y| L_j_x.evaluate(&y).unwrap())
        .fold(G::Scalar::ZERO, |acc, result| acc + result);
      if v_i != sum_L_j_x {
        satisfied = false;
      }
    }

    assert!(!satisfied);
  }

  #[test]
  fn satisfied_ccs_is_satisfied_lcccs() {
    satisfied_ccs_is_satisfied_lcccs_with::<Ep>();
  }

  #[test]
  /// Test linearized CCCS v_j against the L_j(x)
  fn test_lcccs_v_j() {
    test_lcccs_v_j_with::<Ep>();
  }

  /// Given a bad z, check that the v_j should not match with the L_j(x)
  #[test]
  fn test_bad_v_j() {
    test_bad_v_j_with::<Ep>();
  }
}
