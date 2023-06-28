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

use super::cccs::fix_variables;
use super::CCSShape;
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
    let M_y = fix_variables(&M_mle, &y);

    // reverse y to match spartan/polynomial evaluate
    let y_rev: Vec<G::Scalar> = y.into_iter().rev().collect();
    let z_y = z.evaluate(&y_rev);
    let M_z = M_y.scalar_mul(&z_y);
    // XXX: It's crazy to have results in the ops impls. Remove them!
    sum_Mz = sum_Mz.clone().add(M_z).expect("This should not fail");
  }

  sum_Mz
}
