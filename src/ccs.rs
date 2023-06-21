//! This module defines CCS related types and functions.
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::type_complexity)]

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
use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use itertools::concat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};

// TODO: Committed CCS using MLE (see src/spartan/pp.rs)
// TODO: Linearized CCS struct and methods, separate struct similar to RelaxedR1CS

/// Public parameters for a given CCS
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCS<G: Group> {
  _p: PhantomData<G>,
}

/// A type that holds the shape of a CCS instance
/// Unlike R1CS we have a list of matrices M instead of only A, B, C
/// We also have t, q, d constants and c (vector), S (set)
/// As well as m, n, s, s_prime
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCSShape<G: Group> {
  pub(crate) num_cons: usize,

  pub(crate) M: Vec<SparseMatrix<G>>,
  // Num vars
  pub(crate) t: usize,
  // Number of public witness
  pub(crate) l: usize,
  pub(crate) q: usize,
  pub(crate) d: usize,
  pub(crate) S: Vec<Vec<usize>>,

  // Was: usize
  pub(crate) c: Vec<G::Scalar>,

  // n is the number of columns in M_i
  pub(crate) n: usize,
  // m is the number of rows in M_i
  pub(crate) m: usize,
  // s = log m
  pub(crate) s: usize,
  // s_prime = log n
  pub(crate) s_prime: usize,
}

/// A type that holds a witness for a given CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CCSWitness<G: Group> {
  // Vector W in F^{n - l - 1}
  w: Vec<G::Scalar>,
}

// XXX: Not sure this type is needed if we do have CCCSInstance and LCCCSInstance.
/// A type that holds an CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCSInstance<G: Group> {
  // XXX: Move commitment out of CCSInstance for more clean conceptual separation?
  // (Pedersen) Commitment to a witness
  pub(crate) comm_w: Commitment<G>,

  // Public input x in F^l
  pub(crate) x: Vec<G::Scalar>,
}

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

/// A type that holds a LCCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LCCCSInstance<G: Group> {
  pub(crate) C: Commitment<G>,
  pub(crate) x: Vec<G::Scalar>,
  pub(crate) u: G::Scalar,
  pub(crate) v: Vec<G::Scalar>,
}

// NOTE: We deal with `r` parameter later in `nimfs.rs` when running `execute_sequence` with `ro_consts`
/// A type that holds a witness for a given LCCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LCCCSWitness<G: Group> {
  pub(crate) w_mle: Vec<G::Scalar>,
}

impl<G: Group> CCS<G> {
  // TODO: Update commitment_key variables here? This is currently based on R1CS with M length
  /// Samples public parameters for the specified number of constraints and variables in an CCS
  pub fn commitment_key(S: &CCSShape<G>) -> CommitmentKey<G> {
    let num_cons = S.num_cons;
    let num_vars = S.t;
    let total_nz = S.M.iter().fold(0, |acc, m| acc + m.0.len());

    G::CE::setup(b"ck", max(max(num_cons, num_vars), total_nz))
  }
}

impl<G: Group> CCSShape<G> {
  /// Create an object of type `CCSShape` from the explicitly specified CCS matrices
  pub fn new(
    num_cons: usize,
    M: &[SparseMatrix<G>],
    t: usize,
    l: usize,
    q: usize,
    d: usize,
    S: Vec<Vec<usize>>,
    c: Vec<G::Scalar>,
  ) -> Result<CCSShape<G>, NovaError> {
    // Check matrix validity

    // Check that the row and column indexes are within the range of the number of constraints and variables
    M.iter()
      .map(|m| m.is_valid(num_cons, t, l))
      .collect::<Result<Vec<()>, NovaError>>()?;

    // We require the number of public inputs/outputs to be even
    if l % 2 != 0 {
      return Err(NovaError::OddInputLength);
    }

    // Can probably be made more efficient by keeping track fo n_rows/n_cols at creation/insert time
    let m = M
      .iter()
      .fold(usize::MIN, |acc, matrix| max(acc, matrix.n_rows()));
    let n = M
      .iter()
      .fold(usize::MIN, |acc, matrix| max(acc, matrix.n_cols()));

    let s = m.log_2() as usize;
    let s_prime = n.log_2() as usize;

    Ok(CCSShape {
      num_cons,
      M: M.to_vec(),
      t,
      l,
      q,
      d,
      S,
      c,
      m,
      n,
      s,
      s_prime,
    })
  }

  // NOTE: Not using previous used multiply_vec (r1cs.rs), see utils.rs

  // NOTE: Equivalent to is_sat_relaxed (r1cs.rs) but for CCCSS/LCCCS?
  // Either here or as a separate method on LCCCS struct

  /// Checks if the CCS instance is satisfiable given a witness and its shape
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<G>,
    U: &CCSInstance<G>,
    W: &CCSWitness<G>,
  ) -> Result<(), NovaError> {
    assert_eq!(W.w.len(), self.t);
    assert_eq!(U.x.len(), self.l);

    // Sage code to check CCS relation:
    //
    // r = [F(0)] * m
    // for i in range(0, q):
    //     hadamard_output = [F(1)]*m
    //     for j in S[i]:
    //         hadamard_output = hadamard_product(hadamard_output,
    //                 matrix_vector_product(M[j], z))
    //
    //     r = vec_add(r, vec_elem_mul(hadamard_output, c[i]))
    // print("\nCCS relation check (∑ cᵢ ⋅ ◯ Mⱼ z == 0):", r == [0]*m)
    //
    // verify if ∑ cᵢ ⋅ ◯ Mⱼ z == 0

    let z = concat(vec![vec![G::Scalar::ONE], U.x.clone(), W.w.clone()]);

    let r = (0..self.q)
      .into_iter()
      .fold(vec![G::Scalar::ZERO; self.m], |r, idx| {
        let hadamard_output = self.S[idx]
          .iter()
          .fold(vec![G::Scalar::ZERO; self.m], |acc, j| {
            let mvp = matrix_vector_product_sparse(&self.M[*j], &z);
            hadamard_product(&acc, &mvp)
          });

        // Multiply by the coefficient of this step
        let c_M_j_z: Vec<<G as Group>::Scalar> =
          vector_elem_product(&hadamard_output, &self.c[idx]);

        vector_add(&r, &c_M_j_z)
      });

    // verify if comm_W is a commitment to W
    let res_comm: bool = U.comm_w == CE::<G>::commit(ck, &W.w);

    if r == vec![G::Scalar::ZERO; self.m] && res_comm {
      Ok(())
    } else {
      Err(NovaError::UnSat)
    }
  }

  pub fn from_r1cs(r1cs: R1CSShape<G>) -> Self {
    // These contants are used for R1CS-to-CCS, see the paper for more details
    const T: usize = 3;
    const Q: usize = 2;
    const D: usize = 2;
    const S1: [usize; 2] = [0, 1];
    const S2: [usize; 1] = [2];

    let l = r1cs.num_io;
    // NOTE: All matricies have the same number of rows, but in a SparseMatrix we need to check all of them
    // TODO: Consider using SparseMatrix type in R1CSShape too
    // XXX: This can probably be made a lot better
    let A: SparseMatrix<G> = r1cs.A.clone().into();
    let B: SparseMatrix<G> = r1cs.B.clone().into();
    let C: SparseMatrix<G> = r1cs.C.clone().into();

    let m = max(A.n_rows(), max(B.n_rows(), C.n_rows()));
    let n = max(A.n_cols(), max(B.n_cols(), C.n_cols()));

    let s = m.log_2() as usize;
    let s_prime = n.log_2() as usize;

    Self {
      num_cons: r1cs.num_cons,
      M: vec![r1cs.A.into(), r1cs.B.into(), r1cs.C.into()],
      t: T,
      l,
      q: Q,
      d: D,
      S: vec![S1.to_vec(), S2.to_vec()],
      c: vec![G::Scalar::ONE, -G::Scalar::ONE],
      m,
      n,
      s,
      s_prime,
    }
  }

  /// Pads the R1CSShape so that the number of variables is a power of two
  /// Renumbers variables to accomodate padded variables
  pub fn pad(&mut self) {
    let (padded_m, padded_n) = (self.m.next_power_of_two(), self.n.next_power_of_two());

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.n != padded_n {
      // Apply pad for each matrix in M
      self.M.iter_mut().for_each(|m| m.pad(padded_n));
      self.n = padded_n;
    }

    // We always update `m` even if it is the same (no need for `if`s).
    self.m = padded_m;
  }
}

impl<G: Group> CCSWitness<G> {
  /// A method to create a witness object using a vector of scalars
  pub fn new(S: &CCSShape<G>, witness: Vec<G::Scalar>) -> CCSWitness<G> {
    assert_eq!(S.t, witness.len());

    Self { w: witness }
  }

  /// Commits to the witness using the supplied generators
  pub fn commit(&self, ck: &CommitmentKey<G>) -> Commitment<G> {
    CE::<G>::commit(ck, &self.w)
  }
}

impl<G: Group> CCSInstance<G> {
  /// A method to create an instance object using consitituent elements
  pub fn new(
    s: &CCSShape<G>,
    w_comm: &Commitment<G>,
    x: Vec<G::Scalar>,
  ) -> Result<CCSInstance<G>, NovaError> {
    assert_eq!(s.l, x.len());

    Ok(CCSInstance { comm_w: *w_comm, x })
  }
}

impl<G: Group> CCCSShape<G> {
  // TODO: Sanity check this
  pub fn compute_sum_Mz(
    &self,
    M_j: &MultilinearPolynomial<G::Scalar>,
    z: &MultilinearPolynomial<G::Scalar>,
    s_prime: usize,
  ) -> MultilinearPolynomial<G::Scalar> {
    assert_eq!(M_j.get_num_vars(), s_prime);
    assert_eq!(z.get_num_vars(), s_prime);

    let num_vars = M_j.get_num_vars();
    let two_to_num_vars = (2_usize).pow(num_vars as u32);
    let mut result_coefficients = Vec::with_capacity(two_to_num_vars);

    for i in 0..two_to_num_vars {
      let r = bit_decompose(i as u64, num_vars)
        .into_iter()
        .map(|bit| G::Scalar::from(if bit { 1 } else { 0 }))
        .collect::<Vec<_>>();

      let value = M_j.evaluate(&r) * z.evaluate(&r);
      result_coefficients.push(value);
    }

    MultilinearPolynomial::new(result_coefficients)
  }

  // XXX: Take below and util functions with a grain of salt, need to sanity check

  // Computes q(x) = \sum^q c_i * \prod_{j \in S_i} ( \sum_{y \in {0,1}^s'} M_j(x, y) * z(y) )
  // polynomial over x
  pub fn compute_q(
    &self,
    z: &Vec<G::Scalar>,
  ) -> Result<MultilinearPolynomial<G::Scalar>, &'static str> {
    // XXX: Do we need to instrument this to use s_prime as n_vars somehow?
    let z_mle = MultilinearPolynomial::new(z.clone());
    if z_mle.get_num_vars() != self.ccs.s_prime {
      return Err("z_mle number of variables does not match ccs.s_prime");
    }
    let mut q = MultilinearPolynomial::new(vec![G::Scalar::ZERO; self.ccs.s]);

    for i in 0..self.ccs.q {
      let mut prod = MultilinearPolynomial::new(vec![G::Scalar::ONE; self.ccs.s]);

      for j in &self.ccs.S[i] {
        let M_j = sparse_matrix_to_mlp(&self.ccs.M[*j]);

        let sum_Mz = self.compute_sum_Mz(&M_j, &z_mle, self.ccs.s_prime);

        // Fold this sum into the running product
        prod = prod.mul(sum_Mz)?;
      }

      // Multiply the product by the coefficient c_i
      prod = prod.scalar_mul(&self.ccs.c[i]);

      // Add it to the running sum
      q = q.add(prod)?;
    }

    Ok(q)

    // Similar logic in Spartan
    //     let (mut Az, mut Bz, mut Cz) = pk.S.multiply_vec(&z)?;
    //poly_Az: MultilinearPolynomial::new(Az.clone()),
  }
}

#[cfg(test)]
pub mod test {
  use super::*;
  use crate::{
    r1cs::R1CS,
    traits::{Group, ROConstantsTrait},
  };
  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use ff::{Field, PrimeField};
  use rand::rngs::OsRng;

  type S = pasta_curves::pallas::Scalar;
  type G = pasta_curves::pallas::Point;

  #[test]
  fn test_tiny_ccs() {
    // 1. Generate valid R1CS Shape
    // 2. Convert to CCS
    // 3. Test that it is satisfiable

    let one = S::one();
    let (num_cons, num_vars, num_io, A, B, C) = {
      let num_cons = 4;
      let num_vars = 4;
      let num_io = 2;

      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      // The R1CS for this problem consists of the following constraints:
      // `I0 * I0 - Z0 = 0`
      // `Z0 * I0 - Z1 = 0`
      // `(Z1 + I0) * 1 - Z2 = 0`
      // `(Z2 + 5) * 1 - I1 = 0`

      // Relaxed R1CS is a set of three sparse matrices (A B C), where there is a row for every
      // constraint and a column for every entry in z = (vars, u, inputs)
      // An R1CS instance is satisfiable iff:
      // Az \circ Bz = u \cdot Cz + E, where z = (vars, 1, inputs)
      let mut A: Vec<(usize, usize, S)> = Vec::new();
      let mut B: Vec<(usize, usize, S)> = Vec::new();
      let mut C: Vec<(usize, usize, S)> = Vec::new();

      // constraint 0 entries in (A,B,C)
      // `I0 * I0 - Z0 = 0`
      A.push((0, num_vars + 1, one));
      B.push((0, num_vars + 1, one));
      C.push((0, 0, one));

      // constraint 1 entries in (A,B,C)
      // `Z0 * I0 - Z1 = 0`
      A.push((1, 0, one));
      B.push((1, num_vars + 1, one));
      C.push((1, 1, one));

      // constraint 2 entries in (A,B,C)
      // `(Z1 + I0) * 1 - Z2 = 0`
      A.push((2, 1, one));
      A.push((2, num_vars + 1, one));
      B.push((2, num_vars, one));
      C.push((2, 2, one));

      // constraint 3 entries in (A,B,C)
      // `(Z2 + 5) * 1 - I1 = 0`
      A.push((3, 2, one));
      A.push((3, num_vars, one + one + one + one + one));
      B.push((3, num_vars, one));
      C.push((3, num_vars + 2, one));

      (num_cons, num_vars, num_io, A, B, C)
    };

    // create a R1CS shape object
    let S = {
      let res = R1CSShape::new(num_cons, num_vars, num_io, &A, &B, &C);
      assert!(res.is_ok());
      res.unwrap()
    };

    // 2. Take R1CS and convert to CCS
    let S = CCSShape::from_r1cs(S);

    // generate generators and ro constants
    let _ck = CCS::<G>::commitment_key(&S);
    let _ro_consts =
      <<G as Group>::RO as ROTrait<<G as Group>::Base, <G as Group>::Scalar>>::Constants::new();

    // 3. Test that CCS is satisfiable
    let _rand_inst_witness_generator =
      |ck: &CommitmentKey<G>, I: &S| -> (S, CCSInstance<G>, CCSWitness<G>) {
        let i0 = *I;

        // compute a satisfying (vars, X) tuple
        let (O, vars, X) = {
          let z0 = i0 * i0; // constraint 0
          let z1 = i0 * z0; // constraint 1
          let z2 = z1 + i0; // constraint 2
          let i1 = z2 + one + one + one + one + one; // constraint 3

          // store the witness and IO for the instance
          let W = vec![z0, z1, z2, S::zero()];
          let X = vec![i0, i1];
          (i1, W, X)
        };

        let W = {
          let res = CCSWitness::new(&S, &vars);
          assert!(res.is_ok());
          res.unwrap()
        };
        let U = {
          let comm_W = W.commit(ck);
          let res = CCSInstance::new(&S, &comm_W, &X);
          assert!(res.is_ok());
          res.unwrap()
        };

        // check that generated instance is satisfiable
        assert!(S.is_sat(ck, &U, &W).is_ok());

        (O, U, W)
      };
  }
}
