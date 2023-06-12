//! This module defines CCS related types and functions.
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::type_complexity)]

use crate::spartan::math::Math;
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
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,

  pub(crate) M: Vec<SparseMatrix<G>>,
  pub(crate) t: usize,
  pub(crate) q: usize,
  pub(crate) d: usize,
  pub(crate) S: Vec<Vec<usize>>,
  pub(crate) c: Vec<usize>,

  // m is the number of columns in M_i
  pub(crate) m: usize,
  // n is the number of rows in M_i
  pub(crate) n: usize,
  // s = log m
  pub(crate) s: usize,
  // s_prime = log n
  pub(crate) s_prime: usize,
}

/// A type that holds a witness for a given CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CCSWitness<G: Group> {
  // Vector W in F^{n - l - 1}
  W: Vec<G::Scalar>,
}

// TODO: Make sure this is in the right form for committed CCS using MLE, possibly a separate type
/// A type that holds an CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCSInstance<G: Group> {
  // XXX: Move commitment out of CCSInstance for more clean conceptual separation?
  // (Pedersen) Commitment to a witness
  pub(crate) comm_W: Commitment<G>,

  // Public input x in F^l
  pub(crate) X: Vec<G::Scalar>,
}


/// A type that holds the shape of a Committed CCS (CCCS) instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCCSShape<G: Group> {  
  // Sequence of sparse MLE polynomials in s+s' variables M_MLE1, ..., M_MLEt
  // TODO This should be MLE
  // XXX Here atm - look at other example see how it is
  pub(crate) M_MLE: Vec<SparseMatrix<G>>,


  // XXX Embed CCS directly here or do a flat structure?
  // pub(crate) ccs: CCS,

  // q multisets S (same as CCS)
  // q constants c (same as CCS)

}

/// CCCS Instance is (C, x)
/// CCCS Witness is w _mle


// NOTE: We deal with `r` parameter later in `nimfs.rs` when running `execute_sequence` with `ro_consts`
/// A type that holds a CCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCCSInstance<G: Group> {
  // Commitment to a multilinear polynomial in s' - 1 variables
  pub(crate) C: Commitment<G>,

  // $x in F^l$
  pub(crate) X: Vec<G::Scalar>,
}

/// A type that holds a witness for a given CCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CCCSWitness<G: Group> {
  // Multilinear polynomial w_mle in s' - 1 variables
  pub(crate) w_mle: Vec<G::Scalar>,
}

// NOTE: We deal with `r` parameter later in `nimfs.rs` when running `execute_sequence` with `ro_consts`
/// A type that holds a LCCCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LCCCSInstance<G: Group> {
  pub(crate) C: Commitment<G>,
  pub(crate) X: Vec<G::Scalar>,
  pub(crate) u: G::Scalar,
  pub(crate) v: Vec<G::Scalar>,
}

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
    let num_vars = S.num_vars;
    let total_nz = S.M.iter().fold(0, |acc, m| acc + m.0.len());

    G::CE::setup(b"ck", max(max(num_cons, num_vars), total_nz))
  }
}

impl<G: Group> CCSShape<G> {
  /// Create an object of type `CCSSShape` from the explicitly specified CCS matrices
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    M: &[Vec<(usize, usize, G::Scalar)>],
    t: usize,
    q: usize,
    d: usize,
    S: Vec<Vec<usize>>,
    c: Vec<usize>,
  ) -> Result<CCSShape<G>, NovaError> {
    let is_valid = |num_cons: usize,
                    num_vars: usize,
                    num_io: usize,
                    matrix: &[(usize, usize, G::Scalar)]|
     -> Result<(), NovaError> {
      let res = (0..matrix.len())
        .map(|i| {
          let (row, col, _val) = matrix[i];
          if row >= num_cons || col > num_io + num_vars {
            Err(NovaError::InvalidIndex)
          } else {
            Ok(())
          }
        })
        .collect::<Result<Vec<()>, NovaError>>();

      if res.is_err() {
        Err(NovaError::InvalidIndex)
      } else {
        Ok(())
      }
    };

    // Check that the row and column indexes are within the range of the number of constraints and variables
    let res_M = M
      .iter()
      .map(|m| is_valid(num_cons, num_vars, num_io, m))
      .collect::<Result<Vec<()>, NovaError>>();

    // If any of the matricies are invalid, return an error
    if res_M.is_err() {
      return Err(NovaError::InvalidIndex);
    }

    // We require the number of public inputs/outputs to be even
    if num_io % 2 != 0 {
      return Err(NovaError::OddInputLength);
    }

    // We collect the matrixes.
    let M: Vec<SparseMatrix<G>> = M.iter().map(|m| SparseMatrix::from(m)).collect();

    // NOTE: All matricies have the same number of rows, but in a SparseMatrix we need to check all of them
    // Can probably be made more efficient by keeping track fo n_rows/n_cols at creation/insert time
    let m = M.iter().fold(0, |acc, matrix| max(acc, matrix.n_rows()));
    let n = M.iter().fold(0, |acc, matrix| max(acc, matrix.n_cols()));

    let s = m.log_2() as usize;
    let s_prime = n.log_2() as usize;

    let shape = CCSShape {
      num_cons,
      num_vars,
      num_io,
      M,
      t,
      q,
      d,
      S,
      c,
      m,
      n,
      s,
      s_prime,
    };

    Ok(shape)
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
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(U.X.len(), self.num_io);

    // NOTE: All matricies have the same number of rows, but in a SparseMatrix we need to check all of them
    // Can probably be made more efficient by keeping track fo n_rows/n_cols at creation/insert time
    let m = self
      .M
      .iter()
      .fold(0, |acc, matrix| max(acc, matrix.n_rows()));

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
    let res_eq: bool = {
      let mut r = vec![G::Scalar::ZERO; m];
      let z = concat(vec![W.W.clone(), vec![G::Scalar::ONE], U.X.clone()]);

      for i in 0..self.q {
        let mut hadamard_output = vec![G::Scalar::ONE; m];
        for j in &self.S[i] {
          let mvp = matrix_vector_product_sparse(&self.M[*j], &z)?;
          hadamard_output = hadamard_product(&hadamard_output, &mvp)?;
        }

        // XXX: Problem if c[i] is F?
        let civ = G::Scalar::from(self.c[i] as u64);
        let vep = vector_elem_product(&hadamard_output, &civ)?;

        r = vector_add(&r, &vep)?;
      }
      r == vec![G::Scalar::ZERO; m]
    };

    // verify if comm_W is a commitment to W
    let res_comm: bool = U.comm_W == CE::<G>::commit(ck, &W.W);

    if res_eq && res_comm {
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
    const C0: i32 = 1;
    const C1: i32 = -1;

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
      num_vars: r1cs.num_vars,
      num_io: r1cs.num_io,
      M: vec![r1cs.A.into(), r1cs.B.into(), r1cs.C.into()],
      t: T,
      q: Q,
      d: D,
      S: vec![S1.to_vec(), S2.to_vec()],
      c: vec![C0 as usize, C1 as usize],
      m: m,
      n: n,
      s: s,
      s_prime: s_prime,
    }
  }

  /// Pads the R1CSShape so that the number of variables is a power of two
  /// Renumbers variables to accomodate padded variables
  pub fn pad(&mut self) {
    // XXX: Is this definitely always the same as m number of rows?
    // equalize the number of variables and constraints
    let m = max(self.num_vars, self.num_cons).next_power_of_two();

    // check if the provided R1CSShape is already as required
    if self.num_vars == m && self.num_cons == m {
      return;
    }

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.num_vars == m {
      *self = CCSShape {
        num_cons: m,
        num_vars: m,
        num_io: self.num_io,
        M: self.M.clone(),
        t: self.t,
        q: self.q,
        d: self.d,
        S: self.S.clone(),
        c: self.c.clone(),
        m: self.m,
        n: self.n,
        s: self.s,
        s_prime: self.s_prime,
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = m;
    let apply_pad = |M: &mut SparseMatrix<G>| {
      M.0.par_iter_mut().for_each(|(_, c, _)| {
        *c = if *c >= self.num_vars {
          *c + num_vars_padded - self.num_vars
        } else {
          *c
        };
      });
    };

    // Apply pad for each matrix in M
    let mut M_padded = self.M.clone();
    M_padded.iter_mut().for_each(|m| apply_pad(m));

    // TODO: Sanity check if CCS padding is correct here
  }
}

impl<G: Group> CCSWitness<G> {
  /// A method to create a witness object using a vector of scalars
  pub fn new(S: &CCSShape<G>, W: &[G::Scalar]) -> Result<CCSWitness<G>, NovaError> {
    if S.num_vars != W.len() {
      Err(NovaError::InvalidWitnessLength)
    } else {
      Ok(CCSWitness { W: W.to_owned() })
    }
  }

  /// Commits to the witness using the supplied generators
  pub fn commit(&self, ck: &CommitmentKey<G>) -> Commitment<G> {
    CE::<G>::commit(ck, &self.W)
  }
}

impl<G: Group> CCSInstance<G> {
  /// A method to create an instance object using consitituent elements
  pub fn new(
    S: &CCSShape<G>,
    comm_W: &Commitment<G>,
    X: &[G::Scalar],
  ) -> Result<CCSInstance<G>, NovaError> {
    if S.num_io != X.len() {
      Err(NovaError::InvalidInputLength)
    } else {
      Ok(CCSInstance {
        comm_W: *comm_W,
        X: X.to_owned(),
      })
    }
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
