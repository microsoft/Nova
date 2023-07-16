//! This module defines CCS related types and functions.
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused)]
#![allow(clippy::type_complexity)]

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
use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use itertools::concat;
use rand_core::RngCore;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::ops::{Add, Mul};

use self::cccs::{CCCSInstance, CCCSShape};
use self::lcccs::LCCCS;
use self::util::compute_all_sum_Mz_evals;

mod cccs;
mod lcccs;
mod multifolding;
mod util;

/// A type that holds the shape of a CCS instance
/// Unlike R1CS we have a list of matrices M instead of only A, B, C
/// We also have t, q, d constants and c (vector), S (set)
/// As well as m, n, s, s_prime
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCSShape<G: Group> {
  pub(crate) M: Vec<SparseMatrix<G::Scalar>>,
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

impl<G: Group> CCSShape<G> {
  /// Create an object of type `CCSShape` from the explicitly specified CCS matrices
  pub fn new(
    M: &[SparseMatrix<G::Scalar>],
    t: usize,
    l: usize,
    q: usize,
    d: usize,
    S: Vec<Vec<usize>>,
    c: Vec<G::Scalar>,
  ) -> CCSShape<G> {
    // Can probably be made more efficient by keeping track fo n_rows/n_cols at creation/insert time
    let m = M
      .iter()
      .fold(usize::MIN, |acc, matrix| max(acc, matrix.n_rows()));
    let n = M
      .iter()
      .fold(usize::MIN, |acc, matrix| max(acc, matrix.n_cols()));

    // Check that the row and column indexes are within the range of the number of constraints and variables
    assert!(M
      .iter()
      .map(|matrix| matrix.is_valid(m, t, l))
      .collect::<Result<Vec<()>, NovaError>>()
      .is_ok());

    // We require the number of public inputs/outputs to be even
    assert_ne!(l % 2, 0, " number of public i/o has to be even");

    let s = m.log_2();
    let s_prime = n.log_2();

    CCSShape {
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
    }
  }

  pub(crate) fn to_cccs_shape(&self) -> CCCSShape<G> {
    let M_mle = self.M.iter().map(|matrix| matrix.to_mle()).collect();
    CCCSShape {
      M_MLE: M_mle,
      ccs: self.clone(),
    }
  }

  // Transform the CCS instance into a CCCS instance by providing a commitment key.
  pub fn to_cccs<R: RngCore>(
    &self,
    rng: &mut R,
    ck: &<<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey,
    z: &[G::Scalar],
  ) -> (CCCSInstance<G>, CCSWitness<G>, CCCSShape<G>) {
    let w: Vec<G::Scalar> = z[(1 + self.l)..].to_vec();
    // XXX: API doesn't offer a way to handle this apparently?
    // Need to investigate
    let _r_w = G::Scalar::random(rng);
    let C = <<G as Group>::CE as CommitmentEngineTrait<G>>::commit(ck, &w);

    (
      CCCSInstance {
        C,
        x: z[1..(1 + self.l)].to_vec(),
      },
      CCSWitness { w },
      self.to_cccs_shape(),
    )
  }

  /// Transform the CCS instance into an LCCCS instance by providing a commitment key.
  pub fn to_lcccs<R: RngCore>(
    &self,
    mut rng: &mut R,
    ck: &<<G as Group>::CE as CommitmentEngineTrait<G>>::CommitmentKey,
    z: &[G::Scalar],
  ) -> (LCCCS<G>, CCSWitness<G>) {
    let w: Vec<G::Scalar> = z[(1 + self.l)..].to_vec();
    let r_w = G::Scalar::random(&mut rng);
    let C = <<G as Group>::CE as CommitmentEngineTrait<G>>::commit(ck, &w);

    // XXX: API doesn't offer a way to handle this??
    let _r_x: Vec<G::Scalar> = (0..self.s).map(|_| G::Scalar::random(&mut rng)).collect();

    let v = self.compute_v_j(z, &_r_x);
    // XXX: Is absurd to compute these again here. We should take care of this.
    let matrix_mles: Vec<MultilinearPolynomial<G::Scalar>> =
      self.M.iter().map(|matrix| matrix.to_mle()).collect();

    (
      LCCCS::<G> {
        ccs: self.clone(),
        C,
        u: G::Scalar::ONE,
        x: z[1..(1 + self.l)].to_vec(),
        r_x: _r_x,
        v,
        matrix_mles,
      },
      CCSWitness::<G> { w },
    )
  }

  /// Compute v_j values of the linearized committed CCS form
  /// Given `r`, compute:  \sum_{y \in {0,1}^s'} M_j(r, y) * z(y)
  fn compute_v_j(&self, z: &[G::Scalar], r: &[G::Scalar]) -> Vec<G::Scalar> {
    // XXX: Should these be MLE already?
    let M_mle: Vec<MultilinearPolynomial<G::Scalar>> =
      self.M.iter().map(|matrix| matrix.to_mle()).collect();
    compute_all_sum_Mz_evals::<G>(&M_mle, &z.to_vec(), r, self.s_prime)
  }

  // XXX: Update commitment_key variables here? This is currently based on R1CS with M length
  /// Samples public parameters for the specified number of constraints and variables in an CCS
  pub fn commitment_key(&self) -> CommitmentKey<G> {
    let total_nz = self.M.iter().fold(0, |acc, m| acc + m.coeffs().len());

    G::CE::setup(b"ck", max(max(self.m, self.t), total_nz))
  }

  /// Checks if the CCS instance is satisfiable given a witness and its shape
  // XXX: Probably is better to completelly remove the abstraction of Instance and witness and just deal with Z.
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<G>,
    U: &CCSInstance<G>,
    W: &CCSWitness<G>,
  ) -> Result<(), NovaError> {
    assert_eq!(W.w.len(), self.n - self.l - 1);
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

    let r = (0..self.q).fold(vec![G::Scalar::ZERO; self.m], |r, idx| {
      let hadamard_output = self.S[idx]
        .iter()
        .fold(vec![G::Scalar::ZERO; self.m], |acc, j| {
          let mvp = matrix_vector_product_sparse(&self.M[*j], &z);
          hadamard_product(&acc, &mvp)
        });

      // Multiply by the coefficient of this step
      let c_M_j_z: Vec<<G as Group>::Scalar> = vector_elem_product(&hadamard_output, self.c[idx]);

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

    // Generate the SparseMatrix vec
    let A = SparseMatrix::with_coeffs(r1cs.num_cons, r1cs.num_vars, r1cs.A);
    let B = SparseMatrix::with_coeffs(r1cs.num_cons, r1cs.num_vars, r1cs.B);
    let C = SparseMatrix::with_coeffs(r1cs.num_cons, r1cs.num_vars, r1cs.C);

    // Assert all matrixes have the same row/column length.
    assert_eq!(A.n_cols(), B.n_cols());
    assert_eq!(B.n_cols(), C.n_cols());
    assert_eq!(A.n_rows(), B.n_rows());
    assert_eq!(B.n_rows(), C.n_rows());

    Self {
      M: vec![A, B, C],
      t: T,
      l: r1cs.num_io,
      q: Q,
      d: D,
      S: vec![S1.to_vec(), S2.to_vec()],
      c: vec![G::Scalar::ONE, -G::Scalar::ONE],
      m: r1cs.num_cons,
      n: r1cs.num_vars,
      s: r1cs.num_cons.log_2(),
      s_prime: r1cs.num_vars.log_2(),
    }
  }

  /// Pads the CCSShape so that the number of variables is a power of two
  /// Renumbers variables to accomodate padded variables
  pub fn pad(&mut self) {
    let padded_n = self.n.next_power_of_two();

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.n != padded_n {
      // Apply pad for each matrix in M
      self.M.iter_mut().for_each(|m| {
        m.pad();
        *m.n_rows_mut() = padded_n
      });
      self.n = padded_n;
    }
  }

  #[cfg(test)]
  pub(crate) fn gen_test_ccs(z: &[G::Scalar]) -> (CCSShape<G>, CCSWitness<G>, CCSInstance<G>) {
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
    // TODO: The third argument should be 2 or similar, need to adjust test case
    // See https://github.com/privacy-scaling-explorations/Nova/issues/30
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

  #[cfg(test)]
  /// Computes the z vector for the given input for Vitalik's equation.
  pub(crate) fn get_test_z(input: u64) -> Vec<G::Scalar> {
    // z = (1, io, w)
    let input = G::Scalar::from(input);
    vec![
      G::Scalar::ONE,
      input,
      input * input * input + input + G::Scalar::from(5u64), // x^3 + x + 5
      input * input,                                         // x^2
      input * input * input,                                 // x^2 * x
      input * input * input + input,                         // x^3 + x
    ]
  }
}

/// A type that holds a witness for a given CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CCSWitness<G: Group> {
  // Vector W in F^{n - l - 1}
  w: Vec<G::Scalar>,
}

impl<G: Group> CCSWitness<G> {
  /// Create a CCSWitness instance from the witness vector.
  pub fn new(witness: Vec<G::Scalar>) -> Self {
    Self { w: witness }
  }

  /// Commits to the witness using the supplied generators
  pub fn commit(&self, ck: &CommitmentKey<G>) -> Commitment<G> {
    CE::<G>::commit(ck, &self.w)
  }
}
/// A type that holds an CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCSInstance<G: Group> {
  // (Pedersen) Commitment to a witness
  pub(crate) comm_w: Commitment<G>,

  // Public input x in F^l
  pub(crate) x: Vec<G::Scalar>,
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
    let _ck = S.commitment_key();
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

        let ccs_w = CCSWitness::new(vars);

        let U = {
          let comm_W = ccs_w.commit(ck);
          let res = CCSInstance::new(&S, &comm_W, X);
          assert!(res.is_ok());
          res.unwrap()
        };

        // check that generated instance is satisfiable
        assert!(S.is_sat(ck, &U, &ccs_w).is_ok());

        (O, U, ccs_w)
      };
  }
}
