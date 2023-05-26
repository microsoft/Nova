//! This module defines CCS related types and functions.
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(clippy::type_complexity)]
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  r1cs::R1CSShape,
  traits::{
    commitment::CommitmentEngineTrait, AbsorbInROTrait, Group, ROTrait, TranscriptReprTrait,
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

// TODO, based on r1cs.rs:
// x CCS struct
// x CCS basic impl
// x CCS basic is_sat
// - Clean up old R1CS stuff we don't need
// - Get rid of hardcoded R1CS
// - Linearized/Committed CCS
// - R1CS to CCS

/// Public parameters for a given CCS
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCS<G: Group> {
  _p: PhantomData<G>,
}

// TODO Pull out matrix type?

// A type that holds the shape of a CCS instance
// Unlike R1CS we have a list of matrices M instead of only A, B, C
// We also have t, q, d constants and c (vector), S (set)
// TODO Add m, n, or infer from M?
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CCSShape<G: Group> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,
  pub(crate) M: Vec<Vec<(usize, usize, G::Scalar)>>,
  pub(crate) t: usize,
  pub(crate) q: usize,
  pub(crate) d: usize,
  pub(crate) S: Vec<Vec<usize>>,
  pub(crate) c: Vec<usize>,
}

/// A type that holds a witness for a given CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CCSWitness<G: Group> {
  W: Vec<G::Scalar>,
}

/// A type that holds an CCS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct CCSInstance<G: Group> {
  pub(crate) comm_W: Commitment<G>,
  pub(crate) X: Vec<G::Scalar>,
}

// TODO: Type for other CCS types, eqv to RelaxedR1CS

// TODO: Function to convert R1CS to CCS
// Put here or in r1cs module

// TODO: Util fn to create new CCSShape for R1CS with following values
// n=n, m=m, N=N, l=l, t=3, q=2, d=2
// M={A,B,C}, S={{0, 1}, {2}}, c={1,−1}

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

    let shape = CCSShape {
      num_cons,
      num_vars,
      num_io,
      M: M.to_vec(),
      t,
      q,
      d,
      S,
      c,
      //     S: vec![S1.to_vec(), S2.to_vec()],
      //     c: vec![C0 as usize, C1 as usize],
    };

    Ok(shape)
  }

  // NOTE: Not currently used
  // TODO This has to be updated for CCS to not just return Az, Bz, Cz
  // pub fn multiply_vec(
  //   &self,
  //   z: &[G::Scalar],
  // ) -> Result<Vec<Vec<G::Scalar>>, NovaError> {
  //   if z.len() != self.num_io + self.num_vars + 1 {
  //     return Err(NovaError::InvalidWitnessLength);
  //   }

  //   // computes a product between a sparse matrix `matrix` and a vector `z`
  //   // This does not perform any validation of entries in M (e.g., if entries in `M` reference indexes outside the range of `z`)
  //   // This is safe since we know that `M` is valid
  //   let sparse_matrix_vec_product =
  //     |matrix: &Vec<(usize, usize, G::Scalar)>, num_rows: usize, z: &[G::Scalar]| -> Vec<G::Scalar> {
  //       (0..matrix.len())
  //         .map(|i| {
  //           let (row, col, val) =matrix[i];
  //           (row, val * z[col])
  //         })
  //         .fold(vec![G::Scalar::ZERO; num_rows], |mut Mz, (r, v)| {
  //           Mz[r] += v;
  //           Mz
  //         })
  //     };

  //     // // XXX: Hacky, assumes M is A, B, C (true for R1CS)
  //     // let A = self.M[0].clone();
  //     // let B = self.M[1].clone();
  //     // let C = self.M[2].clone();

  //     // let (Az, (Bz, Cz)) = rayon::join(
  //     //   || sparse_matrix_vec_product(&A, self.num_cons, z),
  //     //   || {
  //     //     rayon::join(
  //     //       || sparse_matrix_vec_product(&B, self.num_cons, z),
  //     //       || sparse_matrix_vec_product(&C, self.num_cons, z),
  //     //     )
  //     //   },
  //     // );

  //   // TODO Use rayon to parallelize
  //   let Mzs = self.M.iter().map(|m| sparse_matrix_vec_product(m, self.num_cons, z)).collect::<Vec<_>>();

  //   Ok(Mzs)
  // }

  /// Checks if the Relaxed R1CS instance is satisfiable given a witness and its shape
  // pub fn is_sat_relaxed(
  //   &self,
  //   ck: &CommitmentKey<G>,
  //   U: &RelaxedR1CSInstance<G>,
  //   W: &RelaxedR1CSWitness<G>,
  // ) -> Result<(), NovaError> {
  //   assert_eq!(W.W.len(), self.num_vars);
  //   assert_eq!(W.E.len(), self.num_cons);
  //   assert_eq!(U.X.len(), self.num_io);

  //   // verify if Az * Bz = u*Cz + E
  //   let res_eq: bool = {
  //     let z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);
  //     let (Az, Bz, Cz) = self.multiply_vec(&z)?;
  //     assert_eq!(Az.len(), self.num_cons);
  //     assert_eq!(Bz.len(), self.num_cons);
  //     assert_eq!(Cz.len(), self.num_cons);

  //     let res: usize = (0..self.num_cons)
  //       .map(|i| usize::from(Az[i] * Bz[i] != U.u * Cz[i] + W.E[i]))
  //       .sum();

  //     res == 0
  //   };

  //   // verify if comm_E and comm_W are commitments to E and W
  //   let res_comm: bool = {
  //     let (comm_W, comm_E) =
  //       rayon::join(|| CE::<G>::commit(ck, &W.W), || CE::<G>::commit(ck, &W.E));
  //     U.comm_W == comm_W && U.comm_E == comm_E
  //   };

  //   if res_eq && res_comm {
  //     Ok(())
  //   } else {
  //     Err(NovaError::UnSat)
  //   }
  // }

  /// Checks if the CCS instance is satisfiable given a witness and its shape
  pub fn is_sat(
    &self,
    ck: &CommitmentKey<G>,
    U: &CCSInstance<G>,
    W: &CCSWitness<G>,
  ) -> Result<(), NovaError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(U.X.len(), self.num_io);

    let m = self.M[0].len();

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

        // XXX Problem if c[i] is F?
        let civ = G::Scalar::from(self.c[i] as u64);
        let vep = vector_elem_product(&hadamard_output, &civ)?;

        r = vector_add(&r, &vep)?;
      }
      r == vec![G::Scalar::ZERO; m]
    };

    // NOTE: Previous R1CS code for reference
    // // verify if Az * Bz = u*Cz
    // let res_eq: bool = {
    //   let z = concat(vec![W.W.clone(), vec![G::Scalar::ONE], U.X.clone()]);
    //   let (Az, Bz, Cz) = self.multiply_vec(&z)?;
    //   assert_eq!(Az.len(), self.num_cons);
    //   assert_eq!(Bz.len(), self.num_cons);
    //   assert_eq!(Cz.len(), self.num_cons);

    //   let res: usize = (0..self.num_cons)
    //     .map(|i| usize::from(Az[i] * Bz[i] != Cz[i]))
    //     .sum();

    //   res == 0
    // };

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

    Self {
      num_cons: r1cs.num_cons,
      num_vars: r1cs.num_vars,
      num_io: r1cs.num_io,
      M: vec![r1cs.A, r1cs.B, r1cs.C],
      t: T,
      q: Q,
      d: D,
      S: vec![S1.to_vec(), S2.to_vec()],
      c: vec![C0 as usize, C1 as usize],
    }
  }

  /// A method to compute a commitment to the cross-term `T` given a
  /// Relaxed R1CS instance-witness pair and an R1CS instance-witness pair
  // pub fn commit_T(
  //   &self,
  //   ck: &CommitmentKey<G>,
  //   U1: &RelaxedR1CSInstance<G>,
  //   W1: &RelaxedR1CSWitness<G>,
  //   U2: &R1CSInstance<G>,
  //   W2: &R1CSWitness<G>,
  // ) -> Result<(Vec<G::Scalar>, Commitment<G>), NovaError> {
  //   let (AZ_1, BZ_1, CZ_1) = {
  //     let Z1 = concat(vec![W1.W.clone(), vec![U1.u], U1.X.clone()]);
  //     self.multiply_vec(&Z1)?
  //   };

  //   let (AZ_2, BZ_2, CZ_2) = {
  //     let Z2 = concat(vec![W2.W.clone(), vec![G::Scalar::ONE], U2.X.clone()]);
  //     self.multiply_vec(&Z2)?
  //   };

  //   let AZ_1_circ_BZ_2 = (0..AZ_1.len())
  //     .into_par_iter()
  //     .map(|i| AZ_1[i] * BZ_2[i])
  //     .collect::<Vec<G::Scalar>>();
  //   let AZ_2_circ_BZ_1 = (0..AZ_2.len())
  //     .into_par_iter()
  //     .map(|i| AZ_2[i] * BZ_1[i])
  //     .collect::<Vec<G::Scalar>>();
  //   let u_1_cdot_CZ_2 = (0..CZ_2.len())
  //     .into_par_iter()
  //     .map(|i| U1.u * CZ_2[i])
  //     .collect::<Vec<G::Scalar>>();
  //   let u_2_cdot_CZ_1 = (0..CZ_1.len())
  //     .into_par_iter()
  //     .map(|i| CZ_1[i])
  //     .collect::<Vec<G::Scalar>>();

  //   let T = AZ_1_circ_BZ_2
  //     .par_iter()
  //     .zip(&AZ_2_circ_BZ_1)
  //     .zip(&u_1_cdot_CZ_2)
  //     .zip(&u_2_cdot_CZ_1)
  //     .map(|(((a, b), c), d)| *a + *b - *c - *d)
  //     .collect::<Vec<G::Scalar>>();

  //   let comm_T = CE::<G>::commit(ck, &T);

  //   Ok((T, comm_T))
  // }

  /// Pads the R1CSShape so that the number of variables is a power of two
  /// Renumbers variables to accomodate padded variables
  pub fn pad(&self) -> Self {
    // equalize the number of variables and constraints
    let m = max(self.num_vars, self.num_cons).next_power_of_two();

    // check if the provided R1CSShape is already as required
    if self.num_vars == m && self.num_cons == m {
      return self.clone();
    }

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.num_vars == m {
      return CCSShape {
        num_cons: m,
        num_vars: m,
        num_io: self.num_io,
        M: self.M.clone(),
        t: self.t,
        q: self.q,
        d: self.d,
        S: self.S.clone(),
        c: self.c.clone(),
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = m;
    let num_cons_padded = m;
    let apply_pad = |M: &[(usize, usize, G::Scalar)]| -> Vec<(usize, usize, G::Scalar)> {
      M.par_iter()
        .map(|(r, c, v)| {
          (
            *r,
            if c >= &self.num_vars {
              c + num_vars_padded - self.num_vars
            } else {
              *c
            },
            *v,
          )
        })
        .collect::<Vec<_>>()
    };

    // Apply pad for each matrix in M
    let M_padded = self.M.iter().map(|m| apply_pad(m)).collect::<Vec<_>>();

    // XXX: Check if CCS padding is correct here
    CCSShape {
      num_cons: num_cons_padded,
      num_vars: num_vars_padded,
      num_io: self.num_io,
      M: M_padded,
      t: self.t,
      q: self.q,
      d: self.d,
      S: self.S.clone(),
      c: self.c.clone(),
    }
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
