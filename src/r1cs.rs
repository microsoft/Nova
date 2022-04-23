//! This module defines R1CS related types and a folding scheme for Relaxed R1CS
#![allow(clippy::type_complexity)]
use super::{
  commitments::{CommitGens, CommitTrait, Commitment, CompressedCommitment},
  errors::NovaError,
  traits::Group,
};
use ff::Field;
use itertools::concat;
use rayon::prelude::*;

/// Public parameters for a given R1CS
#[derive(Debug)]
pub struct R1CSGens<G: Group> {
  pub(crate) gens_W: CommitGens<G>, // TODO: avoid pub(crate)
  pub(crate) gens_E: CommitGens<G>,
}

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSShape<G: Group> {
  num_cons: usize,
  num_vars: usize,
  num_io: usize,
  A: Vec<(usize, usize, G::Scalar)>,
  B: Vec<(usize, usize, G::Scalar)>,
  C: Vec<(usize, usize, G::Scalar)>,
}

/// A type that holds a witness for a given R1CS instance
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSWitness<G: Group> {
  W: Vec<G::Scalar>,
}

/// A type that holds an R1CS instance
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSInstance<G: Group> {
  pub(crate) comm_W: Commitment<G>,
  pub(crate) X: Vec<G::Scalar>,
}

/// A type that holds a witness for a given Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedR1CSWitness<G: Group> {
  W: Vec<G::Scalar>,
  E: Vec<G::Scalar>,
}

/// A type that holds a Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedR1CSInstance<G: Group> {
  pub(crate) comm_W: Commitment<G>,
  pub(crate) comm_E: Commitment<G>,
  pub(crate) X: Vec<G::Scalar>,
  pub(crate) u: G::Scalar,
  Y_last: Vec<G::Scalar>, // output of the last instance that was folded
  counter: usize,         // the number of folds thus far
}

impl<G: Group> R1CSGens<G> {
  /// Samples public parameters for the specified number of constraints and variables in an R1CS
  pub fn new(num_cons: usize, num_vars: usize) -> R1CSGens<G> {
    // generators to commit to witness vector `W`
    let gens_W = CommitGens::new(b"gens_W", num_vars);

    // generators to commit to the error/slack vector `E`
    let gens_E = CommitGens::new(b"gens_E", num_cons);

    R1CSGens { gens_E, gens_W }
  }
}

impl<G: Group> R1CSShape<G> {
  /// Create an object of type `R1CSShape` from the explicitly specified R1CS matrices
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: &[(usize, usize, G::Scalar)],
    B: &[(usize, usize, G::Scalar)],
    C: &[(usize, usize, G::Scalar)],
  ) -> Result<R1CSShape<G>, NovaError> {
    let is_valid = |num_cons: usize,
                    num_vars: usize,
                    num_io: usize,
                    M: &[(usize, usize, G::Scalar)]|
     -> Result<(), NovaError> {
      let res = (0..M.len())
        .map(|i| {
          let (row, col, _val) = M[i];
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

    let res_A = is_valid(num_cons, num_vars, num_io, A);
    let res_B = is_valid(num_cons, num_vars, num_io, B);
    let res_C = is_valid(num_cons, num_vars, num_io, C);

    if res_A.is_err() || res_B.is_err() || res_C.is_err() {
      return Err(NovaError::InvalidIndex);
    }

    // We require the number of public inputs/outputs to be even
    if num_io % 2 != 0 {
      return Err(NovaError::OddInputLength);
    }

    let shape = R1CSShape {
      num_cons,
      num_vars,
      num_io,
      A: A.to_owned(),
      B: B.to_owned(),
      C: C.to_owned(),
    };

    Ok(shape)
  }

  fn multiply_vec(
    &self,
    z: &[G::Scalar],
  ) -> Result<(Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>), NovaError> {
    if z.len() != self.num_io + self.num_vars + 1 {
      return Err(NovaError::InvalidWitnessLength);
    }

    // computes a product between a sparse matrix `M` and a vector `z`
    // This does not perform any validation of entries in M (e.g., if entries in `M` reference indexes outside the range of `z`)
    // This is safe since we know that `M` is valid
    let sparse_matrix_vec_product =
      |M: &Vec<(usize, usize, G::Scalar)>, num_rows: usize, z: &[G::Scalar]| -> Vec<G::Scalar> {
        (0..M.len())
          .map(|i| {
            let (row, col, val) = M[i];
            (row, val * z[col])
          })
          .fold(vec![G::Scalar::zero(); num_rows], |mut Mz, (r, v)| {
            Mz[r] += v;
            Mz
          })
      };

    let Az = sparse_matrix_vec_product(&self.A, self.num_cons, z);
    let Bz = sparse_matrix_vec_product(&self.B, self.num_cons, z);
    let Cz = sparse_matrix_vec_product(&self.C, self.num_cons, z);

    Ok((Az, Bz, Cz))
  }

  /// Checks if the Relaxed R1CS instance is satisfiable given a witness and its shape
  pub fn is_sat_relaxed(
    &self,
    gens: &R1CSGens<G>,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<(), NovaError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(W.E.len(), self.num_cons);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz = u*Cz + E
    let res_eq: bool = {
      let z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);
      let (Az, Bz, Cz) = self.multiply_vec(&z)?;
      assert_eq!(Az.len(), self.num_cons);
      assert_eq!(Bz.len(), self.num_cons);
      assert_eq!(Cz.len(), self.num_cons);

      let res: usize = (0..self.num_cons)
        .map(|i| {
          if Az[i] * Bz[i] == U.u * Cz[i] + W.E[i] {
            0
          } else {
            1
          }
        })
        .sum();

      res == 0
    };

    // verify if comm_E and comm_W are commitments to E and W
    let res_comm: bool = {
      let comm_W = W.W.commit(&gens.gens_W);
      let comm_E = W.E.commit(&gens.gens_E);

      U.comm_W == comm_W && U.comm_E == comm_E
    };

    if res_eq && res_comm {
      Ok(())
    } else {
      Err(NovaError::UnSat)
    }
  }

  /// Checks if the R1CS instance is satisfiable given a witness and its shape
  pub fn is_sat(
    &self,
    gens: &R1CSGens<G>,
    U: &R1CSInstance<G>,
    W: &R1CSWitness<G>,
  ) -> Result<(), NovaError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(U.X.len(), self.num_io);

    // verify if Az * Bz = u*Cz
    let res_eq: bool = {
      let z = concat(vec![W.W.clone(), vec![G::Scalar::one()], U.X.clone()]);
      let (Az, Bz, Cz) = self.multiply_vec(&z)?;
      assert_eq!(Az.len(), self.num_cons);
      assert_eq!(Bz.len(), self.num_cons);
      assert_eq!(Cz.len(), self.num_cons);

      let res: usize = (0..self.num_cons)
        .map(|i| if Az[i] * Bz[i] == Cz[i] { 0 } else { 1 })
        .sum();

      res == 0
    };

    // verify if comm_W is a commitment to W
    let res_comm: bool = U.comm_W == W.W.commit(&gens.gens_W);

    if res_eq && res_comm {
      Ok(())
    } else {
      Err(NovaError::UnSat)
    }
  }

  /// A method to compute a commitment to the cross-term `T` given a
  /// Relaxed R1CS instance-witness pair and an R1CS instance-witness pair
  pub fn commit_T(
    &self,
    gens: &R1CSGens<G>,
    U1: &RelaxedR1CSInstance<G>,
    W1: &RelaxedR1CSWitness<G>,
    U2: &R1CSInstance<G>,
    W2: &R1CSWitness<G>,
  ) -> Result<
    (
      Vec<G::Scalar>,
      CompressedCommitment<G::CompressedGroupElement>,
    ),
    NovaError,
  > {
    let (AZ_1, BZ_1, CZ_1) = {
      let Z1 = concat(vec![W1.W.clone(), vec![U1.u], U1.X.clone()]);
      self.multiply_vec(&Z1)?
    };

    let (AZ_2, BZ_2, CZ_2) = {
      let Z2 = concat(vec![W2.W.clone(), vec![G::Scalar::one()], U2.X.clone()]);
      self.multiply_vec(&Z2)?
    };

    let AZ_1_circ_BZ_2 = (0..AZ_1.len())
      .map(|i| AZ_1[i] * BZ_2[i])
      .collect::<Vec<G::Scalar>>();
    let AZ_2_circ_BZ_1 = (0..AZ_2.len())
      .map(|i| AZ_2[i] * BZ_1[i])
      .collect::<Vec<G::Scalar>>();
    let u_1_cdot_CZ_2 = (0..CZ_2.len())
      .map(|i| U1.u * CZ_2[i])
      .collect::<Vec<G::Scalar>>();
    let u_2_cdot_CZ_1 = (0..CZ_1.len()).map(|i| CZ_1[i]).collect::<Vec<G::Scalar>>();

    let T = AZ_1_circ_BZ_2
      .par_iter()
      .zip(&AZ_2_circ_BZ_1)
      .zip(&u_1_cdot_CZ_2)
      .zip(&u_2_cdot_CZ_1)
      .map(|(((a, b), c), d)| *a + *b - *c - *d)
      .collect::<Vec<G::Scalar>>();

    let comm_T = T.commit(&gens.gens_E).compress();

    Ok((T, comm_T))
  }
}

impl<G: Group> R1CSWitness<G> {
  /// A method to create a witness object using a vector of scalars
  pub fn new(S: &R1CSShape<G>, W: &[G::Scalar]) -> Result<R1CSWitness<G>, NovaError> {
    if S.num_vars != W.len() {
      Err(NovaError::InvalidWitnessLength)
    } else {
      Ok(R1CSWitness { W: W.to_owned() })
    }
  }

  /// Commits to the witness using the supplied generators
  pub fn commit(&self, gens: &R1CSGens<G>) -> Commitment<G> {
    self.W.commit(&gens.gens_W)
  }
}

impl<G: Group> R1CSInstance<G> {
  /// A method to create an instance object using consitituent elements
  pub fn new(
    S: &R1CSShape<G>,
    comm_W: &Commitment<G>,
    X: &[G::Scalar],
  ) -> Result<R1CSInstance<G>, NovaError> {
    if S.num_io != X.len() {
      Err(NovaError::InvalidInputLength)
    } else {
      Ok(R1CSInstance {
        comm_W: *comm_W,
        X: X.to_owned(),
      })
    }
  }
}

impl<G: Group> RelaxedR1CSWitness<G> {
  /// Produces a default RelaxedR1CSWitness given an R1CSShape
  pub fn default(S: &R1CSShape<G>) -> RelaxedR1CSWitness<G> {
    RelaxedR1CSWitness {
      W: vec![G::Scalar::zero(); S.num_vars],
      E: vec![G::Scalar::zero(); S.num_cons],
    }
  }

  /// Commits to the witness using the supplied generators
  pub fn commit(&self, gens: &R1CSGens<G>) -> (Commitment<G>, Commitment<G>) {
    (self.W.commit(&gens.gens_W), self.E.commit(&gens.gens_E))
  }

  /// Folds an incoming R1CSWitness into the current one
  pub fn fold(
    &self,
    W2: &R1CSWitness<G>,
    T: &[G::Scalar],
    r: &G::Scalar,
  ) -> Result<RelaxedR1CSWitness<G>, NovaError> {
    let (W1, E1) = (&self.W, &self.E);
    let W2 = &W2.W;

    if W1.len() != W2.len() {
      return Err(NovaError::InvalidWitnessLength);
    }

    let W = W1
      .par_iter()
      .zip(W2)
      .map(|(a, b)| *a + *r * *b)
      .collect::<Vec<G::Scalar>>();
    let E = E1
      .par_iter()
      .zip(T)
      .map(|(a, b)| *a + *r * *b)
      .collect::<Vec<G::Scalar>>();
    Ok(RelaxedR1CSWitness { W, E })
  }
}

impl<G: Group> RelaxedR1CSInstance<G> {
  /// Produces a default RelaxedR1CSInstance given R1CSGens and R1CSShape
  pub fn default(gens: &R1CSGens<G>, S: &R1CSShape<G>) -> RelaxedR1CSInstance<G> {
    let (comm_W, comm_E) = RelaxedR1CSWitness::default(S).commit(gens);
    RelaxedR1CSInstance {
      comm_W,
      comm_E,
      u: G::Scalar::zero(),
      X: vec![G::Scalar::zero(); S.num_io],
      Y_last: vec![G::Scalar::zero(); S.num_io / 2],
      counter: 0,
    }
  }

  /// Folds an incoming RelaxedR1CSInstance into the current one
  pub fn fold(
    &self,
    U2: &R1CSInstance<G>,
    comm_T: &CompressedCommitment<G::CompressedGroupElement>,
    r: &G::Scalar,
  ) -> Result<RelaxedR1CSInstance<G>, NovaError> {
    let comm_T_unwrapped = comm_T.decompress()?;
    let (X1, u1, comm_W_1, comm_E_1) =
      (&self.X, &self.u, &self.comm_W.clone(), &self.comm_E.clone());
    let (X2, comm_W_2) = (&U2.X, &U2.comm_W);

    // check if the input of the incoming instance matches the output
    // of the incremental computation thus far if counter > 0
    if self.counter > 0 {
      if self.Y_last.len() != U2.X.len() / 2 {
        return Err(NovaError::InvalidInputLength);
      }
      for i in 0..self.Y_last.len() {
        if self.Y_last[i] != U2.X[i] {
          return Err(NovaError::InputOutputMismatch);
        }
      }
    }

    // weighted sum of X, comm_W, comm_E, and u
    let X = X1
      .par_iter()
      .zip(X2)
      .map(|(a, b)| *a + *r * *b)
      .collect::<Vec<G::Scalar>>();
    let comm_W = comm_W_1 + comm_W_2 * r;
    let comm_E = *comm_E_1 + comm_T_unwrapped * *r;
    let u = *u1 + *r;

    Ok(RelaxedR1CSInstance {
      comm_W,
      comm_E,
      X,
      u,
      Y_last: U2.X[U2.X.len() / 2..].to_owned(),
      counter: self.counter + 1,
    })
  }
}
