#![allow(clippy::type_complexity)]
use super::commitments::{CommitGens, CommitTrait, Commitment};
use super::errors::NovaError;
use super::traits::{Group, PrimeField};
use itertools::concat;
use rayon::prelude::*;

pub struct R1CSGens<G: Group> {
  gens_W: CommitGens<G>,
  gens_E: CommitGens<G>,
}

#[derive(Debug)]
pub struct R1CSShape<G: Group> {
  num_cons: usize,
  num_vars: usize,
  num_inputs: usize,
  A: Vec<(usize, usize, G::Scalar)>,
  B: Vec<(usize, usize, G::Scalar)>,
  C: Vec<(usize, usize, G::Scalar)>,
}

#[derive(Clone, Debug)]
pub struct R1CSWitness<G: Group> {
  W: Vec<G::Scalar>,
  E: Vec<G::Scalar>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSInstance<G: Group> {
  comm_W: Commitment<G>,
  comm_E: Commitment<G>,
  X: Vec<G::Scalar>,
  u: G::Scalar,
}

impl<G: Group> R1CSGens<G> {
  pub fn new(num_cons: usize, num_vars: usize) -> R1CSGens<G> {
    // generators to commit to witness vector `W`
    let gens_W = CommitGens::new(b"gens_W", num_vars);

    // generators to commit to the error/slack vector `E`
    let gens_E = CommitGens::new(b"gens_E", num_cons);

    R1CSGens { gens_E, gens_W }
  }
}

impl<G: Group> R1CSShape<G> {
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_inputs: usize,
    A: &[(usize, usize, G::Scalar)],
    B: &[(usize, usize, G::Scalar)],
    C: &[(usize, usize, G::Scalar)],
  ) -> Result<R1CSShape<G>, NovaError> {
    let is_valid = |num_cons: usize,
                    num_vars: usize,
                    num_io: usize,
                    M: &[(usize, usize, G::Scalar)]|
     -> Result<(), NovaError> {
      let res = (0..num_cons)
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

    let res_A = is_valid(num_cons, num_vars, num_inputs, A);
    let res_B = is_valid(num_cons, num_vars, num_inputs, B);
    let res_C = is_valid(num_cons, num_vars, num_inputs, C);

    if res_A.is_err() || res_B.is_err() || res_C.is_err() {
      return Err(NovaError::InvalidIndex);
    }

    let shape = R1CSShape {
      num_cons,
      num_vars,
      num_inputs,
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
    if z.len() != self.num_inputs + self.num_vars + 1 {
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

    let Az = sparse_matrix_vec_product(&self.A, self.num_cons, &z);
    let Bz = sparse_matrix_vec_product(&self.B, self.num_cons, &z);
    let Cz = sparse_matrix_vec_product(&self.C, self.num_cons, &z);

    Ok((Az, Bz, Cz))
  }

  pub fn is_sat(
    &self,
    gens: &R1CSGens<G>,
    U: &R1CSInstance<G>,
    W: &R1CSWitness<G>,
  ) -> Result<(), NovaError> {
    assert_eq!(W.W.len(), self.num_vars);
    assert_eq!(W.E.len(), self.num_cons);
    assert_eq!(U.X.len(), self.num_inputs);

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

  pub fn commit_T(
    &self,
    gens: &R1CSGens<G>,
    U1: &R1CSInstance<G>,
    W1: &R1CSWitness<G>,
    U2: &R1CSInstance<G>,
    W2: &R1CSWitness<G>,
  ) -> Result<(Vec<G::Scalar>, Commitment<G>), NovaError> {
    let (AZ_1, BZ_1, CZ_1) = {
      let Z1 = concat(vec![W1.W.clone(), vec![U1.u], U1.X.clone()]);
      self.multiply_vec(&Z1)?
    };

    let (AZ_2, BZ_2, CZ_2) = {
      let Z2 = concat(vec![W2.W.clone(), vec![U2.u], U2.X.clone()]);
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
    let u_2_cdot_CZ_1 = (0..CZ_1.len())
      .map(|i| U2.u * CZ_1[i])
      .collect::<Vec<G::Scalar>>();

    let T = AZ_1_circ_BZ_2
      .par_iter()
      .zip(&AZ_2_circ_BZ_1)
      .zip(&u_1_cdot_CZ_2)
      .zip(&u_2_cdot_CZ_1)
      .map(|(((a, b), c), d)| *a + *b - *c - *d)
      .collect::<Vec<G::Scalar>>();

    let comm_T = T.commit(&gens.gens_E);

    Ok((T, comm_T))
  }
}

impl<G: Group> R1CSWitness<G> {
  pub fn new(
    S: &R1CSShape<G>,
    W: &[G::Scalar],
    E: &[G::Scalar],
  ) -> Result<R1CSWitness<G>, NovaError> {
    if S.num_vars != W.len() || S.num_cons != E.len() {
      Err(NovaError::InvalidWitnessLength)
    } else {
      Ok(R1CSWitness {
        W: W.to_owned(),
        E: E.to_owned(),
      })
    }
  }

  pub fn commit(&self, gens: &R1CSGens<G>) -> (Commitment<G>, Commitment<G>) {
    (self.W.commit(&gens.gens_W), self.E.commit(&gens.gens_E))
  }

  pub fn fold(
    &self,
    W2: &R1CSWitness<G>,
    T: &[G::Scalar],
    r: &G::Scalar,
  ) -> Result<R1CSWitness<G>, NovaError> {
    let (W1, E1) = (&self.W, &self.E);
    let (W2, E2) = (&W2.W, &W2.E);

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
      .zip(E2)
      .map(|((a, b), c)| *a + *r * *b + *r * *r * *c)
      .collect::<Vec<G::Scalar>>();
    Ok(R1CSWitness { W, E })
  }
}

impl<G: Group> R1CSInstance<G> {
  pub fn new(
    S: &R1CSShape<G>,
    comm_W: &Commitment<G>,
    comm_E: &Commitment<G>,
    X: &[G::Scalar],
    u: &G::Scalar,
  ) -> Result<R1CSInstance<G>, NovaError> {
    if S.num_inputs != X.len() {
      Err(NovaError::InvalidInputLength)
    } else {
      Ok(R1CSInstance {
        comm_W: comm_W.clone(),
        comm_E: comm_E.clone(),
        X: X.to_owned(),
        u: *u,
      })
    }
  }

  pub fn fold(
    &self,
    U2: &R1CSInstance<G>,
    comm_T: &Commitment<G>,
    r: &G::Scalar,
  ) -> R1CSInstance<G> {
    let (X1, u1, comm_W_1, comm_E_1) =
      (&self.X, &self.u, &self.comm_W.clone(), &self.comm_E.clone());
    let (X2, u2, comm_W_2, comm_E_2) = (&U2.X, &U2.u, &U2.comm_W, &U2.comm_E);

    // weighted sum of X
    let X = X1
      .par_iter()
      .zip(X2)
      .map(|(a, b)| *a + *r * *b)
      .collect::<Vec<G::Scalar>>();
    let comm_W = comm_W_1 + *comm_W_2 * *r;
    let comm_E = *comm_E_1 + *comm_T * *r + *comm_E_2 * *r * *r;
    let u = *u1 + *r * *u2;

    R1CSInstance {
      comm_W,
      comm_E,
      X,
      u,
    }
  }
}
