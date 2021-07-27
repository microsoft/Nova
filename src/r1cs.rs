#![allow(clippy::type_complexity)]
use super::commitments::Scalar;
use super::commitments::{CommitGens, CommitTrait, Commitment, CompressedCommitment};
use super::errors::NovaError;
use itertools::concat;
use rayon::prelude::*;

pub struct R1CSGens {
  gens_W: CommitGens,
  gens_E: CommitGens,
}

#[derive(Debug)]
pub struct R1CSShape {
  num_cons: usize,
  num_vars: usize,
  num_inputs: usize,
  A: Vec<(usize, usize, Scalar)>,
  B: Vec<(usize, usize, Scalar)>,
  C: Vec<(usize, usize, Scalar)>,
}

#[derive(Clone, Debug)]
pub struct R1CSWitness {
  W: Vec<Scalar>,
  E: Vec<Scalar>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct R1CSInstance {
  comm_W: Commitment,
  comm_E: Commitment,
  X: Vec<Scalar>,
  u: Scalar,
}

impl R1CSGens {
  pub fn new(num_cons: usize, num_vars: usize) -> R1CSGens {
    // generators to commit to witness vector `W`
    let gens_W = CommitGens::new(b"gens_W", num_vars);

    // generators to commit to the error/slack vector `E`
    let gens_E = CommitGens::new(b"gens_E", num_cons);

    R1CSGens { gens_E, gens_W }
  }
}

impl R1CSShape {
  pub fn new(
    num_cons: usize,
    num_vars: usize,
    num_inputs: usize,
    A: &[(usize, usize, Scalar)],
    B: &[(usize, usize, Scalar)],
    C: &[(usize, usize, Scalar)],
  ) -> Result<R1CSShape, NovaError> {
    let is_valid = |num_cons: usize,
                    num_vars: usize,
                    num_io: usize,
                    M: &[(usize, usize, Scalar)]|
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
    z: &[Scalar],
  ) -> Result<(Vec<Scalar>, Vec<Scalar>, Vec<Scalar>), NovaError> {
    if z.len() != self.num_inputs + self.num_vars + 1 {
      return Err(NovaError::InvalidWitnessLength);
    }

    // computes a product between a sparse matrix `M` and a vector `z`
    // This does not perform any validation of entries in M (e.g., if entries in `M` reference indexes outside the range of `z`)
    // This is safe since we know that `M` is valid
    let sparse_matrix_vec_product =
      |M: &Vec<(usize, usize, Scalar)>, num_rows: usize, z: &[Scalar]| -> Vec<Scalar> {
        (0..M.len())
          .map(|i| {
            let (row, col, val) = M[i];
            (row, val * z[col])
          })
          .fold(vec![Scalar::zero(); num_rows], |mut Mz, (r, v)| {
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
    gens: &R1CSGens,
    U: &R1CSInstance,
    W: &R1CSWitness,
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
    gens: &R1CSGens,
    U1: &R1CSInstance,
    W1: &R1CSWitness,
    U2: &R1CSInstance,
    W2: &R1CSWitness,
  ) -> Result<(Vec<Scalar>, CompressedCommitment), NovaError> {
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
      .collect::<Vec<Scalar>>();
    let AZ_2_circ_BZ_1 = (0..AZ_2.len())
      .map(|i| AZ_2[i] * BZ_1[i])
      .collect::<Vec<Scalar>>();
    let u_1_cdot_CZ_2 = (0..CZ_2.len())
      .map(|i| U1.u * CZ_2[i])
      .collect::<Vec<Scalar>>();
    let u_2_cdot_CZ_1 = (0..CZ_1.len())
      .map(|i| U2.u * CZ_1[i])
      .collect::<Vec<Scalar>>();

    let T = AZ_1_circ_BZ_2
      .par_iter()
      .zip(&AZ_2_circ_BZ_1)
      .zip(&u_1_cdot_CZ_2)
      .zip(&u_2_cdot_CZ_1)
      .map(|(((a, b), c), d)| a + b - c - d)
      .collect::<Vec<Scalar>>();

    let comm_T = T.commit(&gens.gens_E).compress();

    Ok((T, comm_T))
  }
}

impl R1CSWitness {
  pub fn new(S: &R1CSShape, W: &[Scalar], E: &[Scalar]) -> Result<R1CSWitness, NovaError> {
    if S.num_vars != W.len() || S.num_cons != E.len() {
      Err(NovaError::InvalidWitnessLength)
    } else {
      Ok(R1CSWitness {
        W: W.to_owned(),
        E: E.to_owned(),
      })
    }
  }

  pub fn commit(&self, gens: &R1CSGens) -> (Commitment, Commitment) {
    (self.W.commit(&gens.gens_W), self.E.commit(&gens.gens_E))
  }

  pub fn fold(&self, W2: &R1CSWitness, T: &[Scalar], r: &Scalar) -> Result<R1CSWitness, NovaError> {
    let (W1, E1) = (&self.W, &self.E);
    let (W2, E2) = (&W2.W, &W2.E);

    if W1.len() != W2.len() {
      return Err(NovaError::InvalidWitnessLength);
    }

    let W = W1
      .par_iter()
      .zip(W2)
      .map(|(a, b)| a + r * b)
      .collect::<Vec<Scalar>>();
    let E = E1
      .par_iter()
      .zip(T)
      .zip(E2)
      .map(|((a, b), c)| a + r * b + r * r * c)
      .collect::<Vec<Scalar>>();
    Ok(R1CSWitness { W, E })
  }
}

impl R1CSInstance {
  pub fn new(
    S: &R1CSShape,
    comm_W: &Commitment,
    comm_E: &Commitment,
    X: &[Scalar],
    u: &Scalar,
  ) -> Result<R1CSInstance, NovaError> {
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
    U2: &R1CSInstance,
    comm_T: &CompressedCommitment,
    r: &Scalar,
  ) -> Result<R1CSInstance, NovaError> {
    let comm_T_unwrapped = comm_T.decompress()?;
    let (X1, u1, comm_W_1, comm_E_1) =
      (&self.X, &self.u, &self.comm_W.clone(), &self.comm_E.clone());
    let (X2, u2, comm_W_2, comm_E_2) = (&U2.X, &U2.u, &U2.comm_W, &U2.comm_E);

    // weighted sum of X
    let X = X1
      .par_iter()
      .zip(X2)
      .map(|(a, b)| a + r * b)
      .collect::<Vec<Scalar>>();
    let comm_W = comm_W_1 + r * comm_W_2;
    let comm_E = comm_E_1 + r * comm_T_unwrapped + r * r * comm_E_2;
    let u = u1 + r * u2;

    Ok(R1CSInstance {
      comm_W,
      comm_E,
      X,
      u,
    })
  }
}
