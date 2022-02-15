//! This module defines R1CS related types and a folding scheme for Relaxed R1CS
#![allow(clippy::type_complexity)]
use super::commitments::{
  AppendToTranscriptTrait, CommitGens, CommitTrait, Commitment, CompressedCommitment,
};
use super::errors::NovaError;
use super::ipa::{
  FinalInnerProductArgument, FinalInnerProductArgumentAux, InnerProductInstance,
  InnerProductWitness, StepInnerProductArgument,
};
use super::traits::{ChallengeTrait, Group, PrimeField};
use core::cmp::max;
use itertools::concat;
use merlin::Transcript;
use rayon::prelude::*;

/// Public parameters for a given R1CS
#[derive(Debug)]
pub struct R1CSGens<G: Group> {
  gens: CommitGens<G>,
  gens_aux: CommitGens<G>,
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
  comm_W: Commitment<G>,
  X: Vec<G::Scalar>,
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
    // generators to commit to witness vector `W` and slack vector `E`
    let gens = CommitGens::new(b"gens", max(num_vars, num_cons));

    // generators to commit to auxiliary vectors
    let gens_aux = CommitGens::new(b"gens_aux", max(num_vars, num_cons));

    R1CSGens { gens, gens_aux }
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

  fn multiply_left(
    &self,
    r: &[G::Scalar],
  ) -> Result<(Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>), NovaError> {
    if r.len() != self.num_cons {
      return Err(NovaError::InvalidRandomVectorLength);
    }

    // computes a product between a random vector `r` and a sparse matrix `M`
    // This does not perform any validation of entries in M (e.g., if entries in `M` reference indexes outside the range of `r`)
    // This is safe since we know that `M` is valid
    let vec_sparse_matrix_product =
      |r: &[G::Scalar], M: &Vec<(usize, usize, G::Scalar)>, num_cols| -> Vec<G::Scalar> {
        (0..M.len())
          .map(|i| {
            let (row, col, val) = M[i];
            (col, r[row] * val)
          })
          .fold(vec![G::Scalar::zero(); num_cols], |mut rM, (c, v)| {
            rM[c] += v;
            rM
          })
      };

    let rA = vec_sparse_matrix_product(r, &self.A, self.num_vars + 1 + self.num_io);
    let rB = vec_sparse_matrix_product(r, &self.B, self.num_vars + 1 + self.num_io);
    let rC = vec_sparse_matrix_product(r, &self.C, self.num_vars + 1 + self.num_io);

    Ok((rA, rB, rC))
  }

  fn multiply_right(
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
      let (Az, Bz, Cz) = self.multiply_right(&z)?;
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
      let comm_W = W.W.commit(&gens.gens);
      let comm_E = W.E.commit(&gens.gens);

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
      let (Az, Bz, Cz) = self.multiply_right(&z)?;
      assert_eq!(Az.len(), self.num_cons);
      assert_eq!(Bz.len(), self.num_cons);
      assert_eq!(Cz.len(), self.num_cons);

      let res: usize = (0..self.num_cons)
        .map(|i| if Az[i] * Bz[i] == Cz[i] { 0 } else { 1 })
        .sum();

      res == 0
    };

    // verify if comm_W is a commitment to W
    let res_comm: bool = U.comm_W == W.W.commit(&gens.gens);

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
      self.multiply_right(&Z1)?
    };

    let (AZ_2, BZ_2, CZ_2) = {
      let Z2 = concat(vec![W2.W.clone(), vec![G::Scalar::one()], U2.X.clone()]);
      self.multiply_right(&Z2)?
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

    let comm_T = T.commit(&gens.gens).compress();

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
    self.W.commit(&gens.gens)
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
    (self.W.commit(&gens.gens), self.E.commit(&gens.gens))
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

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
pub struct RelaxedR1CSSNARK<G: Group> {
  comm_Az: CompressedCommitment<G::CompressedGroupElement>,
  comm_Bz: CompressedCommitment<G::CompressedGroupElement>,
  comm_Cz: CompressedCommitment<G::CompressedGroupElement>,
  comm_Az_circ_tau: CompressedCommitment<G::CompressedGroupElement>,

  // rowcheck IPAs
  ip_Az_circ_tau__Bz: G::Scalar,
  ip_Cz_tau: G::Scalar,
  ip_E_tau: G::Scalar,
  ip_Az_circ_tau__rho: G::Scalar,

  ipa_Az_circ_tau__Bz: FinalInnerProductArgumentAux<G>, // TODO: fold this with others
  ipa_Cz_tau_and_E_tau: StepInnerProductArgument<G>, // a single step IPA for ipa_Cz_tau ipa_E_tau
  ipa_Az_circ_tau__rho: FinalInnerProductArgument<G>, // TODO: fold this with others
  ipa_Az__rho_circ_tau: StepInnerProductArgument<G>,

  // lincheck IPAs
  ip_rA_w: G::Scalar,
  ip_rB_w: G::Scalar,
  ip_rC_w: G::Scalar,
  ipa_rA_w: StepInnerProductArgument<G>,
  ipa_rB_w: StepInnerProductArgument<G>,
  ipa_rC_w: StepInnerProductArgument<G>,
  ipa_r_Az: StepInnerProductArgument<G>,
  ipa_r_Bz: StepInnerProductArgument<G>,
  ipa_r_Cz: StepInnerProductArgument<G>,

  // final IPA for all step IPAs
  final_ipa: FinalInnerProductArgument<G>,
}

impl<G: Group> RelaxedR1CSSNARK<G> {
  fn protocol_name() -> &'static [u8] {
    b"RelaxedR1CSSNARK"
  }

  /// produces a succinct proof of satisfiability of a RelaxedR1CS instance
  pub fn prove(
    gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
    transcript: &mut Transcript,
  ) -> Result<Self, NovaError> {
    // append the protocol name to the transcript
    transcript.append_message(b"protocol-name", RelaxedR1CSSNARK::<G>::protocol_name());

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);

    // commit to Az, Bz, and Cz vectors
    let (Az, comm_Az, Bz, comm_Bz, Cz, comm_Cz) = {
      let (Az, Bz, Cz) = S.multiply_right(&z)?;
      assert_eq!(Az.len(), S.num_cons);
      assert_eq!(Bz.len(), S.num_cons);
      assert_eq!(Cz.len(), S.num_cons);

      let comm_Az = Az.commit(&gens.gens).compress();
      let comm_Bz = Bz.commit(&gens.gens).compress();
      let comm_Cz = Cz.commit(&gens.gens).compress();

      (Az, comm_Az, Bz, comm_Bz, Cz, comm_Cz)
    };

    // append the commitments to the transcript
    comm_Az.append_to_transcript(b"comm_Az", transcript);
    comm_Bz.append_to_transcript(b"comm_Bz", transcript);
    comm_Cz.append_to_transcript(b"comm_Cz", transcript);

    // (1) Row check
    // produce a challenge vector for the Hadamard product check
    let tau = (0..S.num_cons)
      .map(|_| G::Scalar::challenge(b"tau", transcript))
      .collect::<Vec<G::Scalar>>();

    // compute and commit to Az \circ tau
    let Az_circ_tau = (0..Az.len())
      .map(|i| Az[i] * tau[i])
      .collect::<Vec<G::Scalar>>();
    let comm_Az_circ_tau = Az_circ_tau.commit(&gens.gens_aux).compress();
    comm_Az_circ_tau.append_to_transcript(b"comm_Az_circ_tau", transcript);

    let ip_Az_circ_tau__Bz = FinalInnerProductArgument::<G>::inner_product(&Az_circ_tau, &Bz);
    let ip_Cz_tau = FinalInnerProductArgument::<G>::inner_product(&Cz, &tau);
    let ip_E_tau = FinalInnerProductArgument::<G>::inner_product(&W.E, &tau);
    assert_eq!(ip_Az_circ_tau__Bz, U.u * ip_Cz_tau + ip_E_tau); // check that the Hadamard product check actually holds

    let ipa_Az_circ_tau__Bz = FinalInnerProductArgumentAux::prove(
      &comm_Bz.decompress()?,
      &comm_Az_circ_tau.decompress()?,
      &ip_Az_circ_tau__Bz,
      &Bz,
      &Az_circ_tau,
      &gens.gens,
      &gens.gens_aux,
      transcript,
    )?;

    let (ipa_Cz_tau_and_E_tau, r_U, r_W) = StepInnerProductArgument::prove(
      &InnerProductInstance::new(&comm_Cz, &tau, &ip_Cz_tau)?,
      &InnerProductWitness::new(&Cz),
      &InnerProductInstance::new(&U.comm_E.compress(), &tau, &ip_E_tau).unwrap(),
      &InnerProductWitness::new(&W.E),
      transcript,
    );

    // check if Az_circ_tau = Az \circ tau
    let rho = (0..S.num_cons)
      .map(|_| G::Scalar::challenge(b"tau", transcript))
      .collect::<Vec<G::Scalar>>();
    let ip_Az_circ_tau__rho = FinalInnerProductArgument::<G>::inner_product(&Az_circ_tau, &rho);

    let rho_circ_tau = (0..rho.len())
      .map(|i| rho[i] * tau[i])
      .collect::<Vec<G::Scalar>>();
    let ip_Az_rho_circ_tau = FinalInnerProductArgument::<G>::inner_product(&Az, &rho_circ_tau);
    assert_eq!(ip_Az_circ_tau__rho, ip_Az_rho_circ_tau);

    let ipa_Az_circ_tau__rho = FinalInnerProductArgument::prove(
      &InnerProductInstance::new(&comm_Az_circ_tau, &rho, &ip_Az_circ_tau__rho).unwrap(),
      &InnerProductWitness::new(&Az_circ_tau),
      &gens.gens_aux,
      transcript,
    )?;

    let (ipa_Az__rho_circ_tau, r_U, r_W) = StepInnerProductArgument::prove(
      &r_U,
      &r_W,
      &InnerProductInstance::new(&comm_Az, &rho_circ_tau, &ip_Az_rho_circ_tau).unwrap(),
      &InnerProductWitness::new(&Az),
      transcript,
    );

    // (2) Lin Checks
    // produce a challenge vector of size `S.num_cons`
    let r = (0..S.num_cons)
      .map(|_| G::Scalar::challenge(b"r", transcript))
      .collect::<Vec<G::Scalar>>();

    // multiply R1CS matrices using r from the left
    let (rA, rB, rC) = S.multiply_left(&r)?;
    assert_eq!(rA.len(), S.num_vars + 1 + S.num_io);
    assert_eq!(rB.len(), S.num_vars + 1 + S.num_io);
    assert_eq!(rC.len(), S.num_vars + 1 + S.num_io);

    let ip_rA_w: G::Scalar =
      FinalInnerProductArgument::<G>::inner_product(&rA[0..S.num_vars], &W.W);
    let ip_rB_w: G::Scalar =
      FinalInnerProductArgument::<G>::inner_product(&rB[0..S.num_vars], &W.W);
    let ip_rC_w: G::Scalar =
      FinalInnerProductArgument::<G>::inner_product(&rC[0..S.num_vars], &W.W);

    let ip_r_Az: G::Scalar = FinalInnerProductArgument::<G>::inner_product(&r, &Az);
    let ip_r_Bz: G::Scalar = FinalInnerProductArgument::<G>::inner_product(&r, &Bz);
    let ip_r_Cz: G::Scalar = FinalInnerProductArgument::<G>::inner_product(&r, &Cz);

    let ip_rA_z: G::Scalar = FinalInnerProductArgument::<G>::inner_product(&rA, &z);
    let ip_rB_z: G::Scalar = FinalInnerProductArgument::<G>::inner_product(&rB, &z);
    let ip_rC_z: G::Scalar = FinalInnerProductArgument::<G>::inner_product(&rC, &z);

    // inner product relationships
    debug_assert_eq!(ip_r_Az, ip_rA_z);
    debug_assert_eq!(ip_r_Bz, ip_rB_z);
    debug_assert_eq!(ip_r_Cz, ip_rC_z);

    // prove the inner product relationships
    let (ipa_rA_w, r_U, r_W) = StepInnerProductArgument::prove(
      &r_U,
      &r_W,
      &InnerProductInstance::new(&U.comm_W.compress(), &rA[0..S.num_vars], &ip_rA_w).unwrap(),
      &InnerProductWitness::new(&W.W),
      transcript,
    );
    let (ipa_rB_w, r_U, r_W) = StepInnerProductArgument::prove(
      &r_U,
      &r_W,
      &InnerProductInstance::new(&U.comm_W.compress(), &rB[0..S.num_vars], &ip_rB_w).unwrap(),
      &InnerProductWitness::new(&W.W),
      transcript,
    );
    let (ipa_rC_w, r_U, r_W) = StepInnerProductArgument::prove(
      &r_U,
      &r_W,
      &InnerProductInstance::new(&U.comm_W.compress(), &rC[0..S.num_vars], &ip_rC_w).unwrap(),
      &InnerProductWitness::new(&W.W),
      transcript,
    );

    let (ipa_r_Az, r_U, r_W) = StepInnerProductArgument::prove(
      &r_U,
      &r_W,
      &InnerProductInstance::new(&comm_Az, &r, &ip_r_Az).unwrap(),
      &InnerProductWitness::new(&Az),
      transcript,
    );

    let (ipa_r_Bz, r_U, r_W) = StepInnerProductArgument::prove(
      &r_U,
      &r_W,
      &InnerProductInstance::new(&comm_Bz, &r, &ip_r_Bz).unwrap(),
      &InnerProductWitness::new(&Bz),
      transcript,
    );

    let (ipa_r_Cz, r_U, r_W) = StepInnerProductArgument::prove(
      &r_U,
      &r_W,
      &InnerProductInstance::new(&comm_Cz, &r, &ip_r_Cz).unwrap(),
      &InnerProductWitness::new(&Cz),
      transcript,
    );

    let final_ipa = FinalInnerProductArgument::prove(&r_U, &r_W, &gens.gens, transcript)?;

    Ok(RelaxedR1CSSNARK {
      comm_Az,
      comm_Bz,
      comm_Cz,
      ip_rA_w,
      ip_rB_w,
      ip_rC_w,
      ipa_rA_w,
      ipa_rB_w,
      ipa_rC_w,
      ipa_r_Az,
      ipa_r_Bz,
      ipa_r_Cz,
      comm_Az_circ_tau,
      ip_Az_circ_tau__Bz,
      ip_Cz_tau,
      ip_E_tau,
      ipa_Cz_tau_and_E_tau,
      ip_Az_circ_tau__rho, // we don't need to send ip_Az_rho_circ_tau
      ipa_Az_circ_tau__rho,
      ipa_Az__rho_circ_tau,
      final_ipa,
      ipa_Az_circ_tau__Bz,
    })
  }

  /// verifies a proof of satisfiability of a RelaxedR1CS instance
  pub fn verify(
    &self,
    gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    U: &RelaxedR1CSInstance<G>,
    transcript: &mut Transcript,
  ) -> Result<(), NovaError> {
    // append the protocol name to the transcript
    transcript.append_message(b"protocol-name", RelaxedR1CSSNARK::<G>::protocol_name());

    // append the commitments to the transcript
    self.comm_Az.append_to_transcript(b"comm_Az", transcript);
    self.comm_Bz.append_to_transcript(b"comm_Bz", transcript);
    self.comm_Cz.append_to_transcript(b"comm_Cz", transcript);

    // produce a challenge vector for the Hadamard product check
    let tau = (0..S.num_cons)
      .map(|_| G::Scalar::challenge(b"tau", transcript))
      .collect::<Vec<G::Scalar>>();
    self
      .comm_Az_circ_tau
      .append_to_transcript(b"comm_Az_circ_tau", transcript);
    if self.ip_Az_circ_tau__Bz != U.u * self.ip_Cz_tau + self.ip_E_tau {
      return Err(NovaError::HadamardCheckFailed);
    }

    self.ipa_Az_circ_tau__Bz.verify(
      S.num_cons,
      &self.comm_Bz.decompress()?,
      &self.comm_Az_circ_tau.decompress()?,
      &self.ip_Az_circ_tau__Bz,
      &gens.gens,
      &gens.gens_aux,
      transcript,
    )?;

    let r_U = self.ipa_Cz_tau_and_E_tau.verify(
      &InnerProductInstance::new(&self.comm_Cz, &tau, &self.ip_Cz_tau).unwrap(),
      &InnerProductInstance::new(&U.comm_E.compress(), &tau, &self.ip_E_tau).unwrap(),
      transcript,
    );

    // check if Az_circ_tau = Az \circ tau
    let rho = (0..S.num_cons)
      .map(|_| G::Scalar::challenge(b"tau", transcript))
      .collect::<Vec<G::Scalar>>();
    let rho_circ_tau = (0..S.num_cons)
      .map(|i| rho[i] * tau[i])
      .collect::<Vec<G::Scalar>>();

    self.ipa_Az_circ_tau__rho.verify(
      S.num_cons,
      &InnerProductInstance::new(&self.comm_Az_circ_tau, &rho, &self.ip_Az_circ_tau__rho).unwrap(),
      &gens.gens_aux,
      transcript,
    )?;

    let r_U = self.ipa_Az__rho_circ_tau.verify(
      &r_U,
      &InnerProductInstance::new(&self.comm_Az, &rho_circ_tau, &self.ip_Az_circ_tau__rho).unwrap(),
      transcript,
    );

    // produce a challenge vector of size `S.num_cons`
    let r = (0..S.num_cons)
      .map(|_| G::Scalar::challenge(b"r", transcript))
      .collect::<Vec<G::Scalar>>();

    // multiply R1CS matrices using r from the left
    let (rA, rB, rC) = S.multiply_left(&r)?;
    assert_eq!(rA.len(), S.num_vars + 1 + S.num_io);
    assert_eq!(rB.len(), S.num_vars + 1 + S.num_io);
    assert_eq!(rC.len(), S.num_vars + 1 + S.num_io);

    let (ip_rA_z, ip_rB_z, ip_rC_z) = {
      let uX = concat(vec![vec![U.u], U.X.clone()]);
      let ip_rA_z =
        self.ip_rA_w + FinalInnerProductArgument::<G>::inner_product(&rA[S.num_vars..], &uX);
      let ip_rB_z =
        self.ip_rB_w + FinalInnerProductArgument::<G>::inner_product(&rB[S.num_vars..], &uX);
      let ip_rC_z =
        self.ip_rC_w + FinalInnerProductArgument::<G>::inner_product(&rC[S.num_vars..], &uX);
      (ip_rA_z, ip_rB_z, ip_rC_z)
    };

    // verify the inner product relationships
    let r_U = self.ipa_rA_w.verify(
      &r_U,
      &InnerProductInstance::new(&U.comm_W.compress(), &rA[0..S.num_vars], &self.ip_rA_w).unwrap(),
      transcript,
    );

    let r_U = self.ipa_rB_w.verify(
      &r_U,
      &InnerProductInstance::new(&U.comm_W.compress(), &rB[0..S.num_vars], &self.ip_rB_w).unwrap(),
      transcript,
    );

    let r_U = self.ipa_rC_w.verify(
      &r_U,
      &InnerProductInstance::new(&U.comm_W.compress(), &rC[0..S.num_vars], &self.ip_rC_w).unwrap(),
      transcript,
    );

    let r_U = self.ipa_r_Az.verify(
      &r_U,
      &InnerProductInstance::new(&self.comm_Az, &r, &ip_rA_z).unwrap(),
      transcript,
    );

    let r_U = self.ipa_r_Bz.verify(
      &r_U,
      &InnerProductInstance::new(&self.comm_Bz, &r, &ip_rB_z).unwrap(),
      transcript,
    );

    let r_U = self.ipa_r_Cz.verify(
      &r_U,
      &InnerProductInstance::new(&self.comm_Cz, &r, &ip_rC_z).unwrap(),
      transcript,
    );

    // verify the final IPA proof
    self
      .final_ipa
      .verify(max(S.num_vars, S.num_cons), &r_U, &gens.gens, transcript)?;

    Ok(())
  }
}
