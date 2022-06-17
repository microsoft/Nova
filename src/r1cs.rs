//! This module defines R1CS related types and a folding scheme for Relaxed R1CS
#![allow(clippy::type_complexity)]
use super::{
  commitments::{CommitGens, CommitTrait, Commitment},
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::utils::scalar_as_base,
  // ipa::{FinalInnerProductArgument, StepInnerProductArgument},
  polynomial::{EqPolynomial, MultilinearPolynomial},
  sumcheck::SumcheckProof,
  traits::{AbsorbInROTrait, AppendToTranscriptTrait, ChallengeTrait, Group, HashFuncTrait},
};
use bellperson_nonnative::{mp::bignat::nat_to_limbs, util::convert::f_to_nat};
use ff::{Field, PrimeField};
use flate2::{write::ZlibEncoder, Compression};
use itertools::concat;
use merlin::Transcript;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Public parameters for a given R1CS
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
  digest: G::Scalar, // digest of the rest of R1CSShape
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

    let digest = Self::compute_digest(num_cons, num_vars, num_io, A, B, C);

    let shape = R1CSShape {
      num_cons,
      num_vars,
      num_io,
      A: A.to_owned(),
      B: B.to_owned(),
      C: C.to_owned(),
      digest,
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

  /// Bounds "row" variables of (A, B, C) matrices viewed as 2d multilinear polynomials
  pub fn compute_eval_table_sparse(
    &self,
    rx: &[G::Scalar],
  ) -> (Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>) {
    assert_eq!(rx.len(), self.num_cons);

    let mut A_evals: Vec<G::Scalar> = vec![G::Scalar::zero(); self.num_vars + self.num_io + 1];
    let mut B_evals: Vec<G::Scalar> = vec![G::Scalar::zero(); self.num_vars + self.num_io + 1];
    let mut C_evals: Vec<G::Scalar> = vec![G::Scalar::zero(); self.num_vars + self.num_io + 1];

    // TODO: cut duplicate code
    for i in 0..self.A.len() {
      let (row, col, val) = self.A[i];
      A_evals[col] += rx[row] * val;
    }

    for i in 0..self.B.len() {
      let (row, col, val) = self.B[i];
      B_evals[col] += rx[row] * val;
    }

    for i in 0..self.C.len() {
      let (row, col, val) = self.C[i];
      C_evals[col] += rx[row] * val;
    }

    (A_evals, B_evals, C_evals)
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
  ) -> Result<(Vec<G::Scalar>, Commitment<G>), NovaError> {
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

    let comm_T = T.commit(&gens.gens_E);

    Ok((T, comm_T))
  }

  /// returns the digest of R1CSShape
  pub fn get_digest(&self) -> G::Scalar {
    self.digest
  }

  fn compute_digest(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    A: &[(usize, usize, G::Scalar)],
    B: &[(usize, usize, G::Scalar)],
    C: &[(usize, usize, G::Scalar)],
  ) -> G::Scalar {
    let shape_serialized = R1CSShapeSerialized {
      num_cons,
      num_vars,
      num_io,
      A: A
        .iter()
        .map(|(i, j, v)| (*i, *j, v.to_repr().as_ref().to_vec()))
        .collect(),
      B: B
        .iter()
        .map(|(i, j, v)| (*i, *j, v.to_repr().as_ref().to_vec()))
        .collect(),
      C: C
        .iter()
        .map(|(i, j, v)| (*i, *j, v.to_repr().as_ref().to_vec()))
        .collect(),
    };

    // obtain a vector of bytes representing the R1CS shape
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &shape_serialized).unwrap();
    let shape_bytes = encoder.finish().unwrap();

    // convert shape_bytes into a short digest
    let mut hasher = Sha3_256::new();
    hasher.input(&shape_bytes);
    let digest = hasher.result();

    // truncate the digest to 250 bits
    let bv = (0..NUM_HASH_BITS).map(|i| {
      let (byte_pos, bit_pos) = (i / 8, i % 8);
      let bit = (digest[byte_pos] >> bit_pos) & 1;
      bit == 1
    });

    // turn the bit vector into a scalar
    let mut res = G::Scalar::zero();
    let mut coeff = G::Scalar::one();
    for bit in bv {
      if bit {
        res += coeff;
      }
      coeff += coeff;
    }
    res
  }

  /// Pads the R1CSShape so that the number of variables is a power of two
  /// Renumbers variables to accomodate padded variables
  fn pad(&self) -> Self {
    // check if the provided R1CSShape is already as required
    if self.num_vars.next_power_of_two() == self.num_vars
      && self.num_cons.next_power_of_two() == self.num_cons
    {
      return self.clone();
    }

    // check if the number of variables are as expected, then
    // we simply set the number of constraints to the next power of two
    if self.num_vars.next_power_of_two() == self.num_vars {
      let digest = Self::compute_digest(
        self.num_cons.next_power_of_two(),
        self.num_vars,
        self.num_io,
        &self.A,
        &self.B,
        &self.C,
      );

      return R1CSShape {
        num_cons: self.num_cons.next_power_of_two(),
        num_vars: self.num_vars,
        num_io: self.num_io,
        A: self.A.clone(),
        B: self.B.clone(),
        C: self.C.clone(),
        digest,
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = self.num_vars.next_power_of_two();
    let num_cons_padded = self.num_cons.next_power_of_two();

    // TODO: cut duplicate code
    let A_padded = self
      .A
      .iter()
      .map(|(r, c, v)| {
        if c >= &self.num_vars {
          (*r, c + num_vars_padded - self.num_vars, v.clone())
        } else {
          (*r, *c, v.clone())
        }
      })
      .collect::<Vec<_>>();

    let B_padded = self
      .B
      .iter()
      .map(|(r, c, v)| {
        if c >= &self.num_vars {
          (*r, c + num_vars_padded - self.num_vars, v.clone())
        } else {
          (*r, *c, v.clone())
        }
      })
      .collect::<Vec<_>>();

    let C_padded = self
      .C
      .iter()
      .map(|(r, c, v)| {
        if c >= &self.num_vars {
          (*r, c + num_vars_padded - self.num_vars, v.clone())
        } else {
          (*r, *c, v.clone())
        }
      })
      .collect::<Vec<_>>();

    let digest = Self::compute_digest(
      num_cons_padded,
      num_vars_padded,
      self.num_io,
      &A_padded,
      &B_padded,
      &C_padded,
    );

    R1CSShape {
      num_cons: num_cons_padded,
      num_vars: num_vars_padded,
      num_io: self.num_io,
      A: A_padded,
      B: B_padded,
      C: C_padded,
      digest,
    }
  }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct R1CSShapeSerialized {
  num_cons: usize,
  num_vars: usize,
  num_io: usize,
  A: Vec<(usize, usize, Vec<u8>)>,
  B: Vec<(usize, usize, Vec<u8>)>,
  C: Vec<(usize, usize, Vec<u8>)>,
}

impl<G: Group> AppendToTranscriptTrait for R1CSShape<G> {
  fn append_to_transcript(&self, _label: &'static [u8], transcript: &mut Transcript) {
    self
      .get_digest()
      .append_to_transcript(b"R1CSShape", transcript);
  }
}

impl<G: Group> AbsorbInROTrait<G> for R1CSShape<G> {
  fn absorb_in_ro(&self, ro: &mut G::HashFunc) {
    ro.absorb(scalar_as_base::<G>(self.get_digest()));
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

impl<G: Group> AppendToTranscriptTrait for R1CSInstance<G> {
  fn append_to_transcript(&self, _label: &'static [u8], transcript: &mut Transcript) {
    self.comm_W.append_to_transcript(b"comm_W", transcript);
    self.X.append_to_transcript(b"X", transcript);
  }
}

impl<G: Group> AbsorbInROTrait<G> for R1CSInstance<G> {
  fn absorb_in_ro(&self, ro: &mut G::HashFunc) {
    self.comm_W.absorb_in_ro(ro);
    for x in &self.X {
      ro.absorb(scalar_as_base::<G>(*x));
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

  /// Initializes a new RelaxedR1CSWitness from an R1CSWitness
  pub fn from_r1cs_witness(S: &R1CSShape<G>, witness: &R1CSWitness<G>) -> RelaxedR1CSWitness<G> {
    RelaxedR1CSWitness {
      W: witness.W.clone(),
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

  fn pad(&self, S: &R1CSShape<G>) -> RelaxedR1CSWitness<G> {
    let W = {
      let mut W = self.W.clone();
      W.extend(vec![G::Scalar::zero(); S.num_vars - W.len()]);
      W
    };

    let E = {
      let mut E = self.E.clone();
      E.extend(vec![G::Scalar::zero(); S.num_cons - E.len()]);
      E
    };

    Self { W, E }
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
    }
  }

  /// Initializes a new RelaxedR1CSInstance from an R1CSInstance
  pub fn from_r1cs_instance(
    gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    instance: &R1CSInstance<G>,
  ) -> RelaxedR1CSInstance<G> {
    let mut r_instance = RelaxedR1CSInstance::default(gens, S);
    r_instance.comm_W = instance.comm_W;
    r_instance.u = G::Scalar::one();
    r_instance.X = instance.X.clone();
    r_instance
  }

  /// Folds an incoming RelaxedR1CSInstance into the current one
  pub fn fold(
    &self,
    U2: &R1CSInstance<G>,
    comm_T: &Commitment<G>,
    r: &G::Scalar,
  ) -> Result<RelaxedR1CSInstance<G>, NovaError> {
    let (X1, u1, comm_W_1, comm_E_1) =
      (&self.X, &self.u, &self.comm_W.clone(), &self.comm_E.clone());
    let (X2, comm_W_2) = (&U2.X, &U2.comm_W);

    // weighted sum of X, comm_W, comm_E, and u
    let X = X1
      .par_iter()
      .zip(X2)
      .map(|(a, b)| *a + *r * *b)
      .collect::<Vec<G::Scalar>>();
    let comm_W = comm_W_1 + comm_W_2 * r;
    let comm_E = *comm_E_1 + *comm_T * *r;
    let u = *u1 + *r;

    Ok(RelaxedR1CSInstance {
      comm_W,
      comm_E,
      X,
      u,
    })
  }
}

impl<G: Group> AppendToTranscriptTrait for RelaxedR1CSInstance<G> {
  fn append_to_transcript(&self, _label: &'static [u8], transcript: &mut Transcript) {
    self.comm_W.append_to_transcript(b"comm_W", transcript);
    self.comm_E.append_to_transcript(b"comm_E", transcript);
    self.u.append_to_transcript(b"u", transcript);
    self.X.append_to_transcript(b"X", transcript);
  }
}

impl<G: Group> AbsorbInROTrait<G> for RelaxedR1CSInstance<G> {
  fn absorb_in_ro(&self, ro: &mut G::HashFunc) {
    self.comm_W.absorb_in_ro(ro);
    self.comm_E.absorb_in_ro(ro);
    ro.absorb(scalar_as_base::<G>(self.u));

    // absorb each element of self.X in bignum format
    for x in &self.X {
      let limbs: Vec<G::Scalar> = nat_to_limbs(&f_to_nat(x), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
      for limb in limbs {
        ro.absorb(scalar_as_base::<G>(limb));
      }
    }
  }
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
pub struct RelaxedR1CSSNARK<G: Group> {
  sc_proof_outer: SumcheckProof<G>,
  claims_outer: (G::Scalar, G::Scalar, G::Scalar),
  sc_proof_inner: SumcheckProof<G>,
  eval_W: G::Scalar,
  // ipa_eval_W: StepInnerProductArgument<G>,
  eval_E: G::Scalar,
  // ipa_eval_E: StepInnerProductArgument<G>,
  // ipa_final: FinalInnerProductArgument<G>,
}

impl<G: Group> RelaxedR1CSSNARK<G> {
  fn protocol_name() -> &'static [u8] {
    b"RelaxedR1CSSNARK"
  }

  /// produces a succinct proof of satisfiability of a RelaxedR1CS instance
  pub fn prove(
    _gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
    transcript: &mut Transcript,
  ) -> Result<Self, NovaError> {
    // append the protocol name to the transcript
    transcript.append_message(b"protocol-name", RelaxedR1CSSNARK::<G>::protocol_name());

    // pad the shape and witness
    let S = S.pad();
    let W = W.pad(&S);

    // sanity check that R1CSShape has certain size characteristics
    assert_eq!(S.num_cons.next_power_of_two(), S.num_cons);
    assert_eq!(S.num_vars.next_power_of_two(), S.num_vars);
    assert_eq!(S.num_io.next_power_of_two(), S.num_io);
    assert!(S.num_io < S.num_vars);

    // append the R1CSShape and RelaxedR1CSInstance to the transcript
    S.append_to_transcript(b"S", transcript);
    U.append_to_transcript(b"U", transcript);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    // TODO: cleanup how this gets created?
    let z = {
      let mut z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);
      z.resize(S.num_vars * 2, G::Scalar::zero());
      z
    };

    let (num_rounds_x, num_rounds_y) =
      (S.num_cons.log2() as usize, (S.num_vars.log2() + 1) as usize);

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| G::Scalar::challenge(b"challenge_tau", transcript))
      .collect();
    let poly_tau = EqPolynomial::new(tau).evals();
    let (poly_Az, poly_Bz, poly_Cz, poly_uCz_E) = {
      let (poly_Az, poly_Bz, poly_Cz) = S.multiply_vec(&z)?;
      let poly_uCz_E = (0..S.num_cons)
        .map(|i| U.u * poly_Cz[i] + W.E[i])
        .collect::<Vec<G::Scalar>>();
      (poly_Az, poly_Bz, poly_Cz, poly_uCz_E)
    };

    let comb_func_outer =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };
    let (sc_proof_outer, r_x, claims_outer) = SumcheckProof::prove_cubic_with_additive_term(
      &G::Scalar::zero(), // claim is zero
      num_rounds_x,
      &mut MultilinearPolynomial::new(poly_tau),
      &mut MultilinearPolynomial::new(poly_Az),
      &mut MultilinearPolynomial::new(poly_Bz),
      &mut MultilinearPolynomial::new(poly_uCz_E),
      comb_func_outer,
      transcript,
    );

    // claims from the end of sum-check
    let _claim_tau: G::Scalar = claims_outer[0];
    let (claim_Az, claim_Bz, _claim_uCz_E) = (claims_outer[1], claims_outer[2], claims_outer[3]);
    //claims_outer[0].append_to_transcript(b"claim_tau", transcript);
    claims_outer[1].append_to_transcript(b"claim_Az", transcript);
    claims_outer[2].append_to_transcript(b"claim_Bz", transcript);
    let claim_Cz = MultilinearPolynomial::new(poly_Cz).evaluate(&r_x);
    let eval_E = MultilinearPolynomial::new(W.E.clone()).evaluate(&r_x);
    claim_Cz.append_to_transcript(b"claim_Cz", transcript);
    eval_E.append_to_transcript(b"claim_E", transcript);

    // inner sum-check
    let r_A = G::Scalar::challenge(b"challenge_rA", transcript);
    let r_B = G::Scalar::challenge(b"challenge_rB", transcript);
    let r_C = G::Scalar::challenge(b"challenge_rC", transcript);
    let claim_inner_joint = r_A * claims_outer[1] + r_B * claims_outer[2] + r_C * claim_Cz;

    let poly_ABC = {
      // compute the initial evaluation table for R(\tau, x)
      let evals_rx = EqPolynomial::new(r_x.clone()).evals();
      let (evals_A, evals_B, evals_C) = S.compute_eval_table_sparse(&evals_rx);

      assert_eq!(evals_A.len(), evals_B.len());
      assert_eq!(evals_A.len(), evals_C.len());
      (0..evals_A.len())
        .map(|i| r_A * evals_A[i] + r_B * evals_B[i] + r_C * evals_C[i])
        .collect::<Vec<G::Scalar>>()
    };

    let comb_func = |poly_A_comp: &G::Scalar, poly_B_comp: &G::Scalar| -> G::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_inner, r_y, _claims_inner) = SumcheckProof::prove_quad(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(z),
      &mut MultilinearPolynomial::new(poly_ABC),
      comb_func,
      transcript,
    );

    let eval_W = MultilinearPolynomial::new(W.W.clone()).evaluate(&r_y);

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      claims_outer: (claim_Az, claim_Bz, claim_Cz),
      sc_proof_inner,
      eval_W,
      eval_E,
    })
  }

  /// verifies a proof of satisfiability of a RelaxedR1CS instance
  pub fn verify(
    &self,
    _gens: &R1CSGens<G>,
    S: &R1CSShape<G>,
    U: &RelaxedR1CSInstance<G>,
    transcript: &mut Transcript,
  ) -> Result<(), NovaError> {
    // append the protocol name to the transcript
    transcript.append_message(b"protocol-name", RelaxedR1CSSNARK::<G>::protocol_name());

    // append the R1CSShape and RelaxedR1CSInstance to the transcript
    S.append_to_transcript(b"S", transcript);
    U.append_to_transcript(b"U", transcript);

    let (num_rounds_x, num_rounds_y) =
      (S.num_cons.log2() as usize, (S.num_vars.log2() + 1) as usize);

    // outer sum-check
    let _tau = (0..num_rounds_x)
      .map(|_i| G::Scalar::challenge(b"challenge_tau", transcript))
      .collect::<Vec<G::Scalar>>();

    let (_claim_outer_final, _r_x) =
      self
        .sc_proof_outer
        .verify(G::Scalar::zero(), num_rounds_x, 3, transcript)?;

    // TODO: verify claim_outer_final

    self
      .claims_outer
      .0
      .append_to_transcript(b"claim_Az", transcript);
    self
      .claims_outer
      .1
      .append_to_transcript(b"claim_Bz", transcript);
    self
      .claims_outer
      .2
      .append_to_transcript(b"claim_Cz", transcript);
    self.eval_E.append_to_transcript(b"claim_E", transcript);

    // inner sum-check
    let r_A = G::Scalar::challenge(b"challenge_rA", transcript);
    let r_B = G::Scalar::challenge(b"challenge_rB", transcript);
    let r_C = G::Scalar::challenge(b"challenge_rC", transcript);
    let claim_inner_joint =
      r_A * self.claims_outer.0 + r_B * self.claims_outer.1 + r_C * self.claims_outer.2;

    let (_claim_inner_final, _r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 2, transcript)?;

    // TODO: verify claim_inner_final

    // TODO: verify eval_W and eval_E

    Ok(())
  }
}
