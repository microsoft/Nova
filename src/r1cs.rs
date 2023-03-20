//! This module defines R1CS related types and a folding scheme for Relaxed R1CS
#![allow(clippy::type_complexity)]
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  traits::{
    commitment::CommitmentEngineTrait, AbsorbInROTrait, Group, ROTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, CE,
};
use core::{cmp::max, marker::PhantomData};
use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use itertools::concat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};

/// Public parameters for a given R1CS
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CS<G: Group> {
  _p: PhantomData<G>,
}

/// A type that holds the shape of the R1CS matrices
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSShape<G: Group> {
  pub(crate) num_cons: usize,
  pub(crate) num_vars: usize,
  pub(crate) num_io: usize,
  pub(crate) A: Vec<(usize, usize, G::Scalar)>,
  pub(crate) B: Vec<(usize, usize, G::Scalar)>,
  pub(crate) C: Vec<(usize, usize, G::Scalar)>,
  digest: G::Scalar, // digest of the rest of R1CSShape
}

/// A type that holds a witness for a given R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct R1CSWitness<G: Group> {
  W: Vec<G::Scalar>,
}

/// A type that holds an R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSInstance<G: Group> {
  pub(crate) comm_W: Commitment<G>,
  pub(crate) X: Vec<G::Scalar>,
}

/// A type that holds a witness for a given Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RelaxedR1CSWitness<G: Group> {
  pub(crate) W: Vec<G::Scalar>,
  pub(crate) E: Vec<G::Scalar>,
}

/// A type that holds a Relaxed R1CS instance
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSInstance<G: Group> {
  pub(crate) comm_W: Commitment<G>,
  pub(crate) comm_E: Commitment<G>,
  pub(crate) X: Vec<G::Scalar>,
  pub(crate) u: G::Scalar,
}

impl<G: Group> R1CS<G> {
  /// Samples public parameters for the specified number of constraints and variables in an R1CS
  pub fn commitment_key(S: &R1CSShape<G>) -> CommitmentKey<G> {
    let num_cons = S.num_cons;
    let num_vars = S.num_vars;
    let num_nz = max(max(S.A.len(), S.B.len()), S.C.len());
    G::CE::setup(b"ck", max(max(num_cons, num_vars), num_nz))
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

  pub fn multiply_vec(
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

    let (Az, (Bz, Cz)) = rayon::join(
      || sparse_matrix_vec_product(&self.A, self.num_cons, z),
      || {
        rayon::join(
          || sparse_matrix_vec_product(&self.B, self.num_cons, z),
          || sparse_matrix_vec_product(&self.C, self.num_cons, z),
        )
      },
    );

    Ok((Az, Bz, Cz))
  }

  /// Checks if the Relaxed R1CS instance is satisfiable given a witness and its shape
  pub fn is_sat_relaxed(
    &self,
    ck: &CommitmentKey<G>,
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
        .map(|i| usize::from(Az[i] * Bz[i] != U.u * Cz[i] + W.E[i]))
        .sum();

      res == 0
    };

    // verify if comm_E and comm_W are commitments to E and W
    let res_comm: bool = {
      let (comm_W, comm_E) =
        rayon::join(|| CE::<G>::commit(ck, &W.W), || CE::<G>::commit(ck, &W.E));
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
    ck: &CommitmentKey<G>,
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
        .map(|i| usize::from(Az[i] * Bz[i] != Cz[i]))
        .sum();

      res == 0
    };

    // verify if comm_W is a commitment to W
    let res_comm: bool = U.comm_W == CE::<G>::commit(ck, &W.W);

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
    ck: &CommitmentKey<G>,
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
      .into_par_iter()
      .map(|i| AZ_1[i] * BZ_2[i])
      .collect::<Vec<G::Scalar>>();
    let AZ_2_circ_BZ_1 = (0..AZ_2.len())
      .into_par_iter()
      .map(|i| AZ_2[i] * BZ_1[i])
      .collect::<Vec<G::Scalar>>();
    let u_1_cdot_CZ_2 = (0..CZ_2.len())
      .into_par_iter()
      .map(|i| U1.u * CZ_2[i])
      .collect::<Vec<G::Scalar>>();
    let u_2_cdot_CZ_1 = (0..CZ_1.len())
      .into_par_iter()
      .map(|i| CZ_1[i])
      .collect::<Vec<G::Scalar>>();

    let T = AZ_1_circ_BZ_2
      .par_iter()
      .zip(&AZ_2_circ_BZ_1)
      .zip(&u_1_cdot_CZ_2)
      .zip(&u_2_cdot_CZ_1)
      .map(|(((a, b), c), d)| *a + *b - *c - *d)
      .collect::<Vec<G::Scalar>>();

    let comm_T = CE::<G>::commit(ck, &T);

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
    #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
    struct R1CSShapeWithoutDigest<G: Group> {
      num_cons: usize,
      num_vars: usize,
      num_io: usize,
      A: Vec<(usize, usize, G::Scalar)>,
      B: Vec<(usize, usize, G::Scalar)>,
      C: Vec<(usize, usize, G::Scalar)>,
    }

    let shape = R1CSShapeWithoutDigest::<G> {
      num_cons,
      num_vars,
      num_io,
      A: A.to_vec(),
      B: B.to_vec(),
      C: C.to_vec(),
    };

    // obtain a vector of bytes representing the R1CS shape
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serialize_into(&mut encoder, &shape).unwrap();
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
      let digest = Self::compute_digest(m, self.num_vars, self.num_io, &self.A, &self.B, &self.C);

      return R1CSShape {
        num_cons: m,
        num_vars: m,
        num_io: self.num_io,
        A: self.A.clone(),
        B: self.B.clone(),
        C: self.C.clone(),
        digest,
      };
    }

    // otherwise, we need to pad the number of variables and renumber variable accesses
    let num_vars_padded = m;
    let num_cons_padded = m;
    let apply_pad = |M: &[(usize, usize, G::Scalar)]| -> Vec<(usize, usize, G::Scalar)> {
      M.par_iter()
        .map(|(r, c, v)| {
          if c >= &self.num_vars {
            (*r, c + num_vars_padded - self.num_vars, *v)
          } else {
            (*r, *c, *v)
          }
        })
        .collect::<Vec<_>>()
    };

    let A_padded = apply_pad(&self.A);
    let B_padded = apply_pad(&self.B);
    let C_padded = apply_pad(&self.C);

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

impl<G: Group> TranscriptReprTrait<G> for R1CSShape<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    self.get_digest().to_transcript_bytes()
  }
}

impl<G: Group> AbsorbInROTrait<G> for R1CSShape<G> {
  fn absorb_in_ro(&self, ro: &mut G::RO) {
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
  pub fn commit(&self, ck: &CommitmentKey<G>) -> Commitment<G> {
    CE::<G>::commit(ck, &self.W)
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

impl<G: Group> AbsorbInROTrait<G> for R1CSInstance<G> {
  fn absorb_in_ro(&self, ro: &mut G::RO) {
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
  pub fn commit(&self, ck: &CommitmentKey<G>) -> (Commitment<G>, Commitment<G>) {
    (CE::<G>::commit(ck, &self.W), CE::<G>::commit(ck, &self.E))
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

  /// Pads the provided witness to the correct length
  pub fn pad(&self, S: &R1CSShape<G>) -> RelaxedR1CSWitness<G> {
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
  pub fn default(_ck: &CommitmentKey<G>, S: &R1CSShape<G>) -> RelaxedR1CSInstance<G> {
    let (comm_W, comm_E) = (Commitment::<G>::default(), Commitment::<G>::default());
    RelaxedR1CSInstance {
      comm_W,
      comm_E,
      u: G::Scalar::zero(),
      X: vec![G::Scalar::zero(); S.num_io],
    }
  }

  /// Initializes a new RelaxedR1CSInstance from an R1CSInstance
  pub fn from_r1cs_instance(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
    instance: &R1CSInstance<G>,
  ) -> RelaxedR1CSInstance<G> {
    let mut r_instance = RelaxedR1CSInstance::default(ck, S);
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
    let comm_W = *comm_W_1 + *comm_W_2 * *r;
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

impl<G: Group> TranscriptReprTrait<G> for RelaxedR1CSInstance<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_W.to_transcript_bytes(),
      self.comm_E.to_transcript_bytes(),
      self.u.to_transcript_bytes(),
      self.X.as_slice().to_transcript_bytes(),
    ]
    .concat()
  }
}

impl<G: Group> AbsorbInROTrait<G> for RelaxedR1CSInstance<G> {
  fn absorb_in_ro(&self, ro: &mut G::RO) {
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
