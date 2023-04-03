//! This module implements RelaxedR1CSSNARK traits using a spark-based approach to prove evaluations of
//! sparse multilinear polynomials involved in Spartan's sum-check protocol, thereby providing a preprocessing SNARK
use crate::{
  bellperson::{
    r1cs::{NovaShape, NovaWitness},
    shape_cs::ShapeCS,
    solver::SatisfyingAssignment,
  },
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    math::Math,
    polynomial::{EqPolynomial, MultilinearPolynomial, SparsePolynomial},
    sumcheck::SumcheckProof,
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    circuit::StepCircuit, commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait,
    snark::RelaxedR1CSSNARKTrait, Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey,
};
use bellperson::{gadgets::num::AllocatedNum, Circuit, ConstraintSystem, SynthesisError};
use core::{cmp::max, marker::PhantomData};
use ff::Field;
use itertools::concat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod product;

use product::{IdentityPolynomial, ProductArgument};

/// A type that holds R1CSShape in a form amenable to memory checking
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkRepr<G: Group> {
  N: usize, // size of the vectors

  // dense representation
  row: Vec<G::Scalar>,
  col: Vec<G::Scalar>,
  val_A: Vec<G::Scalar>,
  val_B: Vec<G::Scalar>,
  val_C: Vec<G::Scalar>,

  // timestamp polynomials
  row_read_ts: Vec<G::Scalar>,
  row_audit_ts: Vec<G::Scalar>,
  col_read_ts: Vec<G::Scalar>,
  col_audit_ts: Vec<G::Scalar>,
}

/// A type that holds a commitment to a sparse polynomial
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkCommitment<G: Group> {
  N: usize, // size of each vector

  // commitments to the dense representation
  comm_row: Commitment<G>,
  comm_col: Commitment<G>,
  comm_val_A: Commitment<G>,
  comm_val_B: Commitment<G>,
  comm_val_C: Commitment<G>,

  // commitments to the timestamp polynomials
  comm_row_read_ts: Commitment<G>,
  comm_row_audit_ts: Commitment<G>,
  comm_col_read_ts: Commitment<G>,
  comm_col_audit_ts: Commitment<G>,
}

impl<G: Group> TranscriptReprTrait<G> for R1CSShapeSparkCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_row,
      self.comm_col,
      self.comm_val_A,
      self.comm_val_B,
      self.comm_val_C,
      self.comm_row_read_ts,
      self.comm_row_audit_ts,
      self.comm_col_read_ts,
      self.comm_col_audit_ts,
    ]
    .as_slice()
    .to_transcript_bytes()
  }
}

impl<G: Group> R1CSShapeSparkRepr<G> {
  /// represents R1CSShape in a Spark-friendly format amenable to memory checking
  pub fn new(S: &R1CSShape<G>) -> R1CSShapeSparkRepr<G> {
    let N = {
      let total_nz = S.A.len() + S.B.len() + S.C.len();
      max(total_nz, max(2 * S.num_vars, S.num_cons)).next_power_of_two()
    };

    let row = {
      let mut r = S
        .A
        .iter()
        .chain(S.B.iter())
        .chain(S.C.iter())
        .map(|(r, _, _)| *r)
        .collect::<Vec<usize>>();
      r.resize(N, 0usize);
      r
    };

    let col = {
      let mut c = S
        .A
        .iter()
        .chain(S.B.iter())
        .chain(S.C.iter())
        .map(|(_, c, _)| *c)
        .collect::<Vec<usize>>();
      c.resize(N, 0usize);
      c
    };

    let val_A = {
      let mut val = S.A.iter().map(|(_, _, v)| *v).collect::<Vec<G::Scalar>>();
      val.resize(N, G::Scalar::zero());
      val
    };

    let val_B = {
      // prepend zeros
      let mut val = vec![G::Scalar::zero(); S.A.len()];
      val.extend(S.B.iter().map(|(_, _, v)| *v).collect::<Vec<G::Scalar>>());
      // append zeros
      val.resize(N, G::Scalar::zero());
      val
    };

    let val_C = {
      // prepend zeros
      let mut val = vec![G::Scalar::zero(); S.A.len() + S.B.len()];
      val.extend(S.C.iter().map(|(_, _, v)| *v).collect::<Vec<G::Scalar>>());
      // append zeros
      val.resize(N, G::Scalar::zero());
      val
    };

    // timestamp calculation routine
    let timestamp_calc =
      |num_ops: usize, num_cells: usize, addr_trace: &[usize]| -> (Vec<usize>, Vec<usize>) {
        let mut read_ts = vec![0usize; num_ops];
        let mut audit_ts = vec![0usize; num_cells];

        assert!(num_ops >= addr_trace.len());
        for i in 0..addr_trace.len() {
          let addr = addr_trace[i];
          assert!(addr < num_cells);
          let r_ts = audit_ts[addr];
          read_ts[i] = r_ts;

          let w_ts = r_ts + 1;
          audit_ts[addr] = w_ts;
        }
        (read_ts, audit_ts)
      };

    // timestamp polynomials for row
    let (row_read_ts, row_audit_ts) = timestamp_calc(N, N, &row);
    let (col_read_ts, col_audit_ts) = timestamp_calc(N, N, &col);

    // a routine to turn a vector of usize into a vector scalars
    let to_vec_scalar = |v: &[usize]| -> Vec<G::Scalar> {
      (0..v.len())
        .map(|i| G::Scalar::from(v[i] as u64))
        .collect::<Vec<G::Scalar>>()
    };

    R1CSShapeSparkRepr {
      N,

      // dense representation
      row: to_vec_scalar(&row),
      col: to_vec_scalar(&col),
      val_A,
      val_B,
      val_C,

      // timestamp polynomials
      row_read_ts: to_vec_scalar(&row_read_ts),
      row_audit_ts: to_vec_scalar(&row_audit_ts),
      col_read_ts: to_vec_scalar(&col_read_ts),
      col_audit_ts: to_vec_scalar(&col_audit_ts),
    }
  }

  fn commit(&self, ck: &CommitmentKey<G>) -> R1CSShapeSparkCommitment<G> {
    let comm_vec: Vec<Commitment<G>> = [
      &self.row,
      &self.col,
      &self.val_A,
      &self.val_B,
      &self.val_C,
      &self.row_read_ts,
      &self.row_audit_ts,
      &self.col_read_ts,
      &self.col_audit_ts,
    ]
    .par_iter()
    .map(|v| G::CE::commit(ck, v))
    .collect();

    R1CSShapeSparkCommitment {
      N: self.row.len(),
      comm_row: comm_vec[0],
      comm_col: comm_vec[1],
      comm_val_A: comm_vec[2],
      comm_val_B: comm_vec[3],
      comm_val_C: comm_vec[4],
      comm_row_read_ts: comm_vec[5],
      comm_row_audit_ts: comm_vec[6],
      comm_col_read_ts: comm_vec[7],
      comm_col_audit_ts: comm_vec[8],
    }
  }

  /// evaluates the the provided R1CSShape at (r_x, r_y)
  pub fn multi_evaluate(
    M_vec: &[&[(usize, usize, G::Scalar)]],
    r_x: &[G::Scalar],
    r_y: &[G::Scalar],
  ) -> Vec<G::Scalar> {
    let evaluate_with_table =
      |M: &[(usize, usize, G::Scalar)], T_x: &[G::Scalar], T_y: &[G::Scalar]| -> G::Scalar {
        (0..M.len())
          .collect::<Vec<usize>>()
          .par_iter()
          .map(|&i| {
            let (row, col, val) = M[i];
            T_x[row] * T_y[col] * val
          })
          .reduce(G::Scalar::zero, |acc, x| acc + x)
      };

    let (T_x, T_y) = rayon::join(
      || EqPolynomial::new(r_x.to_vec()).evals(),
      || EqPolynomial::new(r_y.to_vec()).evals(),
    );

    (0..M_vec.len())
      .collect::<Vec<usize>>()
      .par_iter()
      .map(|&i| evaluate_with_table(M_vec[i], &T_x, &T_y))
      .collect()
  }

  // computes evaluation oracles
  fn evaluation_oracles(
    &self,
    S: &R1CSShape<G>,
    r_x: &[G::Scalar],
    z: &[G::Scalar],
  ) -> (
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
  ) {
    let r_x_padded = {
      let mut x = vec![G::Scalar::zero(); self.N.log_2() - r_x.len()];
      x.extend(r_x);
      x
    };

    let mem_row = EqPolynomial::new(r_x_padded).evals();
    let mem_col = {
      let mut z = z.to_vec();
      z.resize(self.N, G::Scalar::zero());
      z
    };

    let mut E_row = S
      .A
      .iter()
      .chain(S.B.iter())
      .chain(S.C.iter())
      .map(|(r, _, _)| mem_row[*r])
      .collect::<Vec<G::Scalar>>();

    let mut E_col = S
      .A
      .iter()
      .chain(S.B.iter())
      .chain(S.C.iter())
      .map(|(_, c, _)| mem_col[*c])
      .collect::<Vec<G::Scalar>>();

    E_row.resize(self.N, mem_row[0]); // we place mem_row[0] since resized row is appended with 0s
    E_col.resize(self.N, mem_col[0]);

    (mem_row, mem_col, E_row, E_col)
  }
}

/// A type that represents the memory-checking argument
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct MemcheckProof<G: Group> {
  sc_prod: ProductArgument<G>,

  eval_init_row: G::Scalar,
  eval_read_row: G::Scalar,
  eval_write_row: G::Scalar,
  eval_audit_row: G::Scalar,
  eval_init_col: G::Scalar,
  eval_read_col: G::Scalar,
  eval_write_col: G::Scalar,
  eval_audit_col: G::Scalar,

  eval_row: G::Scalar,
  eval_row_read_ts: G::Scalar,
  eval_E_row: G::Scalar,
  eval_row_audit_ts: G::Scalar,
  eval_col: G::Scalar,
  eval_col_read_ts: G::Scalar,
  eval_E_col: G::Scalar,
  eval_col_audit_ts: G::Scalar,
  eval_z: G::Scalar,
}

impl<G: Group> MemcheckProof<G> {
  #[allow(clippy::too_many_arguments)]
  /// proves a memory-checking relation
  fn prove(
    ck: &CommitmentKey<G>,
    S_repr: &R1CSShapeSparkRepr<G>,
    S_comm: &R1CSShapeSparkCommitment<G>,
    mem_row: &[G::Scalar],
    comm_E_row: &Commitment<G>,
    E_row: &[G::Scalar],
    mem_col: &[G::Scalar],
    comm_E_col: &Commitment<G>,
    E_col: &[G::Scalar],
    transcript: &mut G::TE,
  ) -> Result<
    (
      MemcheckProof<G>,
      Vec<(PolyEvalWitness<G>, PolyEvalInstance<G>)>,
      G::Scalar,
      Vec<G::Scalar>,
    ),
    NovaError,
  > {
    // we now need to prove that E_row and E_col are well-formed
    // we use memory checking: H(INIT) * H(WS) =? H(RS) * H(FINAL)
    let gamma_1 = transcript.squeeze(b"g1")?;
    let gamma_2 = transcript.squeeze(b"g2")?;

    let gamma_1_sqr = gamma_1 * gamma_1;
    let hash_func = |addr: &G::Scalar, val: &G::Scalar, ts: &G::Scalar| -> G::Scalar {
      (*ts * gamma_1_sqr + *val * gamma_1 + *addr) - gamma_2
    };

    let init_row = (0..mem_row.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &mem_row[i], &G::Scalar::zero()))
      .collect::<Vec<G::Scalar>>();
    let read_row = (0..E_row.len())
      .map(|i| hash_func(&S_repr.row[i], &E_row[i], &S_repr.row_read_ts[i]))
      .collect::<Vec<G::Scalar>>();
    let write_row = (0..E_row.len())
      .map(|i| {
        hash_func(
          &S_repr.row[i],
          &E_row[i],
          &(S_repr.row_read_ts[i] + G::Scalar::one()),
        )
      })
      .collect::<Vec<G::Scalar>>();
    let audit_row = (0..mem_row.len())
      .map(|i| {
        hash_func(
          &G::Scalar::from(i as u64),
          &mem_row[i],
          &S_repr.row_audit_ts[i],
        )
      })
      .collect::<Vec<G::Scalar>>();

    let init_col = (0..mem_col.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &mem_col[i], &G::Scalar::zero()))
      .collect::<Vec<G::Scalar>>();
    let read_col = (0..E_col.len())
      .map(|i| hash_func(&S_repr.col[i], &E_col[i], &S_repr.col_read_ts[i]))
      .collect::<Vec<G::Scalar>>();
    let write_col = (0..E_col.len())
      .map(|i| {
        hash_func(
          &S_repr.col[i],
          &E_col[i],
          &(S_repr.col_read_ts[i] + G::Scalar::one()),
        )
      })
      .collect::<Vec<G::Scalar>>();
    let audit_col = (0..mem_col.len())
      .map(|i| {
        hash_func(
          &G::Scalar::from(i as u64),
          &mem_col[i],
          &S_repr.col_audit_ts[i],
        )
      })
      .collect::<Vec<G::Scalar>>();

    let (sc_prod, evals_prod, r_prod, _evals_input_vec, mut w_u_vec) = ProductArgument::prove(
      ck,
      &[
        init_row, read_row, write_row, audit_row, init_col, read_col, write_col, audit_col,
      ],
      transcript,
    )?;

    // row-related and col-related claims of polynomial evaluations to aid the final check of the sum-check
    let evals = [
      &S_repr.row,
      &S_repr.row_read_ts,
      E_row,
      &S_repr.row_audit_ts,
      &S_repr.col,
      &S_repr.col_read_ts,
      E_col,
      &S_repr.col_audit_ts,
      mem_col,
    ]
    .into_par_iter()
    .map(|p| MultilinearPolynomial::evaluate_with(p, &r_prod))
    .collect::<Vec<G::Scalar>>();

    let eval_row = evals[0];
    let eval_row_read_ts = evals[1];
    let eval_E_row = evals[2];
    let eval_row_audit_ts = evals[3];
    let eval_col = evals[4];
    let eval_col_read_ts = evals[5];
    let eval_E_col = evals[6];
    let eval_col_audit_ts = evals[7];
    let eval_z = evals[8];

    // we can batch all the claims
    transcript.absorb(
      b"e",
      &[
        eval_row,
        eval_row_read_ts,
        eval_E_row,
        eval_row_audit_ts,
        eval_col,
        eval_col_read_ts,
        eval_E_col,
        eval_col_audit_ts,
      ]
      .as_slice(),
    );
    let c = transcript.squeeze(b"c")?;
    let eval_joint = eval_row
      + c * eval_row_read_ts
      + c * c * eval_E_row
      + c * c * c * eval_row_audit_ts
      + c * c * c * c * eval_col
      + c * c * c * c * c * eval_col_read_ts
      + c * c * c * c * c * c * eval_E_col
      + c * c * c * c * c * c * c * eval_col_audit_ts;
    let comm_joint = S_comm.comm_row
      + S_comm.comm_row_read_ts * c
      + *comm_E_row * c * c
      + S_comm.comm_row_audit_ts * c * c * c
      + S_comm.comm_col * c * c * c * c
      + S_comm.comm_col_read_ts * c * c * c * c * c
      + *comm_E_col * c * c * c * c * c * c
      + S_comm.comm_col_audit_ts * c * c * c * c * c * c * c;
    let poly_joint = S_repr
      .row
      .iter()
      .zip(S_repr.row_read_ts.iter())
      .zip(E_row.iter())
      .zip(S_repr.row_audit_ts.iter())
      .zip(S_repr.col.iter())
      .zip(S_repr.col_read_ts.iter())
      .zip(E_col.iter())
      .zip(S_repr.col_audit_ts.iter())
      .map(|(((((((x, y), z), m), n), q), s), t)| {
        *x + c * y
          + c * c * z
          + c * c * c * m
          + c * c * c * c * n
          + c * c * c * c * c * q
          + c * c * c * c * c * c * s
          + c * c * c * c * c * c * c * t
      })
      .collect::<Vec<_>>();

    // add the claim to prove for later
    w_u_vec.push((
      PolyEvalWitness { p: poly_joint },
      PolyEvalInstance {
        c: comm_joint,
        x: r_prod.clone(),
        e: eval_joint,
      },
    ));

    let eval_arg = Self {
      eval_init_row: evals_prod[0],
      eval_read_row: evals_prod[1],
      eval_write_row: evals_prod[2],
      eval_audit_row: evals_prod[3],
      eval_init_col: evals_prod[4],
      eval_read_col: evals_prod[5],
      eval_write_col: evals_prod[6],
      eval_audit_col: evals_prod[7],
      sc_prod,

      eval_row,
      eval_row_read_ts,
      eval_E_row,
      eval_row_audit_ts,
      eval_col,
      eval_col_read_ts,
      eval_E_col,
      eval_col_audit_ts,
      eval_z,
    };

    Ok((eval_arg, w_u_vec, eval_z, r_prod))
  }

  /// verifies a memory-checking relation
  fn verify(
    &self,
    S_comm: &R1CSShapeSparkCommitment<G>,
    comm_E_row: &Commitment<G>,
    comm_E_col: &Commitment<G>,
    r_x: &[G::Scalar],
    transcript: &mut G::TE,
  ) -> Result<(Vec<PolyEvalInstance<G>>, G::Scalar, Vec<G::Scalar>), NovaError> {
    let r_x_padded = {
      let mut x = vec![G::Scalar::zero(); S_comm.N.log_2() - r_x.len()];
      x.extend(r_x);
      x
    };

    // verify if E_row and E_col are well formed
    let gamma_1 = transcript.squeeze(b"g1")?;
    let gamma_2 = transcript.squeeze(b"g2")?;

    // hash function
    let gamma_1_sqr = gamma_1 * gamma_1;
    let hash_func = |addr: &G::Scalar, val: &G::Scalar, ts: &G::Scalar| -> G::Scalar {
      (*ts * gamma_1_sqr + *val * gamma_1 + *addr) - gamma_2
    };

    // check the required multiset relationship
    // row
    if self.eval_init_row * self.eval_write_row != self.eval_read_row * self.eval_audit_row {
      return Err(NovaError::InvalidMultisetProof);
    }
    // col
    if self.eval_init_col * self.eval_write_col != self.eval_read_col * self.eval_audit_col {
      return Err(NovaError::InvalidMultisetProof);
    }

    // verify the product proofs
    let (claims_final, r_prod, mut u_vec) = self.sc_prod.verify(
      &[
        self.eval_init_row,
        self.eval_read_row,
        self.eval_write_row,
        self.eval_audit_row,
        self.eval_init_col,
        self.eval_read_col,
        self.eval_write_col,
        self.eval_audit_col,
      ],
      S_comm.N,
      transcript,
    )?;

    // finish the final step of the sum-check
    let (claim_init_expected_row, claim_audit_expected_row) = {
      let addr = IdentityPolynomial::new(r_prod.len()).evaluate(&r_prod);
      let val = EqPolynomial::new(r_x_padded.to_vec()).evaluate(&r_prod);
      (
        hash_func(&addr, &val, &G::Scalar::zero()),
        hash_func(&addr, &val, &self.eval_row_audit_ts),
      )
    };

    let (claim_read_expected_row, claim_write_expected_row) = {
      (
        hash_func(&self.eval_row, &self.eval_E_row, &self.eval_row_read_ts),
        hash_func(
          &self.eval_row,
          &self.eval_E_row,
          &(self.eval_row_read_ts + G::Scalar::one()),
        ),
      )
    };

    // multiset check for the row
    if claim_init_expected_row != claims_final[0]
      || claim_read_expected_row != claims_final[1]
      || claim_write_expected_row != claims_final[2]
      || claim_audit_expected_row != claims_final[3]
    {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let (claim_init_expected_col, claim_audit_expected_col) = {
      let addr = IdentityPolynomial::new(r_prod.len()).evaluate(&r_prod);
      let val = self.eval_z; // this value is later checked against U.comm_W and u.X
      (
        hash_func(&addr, &val, &G::Scalar::zero()),
        hash_func(&addr, &val, &self.eval_col_audit_ts),
      )
    };

    let (claim_read_expected_col, claim_write_expected_col) = {
      (
        hash_func(&self.eval_col, &self.eval_E_col, &self.eval_col_read_ts),
        hash_func(
          &self.eval_col,
          &self.eval_E_col,
          &(self.eval_col_read_ts + G::Scalar::one()),
        ),
      )
    };

    // multiset check for the col
    if claim_init_expected_col != claims_final[4]
      || claim_read_expected_col != claims_final[5]
      || claim_write_expected_col != claims_final[6]
      || claim_audit_expected_col != claims_final[7]
    {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"e",
      &[
        self.eval_row,
        self.eval_row_read_ts,
        self.eval_E_row,
        self.eval_row_audit_ts,
        self.eval_col,
        self.eval_col_read_ts,
        self.eval_E_col,
        self.eval_col_audit_ts,
      ]
      .as_slice(),
    );
    let c = transcript.squeeze(b"c")?;
    let eval_joint = self.eval_row
      + c * self.eval_row_read_ts
      + c * c * self.eval_E_row
      + c * c * c * self.eval_row_audit_ts
      + c * c * c * c * self.eval_col
      + c * c * c * c * c * self.eval_col_read_ts
      + c * c * c * c * c * c * self.eval_E_col
      + c * c * c * c * c * c * c * self.eval_col_audit_ts;
    let comm_joint = S_comm.comm_row
      + S_comm.comm_row_read_ts * c
      + *comm_E_row * c * c
      + S_comm.comm_row_audit_ts * c * c * c
      + S_comm.comm_col * c * c * c * c
      + S_comm.comm_col_read_ts * c * c * c * c * c
      + *comm_E_col * c * c * c * c * c * c
      + S_comm.comm_col_audit_ts * c * c * c * c * c * c * c;

    u_vec.push(PolyEvalInstance {
      c: comm_joint,
      x: r_prod.clone(),
      e: eval_joint,
    });

    Ok((u_vec, self.eval_z, r_prod))
  }
}

/// A type that represents the prover's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  pk_ee: EE::ProverKey,
  S: R1CSShape<G>,
  S_repr: R1CSShapeSparkRepr<G>,
  S_comm: R1CSShapeSparkCommitment<G>,
}

/// A type that represents the verifier's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  num_cons: usize,
  num_vars: usize,
  vk_ee: EE::VerifierKey,
  S_comm: R1CSShapeSparkCommitment<G>,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  // outer sum-check
  sc_proof_outer: SumcheckProof<G>,

  // claims from the end of the outer sum-check
  eval_Az: G::Scalar,
  eval_Bz: G::Scalar,
  eval_Cz: G::Scalar,
  eval_E: G::Scalar,

  // commitment to oracles for the inner sum-check
  comm_E_row: Commitment<G>,
  comm_E_col: Commitment<G>,

  // inner sum-check
  sc_proof_inner: SumcheckProof<G>,

  // claims from the end of inner sum-check
  eval_E_row: G::Scalar,
  eval_E_col: G::Scalar,
  eval_val_A: G::Scalar,
  eval_val_B: G::Scalar,
  eval_val_C: G::Scalar,

  // memory-checking proof
  mc_proof: MemcheckProof<G>,

  // claim about W evaluation
  eval_W: G::Scalar,

  // batch openings of all multilinear polynomials
  sc_proof_batch: SumcheckProof<G>,
  evals_batch: Vec<G::Scalar>,
  eval_arg: EE::EvaluationArgument,
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> RelaxedR1CSSNARKTrait<G>
  for RelaxedR1CSSNARK<G, EE>
{
  type ProverKey = ProverKey<G, EE>;
  type VerifierKey = VerifierKey<G, EE>;

  fn setup(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck);

    // pad the R1CS matrices
    let S = S.pad();

    let S_repr = R1CSShapeSparkRepr::new(&S);
    let S_comm = S_repr.commit(ck);

    let vk = VerifierKey {
      num_cons: S.num_cons,
      num_vars: S.num_vars,
      S_comm: S_comm.clone(),
      vk_ee,
    };

    let pk = ProverKey {
      pk_ee,
      S,
      S_repr,
      S_comm,
    };

    Ok((pk, vk))
  }

  /// produces a succinct proof of satisfiability of a RelaxedR1CS instance
  fn prove(
    ck: &CommitmentKey<G>,
    pk: &Self::ProverKey,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError> {
    let W = W.pad(&pk.S); // pad the witness
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");

    // a list of polynomial evaluation claims that will be batched
    let mut w_u_vec = Vec::new();

    // sanity check that R1CSShape has certain size characteristics
    assert_eq!(pk.S.num_cons.next_power_of_two(), pk.S.num_cons);
    assert_eq!(pk.S.num_vars.next_power_of_two(), pk.S.num_vars);
    assert_eq!(pk.S.num_io.next_power_of_two(), pk.S.num_io);
    assert!(pk.S.num_io < pk.S.num_vars);

    // append the commitment to R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"C", &pk.S_comm);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);

    let (num_rounds_x, _num_rounds_y) = (
      (pk.S.num_cons as f64).log2() as usize,
      ((pk.S.num_vars as f64).log2() as usize + 1),
    );

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    let mut poly_tau = MultilinearPolynomial::new(EqPolynomial::new(tau).evals());
    let (mut poly_Az, mut poly_Bz, poly_Cz, mut poly_uCz_E) = {
      let (poly_Az, poly_Bz, poly_Cz) = pk.S.multiply_vec(&z)?;
      let poly_uCz_E = (0..pk.S.num_cons)
        .map(|i| U.u * poly_Cz[i] + W.E[i])
        .collect::<Vec<G::Scalar>>();
      (
        MultilinearPolynomial::new(poly_Az),
        MultilinearPolynomial::new(poly_Bz),
        MultilinearPolynomial::new(poly_Cz),
        MultilinearPolynomial::new(poly_uCz_E),
      )
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
      &mut poly_tau,
      &mut poly_Az,
      &mut poly_Bz,
      &mut poly_uCz_E,
      comb_func_outer,
      &mut transcript,
    )?;

    // claims from the end of sum-check
    let (eval_Az, eval_Bz): (G::Scalar, G::Scalar) = (claims_outer[1], claims_outer[2]);
    let eval_Cz = poly_Cz.evaluate(&r_x);
    let eval_E = MultilinearPolynomial::new(W.E.clone()).evaluate(&r_x);
    transcript.absorb(b"o", &[eval_Az, eval_Bz, eval_Cz, eval_E].as_slice());

    // add claim about eval_E to be proven
    w_u_vec.push((
      PolyEvalWitness { p: W.E.clone() },
      PolyEvalInstance {
        c: U.comm_E,
        x: r_x.clone(),
        e: eval_E,
      },
    ));

    // send oracles to aid the inner sum-check
    // E_row(i) = eq(r_x, row(i)) for all i
    // E_col(i) = z(col(i)) for all i
    let (mem_row, z, E_row, E_col) = pk.S_repr.evaluation_oracles(&pk.S, &r_x, &z);
    let (comm_E_row, comm_E_col) =
      rayon::join(|| G::CE::commit(ck, &E_row), || G::CE::commit(ck, &E_col));

    // add E_row and E_col to transcript
    transcript.absorb(b"e", &vec![comm_E_row, comm_E_col].as_slice());

    let r = transcript.squeeze(b"r")?;
    let val = pk
      .S_repr
      .val_A
      .iter()
      .zip(pk.S_repr.val_B.iter())
      .zip(pk.S_repr.val_C.iter())
      .map(|((a, b), c)| *a + r * *b + r * r * *c)
      .collect::<Vec<G::Scalar>>();

    // inner sum-check
    let claim_inner_joint = eval_Az + r * eval_Bz + r * r * eval_Cz;
    let num_rounds_y = pk.S_repr.N.log_2();
    let comb_func = |poly_A_comp: &G::Scalar,
                     poly_B_comp: &G::Scalar,
                     poly_C_comp: &G::Scalar|
     -> G::Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };

    debug_assert_eq!(
      E_row
        .iter()
        .zip(val.iter())
        .zip(E_col.iter())
        .map(|((a, b), c)| *a * *b * *c)
        .fold(G::Scalar::zero(), |acc, item| acc + item),
      claim_inner_joint
    );

    let (sc_proof_inner, r_y, claims_inner) = SumcheckProof::prove_cubic(
      &claim_inner_joint,
      num_rounds_y,
      &mut MultilinearPolynomial::new(E_row.clone()),
      &mut MultilinearPolynomial::new(E_col.clone()),
      &mut MultilinearPolynomial::new(val),
      comb_func,
      &mut transcript,
    )?;

    let eval_E_row = claims_inner[0];
    let eval_E_col = claims_inner[1];
    let eval_val_A = MultilinearPolynomial::evaluate_with(&pk.S_repr.val_A, &r_y);
    let eval_val_B = MultilinearPolynomial::evaluate_with(&pk.S_repr.val_B, &r_y);
    let eval_val_C = MultilinearPolynomial::evaluate_with(&pk.S_repr.val_C, &r_y);

    // since all the five polynomials are opened at r_y,
    // we can combine them into a single polynomial opened at r_y
    transcript.absorb(
      b"e",
      &[eval_E_row, eval_E_col, eval_val_A, eval_val_B, eval_val_C].as_slice(),
    );
    let c = transcript.squeeze(b"c")?;
    let eval_sc_inner = eval_E_row
      + c * eval_E_col
      + c * c * eval_val_A
      + c * c * c * eval_val_B
      + c * c * c * c * eval_val_C;
    let comm_sc_inner = comm_E_row
      + comm_E_col * c
      + pk.S_comm.comm_val_A * c * c
      + pk.S_comm.comm_val_B * c * c * c
      + pk.S_comm.comm_val_C * c * c * c * c;
    let poly_sc_inner = E_row
      .iter()
      .zip(E_col.iter())
      .zip(pk.S_repr.val_A.iter())
      .zip(pk.S_repr.val_B.iter())
      .zip(pk.S_repr.val_C.iter())
      .map(|((((x, y), z), m), n)| *x + c * y + c * c * z + c * c * c * m + c * c * c * c * n)
      .collect::<Vec<_>>();

    w_u_vec.push((
      PolyEvalWitness { p: poly_sc_inner },
      PolyEvalInstance {
        c: comm_sc_inner,
        x: r_y,
        e: eval_sc_inner,
      },
    ));

    // we need to prove that E_row and E_col are well-formed
    let (mc_proof, w_u_vec_mem, _eval_z, r_prod) = MemcheckProof::prove(
      ck,
      &pk.S_repr,
      &pk.S_comm,
      &mem_row,
      &comm_E_row,
      &E_row,
      &z,
      &comm_E_col,
      &E_col,
      &mut transcript,
    )?;

    // add claims from memory-checking
    w_u_vec.extend(w_u_vec_mem);

    // we need to prove that eval_z = z(r_prod) = (1-r_prod[0]) * W.w(r_prod[1..]) + r_prod[0] * U.x(r_prod[1..]).
    // r_prod was padded, so we now remove the padding
    let r_prod_unpad = {
      let l = pk.S_repr.N.log_2() - (2 * pk.S.num_vars).log_2();
      r_prod[l..].to_vec()
    };

    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_prod_unpad[1..]);
    w_u_vec.push((
      PolyEvalWitness { p: W.W },
      PolyEvalInstance {
        c: U.comm_W,
        x: r_prod_unpad[1..].to_vec(),
        e: eval_W,
      },
    ));

    // We will now reduce a vector of claims of evaluations at different points into claims about them at the same point.
    // For example, eval_W =? W(r_y[1..]) and eval_W =? E(r_x) into
    // two claims: eval_W_prime =? W(rz) and eval_E_prime =? E(rz)
    // We can them combine the two into one: eval_W_prime + gamma * eval_E_prime =? (W + gamma*E)(rz),
    // where gamma is a public challenge
    // Since commitments to W and E are homomorphic, the verifier can compute a commitment
    // to the batched polynomial.
    assert!(w_u_vec.len() >= 2);

    let (w_vec, u_vec): (Vec<PolyEvalWitness<G>>, Vec<PolyEvalInstance<G>>) =
      w_u_vec.into_iter().unzip();
    let w_vec_padded = PolyEvalWitness::pad(&w_vec); // pad the polynomials to be of the same size
    let u_vec_padded = PolyEvalInstance::pad(&u_vec); // pad the evaluation points

    let powers = |s: &G::Scalar, n: usize| -> Vec<G::Scalar> {
      assert!(n >= 1);
      let mut powers = Vec::new();
      powers.push(G::Scalar::one());
      for i in 1..n {
        powers.push(powers[i - 1] * s);
      }
      powers
    };

    // generate a challenge
    let rho = transcript.squeeze(b"r")?;
    let num_claims = w_vec_padded.len();
    let powers_of_rho = powers(&rho, num_claims);
    let claim_batch_joint = u_vec_padded
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(u, p)| u.e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let mut polys_left: Vec<MultilinearPolynomial<G::Scalar>> = w_vec_padded
      .iter()
      .map(|w| MultilinearPolynomial::new(w.p.clone()))
      .collect();
    let mut polys_right: Vec<MultilinearPolynomial<G::Scalar>> = u_vec_padded
      .iter()
      .map(|u| MultilinearPolynomial::new(EqPolynomial::new(u.x.clone()).evals()))
      .collect();

    let num_rounds_z = u_vec_padded[0].x.len();
    let comb_func = |poly_A_comp: &G::Scalar, poly_B_comp: &G::Scalar| -> G::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_batch, r_z, claims_batch) = SumcheckProof::prove_quad_batch(
      &claim_batch_joint,
      num_rounds_z,
      &mut polys_left,
      &mut polys_right,
      &powers_of_rho,
      comb_func,
      &mut transcript,
    )?;

    let (claims_batch_left, _): (Vec<G::Scalar>, Vec<G::Scalar>) = claims_batch;

    transcript.absorb(b"l", &claims_batch_left.as_slice());

    // we now combine evaluation claims at the same point rz into one
    let gamma = transcript.squeeze(b"g")?;
    let powers_of_gamma: Vec<G::Scalar> = powers(&gamma, num_claims);
    let comm_joint = u_vec_padded
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(u, g_i)| u.c * *g_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);
    let poly_joint = PolyEvalWitness::weighted_sum(&w_vec_padded, &powers_of_gamma);
    let eval_joint = claims_batch_left
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(e, g_i)| *e * *g_i)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &comm_joint,
      &poly_joint.p,
      &r_z,
      &eval_joint,
    )?;

    Ok(RelaxedR1CSSNARK {
      sc_proof_outer,
      eval_Az,
      eval_Bz,
      eval_Cz,
      eval_E,
      comm_E_row,
      comm_E_col,
      sc_proof_inner,
      eval_E_row,
      eval_val_A,
      eval_val_B,
      eval_val_C,
      eval_E_col,

      mc_proof,

      eval_W,

      sc_proof_batch,
      evals_batch: claims_batch_left,
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a RelaxedR1CS instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError> {
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");
    let mut u_vec: Vec<PolyEvalInstance<G>> = Vec::new();

    // append the commitment to R1CS matrices and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"C", &vk.S_comm);
    transcript.absorb(b"U", U);

    let num_rounds_x = (vk.num_cons as f64).log2() as usize;

    // outer sum-check
    let tau = (0..num_rounds_x)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    let (claim_outer_final, r_x) =
      self
        .sc_proof_outer
        .verify(G::Scalar::zero(), num_rounds_x, 3, &mut transcript)?;

    // verify claim_outer_final
    let taus_bound_rx = EqPolynomial::new(tau).evaluate(&r_x);
    let claim_outer_final_expected =
      taus_bound_rx * (self.eval_Az * self.eval_Bz - U.u * self.eval_Cz - self.eval_E);
    if claim_outer_final != claim_outer_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // absorb the claim about eval_E to be checked later
    u_vec.push(PolyEvalInstance {
      c: U.comm_E,
      x: r_x.clone(),
      e: self.eval_E,
    });

    transcript.absorb(
      b"o",
      &[self.eval_Az, self.eval_Bz, self.eval_Cz, self.eval_E].as_slice(),
    );

    // add claimed oracles
    transcript.absorb(b"e", &vec![self.comm_E_row, self.comm_E_col].as_slice());

    // inner sum-check
    let r = transcript.squeeze(b"r")?;
    let claim_inner_joint = self.eval_Az + r * self.eval_Bz + r * r * self.eval_Cz;
    let num_rounds_y = vk.S_comm.N.log_2();

    let (claim_inner_final, r_y) =
      self
        .sc_proof_inner
        .verify(claim_inner_joint, num_rounds_y, 3, &mut transcript)?;

    // verify claim_inner_final
    let claim_inner_final_expected = self.eval_E_row
      * self.eval_E_col
      * (self.eval_val_A + r * self.eval_val_B + r * r * self.eval_val_C);
    if claim_inner_final != claim_inner_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // add claims about five polynomials used at the end of the inner sum-check
    // since they are all evaluated at r_y, we can batch them into one
    transcript.absorb(
      b"e",
      &[
        self.eval_E_row,
        self.eval_E_col,
        self.eval_val_A,
        self.eval_val_B,
        self.eval_val_C,
      ]
      .as_slice(),
    );
    let c = transcript.squeeze(b"c")?;
    let eval_sc_inner = self.eval_E_row
      + c * self.eval_E_col
      + c * c * self.eval_val_A
      + c * c * c * self.eval_val_B
      + c * c * c * c * self.eval_val_C;
    let comm_sc_inner = self.comm_E_row
      + self.comm_E_col * c
      + vk.S_comm.comm_val_A * c * c
      + vk.S_comm.comm_val_B * c * c * c
      + vk.S_comm.comm_val_C * c * c * c * c;

    u_vec.push(PolyEvalInstance {
      c: comm_sc_inner,
      x: r_y,
      e: eval_sc_inner,
    });

    let (u_vec_mem, eval_Z, r_prod) = self.mc_proof.verify(
      &vk.S_comm,
      &self.comm_E_row,
      &self.comm_E_col,
      &r_x,
      &mut transcript,
    )?;

    u_vec.extend(u_vec_mem);

    // we verify that eval_z = z(r_prod) = (1-r_prod[0]) * W.w(r_prod[1..]) + r_prod[0] * U.x(r_prod[1..]).
    let (eval_Z_expected, r_prod_unpad) = {
      // r_prod was padded, so we now remove the padding
      let (factor, r_prod_unpad) = {
        let l = vk.S_comm.N.log_2() - (2 * vk.num_vars).log_2();

        let mut factor = G::Scalar::one();
        for r_p in r_prod.iter().take(l) {
          factor *= G::Scalar::one() - r_p
        }

        (factor, r_prod[l..].to_vec())
      };

      let eval_X = {
        // constant term
        let mut poly_X = vec![(0, U.u)];
        //remaining inputs
        poly_X.extend(
          (0..U.X.len())
            .map(|i| (i + 1, U.X[i]))
            .collect::<Vec<(usize, G::Scalar)>>(),
        );
        SparsePolynomial::new((vk.num_vars as f64).log2() as usize, poly_X)
          .evaluate(&r_prod_unpad[1..])
      };
      let eval_Z =
        factor * ((G::Scalar::one() - r_prod_unpad[0]) * self.eval_W + r_prod_unpad[0] * eval_X);

      (eval_Z, r_prod_unpad)
    };

    if eval_Z != eval_Z_expected {
      return Err(NovaError::IncorrectWitness);
    }

    u_vec.push(PolyEvalInstance {
      c: U.comm_W,
      x: r_prod_unpad[1..].to_vec(),
      e: self.eval_W,
    });

    let u_vec_padded = PolyEvalInstance::pad(&u_vec); // pad the evaluation points

    let powers = |s: &G::Scalar, n: usize| -> Vec<G::Scalar> {
      assert!(n >= 1);
      let mut powers = Vec::new();
      powers.push(G::Scalar::one());
      for i in 1..n {
        powers.push(powers[i - 1] * s);
      }
      powers
    };

    // generate a challenge
    let rho = transcript.squeeze(b"r")?;
    let num_claims = u_vec.len();
    let powers_of_rho = powers(&rho, num_claims);
    let claim_batch_joint = u_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(u, p)| u.e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let num_rounds_z = u_vec_padded[0].x.len();
    let (claim_batch_final, r_z) =
      self
        .sc_proof_batch
        .verify(claim_batch_joint, num_rounds_z, 2, &mut transcript)?;

    let claim_batch_final_expected = {
      let poly_rz = EqPolynomial::new(r_z.clone());
      let evals = u_vec_padded
        .iter()
        .map(|u| poly_rz.evaluate(&u.x))
        .collect::<Vec<G::Scalar>>();

      evals
        .iter()
        .zip(self.evals_batch.iter())
        .zip(powers_of_rho.iter())
        .map(|((e_i, p_i), rho_i)| *e_i * *p_i * rho_i)
        .fold(G::Scalar::zero(), |acc, item| acc + item)
    };

    if claim_batch_final != claim_batch_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(b"l", &self.evals_batch.as_slice());

    // we now combine evaluation claims at the same point rz into one
    let gamma = transcript.squeeze(b"g")?;
    let powers_of_gamma: Vec<G::Scalar> = powers(&gamma, num_claims);
    let comm_joint = u_vec_padded
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(u, g_i)| u.c * *g_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);
    let eval_joint = self
      .evals_batch
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(e, g_i)| *e * *g_i)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &comm_joint,
      &r_z,
      &eval_joint,
      &self.eval_arg,
    )?;

    Ok(())
  }
}

// provides direct interfaces to call the SNARK implemented in this module
struct SpartanCircuit<G: Group, SC: StepCircuit<G::Scalar>> {
  z_i: Option<Vec<G::Scalar>>, // inputs to the circuit
  sc: SC,                      // step circuit to be executed
}

impl<G: Group, SC: StepCircuit<G::Scalar>> Circuit<G::Scalar> for SpartanCircuit<G, SC> {
  fn synthesize<CS: ConstraintSystem<G::Scalar>>(self, cs: &mut CS) -> Result<(), SynthesisError> {
    // obtain the arity information
    let arity = self.sc.arity();

    // Allocate zi. If inputs.zi is not provided, allocate default value 0
    let zero = vec![G::Scalar::zero(); arity];
    let z_i = (0..arity)
      .map(|i| {
        AllocatedNum::alloc(cs.namespace(|| format!("zi_{i}")), || {
          Ok(self.z_i.as_ref().unwrap_or(&zero)[i])
        })
      })
      .collect::<Result<Vec<AllocatedNum<G::Scalar>>, _>>()?;

    let z_i_plus_one = self.sc.synthesize(&mut cs.namespace(|| "F"), &z_i)?;

    // inputize both z_i and z_i_plus_one
    for (j, input) in z_i.iter().enumerate().take(arity) {
      let _ = input.inputize(cs.namespace(|| format!("input {j}")));
    }
    for (j, output) in z_i_plus_one.iter().enumerate().take(arity) {
      let _ = output.inputize(cs.namespace(|| format!("output {j}")));
    }

    Ok(())
  }
}

/// A type that holds Spartan's prover key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanProverKey<G, EE>
where
  G: Group,
  EE: EvaluationEngineTrait<G, CE = G::CE>,
{
  F_arity: usize,
  S: R1CSShape<G>,
  ck: CommitmentKey<G>,
  pk: ProverKey<G, EE>,
}

/// A type that holds Spartan's verifier key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanVerifierKey<G, EE>
where
  G: Group,
  EE: EvaluationEngineTrait<G, CE = G::CE>,
{
  F_arity: usize,
  vk: VerifierKey<G, EE>,
}

/// A direct SNARK proving a step circuit
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SpartanSNARK<G, EE, C>
where
  G: Group,
  EE: EvaluationEngineTrait<G, CE = G::CE>,
  C: StepCircuit<G::Scalar>,
{
  comm_W: Commitment<G>,          // commitment to the witness
  snark: RelaxedR1CSSNARK<G, EE>, // snark proving the witness is satisfying
  _p: PhantomData<C>,
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>, C: StepCircuit<G::Scalar>>
  SpartanSNARK<G, EE, C>
{
  /// Produces prover and verifier keys for Spartan
  pub fn setup(sc: C) -> Result<(SpartanProverKey<G, EE>, SpartanVerifierKey<G, EE>), NovaError> {
    let F_arity = sc.arity();

    // construct a circuit that can be synthesized
    let circuit: SpartanCircuit<G, C> = SpartanCircuit { z_i: None, sc };

    let mut cs: ShapeCS<G> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (S, ck) = cs.r1cs_shape();

    let (pk, vk) = RelaxedR1CSSNARK::setup(&ck, &S)?;

    let pk = SpartanProverKey { F_arity, S, ck, pk };

    let vk = SpartanVerifierKey { F_arity, vk };

    Ok((pk, vk))
  }

  /// Produces a proof of satisfiability of the provided circuit
  pub fn prove(
    pk: &SpartanProverKey<G, EE>,
    sc: C,
    z_i: Vec<G::Scalar>,
  ) -> Result<Self, NovaError> {
    if z_i.len() != pk.F_arity || sc.output(&z_i).len() != pk.F_arity {
      return Err(NovaError::InvalidInitialInputLength);
    }

    let mut cs: SatisfyingAssignment<G> = SatisfyingAssignment::new();

    let circuit: SpartanCircuit<G, C> = SpartanCircuit { z_i: Some(z_i), sc };

    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs
      .r1cs_instance_and_witness(&pk.S, &pk.ck)
      .map_err(|_e| NovaError::UnSat)?;

    // convert the instance and witness to relaxed form
    let (u_relaxed, w_relaxed) = (
      RelaxedR1CSInstance::from_r1cs_instance_unchecked(&u.comm_W, &u.X),
      RelaxedR1CSWitness::from_r1cs_witness(&pk.S, &w),
    );

    // prove the instance using Spartan
    let snark = RelaxedR1CSSNARK::prove(&pk.ck, &pk.pk, &u_relaxed, &w_relaxed)?;

    Ok(SpartanSNARK {
      comm_W: u.comm_W,
      snark,
      _p: Default::default(),
    })
  }

  /// Verifies a proof of satisfiability
  pub fn verify(
    &self,
    vk: &SpartanVerifierKey<G, EE>,
    z_i: Vec<G::Scalar>,
    z_i_plus_one: Vec<G::Scalar>,
  ) -> Result<(), NovaError> {
    // check if z_i and z_i_plus_one have lengths according to the provided arity
    if z_i.len() != vk.F_arity || z_i_plus_one.len() != vk.F_arity {
      return Err(NovaError::InvalidInitialInputLength);
    }

    // construct an instance using the provided commitment to the witness and z_i and z_{i+1}
    let u_relaxed = RelaxedR1CSInstance::from_r1cs_instance_unchecked(
      &self.comm_W,
      &z_i
        .into_iter()
        .chain(z_i_plus_one.into_iter())
        .collect::<Vec<G::Scalar>>(),
    );

    // verify the snark using the constructed instance
    self.snark.verify(&vk.vk, &u_relaxed)?;

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  type G = pasta_curves::pallas::Point;
  type EE = crate::provider::ipa_pc::EvaluationEngine<G>;
  use ::bellperson::{gadgets::num::AllocatedNum, ConstraintSystem, SynthesisError};
  use core::marker::PhantomData;
  use ff::PrimeField;

  #[derive(Clone, Debug, Default)]
  struct CubicCircuit<F: PrimeField> {
    _p: PhantomData<F>,
  }

  impl<F> StepCircuit<F> for CubicCircuit<F>
  where
    F: PrimeField,
  {
    fn arity(&self) -> usize {
      1
    }

    fn synthesize<CS: ConstraintSystem<F>>(
      &self,
      cs: &mut CS,
      z: &[AllocatedNum<F>],
    ) -> Result<Vec<AllocatedNum<F>>, SynthesisError> {
      // Consider a cubic equation: `x^3 + x + 5 = y`, where `x` and `y` are respectively the input and output.
      let x = &z[0];
      let x_sq = x.square(cs.namespace(|| "x_sq"))?;
      let x_cu = x_sq.mul(cs.namespace(|| "x_cu"), x)?;
      let y = AllocatedNum::alloc(cs.namespace(|| "y"), || {
        Ok(x_cu.get_value().unwrap() + x.get_value().unwrap() + F::from(5u64))
      })?;

      cs.enforce(
        || "y = x^3 + x + 5",
        |lc| {
          lc + x_cu.get_variable()
            + x.get_variable()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
            + CS::one()
        },
        |lc| lc + CS::one(),
        |lc| lc + y.get_variable(),
      );

      Ok(vec![y])
    }

    fn output(&self, z: &[F]) -> Vec<F> {
      vec![z[0] * z[0] * z[0] + z[0] + F::from(5u64)]
    }
  }

  #[test]
  fn test_spartan_snark() {
    let circuit = CubicCircuit::default();

    // produce keys
    let (pk, vk) =
      SpartanSNARK::<G, EE, CubicCircuit<<G as Group>::Scalar>>::setup(circuit.clone()).unwrap();

    let num_steps = 3;

    // setup inputs
    let z0 = vec![<G as Group>::Scalar::zero()];
    let mut z_i = z0;

    for _i in 0..num_steps {
      // produce a SNARK
      let res = SpartanSNARK::prove(&pk, circuit.clone(), z_i.clone());
      assert!(res.is_ok());

      let z_i_plus_one = circuit.output(&z_i);

      let snark = res.unwrap();

      // verify the SNARK
      let res = snark.verify(&vk, z_i.clone(), z_i_plus_one.clone());
      assert!(res.is_ok());

      // set input to the next step
      z_i = z_i_plus_one.clone();
    }

    // sanity: check the claimed output with a direct computation of the same
    assert_eq!(z_i, vec![<G as Group>::Scalar::from(2460515u64)]);
  }
}
