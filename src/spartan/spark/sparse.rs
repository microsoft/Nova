#![allow(clippy::type_complexity)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::needless_range_loop)]
use crate::{
  errors::NovaError,
  spartan::{
    math::Math,
    polynomial::{EqPolynomial, MultilinearPolynomial},
    spark::product::{IdentityPolynomial, ProductArgumentBatched},
    SumcheckProof,
  },
  traits::{
    commitment::CommitmentEngineTrait, evaluation::EvaluationEngineTrait, Group,
    TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, unsafe_serde,
};
use abomonation::Abomonation;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds a sparse polynomial in dense representation
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparsePolynomial<G: Group> {
  ell: (usize, usize), // number of variables in each dimension

  // dense representation
  row: Vec<G::Scalar>,
  col: Vec<G::Scalar>,
  val: Vec<G::Scalar>,

  // timestamp polynomials
  row_read_ts: Vec<G::Scalar>,
  row_audit_ts: Vec<G::Scalar>,
  col_read_ts: Vec<G::Scalar>,
  col_audit_ts: Vec<G::Scalar>,
}

impl<G: Group> Abomonation for SparsePolynomial<G> {
  #[inline]
  unsafe fn entomb<W: std::io::Write>(&self, bytes: &mut W) -> std::io::Result<()> {
    self.ell.entomb(bytes)?;
    unsafe_serde::entomb_vec_T(&self.row, bytes)?;
    unsafe_serde::entomb_vec_T(&self.col, bytes)?;
    unsafe_serde::entomb_vec_T(&self.val, bytes)?;
    unsafe_serde::entomb_vec_T(&self.row_read_ts, bytes)?;
    unsafe_serde::entomb_vec_T(&self.row_audit_ts, bytes)?;
    unsafe_serde::entomb_vec_T(&self.col_read_ts, bytes)?;
    unsafe_serde::entomb_vec_T(&self.col_audit_ts, bytes)?;
    Ok(())
  }

  #[inline]
  unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
    let temp = bytes; bytes = self.ell.exhume(temp)?;
    let temp = bytes; bytes = unsafe_serde::exhume_vec_T(&mut self.row, temp)?;
    let temp = bytes; bytes = unsafe_serde::exhume_vec_T(&mut self.col, temp)?;
    let temp = bytes; bytes = unsafe_serde::exhume_vec_T(&mut self.val, temp)?;
    let temp = bytes; bytes = unsafe_serde::exhume_vec_T(&mut self.row_read_ts, temp)?;
    let temp = bytes; bytes = unsafe_serde::exhume_vec_T(&mut self.row_audit_ts, temp)?;
    let temp = bytes; bytes = unsafe_serde::exhume_vec_T(&mut self.col_read_ts, temp)?;
    let temp = bytes; bytes = unsafe_serde::exhume_vec_T(&mut self.col_audit_ts, temp)?;
    Some(bytes)
  }

  #[inline]
  fn extent(&self) -> usize {
    let mut size = 0;
    size += self.ell.extent();
    size += unsafe_serde::extent_vec_T(&self.row);
    size += unsafe_serde::extent_vec_T(&self.col);
    size += unsafe_serde::extent_vec_T(&self.val);
    size += unsafe_serde::extent_vec_T(&self.row_read_ts);
    size += unsafe_serde::extent_vec_T(&self.row_audit_ts);
    size += unsafe_serde::extent_vec_T(&self.col_read_ts);
    size += unsafe_serde::extent_vec_T(&self.col_audit_ts);
    size
  }
}

/// A type that holds a commitment to a sparse polynomial
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparsePolynomialCommitment<G: Group> {
  ell: (usize, usize), // number of variables
  size: usize,         // size of the dense representation

  // commitments to the dense representation
  comm_row: Commitment<G>,
  comm_col: Commitment<G>,
  comm_val: Commitment<G>,

  // commitments to the timestamp polynomials
  comm_row_read_ts: Commitment<G>,
  comm_row_audit_ts: Commitment<G>,
  comm_col_read_ts: Commitment<G>,
  comm_col_audit_ts: Commitment<G>,
}

impl<G: Group> Abomonation for SparsePolynomialCommitment<G> {
  #[inline]
  unsafe fn entomb<W: std::io::Write>(&self, bytes: &mut W) -> std::io::Result<()> {
    self.ell.entomb(bytes)?;
    self.size.entomb(bytes)?;
    self.comm_row.entomb(bytes)?;
    self.comm_col.entomb(bytes)?;
    self.comm_val.entomb(bytes)?;
    self.comm_row_read_ts.entomb(bytes)?;
    self.comm_row_audit_ts.entomb(bytes)?;
    self.comm_col_read_ts.entomb(bytes)?;
    self.comm_col_audit_ts.entomb(bytes)?;
    Ok(())
  }

  #[inline]
  unsafe fn exhume<'a, 'b>(&'a mut self, mut bytes: &'b mut [u8]) -> Option<&'b mut [u8]> {
    let temp = bytes; bytes = self.ell.exhume(temp)?;
    let temp = bytes; bytes = self.size.exhume(temp)?;
    let temp = bytes; bytes = self.comm_row.exhume(temp)?;
    let temp = bytes; bytes = self.comm_col.exhume(temp)?;
    let temp = bytes; bytes = self.comm_val.exhume(temp)?;
    let temp = bytes; bytes = self.comm_row_read_ts.exhume(temp)?;
    let temp = bytes; bytes = self.comm_row_audit_ts.exhume(temp)?;
    let temp = bytes; bytes = self.comm_col_read_ts.exhume(temp)?;
    let temp = bytes; bytes = self.comm_col_audit_ts.exhume(temp)?;
    Some(bytes)
  }

  #[inline]
  fn extent(&self) -> usize {
    let mut size = 0;
    size += self.ell.extent();
    size += self.size.extent();
    size += self.comm_row.extent();
    size += self.comm_col.extent();
    size += self.comm_val.extent();
    size += self.comm_row_read_ts.extent();
    size += self.comm_row_audit_ts.extent();
    size += self.comm_col_read_ts.extent();
    size += self.comm_col_audit_ts.extent();
    size
  }
}

impl<G: Group> TranscriptReprTrait<G> for SparsePolynomialCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_row,
      self.comm_col,
      self.comm_val,
      self.comm_row_read_ts,
      self.comm_row_audit_ts,
      self.comm_col_read_ts,
      self.comm_col_audit_ts,
    ]
    .as_slice()
    .to_transcript_bytes()
  }
}

impl<G: Group> SparsePolynomial<G> {
  pub fn new(ell: (usize, usize), M: &[(usize, usize, G::Scalar)]) -> Self {
    let mut row = M.iter().map(|(r, _, _)| *r).collect::<Vec<usize>>();
    let mut col = M.iter().map(|(_, c, _)| *c).collect::<Vec<usize>>();
    let mut val = M.iter().map(|(_, _, v)| *v).collect::<Vec<G::Scalar>>();

    let num_ops = M.len().next_power_of_two();
    let num_cells_row = ell.0.pow2();
    let num_cells_col = ell.1.pow2();
    row.resize(num_ops, 0usize);
    col.resize(num_ops, 0usize);
    val.resize(num_ops, G::Scalar::zero());

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
    let (row_read_ts, row_audit_ts) = timestamp_calc(num_ops, num_cells_row, &row);
    let (col_read_ts, col_audit_ts) = timestamp_calc(num_ops, num_cells_col, &col);

    let to_vec_scalar = |v: &[usize]| -> Vec<G::Scalar> {
      (0..v.len())
        .map(|i| G::Scalar::from(v[i] as u64))
        .collect::<Vec<G::Scalar>>()
    };

    Self {
      ell,
      // dense representation
      row: to_vec_scalar(&row),
      col: to_vec_scalar(&col),
      val,

      // timestamp polynomials
      row_read_ts: to_vec_scalar(&row_read_ts),
      row_audit_ts: to_vec_scalar(&row_audit_ts),
      col_read_ts: to_vec_scalar(&col_read_ts),
      col_audit_ts: to_vec_scalar(&col_audit_ts),
    }
  }

  pub fn commit(&self, ck: &CommitmentKey<G>) -> SparsePolynomialCommitment<G> {
    let comm_vec: Vec<Commitment<G>> = [
      &self.row,
      &self.col,
      &self.val,
      &self.row_read_ts,
      &self.row_audit_ts,
      &self.col_read_ts,
      &self.col_audit_ts,
    ]
    .par_iter()
    .map(|v| G::CE::commit(ck, v))
    .collect();

    SparsePolynomialCommitment {
      ell: self.ell,
      size: self.row.len(),
      comm_row: comm_vec[0],
      comm_col: comm_vec[1],
      comm_val: comm_vec[2],
      comm_row_read_ts: comm_vec[3],
      comm_row_audit_ts: comm_vec[4],
      comm_col_read_ts: comm_vec[5],
      comm_col_audit_ts: comm_vec[6],
    }
  }

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

  fn evaluation_oracles(
    M: &[(usize, usize, G::Scalar)],
    r_x: &[G::Scalar],
    r_y: &[G::Scalar],
  ) -> (
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
  ) {
    let evaluation_oracles_with_table = |M: &[(usize, usize, G::Scalar)],
                                         T_x: &[G::Scalar],
                                         T_y: &[G::Scalar]|
     -> (Vec<G::Scalar>, Vec<G::Scalar>) {
      (0..M.len())
        .collect::<Vec<usize>>()
        .par_iter()
        .map(|&i| {
          let (row, col, _val) = M[i];
          (T_x[row], T_y[col])
        })
        .collect::<Vec<(G::Scalar, G::Scalar)>>()
        .into_par_iter()
        .unzip()
    };

    let (T_x, T_y) = rayon::join(
      || EqPolynomial::new(r_x.to_vec()).evals(),
      || EqPolynomial::new(r_y.to_vec()).evals(),
    );

    let (mut E_row, mut E_col) = evaluation_oracles_with_table(M, &T_x, &T_y);

    // resize the returned vectors
    E_row.resize(M.len().next_power_of_two(), T_x[0]); // we place T_x[0] since resized row is appended with 0s
    E_col.resize(M.len().next_power_of_two(), T_y[0]);
    (E_row, E_col, T_x, T_y)
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparseEvaluationArgument<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  // claimed evaluation
  eval: G::Scalar,

  // oracles
  comm_E_row: Commitment<G>,
  comm_E_col: Commitment<G>,

  // proof of correct evaluation wrt oracles
  sc_proof_eval: SumcheckProof<G>,
  eval_E_row: G::Scalar,
  eval_E_col: G::Scalar,
  eval_val: G::Scalar,
  arg_eval: EE::EvaluationArgument,

  // proof that E_row is well-formed
  eval_init_row: G::Scalar,
  eval_read_row: G::Scalar,
  eval_write_row: G::Scalar,
  eval_audit_row: G::Scalar,
  eval_init_col: G::Scalar,
  eval_read_col: G::Scalar,
  eval_write_col: G::Scalar,
  eval_audit_col: G::Scalar,
  sc_prod_init_audit_row: ProductArgumentBatched<G>,
  sc_prod_read_write_row_col: ProductArgumentBatched<G>,
  sc_prod_init_audit_col: ProductArgumentBatched<G>,
  eval_row: G::Scalar,
  eval_row_read_ts: G::Scalar,
  eval_E_row2: G::Scalar,
  eval_row_audit_ts: G::Scalar,
  eval_col: G::Scalar,
  eval_col_read_ts: G::Scalar,
  eval_E_col2: G::Scalar,
  eval_col_audit_ts: G::Scalar,
  arg_row_col_joint: EE::EvaluationArgument,
  arg_row_audit_ts: EE::EvaluationArgument,
  arg_col_audit_ts: EE::EvaluationArgument,
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> SparseEvaluationArgument<G, EE> {
  pub fn prove(
    ck: &CommitmentKey<G>,
    pk_ee: &EE::ProverKey,
    poly: &SparsePolynomial<G>,
    sparse: &[(usize, usize, G::Scalar)],
    comm: &SparsePolynomialCommitment<G>,
    r: &(&[G::Scalar], &[G::Scalar]),
    transcript: &mut G::TE,
  ) -> Result<Self, NovaError> {
    let (r_x, r_y) = r;
    let eval = SparsePolynomial::<G>::multi_evaluate(&[sparse], r_x, r_y)[0];

    // compute oracles to prove the correctness of `eval`
    let (E_row, E_col, T_x, T_y) = SparsePolynomial::<G>::evaluation_oracles(sparse, r_x, r_y);
    let val = poly.val.clone();

    // commit to the two oracles
    let comm_E_row = G::CE::commit(ck, &E_row);
    let comm_E_col = G::CE::commit(ck, &E_col);

    // absorb the commitments and the claimed evaluation
    transcript.absorb(b"E", &vec![comm_E_row, comm_E_col].as_slice());
    transcript.absorb(b"e", &eval);

    let comb_func_eval = |poly_A_comp: &G::Scalar,
                          poly_B_comp: &G::Scalar,
                          poly_C_comp: &G::Scalar|
     -> G::Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };
    let (sc_proof_eval, r_eval, claims_eval) = SumcheckProof::<G>::prove_cubic(
      &eval,
      E_row.len().log_2(), // number of rounds
      &mut MultilinearPolynomial::new(E_row.clone()),
      &mut MultilinearPolynomial::new(E_col.clone()),
      &mut MultilinearPolynomial::new(val.clone()),
      comb_func_eval,
      transcript,
    )?;

    // prove evaluations of E_row, E_col and val at r_eval
    let rho = transcript.squeeze(b"r")?;
    let comm_joint = comm_E_row + comm_E_col * rho + comm.comm_val * rho * rho;
    let eval_joint = claims_eval[0] + rho * claims_eval[1] + rho * rho * claims_eval[2];
    let poly_eval = E_row
      .iter()
      .zip(E_col.iter())
      .zip(val.iter())
      .map(|((a, b), c)| *a + rho * *b + rho * rho * *c)
      .collect::<Vec<G::Scalar>>();
    let arg_eval = EE::prove(
      ck,
      pk_ee,
      transcript,
      &comm_joint,
      &poly_eval,
      &r_eval,
      &eval_joint,
    )?;

    // we now need to prove that E_row and E_col are well-formed
    // we use memory checking: H(INIT) * H(WS) =? H(RS) * H(FINAL)
    let gamma_1 = transcript.squeeze(b"g1")?;
    let gamma_2 = transcript.squeeze(b"g2")?;

    let gamma_1_sqr = gamma_1 * gamma_1;
    let hash_func = |addr: &G::Scalar, val: &G::Scalar, ts: &G::Scalar| -> G::Scalar {
      (*ts * gamma_1_sqr + *val * gamma_1 + *addr) - gamma_2
    };

    let init_row = (0..T_x.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &T_x[i], &G::Scalar::zero()))
      .collect::<Vec<G::Scalar>>();
    let read_row = (0..E_row.len())
      .map(|i| hash_func(&poly.row[i], &E_row[i], &poly.row_read_ts[i]))
      .collect::<Vec<G::Scalar>>();
    let write_row = (0..E_row.len())
      .map(|i| {
        hash_func(
          &poly.row[i],
          &E_row[i],
          &(poly.row_read_ts[i] + G::Scalar::one()),
        )
      })
      .collect::<Vec<G::Scalar>>();
    let audit_row = (0..T_x.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &T_x[i], &poly.row_audit_ts[i]))
      .collect::<Vec<G::Scalar>>();
    let init_col = (0..T_y.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &T_y[i], &G::Scalar::zero()))
      .collect::<Vec<G::Scalar>>();
    let read_col = (0..E_col.len())
      .map(|i| hash_func(&poly.col[i], &E_col[i], &poly.col_read_ts[i]))
      .collect::<Vec<G::Scalar>>();
    let write_col = (0..E_col.len())
      .map(|i| {
        hash_func(
          &poly.col[i],
          &E_col[i],
          &(poly.col_read_ts[i] + G::Scalar::one()),
        )
      })
      .collect::<Vec<G::Scalar>>();
    let audit_col = (0..T_y.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &T_y[i], &poly.col_audit_ts[i]))
      .collect::<Vec<G::Scalar>>();

    let (sc_prod_init_audit_row, eval_init_audit_row, r_init_audit_row) =
      ProductArgumentBatched::prove(
        &[
          &MultilinearPolynomial::new(init_row),
          &MultilinearPolynomial::new(audit_row),
        ],
        transcript,
      )?;

    assert_eq!(init_col.len(), audit_col.len());
    let (sc_prod_init_audit_col, eval_init_audit_col, r_init_audit_col) =
      ProductArgumentBatched::prove(
        &[
          &MultilinearPolynomial::new(init_col),
          &MultilinearPolynomial::new(audit_col),
        ],
        transcript,
      )?;

    assert_eq!(read_row.len(), write_row.len());
    assert_eq!(read_row.len(), read_col.len());
    assert_eq!(read_row.len(), write_col.len());

    let (sc_prod_read_write_row_col, eval_read_write_row_col, r_read_write_row_col) =
      ProductArgumentBatched::prove(
        &[
          &MultilinearPolynomial::new(read_row),
          &MultilinearPolynomial::new(write_row),
          &MultilinearPolynomial::new(read_col),
          &MultilinearPolynomial::new(write_col),
        ],
        transcript,
      )?;

    // row-related claims of polynomial evaluations to aid the final check of the sum-check
    let eval_row = MultilinearPolynomial::evaluate_with(&poly.row, &r_read_write_row_col);
    let eval_row_read_ts =
      MultilinearPolynomial::evaluate_with(&poly.row_read_ts, &r_read_write_row_col);
    let eval_E_row2 = MultilinearPolynomial::evaluate_with(&E_row, &r_read_write_row_col);
    let eval_row_audit_ts =
      MultilinearPolynomial::evaluate_with(&poly.row_audit_ts, &r_init_audit_row);

    // col-related claims of polynomial evaluations to aid the final check of the sum-check
    let eval_col = MultilinearPolynomial::evaluate_with(&poly.col, &r_read_write_row_col);
    let eval_col_read_ts =
      MultilinearPolynomial::evaluate_with(&poly.col_read_ts, &r_read_write_row_col);
    let eval_E_col2 = MultilinearPolynomial::evaluate_with(&E_col, &r_read_write_row_col);
    let eval_col_audit_ts =
      MultilinearPolynomial::evaluate_with(&poly.col_audit_ts, &r_init_audit_col);

    // we can batch prove the first three claims
    transcript.absorb(
      b"e",
      &[
        eval_row,
        eval_row_read_ts,
        eval_E_row2,
        eval_col,
        eval_col_read_ts,
        eval_E_col2,
      ]
      .as_slice(),
    );
    let c = transcript.squeeze(b"c")?;
    let eval_joint = eval_row
      + c * eval_row_read_ts
      + c * c * eval_E_row2
      + c * c * c * eval_col
      + c * c * c * c * eval_col_read_ts
      + c * c * c * c * c * eval_E_col2;
    let comm_joint = comm.comm_row
      + comm.comm_row_read_ts * c
      + comm_E_row * c * c
      + comm.comm_col * c * c * c
      + comm.comm_col_read_ts * c * c * c * c
      + comm_E_col * c * c * c * c * c;
    let poly_joint = poly
      .row
      .iter()
      .zip(poly.row_read_ts.iter())
      .zip(E_row.into_iter())
      .zip(poly.col.iter())
      .zip(poly.col_read_ts.iter())
      .zip(E_col.into_iter())
      .map(|(((((x, y), z), m), n), q)| {
        *x + c * y + c * c * z + c * c * c * m + c * c * c * c * n + c * c * c * c * c * q
      })
      .collect::<Vec<_>>();

    let arg_row_col_joint = EE::prove(
      ck,
      pk_ee,
      transcript,
      &comm_joint,
      &poly_joint,
      &r_read_write_row_col,
      &eval_joint,
    )?;

    let arg_row_audit_ts = EE::prove(
      ck,
      pk_ee,
      transcript,
      &comm.comm_row_audit_ts,
      &poly.row_audit_ts,
      &r_init_audit_row,
      &eval_row_audit_ts,
    )?;

    let arg_col_audit_ts = EE::prove(
      ck,
      pk_ee,
      transcript,
      &comm.comm_col_audit_ts,
      &poly.col_audit_ts,
      &r_init_audit_col,
      &eval_col_audit_ts,
    )?;

    Ok(Self {
      // claimed evaluation
      eval,

      // oracles
      comm_E_row,
      comm_E_col,

      // proof of correct evaluation wrt oracles
      sc_proof_eval,
      eval_E_row: claims_eval[0],
      eval_E_col: claims_eval[1],
      eval_val: claims_eval[2],
      arg_eval,

      // proof that E_row and E_row are well-formed
      eval_init_row: eval_init_audit_row[0],
      eval_read_row: eval_read_write_row_col[0],
      eval_write_row: eval_read_write_row_col[1],
      eval_audit_row: eval_init_audit_row[1],
      eval_init_col: eval_init_audit_col[0],
      eval_read_col: eval_read_write_row_col[2],
      eval_write_col: eval_read_write_row_col[3],
      eval_audit_col: eval_init_audit_col[1],
      sc_prod_init_audit_row,
      sc_prod_read_write_row_col,
      sc_prod_init_audit_col,
      eval_row,
      eval_row_read_ts,
      eval_E_row2,
      eval_row_audit_ts,
      eval_col,
      eval_col_read_ts,
      eval_E_col2,
      eval_col_audit_ts,
      arg_row_col_joint,
      arg_row_audit_ts,
      arg_col_audit_ts,
    })
  }

  pub fn verify(
    &self,
    vk_ee: &EE::VerifierKey,
    comm: &SparsePolynomialCommitment<G>,
    r: &(&[G::Scalar], &[G::Scalar]),
    transcript: &mut G::TE,
  ) -> Result<G::Scalar, NovaError> {
    let (r_x, r_y) = r;

    // append the transcript and scalar
    transcript.absorb(b"E", &vec![self.comm_E_row, self.comm_E_col].as_slice());
    transcript.absorb(b"e", &self.eval);

    // (1) verify the correct evaluation of sparse polynomial
    let (claim_eval_final, r_eval) = self.sc_proof_eval.verify(
      self.eval,
      comm.size.next_power_of_two().log_2(),
      3,
      transcript,
    )?;
    // verify the last step of the sum-check
    if claim_eval_final != self.eval_E_row * self.eval_E_col * self.eval_val {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // prove evaluations of E_row, E_col and val at r_eval
    let rho = transcript.squeeze(b"r")?;
    let comm_joint = self.comm_E_row + self.comm_E_col * rho + comm.comm_val * rho * rho;
    let eval_joint = self.eval_E_row + rho * self.eval_E_col + rho * rho * self.eval_val;
    EE::verify(
      vk_ee,
      transcript,
      &comm_joint,
      &r_eval,
      &eval_joint,
      &self.arg_eval,
    )?;

    // (2) verify if E_row and E_col are well formed
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
    let (claim_init_audit_row, r_init_audit_row) = self.sc_prod_init_audit_row.verify(
      &[self.eval_init_row, self.eval_audit_row],
      comm.ell.0.pow2(),
      transcript,
    )?;
    let (claim_init_audit_col, r_init_audit_col) = self.sc_prod_init_audit_col.verify(
      &[self.eval_init_col, self.eval_audit_col],
      comm.ell.1.pow2(),
      transcript,
    )?;
    let (claim_read_write_row_col, r_read_write_row_col) = self.sc_prod_read_write_row_col.verify(
      &[
        self.eval_read_row,
        self.eval_write_row,
        self.eval_read_col,
        self.eval_write_col,
      ],
      comm.size,
      transcript,
    )?;

    // finish the final step of the three sum-checks
    let (claim_init_expected_row, claim_audit_expected_row) = {
      let addr = IdentityPolynomial::new(r_init_audit_row.len()).evaluate(&r_init_audit_row);
      let val = EqPolynomial::new(r_x.to_vec()).evaluate(&r_init_audit_row);

      (
        hash_func(&addr, &val, &G::Scalar::zero()),
        hash_func(&addr, &val, &self.eval_row_audit_ts),
      )
    };

    let (claim_read_expected_row, claim_write_expected_row) = {
      (
        hash_func(&self.eval_row, &self.eval_E_row2, &self.eval_row_read_ts),
        hash_func(
          &self.eval_row,
          &self.eval_E_row2,
          &(self.eval_row_read_ts + G::Scalar::one()),
        ),
      )
    };

    // multiset check for the row
    if claim_init_expected_row != claim_init_audit_row[0]
      || claim_audit_expected_row != claim_init_audit_row[1]
      || claim_read_expected_row != claim_read_write_row_col[0]
      || claim_write_expected_row != claim_read_write_row_col[1]
    {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let (claim_init_expected_col, claim_audit_expected_col) = {
      let addr = IdentityPolynomial::new(r_init_audit_col.len()).evaluate(&r_init_audit_col);
      let val = EqPolynomial::new(r_y.to_vec()).evaluate(&r_init_audit_col);

      (
        hash_func(&addr, &val, &G::Scalar::zero()),
        hash_func(&addr, &val, &self.eval_col_audit_ts),
      )
    };

    let (claim_read_expected_col, claim_write_expected_col) = {
      (
        hash_func(&self.eval_col, &self.eval_E_col2, &self.eval_col_read_ts),
        hash_func(
          &self.eval_col,
          &self.eval_E_col2,
          &(self.eval_col_read_ts + G::Scalar::one()),
        ),
      )
    };

    // multiset check for the col
    if claim_init_expected_col != claim_init_audit_col[0]
      || claim_audit_expected_col != claim_init_audit_col[1]
      || claim_read_expected_col != claim_read_write_row_col[2]
      || claim_write_expected_col != claim_read_write_row_col[3]
    {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(
      b"e",
      &[
        self.eval_row,
        self.eval_row_read_ts,
        self.eval_E_row2,
        self.eval_col,
        self.eval_col_read_ts,
        self.eval_E_col2,
      ]
      .as_slice(),
    );
    let c = transcript.squeeze(b"c")?;
    let eval_joint = self.eval_row
      + c * self.eval_row_read_ts
      + c * c * self.eval_E_row2
      + c * c * c * self.eval_col
      + c * c * c * c * self.eval_col_read_ts
      + c * c * c * c * c * self.eval_E_col2;
    let comm_joint = comm.comm_row
      + comm.comm_row_read_ts * c
      + self.comm_E_row * c * c
      + comm.comm_col * c * c * c
      + comm.comm_col_read_ts * c * c * c * c
      + self.comm_E_col * c * c * c * c * c;

    EE::verify(
      vk_ee,
      transcript,
      &comm_joint,
      &r_read_write_row_col,
      &eval_joint,
      &self.arg_row_col_joint,
    )?;

    EE::verify(
      vk_ee,
      transcript,
      &comm.comm_row_audit_ts,
      &r_init_audit_row,
      &self.eval_row_audit_ts,
      &self.arg_row_audit_ts,
    )?;

    EE::verify(
      vk_ee,
      transcript,
      &comm.comm_col_audit_ts,
      &r_init_audit_col,
      &self.eval_col_audit_ts,
      &self.arg_col_audit_ts,
    )?;

    Ok(self.eval)
  }
}
