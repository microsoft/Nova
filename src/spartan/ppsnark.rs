//! This module implements `RelaxedR1CSSNARK` traits using a spark-based approach to prove evaluations of
//! sparse multilinear polynomials involved in Spartan's sum-check protocol, thereby providing a preprocessing SNARK
//! The verifier in this preprocessing SNARK maintains a commitment to R1CS matrices. This is beneficial when using a
//! polynomial commitment scheme in which the verifier's costs is succinct.
use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    math::Math,
    polys::{
      eq::EqPolynomial,
      identity::IdentityPolynomial,
      multilinear::MultilinearPolynomial,
      power::PowPolynomial,
      univariate::{CompressedUniPoly, UniPoly},
    },
    powers,
    sumcheck::SumcheckProof,
    PolyEvalInstance, PolyEvalWitness, SparsePolynomial,
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    snark::{DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, CompressedCommitment,
};
use core::cmp::max;
use ff::Field;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

fn padded<G: Group>(v: &[G::Scalar], n: usize, e: &G::Scalar) -> Vec<G::Scalar> {
  let mut v_padded = vec![*e; n];
  for (i, v_i) in v.iter().enumerate() {
    v_padded[i] = *v_i;
  }
  v_padded
}

/// A type that holds `R1CSShape` in a form amenable to memory checking
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
  ts_row: Vec<G::Scalar>,
  ts_col: Vec<G::Scalar>,
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
  comm_ts_row: Commitment<G>,
  comm_ts_col: Commitment<G>,
}

impl<G: Group> TranscriptReprTrait<G> for R1CSShapeSparkCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_row,
      self.comm_col,
      self.comm_val_A,
      self.comm_val_B,
      self.comm_val_C,
      self.comm_ts_row,
      self.comm_ts_col,
    ]
    .as_slice()
    .to_transcript_bytes()
  }
}

impl<G: Group> R1CSShapeSparkRepr<G> {
  /// represents `R1CSShape` in a Spark-friendly format amenable to memory checking
  pub fn new(S: &R1CSShape<G>) -> R1CSShapeSparkRepr<G> {
    let N = {
      let total_nz = S.A.len() + S.B.len() + S.C.len();
      max(total_nz, max(2 * S.num_vars, S.num_cons)).next_power_of_two()
    };

    let (mut row, mut col) = (vec![0; N], vec![N - 1; N]); // we make col lookup into the last entry of z, so we commit to zeros

    for (i, (r, c, _)) in S.A.iter().chain(S.B.iter()).chain(S.C.iter()).enumerate() {
      row[i] = r;
      col[i] = c;
    }
    let val_A = {
      let mut val = vec![G::Scalar::ZERO; N];
      for (i, (_, _, v)) in S.A.iter().enumerate() {
        val[i] = v;
      }
      val
    };

    let val_B = {
      let mut val = vec![G::Scalar::ZERO; N];
      for (i, (_, _, v)) in S.B.iter().enumerate() {
        val[S.A.len() + i] = v;
      }
      val
    };

    let val_C = {
      let mut val = vec![G::Scalar::ZERO; N];
      for (i, (_, _, v)) in S.C.iter().enumerate() {
        val[S.A.len() + S.B.len() + i] = v;
      }
      val
    };

    // timestamp calculation routine
    let timestamp_calc = |num_ops: usize, num_cells: usize, addr_trace: &[usize]| -> Vec<usize> {
      let mut ts = vec![0usize; num_cells];

      assert!(num_ops >= addr_trace.len());
      for addr in addr_trace {
        assert!(*addr < num_cells);
        ts[*addr] += 1;
      }
      ts
    };

    // timestamp polynomials for row
    let (ts_row, ts_col) =
      rayon::join(|| timestamp_calc(N, N, &row), || timestamp_calc(N, N, &col));

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
      ts_row: to_vec_scalar(&ts_row),
      ts_col: to_vec_scalar(&ts_col),
    }
  }

  fn commit(&self, ck: &CommitmentKey<G>) -> R1CSShapeSparkCommitment<G> {
    let comm_vec: Vec<Commitment<G>> = [
      &self.row,
      &self.col,
      &self.val_A,
      &self.val_B,
      &self.val_C,
      &self.ts_row,
      &self.ts_col,
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
      comm_ts_row: comm_vec[5],
      comm_ts_col: comm_vec[6],
    }
  }

  // computes evaluation oracles
  fn evaluation_oracles(
    &self,
    S: &R1CSShape<G>,
    r_x: &G::Scalar,
    z: &[G::Scalar],
  ) -> (
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
  ) {
    let mem_row = PowPolynomial::new(r_x, self.N.log_2()).evals();
    let mem_col = padded(z, self.N, &G::Scalar::ZERO);

    let (L_row, L_col) = {
      let mut L_row = vec![mem_row[0]; self.N]; // we place mem_row[0] since resized row is appended with 0s
      let mut L_col = vec![mem_col[self.N - 1]; self.N]; // we place mem_col[N-1] since resized col is appended with N-1

      for (i, (val_r, val_c)) in S
        .A
        .iter()
        .chain(S.B.iter())
        .chain(S.C.iter())
        .map(|(r, c, _)| (mem_row[r], mem_col[c]))
        .enumerate()
      {
        L_row[i] = val_r;
        L_col[i] = val_c;
      }
      (L_row, L_col)
    };

    (mem_row, mem_col, L_row, L_col)
  }
}

/// Defines a trait for implementing sum-check in a generic manner
pub trait SumcheckEngine<G: Group>: Send + Sync {
  /// returns the initial claims
  fn initial_claims(&self) -> Vec<G::Scalar>;

  /// degree of the sum-check polynomial
  fn degree(&self) -> usize;

  /// the size of the polynomials
  fn size(&self) -> usize;

  /// returns evaluation points at 0, 2, d-1 (where d is the degree of the sum-check polynomial)
  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>>;

  /// bounds a variable in the constituent polynomials
  fn bound(&mut self, r: &G::Scalar);

  /// returns the final claims
  fn final_claims(&self) -> Vec<Vec<G::Scalar>>;
}

struct LookupSumcheckInstance<G: Group> {
  // row
  w_plus_r_row: MultilinearPolynomial<G::Scalar>,
  t_plus_r_row: MultilinearPolynomial<G::Scalar>,
  t_plus_r_inv_row: MultilinearPolynomial<G::Scalar>,
  w_plus_r_inv_row: MultilinearPolynomial<G::Scalar>,
  ts_row: MultilinearPolynomial<G::Scalar>,

  // col
  w_plus_r_col: MultilinearPolynomial<G::Scalar>,
  t_plus_r_col: MultilinearPolynomial<G::Scalar>,
  t_plus_r_inv_col: MultilinearPolynomial<G::Scalar>,
  w_plus_r_inv_col: MultilinearPolynomial<G::Scalar>,
  ts_col: MultilinearPolynomial<G::Scalar>,

  // eq
  poly_eq: MultilinearPolynomial<G::Scalar>,
}

impl<G: Group> LookupSumcheckInstance<G> {
  pub fn new(
    ck: &CommitmentKey<G>,
    r: &G::Scalar,
    T_row: Vec<G::Scalar>,
    W_row: Vec<G::Scalar>,
    ts_row: Vec<G::Scalar>,
    T_col: Vec<G::Scalar>,
    W_col: Vec<G::Scalar>,
    ts_col: Vec<G::Scalar>,
    transcript: &mut G::TE,
  ) -> Result<(Self, [Commitment<G>; 4], [Vec<G::Scalar>; 4]), NovaError> {
    let batch_invert = |v: &[G::Scalar]| -> Result<Vec<G::Scalar>, NovaError> {
      let mut products = vec![G::Scalar::ZERO; v.len()];
      let mut acc = G::Scalar::ONE;

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // we can compute an inversion only if acc is non-zero
      if acc == G::Scalar::ZERO {
        return Err(NovaError::InternalError);
      }

      // compute the inverse once for all entries
      acc = acc.invert().unwrap();

      let mut inv = vec![G::Scalar::ZERO; v.len()];
      for i in 0..v.len() {
        let tmp = acc * v[v.len() - 1 - i];
        inv[v.len() - 1 - i] = products[v.len() - 1 - i] * acc;
        acc = tmp;
      }

      Ok(inv)
    };

    // compute vectors TS[i]/(T[i] + r) and 1/(W[i] + r)
    let helper = |T: &[G::Scalar],
                  W: &[G::Scalar],
                  TS: &[G::Scalar],
                  r: &G::Scalar|
     -> (
      (
        Result<Vec<G::Scalar>, NovaError>,
        Result<Vec<G::Scalar>, NovaError>,
      ),
      (
        Result<Vec<G::Scalar>, NovaError>,
        Result<Vec<G::Scalar>, NovaError>,
      ),
    ) {
      rayon::join(
        || {
          rayon::join(
            || {
              let inv = batch_invert(&T.par_iter().map(|e| *e + *r).collect::<Vec<G::Scalar>>())?;

              // compute inv[i] * TS[i] in parallel
              Ok(
                inv
                  .par_iter()
                  .zip(TS.par_iter())
                  .map(|(e1, e2)| *e1 * *e2)
                  .collect::<Vec<_>>(),
              )
            },
            || batch_invert(&W.par_iter().map(|e| *e + *r).collect::<Vec<G::Scalar>>()),
          )
        },
        || {
          rayon::join(
            || Ok(T.par_iter().map(|e| *e + *r).collect::<Vec<G::Scalar>>()),
            || Ok(W.par_iter().map(|e| *e + *r).collect::<Vec<G::Scalar>>()),
          )
        },
      )
    };

    let (
      ((t_plus_r_inv_row, w_plus_r_inv_row), (t_plus_r_row, w_plus_r_row)),
      ((t_plus_r_inv_col, w_plus_r_inv_col), (t_plus_r_col, w_plus_r_col)),
    ) = rayon::join(
      || helper(&T_row, &W_row, &ts_row, r),
      || helper(&T_col, &W_col, &ts_col, r),
    );

    let t_plus_r_inv_row = t_plus_r_inv_row?;
    let w_plus_r_inv_row = w_plus_r_inv_row?;
    let t_plus_r_inv_col = t_plus_r_inv_col?;
    let w_plus_r_inv_col = w_plus_r_inv_col?;

    let (
      (comm_t_plus_r_inv_row, comm_w_plus_r_inv_row),
      (comm_t_plus_r_inv_col, comm_w_plus_r_inv_col),
    ) = rayon::join(
      || {
        rayon::join(
          || G::CE::commit(ck, &t_plus_r_inv_row),
          || G::CE::commit(ck, &w_plus_r_inv_row),
        )
      },
      || {
        rayon::join(
          || G::CE::commit(ck, &t_plus_r_inv_col),
          || G::CE::commit(ck, &w_plus_r_inv_col),
        )
      },
    );

    // absorb the commitments
    transcript.absorb(
      b"l",
      &[
        comm_t_plus_r_inv_row,
        comm_w_plus_r_inv_row,
        comm_t_plus_r_inv_col,
        comm_w_plus_r_inv_col,
      ]
      .as_slice(),
    );

    let rho = transcript.squeeze(b"r")?;
    let poly_eq = MultilinearPolynomial::new(PowPolynomial::new(&rho, T_row.len().log_2()).evals());

    let comm_vec = [
      comm_t_plus_r_inv_row,
      comm_w_plus_r_inv_row,
      comm_t_plus_r_inv_col,
      comm_w_plus_r_inv_col,
    ];

    let poly_vec = [
      t_plus_r_inv_row.clone(),
      w_plus_r_inv_row.clone(),
      t_plus_r_inv_col.clone(),
      w_plus_r_inv_col.clone(),
    ];

    Ok((
      Self {
        w_plus_r_row: MultilinearPolynomial::new(w_plus_r_row?),
        t_plus_r_row: MultilinearPolynomial::new(t_plus_r_row?),
        t_plus_r_inv_row: MultilinearPolynomial::new(t_plus_r_inv_row),
        w_plus_r_inv_row: MultilinearPolynomial::new(w_plus_r_inv_row),
        ts_row: MultilinearPolynomial::new(ts_row),
        w_plus_r_col: MultilinearPolynomial::new(w_plus_r_col?),
        t_plus_r_col: MultilinearPolynomial::new(t_plus_r_col?),
        t_plus_r_inv_col: MultilinearPolynomial::new(t_plus_r_inv_col),
        w_plus_r_inv_col: MultilinearPolynomial::new(w_plus_r_inv_col),
        ts_col: MultilinearPolynomial::new(ts_col),
        poly_eq,
      },
      comm_vec,
      poly_vec,
    ))
  }
}

impl<G: Group> SumcheckEngine<G> for LookupSumcheckInstance<G> {
  fn initial_claims(&self) -> Vec<G::Scalar> {
    vec![G::Scalar::ZERO; 6]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    // sanity checks
    assert_eq!(self.w_plus_r_row.len(), self.t_plus_r_row.len());
    assert_eq!(self.w_plus_r_row.len(), self.ts_row.len());
    assert_eq!(self.w_plus_r_row.len(), self.w_plus_r_col.len());
    assert_eq!(self.w_plus_r_row.len(), self.t_plus_r_col.len());
    assert_eq!(self.w_plus_r_row.len(), self.ts_col.len());

    self.w_plus_r_row.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>> {
    let poly_zero = MultilinearPolynomial::new(vec![G::Scalar::ZERO; self.w_plus_r_row.len()]);

    let comb_func = |poly_A_comp: &G::Scalar,
                     poly_B_comp: &G::Scalar,
                     _poly_C_comp: &G::Scalar|
     -> G::Scalar { *poly_A_comp - *poly_B_comp };

    let comb_func2 =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       _poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - G::Scalar::ONE) };

    let comb_func3 =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    // inv related evaluation points
    let (eval_inv_0_row, eval_inv_2_row, eval_inv_3_row) =
      SumcheckProof::<G>::compute_eval_points_cubic(
        &self.t_plus_r_inv_row,
        &self.w_plus_r_inv_row,
        &poly_zero,
        &comb_func,
      );

    let (eval_inv_0_col, eval_inv_2_col, eval_inv_3_col) =
      SumcheckProof::<G>::compute_eval_points_cubic(
        &self.t_plus_r_inv_col,
        &self.w_plus_r_inv_col,
        &poly_zero,
        &comb_func,
      );

    // row related evaluation points
    let (eval_T_0_row, eval_T_2_row, eval_T_3_row) =
      SumcheckProof::<G>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.t_plus_r_inv_row,
        &self.t_plus_r_row,
        &self.ts_row,
        &comb_func3,
      );
    let (eval_W_0_row, eval_W_2_row, eval_W_3_row) =
      SumcheckProof::<G>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.w_plus_r_inv_row,
        &self.w_plus_r_row,
        &poly_zero,
        &comb_func2,
      );

    // column related evaluation points
    let (eval_T_0_col, eval_T_2_col, eval_T_3_col) =
      SumcheckProof::<G>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.t_plus_r_inv_col,
        &self.t_plus_r_col,
        &self.ts_col,
        &comb_func3,
      );
    let (eval_W_0_col, eval_W_2_col, eval_W_3_col) =
      SumcheckProof::<G>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.w_plus_r_inv_col,
        &self.w_plus_r_col,
        &poly_zero,
        &comb_func2,
      );

    vec![
      vec![eval_inv_0_row, eval_inv_2_row, eval_inv_3_row],
      vec![eval_inv_0_col, eval_inv_2_col, eval_inv_3_col],
      vec![eval_T_0_row, eval_T_2_row, eval_T_3_row],
      vec![eval_W_0_row, eval_W_2_row, eval_W_3_row],
      vec![eval_T_0_col, eval_T_2_col, eval_T_3_col],
      vec![eval_W_0_col, eval_W_2_col, eval_W_3_col],
    ]
  }

  fn bound(&mut self, r: &G::Scalar) {
    self.poly_eq.bound_poly_var_top(r);

    self.t_plus_r_row.bound_poly_var_top(r);
    self.t_plus_r_inv_row.bound_poly_var_top(r);
    self.w_plus_r_row.bound_poly_var_top(r);
    self.w_plus_r_inv_row.bound_poly_var_top(r);
    self.ts_row.bound_poly_var_top(r);

    self.t_plus_r_col.bound_poly_var_top(r);
    self.t_plus_r_inv_col.bound_poly_var_top(r);
    self.w_plus_r_col.bound_poly_var_top(r);
    self.w_plus_r_inv_col.bound_poly_var_top(r);
    self.ts_col.bound_poly_var_top(r);
  }

  fn final_claims(&self) -> Vec<Vec<G::Scalar>> {
    let poly_row_final = vec![
      self.t_plus_r_inv_row[0],
      self.w_plus_r_inv_row[0],
      self.ts_row[0],
    ];

    let poly_col_final = vec![
      self.t_plus_r_inv_col[0],
      self.w_plus_r_inv_col[0],
      self.ts_col[0],
    ];

    vec![poly_row_final, poly_col_final]
  }
}

struct OuterSumcheckInstance<G: Group> {
  poly_tau: MultilinearPolynomial<G::Scalar>,
  poly_Az: MultilinearPolynomial<G::Scalar>,
  poly_Bz: MultilinearPolynomial<G::Scalar>,
  poly_uCz_E: MultilinearPolynomial<G::Scalar>,
}

impl<G: Group> SumcheckEngine<G> for OuterSumcheckInstance<G> {
  fn initial_claims(&self) -> Vec<G::Scalar> {
    vec![G::Scalar::ZERO]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_tau.len(), self.poly_Az.len());
    assert_eq!(self.poly_tau.len(), self.poly_Bz.len());
    assert_eq!(self.poly_tau.len(), self.poly_uCz_E.len());
    self.poly_tau.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>> {
    let (poly_A, poly_B, poly_C, poly_D) = (
      &self.poly_tau,
      &self.poly_Az,
      &self.poly_Bz,
      &self.poly_uCz_E,
    );
    let comb_func =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    let (eval_point_0, eval_point_2, eval_point_3) =
      SumcheckProof::<G>::compute_eval_points_cubic_with_additive_term(
        poly_A, poly_B, poly_C, poly_D, &comb_func,
      );

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &G::Scalar) {
    rayon::join(
      || {
        rayon::join(
          || self.poly_tau.bound_poly_var_top(r),
          || self.poly_Az.bound_poly_var_top(r),
        )
      },
      || {
        rayon::join(
          || self.poly_Bz.bound_poly_var_top(r),
          || self.poly_uCz_E.bound_poly_var_top(r),
        )
      },
    );
  }

  fn final_claims(&self) -> Vec<Vec<G::Scalar>> {
    vec![vec![self.poly_Az[0], self.poly_Bz[0]]]
  }
}

struct InnerSumcheckInstance<G: Group> {
  claim: G::Scalar,
  poly_L_row: MultilinearPolynomial<G::Scalar>,
  poly_L_col: MultilinearPolynomial<G::Scalar>,
  poly_val: MultilinearPolynomial<G::Scalar>,
}

impl<G: Group> SumcheckEngine<G> for InnerSumcheckInstance<G> {
  fn initial_claims(&self) -> Vec<G::Scalar> {
    vec![self.claim]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_L_row.len(), self.poly_val.len());
    assert_eq!(self.poly_L_row.len(), self.poly_L_col.len());
    self.poly_L_row.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>> {
    let (poly_A, poly_B, poly_C) = (&self.poly_L_row, &self.poly_L_col, &self.poly_val);
    let comb_func = |poly_A_comp: &G::Scalar,
                     poly_B_comp: &G::Scalar,
                     poly_C_comp: &G::Scalar|
     -> G::Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };

    let (eval_point_0, eval_point_2, eval_point_3) =
      SumcheckProof::<G>::compute_eval_points_cubic(poly_A, poly_B, poly_C, &comb_func);

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &G::Scalar) {
    rayon::join(
      || self.poly_L_row.bound_poly_var_top(r),
      || {
        rayon::join(
          || self.poly_L_col.bound_poly_var_top(r),
          || self.poly_val.bound_poly_var_top(r),
        )
      },
    );
  }

  fn final_claims(&self) -> Vec<Vec<G::Scalar>> {
    vec![vec![self.poly_L_row[0], self.poly_L_col[0]]]
  }
}

/// A type that represents the prover's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G: Group, EE: EvaluationEngineTrait<G>> {
  pk_ee: EE::ProverKey,
  S_repr: R1CSShapeSparkRepr<G>,
  S_comm: R1CSShapeSparkCommitment<G>,
  vk_digest: G::Scalar, // digest of verifier's key
}

/// A type that represents the verifier's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G: Group, EE: EvaluationEngineTrait<G>> {
  num_cons: usize,
  num_vars: usize,
  vk_ee: EE::VerifierKey,
  S_comm: R1CSShapeSparkCommitment<G>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<G::Scalar>,
}

impl<G: Group, EE: EvaluationEngineTrait<G>> SimpleDigestible for VerifierKey<G, EE> {}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<G: Group, EE: EvaluationEngineTrait<G>> {
  // commitment to oracles: the first three are for Az, Bz, Cz,
  // and the last two are for memory reads
  comm_Az: CompressedCommitment<G>,
  comm_Bz: CompressedCommitment<G>,
  comm_Cz: CompressedCommitment<G>,
  comm_L_row: CompressedCommitment<G>,
  comm_L_col: CompressedCommitment<G>,

  // commitments to aid the memory checks
  comm_t_plus_r_inv_row: CompressedCommitment<G>,
  comm_w_plus_r_inv_row: CompressedCommitment<G>,
  comm_t_plus_r_inv_col: CompressedCommitment<G>,
  comm_w_plus_r_inv_col: CompressedCommitment<G>,

  // claims about Az, Bz, and Cz polynomials
  eval_Az_at_tau: G::Scalar,
  eval_Bz_at_tau: G::Scalar,
  eval_Cz_at_tau: G::Scalar,

  // sum-check
  sc: SumcheckProof<G>,

  // claims from the end of sum-check
  eval_Az: G::Scalar,
  eval_Bz: G::Scalar,
  eval_Cz: G::Scalar,
  eval_E: G::Scalar,
  eval_L_row: G::Scalar,
  eval_L_col: G::Scalar,
  eval_val_A: G::Scalar,
  eval_val_B: G::Scalar,
  eval_val_C: G::Scalar,

  eval_W: G::Scalar,

  eval_t_plus_r_inv_row: G::Scalar,
  eval_row: G::Scalar, // address
  eval_w_plus_r_inv_row: G::Scalar,
  eval_ts_row: G::Scalar,

  eval_t_plus_r_inv_col: G::Scalar,
  eval_col: G::Scalar, // address
  eval_w_plus_r_inv_col: G::Scalar,
  eval_ts_col: G::Scalar,

  // a PCS evaluation argument
  eval_arg: EE::EvaluationArgument,
}

impl<G: Group, EE: EvaluationEngineTrait<G>> RelaxedR1CSSNARK<G, EE> {
  fn prove_helper<T1, T2, T3>(
    mem: &mut T1,
    outer: &mut T2,
    inner: &mut T3,
    transcript: &mut G::TE,
  ) -> Result<
    (
      SumcheckProof<G>,
      Vec<G::Scalar>,
      Vec<Vec<G::Scalar>>,
      Vec<Vec<G::Scalar>>,
      Vec<Vec<G::Scalar>>,
    ),
    NovaError,
  >
  where
    T1: SumcheckEngine<G>,
    T2: SumcheckEngine<G>,
    T3: SumcheckEngine<G>,
  {
    // sanity checks
    assert_eq!(mem.size(), outer.size());
    assert_eq!(mem.size(), inner.size());
    assert_eq!(mem.degree(), outer.degree());
    assert_eq!(mem.degree(), inner.degree());

    // these claims are already added to the transcript, so we do not need to add
    let claims = mem
      .initial_claims()
      .into_iter()
      .chain(outer.initial_claims())
      .chain(inner.initial_claims())
      .collect::<Vec<G::Scalar>>();

    let s = transcript.squeeze(b"r")?;
    let coeffs = powers::<G>(&s, claims.len());

    // compute the joint claim
    let claim = claims
      .iter()
      .zip(coeffs.iter())
      .map(|(c_1, c_2)| *c_1 * c_2)
      .sum();

    let mut e = claim;
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<G::Scalar>> = Vec::new();
    let num_rounds = mem.size().log_2();
    for _ in 0..num_rounds {
      let mut evals: Vec<Vec<G::Scalar>> = Vec::new();
      evals.extend(mem.evaluation_points());
      evals.extend(outer.evaluation_points());
      evals.extend(inner.evaluation_points());
      assert_eq!(evals.len(), claims.len());

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i][0] * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i][1] * coeffs[i]).sum();
      let evals_combined_3 = (0..evals.len()).map(|i| evals[i][2] * coeffs[i]).sum();

      let evals = vec![
        evals_combined_0,
        e - evals_combined_0,
        evals_combined_2,
        evals_combined_3,
      ];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      // derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);

      let _ = rayon::join(
        || mem.bound(&r_i),
        || rayon::join(|| outer.bound(&r_i), || inner.bound(&r_i)),
      );

      e = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    let mem_claims = mem.final_claims();
    let outer_claims = outer.final_claims();
    let inner_claims = inner.final_claims();

    Ok((
      SumcheckProof::new(cubic_polys),
      r,
      mem_claims,
      outer_claims,
      inner_claims,
    ))
  }
}

impl<G: Group, EE: EvaluationEngineTrait<G>> VerifierKey<G, EE> {
  fn new(
    num_cons: usize,
    num_vars: usize,
    S_comm: R1CSShapeSparkCommitment<G>,
    vk_ee: EE::VerifierKey,
  ) -> Self {
    VerifierKey {
      num_cons,
      num_vars,
      S_comm,
      vk_ee,
      digest: Default::default(),
    }
  }
}
impl<G: Group, EE: EvaluationEngineTrait<G>> DigestHelperTrait<G> for VerifierKey<G, EE> {
  /// Returns the digest of the verifier's key
  fn digest(&self) -> G::Scalar {
    self
      .digest
      .get_or_try_init(|| {
        let dc = DigestComputer::new(self);
        dc.digest()
      })
      .cloned()
      .expect("Failure to retrieve digest!")
  }
}

impl<G: Group, EE: EvaluationEngineTrait<G>> RelaxedR1CSSNARKTrait<G> for RelaxedR1CSSNARK<G, EE> {
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

    let vk = VerifierKey::new(S.num_cons, S.num_vars, S_comm.clone(), vk_ee);

    let pk = ProverKey {
      pk_ee,
      S_repr,
      S_comm,
      vk_digest: vk.digest(),
    };

    Ok((pk, vk))
  }

  /// produces a succinct proof of satisfiability of a `RelaxedR1CS` instance
  fn prove(
    ck: &CommitmentKey<G>,
    pk: &Self::ProverKey,
    S: &R1CSShape<G>,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError> {
    // pad the R1CSShape
    let S = S.pad();
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());

    let W = W.pad(&S); // pad the witness
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");

    // a list of polynomial evaluation claims that will be batched
    let mut w_u_vec = Vec::new();

    // append the verifier key (which includes commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();

    // compute Az, Bz, Cz
    let (mut Az, mut Bz, mut Cz) = S.multiply_vec(&z)?;

    // commit to Az, Bz, Cz
    let (comm_Az, (comm_Bz, comm_Cz)) = rayon::join(
      || G::CE::commit(ck, &Az),
      || rayon::join(|| G::CE::commit(ck, &Bz), || G::CE::commit(ck, &Cz)),
    );

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());

    // number of rounds of sum-check #1
    let num_rounds_sat = pk.S_repr.N.log_2();
    let tau = transcript.squeeze(b"t")?;
    let tau_coords = PowPolynomial::new(&tau, num_rounds_sat).coordinates();

    // (1) send commitments to Az, Bz, and Cz along with their evaluations at tau
    let (Az, Bz, Cz, W, E) = {
      Az.resize(pk.S_repr.N, G::Scalar::ZERO);
      Bz.resize(pk.S_repr.N, G::Scalar::ZERO);
      Cz.resize(pk.S_repr.N, G::Scalar::ZERO);
      let E = padded::<G>(&W.E, pk.S_repr.N, &G::Scalar::ZERO);
      let W = padded::<G>(&W.W, pk.S_repr.N, &G::Scalar::ZERO);

      (Az, Bz, Cz, W, E)
    };
    let (eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau) = {
      let evals_at_tau = [&Az, &Bz, &Cz]
        .into_par_iter()
        .map(|p| MultilinearPolynomial::evaluate_with(p, &tau_coords))
        .collect::<Vec<G::Scalar>>();
      (evals_at_tau[0], evals_at_tau[1], evals_at_tau[2])
    };

    // (2) send commitments to the following two oracles
    // L_row(i) = eq(tau, row(i)) for all i
    // L_col(i) = z(col(i)) for all i
    let (mem_row, mem_col, L_row, L_col) = pk.S_repr.evaluation_oracles(&S, &tau, &z);
    let (comm_L_row, comm_L_col) =
      rayon::join(|| G::CE::commit(ck, &L_row), || G::CE::commit(ck, &L_col));

    // absorb the claimed evaluations into the transcript
    transcript.absorb(
      b"e",
      &[eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau].as_slice(),
    );
    // absorb commitments to L_row and L_col in the transcript
    transcript.absorb(b"e", &vec![comm_L_row, comm_L_col].as_slice());

    // add claims about Az, Bz, and Cz to be checked later
    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau];
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    let poly_vec = vec![&Az, &Bz, &Cz];
    transcript.absorb(b"e", &eval_vec.as_slice()); // c_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let w: PolyEvalWitness<G> = PolyEvalWitness::batch(&poly_vec, &c);
    let u: PolyEvalInstance<G> = PolyEvalInstance::batch(&comm_vec, &tau_coords, &eval_vec, &c);
    w_u_vec.push((w, u));

    // we now need to prove three claims
    // (1) 0 = \sum_x poly_tau(x) * (poly_Az(x) * poly_Bz(x) - poly_uCz_E(x))
    // (2) eval_Az_at_tau + r * eval_Bz_at_tau + r^2 * eval_Cz_at_tau = \sum_y L_row(y) * (val_A(y) + r * val_B(y) + r^2 * val_C(y)) * L_col(y)
    // (3) L_row(i) = eq(tau, row(i)) and L_col(i) = z(col(i))

    // a sum-check instance to prove the first claim
    let mut outer_sc_inst = OuterSumcheckInstance {
      poly_tau: MultilinearPolynomial::new(PowPolynomial::new(&tau, num_rounds_sat).evals()),
      poly_Az: MultilinearPolynomial::new(Az.clone()),
      poly_Bz: MultilinearPolynomial::new(Bz.clone()),
      poly_uCz_E: {
        let uCz_E = (0..Cz.len())
          .map(|i| U.u * Cz[i] + E[i])
          .collect::<Vec<G::Scalar>>();
        MultilinearPolynomial::new(uCz_E)
      },
    };

    // a sum-check instance to prove the second claim
    let val = pk
      .S_repr
      .val_A
      .iter()
      .zip(pk.S_repr.val_B.iter())
      .zip(pk.S_repr.val_C.iter())
      .map(|((v_a, v_b), v_c)| *v_a + c * *v_b + c * c * *v_c)
      .collect::<Vec<G::Scalar>>();
    let mut inner_sc_inst = InnerSumcheckInstance {
      claim: eval_Az_at_tau + c * eval_Bz_at_tau + c * c * eval_Cz_at_tau,
      poly_L_row: MultilinearPolynomial::new(L_row.clone()),
      poly_L_col: MultilinearPolynomial::new(L_col.clone()),
      poly_val: MultilinearPolynomial::new(val),
    };

    // a third sum-check instance to prove the read-only memory claim
    // we now need to prove that L_row and L_col are well-formed
    let gamma = transcript.squeeze(b"g")?;

    // hash the tuples of (addr,val) memory contents and read responses into a single field element using `hash_func`
    let hash_func_vec = |mem: &[G::Scalar],
                         addr: &[G::Scalar],
                         lookups: &[G::Scalar]|
     -> (Vec<G::Scalar>, Vec<G::Scalar>) {
      let hash_func = |addr: &G::Scalar, val: &G::Scalar| -> G::Scalar { *val * gamma + *addr };
      assert_eq!(addr.len(), lookups.len());
      rayon::join(
        || {
          (0..mem.len())
            .map(|i| hash_func(&G::Scalar::from(i as u64), &mem[i]))
            .collect::<Vec<G::Scalar>>()
        },
        || {
          (0..addr.len())
            .map(|i| hash_func(&addr[i], &lookups[i]))
            .collect::<Vec<G::Scalar>>()
        },
      )
    };

    let ((T_row, W_row), (T_col, W_col)) = rayon::join(
      || hash_func_vec(&mem_row, &pk.S_repr.row, &L_row),
      || hash_func_vec(&mem_col, &pk.S_repr.col, &L_col),
    );

    let r = transcript.squeeze(b"r")?;
    let (mut mem_sc_inst, comm_lookup, polys_lookup) = LookupSumcheckInstance::new(
      ck,
      &r,
      T_row,
      W_row,
      pk.S_repr.ts_row.clone(),
      T_col,
      W_col,
      pk.S_repr.ts_col.clone(),
      &mut transcript,
    )?;

    let (sc, rand_sc, claims_mem, claims_outer, claims_inner) = Self::prove_helper(
      &mut mem_sc_inst,
      &mut outer_sc_inst,
      &mut inner_sc_inst,
      &mut transcript,
    )?;

    // claims from the end of the sum-check
    let eval_Az = claims_outer[0][0];
    let eval_Bz = claims_outer[0][1];

    let eval_L_row = claims_inner[0][0];
    let eval_L_col = claims_inner[0][1];

    let eval_t_plus_r_inv_row = claims_mem[0][0];
    let eval_w_plus_r_inv_row = claims_mem[0][1];
    let eval_ts_row = claims_mem[0][2];

    let eval_t_plus_r_inv_col = claims_mem[1][0];
    let eval_w_plus_r_inv_col = claims_mem[1][1];
    let eval_ts_col = claims_mem[1][2];

    // compute the remaining claims that did not come for free from the sum-check prover
    let (eval_W, eval_Cz, eval_E, eval_val_A, eval_val_B, eval_val_C, eval_row, eval_col) = {
      let e = [
        &W,
        &Cz,
        &E,
        &pk.S_repr.val_A,
        &pk.S_repr.val_B,
        &pk.S_repr.val_C,
        &pk.S_repr.row,
        &pk.S_repr.col,
      ]
      .into_par_iter()
      .map(|p| MultilinearPolynomial::evaluate_with(p, &rand_sc))
      .collect::<Vec<G::Scalar>>();
      (e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7])
    };

    // all the evaluations are at rand_sc, we can fold them into one claim
    let eval_vec = vec![
      eval_W,
      eval_Az,
      eval_Bz,
      eval_Cz,
      eval_E,
      eval_L_row,
      eval_L_col,
      eval_val_A,
      eval_val_B,
      eval_val_C,
      eval_t_plus_r_inv_row,
      eval_row,
      eval_w_plus_r_inv_row,
      eval_ts_row,
      eval_t_plus_r_inv_col,
      eval_col,
      eval_w_plus_r_inv_col,
      eval_ts_col,
    ]
    .into_iter()
    .collect::<Vec<G::Scalar>>();

    let comm_vec = [
      U.comm_W,
      comm_Az,
      comm_Bz,
      comm_Cz,
      U.comm_E,
      comm_L_row,
      comm_L_col,
      pk.S_comm.comm_val_A,
      pk.S_comm.comm_val_B,
      pk.S_comm.comm_val_C,
      comm_lookup[0],
      pk.S_comm.comm_row,
      comm_lookup[1],
      pk.S_comm.comm_ts_row,
      comm_lookup[2],
      pk.S_comm.comm_col,
      comm_lookup[3],
      pk.S_comm.comm_ts_col,
    ];
    let poly_vec = [
      &W,
      &Az,
      &Bz,
      &Cz,
      &E,
      &L_row,
      &L_col,
      &pk.S_repr.val_A,
      &pk.S_repr.val_B,
      &pk.S_repr.val_C,
      polys_lookup[0].as_ref(),
      &pk.S_repr.row,
      polys_lookup[1].as_ref(),
      &pk.S_repr.ts_row,
      polys_lookup[2].as_ref(),
      &pk.S_repr.col,
      polys_lookup[3].as_ref(),
      &pk.S_repr.ts_col,
    ];
    transcript.absorb(b"e", &eval_vec.as_slice()); // comm_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let w: PolyEvalWitness<G> = PolyEvalWitness::batch(&poly_vec, &c);
    let u: PolyEvalInstance<G> = PolyEvalInstance::batch(&comm_vec, &rand_sc, &eval_vec, &c);

    let eval_arg = EE::prove(ck, &pk.pk_ee, &mut transcript, &u.c, &w.p, &rand_sc, &u.e)?;

    Ok(RelaxedR1CSSNARK {
      comm_Az: comm_Az.compress(),
      comm_Bz: comm_Bz.compress(),
      comm_Cz: comm_Cz.compress(),
      comm_L_row: comm_L_row.compress(),
      comm_L_col: comm_L_col.compress(),

      comm_t_plus_r_inv_row: comm_lookup[0].compress(),
      comm_w_plus_r_inv_row: comm_lookup[1].compress(),
      comm_t_plus_r_inv_col: comm_lookup[2].compress(),
      comm_w_plus_r_inv_col: comm_lookup[3].compress(),

      eval_Az_at_tau,
      eval_Bz_at_tau,
      eval_Cz_at_tau,

      sc,

      eval_Az,
      eval_Bz,
      eval_Cz,
      eval_E,
      eval_L_row,
      eval_L_col,
      eval_val_A,
      eval_val_B,
      eval_val_C,

      eval_W,

      eval_t_plus_r_inv_row,
      eval_row,
      eval_w_plus_r_inv_row,
      eval_ts_row,

      eval_col,
      eval_t_plus_r_inv_col,
      eval_w_plus_r_inv_col,
      eval_ts_col,

      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a `RelaxedR1CS` instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError> {
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");
    let mut u_vec: Vec<PolyEvalInstance<G>> = Vec::new();

    // append the verifier key (including commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    let comm_Az = Commitment::<G>::decompress(&self.comm_Az)?;
    let comm_Bz = Commitment::<G>::decompress(&self.comm_Bz)?;
    let comm_Cz = Commitment::<G>::decompress(&self.comm_Cz)?;
    let comm_L_row = Commitment::<G>::decompress(&self.comm_L_row)?;
    let comm_L_col = Commitment::<G>::decompress(&self.comm_L_col)?;
    let comm_t_plus_r_inv_row = Commitment::<G>::decompress(&self.comm_t_plus_r_inv_row)?;
    let comm_w_plus_r_inv_row = Commitment::<G>::decompress(&self.comm_w_plus_r_inv_row)?;
    let comm_t_plus_r_inv_col = Commitment::<G>::decompress(&self.comm_t_plus_r_inv_col)?;
    let comm_w_plus_r_inv_col = Commitment::<G>::decompress(&self.comm_w_plus_r_inv_col)?;

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());

    let num_rounds_sat = vk.S_comm.N.log_2();
    let tau = transcript.squeeze(b"t")?;
    let tau_coords = PowPolynomial::new(&tau, num_rounds_sat).coordinates();

    transcript.absorb(
      b"e",
      &[
        self.eval_Az_at_tau,
        self.eval_Bz_at_tau,
        self.eval_Cz_at_tau,
      ]
      .as_slice(),
    );

    transcript.absorb(b"e", &vec![comm_L_row, comm_L_col].as_slice());

    // add claims about Az, Bz, and Cz to be checked later
    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![
      self.eval_Az_at_tau,
      self.eval_Bz_at_tau,
      self.eval_Cz_at_tau,
    ];
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    transcript.absorb(b"e", &eval_vec.as_slice()); // c_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let u = PolyEvalInstance::batch(&comm_vec, &tau_coords, &eval_vec, &c);
    let claim_inner = u.e;
    u_vec.push(u);

    let gamma = transcript.squeeze(b"g")?;

    let r = transcript.squeeze(b"r")?;

    transcript.absorb(
      b"l",
      &vec![
        comm_t_plus_r_inv_row,
        comm_w_plus_r_inv_row,
        comm_t_plus_r_inv_col,
        comm_w_plus_r_inv_col,
      ]
      .as_slice(),
    );

    let rho = transcript.squeeze(b"r")?;

    let num_claims = 8;
    let s = transcript.squeeze(b"r")?;
    let coeffs = powers::<G>(&s, num_claims);
    let claim = coeffs[7] * claim_inner; // rest are zeros

    // verify sc
    let (claim_sat_final, rand_sc) = self.sc.verify(claim, num_rounds_sat, 3, &mut transcript)?;

    // verify claim_sat_final
    let claim_sat_final_expected = {
      let rand_eq_bound_rand_sc = {
        let poly_eq_coords = PowPolynomial::new(&rho, num_rounds_sat).coordinates();
        EqPolynomial::new(poly_eq_coords).evaluate(&rand_sc)
      };
      let taus_bound_rand_sc = PowPolynomial::new(&tau, num_rounds_sat).evaluate(&rand_sc);

      let eval_t_plus_r_row = {
        let eval_addr_row = IdentityPolynomial::new(num_rounds_sat).evaluate(&rand_sc);
        let eval_val_row = taus_bound_rand_sc;
        let eval_t = eval_addr_row + gamma * eval_val_row;
        eval_t + r
      };

      let eval_w_plus_r_row = {
        let eval_addr_row = self.eval_row;
        let eval_val_row = self.eval_L_row;
        let eval_w = eval_addr_row + gamma * eval_val_row;
        eval_w + r
      };

      let eval_t_plus_r_col = {
        let eval_addr_col = IdentityPolynomial::new(num_rounds_sat).evaluate(&rand_sc);

        // memory contents is z, so we compute eval_Z from eval_W and eval_X
        let eval_val_col = {
          // rand_sc was padded, so we now remove the padding
          let (factor, rand_sc_unpad) = {
            let l = vk.S_comm.N.log_2() - (2 * vk.num_vars).log_2();

            let mut factor = G::Scalar::ONE;
            for r_p in rand_sc.iter().take(l) {
              factor *= G::Scalar::ONE - r_p
            }

            let rand_sc_unpad = {
              let l = vk.S_comm.N.log_2() - (2 * vk.num_vars).log_2();
              rand_sc[l..].to_vec()
            };

            (factor, rand_sc_unpad)
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
            SparsePolynomial::new(vk.num_vars.log_2(), poly_X).evaluate(&rand_sc_unpad[1..])
          };

          self.eval_W + factor * rand_sc_unpad[0] * eval_X
        };
        let eval_t = eval_addr_col + gamma * eval_val_col;
        eval_t + r
      };

      let eval_w_plus_r_col = {
        let eval_addr_col = self.eval_col;
        let eval_val_col = self.eval_L_col;
        let eval_w = eval_addr_col + gamma * eval_val_col;
        eval_w + r
      };

      let claim_mem_final_expected: G::Scalar = coeffs[0]
        * (self.eval_t_plus_r_inv_row - self.eval_w_plus_r_inv_row)
        + coeffs[1] * (self.eval_t_plus_r_inv_col - self.eval_w_plus_r_inv_col)
        + coeffs[2]
          * (rand_eq_bound_rand_sc
            * (self.eval_t_plus_r_inv_row * eval_t_plus_r_row - self.eval_ts_row))
        + coeffs[3]
          * (rand_eq_bound_rand_sc
            * (self.eval_w_plus_r_inv_row * eval_w_plus_r_row - G::Scalar::ONE))
        + coeffs[4]
          * (rand_eq_bound_rand_sc
            * (self.eval_t_plus_r_inv_col * eval_t_plus_r_col - self.eval_ts_col))
        + coeffs[5]
          * (rand_eq_bound_rand_sc
            * (self.eval_w_plus_r_inv_col * eval_w_plus_r_col - G::Scalar::ONE));

      let claim_outer_final_expected = coeffs[6]
        * taus_bound_rand_sc
        * (self.eval_Az * self.eval_Bz - U.u * self.eval_Cz - self.eval_E);
      let claim_inner_final_expected = coeffs[7]
        * self.eval_L_row
        * self.eval_L_col
        * (self.eval_val_A + c * self.eval_val_B + c * c * self.eval_val_C);

      claim_mem_final_expected + claim_outer_final_expected + claim_inner_final_expected
    };

    if claim_sat_final_expected != claim_sat_final {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let eval_vec = vec![
      self.eval_W,
      self.eval_Az,
      self.eval_Bz,
      self.eval_Cz,
      self.eval_E,
      self.eval_L_row,
      self.eval_L_col,
      self.eval_val_A,
      self.eval_val_B,
      self.eval_val_C,
      self.eval_t_plus_r_inv_row,
      self.eval_row,
      self.eval_w_plus_r_inv_row,
      self.eval_ts_row,
      self.eval_t_plus_r_inv_col,
      self.eval_col,
      self.eval_w_plus_r_inv_col,
      self.eval_ts_col,
    ]
    .into_iter()
    .collect::<Vec<G::Scalar>>();
    let comm_vec = [
      U.comm_W,
      comm_Az,
      comm_Bz,
      comm_Cz,
      U.comm_E,
      comm_L_row,
      comm_L_col,
      vk.S_comm.comm_val_A,
      vk.S_comm.comm_val_B,
      vk.S_comm.comm_val_C,
      comm_t_plus_r_inv_row,
      vk.S_comm.comm_row,
      comm_w_plus_r_inv_row,
      vk.S_comm.comm_ts_row,
      comm_t_plus_r_inv_col,
      vk.S_comm.comm_col,
      comm_w_plus_r_inv_col,
      vk.S_comm.comm_ts_col,
    ];
    transcript.absorb(b"e", &eval_vec.as_slice()); // comm_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let u: PolyEvalInstance<G> = PolyEvalInstance::batch(&comm_vec, &rand_sc, &eval_vec, &c);

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &u.c,
      &rand_sc,
      &u.e,
      &self.eval_arg,
    )?;

    Ok(())
  }
}
