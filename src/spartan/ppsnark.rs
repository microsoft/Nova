//! This module implements `RelaxedR1CSSNARK` traits using a spark-based approach to prove evaluations of
//! sparse multilinear polynomials involved in Spartan's sum-check protocol, thereby providing a preprocessing SNARK
//! The verifier in this preprocessing SNARK maintains a commitment to R1CS matrices. This is beneficial when using a
//! polynomial commitment scheme in which the verifier's costs is succinct.
//! This code includes experimental optimizations to reduce runtimes and proof sizes.
//! We have not yet proven the security of these optimizations, so this code is subject to significant changes in the future.
use crate::{
  digest::{DigestComputer, SimpleDigestible},
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    math::Math,
    polys::{
      eq::EqPolynomial,
      identity::IdentityPolynomial,
      masked_eq::MaskedEqPolynomial,
      multilinear::{MultilinearPolynomial, SparsePolynomial},
      power::PowPolynomial,
      univariate::{CompressedUniPoly, UniPoly},
    },
    powers,
    sumcheck::{SumcheckEngine, SumcheckProof},
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait, Len},
    evaluation::EvaluationEngineTrait,
    snark::{DigestHelperTrait, RelaxedR1CSSNARKTrait},
    Engine, TranscriptEngineTrait, TranscriptReprTrait,
  },
  zip_with, Commitment, CommitmentKey, CompressedCommitment,
};
use core::cmp::max;
use ff::Field;
use itertools::Itertools as _;
use once_cell::sync::OnceCell;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

fn padded<E: Engine>(v: &[E::Scalar], n: usize, e: &E::Scalar) -> Vec<E::Scalar> {
  let mut v_padded = vec![*e; n];
  for (i, v_i) in v.iter().enumerate() {
    v_padded[i] = *v_i;
  }
  v_padded
}

/// A type that holds `R1CSShape` in a form amenable to memory checking
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkRepr<E: Engine> {
  N: usize, // size of the vectors

  // dense representation
  row: Vec<E::Scalar>,
  col: Vec<E::Scalar>,
  val_A: Vec<E::Scalar>,
  val_B: Vec<E::Scalar>,
  val_C: Vec<E::Scalar>,

  // timestamp polynomials
  ts_row: Vec<E::Scalar>,
  ts_col: Vec<E::Scalar>,
}

/// A type that holds a commitment to a sparse polynomial
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkCommitment<E: Engine> {
  N: usize, // size of each vector

  // commitments to the dense representation
  comm_row: Commitment<E>,
  comm_col: Commitment<E>,
  comm_val_A: Commitment<E>,
  comm_val_B: Commitment<E>,
  comm_val_C: Commitment<E>,

  // commitments to the timestamp polynomials
  comm_ts_row: Commitment<E>,
  comm_ts_col: Commitment<E>,
}

impl<E: Engine> TranscriptReprTrait<E::GE> for R1CSShapeSparkCommitment<E> {
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

impl<E: Engine> R1CSShapeSparkRepr<E> {
  /// represents `R1CSShape` in a Spark-friendly format amenable to memory checking
  pub fn new(S: &R1CSShape<E>) -> R1CSShapeSparkRepr<E> {
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
      let mut val = vec![E::Scalar::ZERO; N];
      for (i, (_, _, v)) in S.A.iter().enumerate() {
        val[i] = v;
      }
      val
    };

    let val_B = {
      let mut val = vec![E::Scalar::ZERO; N];
      for (i, (_, _, v)) in S.B.iter().enumerate() {
        val[S.A.len() + i] = v;
      }
      val
    };

    let val_C = {
      let mut val = vec![E::Scalar::ZERO; N];
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
    let to_vec_scalar = |v: &[usize]| -> Vec<E::Scalar> {
      (0..v.len())
        .map(|i| E::Scalar::from(v[i] as u64))
        .collect::<Vec<E::Scalar>>()
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

  fn commit(&self, ck: &CommitmentKey<E>) -> R1CSShapeSparkCommitment<E> {
    let comm_vec: Vec<Commitment<E>> = [
      &self.row,
      &self.col,
      &self.val_A,
      &self.val_B,
      &self.val_C,
      &self.ts_row,
      &self.ts_col,
    ]
    .par_iter()
    .map(|v| E::CE::commit(ck, v))
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
    S: &R1CSShape<E>,
    r_x: &E::Scalar,
    z: &[E::Scalar],
  ) -> (
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
    Vec<E::Scalar>,
  ) {
    let mem_row = PowPolynomial::new(r_x, self.N.log_2()).evals();
    let mem_col = padded::<E>(z, self.N, &E::Scalar::ZERO);

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

/// The [WitnessBoundSumcheck] ensures that the witness polynomial W defined over n = log(N) variables,
/// is zero outside of the first `num_vars = 2^m` entries.
///
/// # Details
///
/// The `W` polynomial is padded with zeros to size N = 2^n.
/// The `masked_eq` polynomials is defined as with regards to a random challenge `tau` as
/// the eq(tau) polynomial, where the first 2^m evaluations to 0.
///
/// The instance is given by
///  `0 = ∑_{0≤i<2^n} masked_eq[i] * W[i]`.
/// It is equivalent to the expression
///  `0 = ∑_{2^m≤i<2^n} eq[i] * W[i]`
/// Since `eq` is random, the instance is only satisfied if `W[2^{m}..] = 0`.
pub struct WitnessBoundSumcheck<E: Engine> {
  poly_W: MultilinearPolynomial<E::Scalar>,
  poly_masked_eq: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> WitnessBoundSumcheck<E> {
  fn new(tau: E::Scalar, poly_W_padded: Vec<E::Scalar>, num_vars: usize) -> Self {
    let num_vars_log = num_vars.log_2();
    // When num_vars = num_rounds, we shouldn't have to prove anything
    // but we still want this instance to compute the evaluation of W
    let num_rounds = poly_W_padded.len().log_2();
    assert!(num_vars_log < num_rounds);

    let tau_coords = PowPolynomial::new(&tau, num_rounds).coordinates();
    let poly_masked_eq_evals =
      MaskedEqPolynomial::new(&EqPolynomial::new(tau_coords), num_vars_log).evals();

    Self {
      poly_W: MultilinearPolynomial::new(poly_W_padded),
      poly_masked_eq: MultilinearPolynomial::new(poly_masked_eq_evals),
    }
  }
}
impl<E: Engine> SumcheckEngine<E> for WitnessBoundSumcheck<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![E::Scalar::ZERO]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_W.len(), self.poly_masked_eq.len());
    self.poly_W.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let comb_func = |poly_A_comp: &E::Scalar,
                     poly_B_comp: &E::Scalar,
                     _: &E::Scalar|
     -> E::Scalar { *poly_A_comp * *poly_B_comp };

    let (eval_point_0, eval_point_2, eval_point_3) = SumcheckProof::<E>::compute_eval_points_cubic(
      &self.poly_masked_eq,
      &self.poly_W,
      &self.poly_W, // unused
      &comb_func,
    );

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [&mut self.poly_W, &mut self.poly_masked_eq]
      .par_iter_mut()
      .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    vec![vec![self.poly_W[0], self.poly_masked_eq[0]]]
  }
}

struct MemorySumcheckInstance<E: Engine> {
  // row
  w_plus_r_row: MultilinearPolynomial<E::Scalar>,
  t_plus_r_row: MultilinearPolynomial<E::Scalar>,
  t_plus_r_inv_row: MultilinearPolynomial<E::Scalar>,
  w_plus_r_inv_row: MultilinearPolynomial<E::Scalar>,
  ts_row: MultilinearPolynomial<E::Scalar>,

  // col
  w_plus_r_col: MultilinearPolynomial<E::Scalar>,
  t_plus_r_col: MultilinearPolynomial<E::Scalar>,
  t_plus_r_inv_col: MultilinearPolynomial<E::Scalar>,
  w_plus_r_inv_col: MultilinearPolynomial<E::Scalar>,
  ts_col: MultilinearPolynomial<E::Scalar>,

  // eq
  poly_eq: MultilinearPolynomial<E::Scalar>,

  // zero polynomial
  poly_zero: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> MemorySumcheckInstance<E> {
  /// Computes witnesses for MemoryInstanceSumcheck
  ///
  /// # Description
  /// We use the logUp protocol to prove that
  /// ∑ TS[i]/(T[i] + r) - 1/(W[i] + r) = 0
  /// where
  ///   T_row[i] = mem_row[i]      * gamma + i
  ///            = eq(tau)[i]      * gamma + i
  ///   W_row[i] = L_row[i]        * gamma + addr_row[i]
  ///            = eq(tau)[row[i]] * gamma + addr_row[i]
  ///   T_col[i] = mem_col[i]      * gamma + i
  ///            = z[i]            * gamma + i
  ///   W_col[i] = addr_col[i]     * gamma + addr_col[i]
  ///            = z[col[i]]       * gamma + addr_col[i]
  /// and
  ///   TS_row, TS_col are integer-valued vectors representing the number of reads
  ///   to each memory cell of L_row, L_col
  ///
  /// The function returns oracles for the polynomials TS[i]/(T[i] + r), 1/(W[i] + r),
  /// as well as auxiliary polynomials T[i] + r, W[i] + r
  pub fn compute_oracles(
    ck: &CommitmentKey<E>,
    r: &E::Scalar,
    gamma: &E::Scalar,
    mem_row: &[E::Scalar],
    addr_row: &[E::Scalar],
    L_row: &[E::Scalar],
    ts_row: &[E::Scalar],
    mem_col: &[E::Scalar],
    addr_col: &[E::Scalar],
    L_col: &[E::Scalar],
    ts_col: &[E::Scalar],
  ) -> Result<([Commitment<E>; 4], [Vec<E::Scalar>; 4], [Vec<E::Scalar>; 4]), NovaError> {
    // hash the tuples of (addr,val) memory contents and read responses into a single field element using `hash_func`
    let hash_func_vec = |mem: &[E::Scalar],
                         addr: &[E::Scalar],
                         lookups: &[E::Scalar]|
     -> (Vec<E::Scalar>, Vec<E::Scalar>) {
      let hash_func = |addr: &E::Scalar, val: &E::Scalar| -> E::Scalar { *val * gamma + *addr };
      assert_eq!(addr.len(), lookups.len());
      rayon::join(
        || {
          (0..mem.len())
            .map(|i| hash_func(&E::Scalar::from(i as u64), &mem[i]))
            .collect::<Vec<E::Scalar>>()
        },
        || {
          (0..addr.len())
            .map(|i| hash_func(&addr[i], &lookups[i]))
            .collect::<Vec<E::Scalar>>()
        },
      )
    };

    let ((T_row, W_row), (T_col, W_col)) = rayon::join(
      || hash_func_vec(mem_row, addr_row, L_row),
      || hash_func_vec(mem_col, addr_col, L_col),
    );

    let batch_invert = |v: &[E::Scalar]| -> Result<Vec<E::Scalar>, NovaError> {
      let mut products = vec![E::Scalar::ZERO; v.len()];
      let mut acc = E::Scalar::ONE;

      for i in 0..v.len() {
        products[i] = acc;
        acc *= v[i];
      }

      // we can compute an inversion only if acc is non-zero
      if acc == E::Scalar::ZERO {
        return Err(NovaError::InternalError);
      }

      // compute the inverse once for all entries
      acc = acc.invert().unwrap();

      let mut inv = vec![E::Scalar::ZERO; v.len()];
      for i in 0..v.len() {
        let tmp = acc * v[v.len() - 1 - i];
        inv[v.len() - 1 - i] = products[v.len() - 1 - i] * acc;
        acc = tmp;
      }

      Ok(inv)
    };

    // compute vectors TS[i]/(T[i] + r) and 1/(W[i] + r)
    let helper = |T: &[E::Scalar],
                  W: &[E::Scalar],
                  TS: &[E::Scalar],
                  r: &E::Scalar|
     -> (
      (
        Result<Vec<E::Scalar>, NovaError>,
        Result<Vec<E::Scalar>, NovaError>,
      ),
      (
        Result<Vec<E::Scalar>, NovaError>,
        Result<Vec<E::Scalar>, NovaError>,
      ),
    ) {
      rayon::join(
        || {
          rayon::join(
            || {
              let inv = batch_invert(&T.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>())?;

              // compute inv[i] * TS[i] in parallel
              Ok(
                zip_with!((inv.into_par_iter(), TS.par_iter()), |e1, e2| e1 * *e2)
                  .collect::<Vec<_>>(),
              )
            },
            || batch_invert(&W.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>()),
          )
        },
        || {
          rayon::join(
            || Ok(T.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>()),
            || Ok(W.par_iter().map(|e| *e + *r).collect::<Vec<E::Scalar>>()),
          )
        },
      )
    };

    let (
      ((t_plus_r_inv_row, w_plus_r_inv_row), (t_plus_r_row, w_plus_r_row)),
      ((t_plus_r_inv_col, w_plus_r_inv_col), (t_plus_r_col, w_plus_r_col)),
    ) = rayon::join(
      || helper(&T_row, &W_row, ts_row, r),
      || helper(&T_col, &W_col, ts_col, r),
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
          || E::CE::commit(ck, &t_plus_r_inv_row),
          || E::CE::commit(ck, &w_plus_r_inv_row),
        )
      },
      || {
        rayon::join(
          || E::CE::commit(ck, &t_plus_r_inv_col),
          || E::CE::commit(ck, &w_plus_r_inv_col),
        )
      },
    );

    let comm_vec = [
      comm_t_plus_r_inv_row,
      comm_w_plus_r_inv_row,
      comm_t_plus_r_inv_col,
      comm_w_plus_r_inv_col,
    ];

    let poly_vec = [
      t_plus_r_inv_row,
      w_plus_r_inv_row,
      t_plus_r_inv_col,
      w_plus_r_inv_col,
    ];

    let aux_poly_vec = [t_plus_r_row?, w_plus_r_row?, t_plus_r_col?, w_plus_r_col?];

    Ok((comm_vec, poly_vec, aux_poly_vec))
  }

  pub fn new(
    polys_oracle: [Vec<E::Scalar>; 4],
    polys_aux: [Vec<E::Scalar>; 4],
    poly_eq: Vec<E::Scalar>,
    ts_row: Vec<E::Scalar>,
    ts_col: Vec<E::Scalar>,
  ) -> Self {
    let [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col] = polys_oracle;
    let [t_plus_r_row, w_plus_r_row, t_plus_r_col, w_plus_r_col] = polys_aux;

    let zero = vec![E::Scalar::ZERO; poly_eq.len()];

    Self {
      w_plus_r_row: MultilinearPolynomial::new(w_plus_r_row),
      t_plus_r_row: MultilinearPolynomial::new(t_plus_r_row),
      t_plus_r_inv_row: MultilinearPolynomial::new(t_plus_r_inv_row),
      w_plus_r_inv_row: MultilinearPolynomial::new(w_plus_r_inv_row),
      ts_row: MultilinearPolynomial::new(ts_row),
      w_plus_r_col: MultilinearPolynomial::new(w_plus_r_col),
      t_plus_r_col: MultilinearPolynomial::new(t_plus_r_col),
      t_plus_r_inv_col: MultilinearPolynomial::new(t_plus_r_inv_col),
      w_plus_r_inv_col: MultilinearPolynomial::new(w_plus_r_inv_col),
      ts_col: MultilinearPolynomial::new(ts_col),
      poly_eq: MultilinearPolynomial::new(poly_eq),
      poly_zero: MultilinearPolynomial::new(zero),
    }
  }
}

impl<E: Engine> SumcheckEngine<E> for MemorySumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![E::Scalar::ZERO; 6]
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

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let comb_func = |poly_A_comp: &E::Scalar,
                     poly_B_comp: &E::Scalar,
                     _poly_C_comp: &E::Scalar|
     -> E::Scalar { *poly_A_comp - *poly_B_comp };

    let comb_func2 =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       _poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - E::Scalar::ONE) };

    let comb_func3 =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    // inv related evaluation points
    // 0 = ∑ TS[i]/(T[i] + r) - 1/(W[i] + r)
    let (eval_inv_0_row, eval_inv_2_row, eval_inv_3_row) =
      SumcheckProof::<E>::compute_eval_points_cubic(
        &self.t_plus_r_inv_row,
        &self.w_plus_r_inv_row,
        &self.poly_zero,
        &comb_func,
      );

    let (eval_inv_0_col, eval_inv_2_col, eval_inv_3_col) =
      SumcheckProof::<E>::compute_eval_points_cubic(
        &self.t_plus_r_inv_col,
        &self.w_plus_r_inv_col,
        &self.poly_zero,
        &comb_func,
      );

    // row related evaluation points
    // 0 = ∑ eq[i] * (inv_T[i] * (T[i] + r) - TS[i]))
    let (eval_T_0_row, eval_T_2_row, eval_T_3_row) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.t_plus_r_inv_row,
        &self.t_plus_r_row,
        &self.ts_row,
        &comb_func3,
      );
    // 0 = ∑ eq[i] * (inv_W[i] * (T[i] + r) - 1))
    let (eval_W_0_row, eval_W_2_row, eval_W_3_row) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.w_plus_r_inv_row,
        &self.w_plus_r_row,
        &self.poly_zero,
        &comb_func2,
      );

    // column related evaluation points
    let (eval_T_0_col, eval_T_2_col, eval_T_3_col) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.t_plus_r_inv_col,
        &self.t_plus_r_col,
        &self.ts_col,
        &comb_func3,
      );
    let (eval_W_0_col, eval_W_2_col, eval_W_3_col) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_eq,
        &self.w_plus_r_inv_col,
        &self.w_plus_r_col,
        &self.poly_zero,
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

  fn bound(&mut self, r: &E::Scalar) {
    [
      &mut self.t_plus_r_row,
      &mut self.t_plus_r_inv_row,
      &mut self.w_plus_r_row,
      &mut self.w_plus_r_inv_row,
      &mut self.ts_row,
      &mut self.t_plus_r_col,
      &mut self.t_plus_r_inv_col,
      &mut self.w_plus_r_col,
      &mut self.w_plus_r_inv_col,
      &mut self.ts_col,
      &mut self.poly_eq,
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
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

struct OuterSumcheckInstance<E: Engine> {
  poly_tau: MultilinearPolynomial<E::Scalar>,
  poly_Az: MultilinearPolynomial<E::Scalar>,
  poly_Bz: MultilinearPolynomial<E::Scalar>,
  poly_uCz_E: MultilinearPolynomial<E::Scalar>,

  poly_Mz: MultilinearPolynomial<E::Scalar>,
  eval_Mz_at_tau: E::Scalar,

  poly_zero: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> OuterSumcheckInstance<E> {
  pub fn new(
    tau: Vec<E::Scalar>,
    Az: Vec<E::Scalar>,
    Bz: Vec<E::Scalar>,
    uCz_E: Vec<E::Scalar>,
    Mz: Vec<E::Scalar>,
    eval_Mz_at_tau: &E::Scalar,
  ) -> Self {
    let zero = vec![E::Scalar::ZERO; tau.len()];
    Self {
      poly_tau: MultilinearPolynomial::new(tau),
      poly_Az: MultilinearPolynomial::new(Az),
      poly_Bz: MultilinearPolynomial::new(Bz),
      poly_uCz_E: MultilinearPolynomial::new(uCz_E),
      poly_Mz: MultilinearPolynomial::new(Mz),
      eval_Mz_at_tau: *eval_Mz_at_tau,
      poly_zero: MultilinearPolynomial::new(zero),
    }
  }
}

impl<E: Engine> SumcheckEngine<E> for OuterSumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    vec![E::Scalar::ZERO, self.eval_Mz_at_tau]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_tau.len(), self.poly_Az.len());
    assert_eq!(self.poly_tau.len(), self.poly_Bz.len());
    assert_eq!(self.poly_tau.len(), self.poly_uCz_E.len());
    assert_eq!(self.poly_tau.len(), self.poly_Mz.len());
    self.poly_tau.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let comb_func =
      |poly_A_comp: &E::Scalar,
       poly_B_comp: &E::Scalar,
       poly_C_comp: &E::Scalar,
       poly_D_comp: &E::Scalar|
       -> E::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    let (eval_point_h_0, eval_point_h_2, eval_point_h_3) =
      SumcheckProof::<E>::compute_eval_points_cubic_with_additive_term(
        &self.poly_tau,
        &self.poly_Az,
        &self.poly_Bz,
        &self.poly_uCz_E,
        &comb_func,
      );

    let comb_func2 = |poly_A_comp: &E::Scalar,
                      poly_B_comp: &E::Scalar,
                      _poly_C_comp: &E::Scalar|
     -> E::Scalar { *poly_A_comp * *poly_B_comp };

    let (eval_point_e_0, eval_point_e_2, eval_point_e_3) =
      SumcheckProof::<E>::compute_eval_points_cubic(
        &self.poly_tau,
        &self.poly_Mz,
        &self.poly_zero,
        &comb_func2,
      );

    vec![
      vec![eval_point_h_0, eval_point_h_2, eval_point_h_3],
      vec![eval_point_e_0, eval_point_e_2, eval_point_e_3],
    ]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [
      &mut self.poly_tau,
      &mut self.poly_Az,
      &mut self.poly_Bz,
      &mut self.poly_uCz_E,
      &mut self.poly_Mz,
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    vec![vec![self.poly_Az[0], self.poly_Bz[0]]]
  }
}

struct InnerSumcheckInstance<E: Engine> {
  claim: E::Scalar,
  poly_L_row: MultilinearPolynomial<E::Scalar>,
  poly_L_col: MultilinearPolynomial<E::Scalar>,
  poly_val: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> SumcheckEngine<E> for InnerSumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
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

  fn evaluation_points(&self) -> Vec<Vec<E::Scalar>> {
    let (poly_A, poly_B, poly_C) = (&self.poly_L_row, &self.poly_L_col, &self.poly_val);
    let comb_func = |poly_A_comp: &E::Scalar,
                     poly_B_comp: &E::Scalar,
                     poly_C_comp: &E::Scalar|
     -> E::Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };

    let (eval_point_0, eval_point_2, eval_point_3) =
      SumcheckProof::<E>::compute_eval_points_cubic(poly_A, poly_B, poly_C, &comb_func);

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &E::Scalar) {
    [
      &mut self.poly_L_row,
      &mut self.poly_L_col,
      &mut self.poly_val,
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    vec![vec![self.poly_L_row[0], self.poly_L_col[0]]]
  }
}

/// A type that represents the prover's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  pk_ee: EE::ProverKey,
  S_repr: R1CSShapeSparkRepr<E>,
  S_comm: R1CSShapeSparkCommitment<E>,
  vk_digest: E::Scalar, // digest of verifier's key
}

/// A type that represents the verifier's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<E: Engine, EE: EvaluationEngineTrait<E>> {
  num_cons: usize,
  num_vars: usize,
  vk_ee: EE::VerifierKey,
  S_comm: R1CSShapeSparkCommitment<E>,
  #[serde(skip, default = "OnceCell::new")]
  digest: OnceCell<E::Scalar>,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> SimpleDigestible for VerifierKey<E, EE> {}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<E: Engine, EE: EvaluationEngineTrait<E>> {
  // commitment to oracles: the first three are for Az, Bz, Cz,
  // and the last two are for memory reads
  comm_Az: CompressedCommitment<E>,
  comm_Bz: CompressedCommitment<E>,
  comm_Cz: CompressedCommitment<E>,
  comm_L_row: CompressedCommitment<E>,
  comm_L_col: CompressedCommitment<E>,

  // commitments to aid the memory checks
  comm_t_plus_r_inv_row: CompressedCommitment<E>,
  comm_w_plus_r_inv_row: CompressedCommitment<E>,
  comm_t_plus_r_inv_col: CompressedCommitment<E>,
  comm_w_plus_r_inv_col: CompressedCommitment<E>,

  // claims about Az, Bz, and Cz polynomials
  eval_Az_at_tau: E::Scalar,
  eval_Bz_at_tau: E::Scalar,
  eval_Cz_at_tau: E::Scalar,

  // sum-check
  sc: SumcheckProof<E>,

  // claims from the end of sum-check
  eval_Az: E::Scalar,
  eval_Bz: E::Scalar,
  eval_Cz: E::Scalar,
  eval_E: E::Scalar,
  eval_L_row: E::Scalar,
  eval_L_col: E::Scalar,
  eval_val_A: E::Scalar,
  eval_val_B: E::Scalar,
  eval_val_C: E::Scalar,

  eval_W: E::Scalar,

  eval_t_plus_r_inv_row: E::Scalar,
  eval_row: E::Scalar, // address
  eval_w_plus_r_inv_row: E::Scalar,
  eval_ts_row: E::Scalar,

  eval_t_plus_r_inv_col: E::Scalar,
  eval_col: E::Scalar, // address
  eval_w_plus_r_inv_col: E::Scalar,
  eval_ts_col: E::Scalar,

  // a PCS evaluation argument
  eval_arg: EE::EvaluationArgument,
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARK<E, EE> {
  fn prove_helper<T1, T2, T3, T4>(
    mem: &mut T1,
    outer: &mut T2,
    inner: &mut T3,
    witness: &mut T4,
    transcript: &mut E::TE,
  ) -> Result<
    (
      SumcheckProof<E>,
      Vec<E::Scalar>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
      Vec<Vec<E::Scalar>>,
    ),
    NovaError,
  >
  where
    T1: SumcheckEngine<E>,
    T2: SumcheckEngine<E>,
    T3: SumcheckEngine<E>,
    T4: SumcheckEngine<E>,
  {
    // sanity checks
    assert_eq!(mem.size(), outer.size());
    assert_eq!(mem.size(), inner.size());
    assert_eq!(mem.size(), witness.size());
    assert_eq!(mem.degree(), outer.degree());
    assert_eq!(mem.degree(), inner.degree());
    assert_eq!(mem.degree(), witness.degree());

    // these claims are already added to the transcript, so we do not need to add
    let claims = mem
      .initial_claims()
      .into_iter()
      .chain(outer.initial_claims())
      .chain(inner.initial_claims())
      .chain(witness.initial_claims())
      .collect::<Vec<E::Scalar>>();

    let s = transcript.squeeze(b"r")?;
    let coeffs = powers::<E>(&s, claims.len());

    // compute the joint claim
    let claim = zip_with!((claims.iter(), coeffs.iter()), |c_1, c_2| *c_1 * c_2).sum();

    let mut e = claim;
    let mut r: Vec<E::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::new();
    let num_rounds = mem.size().log_2();
    for _ in 0..num_rounds {
      let ((evals_mem, evals_outer), (evals_inner, evals_witness)) = rayon::join(
        || rayon::join(|| mem.evaluation_points(), || outer.evaluation_points()),
        || rayon::join(|| inner.evaluation_points(), || witness.evaluation_points()),
      );

      let evals: Vec<Vec<E::Scalar>> = evals_mem
        .into_iter()
        .chain(evals_outer.into_iter())
        .chain(evals_inner.into_iter())
        .chain(evals_witness.into_iter())
        .collect::<Vec<Vec<E::Scalar>>>();
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
        || rayon::join(|| mem.bound(&r_i), || outer.bound(&r_i)),
        || rayon::join(|| inner.bound(&r_i), || witness.bound(&r_i)),
      );

      e = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    let mem_claims = mem.final_claims();
    let outer_claims = outer.final_claims();
    let inner_claims = inner.final_claims();
    let witness_claims = witness.final_claims();

    Ok((
      SumcheckProof::new(cubic_polys),
      r,
      mem_claims,
      outer_claims,
      inner_claims,
      witness_claims,
    ))
  }
}

impl<E: Engine, EE: EvaluationEngineTrait<E>> VerifierKey<E, EE> {
  fn new(
    num_cons: usize,
    num_vars: usize,
    S_comm: R1CSShapeSparkCommitment<E>,
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
impl<E: Engine, EE: EvaluationEngineTrait<E>> DigestHelperTrait<E> for VerifierKey<E, EE> {
  /// Returns the digest of the verifier's key
  fn digest(&self) -> E::Scalar {
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

impl<E: Engine, EE: EvaluationEngineTrait<E>> RelaxedR1CSSNARKTrait<E> for RelaxedR1CSSNARK<E, EE> {
  type ProverKey = ProverKey<E, EE>;
  type VerifierKey = VerifierKey<E, EE>;

  fn ck_floor() -> Box<dyn for<'a> Fn(&'a R1CSShape<E>) -> usize> {
    Box::new(|shape: &R1CSShape<E>| -> usize {
      // the commitment key should be large enough to commit to the R1CS matrices
      shape.A.len() + shape.B.len() + shape.C.len()
    })
  }

  fn setup(
    ck: &CommitmentKey<E>,
    S: &R1CSShape<E>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    // check the provided commitment key meets minimal requirements
    if ck.length() < Self::ck_floor()(S) {
      return Err(NovaError::InvalidCommitmentKeyLength);
    }
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
    ck: &CommitmentKey<E>,
    pk: &Self::ProverKey,
    S: &R1CSShape<E>,
    U: &RelaxedR1CSInstance<E>,
    W: &RelaxedR1CSWitness<E>,
  ) -> Result<Self, NovaError> {
    // pad the R1CSShape
    let S = S.pad();
    // sanity check that R1CSShape has all required size characteristics
    assert!(S.is_regular_shape());

    let W = W.pad(&S); // pad the witness
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the verifier key (which includes commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();

    // compute Az, Bz, Cz
    let (mut Az, mut Bz, mut Cz) = S.multiply_vec(&z)?;

    // commit to Az, Bz, Cz
    let (comm_Az, (comm_Bz, comm_Cz)) = rayon::join(
      || E::CE::commit(ck, &Az),
      || rayon::join(|| E::CE::commit(ck, &Bz), || E::CE::commit(ck, &Cz)),
    );

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());

    // number of rounds of sum-check
    let num_rounds_sc = pk.S_repr.N.log_2();
    let tau = transcript.squeeze(b"t")?;
    let tau_coords = PowPolynomial::new(&tau, num_rounds_sc).coordinates();

    // (1) send commitments to Az, Bz, and Cz along with their evaluations at tau
    let (Az, Bz, Cz, W, E) = {
      Az.resize(pk.S_repr.N, E::Scalar::ZERO);
      Bz.resize(pk.S_repr.N, E::Scalar::ZERO);
      Cz.resize(pk.S_repr.N, E::Scalar::ZERO);
      let E = padded::<E>(&W.E, pk.S_repr.N, &E::Scalar::ZERO);
      let W = padded::<E>(&W.W, pk.S_repr.N, &E::Scalar::ZERO);

      (Az, Bz, Cz, W, E)
    };
    let (eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau) = {
      let evals_at_tau = [&Az, &Bz, &Cz]
        .into_par_iter()
        .map(|p| MultilinearPolynomial::evaluate_with(p, &tau_coords))
        .collect::<Vec<E::Scalar>>();
      (evals_at_tau[0], evals_at_tau[1], evals_at_tau[2])
    };

    // (2) send commitments to the following two oracles
    // L_row(i) = eq(tau, row(i)) for all i
    // L_col(i) = z(col(i)) for all i
    let (mem_row, mem_col, L_row, L_col) = pk.S_repr.evaluation_oracles(&S, &tau, &z);
    let (comm_L_row, comm_L_col) =
      rayon::join(|| E::CE::commit(ck, &L_row), || E::CE::commit(ck, &L_col));

    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau];

    // absorb the claimed evaluations into the transcript
    transcript.absorb(b"e", &eval_vec.as_slice());
    // absorb commitments to L_row and L_col in the transcript
    transcript.absorb(b"e", &vec![comm_L_row, comm_L_col].as_slice());
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    let poly_vec = vec![&Az, &Bz, &Cz];
    let c = transcript.squeeze(b"c")?;
    let w: PolyEvalWitness<E> = PolyEvalWitness::batch(&poly_vec, &c);
    let u: PolyEvalInstance<E> = PolyEvalInstance::batch(&comm_vec, &tau_coords, &eval_vec, &c);

    // we now need to prove four claims
    // (1) 0 = \sum_x poly_tau(x) * (poly_Az(x) * poly_Bz(x) - poly_uCz_E(x)), and eval_Az_at_tau + r * eval_Bz_at_tau + r^2 * eval_Cz_at_tau = (Az+r*Bz+r^2*Cz)(tau)
    // (2) eval_Az_at_tau + c * eval_Bz_at_tau + c^2 * eval_Cz_at_tau = \sum_y L_row(y) * (val_A(y) + c * val_B(y) + c^2 * val_C(y)) * L_col(y)
    // (3) L_row(i) = eq(tau, row(i)) and L_col(i) = z(col(i))
    // (4) Check that the witness polynomial W is well-formed e.g., it is padded with only zeros
    let gamma = transcript.squeeze(b"g")?;
    let r = transcript.squeeze(b"r")?;

    let ((mut outer_sc_inst, mut inner_sc_inst), mem_res) = rayon::join(
      || {
        // a sum-check instance to prove the first claim
        let outer_sc_inst = OuterSumcheckInstance::new(
          PowPolynomial::new(&tau, num_rounds_sc).evals(),
          Az.clone(),
          Bz.clone(),
          (0..Cz.len())
            .map(|i| U.u * Cz[i] + E[i])
            .collect::<Vec<E::Scalar>>(),
          w.p.clone(), // Mz = Az + r * Bz + r^2 * Cz
          &u.e,        // eval_Az_at_tau + r * eval_Az_at_tau + r^2 * eval_Cz_at_tau
        );

        // a sum-check instance to prove the second claim
        let val = zip_with!(
          par_iter,
          (pk.S_repr.val_A, pk.S_repr.val_B, pk.S_repr.val_C),
          |v_a, v_b, v_c| *v_a + c * *v_b + c * c * *v_c
        )
        .collect::<Vec<E::Scalar>>();
        let inner_sc_inst = InnerSumcheckInstance {
          claim: eval_Az_at_tau + c * eval_Bz_at_tau + c * c * eval_Cz_at_tau,
          poly_L_row: MultilinearPolynomial::new(L_row.clone()),
          poly_L_col: MultilinearPolynomial::new(L_col.clone()),
          poly_val: MultilinearPolynomial::new(val),
        };

        (outer_sc_inst, inner_sc_inst)
      },
      || {
        // a third sum-check instance to prove the read-only memory claim
        // we now need to prove that L_row and L_col are well-formed

        // hash the tuples of (addr,val) memory contents and read responses into a single field element using `hash_func`

        let (comm_mem_oracles, mem_oracles, mem_aux) =
          MemorySumcheckInstance::<E>::compute_oracles(
            ck,
            &r,
            &gamma,
            &mem_row,
            &pk.S_repr.row,
            &L_row,
            &pk.S_repr.ts_row,
            &mem_col,
            &pk.S_repr.col,
            &L_col,
            &pk.S_repr.ts_col,
          )?;
        // absorb the commitments
        transcript.absorb(b"l", &comm_mem_oracles.as_slice());

        let rho = transcript.squeeze(b"r")?;
        let poly_eq = MultilinearPolynomial::new(PowPolynomial::new(&rho, num_rounds_sc).evals());

        Ok::<_, NovaError>((
          MemorySumcheckInstance::new(
            mem_oracles.clone(),
            mem_aux,
            poly_eq.Z,
            pk.S_repr.ts_row.clone(),
            pk.S_repr.ts_col.clone(),
          ),
          comm_mem_oracles,
          mem_oracles,
        ))
      },
    );

    let (mut mem_sc_inst, comm_mem_oracles, mem_oracles) = mem_res?;

    let mut witness_sc_inst = WitnessBoundSumcheck::new(tau, W.clone(), S.num_vars);

    let (sc, rand_sc, claims_mem, claims_outer, claims_inner, claims_witness) = Self::prove_helper(
      &mut mem_sc_inst,
      &mut outer_sc_inst,
      &mut inner_sc_inst,
      &mut witness_sc_inst,
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
    let eval_W = claims_witness[0][0];

    // compute the remaining claims that did not come for free from the sum-check prover
    let (eval_Cz, eval_E, eval_val_A, eval_val_B, eval_val_C, eval_row, eval_col) = {
      let e = [
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
      .collect::<Vec<E::Scalar>>();
      (e[0], e[1], e[2], e[3], e[4], e[5], e[6])
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
    .collect::<Vec<E::Scalar>>();

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
      comm_mem_oracles[0],
      pk.S_comm.comm_row,
      comm_mem_oracles[1],
      pk.S_comm.comm_ts_row,
      comm_mem_oracles[2],
      pk.S_comm.comm_col,
      comm_mem_oracles[3],
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
      mem_oracles[0].as_ref(),
      &pk.S_repr.row,
      mem_oracles[1].as_ref(),
      &pk.S_repr.ts_row,
      mem_oracles[2].as_ref(),
      &pk.S_repr.col,
      mem_oracles[3].as_ref(),
      &pk.S_repr.ts_col,
    ];
    transcript.absorb(b"e", &eval_vec.as_slice()); // comm_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let w: PolyEvalWitness<E> = PolyEvalWitness::batch(&poly_vec, &c);
    let u: PolyEvalInstance<E> = PolyEvalInstance::batch(&comm_vec, &rand_sc, &eval_vec, &c);

    let eval_arg = EE::prove(ck, &pk.pk_ee, &mut transcript, &u.c, &w.p, &rand_sc, &u.e)?;

    Ok(RelaxedR1CSSNARK {
      comm_Az: comm_Az.compress(),
      comm_Bz: comm_Bz.compress(),
      comm_Cz: comm_Cz.compress(),
      comm_L_row: comm_L_row.compress(),
      comm_L_col: comm_L_col.compress(),

      comm_t_plus_r_inv_row: comm_mem_oracles[0].compress(),
      comm_w_plus_r_inv_row: comm_mem_oracles[1].compress(),
      comm_t_plus_r_inv_col: comm_mem_oracles[2].compress(),
      comm_w_plus_r_inv_col: comm_mem_oracles[3].compress(),

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
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<E>) -> Result<(), NovaError> {
    let mut transcript = E::TE::new(b"RelaxedR1CSSNARK");

    // append the verifier key (including commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest());
    transcript.absorb(b"U", U);

    let comm_Az = Commitment::<E>::decompress(&self.comm_Az)?;
    let comm_Bz = Commitment::<E>::decompress(&self.comm_Bz)?;
    let comm_Cz = Commitment::<E>::decompress(&self.comm_Cz)?;
    let comm_L_row = Commitment::<E>::decompress(&self.comm_L_row)?;
    let comm_L_col = Commitment::<E>::decompress(&self.comm_L_col)?;
    let comm_t_plus_r_inv_row = Commitment::<E>::decompress(&self.comm_t_plus_r_inv_row)?;
    let comm_w_plus_r_inv_row = Commitment::<E>::decompress(&self.comm_w_plus_r_inv_row)?;
    let comm_t_plus_r_inv_col = Commitment::<E>::decompress(&self.comm_t_plus_r_inv_col)?;
    let comm_w_plus_r_inv_col = Commitment::<E>::decompress(&self.comm_w_plus_r_inv_col)?;

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());

    let num_rounds_sc = vk.S_comm.N.log_2();
    let tau = transcript.squeeze(b"t")?;
    let tau_coords = PowPolynomial::new(&tau, num_rounds_sc).coordinates();

    // add claims about Az, Bz, and Cz to be checked later
    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![
      self.eval_Az_at_tau,
      self.eval_Bz_at_tau,
      self.eval_Cz_at_tau,
    ];

    transcript.absorb(b"e", &eval_vec.as_slice());

    transcript.absorb(b"e", &vec![comm_L_row, comm_L_col].as_slice());
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    let c = transcript.squeeze(b"c")?;
    let u: PolyEvalInstance<E> = PolyEvalInstance::batch(&comm_vec, &tau_coords, &eval_vec, &c);
    let claim = u.e;

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

    let num_claims = 10;
    let s = transcript.squeeze(b"r")?;
    let coeffs = powers::<E>(&s, num_claims);
    let claim = (coeffs[7] + coeffs[8]) * claim; // rest are zeros

    // verify sc
    let (claim_sc_final, rand_sc) = self.sc.verify(claim, num_rounds_sc, 3, &mut transcript)?;

    // verify claim_sc_final
    let claim_sc_final_expected = {
      let rand_eq_bound_rand_sc = {
        let poly_eq_coords = PowPolynomial::new(&rho, num_rounds_sc).coordinates();
        EqPolynomial::new(poly_eq_coords).evaluate(&rand_sc)
      };
      let taus_coords = PowPolynomial::new(&tau, num_rounds_sc).coordinates();
      let eq_tau = EqPolynomial::new(taus_coords);

      let taus_bound_rand_sc = eq_tau.evaluate(&rand_sc);
      let taus_masked_bound_rand_sc =
        MaskedEqPolynomial::new(&eq_tau, vk.num_vars.log_2()).evaluate(&rand_sc);

      let eval_t_plus_r_row = {
        let eval_addr_row = IdentityPolynomial::new(num_rounds_sc).evaluate(&rand_sc);
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
        let eval_addr_col = IdentityPolynomial::new(num_rounds_sc).evaluate(&rand_sc);

        // memory contents is z, so we compute eval_Z from eval_W and eval_X
        let eval_val_col = {
          // rand_sc was padded, so we now remove the padding
          let (factor, rand_sc_unpad) = {
            let l = vk.S_comm.N.log_2() - (2 * vk.num_vars).log_2();

            let mut factor = E::Scalar::ONE;
            for r_p in rand_sc.iter().take(l) {
              factor *= E::Scalar::ONE - r_p
            }

            let rand_sc_unpad = rand_sc[l..].to_vec();

            (factor, rand_sc_unpad)
          };

          let eval_X = {
            // public IO is (u, X)
            let X = vec![U.u]
              .into_iter()
              .chain(U.X.iter().cloned())
              .collect::<Vec<E::Scalar>>();

            // evaluate the sparse polynomial at rand_sc_unpad[1..]
            let poly_X = SparsePolynomial::new(rand_sc_unpad.len() - 1, X);
            poly_X.evaluate(&rand_sc_unpad[1..])
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

      let claim_mem_final_expected: E::Scalar = coeffs[0]
        * (self.eval_t_plus_r_inv_row - self.eval_w_plus_r_inv_row)
        + coeffs[1] * (self.eval_t_plus_r_inv_col - self.eval_w_plus_r_inv_col)
        + coeffs[2]
          * (rand_eq_bound_rand_sc
            * (self.eval_t_plus_r_inv_row * eval_t_plus_r_row - self.eval_ts_row))
        + coeffs[3]
          * (rand_eq_bound_rand_sc
            * (self.eval_w_plus_r_inv_row * eval_w_plus_r_row - E::Scalar::ONE))
        + coeffs[4]
          * (rand_eq_bound_rand_sc
            * (self.eval_t_plus_r_inv_col * eval_t_plus_r_col - self.eval_ts_col))
        + coeffs[5]
          * (rand_eq_bound_rand_sc
            * (self.eval_w_plus_r_inv_col * eval_w_plus_r_col - E::Scalar::ONE));

      let claim_outer_final_expected = coeffs[6]
        * taus_bound_rand_sc
        * (self.eval_Az * self.eval_Bz - U.u * self.eval_Cz - self.eval_E)
        + coeffs[7] * taus_bound_rand_sc * (self.eval_Az + c * self.eval_Bz + c * c * self.eval_Cz);
      let claim_inner_final_expected = coeffs[8]
        * self.eval_L_row
        * self.eval_L_col
        * (self.eval_val_A + c * self.eval_val_B + c * c * self.eval_val_C);

      let claim_witness_final_expected = coeffs[9] * taus_masked_bound_rand_sc * self.eval_W;

      claim_mem_final_expected
        + claim_outer_final_expected
        + claim_inner_final_expected
        + claim_witness_final_expected
    };

    if claim_sc_final_expected != claim_sc_final {
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
    .collect::<Vec<E::Scalar>>();
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
    let u: PolyEvalInstance<E> = PolyEvalInstance::batch(&comm_vec, &rand_sc, &eval_vec, &c);

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
