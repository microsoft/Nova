//! Inverse-logup memory-check for ppSNARK (the original Microsoft Nova
//! implementation).
//!
//! Selected when the `logup-no-gkr` feature is enabled; otherwise ppSNARK uses
//! the Logup-GKR memory-check in [`super::mem_check_logup_gkr`]. This module
//! holds the pieces specific to the inverse-logup approach — the
//! [`MemorySumcheckInstance`] (six-route sumcheck proving
//! `Σ TS[i]/(T[i]+r) − 1/(W[i]+r) = 0` per row/col via committed inverse
//! oracles) and the [`LogupProofData`] bundle of its proof fields — so they can
//! be feature-gated out of the SNARK cleanly.

use crate::{
  errors::NovaError,
  spartan::{
    batch_invert,
    math::Math,
    polys::{
      eq::EqPolynomial, identity::IdentityPolynomial, multilinear::MultilinearPolynomial,
      multilinear::SparsePolynomial,
    },
    sumcheck::{eq_sumcheck::EqSumCheckInstance, SumcheckEngine, SumcheckProof},
  },
  traits::{
    commitment::CommitmentEngineTrait, evm_serde::EvmCompatSerde, Engine, TranscriptEngineTrait,
  },
  zip_with, Commitment, CommitmentKey,
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// The inverse-logup memory-check proof fields carried in the ppSNARK proof:
/// commitments to the four inverse oracles (`1/(T+r)·TS`, `1/(W+r)` for row/col)
/// and their evaluations at the shared inner point. Bundled so the SNARK can
/// gate the whole inverse-logup path behind one field.
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogupProofData<E: Engine> {
  /// Commitment to `TS_row/(T_row+r)`.
  pub comm_t_plus_r_inv_row: Commitment<E>,
  /// Commitment to `1/(W_row+r)`.
  pub comm_w_plus_r_inv_row: Commitment<E>,
  /// Commitment to `TS_col/(T_col+r)`.
  pub comm_t_plus_r_inv_col: Commitment<E>,
  /// Commitment to `1/(W_col+r)`.
  pub comm_w_plus_r_inv_col: Commitment<E>,
  /// Evaluation of `TS_row/(T_row+r)` at the inner point.
  #[serde_as(as = "EvmCompatSerde")]
  pub eval_t_plus_r_inv_row: E::Scalar,
  /// Evaluation of `1/(W_row+r)` at the inner point.
  #[serde_as(as = "EvmCompatSerde")]
  pub eval_w_plus_r_inv_row: E::Scalar,
  /// Evaluation of `TS_col/(T_col+r)` at the inner point.
  #[serde_as(as = "EvmCompatSerde")]
  pub eval_t_plus_r_inv_col: E::Scalar,
  /// Evaluation of `1/(W_col+r)` at the inner point.
  #[serde_as(as = "EvmCompatSerde")]
  pub eval_w_plus_r_inv_col: E::Scalar,
}

/// Memory sumcheck instance for PPSNARK LogUp
pub struct MemorySumcheckInstance<E: Engine> {
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

  eq_sumcheck: EqSumCheckInstance<E>,

  // Per-claim running claims and saved evaluation points (BDDT, eprint 2025/1117 Section 6.2)
  running_claims: [E::Scalar; 6],
  saved_evals: [[E::Scalar; 3]; 6],
}

impl<E: Engine> MemorySumcheckInstance<E> {
  /// Computes witnesses for MemoryInstanceSumcheck
  ///
  /// # Description
  /// We use the logUp protocol to prove that
  /// sum TS\[i\]/(T\[i\] + r) - 1/(W\[i\] + r) = 0
  /// where
  ///   T_row\[i\] = mem_row\[i\]      * gamma + i
  ///            = eq(tau)\[i\]      * gamma + i
  ///   W_row\[i\] = L_row\[i\]        * gamma + addr_row\[i\]
  ///            = eq(tau)\[row\[i\]\] * gamma + addr_row\[i\]
  ///   T_col\[i\] = mem_col\[i\]      * gamma + i
  ///            = z\[i\]            * gamma + i
  ///   W_col\[i\] = L_col\[i\]     * gamma + addr_col\[i\]
  ///            = z\[col\[i\]\]       * gamma + addr_col\[i\]
  /// and
  ///   TS_row, TS_col are integer-valued vectors representing the number of reads
  ///   to each memory cell of L_row, L_col
  ///
  /// The function returns oracles for the polynomials TS\[i\]/(T\[i\] + r), 1/(W\[i\] + r),
  /// as well as auxiliary polynomials T\[i\] + r, W\[i\] + r
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

    // compute vectors TS[i]/(T[i] + r) and 1/(W[i] + r)
    let helper = |T: &[E::Scalar],
                  W: &[E::Scalar],
                  TS: &[E::Scalar],
                  r: &E::Scalar|
     -> Result<
      (
        Vec<E::Scalar>,
        Vec<E::Scalar>,
        Vec<E::Scalar>,
        Vec<E::Scalar>,
      ),
      NovaError,
    > {
      let t_plus_r_and_w_plus_r = T
        .par_iter()
        .chain(W.par_iter())
        .map(|e| *e + *r)
        .collect::<Vec<E::Scalar>>();

      let inv = batch_invert(&t_plus_r_and_w_plus_r)?;

      let mut t_plus_r = t_plus_r_and_w_plus_r;
      let w_plus_r = t_plus_r.split_off(T.len());

      let mut t_plus_r_inv = inv;
      let w_plus_r_inv = t_plus_r_inv.split_off(T.len());

      // compute inv[i] * TS[i] in parallel
      t_plus_r_inv = zip_with!((t_plus_r_inv.into_par_iter(), TS.par_iter()), |e1, e2| e1
        * *e2)
      .collect::<Vec<_>>();

      Ok((t_plus_r_inv, w_plus_r_inv, t_plus_r, w_plus_r))
    };

    let (row, col) = rayon::join(
      || helper(&T_row, &W_row, ts_row, r),
      || helper(&T_col, &W_col, ts_col, r),
    );

    let (t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_row, w_plus_r_row) = row?;
    let (t_plus_r_inv_col, w_plus_r_inv_col, t_plus_r_col, w_plus_r_col) = col?;

    let (
      (comm_t_plus_r_inv_row, comm_w_plus_r_inv_row),
      (comm_t_plus_r_inv_col, comm_w_plus_r_inv_col),
    ) = rayon::join(
      || {
        rayon::join(
          || E::CE::commit(ck, &t_plus_r_inv_row, &E::Scalar::ZERO),
          || E::CE::commit(ck, &w_plus_r_inv_row, &E::Scalar::ZERO),
        )
      },
      || {
        rayon::join(
          || E::CE::commit(ck, &t_plus_r_inv_col, &E::Scalar::ZERO),
          || E::CE::commit(ck, &w_plus_r_inv_col, &E::Scalar::ZERO),
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

    let aux_poly_vec = [t_plus_r_row, w_plus_r_row, t_plus_r_col, w_plus_r_col];

    Ok((comm_vec, poly_vec, aux_poly_vec))
  }

  /// Create a new memory sumcheck instance
  pub fn new(
    polys_oracle: [Vec<E::Scalar>; 4],
    polys_aux: [Vec<E::Scalar>; 4],
    rhos: Vec<E::Scalar>,
    ts_row: Vec<E::Scalar>,
    ts_col: Vec<E::Scalar>,
  ) -> Self {
    let [t_plus_r_inv_row, w_plus_r_inv_row, t_plus_r_inv_col, w_plus_r_inv_col] = polys_oracle;
    let [t_plus_r_row, w_plus_r_row, t_plus_r_col, w_plus_r_col] = polys_aux;

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
      eq_sumcheck: EqSumCheckInstance::new(rhos),
      running_claims: [E::Scalar::ZERO; 6],
      saved_evals: [[E::Scalar::ZERO; 3]; 6],
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

  fn evaluation_points(&mut self) -> Vec<Vec<E::Scalar>> {
    // Pre-borrow all fields as shared references for parallel access
    let eq = &self.eq_sumcheck;
    let running_claims = &self.running_claims;
    let t_plus_r_inv_row = &self.t_plus_r_inv_row;
    let w_plus_r_inv_row = &self.w_plus_r_inv_row;
    let t_plus_r_row = &self.t_plus_r_row;
    let w_plus_r_row = &self.w_plus_r_row;
    let ts_row = &self.ts_row;
    let t_plus_r_inv_col = &self.t_plus_r_inv_col;
    let w_plus_r_inv_col = &self.w_plus_r_inv_col;
    let t_plus_r_col = &self.t_plus_r_col;
    let w_plus_r_col = &self.w_plus_r_col;
    let ts_col = &self.ts_col;

    // inv related evaluation points for linear (A - B) pattern (no claim derivation)
    // 0 = sum TS[i]/(T[i] + r) - 1/(W[i] + r)
    let (
      ((eval_inv_0_row, eval_inv_3_row), (eval_inv_0_col, eval_inv_3_col)),
      (
        ((eval_T_0_row, eval_T_2_row, eval_T_3_row), (eval_W_0_row, eval_W_2_row, eval_W_3_row)),
        ((eval_T_0_col, eval_T_2_col, eval_T_3_col), (eval_W_0_col, eval_W_2_col, eval_W_3_col)),
      ),
    ) = rayon::join(
      || {
        rayon::join(
          || SumcheckProof::<E>::compute_eval_points_linear(t_plus_r_inv_row, w_plus_r_inv_row),
          || SumcheckProof::<E>::compute_eval_points_linear(t_plus_r_inv_col, w_plus_r_inv_col),
        )
      },
      || {
        rayon::join(
          || {
            // Row evaluation points (claim-derived, BDDT Section 6.2)
            rayon::join(
              || {
                // 0 = sum eq[i] * (inv_T[i] * (T[i] + r) - TS[i]))
                eq.evaluation_points_cubic_with_three_inputs(
                  t_plus_r_inv_row,
                  t_plus_r_row,
                  ts_row,
                  running_claims[2],
                )
              },
              || {
                // 0 = sum eq[i] * (inv_W[i] * (W[i] + r) - 1))
                eq.evaluation_points_cubic_with_two_inputs(
                  w_plus_r_inv_row,
                  w_plus_r_row,
                  running_claims[3],
                )
              },
            )
          },
          || {
            // Column evaluation points (claim-derived, BDDT Section 6.2)
            rayon::join(
              || {
                eq.evaluation_points_cubic_with_three_inputs(
                  t_plus_r_inv_col,
                  t_plus_r_col,
                  ts_col,
                  running_claims[4],
                )
              },
              || {
                eq.evaluation_points_cubic_with_two_inputs(
                  w_plus_r_inv_col,
                  w_plus_r_col,
                  running_claims[5],
                )
              },
            )
          },
        )
      },
    );

    // Save evaluation points for running claim updates in bound()
    self.saved_evals = [
      [eval_inv_0_row, E::Scalar::ZERO, eval_inv_3_row],
      [eval_inv_0_col, E::Scalar::ZERO, eval_inv_3_col],
      [eval_T_0_row, eval_T_2_row, eval_T_3_row],
      [eval_W_0_row, eval_W_2_row, eval_W_3_row],
      [eval_T_0_col, eval_T_2_col, eval_T_3_col],
      [eval_W_0_col, eval_W_2_col, eval_W_3_col],
    ];

    self.saved_evals.iter().map(|e| e.to_vec()).collect()
  }

  fn bound(&mut self, r: &E::Scalar) {
    for j in 0..6 {
      self.running_claims[j] =
        SumcheckProof::<E>::update_claim(self.running_claims[j], &self.saved_evals[j], r);
    }

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
    ]
    .par_iter_mut()
    .for_each(|poly| poly.bind_poly_var_top(r));

    self.eq_sumcheck.bound(r);
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

/// Prover side of the inverse-logup memory-check: builds the inverse oracles,
/// commits to them, absorbs the commitments, squeezes the `rho` challenges, and
/// returns the `MemorySumcheckInstance` (the first `prove_helper` slot) plus the
/// oracle commitments and polynomials (needed later for the batched PCS
/// opening). `addr_row`/`addr_col` are `S_repr.row`/`col`.
#[allow(clippy::too_many_arguments)]
pub fn prove_step<E: Engine>(
  ck: &CommitmentKey<E>,
  r: E::Scalar,
  gamma: E::Scalar,
  mem_row: &[E::Scalar],
  addr_row: &[E::Scalar],
  L_row: &[E::Scalar],
  ts_row: &[E::Scalar],
  mem_col: &[E::Scalar],
  addr_col: &[E::Scalar],
  L_col: &[E::Scalar],
  ts_col: &[E::Scalar],
  num_rounds_inner: usize,
  transcript: &mut E::TE,
) -> Result<
  (
    MemorySumcheckInstance<E>,
    [Commitment<E>; 4],
    [Vec<E::Scalar>; 4],
  ),
  NovaError,
> {
  let (comm_mem_oracles, mem_oracles, mem_aux) = MemorySumcheckInstance::<E>::compute_oracles(
    ck, &r, &gamma, mem_row, addr_row, L_row, ts_row, mem_col, addr_col, L_col, ts_col,
  )?;
  transcript.absorb(b"l", &comm_mem_oracles.as_slice());
  let rho = (0..num_rounds_inner)
    .map(|_| transcript.squeeze(b"r"))
    .collect::<Result<Vec<_>, NovaError>>()?;
  let inst = MemorySumcheckInstance::new(
    mem_oracles.clone(),
    mem_aux,
    rho,
    ts_row.to_vec(),
    ts_col.to_vec(),
  );
  Ok((inst, comm_mem_oracles, mem_oracles))
}

/// Number of batched-inner claims the memory-check slot contributes (the six
/// inverse-logup routes at coeffs `[0, 6)`). `prove_helper` places the
/// memory-check slot first, so the inner/witness claims come after these.
pub const NUM_MEM_CLAIMS: usize = 6;

/// The inverse-logup contribution to the batched inner sumcheck's **initial**
/// claim. The six memory routes prove `0 = Σ ...`, so their combined initial
/// claim is zero — this exists for symmetry with the Logup-GKR slot's
/// `verify_initial_claim`, so the caller adds one memory-check contribution
/// regardless of which implementation is active.
pub fn verify_initial_claim<E: Engine>(_coeffs: &[E::Scalar]) -> E::Scalar {
  E::Scalar::ZERO
}

/// Verifier side, transcript phase: absorbs the four inverse-oracle commitments
/// and squeezes the `rho` challenges (the eq randomness for the memory
/// sumcheck), mirroring [`prove_step`]. Must run before the inner sumcheck's `s`
/// challenge is drawn.
pub fn verify_pre_inner<E: Engine>(
  data: &LogupProofData<E>,
  num_rounds_inner: usize,
  transcript: &mut E::TE,
) -> Result<Vec<E::Scalar>, NovaError> {
  transcript.absorb(
    b"l",
    &vec![
      data.comm_t_plus_r_inv_row,
      data.comm_w_plus_r_inv_row,
      data.comm_t_plus_r_inv_col,
      data.comm_w_plus_r_inv_col,
    ]
    .as_slice(),
  );
  (0..num_rounds_inner)
    .map(|_| transcript.squeeze(b"r"))
    .collect()
}

/// Verifier side, final-claim phase: reconstructs the memory-check contribution
/// to the batched inner sumcheck's final claim — the six routes at coeff indices
/// `0..6`, proving `Σ TS/(T+r) − 1/(W+r) = 0` and the well-formedness of the
/// four inverse oracles. `rho` is from [`verify_pre_inner`]; the eval arguments
/// are the shared evaluations at `r_inner_batched`.
#[allow(clippy::too_many_arguments)]
pub fn verify_final_claim<E: Engine>(
  data: &LogupProofData<E>,
  coeffs: &[E::Scalar],
  rho: Vec<E::Scalar>,
  gamma: E::Scalar,
  r: E::Scalar,
  r_outer_full: &[E::Scalar],
  r_inner_batched: &[E::Scalar],
  num_rounds_inner: usize,
  num_vars: usize,
  n: usize,
  eval_W: E::Scalar,
  eval_L_row: E::Scalar,
  eval_L_col: E::Scalar,
  eval_row: E::Scalar,
  eval_col: E::Scalar,
  eval_ts_row: E::Scalar,
  eval_ts_col: E::Scalar,
  x: &[E::Scalar],
) -> E::Scalar {
  let rand_eq_bound = EqPolynomial::new(rho).evaluate(r_inner_batched);
  let eq_r_outer = EqPolynomial::new(r_outer_full.to_vec());
  let eq_r_outer_at_r_inner = eq_r_outer.evaluate(r_inner_batched);

  // mem_col = z at r_inner_batched, reconstructed from eval_W and public IO.
  let eval_mem_col_at_r_inner = {
    let (fac, unpad) = {
      let l = n.log_2() - (2 * num_vars).log_2();
      let mut fac = E::Scalar::ONE;
      for r_p in r_inner_batched.iter().take(l) {
        fac *= E::Scalar::ONE - r_p
      }
      (fac, r_inner_batched[l..].to_vec())
    };
    let eval_x = {
      let poly_x = SparsePolynomial::new(unpad.len() - 1, x.to_vec());
      poly_x.evaluate(&unpad[1..])
    };
    eval_W + fac * unpad[0] * eval_x
  };

  let eval_t_plus_r_row = {
    let eval_addr = IdentityPolynomial::new(num_rounds_inner).evaluate(r_inner_batched);
    eval_addr + gamma * eq_r_outer_at_r_inner + r // mem_row = eq(r_outer_full, ·)
  };
  let eval_w_plus_r_row = eval_row + gamma * eval_L_row + r;
  let eval_t_plus_r_col = {
    let eval_addr = IdentityPolynomial::new(num_rounds_inner).evaluate(r_inner_batched);
    eval_addr + gamma * eval_mem_col_at_r_inner + r
  };
  let eval_w_plus_r_col = eval_col + gamma * eval_L_col + r;

  coeffs[0] * (data.eval_t_plus_r_inv_row - data.eval_w_plus_r_inv_row)
    + coeffs[1] * (data.eval_t_plus_r_inv_col - data.eval_w_plus_r_inv_col)
    + coeffs[2] * (rand_eq_bound * (data.eval_t_plus_r_inv_row * eval_t_plus_r_row - eval_ts_row))
    + coeffs[3]
      * (rand_eq_bound * (data.eval_w_plus_r_inv_row * eval_w_plus_r_row - E::Scalar::ONE))
    + coeffs[4] * (rand_eq_bound * (data.eval_t_plus_r_inv_col * eval_t_plus_r_col - eval_ts_col))
    + coeffs[5]
      * (rand_eq_bound * (data.eval_w_plus_r_inv_col * eval_w_plus_r_col - E::Scalar::ONE))
}
