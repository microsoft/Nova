//! Bridge layer wiring Logup-GKR to ppSNARK's memory-check.
//!
//! This module is the *host* half of the Logup-GKR memory-check. The
//! `logup_gkr` core owns the fractional-sum reduction but no PCS; `ppsnark` owns
//! the commitments and inner sumcheck. This bridge supplies ppSNARK's four
//! equal-height sub-instances and fuses the final GKR layer with ppSNARK's inner
//! sumcheck so both reductions land on one shared evaluation point.
//!
//! The verifier closes soundness by (1) reconstructing each input-layer
//! fraction from the column evaluations claimed at the shared evaluation point,
//! (2) comparing those fractions with the GKR reduction, and (3) checking the
//! two fractional balances. [`MemCheckOpenings`] fixes the claimed columns and
//! the fused inner sumcheck plus batched PCS opening bind those claims to the
//! committed columns at the same point.
//!
//! ## The four sub-instances (why four, not two)
//! ppSNARK's memory-check is two logup relations (`row`, `col`), each a balance
//! `Σ ts/(T+r) = Σ 1/(W+r)`. We encode each relation as **two** height-N
//! fractional-sum sub-instances — a *table* side and an *access* side — so all
//! four share one GKR depth `log N`, as required by the batched GKR verifier;
//! every side is exactly N because ppSNARK pads every memory-check column to N
//! in setup. A single
//! 2N-leaf signed-multiplicity tree would instead emit a `log(2N)` point, which
//! cannot be rerandomized against the N-variable inner sumcheck; the two N-leaf
//! trees for each relation keep the point at `log N`. Instance order is fixed:
//!
//! | idx | name        | num       | den                         |
//! |-----|-------------|-----------|-----------------------------|
//! | 0   | row_table   | `ts_row`  | `mem_row·γ + id + r`        |
//! | 1   | row_access  | `-1`      | `L_row·γ + addr_row + r`    |
//! | 2   | col_table   | `ts_col`  | `mem_col·γ + id + r`        |
//! | 3   | col_access  | `-1`      | `L_col·γ + addr_col + r`    |
//!
//! where `id = IdentityPolynomial` (the cell address `i`), `mem_row =
//! eq(r_outer_full, ·)`, `mem_col = z`, `addr_row = row`, `addr_col = col`, and
//! `(γ, r)` are the ppSNARK memory-check fingerprint challenges.
//!
//! ## Balance (the host's `0/den` check)
//! The GKR verifier reduces each sub-instance to a root fraction but does **not**
//! check the relation balances — that is this module's job. Balance is a
//! property of the whole relation, i.e. the *root* fractions (the proof's
//! `initial_claims`, bound to the reduction by the GKR verifier), not the
//! input-layer evaluations. The access side carries `num = -1`, so `root_table +
//! root_access = Σ ts/(T+r) − Σ 1/(W+r)`, which must vanish. In projective form
//! that means the *sum's numerator* is zero (the denominator, a product of
//! nonzero dens, cannot be), checked for the row pair `(0,1)` and the col pair
//! `(2,3)`.

use crate::errors::NovaError;
use crate::spartan::logup_gkr::fraction::Fraction;
use crate::spartan::logup_gkr::layer::Layer;
use crate::spartan::logup_gkr::proof::LogupGkrPrefixProof;
use crate::spartan::mem_check_logup_gkr_fused::{
  prove_fused, verify_fused, verify_gkr_prefix, FusedEndpointEvals, FusedGkrInnerProof,
  FusedInnerInputs,
};
use crate::spartan::polys::eq::EqPolynomial;
use crate::spartan::polys::identity::IdentityPolynomial;
use crate::spartan::polys::masked_eq::MaskedEqPolynomial;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::spartan::polys::multilinear::SparsePolynomial;
use crate::traits::Engine;
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Fixed sub-instance count (`row_table, row_access, col_table, col_access`).
pub const NUM_SUB_INSTANCES: usize = 4;

/// The column evaluations a prover claims at the GKR
/// [`eval_point`](crate::spartan::logup_gkr::LogupGkrOpeningClaim::eval_point).
///
/// Every field is one column evaluated at the shared GKR point. The host uses
/// them to reconstruct the four sub-instance fractions. The fused inner
/// sumcheck carries the same point through to the batched PCS opening. The
/// verifier computes `id` and `mem_row = eq(r_outer_full, ·)` directly, so
/// neither needs a claim here.
#[derive(Clone, Copy, Debug)]
pub struct MemCheckOpenings<E: Engine> {
  /// `L_row(eval_point)` — the row lookup column.
  pub eval_L_row: E::Scalar,
  /// `L_col(eval_point)` — the col lookup column.
  pub eval_L_col: E::Scalar,
  /// `row(eval_point)` — the row access-address column (`addr_row`).
  pub eval_row: E::Scalar,
  /// `col(eval_point)` — the col access-address column (`addr_col`).
  pub eval_col: E::Scalar,
  /// `ts_row(eval_point)` — the row multiplicity column.
  pub eval_ts_row: E::Scalar,
  /// `ts_col(eval_point)` — the col multiplicity column.
  pub eval_ts_col: E::Scalar,
  /// `mem_col(eval_point) = z(eval_point)` — the col table-value column.
  ///
  /// `mem_row` is `eq(r_outer_full, ·)`, which the verifier evaluates directly,
  /// so only `mem_col` needs a claim.
  pub eval_mem_col: E::Scalar,
}

/// The prover's memory-check witness: the raw length-N columns fed into
/// Logup-GKR.
///
/// These are exactly ppSNARK's padded memory-check columns (all length
/// `N = 2^{log N}`). [`build_input_layers`]
/// turns them into the four GKR input layers, and [`prove_step_fused`] runs the
/// fused last-layer sumcheck over them while claiming the evaluations named by
/// [`MemCheckOpenings`] at the shared point. Field meanings match the
/// four-sub-instance table in the module docs:
/// - `mem_row = eq(r_outer_full, ·)`, `mem_col = z` (table values);
/// - `L_row`/`L_col` the lookup columns, `addr_row = row`/`addr_col = col` the
///   access addresses, `ts_row`/`ts_col` the multiplicities.
///
/// The columns are borrowed, not owned: [`build_input_layers`] only reads them
/// (deriving fresh fraction vectors), so the caller keeps ownership and pays no
/// per-column clone to assemble this witness.
pub struct MemCheckWitness<'a, E: Engine> {
  /// Row table values `mem_row = eq(r_outer_full, ·)`.
  pub mem_row: &'a [E::Scalar],
  /// Col table values `mem_col = z`.
  pub mem_col: &'a [E::Scalar],
  /// Row lookup column `L_row`.
  pub L_row: &'a [E::Scalar],
  /// Col lookup column `L_col`.
  pub L_col: &'a [E::Scalar],
  /// Row access addresses `addr_row = row`.
  pub addr_row: &'a [E::Scalar],
  /// Col access addresses `addr_col = col`.
  pub addr_col: &'a [E::Scalar],
  /// Row multiplicities `ts_row`.
  pub ts_row: &'a [E::Scalar],
  /// Col multiplicities `ts_col`.
  pub ts_col: &'a [E::Scalar],
}

/// Builds the four GKR input layers `[row_table, row_access, col_table,
/// col_access]` from the raw columns and the fingerprint `(gamma, r)`.
///
/// This is the prover-side dual of the verifier's per-instance reconstruction
/// in `reconcile_and_balance`: it must produce, for every leaf `i`, exactly
/// the fractions the verifier recomputes at `eval_point`. The layers, in order (matching the
/// module-doc table and [`NUM_SUB_INSTANCES`]):
///
/// | idx | name        | num        | den                         |
/// |-----|-------------|------------|-----------------------------|
/// | 0   | row_table   | `ts_row`   | `mem_row·γ + id + r`        |
/// | 1   | row_access  | `-1`       | `L_row·γ + addr_row + r`    |
/// | 2   | col_table   | `ts_col`   | `mem_col·γ + id + r`        |
/// | 3   | col_access  | `-1`       | `L_col·γ + addr_col + r`    |
///
/// `id[i] = i` is the cell address. All columns must share one length `N`, a
/// power of two (`debug_assert`ed); the returned layers each have `log N`
/// variables, the shared GKR depth.
pub fn build_input_layers<E: Engine>(
  cols: &MemCheckWitness<'_, E>,
  gamma: E::Scalar,
  r: E::Scalar,
) -> Vec<Layer<E>> {
  let n = cols.mem_row.len();
  debug_assert!(
    n.is_power_of_two() && n >= 2,
    "N must be a power of two >= 2"
  );
  for col in [
    &cols.mem_col,
    &cols.L_row,
    &cols.L_col,
    &cols.addr_row,
    &cols.addr_col,
    &cols.ts_row,
    &cols.ts_col,
  ] {
    debug_assert_eq!(col.len(), n, "all memory-check columns must share length N");
  }

  let neg_one = -E::Scalar::ONE;
  // Table-side den: mem·γ + id + r, where id[i] = i (the cell address).
  //
  // `id[i]` is built by per-chunk accumulation instead of `Scalar::from(i)` per
  // element: each chunk pays ONE `from` for its base index, then walks its cells
  // with a field `+ ONE`, which is far cheaper than a u64→Montgomery conversion.
  // The chunks run in parallel (`par_chunks_mut`), so this keeps full width
  // while dropping N `from`s to `N / chunk_size`.
  let one = E::Scalar::ONE;
  let chunk_size = 1 + n / rayon::current_num_threads().max(1);
  let den_table = |mem: &[E::Scalar]| -> Vec<E::Scalar> {
    let mut out = vec![E::Scalar::ZERO; n];
    out
      .par_chunks_mut(chunk_size)
      .enumerate()
      .for_each(|(c, chunk)| {
        let mut id = E::Scalar::from((c * chunk_size) as u64); // base index of this chunk
        for (out_i, mem_i) in chunk.iter_mut().zip(&mem[c * chunk_size..]) {
          *out_i = *mem_i * gamma + id + r;
          id += one;
        }
      });
    out
  };
  // Access-side den: L·γ + addr + r.
  let den_access = |l: &[E::Scalar], addr: &[E::Scalar]| -> Vec<E::Scalar> {
    (0..n)
      .into_par_iter()
      .map(|i| l[i] * gamma + addr[i] + r)
      .collect()
  };
  let mle = |v: Vec<E::Scalar>| MultilinearPolynomial::new(v);

  vec![
    // idx 0: row_table
    Layer::<E> {
      num: mle(cols.ts_row.to_vec()),
      den: mle(den_table(cols.mem_row)),
    },
    // idx 1: row_access
    Layer::<E> {
      num: mle(vec![neg_one; n]),
      den: mle(den_access(cols.L_row, cols.addr_row)),
    },
    // idx 2: col_table
    Layer::<E> {
      num: mle(cols.ts_col.to_vec()),
      den: mle(den_table(cols.mem_col)),
    },
    // idx 3: col_access
    Layer::<E> {
      num: mle(vec![neg_one; n]),
      den: mle(den_access(cols.L_col, cols.addr_col)),
    },
  ]
}

/// Reconciles the four input-layer fractions reconstructed from the claimed
/// columns at `eval_point` against the GKR-reduced `reduced` fractions
/// (component-wise, exact), then checks the two relation balances on the `roots`
/// with an explicit denominator-nonzero guard. Shared by the standalone
/// ppSNARK fused path (PLAN section 6.2).
pub(crate) fn reconcile_and_balance<E: Engine>(
  reduced: &[Fraction<E::Scalar>],
  roots: &[Fraction<E::Scalar>],
  eval_point: &[E::Scalar],
  gamma: E::Scalar,
  r: E::Scalar,
  r_outer_full: &[E::Scalar],
  openings: &MemCheckOpenings<E>,
) -> Result<(), NovaError> {
  if reduced.len() != NUM_SUB_INSTANCES {
    return Err(NovaError::InvalidNumInstances);
  }
  // Bind GKR depth to the trusted `log N`; otherwise `EqPolynomial::evaluate`
  // would panic on a forged depth rather than return `NovaError`.
  if eval_point.len() != r_outer_full.len() {
    return Err(NovaError::InvalidSumcheckProof);
  }

  // (2) Recompute the four input-layer fractions from the claimed columns.
  let eval_id = IdentityPolynomial::<E::Scalar>::new(eval_point.len()).evaluate(eval_point);
  let eval_mem_row = EqPolynomial::new(r_outer_full.to_vec()).evaluate(eval_point);
  let neg_one = -E::Scalar::ONE;

  let row_table = Fraction::new(openings.eval_ts_row, eval_mem_row * gamma + eval_id + r);
  let row_access = Fraction::new(neg_one, openings.eval_L_row * gamma + openings.eval_row + r);
  let col_table = Fraction::new(
    openings.eval_ts_col,
    openings.eval_mem_col * gamma + eval_id + r,
  );
  let col_access = Fraction::new(neg_one, openings.eval_L_col * gamma + openings.eval_col + r);
  let recomputed = [row_table, row_access, col_table, col_access];

  // Each recomputed fraction must match the GKR-reduced claim component-wise.
  for (rc, red) in recomputed.iter().zip(reduced.iter()) {
    if rc.num != red.num || rc.den != red.den {
      return Err(NovaError::InvalidSumcheckProof);
    }
  }

  // (3) Balance on the root fractions (the whole-relation sums).
  let [row_table_root, row_access_root, col_table_root, col_access_root] = roots[..] else {
    return Err(NovaError::InvalidNumInstances);
  };
  let all_dens_nonzero = [
    row_table_root,
    row_access_root,
    col_table_root,
    col_access_root,
  ]
  .iter()
  .all(|f| f.den != E::Scalar::ZERO);
  if !all_dens_nonzero {
    return Err(NovaError::InvalidSumcheckProof);
  }
  let row_balanced = (row_table_root + row_access_root).num == E::Scalar::ZERO;
  let col_balanced = (col_table_root + col_access_root).num == E::Scalar::ZERO;
  if !row_balanced || !col_balanced {
    return Err(NovaError::InvalidSumcheckProof);
  }
  Ok(())
}

/// The fused memory-check proof carried in the ppSNARK proof on the default
/// path: the GKR prefix plus the fused last-layer + Inner sumcheck proof. GKR
/// and Inner now land on one shared `r_shared`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FusedMemCheckProof<E: Engine> {
  /// The GKR prefix (every layer reduction except the last).
  pub prefix: LogupGkrPrefixProof<E>,
  /// The fused GKR-last-layer + Inner sumcheck proof.
  pub fused: FusedGkrInnerProof<E>,
}

/// Fused-path prover: build the four GKR input layers, run the prefix, then the
/// fused last-layer + Inner sumcheck. Returns the proof, `r_shared`, and the
/// Inner endpoint evaluations for the host's PCS opening.
pub fn prove_step_fused<E: Engine>(
  witness: MemCheckWitness<'_, E>,
  gamma: E::Scalar,
  r: E::Scalar,
  inner: FusedInnerInputs<E>,
  transcript: &mut E::TE,
) -> Result<(FusedMemCheckProof<E>, Vec<E::Scalar>, FusedEndpointEvals<E>), NovaError> {
  let layers = build_input_layers(&witness, gamma, r);
  let prefix = crate::spartan::logup_gkr::prover::prove_prefix::<E>(layers, transcript)?;
  let prefix_proof = LogupGkrPrefixProof {
    initial_claims: prefix.initial_claims.clone(),
    prefix_final_claims: prefix.final_claims_by_layer.clone(),
    prefix_sumchecks: prefix.sumchecks.clone(),
  };
  let (fused, r_shared, endpoints) = prove_fused::<E>(prefix, inner, transcript)?;
  Ok((
    FusedMemCheckProof {
      prefix: prefix_proof,
      fused,
    },
    r_shared,
    endpoints,
  ))
}

/// Read-only context for the fused verifier: the memory-check fingerprint, the
/// outer evaluation point, the public IO, and the trusted padding/masking
/// dimensions. Every field is derived from the verifier key or already-bound
/// transcript values, never from the proof.
pub struct FusedVerifyContext<'a, E: Engine> {
  /// Memory-check fingerprint challenge `gamma`.
  pub gamma: E::Scalar,
  /// Memory-check fingerprint challenge `r`.
  pub r: E::Scalar,
  /// The extended outer point `q = r_outer_full` (length `log N`).
  pub r_outer_full: &'a [E::Scalar],
  /// Public IO `[u, X...]`, used to reconstruct `mem_col`.
  pub public_io: &'a [E::Scalar],
  /// `log2(num_vars)` for the masked-eq witness factor.
  pub num_masked_vars: usize,
  /// `log2(N)` — the trusted inner-sumcheck length.
  pub snark_n_log: usize,
  /// `log2(2·num_vars)` — the unpadded public-IO boundary.
  pub two_num_vars_log: usize,
}

/// The Inner-relation initial claims (PLAN section 5) folded into the fused
/// sumcheck.
pub struct FusedInnerClaims<E: Engine> {
  /// `C_A = factor·(eval_Az + c·eval_Bz + c²·eval_Cz)`.
  pub abc: E::Scalar,
  /// `C_E = factor·eval_E_at_r_outer`.
  pub e: E::Scalar,
  /// `C_W` — the witness-bound relation claim (zero in ppSNARK).
  pub w: E::Scalar,
}

/// The column evaluations claimed at `r_shared`, checked by the fused Inner
/// endpoint (PLAN section 6.1) and the GKR reconcile (section 6.2), and bound to
/// the committed columns by the batched PCS opening. `W` doubles as the value
/// used to reconstruct `mem_col`.
#[allow(non_snake_case)]
pub struct FusedEndpointClaims<E: Engine> {
  /// `val(r_shared) = val_A + c·val_B + c²·val_C`.
  pub val: E::Scalar,
  /// `E(r_shared)`.
  pub E: E::Scalar,
  /// `W(r_shared)` — also reconstructs `mem_col`.
  pub W: E::Scalar,
  /// `L_row(r_shared)`.
  pub L_row: E::Scalar,
  /// `L_col(r_shared)`.
  pub L_col: E::Scalar,
  /// `row(r_shared)` — the row access address.
  pub row: E::Scalar,
  /// `col(r_shared)` — the col access address.
  pub col: E::Scalar,
  /// `ts_row(r_shared)`.
  pub ts_row: E::Scalar,
  /// `ts_col(r_shared)`.
  pub ts_col: E::Scalar,
}

/// Fused-path verifier: replay the prefix, verify the fused sumcheck, check the
/// Inner endpoint `h(f) == inner_expected` independently, then reconcile the four
/// input-layer fractions and check balance (PLAN section 6). Returns `r_shared`.
///
/// `endpoints` are the column evaluations at `r_shared`; `mem_col` is
/// reconstructed internally from `endpoints.W` and the public IO once the fused
/// verifier has derived `r_shared`.
pub fn verify_step_fused<E: Engine>(
  proof: &FusedMemCheckProof<E>,
  ctx: &FusedVerifyContext<'_, E>,
  claims: &FusedInnerClaims<E>,
  endpoints: &FusedEndpointClaims<E>,
  transcript: &mut E::TE,
) -> Result<Vec<E::Scalar>, NovaError> {
  let prefix_state = verify_gkr_prefix::<E>(
    proof.prefix.initial_claims.clone(),
    proof.prefix.prefix_final_claims.clone(),
    proof.prefix.prefix_sumchecks.clone(),
    transcript,
  )?;
  let out = verify_fused::<E>(
    &proof.fused,
    prefix_state,
    claims.abc,
    claims.e,
    claims.w,
    transcript,
  )?;
  let r_shared = out.r_shared;

  // Bind the GKR depth to the trusted inner-sumcheck length before any
  // `EqPolynomial`/`MaskedEqPolynomial::evaluate` below: a forged proof whose
  // GKR depth differs from `log N` yields an `r_shared` of the wrong length,
  // and those evaluators `assert_eq!` on the point length (would panic instead
  // of returning `Err`). `reconcile_and_balance` re-checks this, but only after
  // the Inner endpoint evaluations, so the guard must lead here.
  if r_shared.len() != ctx.r_outer_full.len() {
    return Err(NovaError::InvalidSumcheckProof);
  }

  // Inner endpoint (checked independently of the GKR reconcile, PLAN section 6.1):
  //   h(f) == beta·L_row·L_col·val + beta^2·eq(q,r_shared)·E + beta^3·masked·W
  let eq_r_outer = EqPolynomial::new(ctx.r_outer_full.to_vec());
  let eq_q_at_r = eq_r_outer.evaluate(&r_shared);
  let masked = MaskedEqPolynomial::new(&eq_r_outer, ctx.num_masked_vars).evaluate(&r_shared);
  let beta = out.beta;
  let beta2 = beta * beta;
  let beta3 = beta2 * beta;
  let inner_expected = beta * endpoints.L_row * endpoints.L_col * endpoints.val
    + beta2 * eq_q_at_r * endpoints.E
    + beta3 * masked * endpoints.W;
  if out.e_inner != inner_expected {
    return Err(NovaError::InvalidSumcheckProof);
  }

  let l = ctx
    .snark_n_log
    .checked_sub(ctx.two_num_vars_log)
    .ok_or(NovaError::InvalidSumcheckProof)?;
  if l >= r_shared.len() {
    return Err(NovaError::InvalidSumcheckProof);
  }
  let mut mc_factor = E::Scalar::ONE;
  for r_p in r_shared.iter().take(l) {
    mc_factor *= E::Scalar::ONE - *r_p;
  }
  let r_unpad = r_shared[l..].to_vec();
  let eval_X = {
    let poly_X = SparsePolynomial::new(r_unpad.len() - 1, ctx.public_io.to_vec());
    poly_X.evaluate(&r_unpad[1..])
  };
  let eval_mem_col = endpoints.W + mc_factor * r_unpad[0] * eval_X;
  let openings = MemCheckOpenings::<E> {
    eval_L_row: endpoints.L_row,
    eval_L_col: endpoints.L_col,
    eval_row: endpoints.row,
    eval_col: endpoints.col,
    eval_ts_row: endpoints.ts_row,
    eval_ts_col: endpoints.ts_col,
    eval_mem_col,
  };

  // GKR reconcile + balance at r_shared (checked separately from the endpoint).
  reconcile_and_balance::<E>(
    &out.gkr_fractions,
    &proof.prefix.initial_claims,
    &r_shared,
    ctx.gamma,
    ctx.r,
    ctx.r_outer_full,
    &openings,
  )?;

  Ok(r_shared)
}

#[cfg(all(test, not(feature = "logup-no-gkr")))]
mod tests {
  use super::*;
  use crate::spartan::logup_gkr::proof::{LayerFinalClaim, LogupGkrPrefixProof};
  use crate::spartan::mem_check_logup_gkr_fused::FusedGkrInnerProof;
  use crate::spartan::polys::univariate::UniPoly;
  use crate::traits::{Engine, TranscriptEngineTrait};

  type E = crate::provider::Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  // Regression for the deleted `rejects_wrong_gkr_depth`: a proof whose GKR depth
  // (here `n = 1`) differs from the trusted `log N` must be rejected with `Err`,
  // NOT panic inside `EqPolynomial::evaluate` (whose `assert_eq!` on the point
  // length would otherwise fire before `reconcile_and_balance`'s own guard).
  #[test]
  fn verify_step_fused_rejects_wrong_gkr_depth() {
    let zero = Fr::ZERO;
    let root = Fraction::new(zero, Fr::ONE);
    // `(0,1)+(0,1)` gates to `(0,1)`, so the `n = 1` root check passes and
    // `verify_fused` returns an `r_shared` of length 1.
    let split = LayerFinalClaim {
      left: root,
      right: root,
    };
    let proof = FusedMemCheckProof {
      prefix: LogupGkrPrefixProof {
        initial_claims: vec![root; NUM_SUB_INSTANCES],
        prefix_final_claims: vec![],
        prefix_sumchecks: vec![],
      },
      fused: FusedGkrInnerProof {
        suffix_round_polys: vec![],
        input_splits: [split; NUM_SUB_INSTANCES],
        msb_round_poly: UniPoly::from_evals_deg3(&[zero, zero, zero, zero]).compress(),
      },
    };
    // Trusted `log N = 2` (so `r_outer_full` has length 2) while the forged proof
    // reduces to depth 1.
    let mut transcript = <E as Engine>::TE::new(b"depth-regression");
    let ctx = FusedVerifyContext::<E> {
      gamma: zero,
      r: zero,
      r_outer_full: &[zero, zero],
      public_io: &[zero],
      num_masked_vars: 1,
      snark_n_log: 2,
      two_num_vars_log: 1,
    };
    let claims = FusedInnerClaims::<E> {
      abc: zero,
      e: zero,
      w: zero,
    };
    let endpoints = FusedEndpointClaims::<E> {
      val: zero,
      E: zero,
      W: zero,
      L_row: zero,
      L_col: zero,
      row: zero,
      col: zero,
      ts_row: zero,
      ts_col: zero,
    };
    let verdict = verify_step_fused::<E>(&proof, &ctx, &claims, &endpoints, &mut transcript);
    assert!(
      matches!(verdict, Err(NovaError::InvalidSumcheckProof)),
      "forged GKR depth must be rejected with Err, not panic"
    );
  }

  // Recomputes the four input-layer fractions from the claimed columns exactly as
  // `reconcile_and_balance` does internally, so a test can build a self-consistent
  // "honest" `reduced` set and then perturb `roots` to exercise each guard.
  fn recompute(
    o: &MemCheckOpenings<E>,
    eval_point: &[Fr],
    r_outer_full: &[Fr],
    gamma: Fr,
    r: Fr,
  ) -> [Fraction<Fr>; NUM_SUB_INSTANCES] {
    let eval_id = IdentityPolynomial::<Fr>::new(eval_point.len()).evaluate(eval_point);
    let eval_mem_row = EqPolynomial::new(r_outer_full.to_vec()).evaluate(eval_point);
    let neg_one = -Fr::ONE;
    [
      Fraction::new(o.eval_ts_row, eval_mem_row * gamma + eval_id + r),
      Fraction::new(neg_one, o.eval_L_row * gamma + o.eval_row + r),
      Fraction::new(o.eval_ts_col, o.eval_mem_col * gamma + eval_id + r),
      Fraction::new(neg_one, o.eval_L_col * gamma + o.eval_col + r),
    ]
  }

  fn reconcile_fixture() -> (MemCheckOpenings<E>, Vec<Fr>, Vec<Fr>, Fr, Fr) {
    let openings = MemCheckOpenings::<E> {
      eval_L_row: Fr::from(3),
      eval_L_col: Fr::from(6),
      eval_row: Fr::from(1),
      eval_col: Fr::from(2),
      eval_ts_row: Fr::from(2),
      eval_ts_col: Fr::from(5),
      eval_mem_col: Fr::from(9),
    };
    (
      openings,
      vec![Fr::from(3), Fr::from(5)], // eval_point (log N = 2)
      vec![Fr::from(7), Fr::from(2)], // r_outer_full (same length)
      Fr::from(11),                   // gamma
      Fr::from(4),                    // r
    )
  }

  // Balanced roots: `(1,1)+(-1,1)` has numerator `0` per relation, dens nonzero.
  fn balanced_roots() -> [Fraction<Fr>; NUM_SUB_INSTANCES] {
    let one = Fraction::new(Fr::ONE, Fr::ONE);
    let neg = Fraction::new(-Fr::ONE, Fr::ONE);
    [one, neg, one, neg]
  }

  #[test]
  fn reconcile_accepts_consistent_and_balanced() {
    let (o, eval_point, r_outer_full, gamma, r) = reconcile_fixture();
    let reduced = recompute(&o, &eval_point, &r_outer_full, gamma, r);
    assert!(reconcile_and_balance::<E>(
      &reduced,
      &balanced_roots(),
      &eval_point,
      gamma,
      r,
      &r_outer_full,
      &o,
    )
    .is_ok());
  }

  #[test]
  fn reconcile_rejects_mismatched_column() {
    let (o, eval_point, r_outer_full, gamma, r) = reconcile_fixture();
    let mut reduced = recompute(&o, &eval_point, &r_outer_full, gamma, r);
    reduced[0].num += Fr::ONE; // GKR-reduced fraction disagrees with the column claim
    assert!(reconcile_and_balance::<E>(
      &reduced,
      &balanced_roots(),
      &eval_point,
      gamma,
      r,
      &r_outer_full,
      &o,
    )
    .is_err());
  }

  #[test]
  fn reconcile_rejects_unbalanced_roots() {
    let (o, eval_point, r_outer_full, gamma, r) = reconcile_fixture();
    let reduced = recompute(&o, &eval_point, &r_outer_full, gamma, r);
    // Reconcile passes, but the row pair no longer sums to zero: `(1,1)+(1,1)`.
    let one = Fraction::new(Fr::ONE, Fr::ONE);
    let neg = Fraction::new(-Fr::ONE, Fr::ONE);
    let roots = [one, one, one, neg];
    assert!(
      reconcile_and_balance::<E>(&reduced, &roots, &eval_point, gamma, r, &r_outer_full, &o,)
        .is_err()
    );
  }

  #[test]
  fn reconcile_rejects_zero_root_denominator() {
    let (o, eval_point, r_outer_full, gamma, r) = reconcile_fixture();
    let reduced = recompute(&o, &eval_point, &r_outer_full, gamma, r);
    // A `0/0` root would make the `num == 0` balance vacuously pass, so the
    // explicit den != 0 guard must reject it first.
    let zero_zero = Fraction::new(Fr::ZERO, Fr::ZERO);
    let neg = Fraction::new(-Fr::ONE, Fr::ONE);
    let roots = [zero_zero, neg, Fraction::new(Fr::ONE, Fr::ONE), neg];
    assert!(
      reconcile_and_balance::<E>(&reduced, &roots, &eval_point, gamma, r, &r_outer_full, &o,)
        .is_err()
    );
  }
}
