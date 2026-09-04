//! Bridge layer wiring Logup-GKR to ppSNARK's memory-check.
//!
//! This module is the *host* half of the Logup-GKR memory-check. The
//! `logup_gkr` core owns the fractional-sum reduction but no PCS; `ppsnark` owns
//! the commitments and inner sumcheck. This bridge supplies ppSNARK's four
//! equal-height sub-instances and rerandomizes all seven columns needed for
//! reconcile into the inner sumcheck.
//!
//! The verifier closes soundness by (1) reconstructing each input-layer
//! fraction from the seven column evaluations claimed at the GKR `eval_point`,
//! (2) comparing those fractions with the GKR reduction, and (3) checking the
//! two fractional balances. [`MemCheckOpenings`] fixes the claimed columns and
//! their order; the rerandomize sumcheck and batched PCS opening bind those
//! claims to the committed columns at `r_inner_batched`.
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
use crate::spartan::logup_gkr::proof::LogupGkrProof;
use crate::spartan::logup_gkr::verifier;
use crate::spartan::polys::eq::EqPolynomial;
use crate::spartan::polys::identity::IdentityPolynomial;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::spartan::sumcheck::eq_sumcheck::EqSumCheckInstance;
use crate::spartan::sumcheck::{SumcheckEngine, SumcheckProof};
use crate::traits::evm_serde::EvmCompatSerde;
use crate::traits::{Engine, TranscriptEngineTrait};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_with::serde_as;

/// The Logup-GKR memory-check proof fields carried in the ppSNARK proof: the
/// fractional-sum proof plus the prover-claimed column values at the GKR
/// `eval_point` (the rerandomize instance's initial claims, in
/// [`MemCheckOpenings::rerand_claims`] order). Bundled so the SNARK can gate the
/// whole GKR path behind one field.
#[serde_as]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct GkrProofData<E: Engine> {
  /// The GKR fractional-sum proof.
  pub proof: LogupGkrProof<E>,
  /// Prover-claimed column values at the GKR `eval_point`, in
  /// [`MemCheckOpenings::rerand_claims`] order. The verifier reads these;
  /// reconcile + the inner sumcheck bind them to the real committed columns.
  #[serde_as(as = "[EvmCompatSerde; NUM_RERAND_COLUMNS]")]
  pub rerand_claims: [E::Scalar; NUM_RERAND_COLUMNS],
}

/// Fixed sub-instance count (`row_table, row_access, col_table, col_access`).
pub const NUM_SUB_INSTANCES: usize = 4;

/// Number of columns the rerandomize instance carries from the GKR `eval_point`
/// into the inner sumcheck. These are every column reconcile needs at the GKR
/// point that the verifier cannot self-compute (it computes only `mem_row = eq`
/// and the identity `id`): `L_row, L_col, addr_row, addr_col, ts_row, ts_col,
/// mem_col`. The order is fixed by [`MemCheckOpenings::rerand_claims`] and must
/// match between prover and verifier.
pub const NUM_RERAND_COLUMNS: usize = 7;

/// The column evaluations a prover claims at the GKR
/// [`eval_point`](crate::spartan::logup_gkr::LogupGkrOpeningClaim::eval_point).
///
/// Every field is one column evaluated at the shared GKR point. The host uses
/// them to reconstruct the four sub-instance fractions, while the rerandomize
/// sumcheck carries the same claims to `r_inner_batched` for the batched PCS
/// opening. The verifier computes `id` and `mem_row = eq(r_outer_full, ·)`
/// directly, so neither needs a claim here.
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

impl<E: Engine> MemCheckOpenings<E> {
  /// The claimed column values at the GKR `eval_point`, in the fixed
  /// [`NUM_RERAND_COLUMNS`] order the rerandomize instance and the verifier both
  /// use: `[L_row, L_col, addr_row, addr_col, ts_row, ts_col, mem_col]`.
  pub fn rerand_claims(&self) -> [E::Scalar; NUM_RERAND_COLUMNS] {
    [
      self.eval_L_row,
      self.eval_L_col,
      self.eval_row,
      self.eval_col,
      self.eval_ts_row,
      self.eval_ts_col,
      self.eval_mem_col,
    ]
  }
}

/// The prover's memory-check witness: the raw length-N columns fed into
/// Logup-GKR.
///
/// These are exactly ppSNARK's padded memory-check columns (all length
/// `N = 2^{log N}`). [`build_input_layers`]
/// turns them into the four GKR input layers, and [`prove`] consumes the
/// witness while claiming the seven evaluations named by [`MemCheckOpenings`]
/// at the shared point. Field meanings match the four-sub-instance table in the
/// module docs:
/// - `mem_row = eq(r_outer_full, ·)`, `mem_col = z` (table values);
/// - `L_row`/`L_col` the lookup columns, `addr_row = row`/`addr_col = col` the
///   access addresses, `ts_row`/`ts_col` the multiplicities.
#[derive(Clone)]
pub struct MemCheckWitness<E: Engine> {
  /// Row table values `mem_row = eq(r_outer_full, ·)`.
  pub mem_row: Vec<E::Scalar>,
  /// Col table values `mem_col = z`.
  pub mem_col: Vec<E::Scalar>,
  /// Row lookup column `L_row`.
  pub L_row: Vec<E::Scalar>,
  /// Col lookup column `L_col`.
  pub L_col: Vec<E::Scalar>,
  /// Row access addresses `addr_row = row`.
  pub addr_row: Vec<E::Scalar>,
  /// Col access addresses `addr_col = col`.
  pub addr_col: Vec<E::Scalar>,
  /// Row multiplicities `ts_row`.
  pub ts_row: Vec<E::Scalar>,
  /// Col multiplicities `ts_col`.
  pub ts_col: Vec<E::Scalar>,
}

/// Builds the four GKR input layers `[row_table, row_access, col_table,
/// col_access]` from the raw columns and the fingerprint `(gamma, r)`.
///
/// This is the prover-side dual of the verifier's per-instance reconstruction
/// in [`verify`]: it must produce, for every leaf `i`, exactly the fractions the
/// verifier recomputes at `eval_point`. The layers, in order (matching the
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
  cols: &MemCheckWitness<E>,
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
      num: mle(cols.ts_row.clone()),
      den: mle(den_table(&cols.mem_row)),
    },
    // idx 1: row_access
    Layer::<E> {
      num: mle(vec![neg_one; n]),
      den: mle(den_access(&cols.L_row, &cols.addr_row)),
    },
    // idx 2: col_table
    Layer::<E> {
      num: mle(cols.ts_col.clone()),
      den: mle(den_table(&cols.mem_col)),
    },
    // idx 3: col_access
    Layer::<E> {
      num: mle(vec![neg_one; n]),
      den: mle(den_access(&cols.L_col, &cols.addr_col)),
    },
  ]
}

/// Verifies the ppSNARK memory-check via Logup-GKR.
///
/// Steps: (1) run the GKR verifier to get the shared `eval_point` and four
/// reduced input-layer fractions; (2) reconstruct those fractions from the
/// prover's claimed columns ([`MemCheckOpenings`]) and the fingerprint
/// `(gamma, r)`, and require they match; (3) check the two balances. The
/// returned `eval_point` seeds the rerandomize final-claim check in the inner
/// sumcheck. The transcript must be positioned immediately before the GKR
/// proof.
///
/// `r_outer_full` is ppSNARK's extended outer challenge, defining `mem_row =
/// eq(r_outer_full, ·)`; the verifier evaluates it at `eval_point` itself.
pub fn verify<E: Engine>(
  proof: &LogupGkrProof<E>,
  gamma: E::Scalar,
  r: E::Scalar,
  r_outer_full: &[E::Scalar],
  openings: &MemCheckOpenings<E>,
  transcript: &mut E::TE,
) -> Result<Vec<E::Scalar>, NovaError> {
  // (1) GKR verifier: shape-check, root gates, per-layer sumchecks.
  let claim = verifier::verify::<E>(proof, transcript)?;
  let eval_point = claim.eval_point();
  let reduced = claim.openings();
  if reduced.len() != NUM_SUB_INSTANCES {
    return Err(NovaError::InvalidNumInstances);
  }
  // Bind GKR depth to the trusted `log N` from vk / outer challenge. The proof
  // alone can claim any `final_claims.len()`; without this check a forged depth
  // would panic inside `EqPolynomial::evaluate` (assert on length mismatch)
  // rather than return `NovaError`.
  if eval_point.len() != r_outer_full.len() {
    return Err(NovaError::InvalidSumcheckProof);
  }

  // (2) Recompute the four input-layer fractions from the claimed columns.
  // Pieces the verifier evaluates itself at eval_point:
  let eval_id = IdentityPolynomial::<E::Scalar>::new(eval_point.len()).evaluate(eval_point);
  let eval_mem_row = EqPolynomial::new(r_outer_full.to_vec()).evaluate(eval_point);

  let neg_one = -E::Scalar::ONE;

  // idx 0: row_table    num = ts_row    den = mem_row·γ + id + r
  let row_table = {
    let num = openings.eval_ts_row;
    let den = eval_mem_row * gamma + eval_id + r;
    Fraction::new(num, den)
  };

  // idx 1: row_access   num = -1        den = L_row·γ + addr_row + r
  let row_access = {
    let num = neg_one;
    let den = openings.eval_L_row * gamma + openings.eval_row + r;
    Fraction::new(num, den)
  };

  // idx 2: col_table    num = ts_col    den = mem_col·γ + id + r
  let col_table = {
    let num = openings.eval_ts_col;
    let den = openings.eval_mem_col * gamma + eval_id + r;
    Fraction::new(num, den)
  };

  // idx 3: col_access   num = -1        den = L_col·γ + addr_col + r
  let col_access = {
    let num = neg_one;
    let den = openings.eval_L_col * gamma + openings.eval_col + r;
    Fraction::new(num, den)
  };

  let recomputed = [row_table, row_access, col_table, col_access];

  // Each recomputed fraction must match the GKR-reduced input-layer claim
  // **component-wise**. Cross-multiplication is unsound: a reduced `(0,0)`
  // would cross-equal any recomputed `(a,b)` and disconnect the GKR root from
  // the committed columns (see logup_gkr verifier root-gate comment).
  for (rc, red) in recomputed.iter().zip(reduced.iter()) {
    if rc.num != red.num || rc.den != red.den {
      return Err(NovaError::InvalidSumcheckProof);
    }
  }

  // (3) Balance: table side + access side = 0, for the row relation and the col
  // relation. Balance is a property of the whole relation, i.e. the *root*
  // fractions (`Σ ts/(T+r)` and `Σ -1/(W+r)`) — these are the GKR proof's
  // `initial_claims`, NOT the input-layer `recomputed` fractions from step (2)
  // (those are single-point evaluations at eval_point, a different quantity).
  // The roots are already bound to the reduction by the GKR verifier's root-gate
  // + per-layer sumcheck checks in step (1). The access side carries num = -1,
  // so `root_table + root_access = Σ ts/(T+r) − Σ 1/(W+r)`, which must vanish:
  // in projective form the sum's numerator is zero once its denominator (a
  // product of dens) is confirmed nonzero — checked explicitly just below.
  let [row_table_root, row_access_root, col_table_root, col_access_root] = proof.initial_claims[..]
  else {
    return Err(NovaError::InvalidNumInstances);
  };

  // Guard against a spurious `0/0` balance: `(t+a).num == 0` only certifies the
  // rational `t + a` is zero when its denominator `t.den · a.den` is nonzero.
  // Each root den is `Π_i (fingerprint_i + r)` with `r` a Fiat-Shamir challenge
  // drawn after the prover fixed its columns, so a zero factor happens with
  // probability ≤ N/|F| (negligible) for an honest prover and cannot be forced
  // by a malicious one — but the check is 4 comparisons and closes the path.
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

  Ok(eval_point.to_vec())
}

/// First rerandomize coeff index in the batched inner sumcheck. `prove_helper`
/// places the memory-check slot first, so under Logup-GKR the seven rerandomize
/// columns occupy coeffs `[0, NUM_RERAND_COLUMNS)`, and the inner (ABC, E) and
/// witness claims follow at `NUM_RERAND_COLUMNS ..`.
pub const RERAND_BASE: usize = 0;

/// Number of batched-inner claims the memory-check slot contributes (the seven
/// rerandomize columns). The inner/witness claims come after these.
pub const NUM_MEM_CLAIMS: usize = NUM_RERAND_COLUMNS;

/// Prover side of the Logup-GKR memory-check: folds the four sub-instances into
/// GKR trees, absorbs the claimed column values, and returns the rerandomize
/// instance (the first `prove_helper` slot) and the proof data to store (which
/// itself carries the claimed column values in `rerand_claims`).
pub fn prove_step<E: Engine>(
  witness: MemCheckWitness<E>,
  gamma: E::Scalar,
  r: E::Scalar,
  transcript: &mut E::TE,
) -> Result<(RerandomizeSumcheckInstance<E>, GkrProofData<E>), NovaError> {
  let out = prove::<E>(witness, gamma, r, transcript)?;
  let rerand_claims = out.openings.rerand_claims();
  // Absorb the claimed column values before the inner sumcheck's `s` so the
  // verifier binds the same values in the same order.
  transcript.absorb(b"gkrL", &rerand_claims.as_slice());
  let data = GkrProofData {
    proof: out.proof,
    rerand_claims,
  };
  Ok((out.rerandomize, data))
}

/// Verifier side, transcript phase: replays the GKR proof, reconciles the
/// claimed columns against the reduction, runs the balance check, and absorbs
/// the claimed values — mirroring [`prove_step`]. Returns the shared GKR
/// `eval_point`. Must run before the inner sumcheck's `s` challenge is drawn.
pub fn verify_pre_inner<E: Engine>(
  data: &GkrProofData<E>,
  gamma: E::Scalar,
  r: E::Scalar,
  r_outer_full: &[E::Scalar],
  transcript: &mut E::TE,
) -> Result<Vec<E::Scalar>, NovaError> {
  let c = data.rerand_claims;
  let openings = MemCheckOpenings::<E> {
    eval_L_row: c[0],
    eval_L_col: c[1],
    eval_row: c[2],
    eval_col: c[3],
    eval_ts_row: c[4],
    eval_ts_col: c[5],
    eval_mem_col: c[6],
  };
  let eval_point = verify::<E>(&data.proof, gamma, r, r_outer_full, &openings, transcript)?;
  transcript.absorb(b"gkrL", &data.rerand_claims.as_slice());
  Ok(eval_point)
}

/// The Logup-GKR contribution to the batched inner sumcheck's **initial** claim:
/// `Σ_i coeffs[RERAND_BASE + i] · claimed_column_i(eval_point)`.
pub fn verify_initial_claim<E: Engine>(data: &GkrProofData<E>, coeffs: &[E::Scalar]) -> E::Scalar {
  (0..NUM_RERAND_COLUMNS)
    .map(|i| coeffs[RERAND_BASE + i] * data.rerand_claims[i])
    .sum()
}

/// The Logup-GKR contribution to the batched inner sumcheck's **final** claim:
/// `Σ_i coeffs[RERAND_BASE + i] · eq(eval_point, r_inner_batched) ·
/// column_i(r_inner_batched)`, mirroring the E claim. `rerand_col_evals` are the
/// seven columns' values at `r_inner_batched` in RERAND order.
///
/// `eval_point` and `r_inner_batched` must share length `log N` (already enforced
/// when [`verify`] / [`verify_pre_inner`] returned `eval_point`, and by the
/// inner sumcheck round count). Mismatch returns [`NovaError`] instead of
/// panicking in `EqPolynomial::evaluate`.
pub fn verify_final_claim<E: Engine>(
  coeffs: &[E::Scalar],
  eval_point: &[E::Scalar],
  r_inner_batched: &[E::Scalar],
  rerand_col_evals: &[E::Scalar; NUM_RERAND_COLUMNS],
) -> Result<E::Scalar, NovaError> {
  if eval_point.len() != r_inner_batched.len() {
    return Err(NovaError::InvalidSumcheckProof);
  }
  let eq_gkr = EqPolynomial::new(eval_point.to_vec()).evaluate(r_inner_batched);
  Ok(
    (0..NUM_RERAND_COLUMNS)
      .map(|i| coeffs[RERAND_BASE + i] * eq_gkr * rerand_col_evals[i])
      .sum(),
  )
}

/// Prover output for the Logup-GKR memory-check.
///
/// Bundles the three values consumed by ppSNARK integration:
/// - `proof`: the GKR proof, absorbed into the SNARK and replayed by [`verify`];
/// - `openings`: the seven column evaluations at the GKR `eval_point` that the
///   host reconcile step checks;
/// - `rerandomize`: the [`RerandomizeSumcheckInstance`] that carries those seven
///   claims into the inner sumcheck and binds them at `r_inner_batched`.
pub struct MemCheckProverOutput<E: Engine> {
  /// The GKR fractional-sum proof.
  pub proof: LogupGkrProof<E>,
  /// Column evaluations at the GKR `eval_point` (host reconcile input).
  pub openings: MemCheckOpenings<E>,
  /// Seven-column opening-point reduction into the inner sumcheck.
  pub rerandomize: RerandomizeSumcheckInstance<E>,
  /// The shared GKR evaluation point (length `log N`).
  pub eval_point: Vec<E::Scalar>,
}

/// Proves the ppSNARK memory-check via Logup-GKR from the raw columns.
///
/// This is the prover-side entry point mirroring [`verify`]. It:
/// 1. builds the four GKR input layers ([`build_input_layers`]);
/// 2. runs the GKR prover to fold them and emit the proof plus the shared
///    `eval_point`;
/// 3. evaluates the seven claimed columns at `eval_point` to form
///    [`MemCheckOpenings`];
/// 4. builds the [`RerandomizeSumcheckInstance`] that carries those claims into
///    the inner sumcheck.
///
/// The transcript must be in the same state the verifier expects at the GKR
/// slot (the GKR prover absorbs exactly what [`verify`] replays). `(gamma, r)`
/// are ppSNARK's memory-check fingerprint challenges.
pub fn prove<E: Engine>(
  witness: MemCheckWitness<E>,
  gamma: E::Scalar,
  r: E::Scalar,
  transcript: &mut E::TE,
) -> Result<MemCheckProverOutput<E>, NovaError> {
  // (1)+(2) Build the four input layers and fold them through the GKR trees.
  let layers = build_input_layers(&witness, gamma, r);
  let (proof, claim) = crate::spartan::logup_gkr::prover::prove::<E>(layers, transcript)?;
  let eval_point = claim.eval_point().to_vec();

  // (3) Assemble the opened columns at the shared point. The prover is honest
  // and owns every column, so each opening is taken by the cheapest route that
  // yields the same value:
  // - `ts_row`/`ts_col` ARE the table-side numerators the GKR reduction already
  //   produced (openings 0, 2), so reuse them directly;
  // - `addr_row`/`addr_col`/`mem_col` are fused into the GKR dens
  //   (`L·γ + addr + r`, `mem·γ + id + r`), so invert those closed forms instead
  //   of re-evaluating the MLEs — one field op vs an N-wide evaluation;
  // - `L_row`/`L_col` must be evaluated directly (they seed the rerandomize
  //   claims and are the other unknown in the access dens), so they stay `ev`.
  // This is a pure prover-side shortcut: soundness lives in the verifier, which
  // opens each column against its own commitment (see `verify`).
  let ev = |v: &[E::Scalar]| MultilinearPolynomial::evaluate_with(v, &eval_point);
  let [row_table, row_access, col_table, col_access] = claim.openings()[..] else {
    return Err(NovaError::InvalidNumInstances);
  };
  let eval_L_row = ev(&witness.L_row);
  let eval_L_col = ev(&witness.L_col);
  let eval_id = IdentityPolynomial::<E::Scalar>::new(eval_point.len()).evaluate(&eval_point);
  let gamma_inv = gamma.invert().expect("fingerprint gamma is nonzero");
  let openings = MemCheckOpenings {
    eval_L_row,
    eval_L_col,
    // row_access.den = L_row·γ + addr_row + r  ⇒  addr_row = den − L_row·γ − r
    eval_row: row_access.den - eval_L_row * gamma - r,
    // col_access.den = L_col·γ + addr_col + r  ⇒  addr_col = den − L_col·γ − r
    eval_col: col_access.den - eval_L_col * gamma - r,
    eval_ts_row: row_table.num, // row_table numerator
    eval_ts_col: col_table.num, // col_table numerator
    // col_table.den = mem_col·γ + id + r  ⇒  mem_col = (den − id − r)·γ⁻¹
    eval_mem_col: (col_table.den - eval_id - r) * gamma_inv,
  };

  // (4) Rerandomize instance: carry every column reconcile needs from the GKR
  // eval_point into the inner sumcheck. Columns and claims share the fixed
  // RERAND order [L_row, L_col, addr_row, addr_col, ts_row, ts_col, mem_col].
  // The witness is consumed here, so its columns are moved in (no clone).
  let claims = openings.rerand_claims().to_vec();
  let columns = vec![
    witness.L_row,
    witness.L_col,
    witness.addr_row,
    witness.addr_col,
    witness.ts_row,
    witness.ts_col,
    witness.mem_col,
  ];
  let rerandomize = RerandomizeSumcheckInstance::new(eval_point.clone(), columns, claims);

  Ok(MemCheckProverOutput {
    proof,
    openings,
    rerandomize,
    eval_point,
  })
}

/// Rerandomizes the GKR verifier's per-column evaluation requests at the GKR
/// `eval_point` into the inner sumcheck, so every column reconcile needs is
/// opened at the shared inner point `r_inner_batched` instead of at
/// `eval_point`.
///
/// # Why several columns, not just L
/// The host reconcile ([`verify`]) checks the four GKR-reduced fractions at
/// `eval_point`. Each fraction's num/den is a fingerprint of several columns
/// (`ts`, `L`, `addr`, `mem_col`); the verifier can self-compute only `mem_row =
/// eq(r_outer_full, ·)` and the identity `id`. Every other column it needs at
/// `eval_point` must be carried there. So this instance rerandomizes **all** of
/// them (order fixed by [`MemCheckOpenings::rerand_claims`]): each column `X` becomes one
/// sumcheck `Σ_y eq(eval_point, y) · X(y)` over the same `y ∈ {0,1}^{log N}`
/// domain as the inner ABC/E sumcheck, folding into the shared `prove_helper`
/// bundle and landing at `r_inner_batched`. This is exactly ppSNARK's E-claim
/// mechanism (`Σ eq(r_outer, y) · E(y)`) applied to each column, all sharing one
/// `eq_sumcheck` because they use the same `eval_point`.
///
/// # Degree
/// Each summand `eq(eval_point, ·) · X(·)` is a product of two multilinears, so
/// the true round-polynomial degree is **2**. [`prove_helper`] hardcodes degree
/// 3 (it interpolates every instance with `from_evals_deg3` and asserts all
/// bundled instances share a degree), so [`Self::degree`] reports 3 and the
/// quadratic evals carry a zero cubic coefficient — identical to how the E-claim
/// rides in the degree-3 inner instance.
///
/// [`prove_helper`]: super::ppsnark
pub struct RerandomizeSumcheckInstance<E: Engine> {
  /// Transparent `eq(eval_point, ·)` factor, shared by all columns.
  eq_sumcheck: EqSumCheckInstance<E>,
  /// The columns being rerandomized, order [`MemCheckOpenings::rerand_claims`].
  polys: Vec<MultilinearPolynomial<E::Scalar>>,
  /// Running claim per column (BDDT, eprint 2025/1117 §6.2).
  running_claims: Vec<E::Scalar>,
  /// Saved `[p(0), 0, p(-1)]` per column, used by [`SumcheckEngine::bound`].
  saved_evals: Vec<[E::Scalar; 3]>,
  /// Set by [`SumcheckEngine::fuse_with_coeffs`]. Once the batch coefficients are
  /// known, the seven columns collapse into one random linear combination so the
  /// prover scans and binds a single polynomial per round instead of seven. The
  /// per-round output is still reported as seven triples (the combined triple in
  /// slot 0, zeros elsewhere), so the batched prover's positional
  /// `Σ coeffs[i]·evals[i]` stays byte-identical to the unfused per-column sum.
  fused: Option<FusedRerandomize<E>>,
}

/// Fused single-column state for [`RerandomizeSumcheckInstance`]. Holds the
/// coefficient-weighted combination of all columns and its running claim.
struct FusedRerandomize<E: Engine> {
  /// `Σ_i coeffs[i] · column_i`, bound in lockstep with `eq_sumcheck`.
  poly: MultilinearPolynomial<E::Scalar>,
  /// `Σ_i coeffs[i] · claim_i`, the running claim of the combined column.
  running_claim: E::Scalar,
  /// Saved `[p(0), 0, p(-1)]` of the combined column for [`SumcheckEngine::bound`].
  saved: [E::Scalar; 3],
}

impl<E: Engine> RerandomizeSumcheckInstance<E> {
  /// Builds the instance from the GKR `eval_point`, the columns (order
  /// [`MemCheckOpenings::rerand_claims`]), and their claimed values `X(eval_point)` (the GKR
  /// verifier's requested values, which seed the running claims and are the
  /// instance's initial sumcheck claims). Every column must have length
  /// `N = 2^{eval_point.len()}`.
  pub fn new(
    eval_point: Vec<E::Scalar>,
    columns: Vec<Vec<E::Scalar>>,
    claims: Vec<E::Scalar>,
  ) -> Self {
    assert_eq!(columns.len(), claims.len());
    let saved_evals = vec![[E::Scalar::ZERO; 3]; columns.len()];
    Self {
      eq_sumcheck: EqSumCheckInstance::new(eval_point),
      polys: columns
        .into_iter()
        .map(MultilinearPolynomial::new)
        .collect(),
      running_claims: claims,
      saved_evals,
      fused: None,
    }
  }
}

impl<E: Engine> SumcheckEngine<E> for RerandomizeSumcheckInstance<E> {
  fn initial_claims(&self) -> Vec<E::Scalar> {
    self.running_claims.clone()
  }

  fn degree(&self) -> usize {
    // True degree is 2 (eq · X); reported as 3 to ride in the degree-3
    // prove_helper bundle. See the type docs.
    3
  }

  fn size(&self) -> usize {
    let n = self.polys[0].len();
    debug_assert!(self.polys.iter().all(|p| p.len() == n));
    n
  }

  fn fuse_with_coeffs(&mut self, coeffs: &[E::Scalar]) {
    assert_eq!(coeffs.len(), self.polys.len());
    // Collapse the columns into `Σ_i coeffs[i] · column_i` and the claims into
    // `Σ_i coeffs[i] · claim_i`. Both the BDDT derivation and the N-scaling sum
    // that feed `evaluation_points_quadratic_with_one_input` are linear in
    // `(column, claim)`, so evaluating the combined column at the combined claim
    // equals summing the per-column triples — this makes the fused prover's
    // per-round message byte-identical to the unfused one.
    let n = self.polys[0].len();
    let mut combined = vec![E::Scalar::ZERO; n];
    combined.par_iter_mut().enumerate().for_each(|(idx, out)| {
      *out = self
        .polys
        .iter()
        .zip(coeffs.iter())
        .map(|(poly, &c)| c * poly[idx])
        .sum();
    });
    let running_claim = self
      .running_claims
      .iter()
      .zip(coeffs.iter())
      .map(|(&claim, &c)| c * claim)
      .sum();
    self.fused = Some(FusedRerandomize {
      poly: MultilinearPolynomial::new(combined),
      running_claim,
      saved: [E::Scalar::ZERO; 3],
    });
  }

  fn evaluation_points(&mut self) -> Vec<Vec<E::Scalar>> {
    // Each column is one quadratic `eq(eval_point, ·) · X(·)`, sampled the same
    // way as the E-claim. The cubic coefficient is zero (degree 2).
    if let Some(fused) = self.fused.as_mut() {
      // Fused: evaluate the single combined column, report its triple in slot 0
      // and zeros elsewhere. `prove_helper` computes `Σ coeffs[i]·evals[i]`; with
      // coeffs[0] == 1 (the slot leads the batch) this equals the combined triple
      // — identical to summing the seven per-column triples.
      let (e0, _, einf) = self
        .eq_sumcheck
        .evaluation_points_quadratic_with_one_input(&fused.poly, fused.running_claim);
      fused.saved = [e0, E::Scalar::ZERO, einf];
      let mut out = vec![vec![E::Scalar::ZERO; 3]; self.polys.len()];
      out[0] = vec![e0, E::Scalar::ZERO, einf];
      return out;
    }

    let evals: Vec<[E::Scalar; 3]> = self
      .polys
      .par_iter_mut()
      .zip(self.running_claims.par_iter())
      .map(|(poly, &claim)| {
        let (e0, _, einf) = self
          .eq_sumcheck
          .evaluation_points_quadratic_with_one_input_and_cached_delta(poly, claim);
        [e0, E::Scalar::ZERO, einf]
      })
      .collect();

    self.saved_evals = evals.clone();
    evals.into_iter().map(|e| e.to_vec()).collect()
  }

  fn bound(&mut self, r: &E::Scalar) {
    if let Some(fused) = self.fused.as_mut() {
      fused.running_claim = SumcheckProof::<E>::update_claim(fused.running_claim, &fused.saved, r);
      fused.poly.bind_poly_var_top(r);
      self.eq_sumcheck.bound(r);
      return;
    }

    self.running_claims = self
      .running_claims
      .iter()
      .zip(self.saved_evals.iter())
      .map(|(&claim, saved)| SumcheckProof::<E>::update_claim(claim, saved, r))
      .collect();

    self
      .polys
      .par_iter_mut()
      .for_each(|poly| poly.bind_poly_var_top_with_cached_delta(r));

    self.eq_sumcheck.bound(r);
  }

  fn final_claims(&self) -> Vec<Vec<E::Scalar>> {
    // Fused: only the combined column survives; the ppSNARK GKR path reads
    // ts_row/ts_col directly from the PK columns rather than from here. Return the
    // combined final in slot 0 for symmetry.
    if let Some(fused) = self.fused.as_ref() {
      return vec![vec![fused.poly[0]]];
    }
    self.polys.iter().map(|p| vec![p[0]]).collect()
  }
}

#[cfg(test)]
mod tests {
  //! End-to-end tests through the top-level [`prove`]/[`verify`] pair. Each test
  //! builds four N=4 sub-instances and checks the host accepts a balanced witness
  //! while rejecting tampered multiplicities or mismatched column claims. This
  //! covers input-layer construction, reconcile, balance, and rerandomize claim
  //! ordering.
  use super::*;
  use crate::spartan::logup_gkr::proof::LayerFinalClaim;
  use crate::traits::TranscriptEngineTrait;

  type E = crate::provider::Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  /// A hand-built N=4 memory-check witness whose row and col relations both
  /// balance. We choose the fingerprint pieces directly (γ, r and the per-cell
  /// columns) and derive `mem_row`/`mem_col`/dens so that `Σ ts/(T+r) =
  /// Σ 1/(W+r)` holds on each side.
  ///
  /// Construction: pick 4 distinct table dens `T[i]` freely; the access side is
  /// a multiset of reads into those cells with multiplicities `ts`, so the
  /// access dens are exactly the `T` values repeated per read. With N=4 reads
  /// over the 4 cells and `ts = [ts0..ts3]` summing to 4, the balance is
  /// `Σ ts[i]/(T[i]+r) = Σ_reads 1/(T[read]+r)` — identical multisets, so it
  /// holds by construction. Here we use `ts = [2,1,1,0]` and reads
  /// `[cell0, cell0, cell1, cell2]`.
  struct Witness {
    gamma: Fr,
    r: Fr,
    r_outer_full: Vec<Fr>,
    cols: MemCheckWitness<E>,
  }

  // A balanced N=4 witness. mem_row is eq(r_outer_full, ·) so the verifier can
  // recompute it; we set r_outer_full = [0,0] giving mem_row = [1,0,0,0].
  fn balanced_witness() -> Witness {
    let r_outer_full = vec![Fr::ZERO, Fr::ZERO];
    let mem_row = EqPolynomial::new(r_outer_full.clone()).evals(); // [1,0,0,0]
    let mem_col = vec![Fr::from(5), Fr::from(6), Fr::from(7), Fr::from(8)];
    // reads = [cell0, cell0, cell1, cell2]; ts = [2,1,1,0].
    let reads = [0usize, 0, 1, 2];
    let ts_row = vec![Fr::from(2), Fr::from(1), Fr::from(1), Fr::ZERO];
    let ts_col = ts_row.clone();
    let addr_row: Vec<Fr> = reads.iter().map(|&i| Fr::from(i as u64)).collect();
    let addr_col = addr_row.clone();
    // access lookup value = the table value at the read cell.
    let L_row: Vec<Fr> = reads.iter().map(|&i| mem_row[i]).collect();
    let L_col: Vec<Fr> = reads.iter().map(|&i| mem_col[i]).collect();
    Witness {
      gamma: Fr::from(3),
      r: Fr::from(9),
      r_outer_full,
      cols: MemCheckWitness {
        mem_row,
        mem_col,
        L_row,
        L_col,
        addr_row,
        addr_col,
        ts_row,
        ts_col,
      },
    }
  }

  fn run(w: &Witness) -> Result<Vec<Fr>, NovaError> {
    let mut tr_p = <E as Engine>::TE::new(b"memcheck-test");
    let out = prove::<E>(w.cols.clone(), w.gamma, w.r, &mut tr_p).expect("prove");
    let mut tr_v = <E as Engine>::TE::new(b"memcheck-test");
    verify::<E>(
      &out.proof,
      w.gamma,
      w.r,
      &w.r_outer_full,
      &out.openings,
      &mut tr_v,
    )
  }

  #[test]
  fn accepts_balanced_witness() {
    let w = balanced_witness();
    let pt = run(&w).expect("host verifier must accept a balanced witness");
    assert_eq!(pt.len(), 2, "eval_point has log N = 2 variables");
  }

  #[test]
  fn rejects_tampered_ts() {
    // Break the row balance by bumping a multiplicity: Σ ts/(T+r) no longer
    // equals Σ 1/(W+r). The GKR proof is still built from the tampered layers,
    // so the balance check (not reconcile) is what fails.
    let mut w = balanced_witness();
    w.cols.ts_row[0] += Fr::ONE;
    assert!(run(&w).is_err(), "must reject an unbalanced multiplicity");
  }

  #[test]
  fn rejects_mismatched_opening() {
    // Keep the layers balanced but feed the host a wrong opened column, so the
    // reconcile step (recomputed fraction vs GKR opening) fails.
    let w = balanced_witness();
    let mut tr_p = <E as Engine>::TE::new(b"memcheck-test");
    let mut out = prove::<E>(w.cols.clone(), w.gamma, w.r, &mut tr_p).expect("prove");
    out.openings.eval_L_row += Fr::ONE; // inconsistent with the committed layer
    let mut tr_v = <E as Engine>::TE::new(b"memcheck-test");
    assert!(
      verify::<E>(
        &out.proof,
        w.gamma,
        w.r,
        &w.r_outer_full,
        &out.openings,
        &mut tr_v
      )
      .is_err(),
      "must reject an opening that disagrees with the GKR reduction"
    );
  }

  /// Zero cubic round poly whose `g(0)+g(1)=0` (valid for a zero sumcheck claim).
  fn zero_cubic_compressed() -> crate::spartan::polys::univariate::CompressedUniPoly<Fr> {
    use crate::spartan::polys::univariate::UniPoly;
    UniPoly::from_evals_deg3(&[Fr::ZERO, Fr::ZERO, Fr::ZERO, Fr::ZERO]).compress()
  }

  /// All-`(0,0)` intermediate GKR forged for `num_vars = 2`, `m = 4`.
  /// Variant A: roots `(0,1)` — the documented P0 shape (blocked at root gate).
  /// Variant B: roots `(0,0)` — passes GKR exact root check but fails host reconcile.
  fn forged_zero_gkr(roots_den_one: bool) -> LogupGkrProof<E> {
    let zero = Fraction::new(Fr::ZERO, Fr::ZERO);
    let split = LayerFinalClaim {
      left: zero,
      right: zero,
    };
    let root = if roots_den_one {
      Fraction::new(Fr::ZERO, Fr::ONE)
    } else {
      zero
    };
    LogupGkrProof {
      initial_claims: vec![root; 4],
      final_claims: vec![vec![split; 4], vec![split; 4]],
      sumchecks: vec![crate::spartan::logup_gkr::proof::LayerSumcheck {
        round_polys: vec![zero_cubic_compressed()],
      }],
    }
  }

  #[test]
  fn rejects_p0_zero_zero_chain_with_unit_roots() {
    // Classic P0: roots `(0,1)`, every split `(0,0)`, zero sumchecks, real
    // column openings. Cross-mult accepted this end-to-end; exact equality must
    // reject (at the GKR root gate).
    let w = balanced_witness();
    let mut tr_p = <E as Engine>::TE::new(b"memcheck-p0");
    let out = prove::<E>(w.cols.clone(), w.gamma, w.r, &mut tr_p).expect("prove");
    let forged = forged_zero_gkr(true);
    let mut tr_v = <E as Engine>::TE::new(b"memcheck-p0");
    assert!(
      verify::<E>(
        &forged,
        w.gamma,
        w.r,
        &w.r_outer_full,
        &out.openings,
        &mut tr_v
      )
      .is_err(),
      "must reject P0 (0,1)/(0,0) forgery"
    );
  }

  #[test]
  fn rejects_all_zero_gkr_chain_against_real_openings() {
    // Roots `(0,0)` make the root gate pass under exact equality too, but host
    // reconcile must still refuse `(a,b) == (0,0)`.
    let w = balanced_witness();
    let mut tr_p = <E as Engine>::TE::new(b"memcheck-p0b");
    let out = prove::<E>(w.cols.clone(), w.gamma, w.r, &mut tr_p).expect("prove");
    // Openings are at the honest eval_point; forged proof yields a different
    // point, but reconcile compares fraction components before that matters —
    // and even if GKR completed, reduced is `(0,0)` ≠ recomputed dens.
    // Rebuild openings for the forged eval_point by re-proving is unnecessary:
    // any nonzero fingerprint den against reduced `(0,0)` fails exact match.
    let forged = forged_zero_gkr(false);
    let mut tr_v = <E as Engine>::TE::new(b"memcheck-p0b");
    // Transcript label matches prove so this is a clean replay attempt; GKR
    // absorbs forged claims first. Real openings (wrong point) still give
    // nonzero dens almost surely vs reduced `(0,0)`.
    assert!(
      verify::<E>(
        &forged,
        w.gamma,
        w.r,
        &w.r_outer_full,
        &out.openings,
        &mut tr_v
      )
      .is_err(),
      "must reject all-(0,0) GKR against real column openings"
    );
  }

  #[test]
  fn rejects_wrong_gkr_depth() {
    // Forged depth-1 proof against trusted `r_outer_full` of length 2: must
    // return `Err`, not panic in `EqPolynomial::evaluate`.
    let w = balanced_witness();
    let mut tr_p = <E as Engine>::TE::new(b"memcheck-depth");
    let out = prove::<E>(w.cols.clone(), w.gamma, w.r, &mut tr_p).expect("prove");
    // Root `(0,1)` with children that gate to `(0,1)` so GKR accepts at depth 1.
    let one = Fraction::new(Fr::ZERO, Fr::ONE);
    let split = LayerFinalClaim {
      left: one,
      right: one, // gate = (0,1)+(0,1) = (0,1)
    };
    let forged = LogupGkrProof {
      initial_claims: vec![one; 4],
      final_claims: vec![vec![split; 4]],
      sumchecks: vec![],
    };
    let mut tr_v = <E as Engine>::TE::new(b"memcheck-depth");
    assert!(
      verify::<E>(
        &forged,
        w.gamma,
        w.r,
        &w.r_outer_full,
        &out.openings,
        &mut tr_v
      )
      .is_err(),
      "must reject GKR depth ≠ log N without panicking"
    );
  }

  #[test]
  fn rerandomize_claims_match_openings() {
    // The rerandomize instance's initial claims must be exactly the claimed
    // column values at eval_point, in the fixed RERAND order.
    let w = balanced_witness();
    let mut tr_p = <E as Engine>::TE::new(b"memcheck-test");
    let out = prove::<E>(w.cols.clone(), w.gamma, w.r, &mut tr_p).expect("prove");
    let claims = out.rerandomize.initial_claims();
    assert_eq!(claims, out.openings.rerand_claims().to_vec());
    assert_eq!(claims.len(), NUM_RERAND_COLUMNS);
  }

  #[cfg(feature = "evm")]
  #[test]
  fn gkr_proof_data_has_big_endian_scalar_golden_encoding() {
    let data = GkrProofData::<E> {
      proof: LogupGkrProof {
        initial_claims: vec![Fraction::new(Fr::from(1), Fr::from(2))],
        final_claims: vec![vec![LayerFinalClaim {
          left: Fraction::new(Fr::from(3), Fr::from(4)),
          right: Fraction::new(Fr::from(5), Fr::from(6)),
        }]],
        sumchecks: vec![],
      },
      rerand_claims: core::array::from_fn(|i| Fr::from(i as u64 + 7)),
    };
    let config = bincode::config::legacy()
      .with_big_endian()
      .with_fixed_int_encoding();
    let bytes = bincode::serde::encode_to_vec(&data, config).expect("serialize GKR proof data");

    fn push_len(bytes: &mut Vec<u8>, len: u64) {
      bytes.extend_from_slice(&len.to_be_bytes());
    }

    fn push_scalar(bytes: &mut Vec<u8>, value: u8) {
      bytes.extend_from_slice(&[0u8; 31]);
      bytes.push(value);
    }

    let mut expected = Vec::new();
    push_len(&mut expected, 1); // initial_claims
    push_scalar(&mut expected, 1);
    push_scalar(&mut expected, 2);
    push_len(&mut expected, 1); // final_claims layers
    push_len(&mut expected, 1); // final claims in the layer
    for value in 3..=6 {
      push_scalar(&mut expected, value);
    }
    push_len(&mut expected, 0); // sumchecks
    for value in 7..=13 {
      push_scalar(&mut expected, value);
    }
    assert_eq!(bytes, expected);

    let (decoded, consumed): (GkrProofData<E>, usize) =
      bincode::serde::decode_from_slice(&bytes, config).expect("deserialize GKR proof data");
    assert_eq!(consumed, expected.len());
    assert_eq!(
      bincode::serde::encode_to_vec(decoded, config).expect("re-serialize GKR proof data"),
      expected
    );
  }
}
