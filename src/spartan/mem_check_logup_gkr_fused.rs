//! Fused Logup-GKR last layer + ppSNARK Inner sumcheck.
//!
//! The default ppSNARK memory-check runs the full Logup-GKR argument to an
//! independent `gkr_eval_point`, then a seven-column rerandomize sumcheck carries
//! the reconcile columns from that point into the inner sumcheck's
//! `r_inner_batched`. Both are `O(N)` opening-point reductions of the same
//! length. This module fuses them: the GKR argument runs only to its input layer
//! (the prefix, see `super::logup_gkr::prover::prove_prefix`), and its **last
//! layer** shares the `n-1` suffix rounds and the final MSB fold `f` with the
//! Inner sumcheck's three relations, landing both on one shared point
//! `r_shared = f || s`. No seven-column rerandomize; a single PCS opening.
//!
//! ## The four fused relations
//! A fresh `beta` (sampled after the four initial claims are bound) batches:
//! - `G`: the GKR last layer over four fraction sub-instances (row/col ×
//!   table/access), internally batched by `lambda_last` powers (degree 3, carries
//!   the `eq(tau, ·)` factor);
//! - `A`: the ABC relation `Σ L_row·L_col·val` (degree 3);
//! - `E`: the error relation `Σ eq(q, y)·E(y)` with `q = r_outer_full` (degree 2);
//! - `W`: the witness relation `Σ masked_eq·W` (degree 2).
//!
//! `C_0 = C_G + beta·C_A + beta^2·C_E + beta^3·C_W`. Each suffix round sends one
//! fused cubic. After `n-1` rounds the running claim is `e_suffix`; the boundary
//! subtracts the GKR fold endpoint `G_end` (computed by the verifier from the
//! absorbed input splits) so the Inner MSB polynomial `h` satisfies
//! `h(0)+h(1) = e_suffix - G_end`. A single fold `f` closes both protocols.
//!
//! ## Soundness boundary (PLAN section 18)
//! `G_end` is recomputed by the verifier from the split claims (never absorbed
//! from the prover); the Inner endpoint (`h(f) == inner_expected`) and the GKR
//! reconcile are checked **separately** by the host; the splits and full `h` are
//! absorbed strictly before `f`; `beta != lambda_last` and is sampled after the
//! initial claims are bound.

use crate::errors::NovaError;
use crate::spartan::logup_gkr::fraction::Fraction;
use crate::spartan::logup_gkr::proof::LayerFinalClaim;
use crate::spartan::logup_gkr::prover::GkrProverPrefix;
use crate::spartan::logup_gkr::verifier::GkrVerifierPrefix;
use crate::spartan::polys::eq::EqPolynomial;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::spartan::polys::univariate::{CompressedUniPoly, UniPoly};
use crate::spartan::sumcheck::eq_sumcheck::EqSumCheckInstance;
use crate::spartan::sumcheck::SumcheckProof;
use crate::traits::{Engine, TranscriptEngineTrait};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// The protocol's Fiat-Shamir transcript labels. Narrow, fusion-specific labels
/// (PLAN section 7) so a future refactor cannot silently reorder the phases.
pub mod spec {
  /// Last-layer batching challenge `lambda_last` (8-component GKR gate RLC).
  pub const LAST_LAYER_LAMBDA: &[u8] = b"fll";
  /// Numerator label when absorbing an initial or split fraction.
  pub const NUM: &[u8] = b"fn";
  /// Denominator label when absorbing an initial or split fraction.
  pub const DEN: &[u8] = b"fd";
  /// Cross-protocol batching challenge `beta`.
  pub const BATCH_BETA: &[u8] = b"fbe";
  /// Fused suffix round polynomial.
  pub const ROUND_POLY: &[u8] = b"fp";
  /// Fused suffix round challenge.
  pub const ROUND_CHALLENGE: &[u8] = b"fc";
  /// Inner MSB (boundary) polynomial `h`.
  pub const MSB_POLY: &[u8] = b"fmp";
  /// Shared MSB fold challenge `f`.
  pub const SHARED_FOLD: &[u8] = b"fmf";
  /// Degree bound of every fused round polynomial (`eq`·gate / product = 3).
  pub const FUSED_DEGREE: usize = 3;
  /// Number of GKR fraction sub-instances (row/col × table/access).
  pub const NUM_GKR_INSTANCES: usize = 4;
}

/// A single fused GKR-last-layer + Inner-sumcheck proof.
///
/// `suffix_round_polys` holds the `n-1` shared suffix rounds; `input_splits`
/// holds the four GKR input-layer split claims (fixed length 4); `msb_round_poly`
/// is the single Inner MSB polynomial `h`, decompressed by the verifier with the
/// boundary hint `e_suffix - G_end`. For `n = 1` there are no suffix rounds and
/// `input_splits` are the root reduction's two-cell children.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct FusedGkrInnerProof<E: Engine> {
  /// The `n-1` shared suffix round polynomials (compressed cubics).
  pub suffix_round_polys: Vec<CompressedUniPoly<E::Scalar>>,
  /// The four GKR input-layer split claims, instance order
  /// `[row_table, row_access, col_table, col_access]`.
  pub input_splits: [LayerFinalClaim<E>; 4],
  /// The Inner MSB (boundary) polynomial `h`, compressed.
  pub msb_round_poly: CompressedUniPoly<E::Scalar>,
}

/// Prover inputs for the three Inner relations, plus their initial claims.
///
/// All `Vec`s have length `N = 2^n`. `val = val_A + c·val_B + c^2·val_C` and
/// `masked_eq` are pre-combined by the caller; `q = r_outer_full` has length `n`.
pub struct FusedInnerInputs<E: Engine> {
  /// Lookup polynomial `L_row`.
  #[allow(non_snake_case)]
  pub L_row: Vec<E::Scalar>,
  /// Lookup polynomial `L_col`.
  #[allow(non_snake_case)]
  pub L_col: Vec<E::Scalar>,
  /// Combined value polynomial `val_A + c·val_B + c^2·val_C`.
  pub val: Vec<E::Scalar>,
  /// Error polynomial `E`.
  #[allow(non_snake_case)]
  pub E: Vec<E::Scalar>,
  /// Masked-eq polynomial for the witness-bound relation.
  pub masked_eq: Vec<E::Scalar>,
  /// Witness polynomial `W`.
  #[allow(non_snake_case)]
  pub W: Vec<E::Scalar>,
  /// The point `q = r_outer_full` (length `n`) of the `E` relation.
  pub q: Vec<E::Scalar>,
  /// Initial claim `C_A = Σ L_row·L_col·val`.
  pub claim_abc: E::Scalar,
  /// Initial claim `C_E = Σ eq(q, y)·E(y)`.
  pub claim_e: E::Scalar,
  /// Initial claim `C_W = Σ masked_eq·W`.
  pub claim_w: E::Scalar,
}

/// The Inner MLE evaluations at `r_shared`, produced by the prover for the host's
/// endpoint check and PCS opening.
pub struct FusedEndpointEvals<E: Engine> {
  /// `L_row(r_shared)`.
  #[allow(non_snake_case)]
  pub L_row: E::Scalar,
  /// `L_col(r_shared)`.
  #[allow(non_snake_case)]
  pub L_col: E::Scalar,
  /// `val(r_shared)`.
  pub val: E::Scalar,
  /// `E(r_shared)`.
  #[allow(non_snake_case)]
  pub E: E::Scalar,
  /// `masked_eq(r_shared)`.
  pub masked_eq: E::Scalar,
  /// `W(r_shared)`.
  #[allow(non_snake_case)]
  pub W: E::Scalar,
  /// `ts_row(r_shared)` — the row_table numerator, read for free from the folded
  /// GKR split (no separate opening needed).
  pub ts_row: E::Scalar,
  /// `ts_col(r_shared)` — the col_table numerator, read for free from the folded
  /// GKR split.
  pub ts_col: E::Scalar,
}

/// The fused verifier's output: the shared point, the Inner endpoint value, the
/// batching challenge, and the four reduced GKR input-layer fractions.
pub(crate) struct FusedVerifierOutput<E: Engine> {
  /// The shared evaluation point `r_shared = f || s` (length `n`).
  pub r_shared: Vec<E::Scalar>,
  /// The Inner endpoint value `h(f)`, to be checked against `inner_expected`.
  pub e_inner: E::Scalar,
  /// The cross-protocol batching challenge `beta` (the host needs it to build
  /// `inner_expected`).
  pub beta: E::Scalar,
  /// The four reduced input-layer fractions in instance order, for the host's
  /// component-wise reconcile.
  pub gkr_fractions: [Fraction<E::Scalar>; 4],
}

/// Absorbs a fraction `(num, den)` into the transcript.
fn absorb_fraction<E: Engine>(transcript: &mut E::TE, frac: Fraction<E::Scalar>) {
  transcript.absorb(spec::NUM, &frac.num);
  transcript.absorb(spec::DEN, &frac.den);
}

/// The eight `lambda_last` weights `(a_i, b_i) = (lambda^{2i}, lambda^{2i+1})`.
fn gate_weights<E: Engine>(lambda: E::Scalar) -> [(E::Scalar, E::Scalar); 4] {
  let mut a = E::Scalar::ONE;
  core::array::from_fn(|_| {
    let b = a * lambda;
    let pair = (a, b);
    a = b * lambda;
    pair
  })
}

/// The GKR fold endpoint `G_end = eq(tau, s) · Σ_i [a_i·(nL·dR + nR·dL) + b_i·dL·dR]`
/// reconstructed from the four input splits (verifier recomputes; never absorbed).
fn gkr_endpoint<E: Engine>(
  splits: &[LayerFinalClaim<E>; 4],
  weights: &[(E::Scalar, E::Scalar); 4],
  eq_tau_s: E::Scalar,
) -> E::Scalar {
  let mut acc = E::Scalar::ZERO;
  for (split, (a, b)) in splits.iter().zip(weights.iter()) {
    let (nl, dl) = (split.left.num, split.left.den);
    let (nr, dr) = (split.right.num, split.right.den);
    acc += *a * (nl * dr + nr * dl) + *b * (dl * dr);
  }
  eq_tau_s * acc
}

/// Splits an owned `N`-length column into its MSB halves `(P_0, P_1)` where
/// `P_0(x) = P(0, x)` (indices `0..N/2`) and `P_1(x) = P(1, x)` (indices
/// `N/2..N`). Consumes the vector: the low half stays in place and only the high
/// half is a fresh allocation (`split_off`), halving the split copy cost.
fn split_msb<F: ff::PrimeField>(
  mut v: Vec<F>,
) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
  let n = v.len() / 2;
  let hi = v.split_off(n);
  (
    MultilinearPolynomial::new(v),
    MultilinearPolynomial::new(hi),
  )
}

/// The six Inner MLE `(low, high)` endpoint pairs at the current suffix point
/// `s` — i.e. `(P(0, s), P(1, s))` per column — that the shared MSB fold `f`
/// closes into one evaluation each.
#[allow(non_snake_case)]
struct InnerMsbPairs<F> {
  L_row: (F, F),
  L_col: (F, F),
  val: (F, F),
  E: (F, F),
  masked: (F, F),
  W: (F, F),
}

/// Closes the fused protocol's shared MSB round, the single place the boundary
/// algebra and its transcript order live (both the `n ≥ 2` and `n = 1` provers
/// call it). It:
/// 1. absorbs the four GKR input splits;
/// 2. builds the Inner boundary polynomial `h` (whose `h(0)+h(1)` must equal
///    `h_sum = e_suffix − G_end`) and absorbs it;
/// 3. squeezes the shared fold `f`;
/// 4. folds every Inner pair and the two table-side split numerators
///    (instances 0, 2) to their endpoint evaluation at `f`.
///
/// `eq_qsuf_s` is `eq(q_suffix, s)`, which is `ONE` when there is no suffix
/// (`n = 1`), so the single closure covers both depths. Returns the compressed
/// `h`, the fold `f`, and the endpoint evaluations.
#[allow(non_snake_case)]
#[allow(clippy::too_many_arguments)]
fn close_inner_msb<E: Engine>(
  pairs: &InnerMsbPairs<E::Scalar>,
  q_0: E::Scalar,
  eq_qsuf_s: E::Scalar,
  beta: E::Scalar,
  beta2: E::Scalar,
  beta3: E::Scalar,
  h_sum: E::Scalar,
  input_splits: &[LayerFinalClaim<E>; spec::NUM_GKR_INSTANCES],
  transcript: &mut E::TE,
) -> Result<
  (
    CompressedUniPoly<E::Scalar>,
    E::Scalar,
    FusedEndpointEvals<E>,
  ),
  NovaError,
> {
  let one = E::Scalar::ONE;
  for sp in input_splits {
    absorb_fraction::<E>(transcript, sp.left);
    absorb_fraction::<E>(transcript, sp.right);
  }

  // h(X) = beta·h_A + beta^2·h_E + beta^3·h_W, each factor linearly interpolated
  // between the pair's low and high halves at X.
  let h_at = |x: E::Scalar| -> E::Scalar {
    let bar = |p: (E::Scalar, E::Scalar)| (one - x) * p.0 + x * p.1;
    let ha = bar(pairs.L_row) * bar(pairs.L_col) * bar(pairs.val);
    let eq_q0_x = (one - q_0) * (one - x) + q_0 * x;
    let he = eq_qsuf_s * eq_q0_x * bar(pairs.E);
    let hw = bar(pairs.masked) * bar(pairs.W);
    beta * ha + beta2 * he + beta3 * hw
  };
  let two = one + one;
  let three = two + one;
  let h_poly = UniPoly::from_evals(&[h_at(E::Scalar::ZERO), h_at(one), h_at(two), h_at(three)]);
  debug_assert_eq!(
    h_poly.eval_at_zero() + h_poly.eval_at_one(),
    h_sum,
    "h(0)+h(1) must equal e_suffix - G_end"
  );

  transcript.absorb(spec::MSB_POLY, &h_poly);
  let f = transcript.squeeze(spec::SHARED_FOLD)?;

  let barf = |p: (E::Scalar, E::Scalar)| (one - f) * p.0 + f * p.1;
  let endpoints = FusedEndpointEvals {
    L_row: barf(pairs.L_row),
    L_col: barf(pairs.L_col),
    val: barf(pairs.val),
    E: barf(pairs.E),
    masked_eq: barf(pairs.masked),
    W: barf(pairs.W),
    // ts_row / ts_col are the row_table / col_table numerators (instances 0, 2),
    // read for free from the folded splits at f — no separate opening.
    ts_row: barf((input_splits[0].left.num, input_splits[0].right.num)),
    ts_col: barf((input_splits[2].left.num, input_splits[2].right.num)),
  };
  Ok((h_poly.compress(), f, endpoints))
}

/// Proves the fused GKR-last-layer + Inner argument, returning the proof, the
/// shared point `r_shared`, and the Inner endpoint evaluations for the host.
///
/// `gkr_prefix` is the state from [`super::logup_gkr::prover::prove_prefix`] over
/// the four fraction sub-instances; `inner` supplies the three Inner relations.
/// The prover must have already bound `gkr_prefix`'s prefix into the transcript.
#[allow(non_snake_case)]
pub(crate) fn prove_fused<E: Engine>(
  gkr_prefix: GkrProverPrefix<E>,
  inner: FusedInnerInputs<E>,
  transcript: &mut E::TE,
) -> Result<(FusedGkrInnerProof<E>, Vec<E::Scalar>, FusedEndpointEvals<E>), NovaError> {
  assert_eq!(
    gkr_prefix.m,
    spec::NUM_GKR_INSTANCES,
    "fused GKR needs 4 instances"
  );
  let n = gkr_prefix.num_vars;

  if n == 1 {
    return prove_fused_base(gkr_prefix, inner, transcript);
  }

  let GkrProverPrefix {
    running,
    eval_point: tau,
    input_layers,
    final_claims_by_layer,
    ..
  } = gkr_prefix;
  debug_assert_eq!(tau.len(), n - 1);

  // 1. lambda_last and the batched last-layer initial claim C_G.
  let lambda_last = transcript.squeeze(spec::LAST_LAYER_LAMBDA)?;
  let weights = gate_weights::<E>(lambda_last);
  let c_g: E::Scalar = running
    .iter()
    .zip(weights.iter())
    .map(|((num, den), (a, b))| *a * *num + *b * *den)
    .sum();

  // 2. Split every column into MSB halves. Each split is a fresh hi-half
  //    allocation + copy (page-fault bound), so the 8 GKR splits and 6 Inner
  //    splits run concurrently to overlap those copies across cores.
  type Half<E> = MultilinearPolynomial<<E as Engine>::Scalar>;
  type Quad<E> = (Half<E>, Half<E>, Half<E>, Half<E>);
  let (gkr_halves, inner_halves): (Vec<Quad<E>>, Vec<(Half<E>, Half<E>)>) = rayon::join(
    || {
      input_layers
        .into_par_iter()
        .map(|layer| {
          let (num_lo, num_hi) = split_msb(layer.num.Z);
          let (den_lo, den_hi) = split_msb(layer.den.Z);
          (num_lo, num_hi, den_lo, den_hi)
        })
        .collect()
    },
    || {
      vec![
        inner.L_row,
        inner.L_col,
        inner.val,
        inner.E,
        inner.masked_eq,
        inner.W,
      ]
      .into_par_iter()
      .map(split_msb)
      .collect()
    },
  );

  let mut nl = Vec::with_capacity(4);
  let mut nr = Vec::with_capacity(4);
  let mut dl = Vec::with_capacity(4);
  let mut dr = Vec::with_capacity(4);
  for (num_lo, num_hi, den_lo, den_hi) in gkr_halves {
    nl.push(num_lo);
    nr.push(num_hi);
    dl.push(den_lo);
    dr.push(den_hi);
  }

  // Consume the inner halves in the `vec![]` order above (L_row, L_col, val, E,
  // masked, W); each entry is `(low_half, high_half)`.
  let mut inner_iter = inner_halves.into_iter();
  let (mut l_row_0, mut l_row_1) = inner_iter.next().unwrap();
  let (mut l_col_0, mut l_col_1) = inner_iter.next().unwrap();
  let (mut val_0, mut val_1) = inner_iter.next().unwrap();
  let (mut e_0, mut e_1) = inner_iter.next().unwrap();
  let (mut masked_0, mut masked_1) = inner_iter.next().unwrap();
  let (mut w_0, mut w_1) = inner_iter.next().unwrap();

  let q_0 = inner.q[0];
  let q_suffix: Vec<E::Scalar> = inner.q[1..].to_vec();

  // 4. Bind the four initial claims, sample fresh beta.
  transcript.absorb(
    b"fic",
    &[c_g, inner.claim_abc, inner.claim_e, inner.claim_w].as_slice(),
  );
  let beta = transcript.squeeze(spec::BATCH_BETA)?;
  let beta2 = beta * beta;
  let beta3 = beta2 * beta;

  // Per-relation running claims.
  let mut e_g = c_g;
  let mut e_a = inner.claim_abc;
  let mut e_e = inner.claim_e;
  let mut e_w = inner.claim_w;

  let mut eq_gkr = EqSumCheckInstance::<E>::new(tau.clone());
  let mut eq_e = EqSumCheckInstance::<E>::new(q_suffix.clone());

  let mut suffix_round_polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::with_capacity(n - 1);
  let mut s: Vec<E::Scalar> = Vec::with_capacity(n - 1);

  // 5. The n-1 shared suffix rounds. GKR / ABC / E / W use the cached-delta fast
  // paths (BDDT claim derivation + delta caching for the following bind). Only
  // `e_0`/`e_1` bind non-cached: they carry no round evaluation and are kept
  // solely for the boundary `h`.
  //
  // §20.2: the GKR round-0 `t(0)` is the prefix's last parent-reduction left
  // claims (weighted), so round 0 scans only for `t(inf)` — the same shortcut
  // `prove_layer_sumcheck` applies to every prefix layer (commit 23d83cd). This
  // only reshapes the GKR round-0 evaluation; the numeric result is identical.
  let gkr_round0_t_0: E::Scalar = final_claims_by_layer
    .last()
    .expect("fused last layer follows the prefix parent reduction")
    .iter()
    .zip(weights.iter())
    .map(|(fc, (w_num, w_den))| *w_num * fc.left.num + *w_den * fc.left.den)
    .sum();
  for round in 0..(n - 1) {
    let (g_g0, g_gc, g_gm1) = if round == 0 {
      eq_gkr.evaluation_points_logup_gate_and_cache_deltas_with_t_0(
        &mut nl,
        &mut nr,
        &mut dl,
        &mut dr,
        &weights,
        gkr_round0_t_0,
        e_g,
      )
    } else {
      eq_gkr.evaluation_points_logup_gate_and_cache_deltas(
        &mut nl, &mut nr, &mut dl, &mut dr, &weights, e_g,
      )
    };

    let (a0_0, ac_0, am1_0) = SumcheckProof::<E>::compute_eval_points_cubic_with_cached_deltas(
      &mut l_row_0,
      &mut l_col_0,
      &mut val_0,
    );
    let (a0_1, ac_1, am1_1) = SumcheckProof::<E>::compute_eval_points_cubic_with_cached_deltas(
      &mut l_row_1,
      &mut l_col_1,
      &mut val_1,
    );
    let (g_a0, g_ac, g_am1) = (a0_0 + a0_1, ac_0 + ac_1, am1_0 + am1_1);

    let (g_e0, _e_bound, g_em1) = eq_e
      .evaluation_points_quadratic_with_two_inputs_and_cached_delta(&mut e_0, &mut e_1, q_0, e_e);

    let (w0_0, wm1_0) =
      SumcheckProof::<E>::compute_eval_points_quadratic_with_cached_deltas(&mut masked_0, &mut w_0);
    let (w0_1, wm1_1) =
      SumcheckProof::<E>::compute_eval_points_quadratic_with_cached_deltas(&mut masked_1, &mut w_1);
    let (g_w0, g_wm1) = (w0_0 + w0_1, wm1_0 + wm1_1);

    // Per-relation polynomials advance each individual running claim.
    let g_poly = UniPoly::from_evals_deg3(&[g_g0, e_g - g_g0, g_gc, g_gm1]);
    let a_poly = UniPoly::from_evals_deg3(&[g_a0, e_a - g_a0, g_ac, g_am1]);
    let e_poly = UniPoly::from_evals_deg3(&[g_e0, e_e - g_e0, E::Scalar::ZERO, g_em1]);
    let w_poly = UniPoly::from_evals_deg3(&[g_w0, e_w - g_w0, E::Scalar::ZERO, g_wm1]);

    // Fused round polynomial (E and W contribute 0 to the cubic coefficient).
    let f0 = g_g0 + beta * g_a0 + beta2 * g_e0 + beta3 * g_w0;
    let fc = g_gc + beta * g_ac;
    let fm1 = g_gm1 + beta * g_am1 + beta2 * g_em1 + beta3 * g_wm1;
    let e_comb = e_g + beta * e_a + beta2 * e_e + beta3 * e_w;
    let fused = UniPoly::from_evals_deg3(&[f0, e_comb - f0, fc, fm1]);

    transcript.absorb(spec::ROUND_POLY, &fused);
    let s_j = transcript.squeeze(spec::ROUND_CHALLENGE)?;

    e_g = g_poly.evaluate(&s_j);
    e_a = a_poly.evaluate(&s_j);
    e_e = e_poly.evaluate(&s_j);
    e_w = w_poly.evaluate(&s_j);

    // Cached-delta bind for GKR / ABC / E (their high halves now hold deltas).
    for p in nl
      .iter_mut()
      .chain(nr.iter_mut())
      .chain(dl.iter_mut())
      .chain(dr.iter_mut())
    {
      p.bind_poly_var_top_with_cached_delta(&s_j);
    }
    eq_gkr.bound(&s_j);
    for p in [
      &mut l_row_0,
      &mut l_row_1,
      &mut l_col_0,
      &mut l_col_1,
      &mut val_0,
      &mut val_1,
    ] {
      p.bind_poly_var_top_with_cached_delta(&s_j);
    }
    eq_e.bound(&s_j);
    // Cached-delta bind for the witness + E halves (their high halves hold the
    // deltas cached by this round's W and E evaluations).
    for p in [
      &mut masked_0,
      &mut masked_1,
      &mut w_0,
      &mut w_1,
      &mut e_0,
      &mut e_1,
    ] {
      p.bind_poly_var_top_with_cached_delta(&s_j);
    }

    s.push(s_j);
    suffix_round_polys.push(fused.compress());
  }

  // 6. Input splits + boundary. `G_end` is reconstructed from the splits (never
  // absorbed); `close_inner_msb` then absorbs the splits, `h`, and folds `f`.
  let input_splits: [LayerFinalClaim<E>; 4] = core::array::from_fn(|i| {
    LayerFinalClaim::<E>::new(nl[i].Z[0], nr[i].Z[0], dl[i].Z[0], dr[i].Z[0])
  });

  let eq_tau_s = EqPolynomial::new(tau.clone()).evaluate(&s);
  let g_end = gkr_endpoint::<E>(&input_splits, &weights, eq_tau_s);
  let e_suffix = e_g + beta * e_a + beta2 * e_e + beta3 * e_w;
  let h_sum = e_suffix - g_end;

  let eq_qsuf_s = EqPolynomial::new(q_suffix.clone()).evaluate(&s);
  let pairs = InnerMsbPairs {
    L_row: (l_row_0.Z[0], l_row_1.Z[0]),
    L_col: (l_col_0.Z[0], l_col_1.Z[0]),
    val: (val_0.Z[0], val_1.Z[0]),
    E: (e_0.Z[0], e_1.Z[0]),
    masked: (masked_0.Z[0], masked_1.Z[0]),
    W: (w_0.Z[0], w_1.Z[0]),
  };
  let (msb_round_poly, f, endpoints) = close_inner_msb::<E>(
    &pairs,
    q_0,
    eq_qsuf_s,
    beta,
    beta2,
    beta3,
    h_sum,
    &input_splits,
    transcript,
  )?;

  // r_shared = f || s (fold challenge as MSB, suffix challenges as the tail).
  let mut r_shared = Vec::with_capacity(n);
  r_shared.push(f);
  r_shared.extend_from_slice(&s);

  let proof = FusedGkrInnerProof {
    suffix_round_polys,
    input_splits,
    msb_round_poly,
  };
  Ok((proof, r_shared, endpoints))
}

/// The `n = 1` prover: no suffix rounds. The GKR root reduction is an exact
/// component check (done by the verifier from the absorbed splits); the Inner
/// relations contribute a single MSB polynomial `h` with
/// `h(0)+h(1) = beta·C_A + beta^2·C_E + beta^3·C_W`.
#[allow(non_snake_case)]
fn prove_fused_base<E: Engine>(
  gkr_prefix: GkrProverPrefix<E>,
  inner: FusedInnerInputs<E>,
  transcript: &mut E::TE,
) -> Result<(FusedGkrInnerProof<E>, Vec<E::Scalar>, FusedEndpointEvals<E>), NovaError> {
  let GkrProverPrefix { input_layers, .. } = gkr_prefix;
  let input_splits: [LayerFinalClaim<E>; 4] = core::array::from_fn(|i| {
    LayerFinalClaim::<E>::new(
      input_layers[i].num.Z[0],
      input_layers[i].num.Z[1],
      input_layers[i].den.Z[0],
      input_layers[i].den.Z[1],
    )
  });

  transcript.absorb(
    b"fic",
    &[inner.claim_abc, inner.claim_e, inner.claim_w].as_slice(),
  );
  let beta = transcript.squeeze(spec::BATCH_BETA)?;
  let beta2 = beta * beta;
  let beta3 = beta2 * beta;

  // No suffix rounds, so the Inner pairs are the raw two-cell columns and
  // eq(q_suffix, s) collapses to ONE; the boundary sum is the beta-batched
  // initial claims. `close_inner_msb` handles the rest identically to n >= 2.
  let pairs = InnerMsbPairs {
    L_row: (inner.L_row[0], inner.L_row[1]),
    L_col: (inner.L_col[0], inner.L_col[1]),
    val: (inner.val[0], inner.val[1]),
    E: (inner.E[0], inner.E[1]),
    masked: (inner.masked_eq[0], inner.masked_eq[1]),
    W: (inner.W[0], inner.W[1]),
  };
  let h_sum = beta * inner.claim_abc + beta2 * inner.claim_e + beta3 * inner.claim_w;
  let (msb_round_poly, f, endpoints) = close_inner_msb::<E>(
    &pairs,
    inner.q[0],
    E::Scalar::ONE,
    beta,
    beta2,
    beta3,
    h_sum,
    &input_splits,
    transcript,
  )?;

  let proof = FusedGkrInnerProof {
    suffix_round_polys: vec![],
    input_splits,
    msb_round_poly,
  };
  Ok((proof, vec![f], endpoints))
}

/// Verifies the fused argument, returning the shared point, the Inner endpoint
/// value `h(f)`, the batching challenge `beta`, and the four reduced input-layer
/// fractions. The host then checks `e_inner == inner_expected` and reconciles the
/// four fractions against its committed columns at `r_shared`.
pub(crate) fn verify_fused<E: Engine>(
  proof: &FusedGkrInnerProof<E>,
  gkr_prefix: GkrVerifierPrefix<E>,
  claim_abc: E::Scalar,
  claim_e: E::Scalar,
  claim_w: E::Scalar,
  transcript: &mut E::TE,
) -> Result<FusedVerifierOutput<E>, NovaError> {
  assert_eq!(gkr_prefix.m, spec::NUM_GKR_INSTANCES);
  let n = gkr_prefix.num_vars;

  if n == 1 {
    return verify_fused_base(proof, gkr_prefix, claim_abc, claim_e, claim_w, transcript);
  }

  if proof.suffix_round_polys.len() != n - 1 {
    return Err(NovaError::InvalidSumcheckProof);
  }
  let GkrVerifierPrefix {
    running,
    point: tau,
    ..
  } = gkr_prefix;

  let lambda_last = transcript.squeeze(spec::LAST_LAYER_LAMBDA)?;
  let weights = gate_weights::<E>(lambda_last);
  let c_g: E::Scalar = running
    .iter()
    .zip(weights.iter())
    .map(|(fr, (a, b))| *a * fr.num + *b * fr.den)
    .sum();

  transcript.absorb(b"fic", &[c_g, claim_abc, claim_e, claim_w].as_slice());
  let beta = transcript.squeeze(spec::BATCH_BETA)?;
  let beta2 = beta * beta;
  let beta3 = beta2 * beta;
  let mut e = c_g + beta * claim_abc + beta2 * claim_e + beta3 * claim_w;

  let mut s: Vec<E::Scalar> = Vec::with_capacity(n - 1);
  for poly_c in &proof.suffix_round_polys {
    let poly = poly_c.decompress(&e);
    if poly.degree() > spec::FUSED_DEGREE {
      return Err(NovaError::InvalidSumcheckProof);
    }
    transcript.absorb(spec::ROUND_POLY, &poly);
    let s_j = transcript.squeeze(spec::ROUND_CHALLENGE)?;
    e = poly.evaluate(&s_j);
    s.push(s_j);
  }
  let e_suffix = e;

  for sp in &proof.input_splits {
    absorb_fraction::<E>(transcript, sp.left);
    absorb_fraction::<E>(transcript, sp.right);
  }
  let eq_tau_s = EqPolynomial::new(tau.clone()).evaluate(&s);
  let g_end = gkr_endpoint::<E>(&proof.input_splits, &weights, eq_tau_s);
  let h_sum = e_suffix - g_end;

  let h_poly = proof.msb_round_poly.decompress(&h_sum);
  if h_poly.degree() > spec::FUSED_DEGREE {
    return Err(NovaError::InvalidSumcheckProof);
  }
  transcript.absorb(spec::MSB_POLY, &h_poly);
  let f = transcript.squeeze(spec::SHARED_FOLD)?;

  let mut r_shared = Vec::with_capacity(n);
  r_shared.push(f);
  r_shared.extend_from_slice(&s);
  let e_inner = h_poly.evaluate(&f);
  let gkr_fractions: [Fraction<E::Scalar>; 4] =
    core::array::from_fn(|i| proof.input_splits[i].fold_into_next_claim(f));

  Ok(FusedVerifierOutput {
    r_shared,
    e_inner,
    beta,
    gkr_fractions,
  })
}

/// The `n = 1` verifier: exact component-wise root-gate check (never a
/// probabilistic reduction), then the single Inner MSB polynomial.
fn verify_fused_base<E: Engine>(
  proof: &FusedGkrInnerProof<E>,
  gkr_prefix: GkrVerifierPrefix<E>,
  claim_abc: E::Scalar,
  claim_e: E::Scalar,
  claim_w: E::Scalar,
  transcript: &mut E::TE,
) -> Result<FusedVerifierOutput<E>, NovaError> {
  if !proof.suffix_round_polys.is_empty() {
    return Err(NovaError::InvalidSumcheckProof);
  }
  let root = gkr_prefix.running;

  transcript.absorb(b"fic", &[claim_abc, claim_e, claim_w].as_slice());
  let beta = transcript.squeeze(spec::BATCH_BETA)?;
  let beta2 = beta * beta;
  let beta3 = beta2 * beta;
  let h_sum = beta * claim_abc + beta2 * claim_e + beta3 * claim_w;

  for sp in &proof.input_splits {
    absorb_fraction::<E>(transcript, sp.left);
    absorb_fraction::<E>(transcript, sp.right);
  }
  // R7: the root claim must equal the fraction-add gate of its two children,
  // component-wise (never a scaled / probabilistic check).
  for (rt, sp) in root.iter().zip(proof.input_splits.iter()) {
    let gate = sp.compute_gate();
    if rt.num != gate.num || rt.den != gate.den {
      return Err(NovaError::InvalidSumcheckProof);
    }
  }

  let h_poly = proof.msb_round_poly.decompress(&h_sum);
  if h_poly.degree() > spec::FUSED_DEGREE {
    return Err(NovaError::InvalidSumcheckProof);
  }
  transcript.absorb(spec::MSB_POLY, &h_poly);
  let f = transcript.squeeze(spec::SHARED_FOLD)?;

  let e_inner = h_poly.evaluate(&f);
  let gkr_fractions: [Fraction<E::Scalar>; 4] =
    core::array::from_fn(|i| proof.input_splits[i].fold_into_next_claim(f));

  Ok(FusedVerifierOutput {
    r_shared: vec![f],
    e_inner,
    beta,
    gkr_fractions,
  })
}

/// Reconstructs the [`GkrVerifierPrefix`] from the GKR prefix proof pieces by
/// replaying the prefix reductions. For `n = 1` (no prefix layers) it only binds
/// the four root claims. Shared by the isolated tests and the ppSNARK host.
pub(crate) fn verify_gkr_prefix<E: Engine>(
  initial_claims: Vec<Fraction<E::Scalar>>,
  prefix_final_claims: Vec<Vec<LayerFinalClaim<E>>>,
  prefix_sumchecks: Vec<crate::spartan::logup_gkr::proof::LayerSumcheck<E>>,
  transcript: &mut E::TE,
) -> Result<GkrVerifierPrefix<E>, NovaError> {
  use crate::spartan::logup_gkr::proof::LogupGkrProof;
  // The four sub-instances are fixed (`row/col × table/access`); reject any other
  // count up front rather than relying on the length-4 destructure in
  // `reconcile_and_balance` to catch it downstream.
  if initial_claims.len() != spec::NUM_GKR_INSTANCES {
    return Err(NovaError::InvalidNumInstances);
  }
  if prefix_final_claims.is_empty() {
    // n = 1: no prefix layers; bind the root claims exactly as prove_prefix did.
    for c in &initial_claims {
      crate::spartan::logup_gkr::verifier::absorb_fraction::<E>(transcript, *c);
    }
    return Ok(GkrVerifierPrefix::new(
      spec::NUM_GKR_INSTANCES,
      1,
      initial_claims,
      vec![],
    ));
  }
  let temp = LogupGkrProof {
    initial_claims,
    final_claims: prefix_final_claims,
    sumchecks: prefix_sumchecks,
  };
  let claim = crate::spartan::logup_gkr::verifier::verify::<E>(&temp, transcript)?;
  Ok(GkrVerifierPrefix::new(
    spec::NUM_GKR_INSTANCES,
    claim.eval_point().len() + 1,
    claim.openings().to_vec(),
    claim.eval_point().to_vec(),
  ))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::spartan::logup_gkr::layer::Layer;
  use crate::spartan::logup_gkr::proof::LayerSumcheck;
  use crate::spartan::logup_gkr::prover::prove_prefix;

  type E = crate::provider::Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  struct Rng(u64);
  impl Rng {
    fn next_u64(&mut self) -> u64 {
      self.0 = self
        .0
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      self.0 >> 1
    }
    fn fr(&mut self) -> Fr {
      Fr::from(self.next_u64())
    }
    fn nonzero_fr(&mut self) -> Fr {
      Fr::from(self.next_u64() | 1)
    }
  }

  #[allow(non_snake_case)]
  struct Inputs {
    layer_num: Vec<Vec<Fr>>,
    layer_den: Vec<Vec<Fr>>,
    L_row: Vec<Fr>,
    L_col: Vec<Fr>,
    val: Vec<Fr>,
    E: Vec<Fr>,
    masked: Vec<Fr>,
    W: Vec<Fr>,
    q: Vec<Fr>,
  }

  fn make_inputs(n: usize, seed: u64) -> Inputs {
    let big_n = 1usize << n;
    let mut rng = Rng(seed);
    let layer_num: Vec<Vec<Fr>> = (0..4)
      .map(|_| (0..big_n).map(|_| rng.fr()).collect())
      .collect();
    let layer_den: Vec<Vec<Fr>> = (0..4)
      .map(|_| (0..big_n).map(|_| rng.nonzero_fr()).collect())
      .collect();
    Inputs {
      layer_num,
      layer_den,
      L_row: (0..big_n).map(|_| rng.fr()).collect(),
      L_col: (0..big_n).map(|_| rng.fr()).collect(),
      val: (0..big_n).map(|_| rng.fr()).collect(),
      E: (0..big_n).map(|_| rng.fr()).collect(),
      masked: (0..big_n).map(|_| rng.fr()).collect(),
      W: (0..big_n).map(|_| rng.fr()).collect(),
      q: (0..n).map(|_| rng.fr()).collect(),
    }
  }

  fn mle(v: Vec<Fr>) -> MultilinearPolynomial<Fr> {
    MultilinearPolynomial::new(v)
  }

  fn inner_claims(inp: &Inputs) -> (Fr, Fr, Fr) {
    let c_a: Fr = (0..inp.L_row.len())
      .map(|i| inp.L_row[i] * inp.L_col[i] * inp.val[i])
      .sum();
    let c_e = mle(inp.E.clone()).evaluate(&inp.q);
    let c_w: Fr = (0..inp.masked.len())
      .map(|i| inp.masked[i] * inp.W[i])
      .sum();
    (c_a, c_e, c_w)
  }

  fn make_inner(inp: &Inputs) -> FusedInnerInputs<E> {
    let (c_a, c_e, c_w) = inner_claims(inp);
    FusedInnerInputs {
      L_row: inp.L_row.clone(),
      L_col: inp.L_col.clone(),
      val: inp.val.clone(),
      E: inp.E.clone(),
      masked_eq: inp.masked.clone(),
      W: inp.W.clone(),
      q: inp.q.clone(),
      claim_abc: c_a,
      claim_e: c_e,
      claim_w: c_w,
    }
  }

  type PrefixPieces = (
    Vec<Fraction<Fr>>,
    Vec<Vec<LayerFinalClaim<E>>>,
    Vec<LayerSumcheck<E>>,
  );

  fn prove(
    inp: &Inputs,
    transcript: &mut <E as Engine>::TE,
  ) -> (FusedGkrInnerProof<E>, Vec<Fr>, PrefixPieces, (Fr, Fr, Fr)) {
    let layers: Vec<Layer<E>> = (0..4)
      .map(|i| Layer::<E> {
        num: mle(inp.layer_num[i].clone()),
        den: mle(inp.layer_den[i].clone()),
      })
      .collect();
    let prefix = prove_prefix::<E>(layers, transcript).expect("prove_prefix");
    let pieces = (
      prefix.initial_claims.clone(),
      prefix.final_claims_by_layer.clone(),
      prefix.sumchecks.clone(),
    );
    let claims = inner_claims(inp);
    let inner = make_inner(inp);
    let (proof, r_shared, _endpoints) =
      prove_fused::<E>(prefix, inner, transcript).expect("prove_fused");
    (proof, r_shared, pieces, claims)
  }

  // The host reconcile oracle: direct MLE evaluation at r_shared. Returns Ok(())
  // iff the fused verifier output is consistent with the true columns.
  fn host_check(out: &FusedVerifierOutput<E>, inp: &Inputs) -> Result<(), ()> {
    let r = &out.r_shared;
    for i in 0..4 {
      let num = mle(inp.layer_num[i].clone()).evaluate(r);
      let den = mle(inp.layer_den[i].clone()).evaluate(r);
      if out.gkr_fractions[i].num != num || out.gkr_fractions[i].den != den {
        return Err(());
      }
    }
    let lr = mle(inp.L_row.clone()).evaluate(r);
    let lc = mle(inp.L_col.clone()).evaluate(r);
    let v = mle(inp.val.clone()).evaluate(r);
    let ev = mle(inp.E.clone()).evaluate(r);
    let me = mle(inp.masked.clone()).evaluate(r);
    let w = mle(inp.W.clone()).evaluate(r);
    let eq_q_r = EqPolynomial::new(inp.q.clone()).evaluate(r);
    let beta = out.beta;
    let beta2 = beta * beta;
    let beta3 = beta2 * beta;
    let inner_expected = beta * lr * lc * v + beta2 * eq_q_r * ev + beta3 * me * w;
    if out.e_inner != inner_expected {
      return Err(());
    }
    Ok(())
  }

  fn verify(
    proof: &FusedGkrInnerProof<E>,
    pieces: &PrefixPieces,
    claims: (Fr, Fr, Fr),
    transcript: &mut <E as Engine>::TE,
  ) -> Result<FusedVerifierOutput<E>, NovaError> {
    let prefix = verify_gkr_prefix::<E>(
      pieces.0.clone(),
      pieces.1.clone(),
      pieces.2.clone(),
      transcript,
    )?;
    verify_fused::<E>(proof, prefix, claims.0, claims.1, claims.2, transcript)
  }

  fn roundtrip(n: usize, seed: u64) {
    let inp = make_inputs(n, seed);
    let mut tr_p = <E as Engine>::TE::new(b"fused");
    let (proof, r_shared_p, pieces, claims) = prove(&inp, &mut tr_p);

    let mut tr_v = <E as Engine>::TE::new(b"fused");
    let out = verify(&proof, &pieces, claims, &mut tr_v).expect("verify_fused");

    assert_eq!(out.r_shared, r_shared_p, "n={n}: r_shared mismatch");
    assert_eq!(out.r_shared.len(), n, "n={n}: r_shared length");
    host_check(&out, &inp).unwrap_or_else(|_| panic!("n={n}: host reconcile failed"));
  }

  #[test]
  fn roundtrip_all_depths() {
    for n in 1..=5 {
      roundtrip(n, 100 + n as u64);
    }
  }

  // Gate C/D (endian): r_shared = [f, s_0, ..., s_{n-2}]; the suffix challenges
  // are the tail. The reversed order must NOT reconcile.
  #[test]
  fn endian_r_shared_is_fold_then_suffix() {
    let inp = make_inputs(4, 7);
    let mut tr_p = <E as Engine>::TE::new(b"fused");
    let (proof, r_shared, _pieces, _claims) = prove(&inp, &mut tr_p);
    // Direct fraction eval at r_shared matches the proof's folded splits;
    // at the reversed point it must not (with overwhelming probability).
    let f = r_shared[0];
    let mut reversed: Vec<Fr> = r_shared[1..].to_vec();
    reversed.push(f);
    for i in 0..4 {
      let num_ok = mle(inp.layer_num[i].clone()).evaluate(&r_shared);
      let num_rev = mle(inp.layer_num[i].clone()).evaluate(&reversed);
      assert_eq!(proof.input_splits[i].fold_into_next_claim(f).num, num_ok);
      assert_ne!(num_ok, num_rev, "reversed point must differ");
    }
  }

  fn expect_reject(mutate: impl FnOnce(&mut FusedGkrInnerProof<E>)) {
    let inp = make_inputs(4, 55);
    let mut tr_p = <E as Engine>::TE::new(b"fused");
    let (mut proof, _r, pieces, claims) = prove(&inp, &mut tr_p);
    mutate(&mut proof);
    let mut tr_v = <E as Engine>::TE::new(b"fused");
    let verdict = verify(&proof, &pieces, claims, &mut tr_v)
      .and_then(|out| host_check(&out, &inp).map_err(|_| NovaError::InvalidSumcheckProof));
    assert!(verdict.is_err(), "tampered proof must be rejected");
  }

  #[test]
  fn rejects_tampered_suffix_round_poly() {
    expect_reject(|p| {
      // Replace the first suffix round polynomial: the transcript diverges, so
      // r_shared changes and the reconcile fails.
      let bad = UniPoly::from_evals(&[Fr::from(2), Fr::from(5), Fr::from(11), Fr::from(4)]);
      p.suffix_round_polys[0] = bad.compress();
    });
  }

  #[test]
  fn rejects_tampered_input_split() {
    expect_reject(|p| p.input_splits[1].right.den += Fr::from(3));
  }

  #[test]
  fn rejects_tampered_msb_poly() {
    expect_reject(|p| {
      let bad = UniPoly::from_evals(&[Fr::from(1), Fr::from(2), Fr::from(9), Fr::from(1)]);
      p.msb_round_poly = bad.compress();
    });
  }

  #[test]
  fn rejects_tampered_claim() {
    // A wrong Inner claim must be rejected (the endpoint no longer matches).
    let inp = make_inputs(4, 88);
    let mut tr_p = <E as Engine>::TE::new(b"fused");
    let (proof, _r, pieces, claims) = prove(&inp, &mut tr_p);
    let bad_claims = (claims.0 + Fr::from(1), claims.1, claims.2);
    let mut tr_v = <E as Engine>::TE::new(b"fused");
    let verdict = verify(&proof, &pieces, bad_claims, &mut tr_v)
      .and_then(|out| host_check(&out, &inp).map_err(|_| NovaError::InvalidSumcheckProof));
    assert!(verdict.is_err(), "wrong Inner claim must be rejected");
  }

  // Differential lock (PLAN section 19.3, lightweight form): the fused last-layer
  // gate algebra (`gate_weights` + `gkr_endpoint`) must equal the standalone GKR
  // gate reconstruction — `eq · Σ_k λ^k · [compute_gate.num, compute_gate.den]`
  // (the exact expression `verify_one_reduction` reconciles against). The two
  // drivers share the per-round eval kernel but reimplement this boundary
  // separately, so this catches a future divergence of the λ-power schedule or
  // the fraction-add gate without needing to instrument the prover round-by-round.
  #[test]
  fn fused_gkr_endpoint_matches_standalone_gate_reconstruction() {
    use crate::spartan::logup_gkr::proof::LayerFinalClaim;
    let mut rng = Rng(2024);
    let lambda = rng.fr();
    let eq_tau_s = rng.fr();
    let splits: [LayerFinalClaim<E>; spec::NUM_GKR_INSTANCES] =
      core::array::from_fn(|_| LayerFinalClaim::<E>::new(rng.fr(), rng.fr(), rng.fr(), rng.fr()));

    let weights = gate_weights::<E>(lambda);
    let fused = gkr_endpoint::<E>(&splits, &weights, eq_tau_s);

    // Standalone: eq · horner over the flattened [gate.num, gate.den] per instance
    // (instance i's numerator on λ^{2i}, denominator on λ^{2i+1}).
    let mut standalone = Fr::ZERO;
    let mut pw = Fr::ONE;
    for sp in &splits {
      let g = sp.compute_gate();
      standalone += pw * g.num;
      pw *= lambda;
      standalone += pw * g.den;
      pw *= lambda;
    }
    standalone *= eq_tau_s;

    assert_eq!(
      fused, standalone,
      "fused last-layer gate algebra diverged from the standalone GKR reconstruction"
    );
  }
}
