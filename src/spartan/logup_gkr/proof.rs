//! Proof objects for the Logup-GKR fractional-sum argument.
//!
//! # One batched proof over equal-height trees
//! Each input [`Layer`](super::layer::Layer) defines a separate fractional-sum
//! tree. All inputs must already have the same number of variables; the prover
//! does not pad them. At each depth, the corresponding tree layers are batched
//! into one sumcheck via a fresh `λ`, producing one shared evaluation point for
//! all trees. Their claims are carried in a single proof rather than separate
//! per-tree proofs. Sumcheck round polynomials use Nova's `CompressedUniPoly`.

use crate::spartan::logup_gkr::fraction::Fraction;
use crate::spartan::polys::univariate::CompressedUniPoly;
use crate::traits::Engine;
use serde::{Deserialize, Serialize};

/// `rlc(a, b, r) = a + r·(b - a)` — the two-to-one fold of split claims.
#[inline(always)]
fn rlc<F: ff::Field>(a: F, b: F, r: F) -> F {
  a + r * (b - a)
}

/// A claim about one instance's column pair after a layer: the fraction
/// `num/den`.
pub type LayerClaim<E> = Fraction<<E as Engine>::Scalar>;

/// The split (even/odd) final claim of one instance at one layer: the `left`
/// and `right` child fractions `(nL,dL)` and `(nR,dR)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LayerFinalClaim<E: Engine> {
  /// Left child `(nL, dL)`.
  pub left: Fraction<E::Scalar>,
  /// Right child `(nR, dR)`.
  pub right: Fraction<E::Scalar>,
}

impl<E: Engine> LayerFinalClaim<E> {
  /// Builds from the four folded sumcheck evaluations `nL, nR, dL, dR`.
  ///
  /// WARNING: the layer sumcheck's `bound_evals` come out **interleaved** as
  /// `[nL, dR, nR, dL]` (matching the flattened virtual-polynomial order
  /// `numerator_left, denominator_right, numerator_right, denominator_left`).
  /// Map them explicitly — `new(evals[0], evals[2], evals[3], evals[1])` —
  /// never slice `evals[0..4]` into the parameters positionally.
  pub fn new(nL: E::Scalar, nR: E::Scalar, dL: E::Scalar, dR: E::Scalar) -> Self {
    Self {
      left: Fraction::new(nL, dL),
      right: Fraction::new(nR, dR),
    }
  }

  /// Folds the split claim into the next layer's claim via `rlc` at `r`.
  pub fn fold_into_next_claim(&self, r: E::Scalar) -> LayerClaim<E> {
    Fraction::new(
      rlc(self.left.num, self.right.num, r),
      rlc(self.left.den, self.right.den, r),
    )
  }

  /// The gate output `left + right` (projective fraction add).
  pub fn compute_gate(&self) -> Fraction<E::Scalar> {
    self.left + self.right
  }
}

/// Sumcheck transcript for one batched tree layer: one compressed round
/// polynomial per variable (the instances are batched into this single
/// sumcheck via `λ`).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LayerSumcheck<E: Engine> {
  /// Compressed round polynomials, one per sumcheck round of this layer.
  pub round_polys: Vec<CompressedUniPoly<E::Scalar>>,
}

/// A single **batched** Logup-GKR proof over all input instances.
///
/// Instance-indexed vectors preserve the caller's input order; ppSNARK uses
/// `[row_table, row_access, col_table, col_access]`. `initial_claims` are the
/// per-instance output-layer fractions (observed first). `final_claims[layer]`
/// holds one split claim per instance; `sumchecks[layer]` is the one batched
/// sumcheck for that layer. Layers are ordered output→input, and the top
/// transition (0-variable layer) carries no sumcheck, so
/// `sumchecks.len() + 1 == final_claims.len()`.
///
/// The per-layer batching challenge `λ` is **not** stored here: it is a
/// Fiat-Shamir challenge the verifier re-samples fresh at each layer (reusing
/// one `λ` across layers would let the prover adaptively forge each layer's
/// claims). Likewise `initial_claims` are not bound to committed data by this
/// proof alone; soundness closes at the host's reconcile step, where the
/// returned `openings` must match the fractions reconstructed from the host's
/// claimed column evaluations at `eval_point`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogupGkrProof<E: Engine> {
  /// Per-instance output-layer claims (the root fractions before fold-down).
  pub initial_claims: Vec<LayerClaim<E>>,
  /// Per-layer, per-instance split final claims, output→input.
  pub final_claims: Vec<Vec<LayerFinalClaim<E>>>,
  /// Per-layer batched sumcheck (one fewer than `final_claims`).
  pub sumchecks: Vec<LayerSumcheck<E>>,
}

/// The GKR **prefix** proof: every layer reduction except the last. Carried by
/// the ppSNARK fused memory-check, where the last layer is proven jointly with
/// the Inner sumcheck instead of being folded down here.
///
/// For height `N = 2^n`, `prefix_final_claims` holds the `n-1` prefix layers
/// (output→input) and `prefix_sumchecks` the `n-2` sumchecks (the root reduction
/// carries none). For `n = 1` both are empty (the single layer is the last one).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LogupGkrPrefixProof<E: Engine> {
  /// Per-instance output-layer (root) claims, absorbed first.
  pub initial_claims: Vec<LayerClaim<E>>,
  /// The `n-1` prefix layers' split final claims, output→input.
  pub prefix_final_claims: Vec<Vec<LayerFinalClaim<E>>>,
  /// The `n-2` prefix layer sumchecks.
  pub prefix_sumchecks: Vec<LayerSumcheck<E>>,
}

/// Verifier output — a **continuation token** carrying the shared opening
/// claim, with its point field named Nova's `eval_point`.
///
/// The host uses [`Self::eval_point`] to seed the opening-point reduction and
/// uses `openings` to reconcile every reduced input-layer fraction against its
/// claimed columns. The ppSNARK host also performs the `0/den` zero-sum check;
/// neither operation belongs to the GKR verifier.
#[derive(Clone, Debug)]
pub struct LogupGkrOpeningClaim<E: Engine> {
  eval_point: Vec<E::Scalar>,
  openings: Vec<Fraction<E::Scalar>>,
}

impl<E: Engine> LogupGkrOpeningClaim<E> {
  /// Constructs the token (only the GKR verifier should call this).
  pub fn new(eval_point: Vec<E::Scalar>, openings: Vec<Fraction<E::Scalar>>) -> Self {
    Self {
      eval_point,
      openings,
    }
  }

  /// The shared evaluation point to which all input-layer claims are reduced.
  pub fn eval_point(&self) -> &[E::Scalar] {
    &self.eval_point
  }

  /// The reduced input-layer fractions in caller input order, which the host
  /// reconstructs from its claimed column evaluations and compares against.
  pub fn openings(&self) -> &[Fraction<E::Scalar>] {
    &self.openings
  }
}
