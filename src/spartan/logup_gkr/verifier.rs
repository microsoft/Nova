//! Verifier for the Logup-GKR fractional-sum argument.
//!
//! **This file defines the protocol.** It is written independently of any
//! prover: every check is derived from the GKR fractional-sum argument, and the
//! Fiat-Shamir transcript order is fixed here by soundness reasoning (see the
//! module docs below). A prover is correct **iff** it produces a transcript and
//! claims that make this verifier accept; the prover must satisfy this file, not
//! the other way around.
//!
//! ## What is proven
//! For each logup instance `i`, an input layer of projective fractions folds
//! through a binary tree to a single root fraction `(v_p_i, v_q_i)`. The
//! verifier reduces the per-instance root claims down to a single evaluation of
//! the input layer at a shared point, checking each tree layer with one batched
//! sumcheck. It returns that point and the reduced per-instance fractions; the
//! `0/den` zero-sum balance check is the host's job (this argument owns no PCS).
//!
//! ## Layer reduction (the sumcheck contract)
//! A non-root layer with `t` variables, reduced at evaluation point `τ`
//! (`|τ| = t`), carries a per-instance running claim `(v_p_i, v_q_i)`. The
//! verifier samples a fresh batching challenge `λ` and **demands** a sumcheck
//! proving
//! ```text
//!   Σ_i (λ^{2i}·v_p_i + λ^{2i+1}·v_q_i)  =  Σ_x eq(τ, x) · Σ_i G_i(x),
//!   G_i(x) = λ^{2i}·(nL_i·dR_i + nR_i·dL_i)(x) + λ^{2i+1}·(dL_i·dR_i)(x),
//! ```
//! where `(nL_i, nR_i, dL_i, dR_i)` are the num/den halves of instance `i`'s
//! child layer. The 2m per-instance sub-claims (each instance's numerator and
//! denominator relation) are batched by **distinct powers of λ** — a Horner
//! combination over the flattened `[p_0,q_0,p_1,q_1,…]`. This is essential for
//! soundness with m ≥ 2 instances: a plain `Σ_i (v_p_i + λ·v_q_i)` would expose
//! only `Σ v_p` and `Σ v_q` (two slots regardless of m), so a prover could shift
//! one instance's numerator up and another's down and stay undetected. Distinct
//! λ-powers bind every instance's num and den separately.
//!
//! The sumcheck's final value must equal `eq(τ, r)·Σ_i G_i(r)`, reconstructed
//! here from the prover's claimed `(nL,nR,dL,dR)` at `r`. This is a demand on
//! the sumcheck backend: whatever the prover uses, its final evaluation must
//! reconcile transparently against this expression (Nova's
//! `prove_cubic_with_three_inputs` has this shape; see ppSNARK's outer check).
//!
//! ## Fiat-Shamir order (soundness)
//! The transcript order is chosen so each challenge is unpredictable given the
//! values it must bind — a prover cannot adaptively forge later messages:
//! 1. absorb all root claims `(v_p_i, v_q_i)` before anything is sampled;
//! 2. per layer (root→input): sample `λ` (bound to all prior claims/challenges,
//!    so the layer's claims cannot be chosen after `λ`), run the layer sumcheck
//!    (each round absorbs its polynomial then samples), absorb this layer's
//!    final claims, then sample the fold challenge `r_fold` (bound to those
//!    claims, so the two children cannot be chosen after `r_fold`).
//!
//! Reusing one `λ` across layers, or sampling `r_fold` before the claims it
//! folds, would each break soundness (adaptive-forgery gaps).

use crate::errors::NovaError;
use crate::spartan::logup_gkr::fraction::Fraction;
use crate::spartan::logup_gkr::proof::{LayerClaim, LogupGkrOpeningClaim, LogupGkrProof};
use crate::spartan::polys::eq::EqPolynomial;
use crate::spartan::sumcheck::SumcheckProof;
use crate::traits::{Engine, TranscriptEngineTrait};
use ff::Field;

/// The protocol's Fiat-Shamir transcript labels and sumcheck degree, defined by
/// the verifier. The prover must use exactly these.
pub mod spec {
  /// Round-polynomial label inside a layer sumcheck.
  pub const ROUND_POLY: &[u8] = b"p";
  /// Per-round sumcheck challenge label.
  pub const ROUND_CHALLENGE: &[u8] = b"c";
  /// Per-layer batching challenge `λ`.
  pub const LAMBDA: &[u8] = b"l";
  /// Per-layer fold challenge.
  pub const FOLD: &[u8] = b"f";
  /// Fraction numerator label.
  pub const NUM: &[u8] = b"n";
  /// Fraction denominator label.
  pub const DEN: &[u8] = b"d";
  /// Degree bound of each layer's batched sumcheck round polynomial:
  /// `eq` (degree 1) × gate (degree 2) = degree 3.
  pub const LAYER_SC_DEGREE: usize = 3;
}

/// Absorbs a fraction `(num, den)` into the transcript, in the order the
/// protocol fixes. Shared with the prover (which imports this).
pub fn absorb_fraction<E: Engine>(transcript: &mut E::TE, frac: Fraction<E::Scalar>) {
  transcript.absorb(spec::NUM, &frac.num);
  transcript.absorb(spec::DEN, &frac.den);
}

/// Evaluates `Σ_k coeffs[k] · lambda^k` by Horner's method over the coefficient
/// stream (lowest power first). Callers flatten their per-instance components
/// into this order — e.g. the layer sumcheck feeds `[num_0, den_0, num_1,
/// den_1, ...]`, so instance `i`'s numerator lands on `λ^{2i}` and its
/// denominator on `λ^{2i+1}` (a distinct power per component; see the layer
/// comment for why a shared power would be unsound).
fn horner_eval<F: Field>(coeffs: impl IntoIterator<Item = F>, lambda: F) -> F {
  let mut acc = F::ZERO;
  let mut pw = F::ONE;
  for c in coeffs {
    acc += pw * c;
    pw *= lambda;
  }
  acc
}

/// Verifies the batched fractional-sum proof and returns the shared opening
/// claim (evaluation point + per-instance input-layer fractions). Accepts iff
/// the proof satisfies the protocol defined in this module.
pub fn verify<E: Engine>(
  proof: &LogupGkrProof<E>,
  transcript: &mut E::TE,
) -> Result<LogupGkrOpeningClaim<E>, NovaError> {
  let m = proof.initial_claims.len();
  if m == 0 {
    return Err(NovaError::InvalidNumInstances);
  }
  // Structural well-formedness: `final_claims` has one entry per reduction step
  // (num_vars entries). The root reduction (index 0) carries no sumcheck, so
  // `sumchecks.len() + 1 == final_claims.len()`, and every layer carries one
  // split claim per instance.
  let num_vars = proof.final_claims.len();
  if num_vars == 0 || proof.sumchecks.len() + 1 != num_vars {
    return Err(NovaError::InvalidSumcheckProof);
  }
  if proof.final_claims.iter().any(|layer| layer.len() != m) {
    return Err(NovaError::InvalidSumcheckProof);
  }

  // (1) Bind the root claims before any challenge is drawn.
  for c in &proof.initial_claims {
    absorb_fraction::<E>(transcript, *c);
  }

  // running[i] = v_p_i / v_q_i: the claim about the current layer at `point`.
  let mut running: Vec<LayerClaim<E>> = proof.initial_claims.clone();
  let mut point: Vec<E::Scalar> = Vec::new();

  // (2) Reduce root → input. Step `t` reduces the layer with `t` variables;
  // t = 0 is the root reduction (no sumcheck), t >= 1 uses `sumchecks[t-1]`.
  for (t, layer_finals) in proof.final_claims.iter().enumerate() {
    let lambda = transcript.squeeze(spec::LAMBDA)?;

    if t == 0 {
      // Root reduction: the claimed root must equal the fraction-add gate of its
      // two child cells **component-wise** (`num` and `den` separately).
      // Cross-multiplication is unsound here: `(0,0)` cross-equals any
      // `(a,b)`, so a prover could post roots `(0,1)` with all-zero children and
      // later vacuously pass host reconcile against real columns. HyperPlonk
      // uses the same exact check; honesty already produces matching MLE
      // components, so this rejects only degenerate / scaled forgeries.
      for i in 0..m {
        let gate = layer_finals[i].compute_gate();
        let r = running[i];
        if r.num != gate.num || r.den != gate.den {
          return Err(NovaError::InvalidSumcheckProof);
        }
      }
    } else {
      // Layer sumcheck. The 2m per-instance sub-claims are batched by DISTINCT
      // powers of λ: instance i's numerator gets λ^{2i}, its denominator λ^{2i+1}
      // (Horner over the flattened `[p_0,q_0,p_1,q_1,...]`). A distinct power per
      // component binds every instance separately — a plain `Σ_i (p_i + λ q_i)`
      // would only expose `Σ p` and `Σ q` (two slots regardless of m), letting a
      // prover offset one instance's numerator up and another's down undetected.
      let claim: E::Scalar = horner_eval(running.iter().flat_map(|f| [f.num, f.den]), lambda);
      let sc = SumcheckProof::<E>::new(proof.sumchecks[t - 1].round_polys.clone());
      // The layer at step `t` has `t` variables, and `point` (its evaluation
      // point, i.e. the sumcheck's τ) has length `t`.
      let num_rounds = point.len();
      let (sc_eval, r) = sc.verify(claim, num_rounds, spec::LAYER_SC_DEGREE, transcript)?;

      // The sumcheck contract: its final value must equal
      //   eq(τ, r) · Σ_i [ λ^{2i}·gate_i.num + λ^{2i+1}·gate_i.den ]
      // where gate_i is the fraction-add of instance i's two children.
      let eq_at_r = EqPolynomial::new(point.clone()).evaluate(&r);
      let gate_sum: E::Scalar = horner_eval(
        layer_finals.iter().flat_map(|fc| {
          let g = fc.compute_gate();
          [g.num, g.den]
        }),
        lambda,
      );
      if sc_eval != eq_at_r * gate_sum {
        return Err(NovaError::InvalidSumcheckProof);
      }
      point = r;
    }

    // Bind this layer's claims, then draw the fold challenge (so the children
    // cannot be chosen after it), then fold to the next layer's claims/point.
    for fc in layer_finals {
      absorb_fraction::<E>(transcript, fc.left);
      absorb_fraction::<E>(transcript, fc.right);
    }
    let r_fold = transcript.squeeze(spec::FOLD)?;

    running = layer_finals
      .iter()
      .map(|fc| fc.fold_into_next_claim(r_fold))
      .collect();
    // The next layer's point prepends the fold challenge as the new top (MSB)
    // variable: point' = [r_fold, ...point].
    let mut next_point = Vec::with_capacity(point.len() + 1);
    next_point.push(r_fold);
    next_point.extend_from_slice(&point);
    point = next_point;
  }

  Ok(LogupGkrOpeningClaim::new(point, running))
}

#[cfg(test)]
mod tests {
  //! Independence tests: these build a valid proof **by hand** (from the tree
  //! fold in `layer.rs` plus the transcript spec in this module) with no use of
  //! the `prover` module, demonstrating that the verifier stands on its own.
  use super::*;
  use crate::spartan::logup_gkr::layer::Layer;
  use crate::spartan::logup_gkr::proof::{LayerFinalClaim, LogupGkrProof};
  use crate::spartan::polys::multilinear::MultilinearPolynomial;

  type E = crate::provider::Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  fn mle(v: Vec<u64>) -> MultilinearPolynomial<Fr> {
    MultilinearPolynomial::new(v.into_iter().map(Fr::from).collect())
  }

  // Hand-build the proof for a single-instance, 2-leaf tree (num_vars = 1):
  // only the root reduction (base case) runs — no sumcheck — so the whole proof
  // is constructible from `layer.rs` alone, following this module's spec.
  fn hand_built_2leaf(num: Vec<u64>, den: Vec<u64>) -> LogupGkrProof<E> {
    assert_eq!(num.len(), 2);
    let input = Layer::<E> {
      num: mle(num.clone()),
      den: mle(den.clone()),
    };
    let tree = input.build_tree(); // [input(1 var), root(0 var)]
    let (rn, rd) = tree[1].output_fraction();
    // Base layer split claim = the two child cells directly (nL,nR,dL,dR).
    let child = &tree[0];
    let final_claim = LayerFinalClaim::<E>::new(
      child.num.Z[0],
      child.num.Z[1],
      child.den.Z[0],
      child.den.Z[1],
    );
    LogupGkrProof {
      initial_claims: vec![LayerClaim::<E>::new(rn, rd)],
      final_claims: vec![vec![final_claim]],
      sumchecks: vec![],
    }
  }

  #[test]
  fn verifier_accepts_hand_built_valid_proof() {
    let proof = hand_built_2leaf(vec![3, 5], vec![7, 11]);
    let mut tr = <E as Engine>::TE::new(b"gkr-indep");
    let claim =
      verify::<E>(&proof, &mut tr).expect("verifier must accept a valid hand-built proof");
    // One instance, one variable → eval_point has length 1, one opening.
    assert_eq!(claim.eval_point().len(), 1);
    assert_eq!(claim.openings().len(), 1);
  }

  #[test]
  fn verifier_rejects_wrong_root() {
    let mut proof = hand_built_2leaf(vec![3, 5], vec![7, 11]);
    // Corrupt the root claim so it no longer equals the gate of its children.
    proof.initial_claims[0].num += Fr::from(1);
    let mut tr = <E as Engine>::TE::new(b"gkr-indep");
    assert!(verify::<E>(&proof, &mut tr).is_err());
  }

  #[test]
  fn verifier_rejects_malformed_shape() {
    let mut proof = hand_built_2leaf(vec![3, 5], vec![7, 11]);
    // sumchecks.len() + 1 must equal final_claims.len(); break it.
    proof
      .sumchecks
      .push(crate::spartan::logup_gkr::proof::LayerSumcheck {
        round_polys: vec![],
      });
    let mut tr = <E as Engine>::TE::new(b"gkr-indep");
    assert!(verify::<E>(&proof, &mut tr).is_err());
  }

  #[test]
  fn verifier_rejects_zero_zero_children_of_unit_root() {
    // P0 forgery fragment: root `(0,1)` with children `(0,0)`. Under
    // cross-multiplication `0·0 == 0·1` this would pass; exact equality
    // rejects it. gate = (0,0) ≠ (0,1).
    let zero = Fraction::new(Fr::ZERO, Fr::ZERO);
    let proof = LogupGkrProof {
      initial_claims: vec![Fraction::new(Fr::ZERO, Fr::ONE)],
      final_claims: vec![vec![LayerFinalClaim {
        left: zero,
        right: zero,
      }]],
      sumchecks: vec![],
    };
    let mut tr = <E as Engine>::TE::new(b"gkr-p0");
    assert!(
      verify::<E>(&proof, &mut tr).is_err(),
      "must reject (0,1) root with (0,0) children"
    );
  }

  #[test]
  fn verifier_rejects_scaled_root() {
    // Exact equality also rejects a nonzero scale of an otherwise valid gate
    // (cross-mult would accept).
    let mut proof = hand_built_2leaf(vec![3, 5], vec![7, 11]);
    proof.initial_claims[0].num *= Fr::from(2);
    proof.initial_claims[0].den *= Fr::from(2);
    let mut tr = <E as Engine>::TE::new(b"gkr-indep");
    assert!(verify::<E>(&proof, &mut tr).is_err());
  }
}
