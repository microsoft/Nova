//! Prover for the Logup-GKR fractional-sum argument.
//!
//! Implements the prover side of the Logup-GKR protocol defined in
//! `verifier.rs`, sharing its transcript labels and fraction-absorption order.
//!
//! It builds one equal-height tree per input, folds them leaf→root, and for each
//! internal depth runs one transparent cubic sumcheck
//! (`prove_layer_sumcheck`) reducing the merged fraction-sum claim to the next
//! layer.
//!
//! ## Per-layer gate (the verifier's contract)
//! A layer proves, for all instances batched by a per-layer `λ`,
//! `Σ_i (v_p_i + λ v_q_i) = Σ_x eq(τ,x) · Σ_i [nL·dR + nR·dL + λ·dL·dR]_i`.
//! The `eq(τ, ·)` factor is carried as an explicit MLE so the sumcheck's final
//! value reconciles as the verifier demands: `eq(τ,r)·G(r)`. Round polynomials
//! are degree 3 (`eq` deg 1 × gate deg 2 = `spec::LAYER_SC_DEGREE`).
//!
//! ## Endianness
//! The tree folds MSB-first (`Layer::fold_up`, halves `i`/`i+n`), and the
//! sumcheck binds the top variable first, so the sumcheck point is directly the
//! fraction evaluation point of the next layer.

use crate::errors::NovaError;
use crate::spartan::logup_gkr::layer::Layer;
use crate::spartan::logup_gkr::proof::{
  LayerClaim, LayerFinalClaim, LayerSumcheck, LogupGkrOpeningClaim, LogupGkrProof,
};
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::traits::{Engine, TranscriptEngineTrait};
use ff::Field;
use rayon::prelude::*;

// The prover satisfies the protocol defined by the verifier: it uses the
// verifier's transcript labels (`spec`) and fraction-absorb order, and produces
// a sumcheck whose final value reconciles as `eq(τ,r)·G(r)` (the verifier's
// contract).
use crate::spartan::logup_gkr::verifier::{absorb_fraction, spec};

/// Transparent cubic sumcheck for one GKR layer, proving
/// `claim = Σ_x eq(τ, x) · Σ_i [ nLᵢ·dRᵢ + nRᵢ·dLᵢ + λ·dLᵢ·dRᵢ ](x)`
/// over `num_rounds = |τ|` variables.
///
/// The `eq(τ, ·)` factor is handled by the shared `EqSumCheckInstance`
/// (Gruen eq-factoring, eprint 2024/108: half-size eq tables + O(1) per-round
/// bind), and BDDT claim derivation (eprint 2025/1117 §6.2). The four-instance
/// path reuses the preceding layer's left claims for round-0 `t(0)`, so that
/// round scans only for `t(∞)`; later rounds normally scan for both values.
///
/// The four half-MLEs are passed struct-of-arrays (`nl`/`nr`/`dl`/`dr`, one
/// entry per instance) so the eq instance can index them directly. Returns the
/// compressed round polynomials, the sumcheck point `r`, and each instance's
/// `(nL, nR, dL, dR)` evaluated at `r`.
///
/// ppSNARK's four-sub-instance path caches each top-variable delta during
/// evaluation and reuses it during binding; other instance counts retain the
/// general non-mutating evaluation path. `previous_finals` must be the preceding
/// parent reduction's claims in the same instance order as the four MLE slices.
#[allow(clippy::type_complexity)]
fn prove_layer_sumcheck<E: Engine>(
  claim: E::Scalar,
  taus: &[E::Scalar],
  nl: &mut [MultilinearPolynomial<E::Scalar>],
  nr: &mut [MultilinearPolynomial<E::Scalar>],
  dl: &mut [MultilinearPolynomial<E::Scalar>],
  dr: &mut [MultilinearPolynomial<E::Scalar>],
  lambda: E::Scalar,
  previous_finals: &[LayerFinalClaim<E>],
  transcript: &mut E::TE,
) -> Result<
  (
    Vec<crate::spartan::polys::univariate::CompressedUniPoly<E::Scalar>>,
    Vec<E::Scalar>,
    Vec<[E::Scalar; 4]>,
  ),
  NovaError,
> {
  use crate::spartan::polys::univariate::{CompressedUniPoly, UniPoly};
  use crate::spartan::sumcheck::eq_sumcheck::EqSumCheckInstance;

  let num_rounds = taus.len();
  let m = nl.len();

  let mut r: Vec<E::Scalar> = Vec::with_capacity(num_rounds);
  let mut polys: Vec<CompressedUniPoly<E::Scalar>> = Vec::with_capacity(num_rounds);
  let mut claim_per_round = claim;

  let mut eq = EqSumCheckInstance::<E>::new(taus.to_vec());
  let cache_deltas = m == 4;

  // Per-instance λ-powers: instance i's numerator gets λ^{2i}, denominator
  // λ^{2i+1}. λ^{2i} accumulates by a running product (O(m) muls) instead of
  // per-instance pow_vartime.
  let mut w_num = E::Scalar::ONE;
  let weights: Vec<(E::Scalar, E::Scalar)> = (0..m)
    .map(|_| {
      let w_den = w_num * lambda;
      let pair = (w_num, w_den);
      w_num = w_den * lambda;
      pair
    })
    .collect();

  // The preceding layer's left claims are this layer's round-0 t(0)
  // evaluations at `taus[1..]`.
  let round0_t_0 = cache_deltas.then(|| {
    assert_eq!(previous_finals.len(), m);
    previous_finals
      .iter()
      .zip(&weights)
      .map(|(fc, (w_num, w_den))| *w_num * fc.left.num + *w_den * fc.left.den)
      .sum()
  });

  for round in 0..num_rounds {
    // Round polynomial s(X) = eq(τ,X) · G(X), degree 3. BDDT derivation returns
    // (s(0), cubic coeff, s(-1)); the verifier reconstructs s(1) = claim - s(0).
    let (s_0, s_cubic, s_m1) = if cache_deltas {
      match (round, round0_t_0) {
        (0, Some(t_0)) => eq.evaluation_points_logup_gate_and_cache_deltas_with_t_0(
          nl,
          nr,
          dl,
          dr,
          &weights,
          t_0,
          claim_per_round,
        ),
        _ => eq.evaluation_points_logup_gate_and_cache_deltas(
          nl,
          nr,
          dl,
          dr,
          &weights,
          claim_per_round,
        ),
      }
    } else {
      eq.evaluation_points_logup_gate(nl, nr, dl, dr, &weights, claim_per_round)
    };

    let poly = UniPoly::from_evals_deg3(&[s_0, claim_per_round - s_0, s_cubic, s_m1]);

    transcript.absorb(spec::ROUND_POLY, &poly);
    let r_i = transcript.squeeze(spec::ROUND_CHALLENGE)?;
    r.push(r_i);
    polys.push(poly.compress());
    claim_per_round = poly.evaluate(&r_i);

    // Bind the top variable of every half-MLE (eq binds in O(1) via the instance).
    nl.par_iter_mut()
      .chain(nr.par_iter_mut())
      .chain(dl.par_iter_mut())
      .chain(dr.par_iter_mut())
      .for_each(|p| {
        if cache_deltas {
          p.bind_poly_var_top_with_cached_delta(&r_i);
        } else {
          p.bind_poly_var_top(&r_i);
        }
      });
    eq.bound(&r_i);
  }

  let _ = claim_per_round;
  let finals: Vec<[E::Scalar; 4]> = (0..m)
    .map(|i| [nl[i].Z[0], nr[i].Z[0], dl[i].Z[0], dr[i].Z[0]])
    .collect();
  Ok((polys, r, finals))
}

/// Proves the fractional-sum identity `Σ p/q = root` for all equal-height input
/// trees in one batched proof, returning the proof and the shared opening claim.
///
/// `inputs` holds one input [`Layer`] per instance. Every layer must have the
/// same positive `num_vars`, so every coefficient vector has the same
/// power-of-two length of at least two. The soundness invariant is to absorb
/// every root and layer claim before sampling the challenge that depends on it;
/// `initial_claims` and each layer's `final_claims` enforce this order.
pub fn prove<E: Engine>(
  inputs: Vec<Layer<E>>,
  transcript: &mut E::TE,
) -> Result<(LogupGkrProof<E>, LogupGkrOpeningClaim<E>), NovaError> {
  let m = inputs.len();
  if m == 0 {
    return Err(NovaError::InvalidNumInstances);
  }
  let num_vars = inputs[0].num_vars();
  if num_vars == 0 {
    // A single-cell input has nothing to fold; the argument needs height ≥ 2.
    return Err(NovaError::InvalidSumcheckProof);
  }
  for inp in &inputs {
    if inp.num_vars() != num_vars {
      return Err(NovaError::InvalidSumcheckProof);
    }
  }

  // Build every instance's tree, leaf→root. trees[instance][layer], where
  // trees[i][j] has (num_vars - j) variables; [0] = input, [num_vars] = root.
  //
  // Memory: layers are consumed root→leaf (step j reads layer j-1, for
  // j = num_vars … 1), which is exactly descending index. So each tree is drained
  // via `pop()` — the just-proven layer is dropped the instant it is consumed,
  // instead of every layer staying live for the whole proof. Peak during the
  // sumcheck loop drops from ~3N to ~N per instance (the build itself is still
  // bottom-up, so the transient right after `build_tree` is unavoidable).
  let mut trees: Vec<Vec<Layer<E>>> = inputs.into_iter().map(|l| l.build_tree()).collect();

  // initial_claims = each instance's output (root) fraction, absorbed first. The
  // root layer (index num_vars, a single cell) is popped and dropped here; only
  // its two scalars survive in `initial_claims`.
  let initial_claims: Vec<LayerClaim<E>> = trees
    .iter_mut()
    .map(|t| {
      let root = t.pop().expect("tree has a root layer");
      let (n, d) = root.output_fraction();
      LayerClaim::<E>::new(n, d)
    })
    .collect();
  for c in &initial_claims {
    absorb_fraction::<E>(transcript, *c);
  }

  // Running per-instance claims (v_p_i, v_q_i) about `layers[j]` at `eval_point`.
  // Start at the root (j = num_vars, empty point).
  let mut running: Vec<(E::Scalar, E::Scalar)> =
    initial_claims.iter().map(|c| (c.num, c.den)).collect();
  let mut eval_point: Vec<E::Scalar> = Vec::new();

  // Ordered output→input as we go: step j reduces a claim about layers[j] to
  // layers[j-1], for j = num_vars, num_vars-1, ..., 1.
  let mut sumchecks: Vec<LayerSumcheck<E>> = Vec::with_capacity(num_vars.saturating_sub(1));
  let mut final_claims_by_layer: Vec<Vec<LayerFinalClaim<E>>> = Vec::with_capacity(num_vars);

  for j in (1..=num_vars).rev() {
    // Fresh λ per layer, after the previous claims were absorbed.
    let lambda = transcript.squeeze(spec::LAMBDA)?;

    // Children are the two halves of layers[j-1] (which has num_vars-j+1 vars,
    // so each half has num_vars-j vars). nL/nR = num halves, dL/dR = den halves.
    // Pop layer j-1 off each tree: the loop consumes layers strictly root→leaf
    // (descending index), so once popped, layer j-1 is never read again and its
    // buffers can be moved into the sumcheck (and dropped at end of iteration).
    let mut children: Vec<Layer<E>> = trees.iter_mut().map(|t| t.pop().unwrap()).collect();
    let child_len = 1usize << (num_vars - j + 1);
    let n = child_len / 2;

    // final claims (nL, nR, dL, dR) per instance for this layer.
    let mut layer_finals: Vec<LayerFinalClaim<E>> = Vec::with_capacity(m);

    if num_vars - j == 0 {
      // Base case (j = num_vars, root reduction): the child layer has exactly
      // two cells; read the split directly, no sumcheck.
      for c in &children {
        layer_finals.push(LayerFinalClaim::<E>::new(
          c.num.Z[0], // nL
          c.num.Z[1], // nR
          c.den.Z[0], // dL
          c.den.Z[1], // dR
        ));
      }
      let _ = (lambda, n); // λ unused at the base (single term, no batching)
    } else {
      // Sumcheck over (num_vars - j) variables. The 2m sub-claims are batched by
      // distinct powers of λ (Horner over [p_0,q_0,p_1,q_1,...]) — see verifier.
      let previous_finals = final_claims_by_layer
        .last()
        .expect("every sumcheck layer follows a completed parent reduction");
      let claim: E::Scalar = {
        let mut acc = E::Scalar::ZERO;
        let mut pw = E::Scalar::ONE;
        for (p, q) in &running {
          acc += pw * *p;
          pw *= lambda;
          acc += pw * *q;
          pw *= lambda;
        }
        acc
      };

      // Half-MLEs struct-of-arrays: nL/nR = numerator halves, dL/dR = den halves.
      // Move the child buffers into the halves (split each Z at n) instead of
      // cloning: `children` is dropped at the end of this iteration anyway.
      let mut nl: Vec<MultilinearPolynomial<E::Scalar>> = Vec::with_capacity(m);
      let mut nr: Vec<MultilinearPolynomial<E::Scalar>> = Vec::with_capacity(m);
      let mut dl: Vec<MultilinearPolynomial<E::Scalar>> = Vec::with_capacity(m);
      let mut dr: Vec<MultilinearPolynomial<E::Scalar>> = Vec::with_capacity(m);
      for c in children.drain(..) {
        let mut num_z = c.num.Z;
        let mut den_z = c.den.Z;
        let num_hi = num_z.split_off(n);
        let den_hi = den_z.split_off(n);
        nl.push(MultilinearPolynomial::new(num_z));
        nr.push(MultilinearPolynomial::new(num_hi));
        dl.push(MultilinearPolynomial::new(den_z));
        dr.push(MultilinearPolynomial::new(den_hi));
      }

      let (round_polys, r, finals) = prove_layer_sumcheck::<E>(
        claim,
        &eval_point,
        &mut nl,
        &mut nr,
        &mut dl,
        &mut dr,
        lambda,
        previous_finals,
        transcript,
      )?;

      for f in &finals {
        layer_finals.push(LayerFinalClaim::<E>::new(f[0], f[1], f[2], f[3]));
      }
      sumchecks.push(LayerSumcheck { round_polys });
      eval_point = r; // sumcheck point (length num_vars - j)
    }

    // Absorb final claims, sample fold challenge, update running claims + point.
    for fc in &layer_finals {
      absorb_fraction::<E>(transcript, fc.left);
      absorb_fraction::<E>(transcript, fc.right);
    }
    let fold_r = transcript.squeeze(spec::FOLD)?;

    running = layer_finals
      .iter()
      .map(|fc| {
        let c = fc.fold_into_next_claim(fold_r);
        (c.num, c.den)
      })
      .collect();
    // New point for layers[j-1] is [fold_r, ...sumcheck_point] (fold_r as MSB).
    let mut next_point = Vec::with_capacity(eval_point.len() + 1);
    next_point.push(fold_r);
    next_point.extend_from_slice(&eval_point);
    eval_point = next_point;

    final_claims_by_layer.push(layer_finals);
  }

  // Proof is ordered output→input (the order we produced).
  // openings = each instance's input-layer fraction at the final eval_point
  // (length num_vars); equals the last running claim by construction.
  let openings: Vec<LayerClaim<E>> = running
    .iter()
    .map(|(n, d)| LayerClaim::<E>::new(*n, *d))
    .collect();

  let proof = LogupGkrProof {
    initial_claims,
    final_claims: final_claims_by_layer,
    sumchecks,
  };
  let claim = LogupGkrOpeningClaim::new(eval_point, openings);
  Ok((proof, claim))
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::spartan::logup_gkr::verifier;
  use crate::spartan::polys::multilinear::MultilinearPolynomial;
  use crate::traits::TranscriptEngineTrait;

  type E = crate::provider::Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  fn mle(v: Vec<u64>) -> MultilinearPolynomial<Fr> {
    MultilinearPolynomial::new(v.into_iter().map(Fr::from).collect())
  }

  // Full prove→verify round trip: the verifier must accept, and its returned
  // openings must equal the prover's actual input-layer fractions at the shared
  // evaluation point.
  fn round_trip(inputs: Vec<(Vec<u64>, Vec<u64>)>) {
    let layers: Vec<Layer<E>> = inputs
      .iter()
      .map(|(n, d)| Layer::<E> {
        num: mle(n.clone()),
        den: mle(d.clone()),
      })
      .collect();
    // Keep copies to independently evaluate at the eval_point.
    let raw: Vec<(MultilinearPolynomial<Fr>, MultilinearPolynomial<Fr>)> = inputs
      .iter()
      .map(|(n, d)| (mle(n.clone()), mle(d.clone())))
      .collect();

    let mut tr_p = <E as Engine>::TE::new(b"gkr-test");
    let (proof, claim) = prove::<E>(layers, &mut tr_p).expect("prove");

    let mut tr_v = <E as Engine>::TE::new(b"gkr-test");
    let vclaim = verifier::verify::<E>(&proof, &mut tr_v).expect("verify");

    // The verifier point must match the prover point; opening components must
    // match direct input-MLE evaluations there.
    assert_eq!(
      vclaim.eval_point(),
      claim.eval_point(),
      "eval_point mismatch"
    );
    let pt = vclaim.eval_point();
    for (i, (n, d)) in raw.iter().enumerate() {
      let en = n.evaluate(pt);
      let ed = d.evaluate(pt);
      let op = vclaim.openings()[i];
      assert_eq!(op.num, en, "opening[{i}] numerator mismatch");
      assert_eq!(op.den, ed, "opening[{i}] denominator mismatch");
    }
  }

  // Full prove -> verify round trips over the transparent-eq layer sumcheck.
  #[test]
  fn round_trip_single_instance_n4() {
    round_trip(vec![(vec![1, 2, 3, 4], vec![5, 6, 7, 8])]);
  }

  #[test]
  fn round_trip_two_instances_n4() {
    round_trip(vec![
      (vec![1, 2, 3, 4], vec![5, 6, 7, 8]),
      (vec![9, 8, 7, 6], vec![2, 3, 4, 5]),
    ]);
  }

  #[test]
  fn round_trip_two_instances_n8() {
    round_trip(vec![
      (vec![3, 1, 4, 1, 5, 9, 2, 6], vec![2, 7, 1, 8, 2, 8, 1, 8]),
      (vec![1, 1, 1, 1, 1, 1, 1, 1], vec![3, 1, 4, 1, 5, 9, 2, 6]),
    ]);
  }

  // Exercises cached sumchecks with 1-4 rounds and both eq-factor branches.
  #[test]
  fn round_trip_four_instances_n32() {
    let gen = |seed: u64| -> (Vec<u64>, Vec<u64>) {
      let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
      let mut next = || {
        s = s
          .wrapping_mul(6364136223846793005)
          .wrapping_add(1442695040888963407);
        (s >> 33) | 1
      };
      let num = (0..32).map(|_| next()).collect();
      let den = (0..32).map(|_| next()).collect();
      (num, den)
    };
    round_trip(vec![gen(5), gen(6), gen(7), gen(8)]);
  }

  #[test]
  fn verify_rejects_tampered_final_claim() {
    let layers = vec![Layer::<E> {
      num: mle(vec![1, 2, 3, 4]),
      den: mle(vec![5, 6, 7, 8]),
    }];
    let mut tr_p = <E as Engine>::TE::new(b"gkr-test");
    let (mut proof, _) = prove::<E>(layers, &mut tr_p).unwrap();
    // Corrupt one final claim.
    proof.final_claims[0][0].left.num += Fr::from(1);
    let mut tr_v = <E as Engine>::TE::new(b"gkr-test");
    assert!(verifier::verify::<E>(&proof, &mut tr_v).is_err());
  }

  // Tamper-rejection with m >= 2: mutating per-instance final claims makes the
  // verifier reject (caught by transcript binding + the gate's bilinearity).
  #[test]
  fn verify_rejects_complementary_instance_offset() {
    let inputs = [
      (vec![1u64, 2, 3, 4], vec![5u64, 6, 7, 8]),
      (vec![9u64, 8, 7, 6], vec![2u64, 3, 4, 5]),
    ];
    let layers: Vec<Layer<E>> = inputs
      .iter()
      .map(|(n, d)| Layer::<E> {
        num: mle(n.clone()),
        den: mle(d.clone()),
      })
      .collect();
    let mut tr_p = <E as Engine>::TE::new(b"gkr-test");
    let (proof_ok, _) = prove::<E>(layers, &mut tr_p).unwrap();

    // Sanity: the honest proof verifies.
    let mut tr_v0 = <E as Engine>::TE::new(b"gkr-test");
    assert!(verifier::verify::<E>(&proof_ok, &mut tr_v0).is_ok());

    // Target a SUMCHECK layer (final_claims index >= 1; index 0 is the base
    // case whose per-instance cross-mult check is not batched). For num_vars=2,
    // index 1 is the input-layer reduction. Apply a complementary offset to the
    // two instances' left numerators.
    let mut proof = proof_ok.clone();
    assert!(proof.final_claims.len() >= 2 && proof.final_claims[1].len() == 2);
    let delta = Fr::from(7);
    proof.final_claims[1][0].left.num += delta;
    proof.final_claims[1][1].left.num -= delta;

    let mut tr_v = <E as Engine>::TE::new(b"gkr-test");
    assert!(
      verifier::verify::<E>(&proof, &mut tr_v).is_err(),
      "tampered per-instance final claims must be rejected"
    );
  }
}
