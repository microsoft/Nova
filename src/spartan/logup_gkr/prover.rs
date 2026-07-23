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

/// Reduces one GKR layer: samples `λ`, either reads the two-cell split (root
/// reduction, `num_vars - j == 0`) or runs `prove_layer_sumcheck`, then absorbs
/// the split final claims, samples the fold challenge, and folds the running
/// claims and evaluation point into the child layer.
///
/// `children` are the popped child layers (`layers[j-1]`) whose halves the gate
/// consumes. This is the single per-layer body shared by the standalone prefix
/// loop and the standalone last-layer finish, so both drive an identical
/// transcript; the fused ppSNARK driver deliberately does NOT call it for the
/// last layer (it batches that layer with the Inner sumcheck instead).
#[allow(clippy::too_many_arguments)]
fn prove_one_reduction<E: Engine>(
  j: usize,
  num_vars: usize,
  m: usize,
  mut children: Vec<Layer<E>>,
  running: &mut Vec<(E::Scalar, E::Scalar)>,
  eval_point: &mut Vec<E::Scalar>,
  sumchecks: &mut Vec<LayerSumcheck<E>>,
  final_claims_by_layer: &mut Vec<Vec<LayerFinalClaim<E>>>,
  transcript: &mut E::TE,
) -> Result<(), NovaError> {
  // Fresh λ per layer, after the previous claims were absorbed.
  let lambda = transcript.squeeze(spec::LAMBDA)?;
  let n = 1usize << (num_vars - j);

  let mut layer_finals: Vec<LayerFinalClaim<E>> = Vec::with_capacity(m);
  if num_vars - j == 0 {
    // Base case (root reduction): the child layer has exactly two cells; read
    // the split directly, no sumcheck.
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
      for (p, q) in running.iter() {
        acc += pw * *p;
        pw *= lambda;
        acc += pw * *q;
        pw *= lambda;
      }
      acc
    };

    // Half-MLEs struct-of-arrays: nL/nR = numerator halves, dL/dR = den halves.
    // Move the child buffers into the halves (split each Z at n) instead of
    // cloning: `children` is dropped at the end of this call anyway.
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
      eval_point,
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
    *eval_point = r; // sumcheck point (length num_vars - j)
  }

  // Absorb final claims, sample fold challenge, update running claims + point.
  for fc in &layer_finals {
    absorb_fraction::<E>(transcript, fc.left);
    absorb_fraction::<E>(transcript, fc.right);
  }
  let fold_r = transcript.squeeze(spec::FOLD)?;

  *running = layer_finals
    .iter()
    .map(|fc| {
      let c = fc.fold_into_next_claim(fold_r);
      (c.num, c.den)
    })
    .collect();
  // New point for layers[j-1] is [fold_r, ...sumcheck_point] (fold_r as MSB).
  let mut next_point = Vec::with_capacity(eval_point.len() + 1);
  next_point.push(fold_r);
  next_point.extend_from_slice(eval_point);
  *eval_point = next_point;

  final_claims_by_layer.push(layer_finals);
  Ok(())
}

/// The prover state captured after the GKR prefix (every layer reduction except
/// the last), holding everything needed to finish the argument — either the
/// standalone last layer, or the fused ppSNARK driver that batches the last
/// layer with the Inner sumcheck. No hidden transcript side effects: all data to
/// resume is explicit here.
pub(crate) struct GkrProverPrefix<E: Engine> {
  /// Number of instances.
  pub(crate) m: usize,
  /// `log2` height of every input layer.
  pub(crate) num_vars: usize,
  /// Per-instance output-layer (root) fractions, already absorbed.
  pub(crate) initial_claims: Vec<LayerClaim<E>>,
  /// Prefix layer sumchecks produced so far, output→input.
  pub(crate) sumchecks: Vec<LayerSumcheck<E>>,
  /// Prefix per-layer split final claims produced so far, output→input.
  pub(crate) final_claims_by_layer: Vec<Vec<LayerFinalClaim<E>>>,
  /// Running per-instance `(num, den)` claim about the last layer's parent
  /// (`layers[1]`), or the root claim when `num_vars == 1`.
  pub(crate) running: Vec<(E::Scalar, E::Scalar)>,
  /// Evaluation point `τ` for the last-layer sumcheck (length `num_vars - 1`;
  /// empty when `num_vars == 1`).
  pub(crate) eval_point: Vec<E::Scalar>,
  /// The remaining input layer per instance, consumed by the last reduction.
  pub(crate) input_layers: Vec<Layer<E>>,
}

/// Runs the GKR prefix: builds the trees, absorbs the root claims, and reduces
/// every layer **except the last**, returning the state needed to finish. For
/// `num_vars == 1` the prefix is empty (the single layer is the last one).
///
/// See [`prove`] for the full-argument contract; splitting the prefix out lets
/// the ppSNARK fused driver take over the last layer.
pub(crate) fn prove_prefix<E: Engine>(
  inputs: Vec<Layer<E>>,
  transcript: &mut E::TE,
) -> Result<GkrProverPrefix<E>, NovaError> {
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
  // Layers are consumed root→leaf via `pop()` (see the loop), so the just-proven
  // layer is dropped the instant it is consumed.
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

  // Running per-instance claims about `layers[j]`, starting at the root.
  let mut running: Vec<(E::Scalar, E::Scalar)> =
    initial_claims.iter().map(|c| (c.num, c.den)).collect();
  let mut eval_point: Vec<E::Scalar> = Vec::new();

  let mut sumchecks: Vec<LayerSumcheck<E>> = Vec::with_capacity(num_vars.saturating_sub(1));
  let mut final_claims_by_layer: Vec<Vec<LayerFinalClaim<E>>> = Vec::with_capacity(num_vars);

  // Reduce every layer except the last (j = num_vars … 2). For num_vars == 1
  // this range is empty and the single layer is finished by the caller.
  for j in (2..=num_vars).rev() {
    let children: Vec<Layer<E>> = trees.iter_mut().map(|t| t.pop().unwrap()).collect();
    prove_one_reduction::<E>(
      j,
      num_vars,
      m,
      children,
      &mut running,
      &mut eval_point,
      &mut sumchecks,
      &mut final_claims_by_layer,
      transcript,
    )?;
  }

  // The last remaining layer of each tree is the input layer, consumed by the
  // last reduction (standalone finish or fused driver).
  let input_layers: Vec<Layer<E>> = trees
    .iter_mut()
    .map(|t| t.pop().expect("input layer remains"))
    .collect();

  Ok(GkrProverPrefix {
    m,
    num_vars,
    initial_claims,
    sumchecks,
    final_claims_by_layer,
    running,
    eval_point,
    input_layers,
  })
}

/// Finishes the standalone argument from the prefix state by reducing the last
/// layer with the same `prove_one_reduction` body the prefix uses, then
/// assembles the batched proof and shared opening claim.
pub(crate) fn finish_last_layer_standalone<E: Engine>(
  prefix: GkrProverPrefix<E>,
  transcript: &mut E::TE,
) -> Result<(LogupGkrProof<E>, LogupGkrOpeningClaim<E>), NovaError> {
  let GkrProverPrefix {
    m,
    num_vars,
    initial_claims,
    mut sumchecks,
    mut final_claims_by_layer,
    mut running,
    mut eval_point,
    input_layers,
  } = prefix;

  // The last reduction (j = 1) consumes the input layers.
  prove_one_reduction::<E>(
    1,
    num_vars,
    m,
    input_layers,
    &mut running,
    &mut eval_point,
    &mut sumchecks,
    &mut final_claims_by_layer,
    transcript,
  )?;

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

/// Proves the fractional-sum identity `Σ p/q = root` for all equal-height input
/// trees in one batched proof, returning the proof and the shared opening claim.
///
/// `inputs` holds one input [`Layer`] per instance. Every layer must have the
/// same positive `num_vars`, so every coefficient vector has the same
/// power-of-two length of at least two. The soundness invariant is to absorb
/// every root and layer claim before sampling the challenge that depends on it;
/// `initial_claims` and each layer's `final_claims` enforce this order.
///
/// This is the prefix + last-layer composition; the two stages share the
/// per-layer body `prove_one_reduction`, so the transcript is identical to the
/// former monolithic loop.
pub fn prove<E: Engine>(
  inputs: Vec<Layer<E>>,
  transcript: &mut E::TE,
) -> Result<(LogupGkrProof<E>, LogupGkrOpeningClaim<E>), NovaError> {
  let prefix = prove_prefix::<E>(inputs, transcript)?;
  finish_last_layer_standalone::<E>(prefix, transcript)
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

  // ---- Gate A: staged-refactor byte-equality ----
  //
  // Pre-refactor monolithic prover, kept verbatim so the staged
  // `prove_prefix` + `finish_last_layer_standalone` split can be proven to
  // reproduce the exact same transcript, proof bytes and opening. Delete once
  // the staged equivalence is trusted (see PLAN section 17.4).
  fn prove_reference<E2: Engine>(
    inputs: Vec<Layer<E2>>,
    transcript: &mut E2::TE,
  ) -> Result<(LogupGkrProof<E2>, LogupGkrOpeningClaim<E2>), NovaError> {
    let m = inputs.len();
    if m == 0 {
      return Err(NovaError::InvalidNumInstances);
    }
    let num_vars = inputs[0].num_vars();
    if num_vars == 0 {
      return Err(NovaError::InvalidSumcheckProof);
    }
    for inp in &inputs {
      if inp.num_vars() != num_vars {
        return Err(NovaError::InvalidSumcheckProof);
      }
    }
    let mut trees: Vec<Vec<Layer<E2>>> = inputs.into_iter().map(|l| l.build_tree()).collect();
    let initial_claims: Vec<LayerClaim<E2>> = trees
      .iter_mut()
      .map(|t| {
        let root = t.pop().expect("tree has a root layer");
        let (n, d) = root.output_fraction();
        LayerClaim::<E2>::new(n, d)
      })
      .collect();
    for c in &initial_claims {
      absorb_fraction::<E2>(transcript, *c);
    }
    let mut running: Vec<(E2::Scalar, E2::Scalar)> =
      initial_claims.iter().map(|c| (c.num, c.den)).collect();
    let mut eval_point: Vec<E2::Scalar> = Vec::new();
    let mut sumchecks: Vec<LayerSumcheck<E2>> = Vec::with_capacity(num_vars.saturating_sub(1));
    let mut final_claims_by_layer: Vec<Vec<LayerFinalClaim<E2>>> = Vec::with_capacity(num_vars);
    for j in (1..=num_vars).rev() {
      let lambda = transcript.squeeze(spec::LAMBDA)?;
      let mut children: Vec<Layer<E2>> = trees.iter_mut().map(|t| t.pop().unwrap()).collect();
      let child_len = 1usize << (num_vars - j + 1);
      let n = child_len / 2;
      let mut layer_finals: Vec<LayerFinalClaim<E2>> = Vec::with_capacity(m);
      if num_vars - j == 0 {
        for c in &children {
          layer_finals.push(LayerFinalClaim::<E2>::new(
            c.num.Z[0], c.num.Z[1], c.den.Z[0], c.den.Z[1],
          ));
        }
        let _ = (lambda, n);
      } else {
        let previous_finals = final_claims_by_layer
          .last()
          .expect("every sumcheck layer follows a completed parent reduction");
        let claim: E2::Scalar = {
          let mut acc = E2::Scalar::ZERO;
          let mut pw = E2::Scalar::ONE;
          for (p, q) in &running {
            acc += pw * *p;
            pw *= lambda;
            acc += pw * *q;
            pw *= lambda;
          }
          acc
        };
        let mut nl: Vec<MultilinearPolynomial<E2::Scalar>> = Vec::with_capacity(m);
        let mut nr: Vec<MultilinearPolynomial<E2::Scalar>> = Vec::with_capacity(m);
        let mut dl: Vec<MultilinearPolynomial<E2::Scalar>> = Vec::with_capacity(m);
        let mut dr: Vec<MultilinearPolynomial<E2::Scalar>> = Vec::with_capacity(m);
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
        let (round_polys, r, finals) = prove_layer_sumcheck::<E2>(
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
          layer_finals.push(LayerFinalClaim::<E2>::new(f[0], f[1], f[2], f[3]));
        }
        sumchecks.push(LayerSumcheck { round_polys });
        eval_point = r;
      }
      for fc in &layer_finals {
        absorb_fraction::<E2>(transcript, fc.left);
        absorb_fraction::<E2>(transcript, fc.right);
      }
      let fold_r = transcript.squeeze(spec::FOLD)?;
      running = layer_finals
        .iter()
        .map(|fc| {
          let c = fc.fold_into_next_claim(fold_r);
          (c.num, c.den)
        })
        .collect();
      let mut next_point = Vec::with_capacity(eval_point.len() + 1);
      next_point.push(fold_r);
      next_point.extend_from_slice(&eval_point);
      eval_point = next_point;
      final_claims_by_layer.push(layer_finals);
    }
    let openings: Vec<LayerClaim<E2>> = running
      .iter()
      .map(|(n, d)| LayerClaim::<E2>::new(*n, *d))
      .collect();
    let proof = LogupGkrProof {
      initial_claims,
      final_claims: final_claims_by_layer,
      sumchecks,
    };
    let claim = LogupGkrOpeningClaim::new(eval_point, openings);
    Ok((proof, claim))
  }

  // Instance sets spanning num_vars = 1..5 and m = 1, 2, 4 (both eq-factor
  // branches and both the num_vars == 1 base case and the sumcheck path).
  fn sample_sets() -> Vec<Vec<(Vec<u64>, Vec<u64>)>> {
    let gen = |seed: u64, len: usize| -> (Vec<u64>, Vec<u64>) {
      let mut s = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
      let mut next = || {
        s = s
          .wrapping_mul(6364136223846793005)
          .wrapping_add(1442695040888963407);
        (s >> 33) | 1
      };
      let num = (0..len).map(|_| next()).collect();
      let den = (0..len).map(|_| next()).collect();
      (num, den)
    };
    vec![
      vec![(vec![3, 5], vec![7, 11])],            // num_vars = 1, m = 1
      vec![(vec![1, 2, 3, 4], vec![5, 6, 7, 8])], // num_vars = 2, m = 1
      vec![gen(1, 8)],                            // num_vars = 3, m = 1
      vec![gen(2, 4), gen(3, 4)],                 // num_vars = 2, m = 2
      vec![gen(4, 16), gen(5, 16), gen(6, 16), gen(7, 16)], // num_vars = 4, m = 4
      vec![gen(8, 32), gen(9, 32), gen(10, 32), gen(11, 32)], // num_vars = 5, m = 4
    ]
  }

  fn to_layers(set: &[(Vec<u64>, Vec<u64>)]) -> Vec<Layer<E>> {
    set
      .iter()
      .map(|(n, d)| Layer::<E> {
        num: mle(n.clone()),
        den: mle(d.clone()),
      })
      .collect()
  }

  // Gate A (prover): the staged `prove` must byte-for-byte reproduce the
  // pre-refactor monolithic prover — same proof bytes, same opening, same
  // post-proof transcript sentinel — on every sample instance set.
  #[test]
  fn staged_prove_matches_reference() {
    let cfg = bincode::config::standard();
    for set in sample_sets() {
      let mut tr_new = <E as Engine>::TE::new(b"gkr-gate-a");
      let (proof_new, claim_new) = prove::<E>(to_layers(&set), &mut tr_new).expect("staged prove");
      let sentinel_new: Fr = tr_new.squeeze(b"sentinel").expect("sentinel");

      let mut tr_ref = <E as Engine>::TE::new(b"gkr-gate-a");
      let (proof_ref, claim_ref) =
        prove_reference::<E>(to_layers(&set), &mut tr_ref).expect("reference prove");
      let sentinel_ref: Fr = tr_ref.squeeze(b"sentinel").expect("sentinel");

      let bytes_new = bincode::serde::encode_to_vec(&proof_new, cfg).expect("encode new");
      let bytes_ref = bincode::serde::encode_to_vec(&proof_ref, cfg).expect("encode ref");
      assert_eq!(bytes_new, bytes_ref, "proof bytes differ from reference");
      assert_eq!(
        claim_new.eval_point(),
        claim_ref.eval_point(),
        "opening eval_point differs"
      );
      assert_eq!(
        claim_new.openings(),
        claim_ref.openings(),
        "opening fractions differ"
      );
      assert_eq!(
        sentinel_new, sentinel_ref,
        "post-proof transcript sentinel differs"
      );
    }
  }
}
