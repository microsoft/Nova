//! Layer type for the Logup-GKR fractional-sum tree.
//!
//! One [`Layer`] is a single level of a tree: two multilinear polynomials
//! (numerator, denominator). The same type serves the input layer, every
//! internal layer, and the output layer — they differ only in height (the
//! coefficient length halves each level, `N → N/2 → … → 1`). The same `Layer`
//! type serves all levels, with no separate input/output variants.
//!
//! ppSNARK constructs separate table and access input layers for each of its row
//! and column relations. Their leaves are `(num, den) = (ts, T+r)` on a table
//! side and `(-1, W+r)` on an access side.
//!
//! # Endianness (critical): MSB-first, pairs are `i` and `i+n`
//! Nova's MLEs are **MSB-first** (`bind_poly_var_top` folds the top variable by
//! `split_at(len/2)`). So a level's two children are the halves `x[i]` (top bit
//! 0) and `x[i+n]` (top bit 1), `n = len/2` — NOT `x[2i]` / `x[2i+1]`. Folding
//! this way keeps the tree consistent with Nova's sumcheck round order
//! (`eval_point` challenges bind MSB→LSB), so the GKR point matches the sumcheck
//! point.

use crate::spartan::logup_gkr::fraction::Fraction;
use crate::spartan::polys::multilinear::MultilinearPolynomial;
use crate::spartan::polys::multilinear::MultilinearPolynomial as MLE;
use crate::traits::Engine;
use rayon::prelude::*;

/// Parallelize `fold_up` only above this many output cells. A fold cell is a
/// few field multiplications, so rayon's scheduling overhead dominates on
/// smaller layers. This threshold is intentionally higher than the crate's
/// `PARALLEL_THRESHOLD`, which is tuned for more expensive per-element curve
/// operations such as MSMs.
const FOLD_PARALLEL_THRESHOLD: usize = 1 << 16;

/// One level of a fractional-sum tree: parallel numerator and denominator
/// multilinear polynomials over `{0,1}^{log len}`.
///
/// Storage is **two** MLEs. The "left/right" children the fraction gate consumes
/// are the two halves of these MLEs (`x[i]` and `x[i+n]`, `n = len/2`; MSB-first,
/// see module docs), read during the fold. So a layer holds 2 polynomials; a
/// per-layer *claim* exposes 4 values (`nL, nR, dL, dR`; see
/// `proof::LayerFinalClaim`). Do not confuse the two.
///
/// Invariant: `num.len() == den.len()` and both lengths are a power of two.
///
/// # Numerator/denominator convention (read before constructing)
/// `num` is the **signed weight** side (`ts` on a table side, `-1` on an access
/// side); `den` is the **fingerprint** side (`T+r` / `W+r`, never
/// inverted). To make an accidental swap impossible, this type has **no
/// positional constructor**: build it with field-init syntax so `num`/`den` are
/// named explicitly, e.g. `Layer { num: ts, den: t_plus_r }`.
pub struct Layer<E: Engine> {
  /// Numerator = signed weight (`ts` on a table side, `-1` on an access side).
  pub num: MultilinearPolynomial<E::Scalar>,
  /// Denominator = fingerprint (`T+r` / `W+r`), never inverted.
  pub den: MultilinearPolynomial<E::Scalar>,
}

impl<E: Engine> Layer<E> {
  /// Number of hypercube variables of this layer (`log2(len)`).
  pub fn num_vars(&self) -> usize {
    self.num.get_num_vars()
  }

  /// True when the layer has collapsed to a single cell (the output/root).
  pub fn is_output(&self) -> bool {
    self.num.len() == 1
  }

  /// Folds this layer one level up via the fraction-add gate, MSB-first: the two
  /// children of output cell `i` are the halves `i` (top bit 0) and `i+n` (top
  /// bit 1), `n = len/2`. Returns the parent layer of half the height.
  ///
  /// `(num[i]/den[i]) + (num[i+n]/den[i+n]) =
  ///   ((num[i]·den[i+n] + num[i+n]·den[i]) / (den[i]·den[i+n]))`
  pub fn fold_up(&self) -> Self {
    let len = self.num.len();
    debug_assert_eq!(len, self.den.len());
    debug_assert!(len >= 2 && len.is_power_of_two());
    let n = len / 2;

    // Each output cell i is the fraction-add of the two MSB-halves (child cells
    // i and i+n) — independent across i, so parallelize above the threshold.
    let fold = |i: usize| -> (E::Scalar, E::Scalar) {
      let child = Fraction::new(self.num.Z[i], self.den.Z[i])
        + Fraction::new(self.num.Z[i + n], self.den.Z[i + n]);
      (child.num, child.den)
    };

    let (next_num, next_den): (Vec<_>, Vec<_>) = if n < FOLD_PARALLEL_THRESHOLD {
      (0..n).map(fold).unzip()
    } else {
      (0..n).into_par_iter().map(fold).unzip()
    };

    Self {
      num: MLE::new(next_num),
      den: MLE::new(next_den),
    }
  }

  /// Builds the full tree from this input layer, leaf→root. Result has
  /// `log2(len) + 1` entries; `[0]` is the input layer, the last is the single
  /// output cell.
  pub fn build_tree(self) -> Vec<Self> {
    let depth = self.num.get_num_vars() + 1;
    let mut layers = Vec::with_capacity(depth);
    let mut cur = self;
    loop {
      if cur.is_output() {
        layers.push(cur);
        break;
      }
      let next = cur.fold_up();
      layers.push(cur);
      cur = next;
    }
    layers
  }

  /// The output cell's fraction `(num, den)` as a two-element claim vector,
  /// used to seed the verifier's fold-down (`initial_claims`).
  pub fn output_fraction(&self) -> (E::Scalar, E::Scalar) {
    debug_assert!(self.is_output());
    (self.num.Z[0], self.den.Z[0])
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::Bn256EngineKZG;
  use ff::Field;

  type E = Bn256EngineKZG;
  type Fr = <E as Engine>::Scalar;

  // Reference: the rational sum Σ num[i]/den[i] as a single reduced fraction,
  // computed with real field inversion (only for the test oracle).
  fn ref_sum(num: &[Fr], den: &[Fr]) -> (Fr, Fr) {
    // accumulate as projective fraction a/b + c/d = (ad+cb, bd)
    let mut acc = (Fr::ZERO, Fr::ONE);
    for i in 0..num.len() {
      acc = (acc.0 * den[i] + num[i] * acc.1, acc.1 * den[i]);
    }
    acc
  }

  // Cross-multiplication equality for projective fractions.
  fn frac_eq((n0, d0): (Fr, Fr), (n1, d1): (Fr, Fr)) -> bool {
    n0 * d1 == n1 * d0
  }

  fn layer_from(num: Vec<u64>, den: Vec<u64>) -> Layer<E> {
    Layer {
      num: MultilinearPolynomial::new(num.into_iter().map(Fr::from).collect()),
      den: MultilinearPolynomial::new(den.into_iter().map(Fr::from).collect()),
    }
  }

  #[test]
  fn fold_up_matches_reference_n4() {
    let num = vec![1u64, 2, 3, 4];
    let den = vec![5u64, 6, 7, 8];
    let l = layer_from(num.clone(), den.clone());
    let tree = l.build_tree();
    assert_eq!(tree.len(), 3); // 2 vars → depth 3
    let root = tree.last().unwrap().output_fraction();
    let expected = ref_sum(
      &num.iter().map(|&x| Fr::from(x)).collect::<Vec<_>>(),
      &den.iter().map(|&x| Fr::from(x)).collect::<Vec<_>>(),
    );
    assert!(
      frac_eq(root, expected),
      "tree root fraction must equal the reference rational sum"
    );
  }

  #[test]
  fn fold_up_matches_reference_n8() {
    let num: Vec<u64> = vec![3, 1, 4, 1, 5, 9, 2, 6];
    let den: Vec<u64> = vec![2, 7, 1, 8, 2, 8, 1, 8];
    let l = layer_from(num.clone(), den.clone());
    let tree = l.build_tree();
    assert_eq!(tree.len(), 4);
    let root = tree.last().unwrap().output_fraction();
    let expected = ref_sum(
      &num.iter().map(|&x| Fr::from(x)).collect::<Vec<_>>(),
      &den.iter().map(|&x| Fr::from(x)).collect::<Vec<_>>(),
    );
    assert!(frac_eq(root, expected));
  }

  #[test]
  fn logup_balance_gives_zero_numerator() {
    // A balanced multiset: table {a,b} with multiplicities {1,1}, lookups {a,b}.
    // Σ 1/(α-a) + 1/(α-b) - 1/(α-a) - 1/(α-b) = 0. Encode as one instance's
    // input layer with num = [+1,+1,-1,-1], den = [α-a, α-b, α-a, α-b].
    let alpha = Fr::from(100);
    let a = Fr::from(7);
    let b = Fr::from(9);
    let num = vec![Fr::ONE, Fr::ONE, -Fr::ONE, -Fr::ONE];
    let den = vec![alpha - a, alpha - b, alpha - a, alpha - b];
    let l = Layer::<E> {
      num: MultilinearPolynomial::new(num),
      den: MultilinearPolynomial::new(den),
    };
    let root = l.build_tree().last().unwrap().output_fraction();
    assert_eq!(
      root.0,
      Fr::ZERO,
      "balanced logup must have zero root numerator"
    );
    assert_ne!(root.1, Fr::ZERO, "root denominator must be non-zero");
  }
}
