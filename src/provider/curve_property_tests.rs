//! Property / regression tests for the `halo2curves` v0.9.0 primitives this crate
//! relies on: finite-field arithmetic, the elliptic-curve group law, the BN256
//! pairing, and `msm_best`.
//!
//! These check the algebraic identities the higher layers depend on, so a future
//! `halo2curves` version bump that changes field, curve, or pairing arithmetic is
//! caught here rather than silently. The `asm` feature is enabled by default on
//! x86_64, so by default this exercises the assembly field backend; build with and
//! without `--features asm` to cover both backends.
//!
//! Everything uses the public `halo2curves` API (re-exporting `ff`/`group`/`pairing`),
//! `num-bigint`, and a fixed-seed RNG for reproducibility.

use halo2curves::bn256::{Bn256, Fq as Bn256Fq, Fr as Bn256Fr, G1Affine, G2Affine, Gt, G1, G2};
use halo2curves::ff::{Field, FromUniformBytes, PrimeField};
use halo2curves::group::{prime::PrimeCurveAffine, Curve, Group};
use halo2curves::grumpkin::G1 as GrumpkinG1;
use halo2curves::msm::msm_best;
use halo2curves::pairing::Engine;
use halo2curves::secp256k1::Fq as Secp256k1Fq;
use num_bigint::BigUint;
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

/// Fixed seed so failures are reproducible across runs / CI.
const SEED: u64 = 0x_5EED_C0DE_1234_5678;
const FIELD_ITERS: usize = 64;
const GROUP_ITERS: usize = 50;
const PAIRING_ITERS: usize = 16;

fn rng() -> ChaCha20Rng {
  ChaCha20Rng::seed_from_u64(SEED)
}

// =====================================================================================
// 1. FIELD arithmetic + unbiased `from_uniform_bytes` reduction
// =====================================================================================

/// Generic field property check, run for every prime field of interest.
fn field_properties<F: PrimeField + FromUniformBytes<64>>() {
  let mut rng = rng();

  // invert(ZERO) is none.
  assert!(bool::from(F::ZERO.invert().is_none()));

  // The modulus p = (ZERO - ONE) + 1, computed from the canonical little-endian repr.
  let p = BigUint::from_bytes_le((F::ZERO - F::ONE).to_repr().as_ref()) + BigUint::from(1u8);

  for _ in 0..FIELD_ITERS {
    let a = F::random(&mut rng);
    let b = F::random(&mut rng);

    // a * invert(a) == ONE (for nonzero a, which random() yields with overwhelming prob.)
    if bool::from(!a.is_zero()) {
      assert_eq!(a * a.invert().unwrap(), F::ONE);
    }

    // Commutativity of multiplication.
    assert_eq!(a * b, b * a);

    // Difference of squares: (a + b)(a - b) == a^2 - b^2.
    assert_eq!((a + b) * (a - b), a.square() - b.square());

    // from_uniform_bytes::<64>(x) == (x interpreted little-endian) mod p (unbiased).
    let mut x = [0u8; 64];
    rng.fill_bytes(&mut x);
    let fe = F::from_uniform_bytes(&x);
    let expected = BigUint::from_bytes_le(&x) % &p;
    let got = BigUint::from_bytes_le(fe.to_repr().as_ref());
    assert_eq!(got, expected);
  }
}

#[test]
fn field_bn256_fr() {
  field_properties::<Bn256Fr>();
}

#[test]
fn field_bn256_fq() {
  field_properties::<Bn256Fq>();
}

#[test]
fn field_secp256k1_fq() {
  field_properties::<Secp256k1Fq>();
}

// =====================================================================================
// 2. GROUP law (RCB-2015 complete addition), generic over `group::Group`
// =====================================================================================

fn group_law<G: Group>() {
  let mut rng = rng();
  for _ in 0..GROUP_ITERS {
    let p = G::random(&mut rng);
    let q = G::random(&mut rng);
    let r = G::random(&mut rng);

    // P + (-P) == identity.
    assert_eq!(p + (-p), G::identity());

    // P + identity == P.
    assert_eq!(p + G::identity(), p);

    // double == P + P.
    assert_eq!(p.double(), p + p);

    // Associativity.
    assert_eq!((p + q) + r, p + (q + r));

    // Scalar-mul distributes over addition: [k](P + Q) == [k]P + [k]Q.
    let k = G::Scalar::random(&mut rng);
    assert_eq!((p + q) * k, p * k + q * k);
  }
}

#[test]
fn group_law_bn256_g1() {
  group_law::<G1>();
}

#[test]
fn group_law_grumpkin_g1() {
  group_law::<GrumpkinG1>();
}

// =====================================================================================
// 3. PAIRING (BN256): non-degeneracy + bilinearity
// =====================================================================================

#[test]
fn pairing_properties() {
  let mut rng = rng();

  let p_aff = G1Affine::generator();
  let q_aff = G2Affine::generator();
  let g1 = G1::generator();
  let g2 = G2::generator();

  // Non-degeneracy: e(G1gen, G2gen) != identity.
  let base = Bn256::pairing(&p_aff, &q_aff);
  assert_ne!(base, Gt::identity());

  for _ in 0..PAIRING_ITERS {
    let a = Bn256Fr::random(&mut rng);
    let b = Bn256Fr::random(&mut rng);

    // Bilinearity: e([a]P, [b]Q) == e(P, Q) * (a*b).
    // Gt is an additive-notation Group whose scalar-mul by Fr is exponentiation.
    let ap = (g1 * a).to_affine();
    let bq = (g2 * b).to_affine();
    assert_eq!(Bn256::pairing(&ap, &bq), base * (a * b));

    // Additivity in the first argument (additive Gt notation = multiplicative target):
    // e([a]P, Q) + e([b]P, Q) == e([a+b]P, Q).
    let abp = (g1 * (a + b)).to_affine();
    let lhs =
      Bn256::pairing(&(g1 * a).to_affine(), &q_aff) + Bn256::pairing(&(g1 * b).to_affine(), &q_aff);
    assert_eq!(lhs, Bn256::pairing(&abp, &q_aff));
  }
}

// =====================================================================================
// 4. MSM: `halo2curves::msm::msm_best` vs naive sum, across the affine-batch threshold
// =====================================================================================
//
// `halo2curves::msm::msm_best` is public (used by `crate::provider::msm`). It switches
// to an affine-batch strategy around n >= 8104, so the n set below straddles that
// boundary: small (16, 100), just-at (8104) and just-over (8200).

fn naive_msm(scalars: &[Bn256Fr], bases: &[G1Affine]) -> G1 {
  scalars
    .iter()
    .zip(bases.iter())
    .fold(G1::identity(), |acc, (s, b)| acc + *b * s)
}

#[test]
fn msm_best_matches_naive() {
  let mut rng = rng();

  for &n in &[16usize, 100, 8104, 8200] {
    // Distinct bn256 G1 bases: a random start point plus successive generator steps,
    // batch-normalized to affine (avoids one inversion per base).
    let g = G1::generator();
    let mut acc = g * Bn256Fr::random(&mut rng);
    let mut proj = Vec::with_capacity(n);
    for _ in 0..n {
      proj.push(acc);
      acc += g;
    }
    let mut bases = vec![G1Affine::identity(); n];
    G1::batch_normalize(&proj, &mut bases);

    // Scalar set A: random.
    let random_scalars: Vec<Bn256Fr> = (0..n).map(|_| Bn256Fr::random(&mut rng)).collect();

    // Scalar set B: all equal (to a single random value).
    let equal_scalars = vec![Bn256Fr::random(&mut rng); n];

    // Scalar set C: mix of zero and (r-1) == -ONE (the maximal scalar).
    let r_minus_one = -Bn256Fr::ONE;
    let mix_scalars: Vec<Bn256Fr> = (0..n)
      .map(|i| {
        if i % 2 == 0 {
          Bn256Fr::ZERO
        } else {
          r_minus_one
        }
      })
      .collect();

    for scalars in [&random_scalars, &equal_scalars, &mix_scalars] {
      assert_eq!(msm_best(scalars, &bases), naive_msm(scalars, &bases));
    }
  }
}

// =====================================================================================
// 5. CONSTANT sanity round-trips
// =====================================================================================

#[test]
fn constant_sanity() {
  // (ZERO - ONE) round-trips through to_repr / from_repr.
  let x = Bn256Fr::ZERO - Bn256Fr::ONE;
  let repr = x.to_repr();
  let y = Bn256Fr::from_repr(repr).unwrap();
  assert_eq!(x, y);

  // (-ONE) + ONE == ZERO.
  assert_eq!((-Bn256Fr::ONE) + Bn256Fr::ONE, Bn256Fr::ZERO);
}
