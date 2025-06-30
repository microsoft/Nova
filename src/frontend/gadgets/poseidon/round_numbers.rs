//! A port of `calc_round_numbers.py`
//! <https://extgit.iaik.tugraz.at/krypto/hadeshash/-/blob/9d80ec0473ad7cde5a12f3aac46439ad0da68c0a/code/scripts/calc_round_numbers.py>
//! from Python2 to Rust for a (roughly) 256-bit prime field (e.g. BLS12-381's scalar field) and
//! 128-bit security level.

// The number of bits of the Poseidon prime field modulus. Denoted `n` in the Poseidon paper
// (where `n = ceil(log2(p))`). Note that BLS12-381's scalar field modulus is 255 bits, however we
// use 256 bits for simplicity when operating on bytes as the single bit difference does not affect
// the round number security properties.
const PRIME_BITLEN: usize = 256;

// Security level (in bits), denoted `M` in the Poseidon paper.
const M: usize = 128;

// The number of S-boxes (also called the "cost") given by equation (14) in the Poseidon paper:
// `cost = t * R_F + R_P`.
#[inline]
const fn n_sboxes(t: usize, rf: usize, rp: usize) -> usize {
  t * rf + rp
}

// Returns the round numbers for a given arity `(R_F, R_P)`.
pub(crate) fn round_numbers_base(arity: usize) -> (usize, usize) {
  let t = arity + 1;
  calc_round_numbers(t, true)
}

// In case of newly-discovered attacks, we may need stronger security.
// This option exists so we can preemptively create circuits in order to switch
// to them quickly if needed.
//
// "A realistic alternative is to increase the number of partial rounds by 25%.
// Then it is unlikely that a new attack breaks through this number,
// but even if this happens then the complexity is almost surely above 2^64, and you will be safe."
// - D Khovratovich
pub(crate) fn round_numbers_strengthened(arity: usize) -> (usize, usize) {
  let (full_round, partial_rounds) = round_numbers_base(arity);

  // Increase by 25%, rounding up.
  let strengthened_partial_rounds = f64::ceil(partial_rounds as f64 * 1.25) as usize;

  (full_round, strengthened_partial_rounds)
}

// Returns the round numbers for a given width `t`. Here, the `security_margin` parameter does not
// indicate that we are calculating `R_F` and `R_P` for the "strengthened" round numbers, done in
// the function `round_numbers_strengthened()`.
pub(crate) fn calc_round_numbers(t: usize, security_margin: bool) -> (usize, usize) {
  let mut rf = 0;
  let mut rp = 0;
  let mut n_sboxes_min = usize::MAX;

  for mut rf_test in (2..=1000).step_by(2) {
    for mut rp_test in 4..200 {
      if round_numbers_are_secure(t, rf_test, rp_test) {
        if security_margin {
          rf_test += 2;
          rp_test = (1.075 * rp_test as f32).ceil() as usize;
        }
        let n_sboxes = n_sboxes(t, rf_test, rp_test);
        if n_sboxes < n_sboxes_min || (n_sboxes == n_sboxes_min && rf_test < rf) {
          rf = rf_test;
          rp = rp_test;
          n_sboxes_min = n_sboxes;
        }
      }
    }
  }

  (rf, rp)
}

// Returns `true` if the provided round numbers satisfy the security inequalities specified in the
// Poseidon paper.
fn round_numbers_are_secure(t: usize, rf: usize, rp: usize) -> bool {
  let (rp, t, n, m) = (rp as f32, t as f32, PRIME_BITLEN as f32, M as f32);
  let rf_stat = if m <= (n - 3.0) * (t + 1.0) {
    6.0
  } else {
    10.0
  };
  let rf_interp = 0.43 * m + t.log2() - rp;
  let rf_grob_1 = 0.21 * n - rp;
  let rf_grob_2 = (0.14 * n - 1.0 - rp) / (t - 1.0);
  let rf_max = [rf_stat, rf_interp, rf_grob_1, rf_grob_2]
    .iter()
    .map(|rf| rf.ceil() as usize)
    .max()
    .unwrap();
  rf >= rf_max
}
