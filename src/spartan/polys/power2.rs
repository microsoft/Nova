//! Implements the Power polynomial, the multilinear polynomial that encodes powers of tau

use crate::{
  errors::NovaError,
  traits::{commitment::CommitmentEngineTrait, Engine},
  Commitment, CommitmentKey, CE,
};
use ff::PrimeField;
use itertools::Itertools;

use super::multilinear::MultilinearPolynomial;

#[derive(Debug, Clone)]
pub struct PowPoly<F>
where
  F: PrimeField,
{
  e1: Vec<F>,
  e2: Vec<F>,
  ell: usize,
}

impl<F> PowPoly<F>
where
  F: PrimeField,
{
  /// Create a new instance of [`PowPoly`]
  pub fn new(tau: F, ell: usize) -> Self {
    // compute e1
    let (left_num_vars, right_num_vars) = compute_split_point_dims(ell);
    // We use `right_num_vars` here becuase we treat the boolean hypercube as big
    // endian
    let e1_len = 2usize.pow(right_num_vars as u32);
    let mut e1 = vec![F::ONE; e1_len];

    for i in 1..e1_len {
      e1[i] = e1[i - 1] * tau;
    }

    // compute e2
    let e2_len = 2usize.pow(left_num_vars as u32);

    let mut e2 = vec![F::ONE; e2_len];

    let e_sqrt_m_1 = e1[e1_len - 1] * tau;

    for i in 1..e2_len {
      e2[i] = e2[i - 1] * e_sqrt_m_1
    }

    PowPoly { e1, e2, ell }
  }

  /// Get tau from pow poly
  pub fn tau(&self) -> F {
    self.e1[1]
  }

  pub fn h_polys(&self) -> (MultilinearPolynomial<F>, MultilinearPolynomial<F>) {
    let num_vars = 2usize.pow(self.ell as u32);
    let (_, right_num_vars) = compute_split_point_dims(self.ell);
    let right_num_evals = 2usize.pow(right_num_vars as u32);

    let h1 = {
      let e1 = self.e1.clone().into_iter().cycle().take(num_vars).collect();
      MultilinearPolynomial::new(e1)
    };
    let h2 = {
      let e2 = self
        .e2
        .clone()
        .into_iter()
        .flat_map(|e2_i| vec![e2_i; right_num_evals])
        .collect();
      MultilinearPolynomial::new(e2)
    };

    (h1, h2)
  }

  /// Fold two [`PowPoly`]'s
  pub fn fold(&self, other: &Self, r_b: F) -> Self {
    let inner_fold = |w1: &[F], w2: &[F]| -> Vec<F> {
      w1.iter()
        .zip_eq(w2.iter())
        .map(|(w1, w2)| *w1 * (F::ONE - r_b) + *w2 * r_b)
        .collect()
    };

    let e1 = inner_fold(&self.e1, &other.e1);
    let e2 = inner_fold(&self.e2, &other.e2);

    PowPoly {
      e1,
      e2,
      ell: self.ell,
    }
  }

  /// Compute a commitment to the [`PowPoly`]
  pub fn commit<E>(&self, ck: &CommitmentKey<E>, r_e: E::Scalar) -> Commitment<E>
  where
    E: Engine<Scalar = F>,
  {
    CE::<E>::commit(ck, &[self.e1.clone(), self.e2.clone()].concat(), &r_e) // TODO: remove clone
  }

  #[allow(clippy::manual_memcpy, clippy::needless_range_loop)]
  /// Power check to zero check
  pub fn pc_to_zc(
    &self,
  ) -> Result<
    (
      MultilinearPolynomial<F>,
      MultilinearPolynomial<F>,
      MultilinearPolynomial<F>,
    ),
    NovaError,
  > {
    let tau = self.tau();
    let g1 = [self.e1.clone(), self.e2.clone()].concat();

    let sqrt_m = self.e1.len();

    /*
     * * Compute g2 ***********************
     */
    let mut g2 = vec![F::ONE; g1.len()];
    for i in 1..sqrt_m {
      g2[i] = g1[i - 1];
    }

    g2[sqrt_m + 1] = g1[sqrt_m - 1];

    if g1.len() > sqrt_m + 2 {
      g2[sqrt_m + 2] = g1[sqrt_m + 1];

      for i in sqrt_m + 3..g1.len() {
        g2[i] = g1[sqrt_m + 1]
      }
    }

    /*
     * * Compute g3 ***********************
     */
    let mut g3 = vec![F::ONE; g1.len()];
    for i in 1..sqrt_m {
      g3[i] = tau;
    }
    g3[sqrt_m + 1] = tau;

    for i in sqrt_m + 2..g1.len() {
      g3[i] = g1[i - 1]
    }

    let g1 = MultilinearPolynomial::new_padded(g1, self.ell);
    let g2 = MultilinearPolynomial::new_padded(g2, self.ell);
    let g3 = MultilinearPolynomial::new_padded(g3, self.ell);

    Ok((g1, g2, g3))
  }

  // pub fn ell(&self) -> usize {
  //   self.ell
  // }

  // pub fn e1_e2(&self) -> (&[F], &[F]) {
  //   (&self.e1, &self.e2)
  // }
}

fn compute_split_point_dims(ell: usize) -> (usize, usize) {
  let l = ell / 2;
  let r = ell - l;

  (l, r)
}
