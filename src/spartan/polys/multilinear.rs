//! Main components:
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.

use std::ops::{Add, Index};

use ff::PrimeField;
use rayon::prelude::{
  IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
  IntoParallelRefMutIterator, ParallelIterator,
};
use serde::{Deserialize, Serialize};

use crate::spartan::{math::Math, polys::eq::EqPolynomial};

/// A multilinear extension of a polynomial $Z(\cdot)$, denote it as $\tilde{Z}(x_1, ..., x_m)$
/// where the degree of each variable is at most one.
///
/// This is the dense representation of a multilinear poynomial.
/// Let it be $\mathbb{G}(\cdot): \mathbb{F}^m \rightarrow \mathbb{F}$, it can be represented uniquely by the list of
/// evaluations of $\mathbb{G}(\cdot)$ over the Boolean hypercube $\{0, 1\}^m$.
///
/// For example, a 3 variables multilinear polynomial can be represented by evaluation
/// at points $[0, 2^3-1]$.
///
/// The implementation follows
/// $$
/// \tilde{Z}(x_1, ..., x_m) = \sum_{e\in {0,1}^m}Z(e) \cdot \prod_{i=1}^m(x_i \cdot e_i + (1-x_i) \cdot (1-e_i))
/// $$
///
/// Vector $Z$ indicates $Z(e)$ where $e$ ranges from $0$ to $2^m-1$.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MultilinearPolynomial<Scalar: PrimeField> {
  num_vars: usize,           // the number of variables in the multilinear polynomial
  pub(crate) Z: Vec<Scalar>, // evaluations of the polynomial in all the 2^num_vars Boolean inputs
}

impl<Scalar: PrimeField> MultilinearPolynomial<Scalar> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<Scalar>) -> Self {
    assert_eq!(Z.len(), (2_usize).pow((Z.len() as f64).log2() as u32));
    MultilinearPolynomial {
      num_vars: usize::try_from(Z.len().ilog2()).unwrap(),
      Z,
    }
  }

  /// Returns the number of variables in the multilinear polynomial
  pub const fn get_num_vars(&self) -> usize {
    self.num_vars
  }

  /// Returns the total number of evaluations.
  pub fn len(&self) -> usize {
    self.Z.len()
  }

  /// Bounds the polynomial's top variable using the given scalar.
  ///
  /// This operation modifies the polynomial in-place.
  pub fn bind_poly_var_top(&mut self, r: &Scalar) {
    let n = self.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);

    left
      .par_iter_mut()
      .zip(right.par_iter())
      .for_each(|(a, b)| {
        *a += *r * (*b - *a);
      });

    self.Z.resize(n, Scalar::ZERO);
    self.num_vars -= 1;
  }

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time.
  ///
  /// The point must have a value for each variable.
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    // r must have a value for each variable
    assert_eq!(r.len(), self.get_num_vars());
    let chis = EqPolynomial::new(r.to_vec()).evals();
    assert_eq!(chis.len(), self.Z.len());

    (0..chis.len())
      .into_par_iter()
      .map(|i| chis[i] * self.Z[i])
      .sum()
  }

  /// Evaluates the polynomial with the given evaluations and point.
  pub fn evaluate_with(Z: &[Scalar], r: &[Scalar]) -> Scalar {
    EqPolynomial::new(r.to_vec())
      .evals()
      .into_par_iter()
      .zip(Z.into_par_iter())
      .map(|(a, b)| a * b)
      .sum()
  }
}

impl<Scalar: PrimeField> Index<usize> for MultilinearPolynomial<Scalar> {
  type Output = Scalar;

  #[inline(always)]
  fn index(&self, _index: usize) -> &Scalar {
    &(self.Z[_index])
  }
}

/// Sparse multilinear polynomial, which means the $Z(\cdot)$ is zero at most points.
/// So we do not have to store every evaluations of $Z(\cdot)$, only store the non-zero points.
///
/// For example, the evaluations are [0, 0, 0, 1, 0, 1, 0, 2].
/// The sparse polynomial only store the non-zero values, [(3, 1), (5, 1), (7, 2)].
/// In the tuple, the first is index, the second is value.
pub(crate) struct SparsePolynomial<Scalar: PrimeField> {
  num_vars: usize,
  Z: Vec<(usize, Scalar)>,
}

impl<Scalar: PrimeField> SparsePolynomial<Scalar> {
  pub fn new(num_vars: usize, Z: Vec<(usize, Scalar)>) -> Self {
    SparsePolynomial { num_vars, Z }
  }

  /// Computes the $\tilde{eq}$ extension polynomial.
  /// return 1 when a == r, otherwise return 0.
  fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
    assert_eq!(a.len(), r.len());
    let mut chi_i = Scalar::ONE;
    for j in 0..r.len() {
      if a[j] {
        chi_i *= r[j];
      } else {
        chi_i *= Scalar::ONE - r[j];
      }
    }
    chi_i
  }

  // Takes O(n log n)
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars, r.len());

    (0..self.Z.len())
      .into_par_iter()
      .map(|i| {
        let bits = (self.Z[i].0).get_bits(r.len());
        SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
      })
      .sum()
  }
}

/// Adds another multilinear polynomial to `self`.
/// Assumes the two polynomials have the same number of variables.
impl<Scalar: PrimeField> Add for MultilinearPolynomial<Scalar> {
  type Output = Result<Self, &'static str>;

  fn add(self, other: Self) -> Self::Output {
    if self.get_num_vars() != other.get_num_vars() {
      return Err("The two polynomials must have the same number of variables");
    }

    let sum: Vec<Scalar> = self
      .Z
      .iter()
      .zip(other.Z.iter())
      .map(|(a, b)| *a + *b)
      .collect();

    Ok(MultilinearPolynomial::new(sum))
  }
}

#[cfg(test)]
mod tests {
  use crate::provider::{self, bn256_grumpkin::bn256, secp_secq::secp256k1};

  use super::*;
  use rand_chacha::ChaCha20Rng;
  use rand_core::{CryptoRng, RngCore, SeedableRng};

  fn make_mlp<F: PrimeField>(len: usize, value: F) -> MultilinearPolynomial<F> {
    MultilinearPolynomial {
      num_vars: len.count_ones() as usize,
      Z: vec![value; len],
    }
  }

  fn test_multilinear_polynomial_with<F: PrimeField>() {
    // Let the polynomial has 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

    let TWO = F::from(2);

    let Z = vec![
      F::ZERO,
      F::ZERO,
      F::ZERO,
      F::ONE,
      F::ZERO,
      F::ONE,
      F::ZERO,
      TWO,
    ];
    let m_poly = MultilinearPolynomial::<F>::new(Z.clone());
    assert_eq!(m_poly.get_num_vars(), 3);

    let x = vec![F::ONE, F::ONE, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), TWO);

    let y = MultilinearPolynomial::<F>::evaluate_with(Z.as_slice(), x.as_slice());
    assert_eq!(y, TWO);
  }

  fn test_sparse_polynomial_with<F: PrimeField>() {
    // Let the polynomial have 3 variables, p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations of the polynomial at boolean cube are [0, 0, 0, 1, 0, 1, 0, 2].

    let TWO = F::from(2);
    let Z = vec![(3, F::ONE), (5, F::ONE), (7, TWO)];
    let m_poly = SparsePolynomial::<F>::new(3, Z);

    let x = vec![F::ONE, F::ONE, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), TWO);

    let x = vec![F::ONE, F::ZERO, F::ONE];
    assert_eq!(m_poly.evaluate(x.as_slice()), F::ONE);
  }

  #[test]
  fn test_multilinear_polynomial() {
    test_multilinear_polynomial_with::<pasta_curves::Fp>();
  }

  #[test]
  fn test_sparse_polynomial() {
    test_sparse_polynomial_with::<pasta_curves::Fp>();
  }

  fn test_mlp_add_with<F: PrimeField>() {
    let mlp1 = make_mlp(4, F::from(3));
    let mlp2 = make_mlp(4, F::from(7));

    let mlp3 = mlp1.add(mlp2).unwrap();

    assert_eq!(mlp3.Z, vec![F::from(10); 4]);
  }

  #[test]
  fn test_mlp_add() {
    test_mlp_add_with::<pasta_curves::Fp>();
    test_mlp_add_with::<bn256::Scalar>();
    test_mlp_add_with::<secp256k1::Scalar>();
  }

  fn test_evaluation_with<F: PrimeField>() {
    let num_evals = 4;
    let mut evals: Vec<F> = Vec::with_capacity(num_evals);
    for _ in 0..num_evals {
      evals.push(F::from_u128(8));
    }
    let dense_poly: MultilinearPolynomial<F> = MultilinearPolynomial::new(evals.clone());

    // Evaluate at 3:
    // (0, 0) = 1
    // (0, 1) = 1
    // (1, 0) = 1
    // (1, 1) = 1
    // g(x_0,x_1) => c_0*(1 - x_0)(1 - x_1) + c_1*(1-x_0)(x_1) + c_2*(x_0)(1-x_1) + c_3*(x_0)(x_1)
    // g(3, 4) = 8*(1 - 3)(1 - 4) + 8*(1-3)(4) + 8*(3)(1-4) + 8*(3)(4) = 48 + -64 + -72 + 96  = 8
    // g(5, 10) = 8*(1 - 5)(1 - 10) + 8*(1 - 5)(10) + 8*(5)(1-10) + 8*(5)(10) = 96 + -16 + -72 + 96  = 8
    assert_eq!(
      dense_poly.evaluate(vec![F::from(3), F::from(4)].as_slice()),
      F::from(8)
    );
  }

  #[test]
  fn test_evaluation() {
    test_evaluation_with::<pasta_curves::Fp>();
    test_evaluation_with::<provider::bn256_grumpkin::bn256::Scalar>();
    test_evaluation_with::<provider::secp_secq::secp256k1::Scalar>();
  }

  /// Returns a random ML polynomial
  fn random<R: RngCore + CryptoRng, Scalar: PrimeField>(
    num_vars: usize,
    mut rng: &mut R,
  ) -> MultilinearPolynomial<Scalar> {
    MultilinearPolynomial::new(
      std::iter::from_fn(|| Some(Scalar::random(&mut rng)))
        .take(1 << num_vars)
        .collect(),
    )
  }

  /// This binds the variables of a multilinear polynomial to a provided sequence
  /// of values.
  ///
  /// Assuming `bind_poly_var_top` defines the "top" variable of the polynomial,
  /// this aims to test whether variables should be provided to the
  /// `evaluate` function in topmost-first (big endian) of topmost-last (lower endian)
  /// order.
  fn bind_sequence<F: PrimeField>(
    poly: &MultilinearPolynomial<F>,
    values: &[F],
  ) -> MultilinearPolynomial<F> {
    // Assert that the size of the polynomial being evaluated is a power of 2 greater than (1 << values.len())
    assert!(poly.Z.len().is_power_of_two());
    assert!(poly.Z.len() >= 1 << values.len());

    let mut tmp = poly.clone();
    for v in values.iter() {
      tmp.bind_poly_var_top(v);
    }
    tmp
  }

  fn bind_and_evaluate_with<F: PrimeField>() {
    for i in 0..50 {
      // Initialize a random polynomial
      let n = 7;
      let mut rng = ChaCha20Rng::from_seed([i as u8; 32]);
      let poly = random(n, &mut rng);

      // draw a random point
      let pt: Vec<_> = std::iter::from_fn(|| Some(F::random(&mut rng)))
        .take(n)
        .collect();
      // this shows the order in which coordinates are evaluated
      assert_eq!(poly.evaluate(&pt), bind_sequence(&poly, &pt).Z[0])
    }
  }

  #[test]
  fn test_bind_and_evaluate() {
    bind_and_evaluate_with::<pasta_curves::Fp>();
    bind_and_evaluate_with::<bn256::Scalar>();
    bind_and_evaluate_with::<secp256k1::Scalar>();
  }
}
