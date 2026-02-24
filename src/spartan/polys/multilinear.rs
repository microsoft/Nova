//! Main components:
//! - `MultilinearPolynomial`: Dense representation of multilinear polynomials, represented by evaluations over all possible binary inputs.
//! - `SparsePolynomial`: Efficient representation of sparse multilinear polynomials, storing only non-zero evaluations.

use crate::spartan::{math::Math, polys::eq::EqPolynomial};
use core::ops::{Add, Index};
use ff::PrimeField;
use itertools::Itertools as _;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
  num_vars: usize, // the number of variables in the multilinear polynomial
  /// The evaluations of the polynomial in all the 2^num_vars Boolean inputs
  pub Z: Vec<Scalar>,
}

impl<Scalar: PrimeField> MultilinearPolynomial<Scalar> {
  /// Creates a new `MultilinearPolynomial` from the given evaluations.
  ///
  /// # Panics
  /// The number of evaluations must be a power of two.
  pub fn new(Z: Vec<Scalar>) -> Self {
    let num_vars = Z.len().log_2();
    assert_eq!(Z.len(), 1 << num_vars);
    MultilinearPolynomial { num_vars, Z }
  }

  /// Returns the number of variables in the multilinear polynomial
  pub const fn get_num_vars(&self) -> usize {
    self.num_vars
  }

  /// Returns the total number of evaluations.
  pub fn len(&self) -> usize {
    self.Z.len()
  }

  /// Returns true if the polynomial has no evaluations.
  pub fn is_empty(&self) -> bool {
    self.Z.is_empty()
  }

  /// Binds the polynomial's top variable using the given scalar.
  ///
  /// This operation modifies the polynomial in-place.
  pub fn bind_poly_var_top(&mut self, r: &Scalar) {
    assert!(self.num_vars > 0);

    let n = self.len() / 2;

    let (left, right) = self.Z.split_at_mut(n);

    zip_with_for_each!((left.par_iter_mut(), right.par_iter()), |a, b| {
      *a += *r * (*b - *a);
    });

    self.Z.resize(n, Scalar::ZERO);
    self.num_vars -= 1;
  }

  /// Evaluates the polynomial at the given point.
  /// Returns Z(r) in O(n) time using O(sqrt(n)) memory for eq tables.
  ///
  /// The point must have a value for each variable.
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(r.len(), self.get_num_vars());
    Self::evaluate_with(&self.Z, r)
  }

  /// Evaluates the polynomial with the given evaluations and point.
  /// Uses sqrt-decomposition: splits r into two halves, builds two O(sqrt(n))
  /// eq tables, reduces Z to sqrt(n), then dots with the other eq table.
  pub fn evaluate_with(Z: &[Scalar], r: &[Scalar]) -> Scalar {
    let s = r.len();
    let s_right = s / 2;
    let s_left = s - s_right;
    let n_left = 1 << s_left;
    let n_right = 1 << s_right;

    let eq_left = EqPolynomial::evals_from_points(&r[..s_left]);
    let eq_right = EqPolynomial::evals_from_points(&r[s_left..]);

    // reduce Z from 2^s to 2^s_left by dotting each row with eq_right
    let reduced: Vec<Scalar> = (0..n_left)
      .into_par_iter()
      .map(|i| {
        let chunk = &Z[i * n_right..(i + 1) * n_right];
        chunk
          .iter()
          .zip(eq_right.iter())
          .map(|(z, e)| *z * *e)
          .sum()
      })
      .collect();

    // dot reduced with eq_left
    zip_with!(
      (eq_left.into_par_iter(), reduced.into_par_iter()),
      |a, b| a * b
    )
    .sum()
  }

  /// Evaluates multiple polynomials at the same point, reusing sqrt-sized eq tables.
  pub fn multi_evaluate_with(Zs: &[&[Scalar]], r: &[Scalar]) -> Vec<Scalar> {
    let s = r.len();
    let s_right = s / 2;
    let s_left = s - s_right;
    let n_left = 1 << s_left;
    let n_right = 1 << s_right;

    let eq_left = EqPolynomial::evals_from_points(&r[..s_left]);
    let eq_right = EqPolynomial::evals_from_points(&r[s_left..]);

    Zs.iter()
      .map(|Z| {
        let reduced: Vec<Scalar> = (0..n_left)
          .into_par_iter()
          .map(|i| {
            let chunk = &Z[i * n_right..(i + 1) * n_right];
            chunk
              .iter()
              .zip(eq_right.iter())
              .map(|(z, e)| *z * *e)
              .sum()
          })
          .collect();

        zip_with!((eq_left.par_iter(), reduced.into_par_iter()), |a, b| *a * b).sum()
      })
      .collect()
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
/// In our context, sparse polynomials are non-zeros over the hypercube at locations that map to "small" integers
/// We exploit this property to implement a time-optimal algorithm
pub struct SparsePolynomial<Scalar: PrimeField> {
  num_vars: usize,
  /// The non-zero evaluations
  pub Z: Vec<Scalar>,
}

impl<Scalar: PrimeField> SparsePolynomial<Scalar> {
  /// Creates a new `SparsePolynomial` from the given number of variables and evaluations.
  pub fn new(num_vars: usize, Z: Vec<Scalar>) -> Self {
    SparsePolynomial { num_vars, Z }
  }

  /// A time-optimal algorithm to evaluate sparse polynomials
  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.num_vars, r.len());

    let num_vars_z = self.Z.len().next_power_of_two().log_2();
    let chis = EqPolynomial::evals_from_points(&r[self.num_vars - 1 - num_vars_z..]);
    let eval_partial: Scalar = self
      .Z
      .iter()
      .zip(chis.iter())
      .map(|(z, chi)| *z * *chi)
      .sum();

    let common = (0..self.num_vars - 1 - num_vars_z)
      .map(|i| Scalar::ONE - r[i])
      .product::<Scalar>();

    common * eval_partial
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

    let sum: Vec<Scalar> = zip_with!(into_iter, (self.Z, other.Z), |a, b| a + b).collect();

    Ok(MultilinearPolynomial::new(sum))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::provider::{bn256_grumpkin::bn256, pasta::pallas, secp_secq::secp256k1};
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
    // Let the polynomial have 4 variables, but is non-zero at only 3 locations (out of 2^4 = 16) over the hypercube
    let mut Z = vec![F::ONE, F::ONE, F::from(2)];
    let m_poly = SparsePolynomial::<F>::new(4, Z.clone());

    Z.resize(16, F::ZERO); // append with zeros to make it a dense polynomial
    let m_poly_dense = MultilinearPolynomial::new(Z);

    // evaluation point
    let x = vec![F::from(5), F::from(8), F::from(5), F::from(3)];

    // check evaluations
    assert_eq!(
      m_poly.evaluate(x.as_slice()),
      m_poly_dense.evaluate(x.as_slice())
    );
  }

  #[test]
  fn test_multilinear_polynomial() {
    test_multilinear_polynomial_with::<pallas::Scalar>();
  }

  #[test]
  fn test_sparse_polynomial() {
    test_sparse_polynomial_with::<pallas::Scalar>();
  }

  fn test_mlp_add_with<F: PrimeField>() {
    let mlp1 = make_mlp(4, F::from(3));
    let mlp2 = make_mlp(4, F::from(7));

    let mlp3 = mlp1.add(mlp2).unwrap();

    assert_eq!(mlp3.Z, vec![F::from(10); 4]);
  }

  #[test]
  fn test_mlp_add() {
    test_mlp_add_with::<pallas::Scalar>();
    test_mlp_add_with::<bn256::Scalar>();
    test_mlp_add_with::<secp256k1::Scalar>();
  }

  fn test_evaluation_with<F: PrimeField>() {
    let num_evals = 4;
    let mut evals: Vec<F> = Vec::with_capacity(num_evals);
    for _ in 0..num_evals {
      evals.push(F::from(8));
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
    test_evaluation_with::<pallas::Scalar>();
    test_evaluation_with::<bn256::Scalar>();
    test_evaluation_with::<secp256k1::Scalar>();
  }

  /// Returns a random ML polynomial
  #[allow(clippy::needless_borrows_for_generic_args)]
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
    bind_and_evaluate_with::<pallas::Scalar>();
    bind_and_evaluate_with::<bn256::Scalar>();
    bind_and_evaluate_with::<secp256k1::Scalar>();
  }

  fn test_multi_evaluate_with_matches_single<F: PrimeField>() {
    let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
    let num_vars = 6;

    // Create multiple random polynomials
    let poly1 = random::<_, F>(num_vars, &mut rng);
    let poly2 = random::<_, F>(num_vars, &mut rng);
    let poly3 = random::<_, F>(num_vars, &mut rng);

    // Draw a random evaluation point
    let pt: Vec<F> = std::iter::from_fn(|| Some(F::random(&mut rng)))
      .take(num_vars)
      .collect();

    // Evaluate each polynomial individually
    let expected: Vec<F> = [&poly1, &poly2, &poly3]
      .iter()
      .map(|p| MultilinearPolynomial::evaluate_with(&p.Z, &pt))
      .collect();

    // Evaluate all at once using multi_evaluate_with
    let zs: Vec<&[F]> = vec![&poly1.Z, &poly2.Z, &poly3.Z];
    let result = MultilinearPolynomial::multi_evaluate_with(&zs, &pt);

    assert_eq!(result, expected);
  }

  fn test_multi_evaluate_with_single_poly<F: PrimeField>() {
    let mut rng = ChaCha20Rng::from_seed([7u8; 32]);
    let num_vars = 4;
    let poly = random::<_, F>(num_vars, &mut rng);
    let pt: Vec<F> = std::iter::from_fn(|| Some(F::random(&mut rng)))
      .take(num_vars)
      .collect();

    let expected = MultilinearPolynomial::evaluate_with(&poly.Z, &pt);
    let zs: Vec<&[F]> = vec![&poly.Z];
    let result = MultilinearPolynomial::multi_evaluate_with(&zs, &pt);

    assert_eq!(result, vec![expected]);
  }

  fn test_multi_evaluate_with_known_values<F: PrimeField>() {
    // p(x_1, x_2, x_3) = (x_1 + x_2) * x_3
    // Evaluations are indexed with x_1 as the MSB and x_3 as the LSB
    // (i.e., index = x_1 * 4 + x_2 * 2 + x_3):
    // index 0 = (x_1=0,x_2=0,x_3=0)=0, index 1 = (x_1=0,x_2=0,x_3=1)=0,
    // index 2 = (x_1=0,x_2=1,x_3=0)=0, index 3 = (x_1=0,x_2=1,x_3=1)=1,
    // index 4 = (x_1=1,x_2=0,x_3=0)=0, index 5 = (x_1=1,x_2=0,x_3=1)=1,
    // index 6 = (x_1=1,x_2=1,x_3=0)=0, index 7 = (x_1=1,x_2=1,x_3=1)=2
    let two = F::from(2);
    let z1 = vec![
      F::ZERO,
      F::ZERO,
      F::ZERO,
      F::ONE,
      F::ZERO,
      F::ONE,
      F::ZERO,
      two,
    ];
    // Constant polynomial with all evaluations equal to 5
    let z2 = vec![F::from(5); 8];

    let pt = vec![F::ONE, F::ONE, F::ONE];

    let result = MultilinearPolynomial::multi_evaluate_with(&[z1.as_slice(), z2.as_slice()], &pt);

    assert_eq!(result[0], two); // (1+1)*1 = 2
    assert_eq!(result[1], F::from(5)); // constant poly evaluates to 5 everywhere
  }

  #[test]
  fn test_multi_evaluate_with() {
    test_multi_evaluate_with_matches_single::<pallas::Scalar>();
    test_multi_evaluate_with_matches_single::<bn256::Scalar>();
    test_multi_evaluate_with_matches_single::<secp256k1::Scalar>();

    test_multi_evaluate_with_single_poly::<pallas::Scalar>();
    test_multi_evaluate_with_single_poly::<bn256::Scalar>();
    test_multi_evaluate_with_single_poly::<secp256k1::Scalar>();

    test_multi_evaluate_with_known_values::<pallas::Scalar>();
    test_multi_evaluate_with_known_values::<bn256::Scalar>();
    test_multi_evaluate_with_known_values::<secp256k1::Scalar>();
  }
}
