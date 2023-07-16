use crate::hypercube::BooleanHypercube;
use crate::spartan::math::Math;
use crate::spartan::polynomial::MultilinearPolynomial;
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS, NUM_FE_FOR_RO, NUM_HASH_BITS},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, R1CS},
  traits::{
    commitment::CommitmentEngineTrait, commitment::CommitmentTrait, AbsorbInROTrait, Group, ROTrait,
  },
  utils::*,
  Commitment, CommitmentKey, CE,
};
use bitvec::vec;
use core::{cmp::max, marker::PhantomData};
use ff::{Field, PrimeField};
use flate2::{write::ZlibEncoder, Compression};
use itertools::concat;
use rand::Rng;
use rand_core::RngCore;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::HashMap;
use std::ops::{Add, Mul};
use std::sync::Arc;

// A bit of collage-programming here.
// As a tmp way to have multilinear polynomial product+addition.
// The idea is to re-evaluate once everything works and decide if we replace this code
// by something else.
//
// THIS CODE HAS BEEN TAKEN FpOM THE ESPRESSO SYSTEMS LIB:
// <https://github.com/EspressoSystems/hyperplonk/blob/main/arithmetic/src/virtual_polynomial.rs#L22-L332>
//
#[rustfmt::skip]
/// A virtual polynomial is a sum of products of multilinear polynomials;
/// where the multilinear polynomials are stored via their multilinear
/// extensions:  `(coefficient, DenseMultilinearExtension)`
///
/// * Number of products n = `polynomial.products.len()`,
/// * Number of multiplicands of ith product m_i =
///   `polynomial.products[i].1.len()`,
/// * Coefficient of ith product c_i = `polynomial.products[i].0`
///
/// The resulting polynomial is
///
/// $$ \sum_{i=0}^{n} c_i \cdot \prod_{j=0}^{m_i} P_{ij} $$
///
/// Example:
///  f = c0 * f0 * f1 * f2 + c1 * f3 * f4
/// where f0 ... f4 are multilinear polynomials
///
/// - flattened_ml_extensions stores the multilinear extension representation of
///   f0, f1, f2, f3 and f4
/// - products is
///     \[
///         (c0, \[0, 1, 2\]),
///         (c1, \[3, 4\])
///     \]
/// - raw_pointers_lookup_table maps fi to i
///
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VirtualPolynomial<F: PrimeField> {
    /// Aux information about the multilinear polynomial
    pub aux_info: VPAuxInfo<F>,
    /// list of reference to products (as usize) of multilinear extension
    pub products: Vec<(F, Vec<usize>)>,
    /// Stores multilinear extensions in which product multiplicand can refer
    /// to.
    pub flattened_ml_extensions: Vec<Arc<MultilinearPolynomial<F>>>,
    /// Pointers to the above poly extensions
    raw_pointers_lookup_table: HashMap<*const MultilinearPolynomial<F>, usize>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
/// Auxiliary information about the multilinear polynomial
pub struct VPAuxInfo<F: PrimeField> {
  /// max number of multiplicands in each product
  pub max_degree: usize,
  /// number of variables of the polynomial
  pub num_variables: usize,
  /// Associated field
  #[doc(hidden)]
  pub phantom: PhantomData<F>,
}

impl<F: PrimeField> Add for &VirtualPolynomial<F> {
  type Output = VirtualPolynomial<F>;
  fn add(self, other: &VirtualPolynomial<F>) -> Self::Output {
    let mut res = self.clone();
    for products in other.products.iter() {
      let cur: Vec<Arc<MultilinearPolynomial<F>>> = products
        .1
        .iter()
        .map(|&x| other.flattened_ml_extensions[x].clone())
        .collect();

      res
        .add_mle_list(cur, products.0)
        .expect("add product failed");
    }
    res
  }
}

// TODO: convert this into a trait
impl<F: PrimeField> VirtualPolynomial<F> {
  /// Creates an empty virtual polynomial with `num_variables`.
  pub fn new(num_variables: usize) -> Self {
    VirtualPolynomial {
      aux_info: VPAuxInfo {
        max_degree: 0,
        num_variables,
        phantom: PhantomData::default(),
      },
      products: Vec::new(),
      flattened_ml_extensions: Vec::new(),
      raw_pointers_lookup_table: HashMap::new(),
    }
  }

  /// Creates an new virtual polynomial Fpom a MLE and its coefficient.
  pub fn new_from_mle(mle: &Arc<MultilinearPolynomial<F>>, coefficient: F) -> Self {
    let mle_ptr: *const MultilinearPolynomial<F> = Arc::as_ptr(mle);
    let mut hm = HashMap::new();
    hm.insert(mle_ptr, 0);

    VirtualPolynomial {
      aux_info: VPAuxInfo {
        // The max degree is the max degree of any individual variable
        max_degree: 1,
        num_variables: mle.get_num_vars(),
        phantom: PhantomData::default(),
      },
      // here `0` points to the first polynomial of `flattened_ml_extensions`
      products: vec![(coefficient, vec![0])],
      flattened_ml_extensions: vec![mle.clone()],
      raw_pointers_lookup_table: hm,
    }
  }

  /// Add a product of list of multilinear extensions to self
  /// Returns an error if the list is empty, or the MLE has a different
  /// `num_vars` Fpom self.
  ///
  /// The MLEs will be multiplied together, and then multiplied by the scalar
  /// `coefficient`.
  pub fn add_mle_list(
    &mut self,
    mle_list: impl IntoIterator<Item = Arc<MultilinearPolynomial<F>>>,
    coefficient: F,
  ) -> Result<(), NovaError> {
    let mle_list: Vec<Arc<MultilinearPolynomial<F>>> = mle_list.into_iter().collect();
    let mut indexed_product = Vec::with_capacity(mle_list.len());

    if mle_list.is_empty() {
      return Err(NovaError::VpArith);
    }

    self.aux_info.max_degree = max(self.aux_info.max_degree, mle_list.len());

    for mle in mle_list {
      if mle.get_num_vars() != self.aux_info.num_variables {
        return Err(NovaError::VpArith);
      }

      let mle_ptr: *const MultilinearPolynomial<F> = Arc::as_ptr(&mle);
      if let Some(index) = self.raw_pointers_lookup_table.get(&mle_ptr) {
        indexed_product.push(*index)
      } else {
        let curr_index = self.flattened_ml_extensions.len();
        self.flattened_ml_extensions.push(mle.clone());
        self.raw_pointers_lookup_table.insert(mle_ptr, curr_index);
        indexed_product.push(curr_index);
      }
    }
    self.products.push((coefficient, indexed_product));
    Ok(())
  }

  /// Multiple the current VirtualPolynomial by an MLE:
  /// - add the MLE to the MLE list;
  /// - multiple each product by MLE and its coefficient.
  /// Returns an error if the MLE has a different `num_vars` Fpom self.
  pub fn mul_by_mle(
    &mut self,
    mle: Arc<MultilinearPolynomial<F>>,
    coefficient: F,
  ) -> Result<(), NovaError> {
    if mle.get_num_vars() != self.aux_info.num_variables {
      return Err(NovaError::VpArith);
    }

    let mle_ptr: *const MultilinearPolynomial<F> = Arc::as_ptr(&mle);

    // check if this mle already exists in the virtual polynomial
    let mle_index = match self.raw_pointers_lookup_table.get(&mle_ptr) {
      Some(&p) => p,
      None => {
        self
          .raw_pointers_lookup_table
          .insert(mle_ptr, self.flattened_ml_extensions.len());
        self.flattened_ml_extensions.push(mle);
        self.flattened_ml_extensions.len() - 1
      }
    };

    for (prod_coef, indices) in self.products.iter_mut() {
      // - add the MLE to the MLE list;
      // - multiple each product by MLE and its coefficient.
      indices.push(mle_index);
      *prod_coef *= coefficient;
    }

    // increase the max degree by one as the MLE has degree 1.
    self.aux_info.max_degree += 1;

    Ok(())
  }

  /// Given virtual polynomial `p(x)` and scalar `s`, compute `s*p(x)`
  pub fn scalar_mul(&mut self, s: &F) {
    for (prod_coef, _) in self.products.iter_mut() {
      *prod_coef *= s;
    }
  }

  /// Evaluate the virtual polynomial at point `point`.
  /// Returns an error is point.len() does not match `num_variables`.
  pub fn evaluate(&self, point: &[F]) -> Result<F, NovaError> {
    if self.aux_info.num_variables != point.len() {
      return Err(NovaError::VpArith);
    }

    // Evaluate all the MLEs at `point`
    let evals: Vec<F> = self
      .flattened_ml_extensions
      .iter()
      .map(|x| x.evaluate(point))
      .collect();

    let res = self
      .products
      .iter()
      .map(|(c, p)| *c * p.iter().map(|&i| evals[i]).product::<F>())
      .sum();

    Ok(res)
  }

  /// Sample a random virtual polynomial, return the polynomial and its sum.
  pub fn rand<R: RngCore>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
    mut rng: &mut R,
  ) -> Result<(Self, F), NovaError> {
    let mut sum = F::ZERO;
    let mut poly = VirtualPolynomial::new(nv);
    for _ in 0..num_products {
      let coefficient = F::random(&mut rng);
      let num_multiplicands = rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
      let (product, product_sum) = random_mle_list(nv, num_multiplicands, rng);

      poly.add_mle_list(product.into_iter(), coefficient)?;
      sum += product_sum * coefficient;
    }
    Ok((poly, sum))
  }

  /// Sample a random virtual polynomial that evaluates to zero everywhere
  /// over the boolean hypercube.
  pub fn rand_zero<R: RngCore>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
    mut rng: &mut R,
  ) -> Result<Self, NovaError> {
    let coefficient = F::random(&mut rng);
    let mut poly = VirtualPolynomial::new(nv);
    for _ in 0..num_products {
      let num_multiplicands = rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
      let product = random_zero_mle_list(nv, num_multiplicands, rng);

      poly.add_mle_list(product.into_iter(), coefficient)?;
    }

    Ok(poly)
  }

  // Input poly f(x) and a random vector r, output
  //      \hat f(x) = \sum_{x_i \in eval_x} f(x_i) eq(x, r)
  // where
  //      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
  //
  // This function is used in ZeroCheck.
  pub fn build_f_hat(&self, r: &[F]) -> Result<Self, NovaError> {
    if self.aux_info.num_variables != r.len() {
      return Err(NovaError::VpArith);
    }

    let eq_x_r = build_eq_x_r(r)?;
    let mut res = self.clone();
    res.mul_by_mle(eq_x_r, F::ONE)?;

    Ok(res)
  }
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<F: PrimeField>(r: &[F]) -> Result<Arc<MultilinearPolynomial<F>>, NovaError> {
  let evals = build_eq_x_r_vec(r)?;
  let mle = MultilinearPolynomial::new(evals);

  Ok(Arc::new(mle))
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_vec<F: PrimeField>(r: &[F]) -> Result<Vec<F>, NovaError> {
  // we build eq(x,r) Fpom its evaluations
  // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
  // for example, with num_vars = 4, x is a binary vector of 4, then
  //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
  //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
  //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
  //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
  //  ....
  //  1 1 1 1 -> r0       * r1        * r2        * r3
  // we will need 2^num_var evaluations

  let mut eval = Vec::new();
  build_eq_x_r_helper(r, &mut eval)?;

  Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_helper<F: PrimeField>(r: &[F], buf: &mut Vec<F>) -> Result<(), NovaError> {
  if r.is_empty() {
    return Err(NovaError::VpArith);
  } else if r.len() == 1 {
    // initializing the buffer with [1-r_0, r_0]
    buf.push(F::ONE - r[0]);
    buf.push(r[0]);
  } else {
    build_eq_x_r_helper(&r[1..], buf)?;

    // suppose at the previous step we received [b_1, ..., b_k]
    // for the current step we will need
    // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
    // if x_0 = 1:   r0 * [b_1, ..., b_k]
    // let mut res = vec![];
    // for &b_i in buf.iter() {
    //     let tmp = r[0] * b_i;
    //     res.push(b_i - tmp);
    //     res.push(tmp);
    // }
    // *buf = res;

    let mut res = vec![F::ZERO; buf.len() << 1];
    res.par_iter_mut().enumerate().for_each(|(i, val)| {
      let bi = buf[i >> 1];
      let tmp = r[0] * bi;
      if i & 1 == 0 {
        *val = bi - tmp;
      } else {
        *val = tmp;
      }
    });
    *buf = res;
  }

  Ok(())
}

#[cfg(test)]
mod test {
  use super::*;
  use crate::hypercube::bit_decompose;
  use pasta_curves::Fp;
  use rand_core::OsRng;

  #[test]
  fn test_virtual_polynomial_additions() -> Result<(), NovaError> {
    let mut rng = OsRng;
    for nv in 2..5 {
      for num_products in 2..5 {
        let base: Vec<Fp> = (0..nv).map(|_| Fp::random(&mut rng)).collect();

        let (a, _a_sum) = VirtualPolynomial::<Fp>::rand(nv, (2, 3), num_products, &mut rng)?;
        let (b, _b_sum) = VirtualPolynomial::<Fp>::rand(nv, (2, 3), num_products, &mut rng)?;
        let c = &a + &b;

        assert_eq!(
          a.evaluate(base.as_ref())? + b.evaluate(base.as_ref())?,
          c.evaluate(base.as_ref())?
        );
      }
    }

    Ok(())
  }

  #[test]
  fn test_virtual_polynomial_mul_by_mle() -> Result<(), NovaError> {
    let mut rng = OsRng;
    for nv in 2..5 {
      for num_products in 2..5 {
        let base: Vec<Fp> = (0..nv).map(|_| Fp::random(&mut rng)).collect();

        let (a, _a_sum) = VirtualPolynomial::<Fp>::rand(nv, (2, 3), num_products, &mut rng)?;
        let (b, _b_sum) = random_mle_list(nv, 1, &mut rng);
        let b_mle = b[0].clone();
        let coeff = Fp::random(&mut rng);
        let b_vp = VirtualPolynomial::new_from_mle(&b_mle, coeff);

        let mut c = a.clone();

        c.mul_by_mle(b_mle, coeff)?;

        assert_eq!(
          a.evaluate(base.as_ref())? * b_vp.evaluate(base.as_ref())?,
          c.evaluate(base.as_ref())?
        );
      }
    }

    Ok(())
  }

  #[test]
  fn test_eq_xr() {
    let mut rng = OsRng;
    for nv in 4..10 {
      let r: Vec<Fp> = (0..nv).map(|_| Fp::random(&mut rng)).collect();
      let eq_x_r = build_eq_x_r(r.as_ref()).unwrap();
      let eq_x_r2 = build_eq_x_r_for_test(r.as_ref());
      assert_eq!(eq_x_r, eq_x_r2);
    }
  }

  /// Naive method to build eq(x, r).
  /// Only used for testing purpose.
  // Evaluate
  //      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
  // over r, which is
  //      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
  fn build_eq_x_r_for_test<F: PrimeField>(r: &[F]) -> Arc<MultilinearPolynomial<F>> {
    // we build eq(x,r) Fpom its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    // First, we build array for {1 - r_i}
    let one_minus_r: Vec<F> = r.iter().map(|ri| F::ONE - ri).collect();

    let num_var = r.len();
    let mut eval = vec![];

    for i in 0..1 << num_var {
      let mut current_eval = F::ONE;
      let bit_sequence = bit_decompose(i, num_var);

      for (&bit, (ri, one_minus_ri)) in bit_sequence.iter().zip(r.iter().zip(one_minus_r.iter())) {
        current_eval *= if bit { *ri } else { *one_minus_ri };
      }
      eval.push(current_eval);
    }

    let mle = MultilinearPolynomial::new(eval);

    Arc::new(mle)
  }
}
