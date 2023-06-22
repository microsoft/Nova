//! Basic utils
use std::{cmp::max, collections::HashMap, marker::PhantomData, ops::Add, sync::Arc};

use crate::errors::NovaError;
use crate::spartan::polynomial::MultilinearPolynomial;
use crate::traits::Group;
use ff::{Field, PrimeField};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

/// A matrix structure represented on a sparse form.
/// First element is row index, second column, third value stored
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SparseMatrix<G: Group>(pub(crate) Vec<(usize, usize, G::Scalar)>);

impl<G: Group> SparseMatrix<G> {
  pub fn new() -> Self {
    Self(vec![])
  }

  pub fn with_capacity(n: usize) -> Self {
    Self(Vec::with_capacity(n))
  }

  // Find the maximum row index in the matrix
  pub fn n_rows(&self) -> usize {
    let max_row_idx = self
      .0
      .iter()
      .copied()
      .map(|r| r.0)
      .fold(usize::MIN, |a, b| a.max(b));
    max_row_idx + 1
  }

  // Find the maximum column index in the matrix
  pub fn n_cols(&self) -> usize {
    let max_col_idx = self
      .0
      .iter()
      .copied()
      .map(|r| r.1)
      .fold(std::usize::MIN, |a, b| a.max(b));
    max_col_idx + 1
  }

  pub(crate) fn is_valid(
    &self,
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
  ) -> Result<(), NovaError> {
    let res = self
      .0
      .iter()
      .copied()
      .map(|(row, col, _val)| {
        if row >= num_cons || col > num_io + num_vars {
          Err(NovaError::InvalidIndex)
        } else {
          Ok(())
        }
      })
      .collect::<Result<Vec<()>, NovaError>>();

    if res.is_err() {
      Err(NovaError::InvalidIndex)
    } else {
      Ok(())
    }
  }

  pub(crate) fn pad(&mut self, n: usize) {
    let prev_n = self.n_cols();
    self.0.par_iter_mut().for_each(|(_, c, _)| {
      *c = if *c >= prev_n { *c + n - prev_n } else { *c };
    });
  }
}

impl<G: Group> From<Vec<(usize, usize, G::Scalar)>> for SparseMatrix<G> {
  fn from(matrix: Vec<(usize, usize, G::Scalar)>) -> SparseMatrix<G> {
    SparseMatrix(matrix)
  }
}

impl<G: Group> From<&Vec<(usize, usize, G::Scalar)>> for SparseMatrix<G> {
  fn from(matrix: &Vec<(usize, usize, G::Scalar)>) -> SparseMatrix<G> {
    SparseMatrix(matrix.clone())
  }
}

pub fn vector_add<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Vec<F> {
  assert_eq!(a.len(), b.len(), "Vector addition with different lengths");
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] + b[i]);
  }

  res
}

pub fn vector_elem_product<F: PrimeField>(a: &Vec<F>, e: &F) -> Vec<F> {
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * e);
  }

  res
}

// XXX: This could be implemented via Mul trait in the lib. We should consider as it will reduce imports.
#[allow(dead_code)]
pub fn matrix_vector_product<F: PrimeField>(matrix: &Vec<Vec<F>>, vector: &Vec<F>) -> Vec<F> {
  assert_ne!(matrix.len(), 0, "empty-row matrix");
  assert_ne!(matrix[0].len(), 0, "empty-col  matrix");
  assert_eq!(
    matrix[0].len(),
    vector.len(),
    "matrix rows != vector length"
  );
  let mut res = Vec::with_capacity(matrix.len());
  for i in 0..matrix.len() {
    let mut sum = F::ZERO;
    for j in 0..matrix[i].len() {
      sum += matrix[i][j] * vector[j];
    }
    res.push(sum);
  }

  res
}

// Matrix vector product where matrix is sparse
// First element is row index, second column, third value stored
// XXX: This could be implemented via Mul trait in the lib. We should consider as it will reduce imports.
pub fn matrix_vector_product_sparse<G: Group>(
  matrix: &SparseMatrix<G>,
  vector: &Vec<G::Scalar>,
) -> Vec<G::Scalar> {
  assert_eq!(
    matrix.n_cols(),
    vector.len(),
    "matrix cols != vector length"
  );
  let mut res = vec![G::Scalar::ZERO; vector.len()];
  for &(row, col, value) in matrix.0.iter() {
    res[row] += value * vector[col];
  }
  res
}

pub fn hadamard_product<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Vec<F> {
  assert_eq!(a.len(), b.len(), "Haddamard needs same len vectors");
  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * b[i]);
  }

  res
}

pub fn sparse_matrix_to_mlp<G: Group>(
  matrix: &SparseMatrix<G>,
) -> MultilinearPolynomial<G::Scalar> {
  let n_rows = matrix.n_rows();
  let n_cols = matrix.n_cols();

  // Since n_rows and n_cols already account for 0 indexing,
  // The total number of elements would be n_rows * n_cols
  let total_elements: usize = n_rows * n_cols;
  let n_vars: usize = total_elements.next_power_of_two().trailing_zeros() as usize;

  // Create a vector of zeros with size 2^n_vars
  let mut vec: Vec<G::Scalar> = vec![G::Scalar::ZERO; 2_usize.pow(n_vars as u32)];

  // Assign non-zero entries from the sparse matrix to the vector
  for &(i, j, val) in matrix.0.iter() {
    let index = i * n_cols + j; // Convert (i, j) into an index for a flat vector
    vec[index] = val;
  }

  // Pad to 2^n_vars
  let vec_padded: Vec<G::Scalar> = [
    vec.clone(),
    std::iter::repeat(G::Scalar::ZERO)
      .take((1 << n_vars) - vec.len())
      .collect(),
  ]
  .concat();

  // Convert this vector into a MultilinearPolynomial
  MultilinearPolynomial::new(vec_padded)
}

/// Decompose an integer into a binary vector in little endian.
pub fn bit_decompose(input: u64, num_var: usize) -> Vec<bool> {
  let mut res = Vec::with_capacity(num_var);
  let mut i = input;
  for _ in 0..num_var {
    res.push(i & 1 == 1);
    i >>= 1;
  }
  res
}

// A bit of collage-programming here.
// As a tmp way to have multilinear polynomial product+addition.
// The idea is to re-evaluate once everything works and decide if we replace this code
// by something else.
//
// THIS CODE HAS BEEN TAKEN FROM THE ESPRESSO SYSTEMS LIB:
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

  /// Creates an new virtual polynomial from a MLE and its coefficient.
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
  /// `num_vars` from self.
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
  /// Returns an error if the MLE has a different `num_vars` from self.
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
  // we build eq(x,r) from its evaluations
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
mod tests {
  use super::*;
  use pasta_curves::{Ep, Fq};

  fn to_F_vec<F: PrimeField>(v: Vec<u64>) -> Vec<F> {
    v.iter().map(|x| F::from(*x)).collect()
  }

  fn to_F_matrix<F: PrimeField>(m: Vec<Vec<u64>>) -> Vec<Vec<F>> {
    m.iter().map(|x| to_F_vec(x.clone())).collect()
  }

  fn to_F_matrix_sparse<F: PrimeField>(m: Vec<(usize, usize, u64)>) -> Vec<(usize, usize, F)> {
    m.iter().map(|x| (x.0, x.1, F::from(x.2))).collect()
  }

  #[test]
  fn test_vector_add() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let b = to_F_vec::<Fq>(vec![4, 5, 6]);
    let res = vector_add(&a, &b);
    assert_eq!(res, to_F_vec::<Fq>(vec![5, 7, 9]));
  }

  #[test]
  fn test_vector_elem_product() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let e = Fq::from(2);
    let res = vector_elem_product(&a, &e);
    assert_eq!(res, to_F_vec::<Fq>(vec![2, 4, 6]));
  }

  #[test]
  fn test_matrix_vector_product() {
    let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
    let vector = vec![1, 2, 3];
    let A = to_F_matrix::<Fq>(matrix);
    let z = to_F_vec::<Fq>(vector);
    let res = matrix_vector_product(&A, &z);

    assert_eq!(res, to_F_vec::<Fq>(vec![14, 32]));
  }

  #[test]
  fn test_hadamard_product() {
    let a = to_F_vec::<Fq>(vec![1, 2, 3]);
    let b = to_F_vec::<Fq>(vec![4, 5, 6]);
    let res = hadamard_product(&a, &b);
    assert_eq!(res, to_F_vec::<Fq>(vec![4, 10, 18]));
  }

  #[test]
  fn test_matrix_vector_product_sparse() {
    let matrix = vec![
      (0, 0, 1),
      (0, 1, 2),
      (0, 2, 3),
      (1, 0, 4),
      (1, 1, 5),
      (1, 2, 6),
    ];
    let vector = vec![1, 2, 3];
    let A = to_F_matrix_sparse::<Fq>(matrix);
    let z = to_F_vec::<Fq>(vector);
    let res = matrix_vector_product_sparse::<Ep>(&(A.into()), &z);

    assert_eq!(res, to_F_vec::<Fq>(vec![14, 32, 0]));
  }

  #[test]
  fn test_sparse_matrix_n_rows() {
    let matrix = vec![
      (0, 0, 1),
      (0, 1, 2),
      (0, 2, 3),
      (1, 0, 4),
      (1, 1, 5),
      (1, 2, 6),
    ];
    let A: SparseMatrix<Ep> = to_F_matrix_sparse::<Fq>(matrix).into();
    assert_eq!(A.n_rows(), 2);
  }

  #[test]
  fn test_sparse_matrix_n_cols() {
    let matrix = vec![
      (0, 0, 1),
      (0, 1, 2),
      (0, 2, 3),
      (1, 0, 4),
      (1, 1, 5),
      (1, 2, 6),
    ];
    let A: SparseMatrix<Ep> = to_F_matrix_sparse::<Fq>(matrix).into();
    assert_eq!(A.n_cols(), 3);
  }

  #[test]
  fn test_sparse_matrix_to_mlp() {
    let matrix = vec![
      (0, 0, 2),
      (0, 1, 3),
      (0, 2, 4),
      (0, 3, 4),
      (1, 0, 4),
      (1, 1, 11),
      (1, 2, 14),
      (1, 3, 14),
      (2, 0, 2),
      (2, 1, 8),
      (2, 2, 17),
      (2, 3, 17),
      (3, 0, 420),
      (3, 1, 4),
      (3, 2, 2),
    ];
    let A: SparseMatrix<Ep> = to_F_matrix_sparse::<Fq>(matrix).into();

    // Convert the sparse matrix to a multilinear polynomial
    let mlp = sparse_matrix_to_mlp(&A);

    // A 4x4 matrix, thus 2bit x 2bit, thus 2^4=16 evals
    assert_eq!(mlp.len(), 16);
  }
}
