//! Basic utils
use crate::{
    errors::NovaError
};
use ff::PrimeField;

#[allow(dead_code)]
pub fn matrix_vector_product<F: PrimeField>(
  matrix: &Vec<Vec<F>>,
  vector: &Vec<F>,
) -> Result<Vec<F>, NovaError> {
  if matrix.len() == 0 || matrix[0].len() == 0 {
    return Err(NovaError::InvalidIndex);
  }

  if matrix[0].len() != vector.len() {
    return Err(NovaError::InvalidIndex);
  }

  let mut res = Vec::with_capacity(matrix.len());
  for i in 0..matrix.len() {
    let mut sum = F::ZERO;
    for j in 0..matrix[i].len() {
      sum += matrix[i][j] * vector[j];
    }
    res.push(sum);
  }

  Ok(res)
}

#[allow(dead_code)]
pub fn hadamard_product<F: PrimeField>(a: &Vec<F>, b: &Vec<F>) -> Result<Vec<F>, NovaError> {
  if a.len() != b.len() {
    return Err(NovaError::InvalidIndex);
  }

  let mut res = Vec::with_capacity(a.len());
  for i in 0..a.len() {
    res.push(a[i] * b[i]);
  }

  Ok(res)
}

#[allow(dead_code)]
pub fn to_F_vec<F: PrimeField>(v: Vec<u64>) -> Vec<F> {
  v.iter().map(|x| F::from(*x)).collect()
}

#[allow(dead_code)]
pub fn to_F_matrix<F: PrimeField>(m: Vec<Vec<u64>>) -> Vec<Vec<F>> {
  m.iter().map(|x| to_F_vec(x.clone())).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pasta_curves::Fq;

    #[test]
    fn test_matrix_vector_product() {

      let matrix = vec![vec![1, 2, 3], vec![4, 5, 6]];
      let vector = vec![1, 2, 3];
      let A = to_F_matrix::<Fq>(matrix);
      let z = to_F_vec::<Fq>(vector);
      let res = matrix_vector_product(&A, &z).unwrap();

      assert_eq!(res, to_F_vec::<Fq>(vec![14, 32]));
  }

  #[test]
  fn test_hadamard_product() {
      let a = to_F_vec::<Fq>(vec![1, 2, 3]);
      let b = to_F_vec::<Fq>(vec![4, 5, 6]);
      let res = hadamard_product(&a, &b).unwrap();
      assert_eq!(res, to_F_vec::<Fq>(vec![4, 10, 18]));
  }

}