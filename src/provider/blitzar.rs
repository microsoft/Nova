//! This module implements variable time multi-scalar multiplication using Blitzar's GPU acceleration
use blitzar;
use halo2curves::bn256::{Fr as Scalar, G1Affine as Affine, G1 as Point};

/// A trait that provides the ability to perform multi-scalar multiplication in variable time
pub fn vartime_multiscalar_mul(scalars: &[Scalar], bases: &[Affine]) -> Point {
  let mut blitzar_commitments = vec![Point::default(); 1];

  let scalar_bytes: Vec<[u8; 32]> = scalars.iter().map(|s| s.to_bytes()).collect();

  blitzar::compute::compute_bn254_g1_uncompressed_commitments_with_halo2_generators(
    &mut blitzar_commitments,
    &[(&scalar_bytes).into()],
    bases,
  );

  blitzar_commitments[0]
}

/// A trait that provides the ability to perform a batch of multi-scalar multiplication in variable time
pub fn batch_vartime_multiscalar_mul(scalars: &[Vec<Scalar>], bases: &[Affine]) -> Vec<Point> {
  let mut blitzar_commitments = vec![Point::default(); scalars.len()];

  let scalar_bytes: Vec<Vec<[u8; 32]>> = scalars
    .iter()
    .map(|s| s.iter().map(|v| v.to_bytes()).collect())
    .collect();

  let scalars_table: Vec<blitzar::sequence::Sequence<'_>> =
    scalar_bytes.iter().map(|s| s.into()).collect();

  blitzar::compute::compute_bn254_g1_uncompressed_commitments_with_halo2_generators(
    &mut blitzar_commitments,
    &scalars_table,
    bases,
  );

  blitzar_commitments
}

#[cfg(test)]
mod tests {
  use super::*;
  use ff::Field;
  use halo2curves::msm::msm_best;

  #[test]
  fn test_vartime_multiscalar_mul_empty() {
    let scalars = vec![];
    let bases = vec![];

    let result = vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(result, Point::default());
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_empty() {
    let scalars = vec![vec![]];
    let bases = vec![];

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(result, [Point::default(); 1]);
  }

  #[test]
  fn test_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();

    let scalars = vec![Scalar::random(&mut rng), Scalar::random(&mut rng)];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];

    let result = vartime_multiscalar_mul(&scalars, &bases);

    let expected = bases[0] * scalars[0] + bases[1] * scalars[1];

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_simple() {
    let mut rng = rand::thread_rng();

    let scalars = vec![
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
      vec![Scalar::random(&mut rng), Scalar::random(&mut rng)],
    ];
    let bases = vec![Affine::random(&mut rng), Affine::random(&mut rng)];

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    assert_eq!(
      result[0],
      bases[0] * scalars[0][0] + bases[1] * scalars[0][1]
    );
    assert_eq!(
      result[1],
      bases[0] * scalars[1][0] + bases[1] * scalars[1][1]
    );
  }

  #[test]
  fn test_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);

    let mut expected = Point::default();
    for i in 0..sample_len {
      expected += bases[i] * scalars[i];
    }

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected: Vec<Point> = scalars
      .iter()
      .map(|scalar_row| {
        scalar_row
          .iter()
          .enumerate()
          .map(|(i, scalar)| bases[i] * scalar)
          .sum()
      })
      .collect();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let sample_len = 100;

    let (scalars, bases): (Vec<_>, Vec<_>) = (0..sample_len)
      .map(|_| (Scalar::random(&mut rng), Affine::random(&mut rng)))
      .unzip();

    let result = vartime_multiscalar_mul(&scalars, &bases);
    let expected = msm_best(&scalars, &bases);

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_len = 100;

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|_| (0..sample_len).map(|_| Scalar::random(&mut rng)).collect())
      .collect();

    let bases: Vec<Affine> = (0..sample_len).map(|_| Affine::random(&mut rng)).collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }

  #[test]
  fn test_batch_vartime_multiscalar_mul_of_varying_sized_scalars_with_msm_best() {
    let mut rng = rand::thread_rng();
    let batch_len = 20;
    let sample_lens: Vec<usize> = (0..batch_len).map(|i| i * 100 / (batch_len - 1)).collect();

    let scalars: Vec<Vec<Scalar>> = (0..batch_len)
      .map(|i| {
        (0..sample_lens[i])
          .map(|_| Scalar::random(&mut rng))
          .collect()
      })
      .collect();

    let bases: Vec<Affine> = (0..sample_lens[batch_len - 1])
      .map(|_| Affine::random(&mut rng))
      .collect();

    let result = batch_vartime_multiscalar_mul(&scalars, &bases);

    let expected = scalars
      .iter()
      .map(|scalar| msm_best(scalar, &bases[..scalar.len()]))
      .collect::<Vec<_>>();

    assert_eq!(result, expected);
  }
}
