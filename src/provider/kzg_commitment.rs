//! Commitment engine for KZG commitments
//!

use std::marker::PhantomData;

use ff::Field;
use group::{prime::PrimeCurveAffine, Curve, Group as _};
use halo2curves::pairing::Engine;
use rand::rngs::StdRng;
use rand_core::{CryptoRng, RngCore, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::traits::{
  commitment::{CommitmentEngineTrait, Len},
  Engine as NovaEngine, Group, TranscriptReprTrait,
};

use crate::provider::{pedersen::Commitment, traits::DlogGroup};

/// `UniversalParams` are the universal parameters for the KZG10 scheme.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::G1Affine: Serialize, E::G2Affine: Serialize",
  deserialize = "E::G1Affine: Deserialize<'de>, E::G2Affine: Deserialize<'de>"
))]
pub struct UniversalKZGParam<E: Engine> {
  /// Group elements of the form `{ β^i G }`, where `i` ranges from 0 to
  /// `degree`.
  pub powers_of_g: Vec<E::G1Affine>,
  /// Group elements of the form `{ β^i H }`, where `i` ranges from 0 to
  /// `degree`.
  pub powers_of_h: Vec<E::G2Affine>,
}

// for the purpose of the Len trait, we count commitment bases, i.e. G1 elements
impl<E: Engine> Len for UniversalKZGParam<E> {
  fn length(&self) -> usize {
    self.powers_of_g.len()
  }
}

/// `UnivariateProverKey` is used to generate a proof
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::G1Affine: Serialize",
  deserialize = "E::G1Affine: Deserialize<'de>"
))]
pub struct KZGProverKey<E: Engine> {
  /// generators
  pub powers_of_g: Vec<E::G1Affine>,
}

/// `UVKZGVerifierKey` is used to check evaluation proofs for a given
/// commitment.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::G1Affine: Serialize, E::G2Affine: Serialize",
  deserialize = "E::G1Affine: Deserialize<'de>, E::G2Affine: Deserialize<'de>"
))]
pub struct KZGVerifierKey<E: Engine> {
  /// The generator of G1.
  pub g: E::G1Affine,
  /// The generator of G2.
  pub h: E::G2Affine,
  /// β times the above generator of G2.
  pub beta_h: E::G2Affine,
}

impl<E: Engine> UniversalKZGParam<E> {
  /// Returns the maximum supported degree
  pub fn max_degree(&self) -> usize {
    self.powers_of_g.len()
  }

  /// Trim the universal parameters to specialize the public parameters
  /// for univariate polynomials to the given `supported_size`, and
  /// returns prover key and verifier key. `supported_size` should
  /// be in range `1..params.len()`
  ///
  /// # Panics
  /// If `supported_size` is greater than `self.max_degree()`, or `self.max_degree()` is zero.
  pub fn trim(&self, supported_size: usize) -> (KZGProverKey<E>, KZGVerifierKey<E>) {
    let powers_of_g = self.powers_of_g[..=supported_size].to_vec();

    let pk = KZGProverKey { powers_of_g };
    let vk = KZGVerifierKey {
      g: self.powers_of_g[0],
      h: self.powers_of_h[0],
      beta_h: self.powers_of_h[1],
    };
    (pk, vk)
  }
}

impl<E: Engine> UniversalKZGParam<E> {
  /// Build SRS for testing.
  /// WARNING: THIS FUNCTION IS FOR TESTING PURPOSE ONLY.
  /// THE OUTPUT SRS SHOULD NOT BE USED IN PRODUCTION.
  pub fn gen_srs_for_testing<R: RngCore + CryptoRng>(mut rng: &mut R, max_degree: usize) -> Self {
    let beta = E::Fr::random(&mut rng);
    let g = E::G1::random(&mut rng);
    let h = E::G2::random(rng);

    let (powers_of_g_projective, powers_of_h_projective) = rayon::join(
      || {
        (0..=max_degree)
          .scan(g, |acc, _| {
            let val = *acc;
            *acc *= beta;
            Some(val)
          })
          .collect::<Vec<E::G1>>()
      },
      || {
        (0..=max_degree)
          .scan(h, |acc, _| {
            let val = *acc;
            *acc *= beta;
            Some(val)
          })
          .collect::<Vec<E::G2>>()
      },
    );

    let mut powers_of_g = vec![E::G1Affine::identity(); powers_of_g_projective.len()];
    let mut powers_of_h = vec![E::G2Affine::identity(); powers_of_h_projective.len()];

    rayon::join(
      || E::G1::batch_normalize(&powers_of_g_projective, &mut powers_of_g),
      || E::G2::batch_normalize(&powers_of_h_projective, &mut powers_of_h),
    );

    Self {
      powers_of_g,
      powers_of_h,
    }
  }
}

/// Commitments
#[derive(Debug, Clone, Copy, Eq, PartialEq, Default, Serialize, Deserialize)]
#[serde(bound(
  serialize = "E::G1Affine: Serialize",
  deserialize = "E::G1Affine: Deserialize<'de>"
))]
pub struct UVKZGCommitment<E: Engine>(
  /// the actual commitment is an affine point.
  pub E::G1Affine,
);

impl<E: Engine> TranscriptReprTrait<E::G1> for UVKZGCommitment<E>
where
  E::G1: DlogGroup,
  // Note: due to the move of the bound TranscriptReprTrait<G> on G::Base from Group to Engine
  <E::G1 as Group>::Base: TranscriptReprTrait<E::G1>,
{
  fn to_transcript_bytes(&self) -> Vec<u8> {
    // TODO: avoid the round-trip through the group (to_curve .. to_coordinates)
    let (x, y, is_infinity) = self.0.to_curve().to_coordinates();
    let is_infinity_byte = (!is_infinity).into();
    [
      x.to_transcript_bytes(),
      y.to_transcript_bytes(),
      [is_infinity_byte].to_vec(),
    ]
    .concat()
  }
}

/// Provides a commitment engine
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct KZGCommitmentEngine<E: Engine> {
  _p: PhantomData<E>,
}

impl<E: Engine, NE: NovaEngine<GE = E::G1, Scalar = E::Fr>> CommitmentEngineTrait<NE>
  for KZGCommitmentEngine<E>
where
  E::G1: DlogGroup<AffineGroupElement = E::G1Affine, Scalar = E::Fr>,
  E::G1Affine: Serialize + for<'de> Deserialize<'de>,
  E::G2Affine: Serialize + for<'de> Deserialize<'de>,
{
  type CommitmentKey = UniversalKZGParam<E>;
  type Commitment = Commitment<NE>;

  fn setup(label: &'static [u8], n: usize) -> Self::CommitmentKey {
    // TODO: this is just for testing, replace by grabbing from a real setup for production
    let mut bytes = [0u8; 32];
    let len = label.len().min(32);
    bytes[..len].copy_from_slice(&label[..len]);
    let rng = &mut StdRng::from_seed(bytes);
    UniversalKZGParam::gen_srs_for_testing(rng, n.next_power_of_two())
  }

  fn commit(ck: &Self::CommitmentKey, v: &[<E::G1 as Group>::Scalar]) -> Self::Commitment {
    assert!(ck.length() >= v.len());
    Commitment {
      comm: E::G1::vartime_multiscalar_mul(v, &ck.powers_of_g[..v.len()]),
    }
  }
}

impl<E: Engine, NE: NovaEngine<GE = E::G1, Scalar = E::Fr>> From<Commitment<NE>>
  for UVKZGCommitment<E>
where
  E::G1: Group,
{
  fn from(c: Commitment<NE>) -> Self {
    UVKZGCommitment(c.comm.to_affine())
  }
}

impl<E: Engine, NE: NovaEngine<GE = E::G1, Scalar = E::Fr>> From<UVKZGCommitment<E>>
  for Commitment<NE>
where
  E::G1: Group,
{
  fn from(c: UVKZGCommitment<E>) -> Self {
    Commitment {
      comm: c.0.to_curve(),
    }
  }
}
