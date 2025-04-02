//! This module implements a non-interactive folding scheme
#![allow(non_snake_case)]
use crate::{
  constants::NUM_CHALLENGE_BITS,
  errors::NovaError,
  gadgets::utils::{base_as_scalar, scalar_as_base},
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness, RelaxedR1CSInstance, RelaxedR1CSWitness},
  traits::{AbsorbInROTrait, Engine, ROConstants, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use rand_core::OsRng;
use serde::{Deserialize, Serialize};

/// An NIFS message from Nova's folding scheme
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct NIFS<E: Engine> {
  pub(crate) comm_T: Commitment<E>,
}

impl<E: Engine> NIFS<E> {
  /// Takes as input a Relaxed EC instance-witness tuple `(U1, W1)` and
  /// an EC instance-witness tuple `(U2, W2)`
  /// defined with respect to the same `ck`, and outputs
  /// a folded Relaxed EC instance-witness tuple `(U, W)` of the same shape `shape`,
  /// with the guarantee that the folded witness `W` satisfies the folded instance `U`
  /// if and only if `W1` satisfies `U1` and `W2` satisfies `U2`.
  pub fn prove(
    ck: &CommitmentKey<E>,
    ro_consts: &ROConstants<E>,
    pp_digest: &E::Scalar,
    S: &ECShape<E>,
    U1: &RelaxedECInstance<E>,
    W1: &RelaxedECWitness<E>,
    U2: &ECInstance<E>,
    W2: &ECWitness<E>,
  ) -> Result<(NIFS<E>, (RelaxedR1CSInstance<E>, RelaxedR1CSWitness<E>)), NovaError> {
    // initialize a new RO
    let mut ro = E::RO::new(ro_consts.clone());

    // append the digest of pp to the transcript
    ro.absorb(scalar_as_base::<E>(*pp_digest));

    // append U2 to transcript, U1 does not need to absorbed since U2.X[0] = Hash(params, U1, i, z0, zi)
    U2.absorb_in_ro(&mut ro);

    // compute a commitment to the cross-term
    let r_T = E::Scalar::random(&mut OsRng);
    let (T, comm_T) = S.commit_T(ck, U1, W1, U2, W2, &r_T)?;

    // append `comm_T` to the transcript and obtain a challenge
    comm_T.absorb_in_ro(&mut ro);

    // compute a challenge from the RO
    let r = base_as_scalar::<E>(ro.squeeze(NUM_CHALLENGE_BITS));

    // fold the instance using `r` and `comm_T`
    let U = U1.fold(U2, &comm_T, &r);

    // fold the witness using `r` and `T`
    let W = W1.fold(W2, &T, &r_T, &r)?;

    // return the folded instance and witness
    Ok((Self { comm_T }, (U, W)))
  }
}