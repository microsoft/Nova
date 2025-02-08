//! This module defines relations used in the Neutron folding scheme
use crate::{
  constants::{BN_LIMB_WIDTH, BN_N_LIMBS},
  errors::NovaError,
  gadgets::{
    nonnative::{bignat::nat_to_limbs, util::f_to_nat},
    utils::scalar_as_base,
  },
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  spartan::math::Math,
  traits::{commitment::CommitmentEngineTrait, AbsorbInROTrait, Engine, ROTrait},
  Commitment, CommitmentKey,
};
use ff::Field;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// A type that holds structure information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Structure<E: Engine> {
  /// Shape of the R1CS relation
  pub(crate) S: R1CSShape<E>,

  /// the number of variables in the Eq polynomial
  pub(crate) ell: usize,
  pub(crate) left: usize,
  pub(crate) right: usize,
}

/// A type that holds witness information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldedWitness<E: Engine> {
  /// Running witness of the main relation
  pub(crate) W: Vec<E::Scalar>,
  r_W: E::Scalar,

  /// eq polynomial in tensor form
  pub(crate) E: Vec<E::Scalar>,
  r_E: E::Scalar,
}

/// A type that holds instance information for a zero-fold relation
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldedInstance<E: Engine> {
  comm_W: Commitment<E>,
  comm_E: Commitment<E>,
  pub(crate) T: E::Scalar,
  pub(crate) X: Vec<E::Scalar>,
  pub(crate) u: E::Scalar,
}

#[allow(unused)]
impl<E: Engine> Structure<E> {
  pub fn new(S: &R1CSShape<E>) -> Self {
    let ell = S.num_cons.next_power_of_two().log_2();
    let (ell1, ell2) = (ell / 2, ell - ell / 2);
    Structure {
      S: S.clone(),
      ell,
      left: 1 << ell1,
      right: 1 << ell2,
    }
  }

  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    U: &FoldedInstance<E>,
    W: &FoldedWitness<E>,
  ) -> Result<(), NovaError> {
    // check if the witness is satisfying
    let z = [W.W.clone(), vec![U.u], U.X.clone()].concat();
    let (Az, Bz, Cz) = self.S.multiply_vec(&z)?;

    // full_E is the outer outer product of E1 and E2
    // E1 and E2 are splits of E
    let (E1, E2) = W.E.split_at(self.left);
    let mut full_E = vec![E::Scalar::ONE; self.left * self.right];
    for i in 0..self.left {
      for j in 0..self.right {
        full_E[i * self.right + j] = E1[i] * E2[j];
      }
    }

    let sum = full_E
      .par_iter()
      .zip(Az.par_iter())
      .zip(Bz.par_iter())
      .zip(Cz.par_iter())
      .map(|(((e, a), b), c)| *e * ((*a) * (*b) - *c))
      .reduce(|| E::Scalar::ZERO, |acc, x| acc + x);

    if sum != U.T {
      println!("sum: {:?}", sum);
      println!("U.T: {:?}", U.T);
      return Err(NovaError::UnSat {
        reason: "sum != U.T".to_string(),
      });
    }

    // check the validity of the commitments
    let comm_W = E::CE::commit(ck, &W.W, &W.r_W);
    let comm_E = E::CE::commit(ck, &W.E, &W.r_E);

    if comm_W != U.comm_W || comm_E != U.comm_E {
      return Err(NovaError::UnSat {
        reason: "comm_W != U.comm_W || comm_E != U.comm_E".to_string(),
      });
    }

    Ok(())
  }
}

#[allow(unused)]
impl<E: Engine> FoldedWitness<E> {
  pub fn default(S: &Structure<E>) -> Self {
    FoldedWitness {
      W: vec![E::Scalar::ZERO; S.S.num_vars],
      r_W: E::Scalar::ZERO,
      E: vec![E::Scalar::ZERO; S.left + S.right],
      r_E: E::Scalar::ZERO,
    }
  }

  /// Fold the witness with another witness
  pub fn fold(
    &self,
    W2: &R1CSWitness<E>,
    E2: &Vec<E::Scalar>,
    r_E2: &E::Scalar,
    r_b: &E::Scalar,
  ) -> Result<Self, NovaError> {
    // we need to compute the weighted sum using weights of (1-r_b) and r_b
    let W = self
      .W
      .par_iter()
      .zip(W2.W.par_iter())
      .map(|(w1, w2)| (E::Scalar::ONE - r_b) * w1 + *r_b * w2)
      .collect::<Vec<_>>();
    let r_W = (E::Scalar::ONE - r_b) * self.r_W + *r_b * W2.r_W;

    let E = self
      .E
      .par_iter()
      .zip(E2.par_iter())
      .map(|(e1, e2)| (E::Scalar::ONE - r_b) * e1 + *r_b * e2)
      .collect::<Vec<_>>();
    let r_E = (E::Scalar::ONE - r_b) * self.r_E + *r_b * r_E2;

    Ok(Self { W, r_W, E, r_E })
  }
}

#[allow(unused)]
impl<E: Engine> FoldedInstance<E> {
  pub fn default(S: &Structure<E>) -> Self {
    FoldedInstance {
      comm_W: Commitment::<E>::default(),
      comm_E: Commitment::<E>::default(),
      T: E::Scalar::ZERO,
      X: vec![E::Scalar::ZERO; S.S.num_io],
      u: E::Scalar::ZERO,
    }
  }

  /// Fold the instance with another instance
  pub fn fold(
    &self,
    U2: &R1CSInstance<E>,
    comm_E: &Commitment<E>,
    r_b: &E::Scalar,
    T_out: &E::Scalar,
  ) -> Result<Self, NovaError> {
    // we need to compute the weighted sum using weights of (1-r_b) and r_b
    // TODO: reduce number of ops
    let comm_W = self.comm_W * (E::Scalar::ONE - r_b) + U2.comm_W * *r_b;
    let comm_E = self.comm_E * (E::Scalar::ONE - r_b) + *comm_E * *r_b;
    let X = self
      .X
      .par_iter()
      .zip(U2.X.par_iter())
      .map(|(x1, x2)| (E::Scalar::ONE - r_b) * x1 + *r_b * x2)
      .collect::<Vec<_>>();
    let u = (E::Scalar::ONE - r_b) * self.u + r_b;

    Ok(Self {
      comm_W,
      comm_E,
      T: *T_out,
      X,
      u,
    })
  }
}

impl<E: Engine> AbsorbInROTrait<E> for FoldedInstance<E> {
  fn absorb_in_ro(&self, ro: &mut E::RO) {
    self.comm_W.absorb_in_ro(ro);
    self.comm_E.absorb_in_ro(ro);

    // absorb T in bignum format
    let limbs: Vec<E::Scalar> =
      nat_to_limbs(&f_to_nat(&self.T), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
    for limb in limbs {
      ro.absorb(scalar_as_base::<E>(limb));
    }

    // absorb each element of self.X in bignum format
    for x in &self.X {
      let limbs: Vec<E::Scalar> = nat_to_limbs(&f_to_nat(x), BN_LIMB_WIDTH, BN_N_LIMBS).unwrap();
      for limb in limbs {
        ro.absorb(scalar_as_base::<E>(limb));
      }
    }

    ro.absorb(scalar_as_base::<E>(self.u));
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::{
    frontend::{
      r1cs::{NovaShape, NovaWitness},
      shape_cs::ShapeCS,
      solver::SatisfyingAssignment,
      Circuit, ConstraintSystem,
    },
    provider::{hyperkzg::EvaluationEngine, Bn256EngineKZG},
    spartan::{direct::DirectCircuit, snark::RelaxedR1CSSNARK},
    spartan::{math::Math, polys::eq::EqPolynomial},
    traits::{circuit::NonTrivialCircuit, snark::RelaxedR1CSSNARKTrait},
  };
  use rand::rngs::OsRng;

  fn test_sat_inner<E: Engine, S: RelaxedR1CSSNARKTrait<E>>() -> Result<(), NovaError> {
    // generate a non-trivial circuit
    let num_cons: usize = 16;
    let log_num_cons = num_cons.log_2();

    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));

    // synthesize the circuit's shape
    let mut cs: ShapeCS<E> = ShapeCS::new();
    let _ = circuit.synthesize(&mut cs);
    let (shape, ck) = cs.r1cs_shape(&*S::ck_floor());
    let S = Structure::new(&shape);

    // test default instance-witness pair under the structure
    let W = FoldedWitness::default(&S);
    let U = FoldedInstance::default(&S);
    S.is_sat(&ck, &U, &W)?;

    // generate a satisfying instance-witness for the r1cs
    let circuit: DirectCircuit<E, NonTrivialCircuit<E::Scalar>> =
      DirectCircuit::new(None, NonTrivialCircuit::<E::Scalar>::new(num_cons));
    let mut cs = SatisfyingAssignment::<E>::new();
    let _ = circuit.synthesize(&mut cs);
    let (u, w) = cs
      .r1cs_instance_and_witness(&shape, &ck)
      .map_err(|_e| NovaError::UnSat {
        reason: "Unable to generate a satisfying witness".to_string(),
      })?;

    // generate a random eq polynomial
    let coords = (0..log_num_cons)
      .map(|_| E::Scalar::random(&mut OsRng))
      .collect::<Vec<_>>();
    let E = EqPolynomial::new(coords).evals();

    let W = FoldedWitness {
      W: w.W.clone(),
      r_W: w.r_W,
      E: E.clone(),
      r_E: E::Scalar::random(&mut OsRng),
    };

    let U = FoldedInstance {
      comm_W: u.comm_W,
      comm_E: E::CE::commit(&ck, &E, &W.r_E),
      T: E::Scalar::ZERO,
      X: u.X.clone(),
      u: E::Scalar::ONE,
    };

    S.is_sat(&ck, &U, &W)
  }

  #[test]
  fn test_sat() {
    type E = Bn256EngineKZG;
    type S = RelaxedR1CSSNARK<E, EvaluationEngine<E>>;
    let res = test_sat_inner::<E, S>();
    assert!(res.is_ok());
  }
}
