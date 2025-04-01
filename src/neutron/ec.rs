use crate::{
  errors::NovaError,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  traits::Engine,
  Commitment, CommitmentKey,
};

#[derive(Clone, Debug)]
pub struct ECShape<E: Engine> {
  S: R1CSShape<E>,
}

#[derive(Clone, Debug)]
pub struct ECWitness<E: Engine> {
  W: Vec<E::Scalar>,
}

#[derive(Clone, Debug)]
pub struct ECInstance<E: Engine> {
  comm_W: Commitment<E>,

  // public inputs
  r: E::Scalar,
  coords: [E::Scalar; 6],
  is_inf: [E::Scalar; 3],
}

#[derive(Clone, Debug)]
pub struct RelaxedECWitness<E: Engine> {
  W: Vec<E::Scalar>,
  E: Vec<E::Scalar>,
}

pub struct RelaxedECInstance<E: Engine> {
  comm_W: Commitment<E>,
  comm_T: Commitment<E>,
  r: E::Scalar,
  coords: [E::Scalar; 6],
  is_inf: [E::Scalar; 3],
} 


impl<E: Engine> From<R1CSShape<E>> for ECShape<E> {
  fn from(shape: R1CSShape<E>) -> Self {
    ECShape { S: shape }
  }
}

impl<E: Engine> From<R1CSInstance<E>> for ECInstance<E> {
  fn from(instance: R1CSInstance<E>) -> Self {
    // Extract comm_W from the R1CSInstance
    let comm_W = instance.comm_W;

    // Parse public inputs X, assuming they're in order: r, P_x, P_y, P_z, Q_x, Q_y, Q_z, R_x, R_y, R_z
    assert!(
      instance.X.len() >= 10,
      "Invalid public inputs length for ECInstance conversion"
    );

    let r = instance.X[0];
    let coords = [
      instance.X[1],
      instance.X[2],
      instance.X[4],
      instance.X[5],
      instance.X[7],
      instance.X[8],
    ];

    let is_inf = [instance.X[3], instance.X[6], instance.X[9]];

    ECInstance {
      comm_W,
      r,
      coords,
      is_inf,
    }
  }
}

impl<E: Engine> ECShape<E> {
  pub fn new(S: R1CSShape<E>) -> Self {
    ECShape { S }
  }

  pub fn is_sat(
    &self,
    ck: &CommitmentKey<E>,
    W: &ECWitness<E>,
    U: &ECInstance<E>,
  ) -> Result<(), NovaError> {
    // convert ECInstance to R1CSInstance
    let U = R1CSInstance::new(
      &self.S,
      &U.comm_W,
      &[
        U.r,
        U.coords[0],
        U.coords[1],
        U.is_inf[0],
        U.coords[2],
        U.coords[3],
        U.is_inf[1],
        U.coords[4],
        U.coords[5],
        U.is_inf[2],
      ],
    )?;

    let W = R1CSWitness::new(&self.S, &W.W)?;

    self.S.is_sat(ck, &U, &W)
  }
}
