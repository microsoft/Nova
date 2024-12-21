use super::running_instance::{NSCInstance, NSCPCWitness, NSCWitness};
use crate::{
  errors::NovaError,
  r1cs::R1CSShape,
  spartan::polys::{eq::EqPolynomial, multilinear::MultilinearPolynomial, univariate::UniPoly},
  traits::{Engine, TranscriptEngineTrait},
};
use ff::Field;
use serde::{Deserialize, Serialize};

/// SumFold proof
///
/// Which is a one round sumcheck proof
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct SumFoldProof<E>
where
  E: Engine,
{
  pub(crate) uni_poly: UniPoly<E::Scalar>,
}

impl<E> SumFoldProof<E>
where
  E: Engine,
{
  pub fn verify(
    &self,
    transcript: &mut E::TE,
    T1: E::Scalar,
    T2: E::Scalar,
  ) -> Result<(E::Scalar, E::Scalar, E::Scalar), NovaError> {
    let beta = transcript.squeeze(b"beta")?;
    // check first claim
    {
      let ml_poly = MultilinearPolynomial::new(vec![T1, T2]);

      if ml_poly.evaluate(&[beta]) != self.uni_poly.eval_at_zero() + self.uni_poly.eval_at_one() {
        return Err(NovaError::InvalidSumcheckProof);
      }
    }
    transcript.absorb(b"uni_poly", &self.uni_poly);
    let r_b = transcript.squeeze(b"r_b")?;

    let c = self.uni_poly.evaluate(&r_b);

    Ok((c, beta, r_b))
  }
}

pub fn sumfold_r1cs_prover<E, F>(
  transcript: &mut E::TE,
  g: &R1CSSumfoldInputs<E>,
  h: &R1CSSumfoldInputs<E>,
  comb_func: F,
) -> Result<(E::Scalar, SumFoldProof<E>, E::Scalar, E::Scalar), NovaError>
//
where
  E: Engine,
  F: Fn(E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) -> E::Scalar,
{
  let beta = transcript.squeeze(b"beta")?;

  let num_vars = g.num_vars();
  let dimension_space = 2usize.pow(num_vars as u32);
  let mut uni_poly_evals = vec![E::Scalar::ZERO; 6];
  let (g0, g1, g2, g3, g4) = g.inner();
  let (h0, h1, h2, h3, h4) = h.inner();

  // TODO: refactor this
  for b in 0..dimension_space {
    uni_poly_evals[0] += comb_func(g0[b], g1[b], g2[b], g3[b], g4[b]);
    uni_poly_evals[1] += comb_func(h0[b], h1[b], h2[b], h3[b], h4[b]);
    // f(2, x) = (1 - 2) • g(x) + 2 • h(x)
    //         = 2 • h(x) - g(x)
    uni_poly_evals[2] += comb_func(
      E::Scalar::from(2u64) * h0[b] - g0[b],
      E::Scalar::from(2u64) * h1[b] - g1[b],
      E::Scalar::from(2u64) * h2[b] - g2[b],
      E::Scalar::from(2u64) * h3[b] - g3[b],
      E::Scalar::from(2u64) * h4[b] - g4[b],
    );

    // f(3, x) = (1 - 3) • g(x) + 3 • h(x)
    //         = 3 • h(x) - 2 • g(x)
    uni_poly_evals[3] += comb_func(
      E::Scalar::from(3u64) * h0[b] - E::Scalar::from(2u64) * g0[b],
      E::Scalar::from(3u64) * h1[b] - E::Scalar::from(2u64) * g1[b],
      E::Scalar::from(3u64) * h2[b] - E::Scalar::from(2u64) * g2[b],
      E::Scalar::from(3u64) * h3[b] - E::Scalar::from(2u64) * g3[b],
      E::Scalar::from(3u64) * h4[b] - E::Scalar::from(2u64) * g4[b],
    );

    uni_poly_evals[4] += comb_func(
      E::Scalar::from(4u64) * h0[b] - E::Scalar::from(3u64) * g0[b],
      E::Scalar::from(4u64) * h1[b] - E::Scalar::from(3u64) * g1[b],
      E::Scalar::from(4u64) * h2[b] - E::Scalar::from(3u64) * g2[b],
      E::Scalar::from(4u64) * h3[b] - E::Scalar::from(3u64) * g3[b],
      E::Scalar::from(4u64) * h4[b] - E::Scalar::from(3u64) * g4[b],
    );

    // TODO: should probably remove this evaluation calculation. Will double check with @srinathsetty
    uni_poly_evals[5] += comb_func(
      E::Scalar::from(5u64) * h0[b] - E::Scalar::from(4u64) * g0[b],
      E::Scalar::from(5u64) * h1[b] - E::Scalar::from(4u64) * g1[b],
      E::Scalar::from(5u64) * h2[b] - E::Scalar::from(4u64) * g2[b],
      E::Scalar::from(5u64) * h3[b] - E::Scalar::from(4u64) * g3[b],
      E::Scalar::from(5u64) * h4[b] - E::Scalar::from(4u64) * g4[b],
    );
  }

  let uni_poly_evals: Vec<E::Scalar> = uni_poly_evals
    .into_iter()
    .enumerate()
    .map(|(i, v)| EqPolynomial::new(vec![beta]).evaluate(&[E::Scalar::from(i as u64)]) * v)
    .collect();

  let uni_poly = UniPoly::vandermonde_interpolation(&uni_poly_evals);
  transcript.absorb(b"uni_poly", &uni_poly);
  let r_b = transcript.squeeze(b"r_b")?;

  Ok((
    uni_poly.evaluate(&r_b),
    SumFoldProof { uni_poly },
    beta,
    r_b,
  ))
}

pub fn sumfold_pc_prover<E, F>(
  transcript: &mut E::TE,
  g: &PCSumFoldInputs<E>,
  h: &PCSumFoldInputs<E>,
  comb_func: F,
) -> Result<(E::Scalar, SumFoldProof<E>, E::Scalar, E::Scalar), NovaError>
//
where
  E: Engine,
  F: Fn(E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) -> E::Scalar,
{
  let beta = transcript.squeeze(b"beta")?;

  let num_vars = g.num_vars();
  let dimension_space = 2usize.pow(num_vars as u32);
  let mut uni_poly_evals = vec![E::Scalar::ZERO; 6];
  let (g0, g1, g2, g3, g4) = g.inner();
  let (h0, h1, h2, h3, h4) = h.inner();

  // TODO: refactor this
  for b in 0..dimension_space {
    uni_poly_evals[0] += comb_func(g0[b], g1[b], g2[b], g3[b], g4[b]);
    uni_poly_evals[1] += comb_func(h0[b], h1[b], h2[b], h3[b], h4[b]);
    // f(2, x) = (1 - 2) • g(x) + 2 • h(x)
    //         = 2 • h(x) - g(x)
    uni_poly_evals[2] += comb_func(
      E::Scalar::from(2u64) * h0[b] - g0[b],
      E::Scalar::from(2u64) * h1[b] - g1[b],
      E::Scalar::from(2u64) * h2[b] - g2[b],
      E::Scalar::from(2u64) * h3[b] - g3[b],
      E::Scalar::from(2u64) * h4[b] - g4[b],
    );

    // f(3, x) = (1 - 3) • g(x) + 3 • h(x)
    //         = 3 • h(x) - 2 • g(x)
    uni_poly_evals[3] += comb_func(
      E::Scalar::from(3u64) * h0[b] - E::Scalar::from(2u64) * g0[b],
      E::Scalar::from(3u64) * h1[b] - E::Scalar::from(2u64) * g1[b],
      E::Scalar::from(3u64) * h2[b] - E::Scalar::from(2u64) * g2[b],
      E::Scalar::from(3u64) * h3[b] - E::Scalar::from(2u64) * g3[b],
      E::Scalar::from(3u64) * h4[b] - E::Scalar::from(2u64) * g4[b],
    );

    uni_poly_evals[4] += comb_func(
      E::Scalar::from(4u64) * h0[b] - E::Scalar::from(3u64) * g0[b],
      E::Scalar::from(4u64) * h1[b] - E::Scalar::from(3u64) * g1[b],
      E::Scalar::from(4u64) * h2[b] - E::Scalar::from(3u64) * g2[b],
      E::Scalar::from(4u64) * h3[b] - E::Scalar::from(3u64) * g3[b],
      E::Scalar::from(4u64) * h4[b] - E::Scalar::from(3u64) * g4[b],
    );

    uni_poly_evals[5] += comb_func(
      E::Scalar::from(5u64) * h0[b] - E::Scalar::from(4u64) * g0[b],
      E::Scalar::from(5u64) * h1[b] - E::Scalar::from(4u64) * g1[b],
      E::Scalar::from(5u64) * h2[b] - E::Scalar::from(4u64) * g2[b],
      E::Scalar::from(5u64) * h3[b] - E::Scalar::from(4u64) * g3[b],
      E::Scalar::from(5u64) * h4[b] - E::Scalar::from(4u64) * g4[b],
    );
  }

  let uni_poly_evals: Vec<E::Scalar> = uni_poly_evals
    .into_iter()
    .enumerate()
    .map(|(i, v)| EqPolynomial::new(vec![beta]).evaluate(&[E::Scalar::from(i as u64)]) * v)
    .collect();

  let uni_poly = UniPoly::vandermonde_interpolation(&uni_poly_evals);
  transcript.absorb(b"uni_poly", &uni_poly);

  let r_b = transcript.squeeze(b"r_b")?;

  Ok((
    uni_poly.evaluate(&r_b),
    SumFoldProof { uni_poly },
    beta,
    r_b,
  ))
}

// G(w,x)
pub fn nsc_to_sumfold_inputs<E>(
  S: &R1CSShape<E>,
  NSC_U: &NSCInstance<E>,
  NSC_W: &NSCWitness<E>,
) -> Result<R1CSSumfoldInputs<E>, NovaError>
where
  E: Engine,
{
  let (U, W) = (NSC_U.U(), NSC_W.W());
  let z = [W.W.clone(), vec![E::Scalar::ONE], U.X.clone()].concat();
  let (az, bz, cz) = S.multiply_vec(&z)?;

  let e = NSC_W.e();

  let (h1, h2) = e.h_polys();

  let r1cs_sumfold_inputs = R1CSSumfoldInputs {
    az: MultilinearPolynomial::new(az),
    bz: MultilinearPolynomial::new(bz),
    cz: MultilinearPolynomial::new(cz),
    h1,
    h2,
  };

  Ok(r1cs_sumfold_inputs)
}

pub fn nsc_pc_to_sumfold_inputs<E>(W_pc: &NSCPCWitness<E>) -> Result<PCSumFoldInputs<E>, NovaError>
where
  E: Engine,
{
  let (g1, g2, g3) = W_pc.e().pc_to_zc()?;
  let (h1, h2) = W_pc.new_e().h_polys();

  let pc_sf_inputs = PCSumFoldInputs { g1, g2, g3, h1, h2 };
  Ok(pc_sf_inputs)
}

pub struct R1CSSumfoldInputs<E>
where
  E: Engine,
{
  pub az: MultilinearPolynomial<E::Scalar>,
  pub bz: MultilinearPolynomial<E::Scalar>,
  pub cz: MultilinearPolynomial<E::Scalar>,
  pub h1: MultilinearPolynomial<E::Scalar>,
  pub h2: MultilinearPolynomial<E::Scalar>,
}

impl<E> R1CSSumfoldInputs<E>
where
  E: Engine,
{
  pub fn num_vars(&self) -> usize {
    self.az.get_num_vars()
  }

  pub fn inner(
    &self,
  ) -> (
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
  ) {
    (&self.az, &self.bz, &self.cz, &self.h1, &self.h2)
  }
}

#[derive(Debug, Clone)]
pub struct PCSumFoldInputs<E>
where
  E: Engine,
{
  pub g1: MultilinearPolynomial<E::Scalar>,
  pub g2: MultilinearPolynomial<E::Scalar>,
  pub g3: MultilinearPolynomial<E::Scalar>,
  pub h1: MultilinearPolynomial<E::Scalar>,
  pub h2: MultilinearPolynomial<E::Scalar>,
}

impl<E> PCSumFoldInputs<E>
where
  E: Engine,
{
  pub fn num_vars(&self) -> usize {
    self.g1.get_num_vars()
  }

  pub fn inner(
    &self,
  ) -> (
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
  ) {
    (&self.g1, &self.g2, &self.g3, &self.h1, &self.h2)
  }
}
