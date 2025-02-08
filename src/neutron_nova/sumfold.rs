use crate::{
  constants::NUM_CHALLENGE_BITS,
  errors::NovaError,
  gadgets::utils::scalar_as_base,
  r1cs::{R1CSInstance, R1CSShape, R1CSWitness},
  spartan::polys::{
    eq::EqPolynomial, multilinear::MultilinearPolynomial, power::PowPoly, univariate::UniPoly,
  },
  traits::{AbsorbInROTrait, Engine, ROConstants, ROTrait},
};
use ff::Field;
use itertools::Itertools;
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
    ro: &mut E::RO,
    ro_consts: &ROConstants<E>,
    T1: E::Scalar,
    T2: E::Scalar,
  ) -> Result<(E::Scalar, E::Scalar, E::Scalar), NovaError> {
    let beta = ro.squeeze(NUM_CHALLENGE_BITS);

    // check first claim
    {
      let ml_poly = MultilinearPolynomial::new(vec![T1, T2]);
      if ml_poly.evaluate(&[beta]) != self.uni_poly.eval_at_zero() + self.uni_poly.eval_at_one() {
        return Err(NovaError::InvalidSumcheckProof);
      }
    }
    let mut ro = E::RO::new(ro_consts.clone());
    ro.absorb(scalar_as_base::<E>(beta));
    <UniPoly<E::Scalar> as AbsorbInROTrait<E>>::absorb_in_ro(&self.uni_poly, &mut ro);
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS);
    let c = self.uni_poly.evaluate(&r_b);
    Ok((c, beta, r_b))
  }
  /// Implements the sumfold prover over structure (F′(g, h_1 ,h_ 2,g_pc, h_pc_1,,h_pc_2), G′(w ,e ,w_pc, e_pc,(x ,x_pc)))
  pub fn prove_zerofold<PCFunc, R1CSFunc>(
    ro: &mut E::RO,
    ro_consts: &ROConstants<E>,
    g: &R1CSSumfoldInputs<E>,
    h: &R1CSSumfoldInputs<E>,
    g_claim: E::Scalar,
    comb_func_r1cs: R1CSFunc,
    g_pc: &PCSumFoldInputs<E>,
    h_pc: &PCSumFoldInputs<E>,
    g_claim_pc: E::Scalar,
    comb_func_pc: PCFunc,
    gamma: E::Scalar,
  ) -> Result<(SumFoldProof<E>, E::Scalar, E::Scalar, E::Scalar), NovaError>
  where
    PCFunc: Fn(E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) -> E::Scalar,
    R1CSFunc: Fn(E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) -> E::Scalar,
  {
    let beta = ro.squeeze(NUM_CHALLENGE_BITS);

    // Get uni-poly evals from the 1 round of sumcheck
    let r1cs_evals = Self::sumfold_evals(g, h, comb_func_r1cs, g_claim)?;
    let pc_evals = Self::sumfold_evals(g_pc, h_pc, comb_func_pc, g_claim_pc)?;

    // Compute the sumfold proof
    let eq_poly = EqPolynomial::new(vec![beta]);
    let uni_poly_evals = r1cs_evals
      .iter()
      .zip(pc_evals.iter())
      .enumerate()
      .map(|(i, (r1cs, pc))| {
        let eq_poly_eval = eq_poly.evaluate(&[E::Scalar::from(i as u64)]);
        eq_poly_eval * (*r1cs + gamma * pc)
      })
      .collect_vec();
    let uni_poly = UniPoly::vandermonde_interpolation(&uni_poly_evals);

    // Squeeze out r_b which is used in the folding for the running zero-fold instance and witness
    let mut ro = E::RO::new(ro_consts.clone());
    ro.absorb(scalar_as_base::<E>(beta));
    <UniPoly<E::Scalar> as AbsorbInROTrait<E>>::absorb_in_ro(&uni_poly, &mut ro);
    let r_b = ro.squeeze(NUM_CHALLENGE_BITS);

    // Get T & T_pc to send to the verifier to confirm T_gamma = T + gamma * T_pc
    let T = UniPoly::vandermonde_interpolation(&r1cs_evals).evaluate(&r_b);
    let T_pc = UniPoly::vandermonde_interpolation(&pc_evals).evaluate(&r_b);
    Ok((SumFoldProof { uni_poly }, r_b, T, T_pc))
  }

  /// Compute the polynomial evalations needed for the 1 round of sumcheck in the sumfold proof
  pub fn sumfold_evals<F>(
    g: &impl SumFoldInputTrait<E>,
    h: &impl SumFoldInputTrait<E>,
    comb_func: F,
    g_claim: E::Scalar,
  ) -> Result<Vec<E::Scalar>, NovaError>
  where
    F: Fn(E::Scalar, E::Scalar, E::Scalar, E::Scalar, E::Scalar) -> E::Scalar,
  {
    let num_vars = g.num_vars();
    let dimension_space = 2usize.pow(num_vars as u32);
    let (g0, g1, g2, g3, g4) = g.inner();
    let (h0, h1, h2, h3, h4) = h.inner();

    let mut eval_point_0 = E::Scalar::ZERO;
    let mut eval_point_2 = E::Scalar::ZERO;
    let mut eval_point_3 = E::Scalar::ZERO;
    let mut eval_point_4 = E::Scalar::ZERO;
    let mut eval_point_5 = E::Scalar::ZERO;
    for b in 0..dimension_space {
      eval_point_0 += comb_func(g0[b], g1[b], g2[b], g3[b], g4[b]);

      // eval 2: bound_func is -A(low) + 2*A(high)
      let poly_A_bound_point = h0[b] + h0[b] - g0[b];
      let poly_B_bound_point = h1[b] + h1[b] - g1[b];
      let poly_C_bound_point = h2[b] + h2[b] - g2[b];
      let poly_D_bound_point = h3[b] + h3[b] - g3[b];
      let poly_E_bound_point = h4[b] + h4[b] - g4[b];
      eval_point_2 += comb_func(
        poly_A_bound_point,
        poly_B_bound_point,
        poly_C_bound_point,
        poly_D_bound_point,
        poly_E_bound_point,
      );

      // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
      let poly_A_bound_point = poly_A_bound_point + h0[b] - g0[b];
      let poly_B_bound_point = poly_B_bound_point + h1[b] - g1[b];
      let poly_C_bound_point = poly_C_bound_point + h2[b] - g2[b];
      let poly_D_bound_point = poly_D_bound_point + h3[b] - g3[b];
      let poly_E_bound_point = poly_E_bound_point + h4[b] - g4[b];
      eval_point_3 += comb_func(
        poly_A_bound_point,
        poly_B_bound_point,
        poly_C_bound_point,
        poly_D_bound_point,
        poly_E_bound_point,
      );

      // eval 4: bound_func is -3A(low) + 4A(high);
      let poly_A_bound_point = poly_A_bound_point + h0[b] - g0[b];
      let poly_B_bound_point = poly_B_bound_point + h1[b] - g1[b];
      let poly_C_bound_point = poly_C_bound_point + h2[b] - g2[b];
      let poly_D_bound_point = poly_D_bound_point + h3[b] - g3[b];
      let poly_E_bound_point = poly_E_bound_point + h4[b] - g4[b];
      eval_point_4 += comb_func(
        poly_A_bound_point,
        poly_B_bound_point,
        poly_C_bound_point,
        poly_D_bound_point,
        poly_E_bound_point,
      );

      // eval 5: bound_func is -4A(low) + 5A(high);
      let poly_A_bound_point = poly_A_bound_point + h0[b] - g0[b];
      let poly_B_bound_point = poly_B_bound_point + h1[b] - g1[b];
      let poly_C_bound_point = poly_C_bound_point + h2[b] - g2[b];
      let poly_D_bound_point = poly_D_bound_point + h3[b] - g3[b];
      let poly_E_bound_point = poly_E_bound_point + h4[b] - g4[b];
      eval_point_5 += comb_func(
        poly_A_bound_point,
        poly_B_bound_point,
        poly_C_bound_point,
        poly_D_bound_point,
        poly_E_bound_point,
      );
    }
    let uni_poly_evals = vec![
      eval_point_0,
      g_claim - eval_point_0,
      eval_point_2,
      eval_point_3,
      eval_point_4,
      eval_point_5,
    ];

    Ok(uni_poly_evals)
  }

  pub fn nsc_to_sumfold_inputs(
    S: &R1CSShape<E>,
    U: &R1CSInstance<E>,
    W: &R1CSWitness<E>,
    e: &PowPoly<E::Scalar>,
  ) -> Result<R1CSSumfoldInputs<E>, NovaError> {
    let z = [W.W.clone(), vec![E::Scalar::ONE], U.X.clone()].concat();
    let (az, bz, cz) = S.multiply_vec(&z)?;
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

  pub fn nsc_pc_to_sumfold_inputs(
    e: &PowPoly<E::Scalar>,
    new_e: &PowPoly<E::Scalar>,
  ) -> Result<PCSumFoldInputs<E>, NovaError> {
    let (g1, g2, g3) = e.pc_to_zc()?;
    let (h1, h2) = new_e.h_polys();
    let pc_sf_inputs = PCSumFoldInputs { g1, g2, g3, h1, h2 };
    Ok(pc_sf_inputs)
  }
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

impl<E> SumFoldInputTrait<E> for R1CSSumfoldInputs<E>
where
  E: Engine,
{
  fn num_vars(&self) -> usize {
    self.az.get_num_vars()
  }

  fn inner(
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

impl<E> SumFoldInputTrait<E> for PCSumFoldInputs<E>
where
  E: Engine,
{
  fn num_vars(&self) -> usize {
    self.g1.get_num_vars()
  }

  fn inner(
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

pub trait SumFoldInputTrait<E>
where
  E: Engine,
{
  fn num_vars(&self) -> usize;
  fn inner(
    &self,
  ) -> (
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
    &MultilinearPolynomial<E::Scalar>,
  );
}
