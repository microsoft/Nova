use crate::{
  errors::NovaError,
  spartan::{
    math::Math,
    polynomial::{EqPolynomial, MultilinearPolynomial},
    sumcheck::{CompressedUniPoly, SumcheckProof, UniPoly},
  },
  traits::{Group, TranscriptEngineTrait},
};
use core::marker::PhantomData;
use ff::{Field, PrimeField};
use serde::{Deserialize, Serialize};

pub(crate) struct IdentityPolynomial<Scalar: PrimeField> {
  ell: usize,
  _p: PhantomData<Scalar>,
}

impl<Scalar: PrimeField> IdentityPolynomial<Scalar> {
  pub fn new(ell: usize) -> Self {
    IdentityPolynomial {
      ell,
      _p: Default::default(),
    }
  }

  pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
    assert_eq!(self.ell, r.len());
    (0..self.ell)
      .map(|i| Scalar::from(2_usize.pow((self.ell - i - 1) as u32) as u64) * r[i])
      .fold(Scalar::zero(), |acc, item| acc + item)
  }
}

impl<G: Group> SumcheckProof<G> {
  pub fn prove_cubic<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_A: &mut MultilinearPolynomial<G::Scalar>,
    poly_B: &mut MultilinearPolynomial<G::Scalar>,
    poly_C: &mut MultilinearPolynomial<G::Scalar>,
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, Vec<G::Scalar>), NovaError>
  where
    F: Fn(&G::Scalar, &G::Scalar, &G::Scalar) -> G::Scalar,
  {
    let mut e = *claim;
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<G>> = Vec::new();
    for _j in 0..num_rounds {
      let mut eval_point_0 = G::Scalar::zero();
      let mut eval_point_2 = G::Scalar::zero();
      let mut eval_point_3 = G::Scalar::zero();

      let len = poly_A.len() / 2;
      for i in 0..len {
        // eval 0: bound_func is A(low)
        eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
        eval_point_2 += comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];

        eval_point_3 += comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );
      }

      let evals = vec![eval_point_0, e - eval_point_0, eval_point_2, eval_point_3];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      //derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);

      // bound all tables to the verifier's challenege
      poly_A.bound_poly_var_top(&r_i);
      poly_B.bound_poly_var_top(&r_i);
      poly_C.bound_poly_var_top(&r_i);
      e = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    Ok((
      Self::new(cubic_polys),
      r,
      vec![poly_A[0], poly_B[0], poly_C[0]],
    ))
  }

  pub fn prove_cubic_batched<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_vec: (
      &mut Vec<&mut MultilinearPolynomial<G::Scalar>>,
      &mut Vec<&mut MultilinearPolynomial<G::Scalar>>,
      &mut MultilinearPolynomial<G::Scalar>,
    ),
    coeffs: &[G::Scalar],
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<
    (
      Self,
      Vec<G::Scalar>,
      (Vec<G::Scalar>, Vec<G::Scalar>, G::Scalar),
    ),
    NovaError,
  >
  where
    F: Fn(&G::Scalar, &G::Scalar, &G::Scalar) -> G::Scalar,
  {
    let (poly_A_vec, poly_B_vec, poly_C) = poly_vec;

    let mut e = *claim;
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<G>> = Vec::new();

    for _j in 0..num_rounds {
      let mut evals: Vec<(G::Scalar, G::Scalar, G::Scalar)> = Vec::new();

      for (poly_A, poly_B) in poly_A_vec.iter().zip(poly_B_vec.iter()) {
        let mut eval_point_0 = G::Scalar::zero();
        let mut eval_point_2 = G::Scalar::zero();
        let mut eval_point_3 = G::Scalar::zero();

        let len = poly_A.len() / 2;
        for i in 0..len {
          // eval 0: bound_func is A(low)
          eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);

          // eval 2: bound_func is -A(low) + 2*A(high)
          let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
          eval_point_2 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
          );

          // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
          let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];

          eval_point_3 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
          );
        }

        evals.push((eval_point_0, eval_point_2, eval_point_3));
      }

      let evals_combined_0 = (0..evals.len())
        .map(|i| evals[i].0 * coeffs[i])
        .fold(G::Scalar::zero(), |acc, item| acc + item);
      let evals_combined_2 = (0..evals.len())
        .map(|i| evals[i].1 * coeffs[i])
        .fold(G::Scalar::zero(), |acc, item| acc + item);
      let evals_combined_3 = (0..evals.len())
        .map(|i| evals[i].2 * coeffs[i])
        .fold(G::Scalar::zero(), |acc, item| acc + item);

      let evals = vec![
        evals_combined_0,
        e - evals_combined_0,
        evals_combined_2,
        evals_combined_3,
      ];
      let poly = UniPoly::from_evals(&evals);

      // append the prover's message to the transcript
      transcript.absorb(b"p", &poly);

      // derive the verifier's challenge for the next round
      let r_i = transcript.squeeze(b"c")?;
      r.push(r_i);

      // bound all tables to the verifier's challenege
      for (poly_A, poly_B) in poly_A_vec.iter_mut().zip(poly_B_vec.iter_mut()) {
        poly_A.bound_poly_var_top(&r_i);
        poly_B.bound_poly_var_top(&r_i);
      }
      poly_C.bound_poly_var_top(&r_i);

      e = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    let poly_A_final = (0..poly_A_vec.len()).map(|i| poly_A_vec[i][0]).collect();
    let poly_B_final = (0..poly_B_vec.len()).map(|i| poly_B_vec[i][0]).collect();
    let claims_prod = (poly_A_final, poly_B_final, poly_C[0]);

    Ok((SumcheckProof::new(cubic_polys), r, claims_prod))
  }
}

#[derive(Debug)]
pub struct ProductArgumentInputs<G: Group> {
  left_vec: Vec<MultilinearPolynomial<G::Scalar>>,
  right_vec: Vec<MultilinearPolynomial<G::Scalar>>,
}

impl<G: Group> ProductArgumentInputs<G> {
  fn compute_layer(
    inp_left: &MultilinearPolynomial<G::Scalar>,
    inp_right: &MultilinearPolynomial<G::Scalar>,
  ) -> (
    MultilinearPolynomial<G::Scalar>,
    MultilinearPolynomial<G::Scalar>,
  ) {
    let len = inp_left.len() + inp_right.len();
    let outp_left = (0..len / 4)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<G::Scalar>>();
    let outp_right = (len / 4..len / 2)
      .map(|i| inp_left[i] * inp_right[i])
      .collect::<Vec<G::Scalar>>();

    (
      MultilinearPolynomial::new(outp_left),
      MultilinearPolynomial::new(outp_right),
    )
  }

  pub fn new(poly: &MultilinearPolynomial<G::Scalar>) -> Self {
    let mut left_vec: Vec<MultilinearPolynomial<G::Scalar>> = Vec::new();
    let mut right_vec: Vec<MultilinearPolynomial<G::Scalar>> = Vec::new();
    let num_layers = poly.len().log_2();
    let (outp_left, outp_right) = poly.split(poly.len() / 2);

    left_vec.push(outp_left);
    right_vec.push(outp_right);

    for i in 0..num_layers - 1 {
      let (outp_left, outp_right) =
        ProductArgumentInputs::<G>::compute_layer(&left_vec[i], &right_vec[i]);
      left_vec.push(outp_left);
      right_vec.push(outp_right);
    }

    Self {
      left_vec,
      right_vec,
    }
  }

  pub fn evaluate(&self) -> G::Scalar {
    let len = self.left_vec.len();
    assert_eq!(self.left_vec[len - 1].get_num_vars(), 0);
    assert_eq!(self.right_vec[len - 1].get_num_vars(), 0);
    self.left_vec[len - 1][0] * self.right_vec[len - 1][0]
  }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct LayerProofBatched<G: Group> {
  proof: SumcheckProof<G>,
  claims_prod_left: Vec<G::Scalar>,
  claims_prod_right: Vec<G::Scalar>,
}

impl<G: Group> LayerProofBatched<G> {
  pub fn verify(
    &self,
    claim: G::Scalar,
    num_rounds: usize,
    degree_bound: usize,
    transcript: &mut G::TE,
  ) -> Result<(G::Scalar, Vec<G::Scalar>), NovaError> {
    self
      .proof
      .verify(claim, num_rounds, degree_bound, transcript)
  }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub(crate) struct ProductArgumentBatched<G: Group> {
  proof: Vec<LayerProofBatched<G>>,
}

impl<G: Group> ProductArgumentBatched<G> {
  pub fn prove(
    poly_vec: &[&MultilinearPolynomial<G::Scalar>],
    transcript: &mut G::TE,
  ) -> Result<(Self, Vec<G::Scalar>, Vec<G::Scalar>), NovaError> {
    let mut prod_circuit_vec: Vec<_> = (0..poly_vec.len())
      .map(|i| ProductArgumentInputs::<G>::new(poly_vec[i]))
      .collect();

    let mut proof_layers: Vec<LayerProofBatched<G>> = Vec::new();
    let num_layers = prod_circuit_vec[0].left_vec.len();
    let evals = (0..prod_circuit_vec.len())
      .map(|i| prod_circuit_vec[i].evaluate())
      .collect::<Vec<G::Scalar>>();

    let mut claims_to_verify = evals.clone();
    let mut rand = Vec::new();
    for layer_id in (0..num_layers).rev() {
      let len = prod_circuit_vec[0].left_vec[layer_id].len()
        + prod_circuit_vec[0].right_vec[layer_id].len();

      let mut poly_C = MultilinearPolynomial::new(EqPolynomial::new(rand.clone()).evals());
      assert_eq!(poly_C.len(), len / 2);

      let num_rounds_prod = poly_C.len().log_2();
      let comb_func_prod = |poly_A_comp: &G::Scalar,
                            poly_B_comp: &G::Scalar,
                            poly_C_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };

      let mut poly_A_batched: Vec<&mut MultilinearPolynomial<G::Scalar>> = Vec::new();
      let mut poly_B_batched: Vec<&mut MultilinearPolynomial<G::Scalar>> = Vec::new();
      for prod_circuit in prod_circuit_vec.iter_mut() {
        poly_A_batched.push(&mut prod_circuit.left_vec[layer_id]);
        poly_B_batched.push(&mut prod_circuit.right_vec[layer_id])
      }
      let poly_vec = (&mut poly_A_batched, &mut poly_B_batched, &mut poly_C);

      // produce a fresh set of coeffs and a joint claim
      let coeff_vec = {
        let s = transcript.squeeze(b"r")?;
        let mut s_vec = vec![s];
        for i in 1..claims_to_verify.len() {
          s_vec.push(s_vec[i - 1] * s);
        }
        s_vec
      };

      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .fold(G::Scalar::zero(), |acc, item| acc + item);

      let (proof, rand_prod, claims_prod) = SumcheckProof::prove_cubic_batched(
        &claim,
        num_rounds_prod,
        poly_vec,
        &coeff_vec,
        comb_func_prod,
        transcript,
      )?;

      let (claims_prod_left, claims_prod_right, _claims_eq) = claims_prod;

      let v = {
        let mut v = claims_prod_left.clone();
        v.extend(&claims_prod_right);
        v
      };
      transcript.absorb(b"p", &v.as_slice());

      // produce a random challenge to condense two claims into a single claim
      let r_layer = transcript.squeeze(b"c")?;

      claims_to_verify = (0..prod_circuit_vec.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<G::Scalar>>();

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;

      proof_layers.push(LayerProofBatched {
        proof,
        claims_prod_left,
        claims_prod_right,
      });
    }

    Ok((
      ProductArgumentBatched {
        proof: proof_layers,
      },
      evals,
      rand,
    ))
  }

  pub fn verify(
    &self,
    claims_prod_vec: &[G::Scalar],
    len: usize,
    transcript: &mut G::TE,
  ) -> Result<(Vec<G::Scalar>, Vec<G::Scalar>), NovaError> {
    let num_layers = len.log_2();

    let mut rand: Vec<G::Scalar> = Vec::new();
    if self.proof.len() != num_layers {
      return Err(NovaError::InvalidProductProof);
    }

    let mut claims_to_verify = claims_prod_vec.to_owned();
    for (num_rounds, i) in (0..num_layers).enumerate() {
      // produce random coefficients, one for each instance
      let coeff_vec = {
        let s = transcript.squeeze(b"r")?;
        let mut s_vec = vec![s];
        for i in 1..claims_to_verify.len() {
          s_vec.push(s_vec[i - 1] * s);
        }
        s_vec
      };

      // produce a joint claim
      let claim = (0..claims_to_verify.len())
        .map(|i| claims_to_verify[i] * coeff_vec[i])
        .fold(G::Scalar::zero(), |acc, item| acc + item);

      let (claim_last, rand_prod) = self.proof[i].verify(claim, num_rounds, 3, transcript)?;

      let claims_prod_left = &self.proof[i].claims_prod_left;
      let claims_prod_right = &self.proof[i].claims_prod_right;
      if claims_prod_left.len() != claims_prod_vec.len()
        || claims_prod_right.len() != claims_prod_vec.len()
      {
        return Err(NovaError::InvalidProductProof);
      }

      let v = {
        let mut v = claims_prod_left.clone();
        v.extend(claims_prod_right);
        v
      };
      transcript.absorb(b"p", &v.as_slice());

      if rand.len() != rand_prod.len() {
        return Err(NovaError::InvalidProductProof);
      }

      let eq: G::Scalar = (0..rand.len())
        .map(|i| {
          rand[i] * rand_prod[i] + (G::Scalar::one() - rand[i]) * (G::Scalar::one() - rand_prod[i])
        })
        .fold(G::Scalar::one(), |acc, item| acc * item);
      let claim_expected: G::Scalar = (0..claims_prod_vec.len())
        .map(|i| coeff_vec[i] * (claims_prod_left[i] * claims_prod_right[i] * eq))
        .fold(G::Scalar::zero(), |acc, item| acc + item);

      if claim_expected != claim_last {
        return Err(NovaError::InvalidProductProof);
      }

      // produce a random challenge
      let r_layer = transcript.squeeze(b"c")?;

      claims_to_verify = (0..claims_prod_left.len())
        .map(|i| claims_prod_left[i] + r_layer * (claims_prod_right[i] - claims_prod_left[i]))
        .collect::<Vec<G::Scalar>>();

      let mut ext = vec![r_layer];
      ext.extend(rand_prod);
      rand = ext;
    }
    Ok((claims_to_verify, rand))
  }
}
