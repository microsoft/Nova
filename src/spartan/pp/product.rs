use crate::{
  errors::NovaError,
  spartan::{
    math::Math,
    polynomial::{EqPolynomial, MultilinearPolynomial},
    sumcheck::{CompressedUniPoly, SumcheckProof, UniPoly},
    PolyEvalInstance, PolyEvalWitness,
  },
  traits::{commitment::CommitmentEngineTrait, Group, TranscriptEngineTrait},
  Commitment, CommitmentKey,
};
use core::marker::PhantomData;
use ff::{Field, PrimeField};
use rayon::prelude::*;
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
  pub fn prove_cubic_with_additive_term_batched<F>(
    claim: &G::Scalar,
    num_rounds: usize,
    poly_vec: (
      &mut MultilinearPolynomial<G::Scalar>,
      &mut Vec<MultilinearPolynomial<G::Scalar>>,
      &mut Vec<MultilinearPolynomial<G::Scalar>>,
      &mut Vec<MultilinearPolynomial<G::Scalar>>,
    ),
    coeffs: &[G::Scalar],
    comb_func: F,
    transcript: &mut G::TE,
  ) -> Result<
    (
      Self,
      Vec<G::Scalar>,
      (G::Scalar, Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>),
    ),
    NovaError,
  >
  where
    F: Fn(&G::Scalar, &G::Scalar, &G::Scalar, &G::Scalar) -> G::Scalar,
  {
    let (poly_A, poly_B_vec, poly_C_vec, poly_D_vec) = poly_vec;

    let mut e = *claim;
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<G>> = Vec::new();

    for _j in 0..num_rounds {
      let mut evals: Vec<(G::Scalar, G::Scalar, G::Scalar)> = Vec::new();

      for ((poly_B, poly_C), poly_D) in poly_B_vec
        .iter()
        .zip(poly_C_vec.iter())
        .zip(poly_D_vec.iter())
      {
        let mut eval_point_0 = G::Scalar::zero();
        let mut eval_point_2 = G::Scalar::zero();
        let mut eval_point_3 = G::Scalar::zero();

        let len = poly_A.len() / 2;
        for i in 0..len {
          // eval 0: bound_func is A(low)
          eval_point_0 += comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

          // eval 2: bound_func is -A(low) + 2*A(high)
          let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
          let poly_D_bound_point = poly_D[len + i] + poly_D[len + i] - poly_D[i];

          eval_point_2 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
            &poly_D_bound_point,
          );

          // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
          let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
          let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
          let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];
          let poly_D_bound_point = poly_D_bound_point + poly_D[len + i] - poly_D[i];

          eval_point_3 += comb_func(
            &poly_A_bound_point,
            &poly_B_bound_point,
            &poly_C_bound_point,
            &poly_D_bound_point,
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
      poly_A.bound_poly_var_top(&r_i);
      for ((poly_B, poly_C), poly_D) in poly_B_vec
        .iter_mut()
        .zip(poly_C_vec.iter_mut())
        .zip(poly_D_vec.iter_mut())
      {
        poly_B.bound_poly_var_top(&r_i);
        poly_C.bound_poly_var_top(&r_i);
        poly_D.bound_poly_var_top(&r_i);
      }

      e = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    let poly_B_final = (0..poly_B_vec.len()).map(|i| poly_B_vec[i][0]).collect();
    let poly_C_final = (0..poly_C_vec.len()).map(|i| poly_C_vec[i][0]).collect();
    let poly_D_final = (0..poly_D_vec.len()).map(|i| poly_D_vec[i][0]).collect();
    let claims_prod = (poly_A[0], poly_B_final, poly_C_final, poly_D_final);

    Ok((SumcheckProof::new(cubic_polys), r, claims_prod))
  }
}

/// Provides a product argument using the algorithm described by Setty-Lee, 2020
#[derive(Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProductArgument<G: Group> {
  comm_output_vec: Vec<Commitment<G>>,
  sc_proof: SumcheckProof<G>,
  eval_left_vec: Vec<G::Scalar>,
  eval_right_vec: Vec<G::Scalar>,
  eval_output_vec: Vec<G::Scalar>,
  eval_input_vec: Vec<G::Scalar>,
  eval_output2_vec: Vec<G::Scalar>,
}

impl<G: Group> ProductArgument<G> {
  pub fn prove(
    ck: &CommitmentKey<G>,
    input_vec: &[Vec<G::Scalar>], // a commitment to the input and the input vector to multiplied together
    transcript: &mut G::TE,
  ) -> Result<
    (
      Self,
      Vec<G::Scalar>,
      Vec<G::Scalar>,
      Vec<G::Scalar>,
      Vec<(PolyEvalWitness<G>, PolyEvalInstance<G>)>,
    ),
    NovaError,
  > {
    let num_claims = input_vec.len();

    let compute_layer = |input: &[G::Scalar]| -> (Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>) {
      let left = (0..input.len() / 2)
        .map(|i| input[2 * i])
        .collect::<Vec<G::Scalar>>();

      let right = (0..input.len() / 2)
        .map(|i| input[2 * i + 1])
        .collect::<Vec<G::Scalar>>();

      assert_eq!(left.len(), right.len());

      let output = (0..left.len())
        .map(|i| left[i] * right[i])
        .collect::<Vec<G::Scalar>>();

      (left, right, output)
    };

    // a closure that returns left, right, output, product
    let prepare_inputs =
      |input: &[G::Scalar]| -> (Vec<G::Scalar>, Vec<G::Scalar>, Vec<G::Scalar>, G::Scalar) {
        let mut left: Vec<G::Scalar> = Vec::new();
        let mut right: Vec<G::Scalar> = Vec::new();
        let mut output: Vec<G::Scalar> = Vec::new();

        let mut out = input.to_vec();
        for _i in 0..input.len().log_2() {
          let (l, r, o) = compute_layer(&out);
          out = o.clone();

          left.extend(l);
          right.extend(r);
          output.extend(o);
        }

        // add a dummy product operation to make the left.len() == right.len() == output.len() == input.len()
        left.push(output[output.len() - 1]);
        right.push(G::Scalar::zero());
        output.push(G::Scalar::zero());

        // output is stored at the last but one position
        let product = output[output.len() - 2];

        assert_eq!(left.len(), right.len());
        assert_eq!(left.len(), output.len());
        (left, right, output, product)
      };

    let mut left_vec = Vec::new();
    let mut right_vec = Vec::new();
    let mut output_vec = Vec::new();
    let mut prod_vec = Vec::new();
    for input in input_vec {
      let (l, r, o, p) = prepare_inputs(input);
      left_vec.push(l);
      right_vec.push(r);
      output_vec.push(o);
      prod_vec.push(p);
    }

    // commit to the outputs
    let comm_output_vec = (0..output_vec.len())
      .into_par_iter()
      .map(|i| G::CE::commit(ck, &output_vec[i]))
      .collect::<Vec<_>>();

    // absorb the output commitment and the claimed product
    transcript.absorb(b"o", &comm_output_vec.as_slice());
    transcript.absorb(b"r", &prod_vec.as_slice());

    // this assumes all vectors passed have the same length
    let num_rounds = output_vec[0].len().log_2();

    // produce a fresh set of coeffs and a joint claim
    let coeff_vec = {
      let s = transcript.squeeze(b"r")?;
      let mut s_vec = vec![s];
      for i in 1..num_claims {
        s_vec.push(s_vec[i - 1] * s);
      }
      s_vec
    };

    // generate randomness for the eq polynomial
    let rand_eq = (0..num_rounds)
      .map(|_i| transcript.squeeze(b"e"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    let mut poly_A = MultilinearPolynomial::new(EqPolynomial::new(rand_eq).evals());
    let mut poly_B_vec = left_vec
      .clone()
      .into_par_iter()
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();
    let mut poly_C_vec = right_vec
      .clone()
      .into_par_iter()
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();
    let mut poly_D_vec = output_vec
      .clone()
      .into_par_iter()
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();

    let comb_func =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    let (sc_proof, rand, _claims) = SumcheckProof::prove_cubic_with_additive_term_batched(
      &G::Scalar::zero(),
      num_rounds,
      (
        &mut poly_A,
        &mut poly_B_vec,
        &mut poly_C_vec,
        &mut poly_D_vec,
      ),
      &coeff_vec,
      comb_func,
      transcript,
    )?;

    // claims[0] is about the Eq polynomial, which the verifier computes directly
    // claims[1] =? weighed sum of left(rand)
    // claims[2] =? weighted sum of right(rand)
    // claims[3] =? weighetd sum of output(rand), which is easy to verify by querying output
    // we also need to prove that output(output.len()-2) = claimed_product
    let eval_left_vec = (0..left_vec.len())
      .into_par_iter()
      .map(|i| MultilinearPolynomial::evaluate_with(&left_vec[i], &rand))
      .collect::<Vec<G::Scalar>>();
    let eval_right_vec = (0..right_vec.len())
      .into_par_iter()
      .map(|i| MultilinearPolynomial::evaluate_with(&right_vec[i], &rand))
      .collect::<Vec<G::Scalar>>();
    let eval_output_vec = (0..output_vec.len())
      .into_par_iter()
      .map(|i| MultilinearPolynomial::evaluate_with(&output_vec[i], &rand))
      .collect::<Vec<G::Scalar>>();

    // we now combine eval_left = left(rand) and eval_right = right(rand)
    // into claims about input and output
    transcript.absorb(b"l", &eval_left_vec.as_slice());
    transcript.absorb(b"r", &eval_right_vec.as_slice());
    transcript.absorb(b"o", &eval_output_vec.as_slice());

    let c = transcript.squeeze(b"c")?;

    // eval = (G::Scalar::one() - c) * eval_left + c * eval_right
    // eval is claimed evaluation of input||output(r, c), which can be proven by proving input(r[1..], c) and output(r[1..], c)
    let rand_ext = {
      let mut r = rand.clone();
      r.extend(&[c]);
      r
    };
    let eval_input_vec = (0..input_vec.len())
      .into_par_iter()
      .map(|i| MultilinearPolynomial::evaluate_with(&input_vec[i], &rand_ext[1..]))
      .collect::<Vec<G::Scalar>>();

    let eval_output2_vec = (0..output_vec.len())
      .into_par_iter()
      .map(|i| MultilinearPolynomial::evaluate_with(&output_vec[i], &rand_ext[1..]))
      .collect::<Vec<G::Scalar>>();

    // add claimed evaluations to the transcript
    transcript.absorb(b"i", &eval_input_vec.as_slice());
    transcript.absorb(b"o", &eval_output2_vec.as_slice());

    // squeeze a challenge to combine multiple claims into one
    let powers_of_rho = {
      let s = transcript.squeeze(b"r")?;
      let mut s_vec = vec![s];
      for i in 1..num_claims {
        s_vec.push(s_vec[i - 1] * s);
      }
      s_vec
    };

    // take weighted sum of input, output, and their commitments
    let product = prod_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let eval_output = eval_output_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let comm_output = comm_output_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(c, r_i)| *c * *r_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);

    let weighted_sum = |W: &[Vec<G::Scalar>], s: &[G::Scalar]| -> Vec<G::Scalar> {
      assert_eq!(W.len(), s.len());
      let mut p = vec![G::Scalar::zero(); W[0].len()];
      for i in 0..W.len() {
        for (j, item) in W[i].iter().enumerate().take(W[i].len()) {
          p[j] += *item * s[i]
        }
      }
      p
    };

    let poly_output = weighted_sum(&output_vec, &powers_of_rho);

    let eval_output2 = eval_output2_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let mut w_u_vec = Vec::new();

    // eval_output = output(rand)
    w_u_vec.push((
      PolyEvalWitness {
        p: poly_output.clone(),
      },
      PolyEvalInstance {
        c: comm_output,
        x: rand.clone(),
        e: eval_output,
      },
    ));

    // claimed_product = output(1, ..., 1, 0)
    let x = {
      let mut x = vec![G::Scalar::one(); rand.len()];
      x[rand.len() - 1] = G::Scalar::zero();
      x
    };
    w_u_vec.push((
      PolyEvalWitness {
        p: poly_output.clone(),
      },
      PolyEvalInstance {
        c: comm_output,
        x,
        e: product,
      },
    ));

    // eval_output2 = output(rand_ext[1..])
    w_u_vec.push((
      PolyEvalWitness { p: poly_output },
      PolyEvalInstance {
        c: comm_output,
        x: rand_ext[1..].to_vec(),
        e: eval_output2,
      },
    ));

    let prod_arg = Self {
      comm_output_vec,
      sc_proof,

      // claimed evaluations at rand
      eval_left_vec,
      eval_right_vec,
      eval_output_vec,

      // claimed evaluations at rand_ext[1..]
      eval_input_vec: eval_input_vec.clone(),
      eval_output2_vec,
    };

    Ok((
      prod_arg,
      prod_vec,
      rand_ext[1..].to_vec(),
      eval_input_vec,
      w_u_vec,
    ))
  }

  pub fn verify(
    &self,
    prod_vec: &[G::Scalar], // claimed products
    len: usize,
    transcript: &mut G::TE,
  ) -> Result<(Vec<G::Scalar>, Vec<G::Scalar>, Vec<PolyEvalInstance<G>>), NovaError> {
    // absorb the provided commitment and claimed output
    transcript.absorb(b"o", &self.comm_output_vec.as_slice());
    transcript.absorb(b"r", &prod_vec.to_vec().as_slice());

    let num_rounds = len.log_2();
    let num_claims = prod_vec.len();

    // produce a fresh set of coeffs and a joint claim
    let coeff_vec = {
      let s = transcript.squeeze(b"r")?;
      let mut s_vec = vec![s];
      for i in 1..num_claims {
        s_vec.push(s_vec[i - 1] * s);
      }
      s_vec
    };

    // generate randomness for the eq polynomial
    let rand_eq = (0..num_rounds)
      .map(|_i| transcript.squeeze(b"e"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    let (final_claim, rand) = self.sc_proof.verify(
      G::Scalar::zero(), // claim
      num_rounds,
      3, // degree bound
      transcript,
    )?;

    // verify the final claim along with output[output.len() - 2 ] = claim
    let eq = EqPolynomial::new(rand_eq).evaluate(&rand);
    let final_claim_expected = (0..num_claims)
      .map(|i| {
        coeff_vec[i]
          * eq
          * (self.eval_left_vec[i] * self.eval_right_vec[i] - self.eval_output_vec[i])
      })
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    if final_claim != final_claim_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(b"l", &self.eval_left_vec.as_slice());
    transcript.absorb(b"r", &self.eval_right_vec.as_slice());
    transcript.absorb(b"o", &self.eval_output_vec.as_slice());

    let c = transcript.squeeze(b"c")?;
    let eval_vec = self
      .eval_left_vec
      .iter()
      .zip(self.eval_right_vec.iter())
      .map(|(l, r)| (G::Scalar::one() - c) * l + c * r)
      .collect::<Vec<G::Scalar>>();

    // eval is claimed evaluation of input||output(r, c), which can be proven by proving input(r[1..], c) and output(r[1..], c)
    let rand_ext = {
      let mut r = rand.clone();
      r.extend(&[c]);
      r
    };

    for (i, eval) in eval_vec.iter().enumerate() {
      if *eval
        != (G::Scalar::one() - rand_ext[0]) * self.eval_input_vec[i]
          + rand_ext[0] * self.eval_output2_vec[i]
      {
        return Err(NovaError::InvalidSumcheckProof);
      }
    }

    transcript.absorb(b"i", &self.eval_input_vec.as_slice());
    transcript.absorb(b"o", &self.eval_output2_vec.as_slice());

    // squeeze a challenge to combine multiple claims into one
    let powers_of_rho = {
      let s = transcript.squeeze(b"r")?;
      let mut s_vec = vec![s];
      for i in 1..num_claims {
        s_vec.push(s_vec[i - 1] * s);
      }
      s_vec
    };

    // take weighted sum of input, output, and their commitments
    let product = prod_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let eval_output = self
      .eval_output_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let comm_output = self
      .comm_output_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(c, r_i)| *c * *r_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);

    let eval_output2 = self
      .eval_output2_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .fold(G::Scalar::zero(), |acc, item| acc + item);

    let mut u_vec = Vec::new();

    // eval_output = output(rand)
    u_vec.push(PolyEvalInstance {
      c: comm_output,
      x: rand.clone(),
      e: eval_output,
    });

    // claimed_product = output(1, ..., 1, 0)
    let x = {
      let mut x = vec![G::Scalar::one(); rand.len()];
      x[rand.len() - 1] = G::Scalar::zero();
      x
    };
    u_vec.push(PolyEvalInstance {
      c: comm_output,
      x,
      e: product,
    });

    // eval_output2 = output(rand_ext[1..])
    u_vec.push(PolyEvalInstance {
      c: comm_output,
      x: rand_ext[1..].to_vec(),
      e: eval_output2,
    });

    // input-related claims are checked by the caller
    Ok((self.eval_input_vec.clone(), rand_ext[1..].to_vec(), u_vec))
  }
}
