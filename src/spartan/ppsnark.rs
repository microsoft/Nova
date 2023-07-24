//! This module implements RelaxedR1CSSNARK traits using a spark-based approach to prove evaluations of
//! sparse multilinear polynomials involved in Spartan's sum-check protocol, thereby providing a preprocessing SNARK
//! The verifier in this preprocessing SNARK maintains a commitment to R1CS matrices. This is beneficial when using a
//! polynomial commitment scheme in which the verifier's costs is succinct.
use crate::{
  compute_digest,
  errors::NovaError,
  r1cs::{R1CSShape, RelaxedR1CSInstance, RelaxedR1CSWitness},
  spartan::{
    math::Math,
    polynomial::{EqPolynomial, MultilinearPolynomial},
    powers,
    sumcheck::{CompressedUniPoly, SumcheckProof, UniPoly},
    PolyEvalInstance, PolyEvalWitness, SparsePolynomial,
  },
  traits::{
    commitment::{CommitmentEngineTrait, CommitmentTrait},
    evaluation::EvaluationEngineTrait,
    snark::RelaxedR1CSSNARKTrait,
    Group, TranscriptEngineTrait, TranscriptReprTrait,
  },
  Commitment, CommitmentKey, CompressedCommitment,
};
use core::{cmp::max, marker::PhantomData};
use ff::{Field, PrimeField};
use itertools::concat;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

fn vec_to_arr<T, const N: usize>(v: Vec<T>) -> [T; N] {
  v.try_into()
    .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

struct IdentityPolynomial<Scalar: PrimeField> {
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
      .fold(Scalar::ZERO, |acc, item| acc + item)
  }
}

/// A type that holds R1CSShape in a form amenable to memory checking
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkRepr<G: Group> {
  N: usize, // size of the vectors

  // dense representation
  row: Vec<G::Scalar>,
  col: Vec<G::Scalar>,
  val_A: Vec<G::Scalar>,
  val_B: Vec<G::Scalar>,
  val_C: Vec<G::Scalar>,

  // timestamp polynomials
  row_read_ts: Vec<G::Scalar>,
  row_audit_ts: Vec<G::Scalar>,
  col_read_ts: Vec<G::Scalar>,
  col_audit_ts: Vec<G::Scalar>,
}

/// A type that holds a commitment to a sparse polynomial
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct R1CSShapeSparkCommitment<G: Group> {
  N: usize, // size of each vector

  // commitments to the dense representation
  comm_row: Commitment<G>,
  comm_col: Commitment<G>,
  comm_val_A: Commitment<G>,
  comm_val_B: Commitment<G>,
  comm_val_C: Commitment<G>,

  // commitments to the timestamp polynomials
  comm_row_read_ts: Commitment<G>,
  comm_row_audit_ts: Commitment<G>,
  comm_col_read_ts: Commitment<G>,
  comm_col_audit_ts: Commitment<G>,
}

impl<G: Group> TranscriptReprTrait<G> for R1CSShapeSparkCommitment<G> {
  fn to_transcript_bytes(&self) -> Vec<u8> {
    [
      self.comm_row,
      self.comm_col,
      self.comm_val_A,
      self.comm_val_B,
      self.comm_val_C,
      self.comm_row_read_ts,
      self.comm_row_audit_ts,
      self.comm_col_read_ts,
      self.comm_col_audit_ts,
    ]
    .as_slice()
    .to_transcript_bytes()
  }
}

impl<G: Group> R1CSShapeSparkRepr<G> {
  /// represents R1CSShape in a Spark-friendly format amenable to memory checking
  pub fn new(S: &R1CSShape<G>) -> R1CSShapeSparkRepr<G> {
    let N = {
      let total_nz = S.A.len() + S.B.len() + S.C.len();
      max(total_nz, max(2 * S.num_vars, S.num_cons)).next_power_of_two()
    };

    let row = {
      let mut r = S
        .A
        .iter()
        .chain(S.B.iter())
        .chain(S.C.iter())
        .map(|(r, _, _)| *r)
        .collect::<Vec<usize>>();
      r.resize(N, 0usize);
      r
    };

    let col = {
      let mut c = S
        .A
        .iter()
        .chain(S.B.iter())
        .chain(S.C.iter())
        .map(|(_, c, _)| *c)
        .collect::<Vec<usize>>();
      c.resize(N, 0usize);
      c
    };

    let val_A = {
      let mut val = S.A.iter().map(|(_, _, v)| *v).collect::<Vec<G::Scalar>>();
      val.resize(N, G::Scalar::ZERO);
      val
    };

    let val_B = {
      // prepend zeros
      let mut val = vec![G::Scalar::ZERO; S.A.len()];
      val.extend(S.B.iter().map(|(_, _, v)| *v).collect::<Vec<G::Scalar>>());
      // append zeros
      val.resize(N, G::Scalar::ZERO);
      val
    };

    let val_C = {
      // prepend zeros
      let mut val = vec![G::Scalar::ZERO; S.A.len() + S.B.len()];
      val.extend(S.C.iter().map(|(_, _, v)| *v).collect::<Vec<G::Scalar>>());
      // append zeros
      val.resize(N, G::Scalar::ZERO);
      val
    };

    // timestamp calculation routine
    let timestamp_calc =
      |num_ops: usize, num_cells: usize, addr_trace: &[usize]| -> (Vec<usize>, Vec<usize>) {
        let mut read_ts = vec![0usize; num_ops];
        let mut audit_ts = vec![0usize; num_cells];

        assert!(num_ops >= addr_trace.len());
        for i in 0..addr_trace.len() {
          let addr = addr_trace[i];
          assert!(addr < num_cells);
          let r_ts = audit_ts[addr];
          read_ts[i] = r_ts;

          let w_ts = r_ts + 1;
          audit_ts[addr] = w_ts;
        }
        (read_ts, audit_ts)
      };

    // timestamp polynomials for row
    let (row_read_ts, row_audit_ts) = timestamp_calc(N, N, &row);
    let (col_read_ts, col_audit_ts) = timestamp_calc(N, N, &col);

    // a routine to turn a vector of usize into a vector scalars
    let to_vec_scalar = |v: &[usize]| -> Vec<G::Scalar> {
      (0..v.len())
        .map(|i| G::Scalar::from(v[i] as u64))
        .collect::<Vec<G::Scalar>>()
    };

    R1CSShapeSparkRepr {
      N,

      // dense representation
      row: to_vec_scalar(&row),
      col: to_vec_scalar(&col),
      val_A,
      val_B,
      val_C,

      // timestamp polynomials
      row_read_ts: to_vec_scalar(&row_read_ts),
      row_audit_ts: to_vec_scalar(&row_audit_ts),
      col_read_ts: to_vec_scalar(&col_read_ts),
      col_audit_ts: to_vec_scalar(&col_audit_ts),
    }
  }

  fn commit(&self, ck: &CommitmentKey<G>) -> R1CSShapeSparkCommitment<G> {
    let comm_vec: Vec<Commitment<G>> = [
      &self.row,
      &self.col,
      &self.val_A,
      &self.val_B,
      &self.val_C,
      &self.row_read_ts,
      &self.row_audit_ts,
      &self.col_read_ts,
      &self.col_audit_ts,
    ]
    .par_iter()
    .map(|v| G::CE::commit(ck, v))
    .collect();

    R1CSShapeSparkCommitment {
      N: self.row.len(),
      comm_row: comm_vec[0],
      comm_col: comm_vec[1],
      comm_val_A: comm_vec[2],
      comm_val_B: comm_vec[3],
      comm_val_C: comm_vec[4],
      comm_row_read_ts: comm_vec[5],
      comm_row_audit_ts: comm_vec[6],
      comm_col_read_ts: comm_vec[7],
      comm_col_audit_ts: comm_vec[8],
    }
  }

  // computes evaluation oracles
  fn evaluation_oracles(
    &self,
    S: &R1CSShape<G>,
    r_x: &[G::Scalar],
    z: &[G::Scalar],
  ) -> (
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
    Vec<G::Scalar>,
  ) {
    let r_x_padded = {
      let mut x = vec![G::Scalar::ZERO; self.N.log_2() - r_x.len()];
      x.extend(r_x);
      x
    };

    let mem_row = EqPolynomial::new(r_x_padded).evals();
    let mem_col = {
      let mut z = z.to_vec();
      z.resize(self.N, G::Scalar::ZERO);
      z
    };

    let mut E_row = S
      .A
      .iter()
      .chain(S.B.iter())
      .chain(S.C.iter())
      .map(|(r, _, _)| mem_row[*r])
      .collect::<Vec<G::Scalar>>();

    let mut E_col = S
      .A
      .iter()
      .chain(S.B.iter())
      .chain(S.C.iter())
      .map(|(_, c, _)| mem_col[*c])
      .collect::<Vec<G::Scalar>>();

    E_row.resize(self.N, mem_row[0]); // we place mem_row[0] since resized row is appended with 0s
    E_col.resize(self.N, mem_col[0]);

    (mem_row, mem_col, E_row, E_col)
  }
}

/// Defines a trait for implementing sum-check in a generic manner
pub trait SumcheckEngine<G: Group> {
  /// returns the initial claims
  fn initial_claims(&self) -> Vec<G::Scalar>;

  /// degree of the sum-check polynomial
  fn degree(&self) -> usize;

  /// the size of the polynomials
  fn size(&self) -> usize;

  /// returns evaluation points at 0, 2, d-1 (where d is the degree of the sum-check polynomial)
  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>>;

  /// bounds a variable in the constituent polynomials
  fn bound(&mut self, r: &G::Scalar);

  /// returns the final claims
  fn final_claims(&self) -> Vec<Vec<G::Scalar>>;
}

struct ProductSumcheckInstance<G: Group> {
  pub(crate) claims: Vec<G::Scalar>, // claimed products
  pub(crate) comm_output_vec: Vec<Commitment<G>>,

  input_vec: Vec<Vec<G::Scalar>>,
  output_vec: Vec<Vec<G::Scalar>>,

  poly_A: MultilinearPolynomial<G::Scalar>,
  poly_B_vec: Vec<MultilinearPolynomial<G::Scalar>>,
  poly_C_vec: Vec<MultilinearPolynomial<G::Scalar>>,
  poly_D_vec: Vec<MultilinearPolynomial<G::Scalar>>,
}

impl<G: Group> ProductSumcheckInstance<G> {
  pub fn new(
    ck: &CommitmentKey<G>,
    input_vec: Vec<Vec<G::Scalar>>, // list of input vectors
    transcript: &mut G::TE,
  ) -> Result<Self, NovaError> {
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
        right.push(G::Scalar::ZERO);
        output.push(G::Scalar::ZERO);

        // output is stored at the last but one position
        let product = output[output.len() - 2];

        assert_eq!(left.len(), right.len());
        assert_eq!(left.len(), output.len());
        (left, right, output, product)
      };

    let mut left_vec = Vec::new();
    let mut right_vec = Vec::new();
    let mut output_vec = Vec::new();
    let mut claims = Vec::new();
    for input in input_vec.iter() {
      let (l, r, o, p) = prepare_inputs(input);
      left_vec.push(l);
      right_vec.push(r);
      output_vec.push(o);
      claims.push(p);
    }

    // commit to the outputs
    let comm_output_vec = (0..output_vec.len())
      .into_par_iter()
      .map(|i| G::CE::commit(ck, &output_vec[i]))
      .collect::<Vec<_>>();

    // absorb the output commitment and the claimed product
    transcript.absorb(b"o", &comm_output_vec.as_slice());
    transcript.absorb(b"c", &claims.as_slice());

    // generate randomness for the eq polynomial
    let num_rounds = output_vec[0].len().log_2();
    let rand_eq = (0..num_rounds)
      .map(|_i| transcript.squeeze(b"e"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    let poly_A = MultilinearPolynomial::new(EqPolynomial::new(rand_eq).evals());
    let poly_B_vec = left_vec
      .clone()
      .into_par_iter()
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();
    let poly_C_vec = right_vec
      .clone()
      .into_par_iter()
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();
    let poly_D_vec = output_vec
      .clone()
      .into_par_iter()
      .map(MultilinearPolynomial::new)
      .collect::<Vec<_>>();

    Ok(Self {
      claims,
      comm_output_vec,
      input_vec,
      output_vec,
      poly_A,
      poly_B_vec,
      poly_C_vec,
      poly_D_vec,
    })
  }
}

impl<G: Group> SumcheckEngine<G> for ProductSumcheckInstance<G> {
  fn initial_claims(&self) -> Vec<G::Scalar> {
    vec![G::Scalar::ZERO; 8]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    for poly_B in &self.poly_B_vec {
      assert_eq!(poly_B.len(), self.poly_A.len());
    }
    for poly_C in &self.poly_C_vec {
      assert_eq!(poly_C.len(), self.poly_A.len());
    }
    for poly_D in &self.poly_D_vec {
      assert_eq!(poly_D.len(), self.poly_A.len());
    }
    self.poly_A.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>> {
    let poly_A = &self.poly_A;

    let comb_func =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };

    self
      .poly_B_vec
      .iter()
      .zip(self.poly_C_vec.iter())
      .zip(self.poly_D_vec.iter())
      .map(|((poly_B, poly_C), poly_D)| {
        let len = poly_B.len() / 2;
        // Make an iterator returning the contributions to the evaluations
        let (eval_point_0, eval_point_2, eval_point_3) = (0..len)
          .into_par_iter()
          .map(|i| {
            // eval 0: bound_func is A(low)
            let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

            // eval 2: bound_func is -A(low) + 2*A(high)
            let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
            let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
            let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
            let poly_D_bound_point = poly_D[len + i] + poly_D[len + i] - poly_D[i];
            let eval_point_2 = comb_func(
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
            let eval_point_3 = comb_func(
              &poly_A_bound_point,
              &poly_B_bound_point,
              &poly_C_bound_point,
              &poly_D_bound_point,
            );
            (eval_point_0, eval_point_2, eval_point_3)
          })
          .reduce(
            || (G::Scalar::ZERO, G::Scalar::ZERO, G::Scalar::ZERO),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
          );
        vec![eval_point_0, eval_point_2, eval_point_3]
      })
      .collect::<Vec<Vec<G::Scalar>>>()
  }

  fn bound(&mut self, r: &G::Scalar) {
    self.poly_A.bound_poly_var_top(r);
    for ((poly_B, poly_C), poly_D) in self
      .poly_B_vec
      .iter_mut()
      .zip(self.poly_C_vec.iter_mut())
      .zip(self.poly_D_vec.iter_mut())
    {
      poly_B.bound_poly_var_top(r);
      poly_C.bound_poly_var_top(r);
      poly_D.bound_poly_var_top(r);
    }
  }
  fn final_claims(&self) -> Vec<Vec<G::Scalar>> {
    let poly_A_final = vec![self.poly_A[0]];
    let poly_B_final = (0..self.poly_B_vec.len())
      .map(|i| self.poly_B_vec[i][0])
      .collect();
    let poly_C_final = (0..self.poly_C_vec.len())
      .map(|i| self.poly_C_vec[i][0])
      .collect();
    let poly_D_final = (0..self.poly_D_vec.len())
      .map(|i| self.poly_D_vec[i][0])
      .collect();

    vec![poly_A_final, poly_B_final, poly_C_final, poly_D_final]
  }
}

struct OuterSumcheckInstance<G: Group> {
  poly_tau: MultilinearPolynomial<G::Scalar>,
  poly_Az: MultilinearPolynomial<G::Scalar>,
  poly_Bz: MultilinearPolynomial<G::Scalar>,
  poly_uCz_E: MultilinearPolynomial<G::Scalar>,
}

impl<G: Group> SumcheckEngine<G> for OuterSumcheckInstance<G> {
  fn initial_claims(&self) -> Vec<G::Scalar> {
    vec![G::Scalar::ZERO]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_tau.len(), self.poly_Az.len());
    assert_eq!(self.poly_tau.len(), self.poly_Bz.len());
    assert_eq!(self.poly_tau.len(), self.poly_uCz_E.len());
    self.poly_tau.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>> {
    let (poly_A, poly_B, poly_C, poly_D) = (
      &self.poly_tau,
      &self.poly_Az,
      &self.poly_Bz,
      &self.poly_uCz_E,
    );
    let comb_func =
      |poly_A_comp: &G::Scalar,
       poly_B_comp: &G::Scalar,
       poly_C_comp: &G::Scalar,
       poly_D_comp: &G::Scalar|
       -> G::Scalar { *poly_A_comp * (*poly_B_comp * *poly_C_comp - *poly_D_comp) };
    let len = poly_A.len() / 2;

    // Make an iterator returning the contributions to the evaluations
    let (eval_point_0, eval_point_2, eval_point_3) = (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i], &poly_D[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
        let poly_D_bound_point = poly_D[len + i] + poly_D[len + i] - poly_D[i];
        let eval_point_2 = comb_func(
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
        let eval_point_3 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
          &poly_D_bound_point,
        );
        (eval_point_0, eval_point_2, eval_point_3)
      })
      .reduce(
        || (G::Scalar::ZERO, G::Scalar::ZERO, G::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      );

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &G::Scalar) {
    self.poly_tau.bound_poly_var_top(r);
    self.poly_Az.bound_poly_var_top(r);
    self.poly_Bz.bound_poly_var_top(r);
    self.poly_uCz_E.bound_poly_var_top(r);
  }

  fn final_claims(&self) -> Vec<Vec<G::Scalar>> {
    vec![vec![
      self.poly_tau[0],
      self.poly_Az[0],
      self.poly_Bz[0],
      self.poly_uCz_E[0],
    ]]
  }
}

struct InnerSumcheckInstance<G: Group> {
  claim: G::Scalar,
  poly_E_row: MultilinearPolynomial<G::Scalar>,
  poly_E_col: MultilinearPolynomial<G::Scalar>,
  poly_val: MultilinearPolynomial<G::Scalar>,
}

impl<G: Group> SumcheckEngine<G> for InnerSumcheckInstance<G> {
  fn initial_claims(&self) -> Vec<G::Scalar> {
    vec![self.claim]
  }

  fn degree(&self) -> usize {
    3
  }

  fn size(&self) -> usize {
    assert_eq!(self.poly_E_row.len(), self.poly_val.len());
    assert_eq!(self.poly_E_row.len(), self.poly_E_col.len());
    self.poly_E_row.len()
  }

  fn evaluation_points(&self) -> Vec<Vec<G::Scalar>> {
    let (poly_A, poly_B, poly_C) = (&self.poly_E_row, &self.poly_E_col, &self.poly_val);
    let comb_func = |poly_A_comp: &G::Scalar,
                     poly_B_comp: &G::Scalar,
                     poly_C_comp: &G::Scalar|
     -> G::Scalar { *poly_A_comp * *poly_B_comp * *poly_C_comp };
    let len = poly_A.len() / 2;

    // Make an iterator returning the contributions to the evaluations
    let (eval_point_0, eval_point_2, eval_point_3) = (0..len)
      .into_par_iter()
      .map(|i| {
        // eval 0: bound_func is A(low)
        let eval_point_0 = comb_func(&poly_A[i], &poly_B[i], &poly_C[i]);

        // eval 2: bound_func is -A(low) + 2*A(high)
        let poly_A_bound_point = poly_A[len + i] + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B[len + i] + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C[len + i] + poly_C[len + i] - poly_C[i];
        let eval_point_2 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );

        // eval 3: bound_func is -2A(low) + 3A(high); computed incrementally with bound_func applied to eval(2)
        let poly_A_bound_point = poly_A_bound_point + poly_A[len + i] - poly_A[i];
        let poly_B_bound_point = poly_B_bound_point + poly_B[len + i] - poly_B[i];
        let poly_C_bound_point = poly_C_bound_point + poly_C[len + i] - poly_C[i];
        let eval_point_3 = comb_func(
          &poly_A_bound_point,
          &poly_B_bound_point,
          &poly_C_bound_point,
        );
        (eval_point_0, eval_point_2, eval_point_3)
      })
      .reduce(
        || (G::Scalar::ZERO, G::Scalar::ZERO, G::Scalar::ZERO),
        |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2),
      );

    vec![vec![eval_point_0, eval_point_2, eval_point_3]]
  }

  fn bound(&mut self, r: &G::Scalar) {
    self.poly_E_row.bound_poly_var_top(r);
    self.poly_E_col.bound_poly_var_top(r);
    self.poly_val.bound_poly_var_top(r);
  }

  fn final_claims(&self) -> Vec<Vec<G::Scalar>> {
    vec![vec![
      self.poly_E_row[0],
      self.poly_E_col[0],
      self.poly_val[0],
    ]]
  }
}

/// A type that represents the prover's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct ProverKey<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  pk_ee: EE::ProverKey,
  S: R1CSShape<G>,
  S_repr: R1CSShapeSparkRepr<G>,
  S_comm: R1CSShapeSparkCommitment<G>,
  vk_digest: G::Scalar, // digest of verifier's key
}

/// A type that represents the verifier's key
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VerifierKey<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  num_cons: usize,
  num_vars: usize,
  vk_ee: EE::VerifierKey,
  S_comm: R1CSShapeSparkCommitment<G>,
  digest: G::Scalar,
}

/// A succinct proof of knowledge of a witness to a relaxed R1CS instance
/// The proof is produced using Spartan's combination of the sum-check and
/// the commitment to a vector viewed as a polynomial commitment
#[derive(Clone, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct RelaxedR1CSSNARK<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> {
  // commitment to oracles
  comm_Az: CompressedCommitment<G>,
  comm_Bz: CompressedCommitment<G>,
  comm_Cz: CompressedCommitment<G>,

  // commitment to oracles for the inner sum-check
  comm_E_row: CompressedCommitment<G>,
  comm_E_col: CompressedCommitment<G>,

  // initial claims
  eval_Az_at_tau: G::Scalar,
  eval_Bz_at_tau: G::Scalar,
  eval_Cz_at_tau: G::Scalar,

  comm_output_arr: [CompressedCommitment<G>; 8],
  claims_product_arr: [G::Scalar; 8],

  // satisfiability sum-check
  sc_sat: SumcheckProof<G>,

  // claims from the end of the sum-check
  eval_Az: G::Scalar,
  eval_Bz: G::Scalar,
  eval_Cz: G::Scalar,
  eval_E: G::Scalar,
  eval_E_row: G::Scalar,
  eval_E_col: G::Scalar,
  eval_val_A: G::Scalar,
  eval_val_B: G::Scalar,
  eval_val_C: G::Scalar,
  eval_left_arr: [G::Scalar; 8],
  eval_right_arr: [G::Scalar; 8],
  eval_output_arr: [G::Scalar; 8],
  eval_input_arr: [G::Scalar; 8],
  eval_output2_arr: [G::Scalar; 8],

  eval_row: G::Scalar,
  eval_row_read_ts: G::Scalar,
  eval_E_row_at_r_prod: G::Scalar,
  eval_row_audit_ts: G::Scalar,
  eval_col: G::Scalar,
  eval_col_read_ts: G::Scalar,
  eval_E_col_at_r_prod: G::Scalar,
  eval_col_audit_ts: G::Scalar,
  eval_W: G::Scalar,

  // batch openings of all multilinear polynomials
  sc_proof_batch: SumcheckProof<G>,
  evals_batch_arr: [G::Scalar; 7],
  eval_arg: EE::EvaluationArgument,
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> RelaxedR1CSSNARK<G, EE> {
  fn prove_inner<T1, T2, T3>(
    mem: &mut T1,
    outer: &mut T2,
    inner: &mut T3,
    transcript: &mut G::TE,
  ) -> Result<
    (
      SumcheckProof<G>,
      Vec<G::Scalar>,
      Vec<Vec<G::Scalar>>,
      Vec<Vec<G::Scalar>>,
      Vec<Vec<G::Scalar>>,
    ),
    NovaError,
  >
  where
    T1: SumcheckEngine<G>,
    T2: SumcheckEngine<G>,
    T3: SumcheckEngine<G>,
  {
    // sanity checks
    assert_eq!(mem.size(), outer.size());
    assert_eq!(mem.size(), inner.size());
    assert_eq!(mem.degree(), outer.degree());
    assert_eq!(mem.degree(), inner.degree());

    // these claims are already added to the transcript, so we do not need to add
    let claims = {
      let claims_mem = mem.initial_claims();
      let claims_outer = outer.initial_claims();
      let claims_inner = inner.initial_claims();
      claims_mem
        .into_iter()
        .chain(claims_outer)
        .chain(claims_inner)
        .collect::<Vec<G::Scalar>>()
    };

    let num_claims = claims.len();
    let coeffs = {
      let s = transcript.squeeze(b"r")?;
      let mut s_vec = vec![s];
      for i in 1..num_claims {
        s_vec.push(s_vec[i - 1] * s);
      }
      s_vec
    };

    // compute the joint claim
    let claim = claims
      .iter()
      .zip(coeffs.iter())
      .map(|(c_1, c_2)| *c_1 * c_2)
      .sum();

    let mut e = claim;
    let mut r: Vec<G::Scalar> = Vec::new();
    let mut cubic_polys: Vec<CompressedUniPoly<G>> = Vec::new();
    let num_rounds = mem.size().log_2();
    for _i in 0..num_rounds {
      let mut evals: Vec<Vec<G::Scalar>> = Vec::new();
      evals.extend(mem.evaluation_points());
      evals.extend(outer.evaluation_points());
      evals.extend(inner.evaluation_points());
      assert_eq!(evals.len(), num_claims);

      let evals_combined_0 = (0..evals.len()).map(|i| evals[i][0] * coeffs[i]).sum();
      let evals_combined_2 = (0..evals.len()).map(|i| evals[i][1] * coeffs[i]).sum();
      let evals_combined_3 = (0..evals.len()).map(|i| evals[i][2] * coeffs[i]).sum();

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

      mem.bound(&r_i);
      outer.bound(&r_i);
      inner.bound(&r_i);

      e = poly.evaluate(&r_i);
      cubic_polys.push(poly.compress());
    }

    let mem_claims = mem.final_claims();
    let outer_claims = outer.final_claims();
    let inner_claims = inner.final_claims();

    Ok((
      SumcheckProof::new(cubic_polys),
      r,
      mem_claims,
      outer_claims,
      inner_claims,
    ))
  }
}

impl<G: Group, EE: EvaluationEngineTrait<G, CE = G::CE>> RelaxedR1CSSNARKTrait<G>
  for RelaxedR1CSSNARK<G, EE>
{
  type ProverKey = ProverKey<G, EE>;
  type VerifierKey = VerifierKey<G, EE>;

  fn setup(
    ck: &CommitmentKey<G>,
    S: &R1CSShape<G>,
  ) -> Result<(Self::ProverKey, Self::VerifierKey), NovaError> {
    let (pk_ee, vk_ee) = EE::setup(ck);

    // pad the R1CS matrices
    let S = S.pad();

    let S_repr = R1CSShapeSparkRepr::new(&S);
    let S_comm = S_repr.commit(ck);

    let vk = {
      let mut vk = VerifierKey {
        num_cons: S.num_cons,
        num_vars: S.num_vars,
        S_comm: S_comm.clone(),
        vk_ee,
        digest: G::Scalar::ZERO,
      };
      vk.digest = compute_digest::<G, VerifierKey<G, EE>>(&vk);
      vk
    };

    let pk = ProverKey {
      pk_ee,
      S,
      S_repr,
      S_comm,
      vk_digest: vk.digest,
    };

    Ok((pk, vk))
  }

  /// produces a succinct proof of satisfiability of a RelaxedR1CS instance
  fn prove(
    ck: &CommitmentKey<G>,
    pk: &Self::ProverKey,
    U: &RelaxedR1CSInstance<G>,
    W: &RelaxedR1CSWitness<G>,
  ) -> Result<Self, NovaError> {
    let W = W.pad(&pk.S); // pad the witness
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");

    // a list of polynomial evaluation claims that will be batched
    let mut w_u_vec = Vec::new();

    // sanity check that R1CSShape has certain size characteristics
    pk.S.check_regular_shape();

    // append the verifier key (which includes commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &pk.vk_digest);
    transcript.absorb(b"U", U);

    // compute the full satisfying assignment by concatenating W.W, U.u, and U.X
    let z = concat(vec![W.W.clone(), vec![U.u], U.X.clone()]);

    // compute Az, Bz, Cz
    let (mut Az, mut Bz, mut Cz) = pk.S.multiply_vec(&z)?;

    // commit to Az, Bz, Cz
    let (comm_Az, (comm_Bz, comm_Cz)) = rayon::join(
      || G::CE::commit(ck, &Az),
      || rayon::join(|| G::CE::commit(ck, &Bz), || G::CE::commit(ck, &Cz)),
    );

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());

    // number of rounds of the satisfiability sum-check
    let num_rounds_sat = pk.S_repr.N.log_2();
    let tau = (0..num_rounds_sat)
      .map(|_| transcript.squeeze(b"t"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    // (1) send commitments to Az, Bz, and Cz along with their evaluations at tau
    let (Az, Bz, Cz, E) = {
      Az.resize(pk.S_repr.N, G::Scalar::ZERO);
      Bz.resize(pk.S_repr.N, G::Scalar::ZERO);
      Cz.resize(pk.S_repr.N, G::Scalar::ZERO);

      let mut E = W.E.clone();
      E.resize(pk.S_repr.N, G::Scalar::ZERO);

      (Az, Bz, Cz, E)
    };
    let (eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau) = {
      let evals_at_tau = [&Az, &Bz, &Cz]
        .into_par_iter()
        .map(|p| MultilinearPolynomial::evaluate_with(p, &tau))
        .collect::<Vec<G::Scalar>>();
      (evals_at_tau[0], evals_at_tau[1], evals_at_tau[2])
    };

    // (2) send commitments to the following two oracles
    // E_row(i) = eq(tau, row(i)) for all i
    // E_col(i) = z(col(i)) for all i
    let (mem_row, mem_col, E_row, E_col) = pk.S_repr.evaluation_oracles(&pk.S, &tau, &z);
    let (comm_E_row, comm_E_col) =
      rayon::join(|| G::CE::commit(ck, &E_row), || G::CE::commit(ck, &E_col));

    // absorb the claimed evaluations into the transcript
    transcript.absorb(
      b"e",
      &[eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau].as_slice(),
    );
    // absorb commitments to E_row and E_col in the transcript
    transcript.absorb(b"e", &vec![comm_E_row, comm_E_col].as_slice());

    // add claims about Az, Bz, and Cz to be checked later
    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![eval_Az_at_tau, eval_Bz_at_tau, eval_Cz_at_tau];
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    let poly_vec = vec![&Az, &Bz, &Cz];
    transcript.absorb(b"e", &eval_vec.as_slice()); // c_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let w = PolyEvalWitness::batch(&poly_vec, &c);
    let u = PolyEvalInstance::batch(&comm_vec, &tau, &eval_vec, &c);
    w_u_vec.push((w, u));

    let c_inner = c;

    // we now need to prove three claims
    // (1) 0 = \sum_x poly_tau(x) * (poly_Az(x) * poly_Bz(x) - poly_uCz_E(x))
    // (2) eval_Az_at_tau + r * eval_Bz_at_tau + r^2 * eval_Cz_at_tau = \sum_y E_row(y) * (val_A(y) + r * val_B(y) + r^2 * val_C(y)) * E_col(y)
    // (3) E_row(i) = eq(tau, row(i)) and E_col(i) = z(col(i))

    // a sum-check instance to prove the first claim
    let mut outer_sc_inst = OuterSumcheckInstance {
      poly_tau: MultilinearPolynomial::new(EqPolynomial::new(tau).evals()),
      poly_Az: MultilinearPolynomial::new(Az.clone()),
      poly_Bz: MultilinearPolynomial::new(Bz.clone()),
      poly_uCz_E: {
        let uCz_E = (0..Cz.len())
          .map(|i| U.u * Cz[i] + E[i])
          .collect::<Vec<G::Scalar>>();
        MultilinearPolynomial::new(uCz_E)
      },
    };

    // a sum-check instance to prove the second claim
    let val = pk
      .S_repr
      .val_A
      .iter()
      .zip(pk.S_repr.val_B.iter())
      .zip(pk.S_repr.val_C.iter())
      .map(|((v_a, v_b), v_c)| *v_a + c_inner * *v_b + c_inner * c_inner * *v_c)
      .collect::<Vec<G::Scalar>>();
    let mut inner_sc_inst = InnerSumcheckInstance {
      claim: eval_Az_at_tau + c_inner * eval_Bz_at_tau + c_inner * c_inner * eval_Cz_at_tau,
      poly_E_row: MultilinearPolynomial::new(E_row.clone()),
      poly_E_col: MultilinearPolynomial::new(E_col.clone()),
      poly_val: MultilinearPolynomial::new(val),
    };

    // a third sum-check instance to prove the memory-related claim
    // we now need to prove that E_row and E_col are well-formed
    // we use memory checking: H(INIT) * H(WS) =? H(RS) * H(FINAL)
    let gamma_1 = transcript.squeeze(b"g1")?;
    let gamma_2 = transcript.squeeze(b"g2")?;

    let gamma_1_sqr = gamma_1 * gamma_1;
    let hash_func = |addr: &G::Scalar, val: &G::Scalar, ts: &G::Scalar| -> G::Scalar {
      (*ts * gamma_1_sqr + *val * gamma_1 + *addr) - gamma_2
    };

    let init_row = (0..mem_row.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &mem_row[i], &G::Scalar::ZERO))
      .collect::<Vec<G::Scalar>>();
    let read_row = (0..E_row.len())
      .map(|i| hash_func(&pk.S_repr.row[i], &E_row[i], &pk.S_repr.row_read_ts[i]))
      .collect::<Vec<G::Scalar>>();
    let write_row = (0..E_row.len())
      .map(|i| {
        hash_func(
          &pk.S_repr.row[i],
          &E_row[i],
          &(pk.S_repr.row_read_ts[i] + G::Scalar::ONE),
        )
      })
      .collect::<Vec<G::Scalar>>();
    let audit_row = (0..mem_row.len())
      .map(|i| {
        hash_func(
          &G::Scalar::from(i as u64),
          &mem_row[i],
          &pk.S_repr.row_audit_ts[i],
        )
      })
      .collect::<Vec<G::Scalar>>();

    let init_col = (0..mem_col.len())
      .map(|i| hash_func(&G::Scalar::from(i as u64), &mem_col[i], &G::Scalar::ZERO))
      .collect::<Vec<G::Scalar>>();
    let read_col = (0..E_col.len())
      .map(|i| hash_func(&pk.S_repr.col[i], &E_col[i], &pk.S_repr.col_read_ts[i]))
      .collect::<Vec<G::Scalar>>();
    let write_col = (0..E_col.len())
      .map(|i| {
        hash_func(
          &pk.S_repr.col[i],
          &E_col[i],
          &(pk.S_repr.col_read_ts[i] + G::Scalar::ONE),
        )
      })
      .collect::<Vec<G::Scalar>>();
    let audit_col = (0..mem_col.len())
      .map(|i| {
        hash_func(
          &G::Scalar::from(i as u64),
          &mem_col[i],
          &pk.S_repr.col_audit_ts[i],
        )
      })
      .collect::<Vec<G::Scalar>>();

    let mut mem_sc_inst = ProductSumcheckInstance::new(
      ck,
      vec![
        init_row, read_row, write_row, audit_row, init_col, read_col, write_col, audit_col,
      ],
      &mut transcript,
    )?;

    let (sc_sat, r_sat, claims_mem, claims_outer, claims_inner) = Self::prove_inner(
      &mut mem_sc_inst,
      &mut outer_sc_inst,
      &mut inner_sc_inst,
      &mut transcript,
    )?;

    // claims[0] is about the Eq polynomial, which the verifier computes directly
    // claims[1] =? weighed sum of left(rand)
    // claims[2] =? weighted sum of right(rand)
    // claims[3] =? weighted sum of output(rand), which is easy to verify by querying output
    // we also need to prove that output(output.len()-2) = claimed_product
    let eval_left_vec = claims_mem[1].clone();
    let eval_right_vec = claims_mem[2].clone();
    let eval_output_vec = claims_mem[3].clone();

    // claims from the end of sum-check
    let (eval_Az, eval_Bz): (G::Scalar, G::Scalar) = (claims_outer[0][1], claims_outer[0][2]);
    let eval_Cz = MultilinearPolynomial::evaluate_with(&Cz, &r_sat);
    let eval_E = MultilinearPolynomial::evaluate_with(&E, &r_sat);
    let eval_E_row = claims_inner[0][0];
    let eval_E_col = claims_inner[0][1];
    let eval_val_A = MultilinearPolynomial::evaluate_with(&pk.S_repr.val_A, &r_sat);
    let eval_val_B = MultilinearPolynomial::evaluate_with(&pk.S_repr.val_B, &r_sat);
    let eval_val_C = MultilinearPolynomial::evaluate_with(&pk.S_repr.val_C, &r_sat);
    let eval_vec = vec![
      eval_Az, eval_Bz, eval_Cz, eval_E, eval_E_row, eval_E_col, eval_val_A, eval_val_B, eval_val_C,
    ]
    .into_iter()
    .chain(eval_left_vec.clone())
    .chain(eval_right_vec.clone())
    .chain(eval_output_vec.clone())
    .collect::<Vec<G::Scalar>>();

    // absorb all the claimed evaluations
    transcript.absorb(b"e", &eval_vec.as_slice());

    // we now combine eval_left = left(rand) and eval_right = right(rand)
    // into claims about input and output
    let c = transcript.squeeze(b"c")?;

    // eval = (G::Scalar::ONE - c) * eval_left + c * eval_right
    // eval is claimed evaluation of input||output(r, c), which can be proven by proving input(r[1..], c) and output(r[1..], c)
    let rand_ext = {
      let mut r = r_sat.clone();
      r.extend(&[c]);
      r
    };
    let eval_input_vec = mem_sc_inst
      .input_vec
      .iter()
      .map(|i| MultilinearPolynomial::evaluate_with(i, &rand_ext[1..]))
      .collect::<Vec<G::Scalar>>();

    let eval_output2_vec = mem_sc_inst
      .output_vec
      .iter()
      .map(|o| MultilinearPolynomial::evaluate_with(o, &rand_ext[1..]))
      .collect::<Vec<G::Scalar>>();

    // add claimed evaluations to the transcript
    let evals = eval_input_vec
      .clone()
      .into_iter()
      .chain(eval_output2_vec.clone())
      .collect::<Vec<G::Scalar>>();
    transcript.absorb(b"e", &evals.as_slice());

    // squeeze a challenge to combine multiple claims into one
    let powers_of_rho = {
      let s = transcript.squeeze(b"r")?;
      let mut s_vec = vec![s];
      for i in 1..mem_sc_inst.initial_claims().len() {
        s_vec.push(s_vec[i - 1] * s);
      }
      s_vec
    };

    // take weighted sum of input, output, and their commitments
    let product = mem_sc_inst
      .claims
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .sum();

    let eval_output = eval_output_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .sum();

    let comm_output = mem_sc_inst
      .comm_output_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(c, r_i)| *c * *r_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);

    let weighted_sum = |W: &[Vec<G::Scalar>], s: &[G::Scalar]| -> Vec<G::Scalar> {
      assert_eq!(W.len(), s.len());
      let mut p = vec![G::Scalar::ZERO; W[0].len()];
      for i in 0..W.len() {
        for (j, item) in W[i].iter().enumerate().take(W[i].len()) {
          p[j] += *item * s[i]
        }
      }
      p
    };

    let poly_output = weighted_sum(&mem_sc_inst.output_vec, &powers_of_rho);

    let eval_output2 = eval_output2_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .sum();

    // eval_output = output(r_sat)
    w_u_vec.push((
      PolyEvalWitness {
        p: poly_output.clone(),
      },
      PolyEvalInstance {
        c: comm_output,
        x: r_sat.clone(),
        e: eval_output,
      },
    ));

    // claimed_product = output(1, ..., 1, 0)
    let x = {
      let mut x = vec![G::Scalar::ONE; r_sat.len()];
      x[r_sat.len() - 1] = G::Scalar::ZERO;
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

    let r_prod = rand_ext[1..].to_vec();
    // row-related and col-related claims of polynomial evaluations to aid the final check of the sum-check
    let evals = [
      &pk.S_repr.row,
      &pk.S_repr.row_read_ts,
      &E_row,
      &pk.S_repr.row_audit_ts,
      &pk.S_repr.col,
      &pk.S_repr.col_read_ts,
      &E_col,
      &pk.S_repr.col_audit_ts,
    ]
    .into_par_iter()
    .map(|p| MultilinearPolynomial::evaluate_with(p, &r_prod))
    .collect::<Vec<G::Scalar>>();

    let eval_row = evals[0];
    let eval_row_read_ts = evals[1];
    let eval_E_row_at_r_prod = evals[2];
    let eval_row_audit_ts = evals[3];
    let eval_col = evals[4];
    let eval_col_read_ts = evals[5];
    let eval_E_col_at_r_prod = evals[6];
    let eval_col_audit_ts = evals[7];

    // we need to prove that eval_z = z(r_prod) = (1-r_prod[0]) * W.w(r_prod[1..]) + r_prod[0] * U.x(r_prod[1..]).
    // r_prod was padded, so we now remove the padding
    let r_prod_unpad = {
      let l = pk.S_repr.N.log_2() - (2 * pk.S.num_vars).log_2();
      r_prod[l..].to_vec()
    };

    let eval_W = MultilinearPolynomial::evaluate_with(&W.W, &r_prod_unpad[1..]);

    // we can batch all the claims
    transcript.absorb(
      b"e",
      &[
        eval_row,
        eval_row_read_ts,
        eval_E_row_at_r_prod,
        eval_row_audit_ts,
        eval_col,
        eval_col_read_ts,
        eval_E_col_at_r_prod,
        eval_col_audit_ts,
        eval_W, // this will not be batched below as it is evaluated at r_prod[1..]
      ]
      .as_slice(),
    );

    let c = transcript.squeeze(b"c")?;
    let eval_vec = [
      eval_row,
      eval_row_read_ts,
      eval_E_row_at_r_prod,
      eval_row_audit_ts,
      eval_col,
      eval_col_read_ts,
      eval_E_col_at_r_prod,
      eval_col_audit_ts,
    ];
    let comm_vec = [
      pk.S_comm.comm_row,
      pk.S_comm.comm_row_read_ts,
      comm_E_row,
      pk.S_comm.comm_row_audit_ts,
      pk.S_comm.comm_col,
      pk.S_comm.comm_col_read_ts,
      comm_E_col,
      pk.S_comm.comm_col_audit_ts,
    ];
    let poly_vec = [
      &pk.S_repr.row,
      &pk.S_repr.row_read_ts,
      &E_row,
      &pk.S_repr.row_audit_ts,
      &pk.S_repr.col,
      &pk.S_repr.col_read_ts,
      &E_col,
      &pk.S_repr.col_audit_ts,
    ];
    let w = PolyEvalWitness::batch(&poly_vec, &c);
    let u = PolyEvalInstance::batch(&comm_vec, &r_prod, &eval_vec, &c);

    // add the claim to prove for later
    w_u_vec.push((w, u));

    w_u_vec.push((
      PolyEvalWitness { p: W.W },
      PolyEvalInstance {
        c: U.comm_W,
        x: r_prod_unpad[1..].to_vec(),
        e: eval_W,
      },
    ));

    // all nine evaluations are at r_sat, we can fold them into one; they were added to the transcript earlier
    let eval_vec = [
      eval_Az, eval_Bz, eval_Cz, eval_E, eval_E_row, eval_E_col, eval_val_A, eval_val_B, eval_val_C,
    ];
    let comm_vec = [
      comm_Az,
      comm_Bz,
      comm_Cz,
      U.comm_E,
      comm_E_row,
      comm_E_col,
      pk.S_comm.comm_val_A,
      pk.S_comm.comm_val_B,
      pk.S_comm.comm_val_C,
    ];
    let poly_vec = [
      &Az,
      &Bz,
      &Cz,
      &E,
      &E_row,
      &E_col,
      &pk.S_repr.val_A,
      &pk.S_repr.val_B,
      &pk.S_repr.val_C,
    ];
    transcript.absorb(b"e", &eval_vec.as_slice()); // c_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let w = PolyEvalWitness::batch(&poly_vec, &c);
    let u = PolyEvalInstance::batch(&comm_vec, &r_sat, &eval_vec, &c);
    w_u_vec.push((w, u));

    // We will now reduce a vector of claims of evaluations at different points into claims about them at the same point.
    // For example, eval_W =? W(r_y[1..]) and eval_W =? E(r_x) into
    // two claims: eval_W_prime =? W(rz) and eval_E_prime =? E(rz)
    // We can them combine the two into one: eval_W_prime + gamma * eval_E_prime =? (W + gamma*E)(rz),
    // where gamma is a public challenge
    // Since commitments to W and E are homomorphic, the verifier can compute a commitment
    // to the batched polynomial.
    assert!(w_u_vec.len() >= 2);

    let (w_vec, u_vec): (Vec<PolyEvalWitness<G>>, Vec<PolyEvalInstance<G>>) =
      w_u_vec.into_iter().unzip();
    let w_vec_padded = PolyEvalWitness::pad(&w_vec); // pad the polynomials to be of the same size
    let u_vec_padded = PolyEvalInstance::pad(&u_vec); // pad the evaluation points

    // generate a challenge
    let rho = transcript.squeeze(b"r")?;
    let num_claims = w_vec_padded.len();
    let powers_of_rho = powers::<G>(&rho, num_claims);
    let claim_batch_joint = u_vec_padded
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(u, p)| u.e * p)
      .sum();

    let mut polys_left: Vec<MultilinearPolynomial<G::Scalar>> = w_vec_padded
      .iter()
      .map(|w| MultilinearPolynomial::new(w.p.clone()))
      .collect();
    let mut polys_right: Vec<MultilinearPolynomial<G::Scalar>> = u_vec_padded
      .iter()
      .map(|u| MultilinearPolynomial::new(EqPolynomial::new(u.x.clone()).evals()))
      .collect();

    let num_rounds_z = u_vec_padded[0].x.len();
    let comb_func = |poly_A_comp: &G::Scalar, poly_B_comp: &G::Scalar| -> G::Scalar {
      *poly_A_comp * *poly_B_comp
    };
    let (sc_proof_batch, r_z, claims_batch) = SumcheckProof::prove_quad_batch(
      &claim_batch_joint,
      num_rounds_z,
      &mut polys_left,
      &mut polys_right,
      &powers_of_rho,
      comb_func,
      &mut transcript,
    )?;

    let (claims_batch_left, _): (Vec<G::Scalar>, Vec<G::Scalar>) = claims_batch;

    transcript.absorb(b"l", &claims_batch_left.as_slice());

    // we now combine evaluation claims at the same point rz into one
    let gamma = transcript.squeeze(b"g")?;
    let powers_of_gamma: Vec<G::Scalar> = powers::<G>(&gamma, num_claims);
    let comm_joint = u_vec_padded
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(u, g_i)| u.c * *g_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);
    let poly_joint = PolyEvalWitness::weighted_sum(&w_vec_padded, &powers_of_gamma);
    let eval_joint = claims_batch_left
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(e, g_i)| *e * *g_i)
      .sum();

    let eval_arg = EE::prove(
      ck,
      &pk.pk_ee,
      &mut transcript,
      &comm_joint,
      &poly_joint.p,
      &r_z,
      &eval_joint,
    )?;

    Ok(RelaxedR1CSSNARK {
      comm_Az: comm_Az.compress(),
      comm_Bz: comm_Bz.compress(),
      comm_Cz: comm_Cz.compress(),
      comm_E_row: comm_E_row.compress(),
      comm_E_col: comm_E_col.compress(),
      eval_Az_at_tau,
      eval_Bz_at_tau,
      eval_Cz_at_tau,
      comm_output_arr: vec_to_arr(
        mem_sc_inst
          .comm_output_vec
          .iter()
          .map(|c| c.compress())
          .collect::<Vec<CompressedCommitment<G>>>(),
      ),
      claims_product_arr: vec_to_arr(mem_sc_inst.claims.clone()),

      sc_sat,

      eval_Az,
      eval_Bz,
      eval_Cz,
      eval_E,
      eval_E_row,
      eval_E_col,
      eval_val_A,
      eval_val_B,
      eval_val_C,

      eval_left_arr: vec_to_arr(eval_left_vec),
      eval_right_arr: vec_to_arr(eval_right_vec),
      eval_output_arr: vec_to_arr(eval_output_vec),
      eval_input_arr: vec_to_arr(eval_input_vec),
      eval_output2_arr: vec_to_arr(eval_output2_vec),

      eval_row,
      eval_row_read_ts,
      eval_E_row_at_r_prod,
      eval_row_audit_ts,
      eval_col,
      eval_col_read_ts,
      eval_E_col_at_r_prod,
      eval_col_audit_ts,
      eval_W,

      sc_proof_batch,
      evals_batch_arr: vec_to_arr(claims_batch_left),
      eval_arg,
    })
  }

  /// verifies a proof of satisfiability of a RelaxedR1CS instance
  fn verify(&self, vk: &Self::VerifierKey, U: &RelaxedR1CSInstance<G>) -> Result<(), NovaError> {
    let mut transcript = G::TE::new(b"RelaxedR1CSSNARK");
    let mut u_vec: Vec<PolyEvalInstance<G>> = Vec::new();

    // append the verifier key (including commitment to R1CS matrices) and the RelaxedR1CSInstance to the transcript
    transcript.absorb(b"vk", &vk.digest);
    transcript.absorb(b"U", U);

    let comm_Az = Commitment::<G>::decompress(&self.comm_Az)?;
    let comm_Bz = Commitment::<G>::decompress(&self.comm_Bz)?;
    let comm_Cz = Commitment::<G>::decompress(&self.comm_Cz)?;
    let comm_E_row = Commitment::<G>::decompress(&self.comm_E_row)?;
    let comm_E_col = Commitment::<G>::decompress(&self.comm_E_col)?;

    transcript.absorb(b"c", &[comm_Az, comm_Bz, comm_Cz].as_slice());

    let num_rounds_sat = vk.S_comm.N.log_2();
    let tau = (0..num_rounds_sat)
      .map(|_i| transcript.squeeze(b"t"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    transcript.absorb(
      b"e",
      &[
        self.eval_Az_at_tau,
        self.eval_Bz_at_tau,
        self.eval_Cz_at_tau,
      ]
      .as_slice(),
    );

    transcript.absorb(b"e", &vec![comm_E_row, comm_E_col].as_slice());

    // add claims about Az, Bz, and Cz to be checked later
    // since all the three polynomials are opened at tau,
    // we can combine them into a single polynomial opened at tau
    let eval_vec = vec![
      self.eval_Az_at_tau,
      self.eval_Bz_at_tau,
      self.eval_Cz_at_tau,
    ];
    let comm_vec = vec![comm_Az, comm_Bz, comm_Cz];
    transcript.absorb(b"e", &eval_vec.as_slice()); // c_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let u = PolyEvalInstance::batch(&comm_vec, &tau, &eval_vec, &c);
    let claim_inner = u.e;
    let c_inner = c;
    u_vec.push(u);

    let gamma_1 = transcript.squeeze(b"g1")?;
    let gamma_2 = transcript.squeeze(b"g2")?;

    // hash function
    let gamma_1_sqr = gamma_1 * gamma_1;
    let hash_func = |addr: &G::Scalar, val: &G::Scalar, ts: &G::Scalar| -> G::Scalar {
      (*ts * gamma_1_sqr + *val * gamma_1 + *addr) - gamma_2
    };

    // check the required multiset relationship
    // row
    if self.claims_product_arr[0] * self.claims_product_arr[2]
      != self.claims_product_arr[1] * self.claims_product_arr[3]
    {
      return Err(NovaError::InvalidMultisetProof);
    }
    // col
    if self.claims_product_arr[4] * self.claims_product_arr[6]
      != self.claims_product_arr[5] * self.claims_product_arr[7]
    {
      return Err(NovaError::InvalidMultisetProof);
    }

    let comm_output_vec = self
      .comm_output_arr
      .iter()
      .map(|c| Commitment::<G>::decompress(c))
      .collect::<Result<Vec<Commitment<G>>, NovaError>>()?;

    transcript.absorb(b"o", &comm_output_vec.as_slice());
    transcript.absorb(b"c", &self.claims_product_arr.as_slice());

    let num_rounds = vk.S_comm.N.log_2();
    let rand_eq = (0..num_rounds)
      .map(|_i| transcript.squeeze(b"e"))
      .collect::<Result<Vec<G::Scalar>, NovaError>>()?;

    let num_claims = 10;
    let coeffs = {
      let s = transcript.squeeze(b"r")?;
      let mut s_vec = vec![s];
      for i in 1..num_claims {
        s_vec.push(s_vec[i - 1] * s);
      }
      s_vec
    };

    let claim = coeffs[9] * claim_inner; // rest are zeros
    let (claim_sat_final, r_sat) = self
      .sc_sat
      .verify(claim, num_rounds_sat, 3, &mut transcript)?;

    // verify claim_sat_final
    let taus_bound_r_sat = EqPolynomial::new(tau.clone()).evaluate(&r_sat);
    let rand_eq_bound_r_sat = EqPolynomial::new(rand_eq).evaluate(&r_sat);
    let claim_mem_final_expected: G::Scalar = (0..8)
      .map(|i| {
        coeffs[i]
          * rand_eq_bound_r_sat
          * (self.eval_left_arr[i] * self.eval_right_arr[i] - self.eval_output_arr[i])
      })
      .sum();
    let claim_outer_final_expected = coeffs[8]
      * taus_bound_r_sat
      * (self.eval_Az * self.eval_Bz - U.u * self.eval_Cz - self.eval_E);
    let claim_inner_final_expected = coeffs[9]
      * self.eval_E_row
      * self.eval_E_col
      * (self.eval_val_A + c_inner * self.eval_val_B + c_inner * c_inner * self.eval_val_C);

    if claim_mem_final_expected + claim_outer_final_expected + claim_inner_final_expected
      != claim_sat_final
    {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // claims from the end of the sum-check
    let eval_vec = [
      self.eval_Az,
      self.eval_Bz,
      self.eval_Cz,
      self.eval_E,
      self.eval_E_row,
      self.eval_E_col,
      self.eval_val_A,
      self.eval_val_B,
      self.eval_val_C,
    ]
    .into_iter()
    .chain(self.eval_left_arr)
    .chain(self.eval_right_arr)
    .chain(self.eval_output_arr)
    .collect::<Vec<G::Scalar>>();

    transcript.absorb(b"e", &eval_vec.as_slice());
    // we now combine eval_left = left(rand) and eval_right = right(rand)
    // into claims about input and output
    let c = transcript.squeeze(b"c")?;

    // eval = (G::Scalar::ONE - c) * eval_left + c * eval_right
    // eval is claimed evaluation of input||output(r, c), which can be proven by proving input(r[1..], c) and output(r[1..], c)
    let rand_ext = {
      let mut r = r_sat.clone();
      r.extend(&[c]);
      r
    };

    // add claimed evaluations to the transcript
    let evals = self
      .eval_input_arr
      .into_iter()
      .chain(self.eval_output2_arr)
      .collect::<Vec<G::Scalar>>();
    transcript.absorb(b"e", &evals.as_slice());

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
    let product = self
      .claims_product_arr
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .sum();

    let eval_output = self
      .eval_output_arr
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .sum();

    let comm_output = comm_output_vec
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(c, r_i)| *c * *r_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);

    let eval_output2 = self
      .eval_output2_arr
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(e, p)| *e * p)
      .sum();

    // eval_output = output(r_sat)
    u_vec.push(PolyEvalInstance {
      c: comm_output,
      x: r_sat.clone(),
      e: eval_output,
    });

    // claimed_product = output(1, ..., 1, 0)
    let x = {
      let mut x = vec![G::Scalar::ONE; r_sat.len()];
      x[r_sat.len() - 1] = G::Scalar::ZERO;
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

    let r_prod = rand_ext[1..].to_vec();
    // row-related and col-related claims of polynomial evaluations to aid the final check of the sum-check
    // we can batch all the claims
    transcript.absorb(
      b"e",
      &[
        self.eval_row,
        self.eval_row_read_ts,
        self.eval_E_row_at_r_prod,
        self.eval_row_audit_ts,
        self.eval_col,
        self.eval_col_read_ts,
        self.eval_E_col_at_r_prod,
        self.eval_col_audit_ts,
        self.eval_W,
      ]
      .as_slice(),
    );
    let c = transcript.squeeze(b"c")?;
    let eval_vec = [
      self.eval_row,
      self.eval_row_read_ts,
      self.eval_E_row_at_r_prod,
      self.eval_row_audit_ts,
      self.eval_col,
      self.eval_col_read_ts,
      self.eval_E_col_at_r_prod,
      self.eval_col_audit_ts,
    ];
    let comm_vec = [
      vk.S_comm.comm_row,
      vk.S_comm.comm_row_read_ts,
      comm_E_row,
      vk.S_comm.comm_row_audit_ts,
      vk.S_comm.comm_col,
      vk.S_comm.comm_col_read_ts,
      comm_E_col,
      vk.S_comm.comm_col_audit_ts,
    ];
    let u = PolyEvalInstance::batch(&comm_vec, &r_prod, &eval_vec, &c);

    // add the claim to prove for later
    u_vec.push(u);

    // compute eval_Z
    let (eval_Z, r_prod_unpad) = {
      // r_prod was padded, so we now remove the padding
      let (factor, r_prod_unpad) = {
        let l = vk.S_comm.N.log_2() - (2 * vk.num_vars).log_2();

        let mut factor = G::Scalar::ONE;
        for r_p in r_prod.iter().take(l) {
          factor *= G::Scalar::ONE - r_p
        }

        let r_prod_unpad = {
          let l = vk.S_comm.N.log_2() - (2 * vk.num_vars).log_2();
          r_prod[l..].to_vec()
        };

        (factor, r_prod_unpad)
      };

      let eval_X = {
        // constant term
        let mut poly_X = vec![(0, U.u)];
        //remaining inputs
        poly_X.extend(
          (0..U.X.len())
            .map(|i| (i + 1, U.X[i]))
            .collect::<Vec<(usize, G::Scalar)>>(),
        );
        SparsePolynomial::new((vk.num_vars as f64).log2() as usize, poly_X)
          .evaluate(&r_prod_unpad[1..])
      };
      let eval_Z =
        factor * ((G::Scalar::ONE - r_prod_unpad[0]) * self.eval_W + r_prod_unpad[0] * eval_X);

      (eval_Z, r_prod_unpad)
    };

    u_vec.push(PolyEvalInstance {
      c: U.comm_W,
      x: r_prod_unpad[1..].to_vec(),
      e: self.eval_W,
    });

    // finish the final step of the sum-check
    let (claim_init_expected_row, claim_audit_expected_row) = {
      let addr = IdentityPolynomial::new(r_prod.len()).evaluate(&r_prod);
      let val = EqPolynomial::new(tau.to_vec()).evaluate(&r_prod);
      (
        hash_func(&addr, &val, &G::Scalar::ZERO),
        hash_func(&addr, &val, &self.eval_row_audit_ts),
      )
    };

    let (claim_read_expected_row, claim_write_expected_row) = {
      (
        hash_func(
          &self.eval_row,
          &self.eval_E_row_at_r_prod,
          &self.eval_row_read_ts,
        ),
        hash_func(
          &self.eval_row,
          &self.eval_E_row_at_r_prod,
          &(self.eval_row_read_ts + G::Scalar::ONE),
        ),
      )
    };

    // multiset check for the row
    if claim_init_expected_row != self.eval_input_arr[0]
      || claim_read_expected_row != self.eval_input_arr[1]
      || claim_write_expected_row != self.eval_input_arr[2]
      || claim_audit_expected_row != self.eval_input_arr[3]
    {
      return Err(NovaError::InvalidSumcheckProof);
    }

    let (claim_init_expected_col, claim_audit_expected_col) = {
      let addr = IdentityPolynomial::new(r_prod.len()).evaluate(&r_prod);
      let val = eval_Z;
      (
        hash_func(&addr, &val, &G::Scalar::ZERO),
        hash_func(&addr, &val, &self.eval_col_audit_ts),
      )
    };

    let (claim_read_expected_col, claim_write_expected_col) = {
      (
        hash_func(
          &self.eval_col,
          &self.eval_E_col_at_r_prod,
          &self.eval_col_read_ts,
        ),
        hash_func(
          &self.eval_col,
          &self.eval_E_col_at_r_prod,
          &(self.eval_col_read_ts + G::Scalar::ONE),
        ),
      )
    };

    // multiset check for the col
    if claim_init_expected_col != self.eval_input_arr[4]
      || claim_read_expected_col != self.eval_input_arr[5]
      || claim_write_expected_col != self.eval_input_arr[6]
      || claim_audit_expected_col != self.eval_input_arr[7]
    {
      return Err(NovaError::InvalidSumcheckProof);
    }

    // since all the nine polynomials are opened at r_sat,
    // we can combine them into a single polynomial opened at r_sat
    let eval_vec = [
      self.eval_Az,
      self.eval_Bz,
      self.eval_Cz,
      self.eval_E,
      self.eval_E_row,
      self.eval_E_col,
      self.eval_val_A,
      self.eval_val_B,
      self.eval_val_C,
    ];
    let comm_vec = [
      comm_Az,
      comm_Bz,
      comm_Cz,
      U.comm_E,
      comm_E_row,
      comm_E_col,
      vk.S_comm.comm_val_A,
      vk.S_comm.comm_val_B,
      vk.S_comm.comm_val_C,
    ];
    transcript.absorb(b"e", &eval_vec.as_slice()); // c_vec is already in the transcript
    let c = transcript.squeeze(b"c")?;
    let u = PolyEvalInstance::batch(&comm_vec, &r_sat, &eval_vec, &c);
    u_vec.push(u);

    let u_vec_padded = PolyEvalInstance::pad(&u_vec); // pad the evaluation points

    // generate a challenge
    let rho = transcript.squeeze(b"r")?;
    let num_claims = u_vec.len();
    let powers_of_rho = powers::<G>(&rho, num_claims);
    let claim_batch_joint = u_vec_padded
      .iter()
      .zip(powers_of_rho.iter())
      .map(|(u, p)| u.e * p)
      .sum();

    let num_rounds_z = u_vec_padded[0].x.len();
    let (claim_batch_final, r_z) =
      self
        .sc_proof_batch
        .verify(claim_batch_joint, num_rounds_z, 2, &mut transcript)?;

    let claim_batch_final_expected = {
      let poly_rz = EqPolynomial::new(r_z.clone());
      let evals = u_vec_padded
        .iter()
        .map(|u| poly_rz.evaluate(&u.x))
        .collect::<Vec<G::Scalar>>();

      evals
        .iter()
        .zip(self.evals_batch_arr.iter())
        .zip(powers_of_rho.iter())
        .map(|((e_i, p_i), rho_i)| *e_i * *p_i * rho_i)
        .sum()
    };

    if claim_batch_final != claim_batch_final_expected {
      return Err(NovaError::InvalidSumcheckProof);
    }

    transcript.absorb(b"l", &self.evals_batch_arr.as_slice());

    // we now combine evaluation claims at the same point rz into one
    let gamma = transcript.squeeze(b"g")?;
    let powers_of_gamma: Vec<G::Scalar> = powers::<G>(&gamma, num_claims);
    let comm_joint = u_vec_padded
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(u, g_i)| u.c * *g_i)
      .fold(Commitment::<G>::default(), |acc, item| acc + item);
    let eval_joint = self
      .evals_batch_arr
      .iter()
      .zip(powers_of_gamma.iter())
      .map(|(e, g_i)| *e * *g_i)
      .sum();

    // verify
    EE::verify(
      &vk.vk_ee,
      &mut transcript,
      &comm_joint,
      &r_z,
      &eval_joint,
      &self.eval_arg,
    )?;

    Ok(())
  }
}
