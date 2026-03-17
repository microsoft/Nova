//! Compiled witness computation program extracted from R1CS constraints.
//!
//! After circuit synthesis determines the R1CS shape (matrices A, B, C), this
//! module can compile a "witness program" — a flat sequence of field operations
//! that computes the auxiliary witness given the public inputs.
//!
//! This eliminates all framework overhead from synthesis (closures, trait
//! dispatch, Option unwrapping, namespace strings) and replaces it with a
//! tight inner loop of field operations on a flat buffer.
//!
//! # Approach
//!
//! Each R1CS constraint has the form: `(A·z) * (B·z) = (C·z)`.
//! For witness generation, we process constraints in order and solve for the
//! single unknown auxiliary variable in each constraint.
//!
//! For any constraint where exactly one variable `x` is unknown, the constraint
//! is linear in `x` (since `x` cannot appear in both A and B simultaneously
//! in well-formed circuits). The general solution is:
//!
//! ```text
//! x = (C_known - A_known * B_known) / (a_x * B_known + b_x * A_known - c_x)
//! ```
//!
//! where `A_known`, `B_known`, `C_known` are the evaluations of each matrix row
//! excluding the unknown variable, and `a_x`, `b_x`, `c_x` are the coefficients
//! of `x` in each matrix.

use ff::PrimeField;

/// A single instruction in the compiled witness program.
///
/// The z-vector layout is: `[W | u | X]` where:
/// - `W[0..num_vars]` are auxiliary variables (to be computed)
/// - `u` is at index `num_vars` (always 1 for fresh instances)
/// - `X[0..num_io]` are public inputs (given)
///
/// Instructions refer to z-vector indices directly.
#[derive(Clone, Debug)]
pub enum WitnessOp<F: PrimeField> {
  /// z[dst] = z[a] * z[b]
  /// Common case: Boolean AND, field multiplication where c_coeff = 1
  Mul { dst: u32, a: u32, b: u32 },

  /// z[dst] = coeff * z[a] * z[b]
  /// Scaled product (e.g., constraint: coeff * a * b = c  →  c = coeff * a * b)
  MulScaled { dst: u32, a: u32, b: u32, coeff: F },

  /// z[dst] = (A_known * B_known - C_known) / divisor
  /// where divisor = -(a_x * B_known + b_x * A_known - c_x)
  ///
  /// Specialization: A and B have exactly 1 term each, dst is in C only.
  /// At compile time, if c_x can be inverted, inv_c stores 1/c_x.
  /// z[dst] = (eval_A * eval_B - C_known_sum) * inv_c
  GeneralC {
    dst: u32,
    a_terms: Vec<(u32, F)>,
    b_terms: Vec<(u32, F)>,
    c_known: Vec<(u32, F)>,
    inv_c_coeff: F,
  },

  /// General linear solver for when dst appears in A or B (and possibly C).
  ///
  /// x = (C_known - A_known * B_known) / (a_x * B_known + b_x * A_known - c_x)
  ///
  /// where A_known/B_known/C_known exclude the dst variable.
  GeneralLinear {
    dst: u32,
    a_known: Vec<(u32, F)>,
    b_known: Vec<(u32, F)>,
    c_known: Vec<(u32, F)>,
    /// Coefficient of dst in A (0 if dst not in A)
    a_dst: F,
    /// Coefficient of dst in B (0 if dst not in B)
    b_dst: F,
    /// Coefficient of dst in C (0 if dst not in C)
    c_dst: F,
  },
}

/// A compiled witness computation program.
///
/// After compilation, calling `execute()` with the public inputs produces
/// the full auxiliary witness — equivalent to running `synthesize()` but
/// 10-50× faster because all framework overhead is eliminated.
#[derive(Clone, Debug)]
pub struct WitnessProgram<F: PrimeField> {
  /// Instructions in dependency order
  pub(crate) ops: Vec<WitnessOp<F>>,
  /// Number of auxiliary variables (W)
  pub(crate) num_vars: usize,
  /// Number of public IO variables (X)
  pub(crate) num_io: usize,
  /// Indices into the z-vector for "free" auxiliary variables
  /// (those not determined by any constraint — must be provided externally)
  pub(crate) free_vars: Vec<u32>,
}

impl<F: PrimeField> WitnessProgram<F> {
  /// Execute the compiled program to compute the auxiliary witness.
  ///
  /// # Arguments
  /// * `inputs` - Public IO values (X), length = num_io
  /// * `free_witness` - Values for free variables (not determined by constraints),
  ///                    in the order given by `self.free_vars`
  ///
  /// # Returns
  /// The auxiliary witness W as a Vec of field elements.
  pub fn execute(&self, inputs: &[F], free_witness: &[F]) -> Vec<F> {
    assert_eq!(inputs.len(), self.num_io);
    assert_eq!(free_witness.len(), self.free_vars.len());

    // Build z-vector: [W | u=1 | X]
    let z_len = self.num_vars + 1 + self.num_io;
    let mut z = vec![F::ZERO; z_len];

    // Set u = 1
    z[self.num_vars] = F::ONE;

    // Set public inputs
    for (i, &x) in inputs.iter().enumerate() {
      z[self.num_vars + 1 + i] = x;
    }

    // Set free variables
    for (&idx, &val) in self.free_vars.iter().zip(free_witness.iter()) {
      z[idx as usize] = val;
    }

    // Execute instruction tape
    for op in &self.ops {
      match op {
        WitnessOp::Mul { dst, a, b } => {
          z[*dst as usize] = z[*a as usize] * z[*b as usize];
        }
        WitnessOp::MulScaled { dst, a, b, coeff } => {
          z[*dst as usize] = *coeff * z[*a as usize] * z[*b as usize];
        }
        WitnessOp::GeneralC {
          dst,
          a_terms,
          b_terms,
          c_known,
          inv_c_coeff,
        } => {
          let eval_a = Self::eval_terms(a_terms, &z);
          let eval_b = Self::eval_terms(b_terms, &z);
          let c_sum = Self::eval_terms(c_known, &z);
          z[*dst as usize] = (eval_a * eval_b - c_sum) * *inv_c_coeff;
        }
        WitnessOp::GeneralLinear {
          dst,
          a_known,
          b_known,
          c_known,
          a_dst,
          b_dst,
          c_dst,
        } => {
          let eval_a_known = Self::eval_terms(a_known, &z);
          let eval_b_known = Self::eval_terms(b_known, &z);
          let eval_c_known = Self::eval_terms(c_known, &z);
          let numerator = eval_c_known - eval_a_known * eval_b_known;
          let denominator = *a_dst * eval_b_known + *b_dst * eval_a_known - *c_dst;
          z[*dst as usize] = numerator * denominator.invert().unwrap();
        }
      }
    }

    // Extract W from z
    z.truncate(self.num_vars);
    z
  }

  #[inline(always)]
  fn eval_terms(terms: &[(u32, F)], z: &[F]) -> F {
    terms.iter().map(|(idx, coeff)| *coeff * z[*idx as usize]).sum()
  }

  /// Number of compiled instructions
  pub fn num_instructions(&self) -> usize {
    self.ops.len()
  }

  /// Number of free (externally-provided) variables
  pub fn num_free_vars(&self) -> usize {
    self.free_vars.len()
  }

  /// Returns the indices of free variables in the auxiliary witness.
  pub fn free_var_indices(&self) -> &[u32] {
    &self.free_vars
  }

  /// Extract free variable values from a full auxiliary witness.
  ///
  /// Given the output of `synthesize()` (the aux_assignment), extracts
  /// the values at free variable positions.
  pub fn extract_free_witness(&self, aux_assignment: &[F]) -> Vec<F> {
    self.free_vars.iter().map(|&i| aux_assignment[i as usize]).collect()
  }

  /// Compile a witness program from R1CS matrices.
  ///
  /// Processes each constraint in order, attempting to solve for a single
  /// unknown auxiliary variable. Variables are considered "known" if they
  /// are inputs (io + u) or have been computed by a previous constraint.
  ///
  /// In bellman-style circuits, many variables are "free" — their values come
  /// from `alloc` closures, not from constraints. The compiler handles this by:
  /// 1. First pass: solve what we can from constraints alone
  /// 2. Mark all remaining unknowns as "free" (externally provided)
  /// 3. Re-process deferred constraints with free variables now known
  ///
  /// Returns the compiled program, or an error message if compilation fails.
  pub fn compile(
    num_cons: usize,
    num_vars: usize,
    num_io: usize,
    a_matrix: &super::SparseMatrix<F>,
    b_matrix: &super::SparseMatrix<F>,
    c_matrix: &super::SparseMatrix<F>,
  ) -> Result<Self, String> {
    let z_len = num_vars + 1 + num_io;

    // Track which variables have been computed
    let mut known = vec![false; z_len];
    for i in num_vars..z_len {
      known[i] = true; // u and X are known
    }

    let mut ops = Vec::with_capacity(num_cons);
    let mut deferred: Vec<usize> = Vec::new();

    // Phase 1: Process constraints in order, solve what we can
    for row in 0..num_cons {
      Self::process_row(row, num_vars, a_matrix, b_matrix, c_matrix, &mut known, &mut ops, &mut deferred);
    }

    // Phase 1b: Iterate on deferred until no progress
    let mut progress = true;
    while progress && !deferred.is_empty() {
      progress = false;
      deferred.retain(|&row| {
        let prev_ops = ops.len();
        Self::process_row(row, num_vars, a_matrix, b_matrix, c_matrix, &mut known, &mut ops, &mut Vec::new());
        if ops.len() > prev_ops {
          progress = true;
          false
        } else if Self::all_vars_known(row, a_matrix, b_matrix, c_matrix, &known) {
          progress = true;
          false
        } else {
          true
        }
      });
    }

    // Phase 2: Mark all remaining unknown aux variables as "free"
    // In bellman circuits, these are variables from alloc() closures whose
    // values must be provided externally (e.g., boolean witness bits,
    // signature bytes, Merkle paths, etc.)
    let mut free_vars = Vec::new();
    for i in 0..num_vars {
      if !known[i] {
        free_vars.push(i as u32);
        known[i] = true; // mark as known for Phase 3
      }
    }

    // Phase 3: Re-process deferred constraints with free variables now known
    // This resolves constraints like XOR/AND where the inputs are free
    // (from alloc closures) and the output is determined by the constraint.
    progress = true;
    while progress && !deferred.is_empty() {
      progress = false;
      deferred.retain(|&row| {
        let prev_ops = ops.len();
        Self::process_row(row, num_vars, a_matrix, b_matrix, c_matrix, &mut known, &mut ops, &mut Vec::new());
        if ops.len() > prev_ops {
          progress = true;
          false
        } else if Self::all_vars_known(row, a_matrix, b_matrix, c_matrix, &known) {
          progress = true;
          false
        } else {
          true
        }
      });
    }

    // Phase 4: Any variables still unknown after all passes are truly unresolvable
    // Check for any remaining unknown variables (shouldn't happen)
    for i in 0..num_vars {
      if !known[i] {
        // This variable was marked free in Phase 2 but somehow became unknown again.
        // This shouldn't happen.
        return Err(format!("Variable {} is still unknown after all passes", i));
      }
    }

    if !deferred.is_empty() {
      return Err(format!(
        "Failed to resolve {} constraints after marking {} free variables",
        deferred.len(),
        free_vars.len()
      ));
    }

    Ok(WitnessProgram {
      ops,
      num_vars,
      num_io,
      free_vars,
    })
  }

  /// Process a single constraint row: if it has exactly one unknown aux variable,
  /// emit an instruction and mark that variable as known.
  fn process_row(
    row: usize,
    num_vars: usize,
    a: &super::SparseMatrix<F>,
    b: &super::SparseMatrix<F>,
    c: &super::SparseMatrix<F>,
    known: &mut [bool],
    ops: &mut Vec<WitnessOp<F>>,
    deferred: &mut Vec<usize>,
  ) {
    // Collect terms from each matrix row
    let get_terms = |m: &super::SparseMatrix<F>| -> Vec<(usize, F)> {
      let start = m.indptr[row];
      let end = m.indptr[row + 1];
      m.indices[start..end]
        .iter()
        .zip(&m.data[start..end])
        .map(|(&col, &val)| (col, val))
        .collect()
    };

    let a_terms = get_terms(a);
    let b_terms = get_terms(b);
    let c_terms = get_terms(c);

    // Find unknown aux variables across all three matrices (deduplicated)
    let mut unknown_vars: Vec<usize> = Vec::new();
    for terms in [&a_terms, &b_terms, &c_terms] {
      for &(col, _) in terms {
        if col < num_vars && !known[col] && !unknown_vars.contains(&col) {
          unknown_vars.push(col);
        }
      }
    }

    if unknown_vars.len() != 1 {
      if !unknown_vars.is_empty() {
        deferred.push(row);
      }
      // 0 unknowns = check constraint, skip
      return;
    }

    let dst = unknown_vars[0];

    // Determine coefficients of dst in each matrix
    let find_coeff = |terms: &[(usize, F)]| -> F {
      terms.iter()
        .find(|&&(col, _)| col == dst)
        .map(|&(_, v)| v)
        .unwrap_or(F::ZERO)
    };

    let a_dst_coeff = find_coeff(&a_terms);
    let b_dst_coeff = find_coeff(&b_terms);
    let c_dst_coeff = find_coeff(&c_terms);

    // Check if dst appears in both A and B (would make it quadratic)
    let in_a = a_dst_coeff != F::ZERO;
    let in_b = b_dst_coeff != F::ZERO;
    let in_c = c_dst_coeff != F::ZERO;

    if in_a && in_b {
      // Quadratic in dst — cannot solve linearly
      deferred.push(row);
      return;
    }

    // Filter out dst from the terms to get "known" terms
    let a_known: Vec<(u32, F)> = a_terms.iter()
      .filter(|&&(col, _)| col != dst)
      .map(|&(col, val)| (col as u32, val))
      .collect();
    let b_known: Vec<(u32, F)> = b_terms.iter()
      .filter(|&&(col, _)| col != dst)
      .map(|&(col, val)| (col as u32, val))
      .collect();
    let c_known: Vec<(u32, F)> = c_terms.iter()
      .filter(|&&(col, _)| col != dst)
      .map(|&(col, val)| (col as u32, val))
      .collect();

    if !in_a && !in_b && in_c {
      // dst only in C — the simple and most common case
      // x = (eval_A * eval_B - C_known) / c_dst_coeff
      let inv = c_dst_coeff.invert();
      if bool::from(inv.is_none()) {
        deferred.push(row);
        return;
      }
      let inv_c = inv.unwrap();

      // Try specialized patterns
      let op = Self::make_op_c_only(
        dst as u32,
        &a_terms,
        &b_terms,
        a_known,
        b_known,
        c_known,
        inv_c,
      );
      known[dst] = true;
      ops.push(op);
    } else {
      // dst in A or B (and possibly C) — general linear case
      // x = (C_known - A_known * B_known) / (a_dst * B_known + b_dst * A_known - c_dst)
      known[dst] = true;
      ops.push(WitnessOp::GeneralLinear {
        dst: dst as u32,
        a_known,
        b_known,
        c_known,
        a_dst: a_dst_coeff,
        b_dst: b_dst_coeff,
        c_dst: c_dst_coeff,
      });
    }
  }

  /// Check if all variables in a constraint row are known
  fn all_vars_known(
    row: usize,
    a: &super::SparseMatrix<F>,
    b: &super::SparseMatrix<F>,
    c: &super::SparseMatrix<F>,
    known: &[bool],
  ) -> bool {
    let check = |m: &super::SparseMatrix<F>| -> bool {
      let start = m.indptr[row];
      let end = m.indptr[row + 1];
      m.indices[start..end].iter().all(|&col| known[col])
    };
    check(a) && check(b) && check(c)
  }

  /// Create an optimized operation for the common case: dst only in C.
  fn make_op_c_only(
    dst: u32,
    a_terms_full: &[(usize, F)],
    b_terms_full: &[(usize, F)],
    a_known: Vec<(u32, F)>,
    b_known: Vec<(u32, F)>,
    c_known: Vec<(u32, F)>,
    inv_c: F,
  ) -> WitnessOp<F> {
    // Pattern: A·z = single_var, B·z = single_var, C_known = empty
    // → z[dst] = a_coeff * z[a] * b_coeff * z[b] * inv_c
    if a_terms_full.len() == 1 && b_terms_full.len() == 1 && c_known.is_empty() {
      let (a_col, a_coeff) = a_terms_full[0];
      let (b_col, b_coeff) = b_terms_full[0];
      let combined = a_coeff * b_coeff * inv_c;

      if combined == F::ONE {
        return WitnessOp::Mul {
          dst,
          a: a_col as u32,
          b: b_col as u32,
        };
      }
      return WitnessOp::MulScaled {
        dst,
        a: a_col as u32,
        b: b_col as u32,
        coeff: combined,
      };
    }

    WitnessOp::GeneralC {
      dst,
      a_terms: a_known,
      b_terms: b_known,
      c_known,
      inv_c_coeff: inv_c,
    }
  }
}
