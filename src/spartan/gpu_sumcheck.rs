#![allow(unsafe_code)]

//! GPU-accelerated sumcheck for BN254 ppsnark.
//!
//! Uploads all 21 polynomials to GPU once, then runs evaluation + bind
//! rounds without re-uploading. Per-round communication is ~1KB.
//!
//! IMPORTANT: All scalars are transferred in Montgomery form (raw internal
//! representation), NOT standard form. sppark's fr_t (mont_t) uses the same
//! Montgomery representation as halo2curves::bn256::Fr.

use halo2curves::bn256;
use std::sync::Mutex;

type Scalar = bn256::Fr;

/// Size of a BN254 scalar in bytes (256 bits = 32 bytes).
const SCALAR_BYTES: usize = 32;

/// Number of polynomials in the GPU sumcheck (19 data + 2 eq).
const NUM_POLYS: usize = 21;

/// Number of claims (10 claims × 3 values = 30 results per round).
const NUM_CLAIMS: usize = 10;
const RESULTS_PER_ROUND: usize = NUM_CLAIMS * 3;

static GPU_SC_LOCK: Mutex<()> = Mutex::new(());

extern "C" {
  fn gpu_sumcheck_setup(poly_ptrs: *const *const u8, n: u32) -> i32;
  fn gpu_sumcheck_eval_round(half_n: u32, results: *mut u8) -> i32;
  fn gpu_sumcheck_bind(r: *const u8, half_n: u32) -> i32;
  fn gpu_sumcheck_get_final(poly_id: u32, result: *mut u8) -> i32;
  fn gpu_sumcheck_get_element(poly_id: u32, idx: u32, result: *mut u8) -> i32;
  fn gpu_sumcheck_free();
}

/// Raw bytes of a scalar in Montgomery form (no conversion).
#[inline]
fn scalar_to_raw(s: &Scalar) -> &[u8] {
  unsafe { std::slice::from_raw_parts(s as *const Scalar as *const u8, SCALAR_BYTES) }
}

/// Interpret raw Montgomery-form bytes as a Scalar (no conversion).
#[inline]
fn scalar_from_raw(bytes: &[u8]) -> Scalar {
  debug_assert_eq!(bytes.len(), SCALAR_BYTES);
  unsafe { std::ptr::read(bytes.as_ptr() as *const Scalar) }
}

/// State for a GPU sumcheck session.
pub struct GpuSumcheckState {
  #[allow(dead_code)]
  n: usize,
  current_half_n: usize,
}

impl GpuSumcheckState {
  /// Upload all 21 polynomials to GPU.
  ///
  /// Polynomial order (matching gpu_sumcheck.cu):
  ///   0-9:   Memory instance (t_row, t_inv_row, w_row, w_inv_row, ts_row,
  ///                           t_col, t_inv_col, w_col, w_inv_col, ts_col)
  ///   10-13: Outer instance (Az, Bz, uCz_E, Mz)
  ///   14-16: Inner instance (L_row, L_col, val)
  ///   17-18: Witness instance (W, masked_eq)
  ///   19-20: Eq polynomials (eq_memory, eq_outer)
  pub fn setup(polys: &[&[Scalar]; NUM_POLYS]) -> Result<Self, String> {
    let n = polys[0].len();
    for (i, p) in polys.iter().enumerate() {
      if p.len() != n {
        return Err(format!("polynomial {} has size {} but expected {}", i, p.len(), n));
      }
    }

    // Pass pointers directly to GPU — no intermediate flat buffer
    let ptrs: [*const u8; NUM_POLYS] = std::array::from_fn(|i| polys[i].as_ptr() as *const u8);

    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe { gpu_sumcheck_setup(ptrs.as_ptr(), n as u32) };
    if ret != 0 {
      return Err("gpu_sumcheck_setup failed".to_string());
    }

    Ok(GpuSumcheckState {
      n,
      current_half_n: n / 2,
    })
  }

  /// Compute evaluation points for one sumcheck round.
  ///
  /// Returns 10 triples of (eval_0, bound_coeff, eval_inf), in claim order:
  ///   [0] = linear row, [1] = linear col,
  ///   [2] = cubic3in row, [3] = cubic2in row,
  ///   [4] = cubic3in col, [5] = cubic2in col,
  ///   [6] = cubic3in outer, [7] = cubic1in outer,
  ///   [8] = cubic_deg3 inner, [9] = quadratic witness
  pub fn eval_round(&self) -> Result<Vec<[Scalar; 3]>, String> {
    let mut raw = vec![0u8; RESULTS_PER_ROUND * SCALAR_BYTES];

    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe {
      gpu_sumcheck_eval_round(self.current_half_n as u32, raw.as_mut_ptr())
    };
    if ret != 0 {
      return Err("gpu_sumcheck_eval_round failed".to_string());
    }

    // Parse 30 scalars into 10 triples
    let mut results = Vec::with_capacity(NUM_CLAIMS);
    for c in 0..NUM_CLAIMS {
      let base = c * 3 * SCALAR_BYTES;
      let s0 = scalar_from_raw(&raw[base..base + SCALAR_BYTES]);
      let s1 = scalar_from_raw(&raw[base + SCALAR_BYTES..base + 2 * SCALAR_BYTES]);
      let s2 = scalar_from_raw(&raw[base + 2 * SCALAR_BYTES..base + 3 * SCALAR_BYTES]);
      results.push([s0, s1, s2]);
    }

    Ok(results)
  }

  /// Bind all polynomials with challenge r, halving their effective size.
  pub fn bind(&mut self, r: &Scalar) -> Result<(), String> {
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe { gpu_sumcheck_bind(scalar_to_raw(r).as_ptr(), self.current_half_n as u32) };
    if ret != 0 {
      return Err("gpu_sumcheck_bind failed".to_string());
    }

    self.current_half_n /= 2;
    Ok(())
  }

  /// Get the final scalar value of polynomial `poly_id` (after all rounds).
  pub fn get_final(&self, poly_id: usize) -> Result<Scalar, String> {
    let mut raw = [0u8; SCALAR_BYTES];

    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe { gpu_sumcheck_get_final(poly_id as u32, raw.as_mut_ptr()) };
    if ret != 0 {
      return Err(format!("gpu_sumcheck_get_final({}) failed", poly_id));
    }

    Ok(scalar_from_raw(&raw))
  }
  /// Read element [idx] from polynomial [poly_id] on GPU (for debugging).
  pub fn get_element(&self, poly_id: usize, idx: usize) -> Result<Scalar, String> {
    let mut raw = [0u8; SCALAR_BYTES];
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret =
      unsafe { gpu_sumcheck_get_element(poly_id as u32, idx as u32, raw.as_mut_ptr()) };
    if ret != 0 {
      return Err(format!("gpu_sumcheck_get_element({},{}) failed", poly_id, idx));
    }
    Ok(scalar_from_raw(&raw))
  }
}

impl Drop for GpuSumcheckState {
  fn drop(&mut self) {
    if let Ok(_lock) = GPU_SC_LOCK.lock() {
      unsafe { gpu_sumcheck_free() };
    }
  }
}

/// Materialize eq(tau, x) for all x in {0,1}^n as a vector of N scalars.
pub fn materialize_eq(tau: &[Scalar]) -> Vec<Scalar> {
  crate::spartan::polys::eq::EqPolynomial::evals_from_points(tau)
}
