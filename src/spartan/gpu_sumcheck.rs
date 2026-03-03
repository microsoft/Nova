#![allow(unsafe_code)]

//! GPU-accelerated sumcheck for BN254 ppsnark.
//!
//! Three-phase design matching transcript ordering:
//! - Phase 0: Static upload (row, col, val_A/B/C, ts_row, ts_col) — once per circuit.
//! - Phase 1: After tau — compute L_row, L_col via GPU gather.
//! - Phase 2: After c, gamma, r — compute val, memory hash, batch inversion.
//! - Phase 3: After rho — construct eq polynomials, setup sumcheck.
//!
//! All scalars transferred in Montgomery form (raw internal representation).

use halo2curves::bn256;
use std::sync::Mutex;

type Scalar = bn256::Fr;

const SCALAR_BYTES: usize = 32;
const NUM_CLAIMS: usize = 10;
const RESULTS_PER_ROUND: usize = NUM_CLAIMS * 3;

static GPU_SC_LOCK: Mutex<()> = Mutex::new(());

extern "C" {
  // Phase 0: Static upload
  fn gpu_static_upload(
    row_int: *const u32, col_int: *const u32,
    row_fr: *const u8, col_fr: *const u8,
    val_A: *const u8, val_B: *const u8, val_C: *const u8,
    ts_row: *const u8, ts_col: *const u8,
    n: u32,
  ) -> i32;
  fn gpu_static_free();

  // Phase 1: After tau
  fn gpu_phase1_init(eq_tau: *const u8, z_padded: *const u8, n: u32) -> i32;
  fn gpu_download_L(L_row: *mut u8, L_col: *mut u8) -> i32;
  fn gpu_upload_to_poly(host_data: *const u8, poly_id: u32, n: u32, d_ptr_out: *mut *mut u8) -> i32;

  // Phase 2: After c, gamma, r — v2 computes uCz_E and Mz on GPU
  fn gpu_phase2_construct_v2(
    Az: *const u8, Bz: *const u8, Cz: *const u8, E: *const u8,
    W: *const u8, masked_eq: *const u8,
    u: *const u8, c: *const u8, gamma: *const u8, r: *const u8,
    n: u32,
  ) -> i32;
  fn gpu_download_mem_oracles(
    t_plus_r_inv_row: *mut u8, w_plus_r_inv_row: *mut u8,
    t_plus_r_inv_col: *mut u8, w_plus_r_inv_col: *mut u8,
    t_plus_r_row: *mut u8, w_plus_r_row: *mut u8,
    t_plus_r_col: *mut u8, w_plus_r_col: *mut u8,
  ) -> i32;

  // Phase 3: After rho
  fn gpu_phase3_setup_sumcheck(rho: *const u8, tau: *const u8, num_rounds: u32) -> i32;

  // Sumcheck operations
  fn gpu_sumcheck_eval_round(half_n: u32, results: *mut u8) -> i32;
  fn gpu_sumcheck_bind(r: *const u8, half_n: u32) -> i32;
  fn gpu_sumcheck_get_final(poly_id: u32, result: *mut u8) -> i32;
  fn gpu_sumcheck_free();

  // Device pointer access
  fn gpu_get_poly_device_ptr(poly_id: u32) -> *mut u8;

  // HyperKZG GPU operations
  fn gpu_hkzg_fold(hat_P: *const u8, n: u32, x_challenges: *const u8, ell: u32) -> i32;
  fn gpu_hkzg_get_level_ptr(level: u32) -> *mut u8;
  fn gpu_hkzg_eval(u_points: *const u8, v_out: *mut u8, ell: u32) -> i32;
  fn gpu_hkzg_batch_poly(q: *const u8, B_out: *mut u8, k_count: u32) -> i32;
  fn gpu_hkzg_free();
  fn gpu_memcpy_dtoh(dst: *mut u8, d_src: *const u8, bytes: usize) -> i32;

  // Mercury GPU operations
  fn gpu_mercury_divide_by_linear(
    coeffs: *const u8,
    poly_len: u32,
    a: *const u8,
    quotient: *mut u8,
    remainder: *mut u8,
  ) -> i32;

  fn gpu_mercury_divide_by_binomial(
    coeffs: *const u8,
    b: u32,
    alpha: *const u8,
    quotient: *mut u8,
    remainder: *mut u8,
    d_quot_out: *mut *mut u8,
  ) -> i32;

  fn gpu_mercury_quot_f(
    f: *const u8,
    f_len: u32,
    q: *const u8,
    q_len: u32,
    scale: *const u8,
    g_zeta: *const u8,
    zeta: *const u8,
    quotient: *mut u8,
    remainder: *mut u8,
    d_quot_out: *mut *mut u8,
  ) -> i32;

  fn gpu_free_device(d_ptr: *mut u8);
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

/// Upload static circuit data to GPU (called once per circuit).
use std::sync::atomic::{AtomicU64, Ordering};

static STATIC_FINGERPRINT: AtomicU64 = AtomicU64::new(0);

/// Compute a fingerprint for the static data to detect changes.
/// Compute a fingerprint for the static data to detect changes.
fn compute_static_fingerprint(row_int: &[u32], n: usize) -> u64 {
  // Use first+last element and size as a fast fingerprint
  let first = if n > 0 { row_int[0] as u64 } else { 0 };
  let last = if n > 0 { row_int[n - 1] as u64 } else { 0 };
  first ^ (last << 32) ^ (n as u64).wrapping_mul(0x517cc1b727220a95)
}

/// Upload static circuit data to GPU. Caches based on fingerprint to skip re-upload.
pub fn static_upload(
  row_int: &[u32],
  col_int: &[u32],
  row_fr: &[Scalar],
  col_fr: &[Scalar],
  val_a: &[Scalar],
  val_b: &[Scalar],
  val_c: &[Scalar],
  ts_row: &[Scalar],
  ts_col: &[Scalar],
) -> Result<(), String> {
  let n = row_int.len();

  // Check if static data is already uploaded (same circuit)
  let fp = compute_static_fingerprint(row_int, n);
  if STATIC_FINGERPRINT.load(Ordering::Relaxed) == fp {
    return Ok(());
  }

  let _lock = GPU_SC_LOCK.lock().unwrap();
  let ret = unsafe {
    gpu_static_upload(
      row_int.as_ptr(), col_int.as_ptr(),
      row_fr.as_ptr() as *const u8, col_fr.as_ptr() as *const u8,
      val_a.as_ptr() as *const u8, val_b.as_ptr() as *const u8, val_c.as_ptr() as *const u8,
      ts_row.as_ptr() as *const u8, ts_col.as_ptr() as *const u8,
      n as u32,
    )
  };
  if ret != 0 {
    return Err("gpu_static_upload failed".to_string());
  }
  STATIC_FINGERPRINT.store(fp, Ordering::Relaxed);
  Ok(())
}

/// Phase 1: Upload eq(tau) and z, compute L_row/L_col via GPU gather.
pub fn phase1_init(eq_tau: &[Scalar], z_padded: &[Scalar]) -> Result<(), String> {
  let n = eq_tau.len() as u32;
  let _lock = GPU_SC_LOCK.lock().unwrap();
  let ret = unsafe {
    gpu_phase1_init(
      eq_tau.as_ptr() as *const u8,
      z_padded.as_ptr() as *const u8,
      n,
    )
  };
  if ret != 0 {
    return Err("gpu_phase1_init failed".to_string());
  }
  Ok(())
}

/// Download L_row and L_col computed by phase 1.
pub fn phase1_download_l(n: usize) -> Result<(Vec<Scalar>, Vec<Scalar>), String> {
  let mut l_row = vec![Scalar::default(); n];
  let mut l_col = vec![Scalar::default(); n];
  let _lock = GPU_SC_LOCK.lock().unwrap();
  let ret = unsafe {
    gpu_download_L(
      l_row.as_mut_ptr() as *mut u8,
      l_col.as_mut_ptr() as *mut u8,
    )
  };
  if ret != 0 {
    return Err("gpu_download_L failed".to_string());
  }
  Ok((l_row, l_col))
}

/// Phase 2 v2: Upload raw Az/Bz/Cz/E, compute uCz_E and Mz on GPU.
/// Saves CPU computation + 2 array uploads vs the original API.
pub fn phase2_construct(
  az: &[Scalar], bz: &[Scalar], cz: &[Scalar], e: &[Scalar],
  w: &[Scalar], masked_eq: &[Scalar],
  u: &Scalar, c: &Scalar, gamma: &Scalar, r: &Scalar,
) -> Result<(), String> {
  let n = az.len() as u32;
  let _lock = GPU_SC_LOCK.lock().unwrap();
  let ret = unsafe {
    gpu_phase2_construct_v2(
      az.as_ptr() as *const u8, bz.as_ptr() as *const u8,
      cz.as_ptr() as *const u8, e.as_ptr() as *const u8,
      w.as_ptr() as *const u8, masked_eq.as_ptr() as *const u8,
      scalar_to_raw(u).as_ptr(), scalar_to_raw(c).as_ptr(),
      scalar_to_raw(gamma).as_ptr(), scalar_to_raw(r).as_ptr(),
      n,
    )
  };
  if ret != 0 {
    return Err("gpu_phase2_construct_v2 failed".to_string());
  }
  Ok(())
}

/// Upload a host array to a GPU poly slot and return device pointer.
pub fn upload_to_poly(data: &[Scalar], poly_id: u32) -> Result<*mut u8, String> {
  let n = data.len() as u32;
  let mut d_ptr: *mut u8 = std::ptr::null_mut();
  let _lock = GPU_SC_LOCK.lock().unwrap();
  let ret = unsafe {
    gpu_upload_to_poly(data.as_ptr() as *const u8, poly_id, n, &mut d_ptr)
  };
  if ret != 0 {
    return Err("gpu_upload_to_poly failed".to_string());
  }
  Ok(d_ptr)
}

/// Download memory oracle + auxiliary polynomials from phase 2.
/// Download only the 4 memory oracle polynomials from GPU (skip aux for GPU path).
pub fn phase2_download_mem_oracles_only(n: usize) -> Result<[Vec<Scalar>; 4], String> {
  let fr_bytes = n * std::mem::size_of::<Scalar>();
  let _lock = GPU_SC_LOCK.lock().unwrap();
  // Oracle poly IDs: 1 (t_inv_row), 3 (w_inv_row), 6 (t_inv_col), 8 (w_inv_col)
  let poly_ids = [1u32, 3, 6, 8];
  let oracle: [Vec<Scalar>; 4] = std::array::from_fn(|i| {
    let mut v = Vec::<Scalar>::with_capacity(n);
    let d_ptr = unsafe { gpu_get_poly_device_ptr(poly_ids[i]) };
    let ret = unsafe { gpu_memcpy_dtoh(v.as_mut_ptr() as *mut u8, d_ptr, fr_bytes) };
    assert_eq!(ret, 0, "gpu_memcpy_dtoh for poly {} failed", poly_ids[i]);
    unsafe { v.set_len(n) };
    v
  });
  Ok(oracle)
}

/// Download all 8 memory oracle + aux polynomials from GPU.
pub fn phase2_download_mem_oracles(n: usize) -> Result<([Vec<Scalar>; 4], [Vec<Scalar>; 4]), String> {
  let mut oracle = [
    vec![Scalar::default(); n],
    vec![Scalar::default(); n],
    vec![Scalar::default(); n],
    vec![Scalar::default(); n],
  ];
  let mut aux = [
    vec![Scalar::default(); n],
    vec![Scalar::default(); n],
    vec![Scalar::default(); n],
    vec![Scalar::default(); n],
  ];
  let _lock = GPU_SC_LOCK.lock().unwrap();
  let ret = unsafe {
    gpu_download_mem_oracles(
      oracle[0].as_mut_ptr() as *mut u8, oracle[1].as_mut_ptr() as *mut u8,
      oracle[2].as_mut_ptr() as *mut u8, oracle[3].as_mut_ptr() as *mut u8,
      aux[0].as_mut_ptr() as *mut u8, aux[1].as_mut_ptr() as *mut u8,
      aux[2].as_mut_ptr() as *mut u8, aux[3].as_mut_ptr() as *mut u8,
    )
  };
  if ret != 0 {
    return Err("gpu_download_mem_oracles failed".to_string());
  }
  Ok((oracle, aux))
}

/// State for a GPU sumcheck session (created by phase 3).
pub struct GpuSumcheckState {
  #[allow(dead_code)]
  n: usize,
  current_half_n: usize,
}

impl GpuSumcheckState {
  /// Phase 3: Construct eq polynomials from rho/tau, setup sumcheck.
  pub fn phase3_setup(rho: &[Scalar], tau: &[Scalar], n: usize) -> Result<Self, String> {
    let num_rounds = rho.len() as u32;
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe {
      gpu_phase3_setup_sumcheck(
        rho.as_ptr() as *const u8,
        tau.as_ptr() as *const u8,
        num_rounds,
      )
    };
    if ret != 0 {
      return Err("gpu_phase3_setup_sumcheck failed".to_string());
    }
    Ok(GpuSumcheckState { n, current_half_n: n / 2 })
  }

  /// Compute evaluation points for one sumcheck round.
  pub fn eval_round(&self) -> Result<Vec<[Scalar; 3]>, String> {
    let mut raw = vec![0u8; RESULTS_PER_ROUND * SCALAR_BYTES];
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe {
      gpu_sumcheck_eval_round(self.current_half_n as u32, raw.as_mut_ptr())
    };
    if ret != 0 {
      return Err("gpu_sumcheck_eval_round failed".to_string());
    }
    let mut results = Vec::with_capacity(NUM_CLAIMS);
    for c in 0..NUM_CLAIMS {
      let base = c * 3 * SCALAR_BYTES;
      results.push([
        scalar_from_raw(&raw[base..base + SCALAR_BYTES]),
        scalar_from_raw(&raw[base + SCALAR_BYTES..base + 2 * SCALAR_BYTES]),
        scalar_from_raw(&raw[base + 2 * SCALAR_BYTES..base + 3 * SCALAR_BYTES]),
      ]);
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
}

impl Drop for GpuSumcheckState {
  fn drop(&mut self) {
    if let Ok(_lock) = GPU_SC_LOCK.lock() {
      unsafe { gpu_sumcheck_free() };
    }
  }
}

/// Free static GPU data.
pub fn static_free() {
  if let Ok(_lock) = GPU_SC_LOCK.lock() {
    unsafe { gpu_static_free() };
  }
}

/// Get device pointer for a sumcheck polynomial (for device-side MSM).
pub fn get_poly_device_ptr(poly_id: u32) -> *mut u8 {
  unsafe { gpu_get_poly_device_ptr(poly_id) }
}

/// Free a device pointer allocated by GPU Mercury functions.
pub fn free_device_ptr(d_ptr: *mut u8) {
  unsafe { gpu_free_device(d_ptr) };
}

/// HyperKZG GPU state for fold, evaluate, and batch polynomial operations.
pub struct GpuHkzgState {
  ell: u32,
  n: u32,
}

impl GpuHkzgState {
  /// Upload hat_P to GPU and fold ell-1 times.
  /// x_challenges[i] corresponds to the challenge used for fold level i
  /// (caller provides x[ell-1-i] from the original point vector).
  pub fn fold(hat_p: &[Scalar], x_challenges: &[Scalar]) -> Result<Self, String> {
    let n = hat_p.len() as u32;
    let ell = x_challenges.len() as u32 + 1;
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe {
      gpu_hkzg_fold(
        hat_p.as_ptr() as *const u8,
        n,
        x_challenges.as_ptr() as *const u8,
        ell,
      )
    };
    if ret != 0 {
      return Err("gpu_hkzg_fold failed".to_string());
    }
    Ok(GpuHkzgState { ell, n })
  }

  /// Get device pointer for a folded polynomial level (for device-side MSM commit).
  pub fn get_level_ptr(&self, level: u32) -> *mut u8 {
    unsafe { gpu_hkzg_get_level_ptr(level) }
  }

  /// Download a folded polynomial level from GPU to CPU.
  pub fn download_level(&self, level: u32) -> Vec<Scalar> {
    let len = (self.n >> level) as usize;
    let mut out = vec![Scalar::default(); len];
    let d_ptr = self.get_level_ptr(level);
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let err = unsafe {
      gpu_memcpy_dtoh(out.as_mut_ptr() as *mut u8, d_ptr as *const u8, len * SCALAR_BYTES)
    };
    assert_eq!(err, 0, "gpu_memcpy_dtoh failed");
    out
  }

  /// Evaluate all folded polynomials at 3 points. Returns v[i][j] = polys[i](u[j]).
  pub fn eval(&self, u: &[Scalar; 3]) -> Result<Vec<[Scalar; 3]>, String> {
    let mut raw = vec![0u8; self.ell as usize * 3 * SCALAR_BYTES];
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe {
      gpu_hkzg_eval(u.as_ptr() as *const u8, raw.as_mut_ptr(), self.ell)
    };
    if ret != 0 {
      return Err("gpu_hkzg_eval failed".to_string());
    }
    let mut results = Vec::with_capacity(self.ell as usize);
    for i in 0..self.ell as usize {
      let base = i * 3 * SCALAR_BYTES;
      results.push([
        scalar_from_raw(&raw[base..base + SCALAR_BYTES]),
        scalar_from_raw(&raw[base + SCALAR_BYTES..base + 2 * SCALAR_BYTES]),
        scalar_from_raw(&raw[base + 2 * SCALAR_BYTES..base + 3 * SCALAR_BYTES]),
      ]);
    }
    Ok(results)
  }

  /// Compute batch polynomial B = sum(q^k * f[k]) on GPU, download to CPU.
  pub fn batch_poly(&self, q: &Scalar) -> Result<Vec<Scalar>, String> {
    let mut b = vec![Scalar::default(); self.n as usize];
    let _lock = GPU_SC_LOCK.lock().unwrap();
    let ret = unsafe {
      gpu_hkzg_batch_poly(
        scalar_to_raw(q).as_ptr(),
        b.as_mut_ptr() as *mut u8,
        self.ell,
      )
    };
    if ret != 0 {
      return Err("gpu_hkzg_batch_poly failed".to_string());
    }
    Ok(b)
  }
}

impl Drop for GpuHkzgState {
  fn drop(&mut self) {
    if let Ok(_lock) = GPU_SC_LOCK.lock() {
      unsafe { gpu_hkzg_free() };
    }
  }
}

// ============== Mercury GPU operations ==============

/// GPU-accelerated polynomial division by (X - a) using parallel prefix scan.
/// Returns (quotient, remainder).
pub fn gpu_poly_divide_by_linear(coeffs: &[Scalar], a: &Scalar) -> Result<(Vec<Scalar>, Scalar), String> {
  let poly_len = coeffs.len();
  if poly_len < 2 {
    return Err("polynomial too short".to_string());
  }

  let mut quotient = vec![Scalar::default(); poly_len - 1];
  let mut remainder = Scalar::default();

  let _lock = GPU_SC_LOCK.lock().map_err(|e| e.to_string())?;
  let ret = unsafe {
    gpu_mercury_divide_by_linear(
      coeffs.as_ptr() as *const u8,
      poly_len as u32,
      a as *const Scalar as *const u8,
      quotient.as_mut_ptr() as *mut u8,
      &mut remainder as *mut Scalar as *mut u8,
    )
  };
  if ret != 0 {
    return Err("gpu_mercury_divide_by_linear failed".to_string());
  }
  Ok((quotient, remainder))
}

/// GPU-accelerated divide_by_binomial: f(X) / (X^b - alpha)
/// Input: f_poly coefficients (b*b elements), b (sqrt of N), alpha
/// Returns: (quotient coeffs, remainder coeffs of length b, optional device pointer)
pub fn gpu_poly_divide_by_binomial(
  coeffs: &[Scalar],
  b: usize,
  alpha: &Scalar,
) -> Result<(Vec<Scalar>, Vec<Scalar>, *mut u8), String> {
  assert_eq!(coeffs.len(), b * b);

  let quot_len = (b - 1) * b;
  let mut quotient = vec![Scalar::default(); quot_len];
  let mut remainder = vec![Scalar::default(); b];
  let mut d_quot: *mut u8 = std::ptr::null_mut();

  let _lock = GPU_SC_LOCK.lock().map_err(|e| e.to_string())?;
  let ret = unsafe {
    gpu_mercury_divide_by_binomial(
      coeffs.as_ptr() as *const u8,
      b as u32,
      alpha as *const Scalar as *const u8,
      quotient.as_mut_ptr() as *mut u8,
      remainder.as_mut_ptr() as *mut u8,
      &mut d_quot,
    )
  };
  if ret != 0 {
    return Err("gpu_mercury_divide_by_binomial failed".to_string());
  }
  Ok((quotient, remainder, d_quot))
}

/// GPU-accelerated quot_f computation:
/// quot_f = (f + scale*q - g_zeta) / (X - zeta)
/// Returns (quotient, remainder, device pointer to quotient for MSM).
pub fn gpu_mercury_compute_quot_f(
  f_coeffs: &[Scalar],
  q_coeffs: &[Scalar],
  scale: &Scalar,
  g_zeta: &Scalar,
  zeta: &Scalar,
) -> Result<(Vec<Scalar>, Scalar, *mut u8), String> {
  let f_len = f_coeffs.len();
  if f_len < 2 {
    return Err("polynomial too short".to_string());
  }

  let mut quotient = vec![Scalar::default(); f_len - 1];
  let mut remainder = Scalar::default();
  let mut d_quot: *mut u8 = std::ptr::null_mut();

  let _lock = GPU_SC_LOCK.lock().map_err(|e| e.to_string())?;
  let ret = unsafe {
    gpu_mercury_quot_f(
      f_coeffs.as_ptr() as *const u8,
      f_len as u32,
      q_coeffs.as_ptr() as *const u8,
      q_coeffs.len() as u32,
      scale as *const Scalar as *const u8,
      g_zeta as *const Scalar as *const u8,
      zeta as *const Scalar as *const u8,
      quotient.as_mut_ptr() as *mut u8,
      &mut remainder as *mut Scalar as *mut u8,
      &mut d_quot,
    )
  };
  if ret != 0 {
    return Err("gpu_mercury_quot_f failed".to_string());
  }
  Ok((quotient, remainder, d_quot))
}
