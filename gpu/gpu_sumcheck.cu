// GPU sumcheck C API for BN254 ppsnark.
//
// Three-phase design matching transcript ordering:
// Phase 0: Static upload (row, col, val_A/B/C, ts_row, ts_col) - once per circuit
// Phase 1: After tau - compute L_row, L_col via gather
// Phase 2: After c, gamma, r - compute val, memory hash, batch_invert
// Phase 3: After rho - construct eq polynomials, setup sumcheck

#include <cuda.h>
#include <cstdio>
#include <cstring>

#include <ff/alt_bn128.hpp>

#include "sumcheck_kernels.cuh"

#define NUM_POLYS 21
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============== Static GPU state ==============
static fr_t* d_static_data = nullptr;
static fr_t* d_val_A = nullptr;
static fr_t* d_val_B = nullptr;
static fr_t* d_val_C = nullptr;
static fr_t* d_ts_row_static = nullptr;
static fr_t* d_ts_col_static = nullptr;
static fr_t* d_row_fr = nullptr;
static fr_t* d_col_fr = nullptr;
static uint32_t* d_row_int = nullptr;
static uint32_t* d_col_int = nullptr;
static uint32_t g_static_n = 0;

// ============== Per-proof GPU state ==============
static fr_t* d_poly_data = nullptr;
static fr_t* d_polys[NUM_POLYS];
static fr_t** d_poly_ptrs = nullptr;
static fr_t* d_block_results = nullptr;
static fr_t* d_challenge = nullptr;
static fr_t* d_eq_tau = nullptr;
static fr_t* d_z_padded = nullptr;
static fr_t* d_products = nullptr;
static uint32_t g_n = 0;
static uint32_t g_num_blocks = 0;
static int g_grid = 0;
static fr_t* h_block_results = nullptr;

// ============== HyperKZG CUDA kernels ==============
// Must be outside #ifndef __CUDA_ARCH__ to compile for both device and host.

// Fold kernel: out[j] = in[2j] + x * (in[2j+1] - in[2j])
__global__
void hkzg_fold_kernel(const fr_t* __restrict__ in, fr_t* __restrict__ out,
                       const fr_t x, uint32_t half_n)
{
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half_n) return;
    out[j] = in[2*j] + x * (in[2*j+1] - in[2*j]);
}

// Chunked Horner: each thread evaluates one chunk via Horner's method
__global__
void hkzg_chunked_horner_kernel(const fr_t* __restrict__ f, fr_t* __restrict__ chunk_results,
                                 const fr_t u, uint32_t n, uint32_t chunk_sz)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = tid * chunk_sz;
    if (start >= n) return;
    uint32_t end = start + chunk_sz;
    if (end > n) end = n;
    fr_t acc;
    acc = f[end - 1];
    for (uint32_t i = end - 1; i > start; ) {
        --i;
        acc = acc * u + f[i];
    }
    chunk_results[tid] = acc;
}

// 3-point chunked Horner: evaluates one chunk at 3 points simultaneously
// Results stored interleaved: chunk_results[tid*3+p] for point p
__global__
void hkzg_chunked_horner_3pt_kernel(const fr_t* __restrict__ f, fr_t* __restrict__ chunk_results,
                                     const fr_t u0, const fr_t u1, const fr_t u2,
                                     uint32_t n, uint32_t chunk_sz)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = tid * chunk_sz;
    if (start >= n) return;
    uint32_t end = start + chunk_sz;
    if (end > n) end = n;
    fr_t a0 = f[end - 1], a1 = a0, a2 = a0;
    for (uint32_t i = end - 1; i > start; ) {
        --i;
        fr_t fi = f[i];
        a0 = a0 * u0 + fi;
        a1 = a1 * u1 + fi;
        a2 = a2 * u2 + fi;
    }
    chunk_results[tid*3]   = a0;
    chunk_results[tid*3+1] = a1;
    chunk_results[tid*3+2] = a2;
}

// SAXPY: out[i] = a[i] + s * b[i] (out may alias a or b)
__global__
void hkzg_saxpy_kernel(fr_t* out, const fr_t* __restrict__ a,
                        const fr_t* __restrict__ b, const fr_t s, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = a[i] + s * b[i];
}

// ============== Mercury CUDA kernels ==============

// Transpose kernel: out[col*num_rows + row] = in[row*num_cols + col]
__global__
void mercury_transpose_kernel(const fr_t* __restrict__ in, fr_t* __restrict__ out,
                               uint32_t num_rows, uint32_t num_cols)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = num_rows * num_cols;
    if (idx >= total) return;
    uint32_t row = idx / num_cols;
    uint32_t col = idx % num_cols;
    out[col * num_rows + row] = in[idx];
}

// Batch Horner: each thread does one column's Horner division
// Input: N elements, logically b rows × b cols, stored row-major
// For column col_id: extract coeffs[col_id], coeffs[col_id + b], coeffs[col_id + 2*b], ...
// Do Horner division by alpha: q[n-1] = c[n-1]; q[i] = c[i] + alpha * q[i+1] (from top)
// Actually: the Horner division f(X)/(X-alpha): q[i] = c[i+1] + alpha*q[i+1], q[b-2] = c[b-1]
// Store quotient in-place, remainder in separate array
__global__
void mercury_batch_horner_strided_kernel(const fr_t* __restrict__ in, fr_t* __restrict__ quot_out,
                                          fr_t* __restrict__ rem_out,
                                          const fr_t alpha, uint32_t num_rows, uint32_t num_cols)
{
    uint32_t col_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (col_id >= num_cols) return;

    // Horner division: polynomial is in[col_id + i*num_cols] for i=0..num_rows-1
    // Division by (X - alpha)
    // q[n-1] is not produced (it would be leading coeff / 1 = c[n-1])
    // Actually, divide_by_linear_polynomial does:
    // for i = (len-2) downto 0: coeffs[i] += coeffs[i+1] * a
    // then remove coeffs[0] as remainder

    // Load column
    fr_t prev = in[col_id + (num_rows - 1) * num_cols];
    for (int i = (int)num_rows - 2; i >= 0; i--) {
        fr_t cur = in[col_id + i * num_cols];
        cur = cur + prev * alpha;
        // quot_out stores in column-major order for each column
        // We write quotient[i] for this column
        // After division, coeffs shift: quotient has length num_rows-1
        // coeffs[i] for i=1..num_rows-1 become quotient[i-1]
        // Wait - let me match the CPU code exactly.
        // CPU: for i in (0..len-1).rev(): coeffs[i] += coeffs[i+1] * a
        //      remainder = coeffs.remove(0)
        // So after the loop, coeffs[0] has the remainder,
        // and coeffs[1..] is the quotient.
        if (i > 0) {
            // This is an intermediate step - store for next iteration
            prev = cur;
        } else {
            // i=0: cur is the remainder
            rem_out[col_id] = cur;
            // prev (from i=1) was already stored as the quotient's first coeff
        }
    }

    // We need to redo this more carefully to store quotient coefficients.
    // Let me use a different approach: load all, compute, store.
    // For num_rows up to ~2048, this fits in registers/local memory.

    // Actually, num_rows = b = sqrt(N) ≈ 1024-2048. Too large for registers.
    // Use global memory: read column, process, write back.
    // The output quotient for this column goes into quot_out at positions:
    // quot_out[col_id + i * num_cols] for i = 0..num_rows-2

    // Recompute from scratch (we messed up above):
    // After Horner division of f by (X - alpha):
    // Process from high to low:
    prev = in[col_id + (num_rows - 1) * num_cols];
    quot_out[col_id + (num_rows - 2) * num_cols] = prev;
    for (int i = (int)num_rows - 2; i >= 1; i--) {
        prev = in[col_id + i * num_cols] + prev * alpha;
        quot_out[col_id + (i - 1) * num_cols] = prev;
    }
    // remainder
    rem_out[col_id] = in[col_id] + prev * alpha;
}

// Parallel prefix scan for polynomial division by (X - a)
// The recurrence: q[n-2] = c[n-1], q[i] = c[i+1] + a*q[i+1]
// Represented as linear function composition (suffix scan):
// f_i(x) = a*x + c[i+1], compose: (m1,b1)∘(m2,b2) = (m1*m2, m1*b2+b1)
// Two-phase Blelloch scan approach.
// Phase 1 (upsweep): compute partial compositions within blocks
// Phase 2 (downsweep): propagate prefix across blocks

// Each block processes BLOCK_SIZE elements
// Block output: composed (m, b) for the entire block
__global__
void mercury_horner_scan_phase1(const fr_t* __restrict__ coeffs, 
                                 fr_t* __restrict__ block_m, fr_t* __restrict__ block_b,
                                 fr_t* __restrict__ local_m, fr_t* __restrict__ local_b,
                                 const fr_t a, uint32_t n)
{
    // n = number of quotient coefficients = poly_len - 1
    // coeffs has poly_len elements, but we use coeffs[1..poly_len] as the addends
    // q[i] = a * q[i+1] + coeffs[i+1], for i = n-2 downto 0
    // q[n-1] = coeffs[n] (the last coeff)
    
    // We process right-to-left. Thread with logical index j handles quotient position (n-1-j).
    // But for a scan, let's reindex left-to-right:
    // Define r[j] = q[n-1-j], then r[0] = q[n-1] = coeffs[n]
    // r[j] = a * r[j-1] + coeffs[n-j], for j >= 1
    // This is a left-to-right scan: r[j] = a * r[j-1] + c'[j]
    // where c'[j] = coeffs[n-j]
    
    uint32_t tid = threadIdx.x;
    uint32_t block_start = blockIdx.x * blockDim.x;
    uint32_t gid = block_start + tid;
    
    // Each element represents the linear function f(x) = a*x + c'[gid]
    // For gid=0, the "initial value" will be handled separately
    fr_t my_m = a;
    fr_t my_b;
    if (gid < n) {
        my_b = coeffs[n - gid];  // c'[gid] = coeffs[n - gid]
    } else {
        // Identity function for out-of-range
        my_m = fr_t::one();
        my_b = fr_t();
    }
    
    // Store local (m, b) for each element before scan
    if (gid < n) {
        local_m[gid] = my_m;
        local_b[gid] = my_b;
    }
    
    // Inclusive scan within block using shared memory
    // fr_t is 32 bytes (8 uint32_t). Block size is 256, so 2*256*32 = 16KB shared mem.
    __shared__ uint32_t s_m_raw[256 * 8], s_b_raw[256 * 8];
    fr_t* s_m = reinterpret_cast<fr_t*>(s_m_raw);
    fr_t* s_b = reinterpret_cast<fr_t*>(s_b_raw);
    s_m[tid] = my_m;
    s_b[tid] = my_b;
    __syncthreads();
    
    // Hillis-Steele inclusive prefix scan (left to right)
    // We want result[j](x) = f_j(f_{j-1}(...f_0(x)...))
    // Compose: (my_m, my_b) applied AFTER (prev_m, prev_b):
    //   composed(x) = my_m * (prev_m * x + prev_b) + my_b = (my_m*prev_m)*x + (my_m*prev_b + my_b)
    for (uint32_t stride = 1; stride < blockDim.x; stride <<= 1) {
        fr_t prev_m, prev_b;
        if (tid >= stride) {
            prev_m = s_m[tid - stride];
            prev_b = s_b[tid - stride];
        }
        __syncthreads();
        if (tid >= stride) {
            // compose: current ∘ prev = (my_m * prev_m, my_m * prev_b + my_b)
            fr_t my_m_cur = s_m[tid];
            fr_t my_b_cur = s_b[tid];
            s_m[tid] = my_m_cur * prev_m;
            s_b[tid] = my_m_cur * prev_b + my_b_cur;
        }
        __syncthreads();
    }
    
    // Write block aggregate (last thread's result)
    if (tid == blockDim.x - 1 || gid == n - 1) {
        block_m[blockIdx.x] = s_m[tid];
        block_b[blockIdx.x] = s_b[tid];
    }
    
    // Write per-element scanned results
    if (gid < n) {
        local_m[gid] = s_m[tid];
        local_b[gid] = s_b[tid];
    }
}

// Phase 2: propagate block prefixes and compute final quotient values
__global__
void mercury_horner_scan_phase2(fr_t* __restrict__ quotient,
                                 const fr_t* __restrict__ local_m, const fr_t* __restrict__ local_b,
                                 const fr_t* __restrict__ prefix_m, const fr_t* __restrict__ prefix_b,
                                 const fr_t initial_val, uint32_t n, uint32_t n_orig)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    
    fr_t m = local_m[gid];
    fr_t b = local_b[gid];
    
    // Apply block prefix if not the first block
    // composed = local ∘ prefix: (m * pm, m * pb + b)
    if (blockIdx.x > 0) {
        fr_t pm = prefix_m[blockIdx.x - 1];
        fr_t pb = prefix_b[blockIdx.x - 1];
        b = m * pb + b;
        m = m * pm;
    }
    
    // r[gid] = m * initial_val + b
    // But for gid=0: r[0] = coeffs[n] = initial_val directly
    // Actually: element 0's function is f(x) = a*x + coeffs[n]
    // After inclusive scan, composed[0] = (a, coeffs[n])
    // r[0] = a * (what?) + coeffs[n]
    // The scan gives us the composed function. We need to apply it to the "seed" value.
    // For the Horner recurrence, r[0] = coeffs[n] (no prior dependence)
    // r[j] = composed_j(r[-1]) where r[-1] is "0" conceptually
    // Actually: the composed function for [0..j] applied to 0 gives r[j]
    // Because: r[0] = a*0 + coeffs[n] = coeffs[n] ✓
    // r[1] = a*r[0] + coeffs[n-1] = composed_{0,1}(0) ✓
    
    fr_t result = b;  // m * 0 + b = b (applying composed function to 0)
    
    // Map back: r[gid] = q[n-1-gid], so quotient[n-1-gid] = result
    uint32_t qi = n - 1 - gid;
    quotient[qi] = result;
}

// Simple element-wise operations for Mercury polynomial arithmetic
__global__
void mercury_poly_sub_scaled_kernel(fr_t* __restrict__ out, const fr_t* __restrict__ a,
                                     const fr_t scale, uint32_t n)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = out[i] + scale * a[i];
}

// compute_h_poly: h[row] = sum_{col=0}^{b-1} f[row*b + col] * eq_col[col]
__global__
void mercury_h_poly_kernel(const fr_t* __restrict__ f, const fr_t* __restrict__ eq_col,
                            fr_t* __restrict__ h, uint32_t b)
{
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    // b is both num_rows and num_cols
    if (row >= b) return;
    fr_t acc;
    acc = fr_t();
    for (uint32_t col = 0; col < b; col++) {
        acc = acc + f[row * b + col] * eq_col[col];
    }
    h[row] = acc;
}

#ifndef __CUDA_ARCH__

// HyperKZG host-side state
static fr_t* d_hkzg_poly = nullptr;
static fr_t** d_hkzg_levels = nullptr;
static uint32_t hkzg_ell = 0;
static uint32_t hkzg_n = 0;

extern "C" {

// ============== Phase 0: Static upload ==============
int gpu_static_upload(
    const uint32_t* row_int, const uint32_t* col_int,
    const void* row_fr, const void* col_fr,
    const void* val_A, const void* val_B, const void* val_C,
    const void* ts_row, const void* ts_col,
    uint32_t n)
{
    g_static_n = n;
    size_t fr_bytes = (size_t)n * sizeof(fr_t);
    size_t int_bytes = (size_t)n * sizeof(uint32_t);

    CHECK_CUDA(cudaMalloc(&d_static_data, 7 * fr_bytes));
    d_row_fr = d_static_data;
    d_col_fr = d_static_data + n;
    d_val_A = d_static_data + 2*(size_t)n;
    d_val_B = d_static_data + 3*(size_t)n;
    d_val_C = d_static_data + 4*(size_t)n;
    d_ts_row_static = d_static_data + 5*(size_t)n;
    d_ts_col_static = d_static_data + 6*(size_t)n;

    CHECK_CUDA(cudaMemcpy(d_row_fr, row_fr, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_fr, col_fr, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val_A, val_A, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val_B, val_B, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_val_C, val_C, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ts_row_static, ts_row, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ts_col_static, ts_col, fr_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_row_int, int_bytes));
    CHECK_CUDA(cudaMalloc(&d_col_int, int_bytes));
    CHECK_CUDA(cudaMemcpy(d_row_int, row_int, int_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_int, col_int, int_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

void gpu_static_free() {
    if (d_static_data) { cudaFree(d_static_data); d_static_data = nullptr; }
    if (d_row_int) { cudaFree(d_row_int); d_row_int = nullptr; }
    if (d_col_int) { cudaFree(d_col_int); d_col_int = nullptr; }
    d_val_A = d_val_B = d_val_C = d_ts_row_static = d_ts_col_static = nullptr;
    d_row_fr = d_col_fr = nullptr;
    g_static_n = 0;
}

// ============== Phase 1: After tau ==============
int gpu_phase1_init(const void* eq_tau_host, const void* z_padded_host, uint32_t n) {
    g_n = n;
    size_t fr_bytes = (size_t)n * sizeof(fr_t);

    int device; cudaGetDevice(&device);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, device);
    g_num_blocks = prop.multiProcessorCount * 2;
    g_grid = (n + 255) / 256;
    if (g_grid > (int)g_num_blocks * 4) g_grid = g_num_blocks * 4;

    CHECK_CUDA(cudaMalloc(&d_poly_data, (size_t)NUM_POLYS * fr_bytes));
    for (int i = 0; i < NUM_POLYS; i++)
        d_polys[i] = d_poly_data + (size_t)i * n;

    CHECK_CUDA(cudaMalloc(&d_eq_tau, fr_bytes));
    CHECK_CUDA(cudaMalloc(&d_z_padded, fr_bytes));
    CHECK_CUDA(cudaMalloc(&d_challenge, 4 * sizeof(fr_t)));

    CHECK_CUDA(cudaMemcpy(d_eq_tau, eq_tau_host, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_z_padded, z_padded_host, fr_bytes, cudaMemcpyHostToDevice));

    gather_kernel<<<g_grid, 256>>>(d_eq_tau, d_row_int, d_polys[14], n);
    gather_kernel<<<g_grid, 256>>>(d_z_padded, d_col_int, d_polys[15], n);
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

int gpu_download_L(void* L_row_host, void* L_col_host) {
    size_t fr_bytes = (size_t)g_n * sizeof(fr_t);
    CHECK_CUDA(cudaMemcpy(L_row_host, d_polys[14], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(L_col_host, d_polys[15], fr_bytes, cudaMemcpyDeviceToHost));
    return 0;
}

// ============== Phase 2: After c, gamma, r ==============
int gpu_phase2_construct(
    const void* Az_host, const void* Bz_host, const void* uCz_E_host, const void* Mz_host,
    const void* W_host, const void* masked_eq_host,
    const void* c_host, const void* gamma_host, const void* r_host, uint32_t n)
{
    size_t fr_bytes = (size_t)n * sizeof(fr_t);

    CHECK_CUDA(cudaMemcpy(d_polys[10], Az_host, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_polys[11], Bz_host, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_polys[12], uCz_E_host, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_polys[13], Mz_host, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_polys[17], W_host, fr_bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_polys[18], masked_eq_host, fr_bytes, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(d_challenge, c_host, sizeof(fr_t), cudaMemcpyHostToDevice));
    compute_val_kernel<<<g_grid, 256>>>(d_val_A, d_val_B, d_val_C, d_challenge, d_polys[16], n);

    CHECK_CUDA(cudaMemcpy(d_polys[4], d_ts_row_static, fr_bytes, cudaMemcpyDeviceToDevice));
    CHECK_CUDA(cudaMemcpy(d_polys[9], d_ts_col_static, fr_bytes, cudaMemcpyDeviceToDevice));

    CHECK_CUDA(cudaMemcpy(d_challenge, gamma_host, sizeof(fr_t), cudaMemcpyHostToDevice));
    mem_hash_kernel<<<g_grid, 256>>>(d_eq_tau, d_row_fr, d_polys[14], d_challenge,
                                      d_polys[0], d_polys[2], n);
    mem_hash_kernel<<<g_grid, 256>>>(d_z_padded, d_col_fr, d_polys[15], d_challenge,
                                      d_polys[5], d_polys[7], n);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(d_challenge, r_host, sizeof(fr_t), cudaMemcpyHostToDevice));
    add_scalar_kernel<<<g_grid, 256>>>(d_polys[0], d_challenge, n);
    add_scalar_kernel<<<g_grid, 256>>>(d_polys[2], d_challenge, n);
    add_scalar_kernel<<<g_grid, 256>>>(d_polys[5], d_challenge, n);
    add_scalar_kernel<<<g_grid, 256>>>(d_polys[7], d_challenge, n);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Batch inversion (Montgomery's trick with chunked prefix products)
    CHECK_CUDA(cudaMalloc(&d_products, fr_bytes));

    auto batch_inv = [&](fr_t* d_arr, fr_t* d_inv_out, uint32_t len) -> int {
        uint32_t csz = 4096;
        uint32_t nc = (len + csz - 1) / csz;
        batch_invert_prefix_kernel<<<(nc+255)/256, 256>>>(d_arr, d_products, len, csz);

        // Gather chunk ends using GPU kernel, then single bulk download
        fr_t* d_ends_tmp;
        CHECK_CUDA(cudaMalloc(&d_ends_tmp, nc * sizeof(fr_t)));
        gather_chunk_ends_kernel<<<(nc+255)/256, 256>>>(d_products, d_ends_tmp, nc, csz, len);
        fr_t* h_ends = (fr_t*)malloc(nc * sizeof(fr_t));
        CHECK_CUDA(cudaMemcpy(h_ends, d_ends_tmp, nc * sizeof(fr_t), cudaMemcpyDeviceToHost));
        cudaFree(d_ends_tmp);

        fr_t* h_prods = (fr_t*)malloc(nc * sizeof(fr_t));
        h_prods[0] = h_ends[0];
        for (uint32_t i = 1; i < nc; i++) h_prods[i] = h_prods[i-1] * h_ends[i];

        fr_t total_inv = h_prods[nc-1].reciprocal();

        fr_t* h_invs = (fr_t*)malloc(nc * sizeof(fr_t));
        for (uint32_t i = nc; i > 1; ) { --i; h_invs[i] = total_inv * h_prods[i-1]; total_inv = total_inv * h_ends[i]; }
        h_invs[0] = total_inv;

        fr_t* d_invs; CHECK_CUDA(cudaMalloc(&d_invs, nc * sizeof(fr_t)));
        CHECK_CUDA(cudaMemcpy(d_invs, h_invs, nc * sizeof(fr_t), cudaMemcpyHostToDevice));
        batch_invert_back_kernel<<<(nc+255)/256, 256>>>(d_arr, d_products, d_invs, d_inv_out, len, csz);
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaFree(d_invs); free(h_ends); free(h_prods); free(h_invs);
        return 0;
    };

    if (batch_inv(d_polys[0], d_polys[1], n) != 0) return -1;
    if (batch_inv(d_polys[2], d_polys[3], n) != 0) return -1;
    if (batch_inv(d_polys[5], d_polys[6], n) != 0) return -1;
    if (batch_inv(d_polys[7], d_polys[8], n) != 0) return -1;

    elemwise_mul_kernel<<<g_grid, 256>>>(d_polys[1], d_polys[4], d_polys[1], n);
    elemwise_mul_kernel<<<g_grid, 256>>>(d_polys[6], d_polys[9], d_polys[6], n);
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

int gpu_download_mem_oracles(
    void* t_inv_row, void* w_inv_row, void* t_inv_col, void* w_inv_col,
    void* t_row, void* w_row, void* t_col, void* w_col)
{
    size_t fr_bytes = (size_t)g_n * sizeof(fr_t);
    CHECK_CUDA(cudaMemcpy(t_inv_row, d_polys[1], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(w_inv_row, d_polys[3], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(t_inv_col, d_polys[6], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(w_inv_col, d_polys[8], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(t_row, d_polys[0], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(w_row, d_polys[2], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(t_col, d_polys[5], fr_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(w_col, d_polys[7], fr_bytes, cudaMemcpyDeviceToHost));
    return 0;
}

// ============== Phase 3: After rho ==============
int gpu_phase3_setup_sumcheck(const void* rho_host, const void* tau_host, uint32_t num_rounds) {
    fr_t one = fr_t::one();

    CHECK_CUDA(cudaMemcpy(d_polys[19], &one, sizeof(fr_t), cudaMemcpyHostToDevice));
    for (uint32_t r = 0; r < num_rounds; r++) {
        uint32_t ps = 1u << r;
        // Process in reverse order to match CPU's evals_from_points(r.iter().rev())
        CHECK_CUDA(cudaMemcpy(d_challenge, (const fr_t*)rho_host + (num_rounds - 1 - r), sizeof(fr_t), cudaMemcpyHostToDevice));
        eq_expand_kernel<<<max(1,(int)((ps+255)/256)), 256>>>(d_polys[19], d_challenge, ps);
    }

    CHECK_CUDA(cudaMemcpy(d_polys[20], &one, sizeof(fr_t), cudaMemcpyHostToDevice));
    for (uint32_t r = 0; r < num_rounds; r++) {
        uint32_t ps = 1u << r;
        // Process in reverse order to match CPU's evals_from_points(r.iter().rev())
        CHECK_CUDA(cudaMemcpy(d_challenge, (const fr_t*)tau_host + (num_rounds - 1 - r), sizeof(fr_t), cudaMemcpyHostToDevice));
        eq_expand_kernel<<<max(1,(int)((ps+255)/256)), 256>>>(d_polys[20], d_challenge, ps);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMalloc(&d_poly_ptrs, NUM_POLYS * sizeof(fr_t*)));
    CHECK_CUDA(cudaMemcpy(d_poly_ptrs, d_polys, NUM_POLYS * sizeof(fr_t*), cudaMemcpyHostToDevice));

    size_t brb = g_num_blocks * 3 * sizeof(fr_t);
    CHECK_CUDA(cudaMalloc(&d_block_results, brb));
    h_block_results = (fr_t*)malloc(brb);

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

// ============== Sumcheck operations ==============
static void reduce_blocks(fr_t* out0, fr_t* out1, fr_t* out2) {
    cudaMemcpy(h_block_results, d_block_results, g_num_blocks*3*sizeof(fr_t), cudaMemcpyDeviceToHost);
    fr_t s0, s1, s2; memset(&s0,0,sizeof(fr_t)); memset(&s1,0,sizeof(fr_t)); memset(&s2,0,sizeof(fr_t));
    for (uint32_t b = 0; b < g_num_blocks; b++) { s0+=h_block_results[b*3]; s1+=h_block_results[b*3+1]; s2+=h_block_results[b*3+2]; }
    *out0=s0; *out1=s1; *out2=s2;
}

int gpu_sumcheck_eval_round(uint32_t half_n, void* results) {
    fr_t* r = (fr_t*)results;
    sc_reduce_linear<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[1],d_polys[3],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[0],&r[1],&r[2]);
    sc_reduce_linear<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[6],d_polys[8],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[3],&r[4],&r[5]);
    sc_reduce_cubic_3in<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[1],d_polys[0],d_polys[4],d_polys[19],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[6],&r[7],&r[8]);
    sc_reduce_cubic_2in<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[3],d_polys[2],d_polys[19],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[9],&r[10],&r[11]);
    sc_reduce_cubic_3in<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[6],d_polys[5],d_polys[9],d_polys[19],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[12],&r[13],&r[14]);
    sc_reduce_cubic_2in<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[8],d_polys[7],d_polys[19],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[15],&r[16],&r[17]);
    sc_reduce_cubic_3in<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[10],d_polys[11],d_polys[12],d_polys[20],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[18],&r[19],&r[20]);
    sc_reduce_cubic_1in<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[13],d_polys[20],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[21],&r[22],&r[23]);
    sc_reduce_cubic_deg3<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[14],d_polys[15],d_polys[16],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[24],&r[25],&r[26]);
    sc_reduce_quadratic<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_polys[18],d_polys[17],half_n,d_block_results); cudaDeviceSynchronize(); reduce_blocks(&r[27],&r[28],&r[29]);
    return 0;
}

int gpu_sumcheck_bind(const void* r, uint32_t half_n) {
    CHECK_CUDA(cudaMemcpy(d_challenge, r, sizeof(fr_t), cudaMemcpyHostToDevice));
    sumcheck_bind_kernel<<<g_num_blocks,SC_BLOCK_SIZE>>>(d_poly_ptrs, NUM_POLYS, d_challenge, half_n);
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

int gpu_sumcheck_get_final(uint32_t poly_id, void* result) {
    if (poly_id >= NUM_POLYS) return -1;
    CHECK_CUDA(cudaMemcpy(result, d_polys[poly_id], sizeof(fr_t), cudaMemcpyDeviceToHost));
    return 0;
}

// Return device pointer for a sumcheck polynomial (for device-side MSM).
void* gpu_get_poly_device_ptr(uint32_t poly_id) {
    if (poly_id >= NUM_POLYS) return nullptr;
    return (void*)d_polys[poly_id];
}

void gpu_sumcheck_free() {
    if (d_poly_data)     { cudaFree(d_poly_data);     d_poly_data = nullptr; }
    if (d_poly_ptrs)     { cudaFree(d_poly_ptrs);     d_poly_ptrs = nullptr; }
    if (d_block_results) { cudaFree(d_block_results); d_block_results = nullptr; }
    if (d_challenge)     { cudaFree(d_challenge);     d_challenge = nullptr; }
    if (d_eq_tau)        { cudaFree(d_eq_tau);        d_eq_tau = nullptr; }
    if (d_z_padded)      { cudaFree(d_z_padded);      d_z_padded = nullptr; }
    if (d_products)      { cudaFree(d_products);      d_products = nullptr; }
    if (h_block_results) { free(h_block_results);     h_block_results = nullptr; }
    g_n = 0; g_num_blocks = 0;
}

// Upload hat_P and fold ell-1 times. Stores device pointers for each level.
// x_challenges[i] is used for fold level i (applied as x[ell-1-i] in caller).
void gpu_hkzg_free(); // forward declaration
int gpu_hkzg_fold(const void* hat_P, uint32_t n, const void* x_challenges, uint32_t ell) {
    // Free any prior allocation (safe for parallel test runs)
    gpu_hkzg_free();

    hkzg_ell = ell;
    hkzg_n = n;

    // Allocate levels array on host
    d_hkzg_levels = (fr_t**)malloc(ell * sizeof(fr_t*));
    if (!d_hkzg_levels) return -1;

    // Allocate device memory for all levels
    // Level 0: n elements, Level 1: n/2, ..., Level ell-1: 1
    // Total: 2n - 1 elements. Allocate 2n for safety.
    CHECK_CUDA(cudaMalloc(&d_hkzg_poly, 2 * (size_t)n * sizeof(fr_t)));

    // Level 0 starts at offset 0
    d_hkzg_levels[0] = d_hkzg_poly;
    CHECK_CUDA(cudaMemcpy(d_hkzg_levels[0], hat_P, (size_t)n * sizeof(fr_t), cudaMemcpyHostToDevice));

    // Set up pointers for subsequent levels (contiguous after level 0)
    size_t offset = n;
    for (uint32_t i = 1; i < ell; i++) {
        d_hkzg_levels[i] = d_hkzg_poly + offset;
        offset += n >> i;
    }

    // Fold ell-1 times
    const fr_t* x = reinterpret_cast<const fr_t*>(x_challenges);
    for (uint32_t i = 0; i < ell - 1; i++) {
        uint32_t half = n >> (i + 1);
        uint32_t blocks = (half + 255) / 256;
        hkzg_fold_kernel<<<blocks, 256>>>(d_hkzg_levels[i], d_hkzg_levels[i+1], x[i], half);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

// Get device pointer for a folded level (for device-side MSM commit).
void* gpu_hkzg_get_level_ptr(uint32_t level) {
    if (!d_hkzg_levels || level >= hkzg_ell) return nullptr;
    return (void*)d_hkzg_levels[level];
}

// Evaluate all folded polynomials at 3 points u[0], u[1], u[2].
// Output: v[i][j] = polys[i](u[j]), stored as v[i*3+j] in row-major order.
// Returns ell * 3 scalars.
int gpu_hkzg_eval(const void* u_host, void* v_host, uint32_t ell) {
    if (!d_hkzg_levels) return -1;

    const fr_t* u = reinterpret_cast<const fr_t*>(u_host);
    fr_t* v = reinterpret_cast<fr_t*>(v_host);

    const uint32_t CHUNK_SZ = 1024;
    const uint32_t CPU_THRESHOLD = 4096;

    // Allocate chunk results buffer once (3× for 3-point kernel)
    uint32_t max_chunks = (hkzg_n + CHUNK_SZ - 1) / CHUNK_SZ;
    fr_t* d_chunks;
    CHECK_CUDA(cudaMalloc(&d_chunks, max_chunks * 3 * sizeof(fr_t)));

    // Pre-compute u^CHUNK_SZ for each point
    fr_t u_pow[3];
    for (int p = 0; p < 3; p++) {
        u_pow[p] = u[p];
        for (uint32_t k = 1; k < CHUNK_SZ; k++) u_pow[p] = u_pow[p] * u[p];
    }

    for (uint32_t i = 0; i < ell; i++) {
        uint32_t poly_len = hkzg_n >> i;

        if (poly_len <= 1) {
            CHECK_CUDA(cudaMemcpy(&v[i*3], d_hkzg_levels[i], sizeof(fr_t), cudaMemcpyDeviceToHost));
            v[i*3+1] = v[i*3];
            v[i*3+2] = v[i*3];
            continue;
        }

        if (poly_len <= CPU_THRESHOLD) {
            fr_t* h_poly = (fr_t*)malloc(poly_len * sizeof(fr_t));
            CHECK_CUDA(cudaMemcpy(h_poly, d_hkzg_levels[i], poly_len * sizeof(fr_t), cudaMemcpyDeviceToHost));
            for (int p = 0; p < 3; p++) {
                fr_t acc = h_poly[poly_len - 1];
                for (uint32_t j = poly_len - 1; j > 0; ) {
                    --j;
                    acc = acc * u[p] + h_poly[j];
                }
                v[i*3+p] = acc;
            }
            free(h_poly);
            continue;
        }

        // GPU: single 3-point kernel per level (1 launch instead of 3)
        uint32_t num_chunks = (poly_len + CHUNK_SZ - 1) / CHUNK_SZ;
        uint32_t blocks = (num_chunks + 255) / 256;
        hkzg_chunked_horner_3pt_kernel<<<blocks, 256>>>(
            d_hkzg_levels[i], d_chunks, u[0], u[1], u[2], poly_len, CHUNK_SZ);
        CHECK_CUDA(cudaDeviceSynchronize());

        fr_t* h_chunks = (fr_t*)malloc(num_chunks * 3 * sizeof(fr_t));
        CHECK_CUDA(cudaMemcpy(h_chunks, d_chunks, num_chunks * 3 * sizeof(fr_t), cudaMemcpyDeviceToHost));

        for (int p = 0; p < 3; p++) {
            fr_t result = h_chunks[(num_chunks - 1) * 3 + p];
            for (uint32_t k = num_chunks - 1; k > 0; ) {
                --k;
                result = result * u_pow[p] + h_chunks[k * 3 + p];
            }
            v[i*3+p] = result;
        }
        free(h_chunks);
    }
    cudaFree(d_chunks);
    return 0;
}

// Compute batch polynomial B = sum(q^k * f[k]) on GPU.
// f[k] = d_hkzg_levels[k], each with length n >> k.
// B has length n (= hkzg_n), zero-padded for shorter polynomials.
int gpu_hkzg_batch_poly(const void* q_host, void* B_host, uint32_t k_count) {
    if (!d_hkzg_levels) return -1;

    fr_t q = *reinterpret_cast<const fr_t*>(q_host);
    uint32_t n = hkzg_n;

    // Allocate output buffer on GPU
    fr_t* d_B;
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)n * sizeof(fr_t)));
    CHECK_CUDA(cudaMemset(d_B, 0, (size_t)n * sizeof(fr_t)));

    // B = sum(q^k * f[k]) — accumulate with Horner: B = f[0] + q*(f[1] + q*(f[2] + ...))
    // Process from last to first for Horner
    // Start with B = f[k_count-1] (smallest poly)
    uint32_t last_len = n >> (k_count - 1);
    CHECK_CUDA(cudaMemcpy(d_B, d_hkzg_levels[k_count - 1], last_len * sizeof(fr_t), cudaMemcpyDeviceToDevice));

    for (int k = (int)k_count - 2; k >= 0; k--) {
        uint32_t len_k = n >> k;
        uint32_t blocks = (len_k + 255) / 256;
        // B[0..len_k] = f[k][0..len_k] + q * B[0..len_k]
        hkzg_saxpy_kernel<<<blocks, 256>>>(d_B, d_hkzg_levels[k], d_B, q, len_k);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Download result
    CHECK_CUDA(cudaMemcpy(B_host, d_B, (size_t)n * sizeof(fr_t), cudaMemcpyDeviceToHost));
    cudaFree(d_B);
    return 0;
}

void gpu_hkzg_free() {
    if (d_hkzg_poly) { cudaFree(d_hkzg_poly); d_hkzg_poly = nullptr; }
    if (d_hkzg_levels) { free(d_hkzg_levels); d_hkzg_levels = nullptr; }
    hkzg_ell = 0; hkzg_n = 0;
}

// Generic device-to-host memcpy wrapper
int gpu_memcpy_dtoh(void* dst, const void* d_src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, d_src, bytes, cudaMemcpyDeviceToHost);
    return err == cudaSuccess ? 0 : -1;
}

// ============== Mercury C API ==============

// Mercury: divide polynomial by (X - a) using GPU parallel prefix scan.
// Input: coeffs (host, poly_len elements), a (host, 1 element)
// Output: quotient (host, poly_len-1 elements), remainder (host, 1 element)
int gpu_mercury_divide_by_linear(const void* h_coeffs, uint32_t poly_len,
                                  const void* h_a, void* h_quotient, void* h_remainder) {
    if (poly_len < 2) return -1;
    
    const fr_t* coeffs = (const fr_t*)h_coeffs;
    const fr_t a = *(const fr_t*)h_a;
    uint32_t n = poly_len - 1;  // quotient length
    
    // Upload polynomial
    fr_t* d_coeffs;
    CHECK_CUDA(cudaMalloc(&d_coeffs, (size_t)poly_len * sizeof(fr_t)));
    CHECK_CUDA(cudaMemcpy(d_coeffs, coeffs, (size_t)poly_len * sizeof(fr_t), cudaMemcpyHostToDevice));
    
    const uint32_t BLOCK = 256;
    uint32_t num_blocks = (n + BLOCK - 1) / BLOCK;
    
    // Allocate scan buffers
    fr_t* d_local_m, *d_local_b;
    fr_t* d_block_m, *d_block_b;
    fr_t* d_quotient;
    CHECK_CUDA(cudaMalloc(&d_local_m, (size_t)n * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_local_b, (size_t)n * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_block_m, (size_t)num_blocks * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_block_b, (size_t)num_blocks * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_quotient, (size_t)n * sizeof(fr_t)));
    
    // Phase 1: block-level inclusive scan
    mercury_horner_scan_phase1<<<num_blocks, BLOCK>>>(d_coeffs, d_block_m, d_block_b,
                                                       d_local_m, d_local_b, a, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Phase 1.5: scan the block aggregates (small enough for CPU or recursive GPU)
    if (num_blocks > 1) {
        // Download block results, scan on CPU, re-upload
        fr_t* h_bm = (fr_t*)malloc(num_blocks * sizeof(fr_t));
        fr_t* h_bb = (fr_t*)malloc(num_blocks * sizeof(fr_t));
        CHECK_CUDA(cudaMemcpy(h_bm, d_block_m, num_blocks * sizeof(fr_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_bb, d_block_b, num_blocks * sizeof(fr_t), cudaMemcpyDeviceToHost));
        
        // Inclusive prefix scan of (m, b) pairs
        // compose: current ∘ prev = (cur_m * prev_m, cur_m * prev_b + cur_b)
        for (uint32_t i = 1; i < num_blocks; i++) {
            fr_t cur_m = h_bm[i], cur_b = h_bb[i];
            h_bm[i] = cur_m * h_bm[i-1];
            h_bb[i] = cur_m * h_bb[i-1] + cur_b;
        }
        
        CHECK_CUDA(cudaMemcpy(d_block_m, h_bm, num_blocks * sizeof(fr_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_block_b, h_bb, num_blocks * sizeof(fr_t), cudaMemcpyHostToDevice));
        free(h_bm);
        free(h_bb);
    }
    
    // Phase 2: compute final quotient values
    fr_t zero;
    memset(&zero, 0, sizeof(fr_t));
    mercury_horner_scan_phase2<<<num_blocks, BLOCK>>>(d_quotient, d_local_m, d_local_b,
                                                       d_block_m, d_block_b, zero, n, poly_len);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Download quotient
    CHECK_CUDA(cudaMemcpy(h_quotient, d_quotient, (size_t)n * sizeof(fr_t), cudaMemcpyDeviceToHost));
    
    // Compute remainder: r = coeffs[0] + a * q[0]
    // Download q[0]
    fr_t q0;
    CHECK_CUDA(cudaMemcpy(&q0, d_quotient, sizeof(fr_t), cudaMemcpyDeviceToHost));
    fr_t rem = coeffs[0] + a * q0;
    memcpy(h_remainder, &rem, sizeof(fr_t));
    
    cudaFree(d_coeffs);
    cudaFree(d_local_m);
    cudaFree(d_local_b);
    cudaFree(d_block_m);
    cudaFree(d_block_b);
    cudaFree(d_quotient);
    return 0;
}

// Mercury: divide_by_binomial on GPU
// f(X) / (X^b - alpha), f has b*b elements in row-major order
// Returns quotient (b*(b-1) elements, transposed to coeff order) and remainder (b elements)
// If d_quot_out is non-null, stores a COPY of the device quotient pointer there (caller must cudaFree)
int gpu_mercury_divide_by_binomial(const void* h_coeffs, uint32_t b,
                                    const void* h_alpha, void* h_quotient, void* h_remainder,
                                    void** d_quot_out) {
    uint32_t n = b * b;
    const fr_t alpha = *(const fr_t*)h_alpha;
    
    fr_t* d_in;
    fr_t* d_quot;
    fr_t* d_rem;
    fr_t* d_transposed = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, (size_t)n * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_quot, (size_t)n * sizeof(fr_t)));  // b cols * (b-1) rows, padded
    CHECK_CUDA(cudaMalloc(&d_rem, (size_t)b * sizeof(fr_t)));
    CHECK_CUDA(cudaMemcpy(d_in, h_coeffs, (size_t)n * sizeof(fr_t), cudaMemcpyHostToDevice));
    
    // Launch batch Horner: one thread per column
    uint32_t blocks = (b + 255) / 256;
    mercury_batch_horner_strided_kernel<<<blocks, 256>>>(d_in, d_quot, d_rem, alpha, b, b);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Transpose quotient: currently stored as b cols × (b-1) rows in strided layout
    // Actually the kernel stores in row-major with stride b (same as input layout but one fewer row)
    // We need to transpose it
    // Output of kernel: quot_out[col_id + i * num_cols] for i = 0..num_rows-2
    // That's (b-1) rows × b cols, row-major
    // We need to transpose to b cols × (b-1) rows = b*(b-1) in column-major
    // Actually, looking at CPU code: it first collects quotient per-column (each length b-1),
    // flattens them (all columns concatenated), then transposes.
    // Our kernel stores the quotient in the SAME row-major layout as input (with one fewer row).
    // The CPU code then does a transpose on the flattened result.
    // Let me match: CPU creates quotients as col-major (each column's quotient is a separate vec),
    // then does transpose(num_rows-1, num_cols) to interleave back.
    
    // Actually our kernel writes quot_out[col_id + i * b] for i=0..b-2
    // This is effectively (b-1) rows × b cols in row-major
    // CPU code: flatten columns → [col0_q0..col0_q(b-2), col1_q0..col1_q(b-2), ...]
    //   then expand each to length b (zero-pad) → b × b
    //   then transpose(b-1, b) on this b×b thing
    // That's complex. Let me just match the CPU output format directly.
    
    // Our kernel output: row i (0..b-2), col j (0..b-1): quot[j + i*b]
    // CPU output after transpose: the coefficient of X^k in the quotient
    // The quotient of f(X)/(X^b - alpha) has degree < b*(b-1)
    // CPU code: quotient.coeffs stores this in standard coeff order
    
    // Actually, let me think about this differently.
    // CPU does: for each col, Horner div gives quotient of length b-1
    // Then it pads each to length b (zero at end), getting b vectors of length b
    // Then concatenates to b*b, then transposes(b-1, b) which treats it as (b-1) rows × b cols → b rows × (b-1) cols
    // Wait, transpose(num_rows, num_cols) where num_rows=b-1, num_cols=b
    // After transpose: result[new_row * (b-1) + new_col] comes from [new_col * b + new_row]
    // Hmm this is getting complex. Let me just allocate a temp and use the transpose kernel.
    
    // Simpler approach: our kernel already puts quotient in row-major (b-1 rows × b cols).
    // The CPU code wants the coefficients of the polynomial quotient.
    // The relationship is: the quotient poly Q(X) of f(X) / (X^b - alpha) has degree < b*(b-1).
    // Q(X) = sum_{k=0}^{b^2-b-1} q_k X^k
    // Each column c (0..b-1) gives the sub-polynomial coefficients:
    // q_{c + j*b} for j=0..b-2 from the Horner division of column c.
    // So Q's coefficient at position (c + j*b) = column_quotient[c][j].
    
    // Our kernel stores quot_out[c + j*b] = column_quotient[c][j], which is exactly q_{c+j*b}.
    // So the quotient coefficients are already in the right order! No transpose needed if we
    // expanded each column to length b (zero-padded).
    // But CPU code does transpose because it concatenates columns (col0_q, col1_q, ...)
    // which gives [q_{0}, q_{b}, q_{2b}, ..., q_{1}, q_{b+1}, q_{2b+1}, ...] — wrong order.
    // It then transposes to fix the interleaving.
    
    // Our kernel stores in ROW-MAJOR: quot_out[c + j*b], which when flattened gives:
    // j=0: [q_0, q_1, ..., q_{b-1}]  (first "row" of quotient coeffs)
    // j=1: [q_b, q_{b+1}, ..., q_{2b-1}]
    // ...
    // This is exactly the standard coefficient order! No transpose needed.
    
    // Download remainder only (quotient stays on device for MSM)
    CHECK_CUDA(cudaMemcpy(h_remainder, d_rem, (size_t)b * sizeof(fr_t), cudaMemcpyDeviceToHost));
    
    // Also download quotient to host (needed later for EE)
    CHECK_CUDA(cudaMemcpy(h_quotient, d_quot, (size_t)(b - 1) * b * sizeof(fr_t), cudaMemcpyDeviceToHost));
    
    // If caller wants device pointer, transfer ownership instead of freeing
    if (d_quot_out) {
        *d_quot_out = d_quot;
    } else {
        cudaFree(d_quot);
    }
    
    cudaFree(d_in);
    cudaFree(d_rem);
    cudaFree(d_transposed);
    return 0;
}

// Mercury: quot_f = (f - scale*q - g_zeta) / (X - zeta)
// Combines batch_add + divide_by_linear into one GPU call
// f_coeffs: N elements, q_coeffs: up to N elements, scale: scalar
// g_zeta: scalar to subtract from coeff[0]
// zeta: divisor point
// Output: quotient (N-1 elements)
int gpu_mercury_quot_f(const void* h_f, uint32_t f_len,
                       const void* h_q, uint32_t q_len,
                       const void* h_scale,
                       const void* h_g_zeta,
                       const void* h_zeta,
                       void* h_quotient,
                       void* h_remainder,
                       void** d_quot_out) {
    const fr_t scale = *(const fr_t*)h_scale;
    const fr_t g_zeta = *(const fr_t*)h_g_zeta;
    const fr_t zeta = *(const fr_t*)h_zeta;
    
    // Upload f and q
    fr_t* d_f;
    fr_t* d_q;
    CHECK_CUDA(cudaMalloc(&d_f, (size_t)f_len * sizeof(fr_t)));
    CHECK_CUDA(cudaMemcpy(d_f, h_f, (size_t)f_len * sizeof(fr_t), cudaMemcpyHostToDevice));
    
    if (q_len > 0) {
        CHECK_CUDA(cudaMalloc(&d_q, (size_t)q_len * sizeof(fr_t)));
        CHECK_CUDA(cudaMemcpy(d_q, h_q, (size_t)q_len * sizeof(fr_t), cudaMemcpyHostToDevice));
        
        // f += scale * q (element-wise, only for first q_len elements)
        uint32_t blocks = (q_len + 255) / 256;
        mercury_poly_sub_scaled_kernel<<<blocks, 256>>>(d_f, d_q, scale, q_len);
        cudaFree(d_q);
    }
    
    // f[0] -= g_zeta
    fr_t neg_g_zeta;
    memset(&neg_g_zeta, 0, sizeof(fr_t));
    // Download f[0], subtract, upload
    fr_t f0;
    CHECK_CUDA(cudaMemcpy(&f0, d_f, sizeof(fr_t), cudaMemcpyDeviceToHost));
    f0 = f0 - g_zeta;
    CHECK_CUDA(cudaMemcpy(d_f, &f0, sizeof(fr_t), cudaMemcpyHostToDevice));
    
    // Now divide d_f by (X - zeta) using parallel prefix scan
    uint32_t n = f_len - 1;  // quotient length
    const uint32_t BLOCK = 256;
    uint32_t num_blocks = (n + BLOCK - 1) / BLOCK;
    
    fr_t* d_local_m, *d_local_b;
    fr_t* d_block_m, *d_block_b;
    fr_t* d_quotient;
    CHECK_CUDA(cudaMalloc(&d_local_m, (size_t)n * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_local_b, (size_t)n * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_block_m, (size_t)num_blocks * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_block_b, (size_t)num_blocks * sizeof(fr_t)));
    CHECK_CUDA(cudaMalloc(&d_quotient, (size_t)n * sizeof(fr_t)));
    
    mercury_horner_scan_phase1<<<num_blocks, BLOCK>>>(d_f, d_block_m, d_block_b,
                                                       d_local_m, d_local_b, zeta, n);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    if (num_blocks > 1) {
        fr_t* h_bm = (fr_t*)malloc(num_blocks * sizeof(fr_t));
        fr_t* h_bb = (fr_t*)malloc(num_blocks * sizeof(fr_t));
        CHECK_CUDA(cudaMemcpy(h_bm, d_block_m, num_blocks * sizeof(fr_t), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_bb, d_block_b, num_blocks * sizeof(fr_t), cudaMemcpyDeviceToHost));
        
        for (uint32_t i = 1; i < num_blocks; i++) {
            fr_t cur_m = h_bm[i], cur_b = h_bb[i];
            h_bm[i] = cur_m * h_bm[i-1];
            h_bb[i] = cur_m * h_bb[i-1] + cur_b;
        }
        
        CHECK_CUDA(cudaMemcpy(d_block_m, h_bm, num_blocks * sizeof(fr_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_block_b, h_bb, num_blocks * sizeof(fr_t), cudaMemcpyHostToDevice));
        free(h_bm);
        free(h_bb);
    }
    
    fr_t zero;
    memset(&zero, 0, sizeof(fr_t));
    mercury_horner_scan_phase2<<<num_blocks, BLOCK>>>(d_quotient, d_local_m, d_local_b,
                                                       d_block_m, d_block_b, zero, n, f_len);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaMemcpy(h_quotient, d_quotient, (size_t)n * sizeof(fr_t), cudaMemcpyDeviceToHost));
    
    // Compute remainder
    fr_t q0;
    CHECK_CUDA(cudaMemcpy(&q0, d_quotient, sizeof(fr_t), cudaMemcpyDeviceToHost));
    fr_t f0_orig;
    CHECK_CUDA(cudaMemcpy(&f0_orig, d_f, sizeof(fr_t), cudaMemcpyDeviceToHost));
    fr_t rem = f0_orig + zeta * q0;
    memcpy(h_remainder, &rem, sizeof(fr_t));
    
    // If caller wants device pointer, transfer ownership instead of freeing
    if (d_quot_out) {
        *d_quot_out = d_quotient;
    } else {
        cudaFree(d_quotient);
    }
    
    cudaFree(d_f);
    cudaFree(d_local_m);
    cudaFree(d_local_b);
    cudaFree(d_block_m);
    cudaFree(d_block_b);
    return 0;
}

// Free a device pointer allocated by the above functions
void gpu_free_device(void* d_ptr) {
    if (d_ptr) cudaFree(d_ptr);
}

} // extern "C"
#endif
