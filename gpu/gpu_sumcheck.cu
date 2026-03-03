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

} // extern "C"
#endif
