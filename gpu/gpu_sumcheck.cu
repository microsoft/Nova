// GPU sumcheck C API for BN254 ppsnark.
//
// Manages polynomial data on GPU and dispatches sumcheck reduction/bind kernels.
// All polynomials uploaded once; rounds alternate between eval and bind.
//
// Built via build.rs when the `sppark` feature is enabled.

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

#define CHECK_CUDA_VOID(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// GPU state for one sumcheck session
static fr_t* d_poly_data = nullptr;   // single allocation for all polynomial data
static fr_t* d_polys[NUM_POLYS];      // device pointers into d_poly_data
static fr_t** d_poly_ptrs = nullptr;  // device copy of d_polys array
static fr_t* d_block_results = nullptr;  // per-block reduction output
static fr_t* d_challenge = nullptr;   // single scalar for bind challenge
static uint32_t g_n = 0;             // original polynomial size
static uint32_t g_num_blocks = 0;

// Host buffer for reading back per-block results
static fr_t* h_block_results = nullptr;

#ifndef __CUDA_ARCH__

extern "C" {

// Upload all 21 polynomials to GPU.
// poly_ptrs: array of 21 host pointers, each pointing to n fr_t values
// n: size of each polynomial (must be power of 2)
// Returns 0 on success, -1 on error.
int gpu_sumcheck_setup(const void* const* poly_ptrs, uint32_t n) {
    g_n = n;

    // Determine grid size: use enough blocks to saturate GPU
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    g_num_blocks = prop.multiProcessorCount * 2;  // 2 blocks per SM

    size_t total_bytes = (size_t)NUM_POLYS * n * sizeof(fr_t);
    size_t poly_bytes = (size_t)n * sizeof(fr_t);

    CHECK_CUDA(cudaMalloc(&d_poly_data, total_bytes));

    // Upload each polynomial directly from its host pointer (no intermediate copy)
    for (int i = 0; i < NUM_POLYS; i++) {
        d_polys[i] = d_poly_data + (size_t)i * n;
        CHECK_CUDA(cudaMemcpy(d_polys[i], poly_ptrs[i], poly_bytes, cudaMemcpyHostToDevice));
    }

    CHECK_CUDA(cudaMalloc(&d_poly_ptrs, NUM_POLYS * sizeof(fr_t*)));
    CHECK_CUDA(cudaMemcpy(d_poly_ptrs, d_polys, NUM_POLYS * sizeof(fr_t*),
                          cudaMemcpyHostToDevice));

    // Allocate per-block reduction buffers (max 3 values per block per claim)
    size_t block_result_bytes = g_num_blocks * 3 * sizeof(fr_t);
    CHECK_CUDA(cudaMalloc(&d_block_results, block_result_bytes));
    h_block_results = (fr_t*)malloc(block_result_bytes);

    CHECK_CUDA(cudaMalloc(&d_challenge, sizeof(fr_t)));

    CHECK_CUDA(cudaDeviceSynchronize());
    return 0;
}

// Internal: reduce per-block results to a single (eval_0, bound, inf) triple.
static void reduce_blocks(fr_t* out_eval0, fr_t* out_bound, fr_t* out_inf) {
    cudaMemcpy(h_block_results, d_block_results,
               g_num_blocks * 3 * sizeof(fr_t), cudaMemcpyDeviceToHost);

    fr_t sum0, sum1, sum2;
    memset(&sum0, 0, sizeof(fr_t));
    memset(&sum1, 0, sizeof(fr_t));
    memset(&sum2, 0, sizeof(fr_t));
    for (uint32_t b = 0; b < g_num_blocks; b++) {
        sum0 += h_block_results[b * 3 + 0];
        sum1 += h_block_results[b * 3 + 1];
        sum2 += h_block_results[b * 3 + 2];
    }
    *out_eval0 = sum0;
    *out_bound = sum1;
    *out_inf   = sum2;
}

// Compute all 10 claims' evaluation points for one sumcheck round.
// half_n: current half-size (n/2 for round 0, n/4 for round 1, etc.)
// results: output array of 30 fr_t values (10 claims × 3)
//
// Claim order matches ppsnark prove_helper:
//   [0..5] = memory (2 linear + 4 cubic)
//   [6..7] = outer (2 cubic)
//   [8]    = inner (1 cubic_deg3)
//   [9]    = witness (1 quadratic)
//
// Returns 0 on success.
int gpu_sumcheck_eval_round(uint32_t half_n, void* results) {
    fr_t* res = (fr_t*)results;

    // Claim 0: Linear row (poly 1 = t_inv_row, poly 3 = w_inv_row)
    sc_reduce_linear<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[1], d_polys[3], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[0], &res[1], &res[2]);

    // Claim 1: Linear col (poly 6 = t_inv_col, poly 8 = w_inv_col)
    sc_reduce_linear<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[6], d_polys[8], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[3], &res[4], &res[5]);

    // Claim 2: Cubic3in row (A=t_inv_row=1, B=t_row=0, C=ts_row=4, eq=eq_mem=19)
    sc_reduce_cubic_3in<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[1], d_polys[0], d_polys[4], d_polys[19], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[6], &res[7], &res[8]);

    // Claim 3: Cubic2in row (A=w_inv_row=3, B=w_row=2, eq=eq_mem=19)
    sc_reduce_cubic_2in<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[3], d_polys[2], d_polys[19], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[9], &res[10], &res[11]);

    // Claim 4: Cubic3in col (A=t_inv_col=6, B=t_col=5, C=ts_col=9, eq=eq_mem=19)
    sc_reduce_cubic_3in<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[6], d_polys[5], d_polys[9], d_polys[19], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[12], &res[13], &res[14]);

    // Claim 5: Cubic2in col (A=w_inv_col=8, B=w_col=7, eq=eq_mem=19)
    sc_reduce_cubic_2in<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[8], d_polys[7], d_polys[19], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[15], &res[16], &res[17]);

    // Claim 6: Cubic3in outer (A=Az=10, B=Bz=11, C=uCz_E=12, eq=eq_outer=20)
    sc_reduce_cubic_3in<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[10], d_polys[11], d_polys[12], d_polys[20], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[18], &res[19], &res[20]);

    // Claim 7: Cubic1in outer (A=Mz=13, eq=eq_outer=20)
    sc_reduce_cubic_1in<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[13], d_polys[20], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[21], &res[22], &res[23]);

    // Claim 8: CubicDeg3 inner (A=L_row=14, B=L_col=15, C=val=16)
    sc_reduce_cubic_deg3<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[14], d_polys[15], d_polys[16], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[24], &res[25], &res[26]);

    // Claim 9: Quadratic witness (A=masked_eq=18, B=W=17)
    sc_reduce_quadratic<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_polys[18], d_polys[17], half_n, d_block_results);
    cudaDeviceSynchronize();
    reduce_blocks(&res[27], &res[28], &res[29]);

    return 0;
}

// Bind all 21 polynomials with challenge r, halving their effective size.
// r: pointer to one fr_t scalar (32 bytes)
// half_n: current half-size
// Returns 0 on success.
int gpu_sumcheck_bind(const void* r, uint32_t half_n) {
    CHECK_CUDA(cudaMemcpy(d_challenge, r, sizeof(fr_t), cudaMemcpyHostToDevice));

    sumcheck_bind_kernel<<<g_num_blocks, SC_BLOCK_SIZE>>>(
        d_poly_ptrs, NUM_POLYS, d_challenge, half_n);
    CHECK_CUDA(cudaDeviceSynchronize());

    return 0;
}

// Get the final scalar value of a polynomial (after all rounds, size=1).
// poly_id: polynomial index (0-20)
// result: output fr_t scalar (32 bytes)
int gpu_sumcheck_get_final(uint32_t poly_id, void* result) {
    if (poly_id >= NUM_POLYS) return -1;
    CHECK_CUDA(cudaMemcpy(result, d_polys[poly_id], sizeof(fr_t),
                          cudaMemcpyDeviceToHost));
    return 0;
}

// Read element [idx] of polynomial [poly_id] from GPU.
int gpu_sumcheck_get_element(uint32_t poly_id, uint32_t idx, void* result) {
    if (poly_id >= NUM_POLYS) return -1;
    CHECK_CUDA(cudaMemcpy(result, d_polys[poly_id] + idx, sizeof(fr_t),
                          cudaMemcpyDeviceToHost));
    return 0;
}

// Free all GPU sumcheck resources.
void gpu_sumcheck_free() {
    if (d_poly_data)     { cudaFree(d_poly_data);     d_poly_data = nullptr; }
    if (d_poly_ptrs)     { cudaFree(d_poly_ptrs);     d_poly_ptrs = nullptr; }
    if (d_block_results) { cudaFree(d_block_results); d_block_results = nullptr; }
    if (d_challenge)     { cudaFree(d_challenge);     d_challenge = nullptr; }
    if (h_block_results) { free(h_block_results);     h_block_results = nullptr; }
    g_n = 0;
    g_num_blocks = 0;
}

} // extern "C"

#endif // __CUDA_ARCH__
