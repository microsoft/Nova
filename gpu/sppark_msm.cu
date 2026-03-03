// GPU MSM for BN254 using sppark's sort-based Pippenger with parallel accumulate.
//
// Generator caching: generators are uploaded to GPU on first call and reused.
// All MSMs use accumulate_parallel (handles both normal and pathological distributions).
//
// Built automatically via build.rs when the `sppark` feature is enabled.

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ff/alt_bn128.hpp>

typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;
typedef affine_t::mem_t affine_h;

#include <msm/pippenger.cuh>
#include "msm_parallel.cuh"

#ifndef __CUDA_ARCH__

// Generator cache with label-based invalidation.
// The label is the base pointer of the Rust Vec<Affine> inside CommitmentKey.
// Because every prefix slice &ck[..len] shares the same base pointer, the
// GPU recognises the same generator set regardless of how many elements
// the Rust caller actually needs for a given MSM.
static affine_h* g_gens = nullptr;
static size_t g_gens_n = 0;
static uint64_t g_gens_label = 0;

static int ensure_generators(const void* points, size_t n_bases, uint64_t label) {
    if (n_bases == 0) {
        g_gens_n = 0;
        return 0;
    }
    // Cache hit: same generator set and we already have enough points.
    if (g_gens && g_gens_label == label && n_bases <= g_gens_n) return 0;

    if (g_gens) cudaFree(g_gens);
    g_gens = nullptr;
    g_gens_label = 0;

    cudaError_t err = cudaMalloc(&g_gens, n_bases * sizeof(affine_h));
    if (err != cudaSuccess) { g_gens_n = 0; return -1; }

    err = cudaMemcpy(g_gens, points, n_bases * sizeof(affine_h), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(g_gens); g_gens = nullptr; g_gens_n = 0; return -1; }

    cudaDeviceSynchronize();
    g_gens_n = n_bases;
    g_gens_label = label;
    return 0;
}

// Internal: run MSM using cached generators with msm_par_t (parallel accumulate).
// Creates a fresh handle per call for optimal wbits, but reuses GPU generator data.
// n_bases:  how many generators the caller provides (for upload).
// n_scalars: how many scalars to use for this MSM (for computation, ≤ n_bases).
static int msm_cached(const void* points, const void* scalars,
                      void* result, size_t n_bases, size_t n_scalars,
                      uint64_t label) {
    if (ensure_generators(points, n_bases, label) != 0)
        return -1;

    // Fresh handle: optimal wbits for this n_scalars, no generator upload
    msm_par_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, n_scalars};
    msm.d_points = g_gens;
    msm.npoints = n_scalars;

    point_t out;
    RustError err = msm.invoke(out, (const affine_t*)nullptr, n_scalars,
                               reinterpret_cast<const scalar_t*>(scalars), true);

    // Prevent destructor from freeing cached generators
    msm.d_points = nullptr;
    msm.npoints = 0;

    memcpy(result, &out, sizeof(out));
    return err.code;
}

extern "C" {

int sppark_msm_with_generators(const void* points, const void* scalars,
                                void* result, uint32_t n_bases,
                                uint32_t n_scalars, uint64_t label) {
    return msm_cached(points, scalars, result, (size_t)n_bases,
                      (size_t)n_scalars, label);
}

void sppark_msm_free() {
    if (g_gens) { cudaFree(g_gens); g_gens = nullptr; g_gens_n = 0; }
}

} // extern "C"

#endif // __CUDA_ARCH__
