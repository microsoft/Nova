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

// Configure CUDA memory pool to retain freed memory, avoiding OS-level
// alloc/free overhead on repeated cudaMallocAsync/cudaFreeAsync cycles.
// This eliminates the alternating fast/slow MSM pattern caused by the
// sppark invoke's internal dev_ptr_t alloc/free of ~192MB per call.
static void init_cuda_pool() {
    static bool done = false;
    if (!done) {
        cudaMemPool_t pool;
        if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
            uint64_t threshold = UINT64_MAX;
            cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        }
        done = true;
    }
}

// Generator cache with label-based invalidation.
// The label is the base pointer of the Rust Vec<Affine> inside CommitmentKey.
// Because every prefix slice &ck[..len] shares the same base pointer, the
// GPU recognises the same generator set regardless of how many elements
// the Rust caller actually needs for a given MSM.
static affine_h* g_gens = nullptr;
static size_t g_gens_n = 0;
static uint64_t g_gens_label = 0;

static int ensure_generators(const void* points, size_t n_bases, uint64_t label) {
    init_cuda_pool();
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

// Cached MSM handle — avoids repeated alloc/sync/free per call.
// The msm_t destructor calls cudaDeviceSynchronize() + cudaFree, which adds
// ~30ms overhead per call. Caching the handle across calls eliminates this.
typedef msm_par_t<bucket_t, point_t, affine_t, scalar_t> msm_handle_t;
static msm_handle_t* g_msm_handle = nullptr;
static size_t g_msm_handle_n = 0;

static msm_handle_t& get_msm_handle(size_t n) {
    if (!g_msm_handle || g_msm_handle_n < n) {
        if (g_msm_handle) {
            g_msm_handle->d_points = nullptr;
            g_msm_handle->npoints = 0;
            g_msm_handle->d_scalars = nullptr;
            delete g_msm_handle;
        }
        g_msm_handle = new msm_handle_t(nullptr, n);
        g_msm_handle_n = n;
    }
    return *g_msm_handle;
}

// Internal: run MSM using cached generators and cached MSM handle.
static int msm_cached(const void* points, const void* scalars,
                      void* result, size_t n_bases, size_t n_scalars,
                      uint64_t label) {
    if (ensure_generators(points, n_bases, label) != 0)
        return -1;

    msm_handle_t& msm = get_msm_handle(n_scalars);
    msm.d_points = g_gens;
    msm.npoints = n_scalars;

    point_t out;
    RustError err = msm.invoke(out, (const affine_t*)nullptr, n_scalars,
                               reinterpret_cast<const scalar_t*>(scalars), true);

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

// MSM using device-side scalars (already on GPU).
// Generators must be cached from a prior sppark_msm_with_generators call
// or sppark_ensure_generators call with sufficient n.
int sppark_msm_from_device(void* d_scalars, void* result, int n) {
    if (!g_gens || g_gens_n < (size_t)n) {
        return -1;
    }

    msm_handle_t& msm = get_msm_handle((size_t)n);
    msm.d_points = g_gens;
    msm.npoints = (size_t)n;
    msm.d_scalars = reinterpret_cast<scalar_t*>(d_scalars);

    point_t out;
    RustError err = msm.invoke(out, (const affine_t*)nullptr, (size_t)n,
                               (const scalar_t*)nullptr, true);

    memcpy(result, &out, sizeof(out));
    return err.code;
}

void sppark_msm_free() {
    if (g_msm_handle) {
        g_msm_handle->d_points = nullptr;
        g_msm_handle->npoints = 0;
        g_msm_handle->d_scalars = nullptr;
        delete g_msm_handle;
        g_msm_handle = nullptr;
        g_msm_handle_n = 0;
    }
    if (g_gens) { cudaFree(g_gens); g_gens = nullptr; g_gens_n = 0; }
}

// Ensure generators are cached with at least n elements.
void sppark_ensure_generators(const void* points, int n) {
    ensure_generators(points, (size_t)n);
}

} // extern "C"

#endif // __CUDA_ARCH__
