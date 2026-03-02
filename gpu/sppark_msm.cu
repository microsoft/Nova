// GPU MSM for BN254 using sppark's sort-based Pippenger with parallel accumulate.
//
// Generator caching: generators are uploaded to GPU on first call and reused.
// All MSMs use accumulate_parallel (handles both normal and pathological distributions).
//
// Compile:
//   nvcc -O3 -arch=sm_80 -I $SPPARK/sppark -I $BLST/src -I gpu/ \
//     -DTAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE -DFEATURE_BN254 \
//     -rdc=true --shared -Xcompiler -fPIC \
//     gpu/sppark_msm.cu $SPPARK/sppark/util/all_gpus.cpp \
//     -Xlinker --whole-archive -L/tmp/blst_build -lblst -Xlinker --no-whole-archive \
//     -o gpu/libsppark_msm.so

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

// Generator cache with fingerprint invalidation
static affine_h* g_gens = nullptr;
static size_t g_gens_n = 0;
static uint64_t g_gens_fp = 0;

static void ensure_generators(const void* points, size_t n) {
    const uint64_t* pts = reinterpret_cast<const uint64_t*>(points);
    // Fingerprint: first word XOR last point's first word XOR count
    uint64_t fp = pts[0] ^ pts[(n-1) * (sizeof(affine_h)/sizeof(uint64_t))] ^ n;
    if (g_gens && g_gens_fp == fp) return;

    if (g_gens) cudaFree(g_gens);
    cudaMalloc(&g_gens, n * sizeof(affine_h));
    cudaMemcpy(g_gens, points, n * sizeof(affine_h), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    g_gens_n = n;
    g_gens_fp = fp;
}

// Internal: run MSM using cached generators with msm_par_t (parallel accumulate).
// Creates a fresh handle per call for optimal wbits, but reuses GPU generator data.
static int msm_cached(const void* points, const void* scalars,
                      void* result, size_t n) {
    ensure_generators(points, n);

    // Fresh handle: optimal wbits for this n, no generator upload
    msm_par_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, n};
    msm.d_points = g_gens;
    msm.npoints = n;

    point_t out;
    RustError err = msm.invoke(out, (const affine_t*)nullptr, n,
                               reinterpret_cast<const scalar_t*>(scalars), true);

    // Prevent destructor from freeing cached generators
    msm.d_points = nullptr;
    msm.npoints = 0;

    memcpy(result, &out, sizeof(out));
    return err.code;
}

extern "C" {

int sppark_msm_with_generators(const void* points, const void* scalars,
                                void* result, int n) {
    return msm_cached(points, scalars, result, (size_t)n);
}

int sppark_msm_parallel(const void* points, const void* scalars,
                         void* result, int n) {
    return msm_cached(points, scalars, result, (size_t)n);
}

void sppark_msm_free() {
    if (g_gens) { cudaFree(g_gens); g_gens = nullptr; g_gens_n = 0; }
}

} // extern "C"

#endif // __CUDA_ARCH__
