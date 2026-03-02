// Thin wrapper around sppark's msm_t for BN254 with persistent generator caching.
// Exposes C API: sppark_msm_set_generators, sppark_msm, sppark_msm_free
//
// Compile:
//   nvcc -O3 -arch=sm_80 -I $SPPARK/sppark -I $BLST/src \
//     -DTAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE -DFEATURE_BN254 \
//     -rdc=true --shared -Xcompiler -fPIC \
//     gpu/sppark_msm.cu $SPPARK/sppark/util/all_gpus.cpp \
//     -Xlinker --whole-archive -L/tmp/blst_build -lblst -Xlinker --no-whole-archive \
//     -o gpu/libsppark_msm.so

#include <cuda.h>

#include <ec/jacobian_t.hpp>
#include <ec/xyzz_t.hpp>
#include <ff/alt_bn128.hpp>

// These typedefs must come BEFORE pippenger.cuh (it uses them for template instantiation)
typedef jacobian_t<fp_t> point_t;
typedef xyzz_t<fp_t> bucket_t;
typedef bucket_t::affine_t affine_t;
typedef fr_t scalar_t;

#include <msm/pippenger.cuh>

#ifndef __CUDA_ARCH__

// Persistent MSM handle — holds pre-uploaded generators on GPU
static msm_t<bucket_t, point_t, affine_t, scalar_t>* g_msm = nullptr;
static size_t g_npoints = 0;

extern "C" {

// Upload generators (affine points in Montgomery form) to GPU once.
void sppark_msm_set_generators(const void* points, int n) {
    if (g_msm) {
        cudaDeviceSynchronize();
        delete g_msm;
        g_msm = nullptr;
        g_npoints = 0;
        cudaDeviceSynchronize();
    }
    if (n > 0 && points) {
        g_msm = new msm_t<bucket_t, point_t, affine_t, scalar_t>(
            reinterpret_cast<const affine_t*>(points), (size_t)n
        );
        g_npoints = (size_t)n;
    }
}

// Perform MSM using cached generators and provided scalars.
// Returns 0 on success, non-zero on error.
int sppark_msm(const void* scalars, void* result, int n) {
    if (n <= 0) return -1;

    point_t out;
    RustError err;

    if (g_msm && (size_t)n <= g_npoints) {
        // Use cached generators via invoke()
        try {
            err = g_msm->invoke(
                out,
                (const affine_t*)nullptr, (size_t)n,
                reinterpret_cast<const scalar_t*>(scalars),
                true
            );
        } catch (...) {
            memset(result, 0, sizeof(point_t));
            return -2;
        }
    } else {
        memset(result, 0, sizeof(point_t));
        return -3;
    }

    memcpy(result, &out, sizeof(out));
    return err.code;
}

// Perform MSM with explicit generators (no caching, fresh allocation each call).
int sppark_msm_with_generators(const void* points, const void* scalars,
                                void* result, int n) {
    point_t out;
    RustError err = mult_pippenger<bucket_t>(
        &out,
        reinterpret_cast<const affine_t*>(points), (size_t)n,
        reinterpret_cast<const scalar_t*>(scalars),
        true
    );
    memcpy(result, &out, sizeof(out));
    return err.code;
}

// Free the cached MSM handle and GPU resources.
void sppark_msm_free() {
    if (g_msm) {
        cudaDeviceSynchronize();
        delete g_msm;
        g_msm = nullptr;
        g_npoints = 0;
    }
}

} // extern "C"

#endif // __CUDA_ARCH__
