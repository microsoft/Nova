// GPU sumcheck kernels for BN254 ppsnark.
//
// Computes 10 sumcheck claims (eval_0, bound_coeff, eval_inf) via parallel
// reduction, and binds all polynomials per round.
//
// Uses sppark's BN254 field types (fr_t) for arithmetic.

#ifndef __SUMCHECK_KERNELS_CUH__
#define __SUMCHECK_KERNELS_CUH__

#include <cstdint>
#include <cooperative_groups.h>

#define SC_BLOCK_SIZE 256

// mont_t() {} leaves val[] uninitialized — must explicitly zero accumulators.
__device__ __forceinline__
fr_t fr_zero() {
    fr_t z;
    memset(&z, 0, sizeof(z));
    return z;
}

// Shared memory for one 3-value block reduction
__shared__ fr_t s_eval0[SC_BLOCK_SIZE];
__shared__ fr_t s_bound[SC_BLOCK_SIZE];
__shared__ fr_t s_inf[SC_BLOCK_SIZE];

// Block-level tree reduction of (eval0, bound, inf) from shared memory.
// Result in s_eval0[0], s_bound[0], s_inf[0].
__device__ __forceinline__
void block_reduce_3(uint32_t tid) {
    __syncthreads();
    for (int s = SC_BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < (uint32_t)s) {
            s_eval0[tid] += s_eval0[tid + s];
            s_bound[tid] += s_bound[tid + s];
            s_inf[tid]   += s_inf[tid + s];
        }
        __syncthreads();
    }
}

// --- LINEAR: Σ (A - B) ---
// eval_0 = Σ (A_lo - B_lo), bound = 0, eval_inf = Σ ((2A_lo-A_hi) - (2B_lo-B_hi))
__global__
void sc_reduce_linear(
    const fr_t* __restrict__ A,
    const fr_t* __restrict__ B,
    uint32_t half_n,
    fr_t* __restrict__ out)  // out[gridDim.x * 3]
{
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t stride = blockDim.x * gridDim.x;

    fr_t acc0 = fr_zero(), acc2 = fr_zero();
    for (uint32_t j = gid; j < half_n; j += stride) {
        fr_t a_lo = A[j], a_hi = A[j + half_n];
        fr_t b_lo = B[j], b_hi = B[j + half_n];
        acc0 += a_lo - b_lo;
        acc2 += (a_lo + a_lo - a_hi) - (b_lo + b_lo - b_hi);
    }

    s_eval0[tid] = acc0;
    s_bound[tid] = fr_zero();
    s_inf[tid]   = acc2;
    block_reduce_3(tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 3;
        out[base + 0] = s_eval0[0];
        out[base + 1] = fr_zero();
        out[base + 2] = s_inf[0];
    }
}

// --- QUADRATIC: Σ (A * B) ---
__global__
void sc_reduce_quadratic(
    const fr_t* __restrict__ A,
    const fr_t* __restrict__ B,
    uint32_t half_n,
    fr_t* __restrict__ out)
{
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t stride = blockDim.x * gridDim.x;

    fr_t acc0 = fr_zero(), acc2 = fr_zero();
    for (uint32_t j = gid; j < half_n; j += stride) {
        fr_t a_lo = A[j], a_hi = A[j + half_n];
        fr_t b_lo = B[j], b_hi = B[j + half_n];
        acc0 += a_lo * b_lo;
        fr_t a_inf = a_lo + a_lo - a_hi;
        fr_t b_inf = b_lo + b_lo - b_hi;
        acc2 += a_inf * b_inf;
    }

    s_eval0[tid] = acc0;
    s_bound[tid] = fr_zero();
    s_inf[tid]   = acc2;
    block_reduce_3(tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 3;
        out[base + 0] = s_eval0[0];
        out[base + 1] = fr_zero();
        out[base + 2] = s_inf[0];
    }
}

// --- CUBIC_1IN: Σ eq * A ---
__global__
void sc_reduce_cubic_1in(
    const fr_t* __restrict__ A,
    const fr_t* __restrict__ EQ,
    uint32_t half_n,
    fr_t* __restrict__ out)
{
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t stride = blockDim.x * gridDim.x;

    fr_t acc0 = fr_zero(), acc2 = fr_zero();
    for (uint32_t j = gid; j < half_n; j += stride) {
        fr_t a_lo = A[j], a_hi = A[j + half_n];
        fr_t e_lo = EQ[j], e_hi = EQ[j + half_n];
        acc0 += e_lo * a_lo;
        acc2 += (e_lo + e_lo - e_hi) * (a_lo + a_lo - a_hi);
    }

    s_eval0[tid] = acc0;
    s_bound[tid] = fr_zero();
    s_inf[tid]   = acc2;
    block_reduce_3(tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 3;
        out[base + 0] = s_eval0[0];
        out[base + 1] = fr_zero();
        out[base + 2] = s_inf[0];
    }
}

// --- CUBIC_2IN: Σ eq * (A*B - 1) ---
__global__
void sc_reduce_cubic_2in(
    const fr_t* __restrict__ A,
    const fr_t* __restrict__ B,
    const fr_t* __restrict__ EQ,
    uint32_t half_n,
    fr_t* __restrict__ out)
{
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t stride = blockDim.x * gridDim.x;

    fr_t acc0 = fr_zero(), acc1 = fr_zero(), acc2 = fr_zero();
    fr_t one = fr_t::one();

    for (uint32_t j = gid; j < half_n; j += stride) {
        fr_t a_lo = A[j], a_hi = A[j + half_n];
        fr_t b_lo = B[j], b_hi = B[j + half_n];
        fr_t e_lo = EQ[j], e_hi = EQ[j + half_n];

        acc0 += e_lo * (a_lo * b_lo - one);
        acc1 += (e_hi - e_lo) * (a_hi - a_lo) * (b_hi - b_lo);
        fr_t a_inf = a_lo + a_lo - a_hi;
        fr_t b_inf = b_lo + b_lo - b_hi;
        acc2 += (e_lo + e_lo - e_hi) * (a_inf * b_inf - one);
    }

    s_eval0[tid] = acc0;
    s_bound[tid] = acc1;
    s_inf[tid]   = acc2;
    block_reduce_3(tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 3;
        out[base + 0] = s_eval0[0];
        out[base + 1] = s_bound[0];
        out[base + 2] = s_inf[0];
    }
}

// --- CUBIC_3IN: Σ eq * (A*B - C) ---
__global__
void sc_reduce_cubic_3in(
    const fr_t* __restrict__ A,
    const fr_t* __restrict__ B,
    const fr_t* __restrict__ C,
    const fr_t* __restrict__ EQ,
    uint32_t half_n,
    fr_t* __restrict__ out)
{
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t stride = blockDim.x * gridDim.x;

    fr_t acc0 = fr_zero(), acc1 = fr_zero(), acc2 = fr_zero();
    for (uint32_t j = gid; j < half_n; j += stride) {
        fr_t a_lo = A[j], a_hi = A[j + half_n];
        fr_t b_lo = B[j], b_hi = B[j + half_n];
        fr_t c_lo = C[j], c_hi = C[j + half_n];
        fr_t e_lo = EQ[j], e_hi = EQ[j + half_n];

        acc0 += e_lo * (a_lo * b_lo - c_lo);
        acc1 += (e_hi - e_lo) * (a_hi - a_lo) * (b_hi - b_lo);
        fr_t a_inf = a_lo + a_lo - a_hi;
        fr_t b_inf = b_lo + b_lo - b_hi;
        fr_t c_inf = c_lo + c_lo - c_hi;
        acc2 += (e_lo + e_lo - e_hi) * (a_inf * b_inf - c_inf);
    }

    s_eval0[tid] = acc0;
    s_bound[tid] = acc1;
    s_inf[tid]   = acc2;
    block_reduce_3(tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 3;
        out[base + 0] = s_eval0[0];
        out[base + 1] = s_bound[0];
        out[base + 2] = s_inf[0];
    }
}

// --- CUBIC_DEG3: Σ A*B*C (no eq) ---
__global__
void sc_reduce_cubic_deg3(
    const fr_t* __restrict__ A,
    const fr_t* __restrict__ B,
    const fr_t* __restrict__ C,
    uint32_t half_n,
    fr_t* __restrict__ out)
{
    uint32_t tid = threadIdx.x;
    uint32_t gid = blockIdx.x * blockDim.x + tid;
    uint32_t stride = blockDim.x * gridDim.x;

    fr_t acc0 = fr_zero(), acc1 = fr_zero(), acc2 = fr_zero();
    for (uint32_t j = gid; j < half_n; j += stride) {
        fr_t a_lo = A[j], a_hi = A[j + half_n];
        fr_t b_lo = B[j], b_hi = B[j + half_n];
        fr_t c_lo = C[j], c_hi = C[j + half_n];

        acc0 += a_lo * b_lo * c_lo;
        acc1 += (a_hi - a_lo) * (b_hi - b_lo) * (c_hi - c_lo);
        fr_t a_inf = a_lo + a_lo - a_hi;
        fr_t b_inf = b_lo + b_lo - b_hi;
        fr_t c_inf = c_lo + c_lo - c_hi;
        acc2 += a_inf * b_inf * c_inf;
    }

    s_eval0[tid] = acc0;
    s_bound[tid] = acc1;
    s_inf[tid]   = acc2;
    block_reduce_3(tid);

    if (tid == 0) {
        uint32_t base = blockIdx.x * 3;
        out[base + 0] = s_eval0[0];
        out[base + 1] = s_bound[0];
        out[base + 2] = s_inf[0];
    }
}

// --- BIND: poly[i] += r * (poly[i+half_n] - poly[i]) for all polynomials ---
__global__
void sumcheck_bind_kernel(
    fr_t** __restrict__ polys,
    uint32_t num_polys,
    const fr_t* __restrict__ d_r,
    uint32_t half_n)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;

    fr_t r = *d_r;

    for (uint32_t j = gid; j < half_n; j += stride) {
        for (uint32_t p = 0; p < num_polys; p++) {
            fr_t lo = polys[p][j];
            fr_t hi = polys[p][j + half_n];
            polys[p][j] = lo + r * (hi - lo);
        }
    }
}

#endif // __SUMCHECK_KERNELS_CUH__
