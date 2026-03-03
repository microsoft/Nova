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

// ============== Polynomial construction kernels ==============

// Gather: out[i] = data[indices[i]]
// Used for L_row[i] = eq_tau[row[i]], L_col[i] = z_padded[col[i]]
__global__
void gather_kernel(
    const fr_t* __restrict__ data,
    const uint32_t* __restrict__ indices,
    fr_t* __restrict__ out,
    uint32_t n)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = gid; i < n; i += stride) {
        out[i] = data[indices[i]];
    }
}

// Compute val = val_A + c*val_B + c²*val_C
__global__
void compute_val_kernel(
    const fr_t* __restrict__ val_A,
    const fr_t* __restrict__ val_B,
    const fr_t* __restrict__ val_C,
    const fr_t* __restrict__ d_c,
    fr_t* __restrict__ out,
    uint32_t n)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    fr_t c = d_c[0];
    fr_t c2 = c * c;
    for (uint32_t i = gid; i < n; i += stride) {
        out[i] = val_A[i] + c * val_B[i] + c2 * val_C[i];
    }
}

// Convert integer to Montgomery form on device
__device__ __forceinline__
fr_t fr_from_uint(uint32_t val) {
    fr_t r = fr_zero();
    // Access underlying storage via pointer cast (operator[] is private)
    uint32_t* limbs = reinterpret_cast<uint32_t*>(&r);
    limbs[0] = val;
    r.to();      // convert to Montgomery form
    return r;
}

// Memory hash: T[i] = mem[i]*gamma + i, W[i] = lookup[i]*gamma + addr[i]
// Both outputs written to T_out and W_out
__global__
void mem_hash_kernel(
    const fr_t* __restrict__ mem,
    const fr_t* __restrict__ addr,
    const fr_t* __restrict__ lookup,
    const fr_t* __restrict__ d_gamma,
    fr_t* __restrict__ T_out,
    fr_t* __restrict__ W_out,
    uint32_t n)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    fr_t gamma = d_gamma[0];
    for (uint32_t i = gid; i < n; i += stride) {
        // T[i] = mem[i] * gamma + i
        T_out[i] = mem[i] * gamma + fr_from_uint(i);
        // W[i] = lookup[i] * gamma + addr[i]
        W_out[i] = lookup[i] * gamma + addr[i];
    }
}

// Add scalar r to every element: out[i] = in[i] + r
__global__
void add_scalar_kernel(
    fr_t* __restrict__ data,
    const fr_t* __restrict__ d_r,
    uint32_t n)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    fr_t r = d_r[0];
    for (uint32_t i = gid; i < n; i += stride) {
        data[i] = data[i] + r;
    }
}

// Element-wise multiply: out[i] = a[i] * b[i]
__global__
void elemwise_mul_kernel(
    const fr_t* __restrict__ a,
    const fr_t* __restrict__ b,
    fr_t* __restrict__ out,
    uint32_t n)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    for (uint32_t i = gid; i < n; i += stride) {
        out[i] = a[i] * b[i];
    }
}

// ============== Batch inversion (Montgomery's trick) ==============
// Phase 1: prefix product. products[i] = a[0]*a[1]*...*a[i]
__global__
void batch_invert_prefix_kernel(
    const fr_t* __restrict__ a,
    fr_t* __restrict__ products,
    uint32_t n,
    uint32_t chunk_size)
{
    uint32_t chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = chunk_id * chunk_size;
    if (start >= n) return;
    uint32_t end = min(start + chunk_size, n);

    fr_t acc = a[start];
    products[start] = acc;
    for (uint32_t i = start + 1; i < end; i++) {
        acc = acc * a[i];
        products[i] = acc;
    }
}

// Phase 3: back-propagate inverses.
// Given products[] and inv of product suffix, compute inverses.
__global__
void batch_invert_back_kernel(
    const fr_t* __restrict__ a,
    const fr_t* __restrict__ products,
    const fr_t* __restrict__ chunk_invs,
    fr_t* __restrict__ out,
    uint32_t n,
    uint32_t chunk_size)
{
    uint32_t chunk_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t start = chunk_id * chunk_size;
    if (start >= n) return;
    uint32_t end = min(start + chunk_size, n);

    fr_t inv = chunk_invs[chunk_id];
    // Back-propagate within chunk
    for (uint32_t i = end; i > start + 1; ) {
        --i;
        out[i] = inv * products[i - 1];
        inv = inv * a[i];
    }
    out[start] = inv;
}

// Compute eq(tau, x) for all x in {0,1}^n.
// eq(tau, x) = prod_{i} (tau_i * x_i + (1-tau_i)*(1-x_i))
// Built iteratively: start with [1], each round doubles the size.
__global__
void eq_expand_kernel(
    fr_t* __restrict__ eq,
    const fr_t* __restrict__ d_tau_i,
    uint32_t prev_size)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t stride = blockDim.x * gridDim.x;
    fr_t tau_i = d_tau_i[0];
    fr_t one_minus_tau = fr_t::one() - tau_i;
    for (uint32_t j = gid; j < prev_size; j += stride) {
        fr_t val = eq[j];
        eq[j] = val * one_minus_tau;
        eq[j + prev_size] = val * tau_i;
    }
}

// Gather chunk-end values for batch inversion
__global__
void gather_chunk_ends_kernel(
    const fr_t* __restrict__ products,
    fr_t* __restrict__ ends,
    uint32_t num_chunks,
    uint32_t chunk_size,
    uint32_t total_n)
{
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_chunks) return;
    uint32_t idx = (gid+1)*chunk_size - 1;
    if (idx >= total_n) idx = total_n - 1;
    ends[gid] = products[idx];
}

#endif // __SUMCHECK_KERNELS_CUH__
