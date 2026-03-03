// Parallel-accumulate MSM: fork of sppark's pippenger with parallel tree
// reduction for large buckets. When pathological scalar distributions cause
// >4096 points in one bucket, the original sequential accumulate takes ~7s.
// This version splits large buckets across all threads in a block via
// shared-memory tree reduction, bringing that down to ~3ms.
//
// Uses sppark's sort, breakdown, batch_addition, and integrate unchanged.
// Only accumulate is replaced.

#ifndef __MSM_PARALLEL_CUH__
#define __MSM_PARALLEL_CUH__

#include <cooperative_groups.h>

#ifndef LARGE_BUCKET_THRESHOLD
# define LARGE_BUCKET_THRESHOLD 4096
#endif
#define MAX_LARGE_BUCKETS 512
#define PARALLEL_REDUCE_SIZE 256  // Must be power-of-2, <= ACCUMULATE_NTHREADS

// Overflow threads fold into shared_buckets[0..PARALLEL_REDUCE_SIZE).
// target = threadIdx.x - PARALLEL_REDUCE_SIZE must be < PARALLEL_REDUCE_SIZE,
// i.e. ACCUMULATE_NTHREADS <= 2 * PARALLEL_REDUCE_SIZE.
static_assert(ACCUMULATE_NTHREADS <= 2 * PARALLEL_REDUCE_SIZE,
              "ACCUMULATE_NTHREADS must be <= 2 * PARALLEL_REDUCE_SIZE");

struct large_bucket_info_t {
    uint32_t win;
    uint32_t bucket;
    uint32_t start;
    uint32_t len;
};

static __device__ uint32_t d_n_large_buckets;
static __device__ large_bucket_info_t d_large_bucket_list[MAX_LARGE_BUCKETS];

// ========== Modified accumulate kernel with parallel large-bucket reduction ==========

template<class bucket_t,
         class affine_h,
         class bucket_h = typename bucket_t::mem_t,
         class affine_t = typename bucket_t::affine_t>
__launch_bounds__(ACCUMULATE_NTHREADS) __global__
void accumulate_parallel(bucket_h buckets_[], uint32_t nwins, uint32_t wbits,
                         /*const*/ affine_h points_[], const vec2d_t<uint32_t> digits,
                         const vec2d_t<uint32_t> histogram, uint32_t sid = 0)
{
    vec2d_t<bucket_h> buckets{buckets_, 1U<<--wbits};
    const affine_h* points = points_;

    static __device__ uint32_t streams_par[MSM_NSTREAMS];
    uint32_t& current = streams_par[sid % MSM_NSTREAMS];
    uint32_t laneid;
    asm("mov.u32 %0, %laneid;" : "=r"(laneid));
    const uint32_t degree = bucket_t::degree;
    const uint32_t warp_sz = WARP_SZ / degree;
    const uint32_t lane_id = laneid / degree;

    if (threadIdx.x == 0 && blockIdx.x == 0)
        d_n_large_buckets = 0;
    // Ensure init visible to all blocks before Phase 1
    cooperative_groups::this_grid().sync();

    uint32_t x, y;
    __shared__ uint32_t xchg;

    if (threadIdx.x == 0)
        xchg = atomicAdd(&current, blockDim.x/degree);
    __syncthreads();
    x = xchg + threadIdx.x/degree;

    // ===== Phase 1: process small buckets, defer large ones =====
    while (x < (nwins << wbits)) {
        y = x >> wbits;
        x &= (1U << wbits) - 1;
        const uint32_t* h = &histogram[y][x];

        uint32_t idx, len = h[0];

        asm("{ .reg.pred %did;"
            "  shfl.sync.up.b32 %0|%did, %1, %2, 0, 0xffffffff;"
            "  @!%did mov.b32 %0, 0;"
            "}" : "=r"(idx) : "r"(len), "r"(degree));

        if (lane_id == 0 && x != 0)
            idx = h[-1];

        if ((len -= idx) && !(x == 0 && y == 0)) {
            if (len > LARGE_BUCKET_THRESHOLD) {
                // Defer to Phase 2 for parallel processing
                uint32_t slot = atomicAdd(&d_n_large_buckets, 1);
                if (slot < MAX_LARGE_BUCKETS) {
                    d_large_bucket_list[slot].win    = y;
                    d_large_bucket_list[slot].bucket = x;
                    d_large_bucket_list[slot].start  = idx;
                    d_large_bucket_list[slot].len    = len;
                    buckets[y][x].inf(); // Placeholder; Phase 2 overwrites
                } else {
                    goto sequential;
                }
            } else {
            sequential:;
                const uint32_t* digs_ptr = &digits[y][idx];
                uint32_t digit = *digs_ptr++;

                affine_t p = points[digit & 0x7fffffff];
                bucket_t bucket = p;
                bucket.cneg(digit >> 31);

                while (--len) {
                    digit = *digs_ptr++;
                    p = points[digit & 0x7fffffff];
                    if (sizeof(bucket) <= 128 || LARGE_L1_CODE_CACHE)
                        bucket.add(p, digit >> 31);
                    else
                        bucket.uadd(p, digit >> 31);
                }

                buckets[y][x] = bucket;
            }
        } else {
            buckets[y][x].inf();
        }

        x = laneid == 0 ? atomicAdd(&current, warp_sz) : 0;
        x = __shfl_sync(0xffffffff, x, 0) + lane_id;
    }

    // Phase 1 done — all large bucket info is written
    cooperative_groups::this_grid().sync();

    // ===== Phase 2: parallel tree reduction for large buckets =====
    uint32_t n_large = d_n_large_buckets;
    extern __shared__ bucket_h shared_buckets[];

    for (uint32_t lb = blockIdx.x; lb < n_large; lb += gridDim.x) {
        uint32_t lb_y     = d_large_bucket_list[lb].win;
        uint32_t lb_x     = d_large_bucket_list[lb].bucket;
        uint32_t lb_start = d_large_bucket_list[lb].start;
        uint32_t lb_len   = d_large_bucket_list[lb].len;

        const uint32_t* digs_ptr = &digits[lb_y][lb_start];

        // Each thread accumulates its stride through the bucket
        bucket_t partial;
        partial.inf();

        for (uint32_t i = threadIdx.x; i < lb_len; i += blockDim.x) {
            uint32_t digit = digs_ptr[i];
            affine_t p = points[digit & 0x7fffffff];
            if (sizeof(bucket_t) <= 128 || LARGE_L1_CODE_CACHE)
                partial.add(p, digit >> 31);
            else
                partial.uadd(p, digit >> 31);
        }

        // Reduce blockDim.x partial sums into PARALLEL_REDUCE_SIZE shared entries.
        // Step 1: first PARALLEL_REDUCE_SIZE threads store directly
        if (threadIdx.x < PARALLEL_REDUCE_SIZE)
            shared_buckets[threadIdx.x] = partial;
        __syncthreads();

        // Step 2: overflow threads (>= PARALLEL_REDUCE_SIZE) fold into [0..overflow)
        if (threadIdx.x >= PARALLEL_REDUCE_SIZE && threadIdx.x < blockDim.x) {
            uint32_t target = threadIdx.x - PARALLEL_REDUCE_SIZE;
            bucket_t existing = shared_buckets[target];
            existing.uadd(partial);
            shared_buckets[target] = existing;
        }
        __syncthreads();

        // Step 3: standard power-of-2 tree reduction on PARALLEL_REDUCE_SIZE entries
        for (uint32_t s = PARALLEL_REDUCE_SIZE / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                bucket_t a = shared_buckets[threadIdx.x];
                bucket_t b = shared_buckets[threadIdx.x + s];
                a.uadd(b);
                shared_buckets[threadIdx.x] = a;
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
            buckets[lb_y][lb_x] = shared_buckets[0];

        __syncthreads();
    }

    // Phase 2 done
    cooperative_groups::this_grid().sync();

    if (threadIdx.x + blockIdx.x == 0) {
        current = 0;
        d_n_large_buckets = 0;
    }
}

// ========== msm_par_t: copy of sppark's msm_t with public members ==========
// Uses accumulate_parallel instead of accumulate in invoke().

template<class bucket_t, class point_t, class affine_t, class scalar_t,
         class affine_h = typename affine_t::mem_t,
         class bucket_h = typename bucket_t::mem_t>
struct msm_par_t {
    const gpu_t& gpu;
    size_t npoints;
    uint32_t wbits, nwins;
    bucket_h *d_buckets;
    affine_h *d_points;
    scalar_t *d_scalars;
    vec2d_t<uint32_t> d_hist;

    template<typename T> using vec_t = slice_t<T>;

    class result_t {
        bucket_t ret[MSM_NTHREADS/bucket_t::degree][2];
    public:
        result_t() {}
        inline operator decltype(ret)&()                    { return ret;    }
        inline const bucket_t* operator[](size_t i) const   { return ret[i]; }
    };

    constexpr static int lg2(size_t n)
    {   int ret=0; while (n>>=1) ret++; return ret;   }

    msm_par_t(const affine_t points[], size_t np,
              size_t ffi_affine_sz = sizeof(affine_t), int device_id = -1)
        : gpu(select_gpu(device_id)), d_points(nullptr), d_scalars(nullptr)
    {
        npoints = (np+WARP_SZ-1) & ((size_t)0-WARP_SZ);
        wbits = 17;
        if (npoints > 192) {
            wbits = std::min(lg2(npoints + npoints/2) - 8, 18);
            if (wbits < 10) wbits = 10;
        } else if (npoints > 0) {
            wbits = 10;
        }
        nwins = (scalar_t::bit_length() - 1) / wbits + 1;

        uint32_t row_sz = 1U << (wbits-1);
        size_t d_buckets_sz = (nwins * row_sz)
                            + (gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);
        size_t d_blob_sz = (d_buckets_sz * sizeof(d_buckets[0]))
                         + (nwins * row_sz * sizeof(uint32_t))
                         + (points ? npoints * sizeof(d_points[0]) : 0);

        d_buckets = reinterpret_cast<decltype(d_buckets)>(gpu.Dmalloc(d_blob_sz));
        d_hist = vec2d_t<uint32_t>(&d_buckets[d_buckets_sz], row_sz);
        if (points) {
            d_points = reinterpret_cast<decltype(d_points)>(d_hist[nwins]);
            gpu.HtoD(d_points, points, np, ffi_affine_sz);
            npoints = np;
        } else {
            npoints = 0;
        }
    }
    inline msm_par_t(vec_t<affine_t> points, size_t ffi_affine_sz = sizeof(affine_t),
                     int device_id = -1)
        : msm_par_t(points, points.size(), ffi_affine_sz, device_id) {};
    inline msm_par_t(int device_id = -1)
        : msm_par_t(nullptr, 0, 0, device_id) {};
    ~msm_par_t()
    {
        gpu.sync();
        if (d_buckets) gpu.Dfree(d_buckets);
    }

    void digits(const scalar_t d_scalars[], size_t len,
                vec2d_t<uint32_t>& d_digits, vec2d_t<uint2>& d_temps, bool mont)
    {
        uint32_t grid_size = gpu.sm_count() / 3;
        while (grid_size & (grid_size - 1))
            grid_size -= (grid_size & (0 - grid_size));

        breakdown<<<2*grid_size, 1024, sizeof(scalar_t)*1024, gpu[2]>>>(
            d_digits, d_scalars, len, nwins, wbits, mont
        );
        CUDA_OK(cudaGetLastError());

        const size_t shared_sz = sizeof(uint32_t) << DIGIT_BITS;
        uint32_t top = scalar_t::bit_length() - wbits * (nwins-1);
        uint32_t win;
        for (win = 0; win < nwins-1; win += 2) {
            gpu[2].launch_coop(sort, {{grid_size, 2}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, wbits-1, win == nwins-2 ? top-1 : wbits-1);
        }
        if (win < nwins) {
            gpu[2].launch_coop(sort, {{grid_size, 1}, SORT_BLOCKDIM, shared_sz},
                            d_digits, len, win, d_temps, d_hist,
                            wbits-1, top-1, 0u);
        }
    }

    // invoke() — identical to sppark's except uses accumulate_parallel
    RustError invoke(point_t& out, const affine_t* points_, size_t npoints,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {
        assert(this->npoints == 0 || npoints <= this->npoints);

        uint32_t lg_npoints = lg2(npoints + npoints/2);
        size_t batch = 1 << (std::max(lg_npoints, wbits) - wbits);
        batch >>= 6;
        batch = batch ? batch : 1;
        uint32_t stride = (npoints + batch - 1) / batch;
        stride = (stride+WARP_SZ-1) & ((size_t)0-WARP_SZ);

        std::vector<result_t> res(nwins);
        std::vector<bucket_t> ones(gpu.sm_count() * BATCH_ADD_BLOCK_SIZE / WARP_SZ);

        out.inf();
        point_t p;

        try {
            size_t temp_sz = scalars ? sizeof(scalar_t) : 0;
            temp_sz = stride * std::max(2*sizeof(uint2), temp_sz);

            const char* points = reinterpret_cast<const char*>(points_);
            size_t d_point_sz = points ? (batch > 1 ? 2*stride : stride) : 0;
            d_point_sz *= sizeof(affine_h);

            size_t digits_sz = nwins * stride * sizeof(uint32_t);

            dev_ptr_t<uint8_t> d_temp{temp_sz + digits_sz + d_point_sz, gpu[2]};

            vec2d_t<uint2> d_temps{&d_temp[0], stride};
            vec2d_t<uint32_t> d_digits{&d_temp[temp_sz], stride};

            scalar_t* d_scalars = scalars ? (scalar_t*)&d_temp[0]
                                          : this->d_scalars;
            affine_h* d_points = points ? (affine_h*)&d_temp[temp_sz + digits_sz]
                                        : this->d_points;

            size_t d_off = 0;
            size_t h_off = 0;
            size_t num = stride > npoints ? npoints : stride;
            event_t ev;

            if (scalars)
                gpu[2].HtoD(&d_scalars[d_off], &scalars[h_off], num);
            digits(&d_scalars[0], num, d_digits, d_temps, mont);
            gpu[2].record(ev);

            if (points)
                gpu[0].HtoD(&d_points[d_off], &points[h_off],
                            num,              ffi_affine_sz);

            for (uint32_t i = 0; i < batch; i++) {
                gpu[i&1].wait(ev);

                batch_addition<bucket_t><<<gpu.sm_count(), BATCH_ADD_BLOCK_SIZE,
                                           0, gpu[i&1]>>>(
                    &d_buckets[nwins << (wbits-1)], &d_points[d_off], num,
                    &d_digits[0][0], d_hist[0][0]
                );
                CUDA_OK(cudaGetLastError());

                // THE KEY CHANGE: accumulate_parallel + shared memory for Phase 2
                size_t accum_shared = sizeof(bucket_h) * PARALLEL_REDUCE_SIZE;
                gpu[i&1].launch_coop(accumulate_parallel<bucket_t, affine_h>,
                    {gpu.sm_count(), 0, accum_shared},
                    d_buckets, nwins, wbits, &d_points[d_off], d_digits, d_hist, i&1
                );
                gpu[i&1].record(ev);

                integrate<bucket_t><<<nwins, MSM_NTHREADS,
                                      sizeof(bucket_t)*MSM_NTHREADS/bucket_t::degree,
                                      gpu[i&1]>>>(
                    d_buckets, nwins, wbits, scalar_t::bit_length()
                );
                CUDA_OK(cudaGetLastError());

                if (i < batch-1) {
                    h_off += stride;
                    num = h_off + stride <= npoints ? stride : npoints - h_off;
                    if (scalars)
                        gpu[2].HtoD(&d_scalars[0], &scalars[h_off], num);
                    gpu[2].wait(ev);
                    digits(&d_scalars[scalars ? 0 : h_off], num,
                           d_digits, d_temps, mont);
                    gpu[2].record(ev);

                    if (points) {
                        size_t j = (i + 1) & 1;
                        d_off = j ? stride : 0;
                        gpu[j].HtoD(&d_points[d_off], &points[h_off*ffi_affine_sz],
                                    num,              ffi_affine_sz);
                    } else {
                        d_off = h_off;
                    }
                }

                if (i > 0) {
                    collect(p, res, ones);
                    out.add(p);
                }

                gpu[i&1].DtoH(ones, d_buckets + (nwins << (wbits-1)));
                gpu[i&1].DtoH(res, d_buckets, sizeof(bucket_h)<<(wbits-1));
                gpu[i&1].sync();
            }
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        collect(p, res, ones);
        out.add(p);

        return RustError{cudaSuccess};
    }

    // Overload for slice_t points
    RustError invoke(point_t& out, vec_t<affine_t> points,
                                   const scalar_t* scalars, bool mont = true,
                                   size_t ffi_affine_sz = sizeof(affine_t))
    {   return invoke(out, &points[0], points.size(), scalars, mont, ffi_affine_sz);   }

private:
    point_t integrate_row(const result_t& row, uint32_t lsbits)
    {
        const int NTHRBITS = lg2(MSM_NTHREADS/bucket_t::degree);
        assert(wbits-1 > NTHRBITS);

        size_t i = MSM_NTHREADS/bucket_t::degree - 1;

        if (lsbits-1 <= NTHRBITS) {
            size_t mask = (1U << (NTHRBITS-(lsbits-1))) - 1;
            bucket_t res, acc = row[i][1];

            if (mask) res.inf();
            else      res = acc;

            while (i--) {
                acc.add(row[i][1]);
                if ((i & mask) == 0)
                    res.add(acc);
            }
            return res;
        }

        point_t  res = row[i][0];
        bucket_t acc = row[i][1];

        while (i--) {
            point_t raise = acc;
            for (size_t j = 0; j < lsbits-1-NTHRBITS; j++)
                raise.dbl();
            res.add(raise);
            res.add(point_t{row[i][0]});
            if (i)
                acc.add(row[i][1]);
        }
        return res;
    }

    void collect(point_t& out, const std::vector<result_t>& res,
                               const std::vector<bucket_t>& ones)
    {
        struct tile_t {
            uint32_t x, y, dy;
            point_t p;
            tile_t() {}
        };
        std::vector<tile_t> grid(nwins);

        uint32_t y = nwins-1, total = 0;

        grid[0].x  = 0;
        grid[0].y  = y;
        grid[0].dy = scalar_t::bit_length() - y*wbits;
        total++;

        while (y--) {
            grid[total].x  = grid[0].x;
            grid[total].y  = y;
            grid[total].dy = wbits;
            total++;
        }

        std::vector<std::atomic<size_t>> row_sync(nwins);
        counter_t<size_t> counter(0);
        channel_t<size_t> ch;

        auto n_workers = min((uint32_t)gpu.ncpus(), total);
        while (n_workers--) {
            gpu.spawn([&, this, total, counter]() {
                for (size_t work; (work = counter++) < total;) {
                    auto item = &grid[work];
                    auto y = item->y;
                    item->p = integrate_row(res[y], item->dy);
                    if (++row_sync[y] == 1)
                        ch.send(y);
                }
            });
        }

        point_t one = sum_up(ones);

        out.inf();
        size_t row = 0, ny = nwins;
        while (ny--) {
            auto y = ch.recv();
            row_sync[y] = -1U;
            while (grid[row].y == y) {
                while (row < total && grid[row].y == y)
                    out.add(grid[row++].p);
                if (y == 0) break;
                for (size_t i = 0; i < wbits; i++)
                    out.dbl();
                if (row_sync[--y] != -1U)
                    break;
            }
        }
        out.add(one);
    }
};

// ========== Top-level wrapper ==========

template<class bucket_t, class point_t, class affine_t, class scalar_t> static
RustError mult_pippenger_par(point_t *out, const affine_t points[], size_t npoints,
                                           const scalar_t scalars[], bool mont = true,
                                           size_t ffi_affine_sz = sizeof(affine_t))
{
    try {
        msm_par_t<bucket_t, point_t, affine_t, scalar_t> msm{nullptr, npoints};
        return msm.invoke(*out, slice_t<affine_t>{points, npoints},
                                scalars, mont, ffi_affine_sz);
    } catch (const cuda_error& e) {
        out->inf();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
        return RustError{e.code(), e.what()};
#else
        return RustError{e.code()};
#endif
    }
}

// Explicit template instantiation for accumulate_parallel (BN254)
template __global__
void accumulate_parallel<bucket_t, affine_t::mem_t>(bucket_t::mem_t buckets_[],
                                                     uint32_t nwins, uint32_t wbits,
                                                     /*const*/ affine_t::mem_t points_[],
                                                     const vec2d_t<uint32_t> digits,
                                                     const vec2d_t<uint32_t> histogram,
                                                     uint32_t sid);

#endif // __MSM_PARALLEL_CUH__
