// fork: top-k over many rows with the radix select of top-k.cu (written for HIP) on CUDA too, included by top-k.cu.
// with CCCL >= 3.4.3 upstream runs cub DeviceTopK once per row: the qwen4exp block top-k of a prompt ubatch (k 513 over
// the KV blocks, a row per token) took 96 ms per call for 2596 rows on the 3090 (CUDA 13.4, cuda_op_timing 11.10),
// 1.15 s of a 2.6K prompt; the radix select does all rows in 10 launches. GGML_CUDA_TOPK_RADIX: 0 = upstream, 1 = the
// radix select for every top-k, default from 8 rows (decode and draft checks keep the upstream path). since 71ad0590f
// (master 16.10) upstream also takes the radix select above 2 rows, except rows up to 512 columns (and up to 1024 while
// the rows fit one wave of SMs), which go to the bitonic sort; the default here still takes radix for those

#include <cstdlib>

#if defined(GGML_CUDA_USE_CUB)
static bool ggml_cuda_fork_topk_radix(int64_t nrows) {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_TOPK_RADIX");
        return env ? atoi(env) : -1;
    }();
    return mode == 1 || (mode != 0 && nrows >= 8);
}
#endif // defined(GGML_CUDA_USE_CUB)

// fork: the radix select with an output that does not depend on the order the blocks run in, for every caller of
// top_k_radix_cuda. the gather of upstream takes a slot per element with atomicAdd on one counter of the row, shared by
// up to 64 blocks: the order of the indices and, at ties with the k-th value, which of the tied elements get in change
// from run to run. on the 3090 two equal runs of the probes parted at the first probe after a ubatch with more than
// 1024 KV blocks (16256 of 32768, 01.11 and 02.11), as on the 395 before the stable variant of topk_radix_select.comp
// (01.11). here the elements above the threshold come in ascending index order, then the ties from the lowest index, as
// there: each block reads a contiguous range of columns, and a block writes after the elements of the blocks before it.
// the passes already count what that needs, without another read of the input: an element above the threshold leaves
// the prefix in the first pass where its digit is above the chosen one, so the bins above the chosen digit, summed over
// the passes, count the elements of a block above the threshold, and the chosen bin of the last pass counts its ties.
// GGML_CUDA_TOPK_RADIX_STABLE: 1 = this (default), 0 = the gather of upstream. warps of 32 lanes: CUDA only
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

// the sum of v over the block, for every thread; warp_sums holds a value per warp
template<int BLOCK_SIZE>
static __device__ __forceinline__ int top_k_stable_block_sum(int v, int * warp_sums) {
    v = warp_reduce_sum(v);
    if (threadIdx.x % WARP_SIZE == 0) {
        warp_sums[threadIdx.x / WARP_SIZE] = v;
    }
    __syncthreads();
    int sum = 0;
#pragma unroll
    for (int w = 0; w < BLOCK_SIZE / WARP_SIZE; ++w) {
        sum += warp_sums[w];
    }
    return sum;
}

// top_k_radix_histogram over the contiguous columns of the block; first adds the elements of the block that the select
// of the previous pass put above the threshold to above[block]
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_histogram_stable(
        const float * __restrict__ src,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ block_histograms,
        int * __restrict__ above,
        int ncols,
        int blocks_per_row,
        int cols_per_block,
        int shift) {
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(NBINS == BLOCK_SIZE, "a thread per bin");

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const float * row_src = src + (size_t) row * ncols;
    int * histogram_out = block_histograms + (size_t) blockIdx.x * NBINS;
    __shared__ int histogram[NBINS];
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    const top_k_radix_state state = states[row];
    const bool first = shift + RADIX_BITS >= 32;
    int n_above = 0;
    if (!first) {
        const int chosen = (state.prefix >> (shift + RADIX_BITS)) & (NBINS - 1);
        n_above = tid > chosen ? histogram_out[tid] : 0;
    }
    histogram[tid] = 0;
    // its barrier also puts the zeroed histogram before the counts below
    n_above = top_k_stable_block_sum<BLOCK_SIZE>(n_above, warp_sums);
    if (tid == 0) {
        above[blockIdx.x] = (first ? 0 : above[blockIdx.x]) + n_above;
    }

    const int64_t begin = (int64_t) row_block * cols_per_block;
    const int64_t end = begin + cols_per_block < ncols ? begin + cols_per_block : ncols;
    for (int64_t col = begin + tid; col < end; col += BLOCK_SIZE) {
        const uint32_t key = top_k_float_to_ordered(row_src[col]);
        if ((key & state.prefix_mask) == state.prefix) {
            atomicAdd(&histogram[(key >> shift) & (NBINS - 1)], 1);
        }
    }
    __syncthreads();

    histogram_out[tid] = histogram[tid];
}

// after the select of the last pass: the rest of above[block] and the ties of the block
template<int BLOCK_SIZE, int RADIX_BITS>
static __global__ void top_k_radix_counts_stable(
        const int * __restrict__ block_histograms,
        const top_k_radix_state * __restrict__ states,
        int * __restrict__ above,
        int * __restrict__ ties,
        int blocks_per_row) {
    constexpr int NBINS = 1 << RADIX_BITS;
    static_assert(NBINS == BLOCK_SIZE, "a thread per bin");

    const int row = blockIdx.x / blocks_per_row;
    const int tid = threadIdx.x;
    __shared__ int warp_sums[BLOCK_SIZE / WARP_SIZE];

    const int chosen = states[row].prefix & (NBINS - 1);
    const int count = block_histograms[(size_t) blockIdx.x * NBINS + tid];
    const int n_above = top_k_stable_block_sum<BLOCK_SIZE>(tid > chosen ? count : 0, warp_sums);
    if (tid == 0) {
        above[blockIdx.x] += n_above;
    }
    if (tid == chosen) {
        ties[blockIdx.x] = count;
    }
}

// the elements of the block above the threshold from the slot after those of the blocks before it, in column order, and
// so the ties from k - rank, as long as they fit
template<int BLOCK_SIZE>
static __global__ void top_k_radix_gather_stable(
        const float * __restrict__ src,
        int * __restrict__ dst,
        const top_k_radix_state * __restrict__ states,
        const int * __restrict__ above,
        const int * __restrict__ ties,
        int ncols,
        int k,
        int blocks_per_row,
        int cols_per_block) {
    constexpr int NWARPS = BLOCK_SIZE / WARP_SIZE;

    const int row = blockIdx.x / blocks_per_row;
    const int row_block = blockIdx.x % blocks_per_row;
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    const int warp = tid / WARP_SIZE;
    const float * row_src = src + (size_t) row * ncols;
    int * row_dst = dst + (size_t) row * k;
    const int * row_above = above + (size_t) row * blocks_per_row;
    const int * row_ties = ties + (size_t) row * blocks_per_row;
    __shared__ int warp_above[NWARPS];
    __shared__ int warp_ties[NWARPS];
    __shared__ int start[2];

    const top_k_radix_state state = states[row];
    const int n_above = k - state.rank;

    if (warp == 0) {
        int a = 0;
        int t = 0;
        for (int b = lane; b < row_block; b += WARP_SIZE) {
            a += row_above[b];
            t += row_ties[b];
        }
        a = warp_reduce_sum(a);
        t = warp_reduce_sum(t);
        if (lane == 0) {
            start[0] = a;
            start[1] = t;
        }
    }
    __syncthreads();
    int a_slot = start[0];
    int t_slot = start[1];

    // the same for the whole block
    if (row_above[row_block] == 0 && (row_ties[row_block] == 0 || t_slot >= state.rank)) {
        return;
    }

    const uint32_t lanes_below = (1u << lane) - 1;
    const int64_t begin = (int64_t) row_block * cols_per_block;
    const int64_t end = begin + cols_per_block < ncols ? begin + cols_per_block : ncols;
    for (int64_t col0 = begin; col0 < end; col0 += BLOCK_SIZE) {
        const int64_t col = col0 + tid;
        const uint32_t key = col < end ? top_k_float_to_ordered(row_src[col]) : 0;
        const bool is_above = col < end && key > state.prefix;
        const bool is_tie = col < end && key == state.prefix;
        const uint32_t ballot_above = __ballot_sync(0xffffffff, is_above);
        const uint32_t ballot_tie = __ballot_sync(0xffffffff, is_tie);
        if (lane == 0) {
            warp_above[warp] = __popc(ballot_above);
            warp_ties[warp] = __popc(ballot_tie);
        }
        __syncthreads();

        int a_before = 0;
        int t_before = 0;
        int a_tile = 0;
        int t_tile = 0;
#pragma unroll
        for (int w = 0; w < NWARPS; ++w) {
            a_before += w < warp ? warp_above[w] : 0;
            t_before += w < warp ? warp_ties[w] : 0;
            a_tile += warp_above[w];
            t_tile += warp_ties[w];
        }
        if (is_above) {
            row_dst[a_slot + a_before + __popc(ballot_above & lanes_below)] = (int) col;
        }
        if (is_tie) {
            const int pos = t_slot + t_before + __popc(ballot_tie & lanes_below);
            if (pos < state.rank) {
                row_dst[n_above + pos] = (int) col;
            }
        }
        a_slot += a_tile;
        t_slot += t_tile;
        // warp_above and warp_ties are written again by the next tile
        __syncthreads();
    }
}

// top_k_radix_cuda with the stable passes; false when off
static bool top_k_radix_stable_fork(
        ggml_cuda_pool & pool,
        const float * src, int * dst, int ncols, int64_t nrows, int k, cudaStream_t stream) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_TOPK_RADIX_STABLE");
        return env == nullptr || atoi(env) != 0;
    }();
    if (!enabled) {
        return false;
    }

    constexpr int BLOCK_SIZE = 256;
    constexpr int RADIX_BITS = 8;
    constexpr int NBINS = 1 << RADIX_BITS;
    // the blocks of a row: as upstream (a block per 1024 columns, at most 64) while the rows are few (a decode, a draft
    // check), fewer when the rows fill the GPU by themselves (a prompt ubatch): enough for 4 waves of 8 blocks per SM.
    // per pass each block writes a histogram that the select reads and the next pass reads again for the counts: 3 KB
    // against 3.9 KB of input per block with 64 blocks over 63040 columns; with the blocks of upstream a call of 4424
    // rows took 12.4 ms against 10.6 ms of the upstream gather (3090 02.11). each block takes whole tiles of BLOCK_SIZE
    const int64_t nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int64_t upstream_blocks = std::min<int64_t>(((int64_t) ncols + 1023) / 1024, 64);
    const int64_t fill_blocks = (32 * nsm + nrows - 1) / std::max<int64_t>(nrows, 1);
    const int blocks_per_row = (int) std::max<int64_t>(1, std::min(upstream_blocks, fill_blocks));
    const int64_t per_block = ((int64_t) ncols + blocks_per_row - 1) / blocks_per_row;
    const int cols_per_block = (int) ((per_block + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);

    // chunk the rows to bound the histogram memory to 64 MB
    const int64_t chunk_nrows = ggml_cuda_chunk_nrows((size_t) blocks_per_row * NBINS * sizeof(int), nrows);

    ggml_cuda_pool_alloc<top_k_radix_state> states_alloc(pool, chunk_nrows);
    ggml_cuda_pool_alloc<int> histograms_alloc(pool, (size_t) chunk_nrows * blocks_per_row * NBINS);
    ggml_cuda_pool_alloc<int> counts_alloc(pool, (size_t) chunk_nrows * blocks_per_row * 2);
    top_k_radix_state * states = states_alloc.get();
    int * histograms = histograms_alloc.get();
    int * above = counts_alloc.get();
    int * ties = above + (size_t) chunk_nrows * blocks_per_row;

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        const int iter_nrows = std::min(chunk_nrows, nrows - i);

        top_k_radix_init<<<(iter_nrows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(states, iter_nrows, k);

        const dim3 row_grid(blocks_per_row * iter_nrows);
        for (int shift = 32 - RADIX_BITS; shift >= 0; shift -= RADIX_BITS) {
            top_k_radix_histogram_stable<BLOCK_SIZE, RADIX_BITS>
                <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                    src, states, histograms, above, ncols, blocks_per_row, cols_per_block, shift);
            top_k_radix_select<BLOCK_SIZE, RADIX_BITS>
                <<<iter_nrows, BLOCK_SIZE, 0, stream>>>(histograms, states, blocks_per_row, shift);
        }

        top_k_radix_counts_stable<BLOCK_SIZE, RADIX_BITS>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(histograms, states, above, ties, blocks_per_row);
        top_k_radix_gather_stable<BLOCK_SIZE>
            <<<row_grid, BLOCK_SIZE, 0, stream>>>(
                src, dst, states, above, ties, ncols, k, blocks_per_row, cols_per_block);

        src += (size_t) ncols * iter_nrows;
        dst += (size_t) k     * iter_nrows;
    }
    return true;
}

#else

static bool top_k_radix_stable_fork(ggml_cuda_pool &, const float *, int *, int, int64_t, int, cudaStream_t) {
    return false;
}

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
