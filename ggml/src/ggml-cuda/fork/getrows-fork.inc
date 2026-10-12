// fork: GET_ROWS of short rows, one thread per element over all rows, included by getrows.cu. upstream launches a
// block of 256 threads per row, so a row of 1 element keeps 1 of 256 busy: the qwen4exp attention gathers rows of 1
// (the position or mask value of each selected cell, 2052 x 4096 per QSA layer of a 4096 prompt) and of 4 (the cells
// of each selected block, 513 x 4096). on the 3090 (12.10, GGML_CUDA_OP_TIMING, two prompts of 4096 at depth 0) the
// flat gather of positions took 30 ms per call (727 ms of 5006), the mask values 13 ms, the block cells 7.4 ms.
// GGML_CUDA_GET_ROWS_FLAT=0: upstream for every row size

#include <cstdlib>

template<typename src0_t, typename dst_t, typename idx_t>
static __global__ void k_get_rows_flat_fork(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const idx_t ne00, const idx_t ne10, const idx_t ne11, const idx_t n,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12,
        const size_t s1, const size_t s2, const size_t s3) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();

    const idx_t i = (idx_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const idx_t i00 = i % ne00;
    idx_t       r   = i / ne00;
    const idx_t i10 = r % ne10;
    r /= ne10;
    const idx_t i11 = r % ne11;
    const idx_t i12 = r / ne11;

    const int32_t i01 = src1[i10*s10 + i11*s11 + i12*s12];
    const src0_t * src0_row = (const src0_t *) ((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);
    dst[i10*s1 + i11*s2 + i12*s3 + i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
}

// true when it took the gather (strides of src1 and dst in elements)
template<typename src0_t, typename dst_t>
static bool ggml_cuda_get_rows_flat_fork(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12,
        const size_t s10, const size_t s11, const size_t s12,
        const size_t s1, const size_t s2, const size_t s3,
        cudaStream_t stream) {
    static const bool enabled = [] {
        const char * env = getenv("GGML_CUDA_GET_ROWS_FLAT");
        return env == nullptr || atoi(env) != 0;
    }();
    // half a block or less per row
    if (!enabled || ne00 > CUDA_GET_ROWS_BLOCK_SIZE / 2) {
        return false;
    }
    const int64_t n = ne00 * ne10 * ne11 * ne12;
    if (n <= 0) {
        return true;
    }
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const dim3 block_nums((unsigned int) ((n + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE), 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    if (n <= (int64_t) UINT32_MAX) {
        ggml_cuda_kernel_launch(k_get_rows_flat_fork<src0_t, dst_t, uint32_t>, launch_params,
            src0_d, src1_d, dst_d,
            (uint32_t) ne00, (uint32_t) ne10, (uint32_t) ne11, (uint32_t) n,
            nb01, nb02, nb03, s10, s11, s12, s1, s2, s3);
    } else {
        ggml_cuda_kernel_launch(k_get_rows_flat_fork<src0_t, dst_t, int64_t>, launch_params,
            src0_d, src1_d, dst_d,
            ne00, ne10, ne11, n,
            nb01, nb02, nb03, s10, s11, s12, s1, s2, s3);
    }
    return true;
}
