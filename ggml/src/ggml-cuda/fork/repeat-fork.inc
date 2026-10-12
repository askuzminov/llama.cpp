// fork: REPEAT of one element into a contiguous dst as a fill, included by binbcast.cu. the qwen4exp QSA mask starts as
// the -inf of a one-element tensor repeated to n_kv x ubatch f16 (1 GiB a layer at depth 122880 x 4096), which the
// broadcast kernel wrote 2 bytes a thread with its index math; here a thread stores 16 bytes

template <typename T>
static __global__ void k_repeat_scalar_fork(const T * __restrict__ src, T * __restrict__ dst, const int64_t n) {
    constexpr int per = sizeof(uint4) / sizeof(T);
    union {
        uint4 u;
        T     t[per];
    } pack;
    const T v = src[0];
#pragma unroll
    for (int j = 0; j < per; ++j) {
        pack.t[j] = v;
    }
    const int64_t n4     = n / per;
    const int64_t stride = (int64_t) gridDim.x * blockDim.x;
    uint4 * dst4 = (uint4 *) dst;
    for (int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        dst4[i] = pack.u;
    }
    for (int64_t i = n4 * per + (int64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = v;
    }
}

static bool ggml_cuda_repeat_scalar_fork(const ggml_tensor * src, ggml_tensor * dst, cudaStream_t stream) {
    const size_t ts = ggml_type_size(dst->type);
    if (ggml_nelements(src) != 1 || !ggml_is_contiguous(dst) || ggml_type_size(src->type) != ts ||
        (ts != 2 && ts != 4) || (uintptr_t) dst->data % sizeof(uint4) != 0) {
        return false;
    }
    const int64_t n       = ggml_nelements(dst);
    const int     threads = 256;
    const int64_t n4      = n * (int64_t) ts / (int64_t) sizeof(uint4);
    const int     blocks  = (int) std::max<int64_t>(1, std::min<int64_t>((n4 + threads - 1) / threads, 65536));
    if (ts == 2) {
        k_repeat_scalar_fork<uint16_t><<<blocks, threads, 0, stream>>>((const uint16_t *) src->data, (uint16_t *) dst->data, n);
    } else {
        k_repeat_scalar_fork<uint32_t><<<blocks, threads, 0, stream>>>((const uint32_t *) src->data, (uint32_t *) dst->data, n);
    }
    return true;
}
