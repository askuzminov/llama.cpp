// fork: top-k over many rows with the radix select of top-k.cu (written for HIP) on CUDA too, included by top-k.cu.
// with CCCL >= 3.4.3 upstream runs cub DeviceTopK once per row: the qwen4exp block top-k of a prompt ubatch (k 513 over
// the KV blocks, a row per token) took 96 ms per call for 2596 rows on the 3090 (CUDA 13.4, cuda_op_timing 11.10),
// 1.15 s of a 2.6K prompt; the radix select does all rows in 10 launches. GGML_CUDA_TOPK_RADIX: 0 = upstream, 1 = the
// radix select for every top-k, default from 8 rows (decode and draft checks keep the upstream path)

#include <cstdlib>

static bool ggml_cuda_fork_topk_radix(int64_t nrows) {
    static const int mode = [] {
        const char * env = getenv("GGML_CUDA_TOPK_RADIX");
        return env ? atoi(env) : -1;
    }();
    return mode == 1 || (mode != 0 && nrows >= 8);
}
