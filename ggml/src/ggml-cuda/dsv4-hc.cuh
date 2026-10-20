#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);

// fork: HC_POST with the post weights s_out*sigmoid(s_in*p_src) computed in the kernel (fork/dsv4-hc-fork.inc). it
// fuses the same chain as the gate_scale_* of ggml_cuda_op_dsv4_hc_post and is tried first (fork/fusion-fork.inc);
// GGML_CUDA_FORK_FUSION=0 leaves the chain to the upstream fusion
void ggml_cuda_op_dsv4_hc_post_gated_fork(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * p_src, float s_in, float s_out);
