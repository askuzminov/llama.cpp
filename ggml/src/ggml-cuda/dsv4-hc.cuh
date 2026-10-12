#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// fork: HC_POST with the post weights s_out*sigmoid(s_in*p_src) computed in the kernel (fork/dsv4-hc-fork.inc)
void ggml_cuda_op_dsv4_hc_post_gated_fork(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * p_src, float s_in, float s_out);
