// fork: CUDA fusions of the qwen4exp hyper-connection ops, included by ggml-cuda.cu and tried first by
// ggml_cuda_try_fuse. a decode step on the 3090 ran about 2800 kernels (cuda_op_timing, 11.10), 24 of the ~58 per layer
// hyper-connection work. Vulkan fuses the same patterns (HC_POST_GATE, ggml-vulkan-fusion-fork.cpp).
//   SCALE, SIGMOID, SCALE, DSV4_HC_POST -> one HC_POST that computes its post weights (3 kernels less per combine)
//   SCALE, SILU                         -> silu(x*s), the low-rank activation of the mix (1 kernel less per mix)
// GGML_CUDA_FORK_FUSION=0: off

static __global__ void k_scale_silu_fork(const float * __restrict__ x, float * __restrict__ dst, const int64_t n, const float s) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i] * s;
        dst[i] = v / (1.0f + expf(-v));
    }
}

static bool ggml_cuda_fork_scale_plain(const ggml_tensor * t) {
    // ggml_scale_bias keeps the bias in op_params[1]
    return t->op == GGML_OP_SCALE && ggml_get_op_params_f32(t, 1) == 0.0f;
}

static int ggml_cuda_fork_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {
    static const bool disabled = [] {
        const char * env = getenv("GGML_CUDA_FORK_FUSION");
        return env != nullptr && atoi(env) == 0;
    }();
    ggml_tensor * s1 = cgraph->nodes[i];
    if (disabled || !ggml_cuda_fork_scale_plain(s1) || s1->type != GGML_TYPE_F32 || s1->src[0]->type != GGML_TYPE_F32) {
        return 0;
    }

    if (i + 3 < cgraph->n_nodes && ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST }, { i + 3 })) {
        const ggml_tensor * sig  = cgraph->nodes[i + 1];
        const ggml_tensor * s2   = cgraph->nodes[i + 2];
        ggml_tensor *       post = cgraph->nodes[i + 3];
        const int out = i + 3;
        if (ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->src[0] == s1 && s2->src[0] == sig && ggml_cuda_fork_scale_plain(s2) &&
                post->src[2] == s2 && post->src[0] != s2 && post->src[1] != s2 && post->src[3] != s2 &&
                ggml_are_same_shape(s1->src[0], s2) && ggml_cuda_check_fusion_memory_ranges(cgraph, i, 4, &out, 1)) {
            ggml_cuda_op_dsv4_hc_post_gated_fork(*cuda_ctx, post, s1->src[0], ggml_get_op_params_f32(s1, 0), ggml_get_op_params_f32(s2, 0));
            return 3;
        }
    }

    if (i + 1 < cgraph->n_nodes && ggml_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY })) {
        ggml_tensor * silu = cgraph->nodes[i + 1];
        const int out = i + 1;
        if (ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == s1 && silu->type == GGML_TYPE_F32 &&
                ggml_is_contiguous(s1->src[0]) && ggml_is_contiguous(silu) && ggml_are_same_shape(s1->src[0], silu) &&
                ggml_cuda_check_fusion_memory_ranges(cgraph, i, 2, &out, 1)) {
            const int64_t n = ggml_nelements(silu);
            const int threads = 256;
            k_scale_silu_fork<<<(n + threads - 1) / threads, threads, 0, cuda_ctx->stream()>>>(
                (const float *) s1->src[0]->data, (float *) silu->data, n, ggml_get_op_params_f32(s1, 0));
            return 1;
        }
    }

    return 0;
}
