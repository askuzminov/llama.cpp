// fork: DSV4_HC_POST with its post weights computed in the kernel, included by dsv4-hc.cu. the qwen4exp combine builds the
// weights as scale(sigmoid(scale(inject))) (2*sigmoid(inject/hc)), three kernels of [hc, n_tokens] before every HC_POST;
// ggml_cuda_fork_try_fuse (fork/fusion-fork.inc) runs the four nodes as this one, as Vulkan does (HC_POST_GATE)

template <bool has_comb>
static __global__ void dsv4_hc_post_gated_f32_fork(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float   s_in,
        float   s_out) {
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    const float w = s_out / (1.0f + expf(-s_in * post[idst*sp0 + it*sp1]));

    float sum = x[i0*sx0 + it*sx1] * w;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

// dst: the HC_POST node, p_src: the input of the first SCALE (its post weights are s_out*sigmoid(s_in*p_src))
void ggml_cuda_op_dsv4_hc_post_gated_fork(ggml_backend_cuda_context & ctx, ggml_tensor * dst, const ggml_tensor * p_src, float s_in, float s_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 && p_src->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);

    auto kernel = comb ? dsv4_hc_post_gated_f32_fork<true> : dsv4_hc_post_gated_f32_fork<false>;
    kernel<<<grid_dims, block_dims, 0, ctx.stream()>>>(
            (const float *) x->data, (const float *) residual->data,
            (const float *) p_src->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            x->nb[0] / sizeof(float), x->nb[1] / sizeof(float),
            residual->nb[0] / sizeof(float), residual->nb[1] / sizeof(float), residual->nb[2] / sizeof(float),
            p_src->nb[0] / sizeof(float), p_src->nb[1] / sizeof(float),
            comb ? comb->nb[0] / sizeof(float) : 0, comb ? comb->nb[1] / sizeof(float) : 0, comb ? comb->nb[2] / sizeof(float) : 0,
            dst->nb[0] / sizeof(float), dst->nb[1] / sizeof(float), dst->nb[2] / sizeof(float),
            s_in, s_out);
}
