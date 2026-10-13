// fork: CPU ops the fork adds to ggml, included at the end of ops.cpp

// ggml_compute_forward_flash_attn_list: each (token, head, stream) row walks its list of cells with an online softmax.
// the reference of the backends' sparse paths (test-backend-ops); a model builds the op only for a backend that runs it
// on its sparse attention (qwen4exp on Vulkan, see src/fork/qwen4exp-fork.cpp)
void ggml_compute_forward_flash_attn_list(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
    const struct ggml_tensor * q    = dst->src[0];
    const struct ggml_tensor * k    = dst->src[1];
    const struct ggml_tensor * v    = dst->src[2];
    const struct ggml_tensor * vals = dst->src[3];
    const struct ggml_tensor * idx  = dst->src[5];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 && ggml_is_contiguous(dst));

    const int64_t DK      = q->ne[0];
    const int64_t DV      = v->ne[0];
    const int64_t n_tok   = q->ne[1];
    const int64_t n_head  = q->ne[2];
    const int64_t n_seq   = q->ne[3];
    const int64_t n_list  = idx->ne[0];
    const int64_t n_kv    = k->ne[1];
    const int64_t rk2     = n_head / k->ne[2];
    const int64_t rv2     = n_head / v->ne[2];
    const int64_t rk3     = n_seq / k->ne[3];
    const int64_t rv3     = n_seq / v->ne[3];
    const int64_t n_vseq  = vals->ne[3];

    float scale;
    memcpy(&scale, (const float *) dst->op_params + 0, sizeof(float));

    ggml_to_float_t const k_to_float = ggml_get_type_traits(k->type)->to_float;
    ggml_to_float_t const v_to_float = ggml_get_type_traits(v->type)->to_float;
    GGML_ASSERT(k_to_float && v_to_float);

    // per thread: the converted K and V rows and the V accumulator, a cache line apart
    float * wdata = (float *) params->wdata + params->ith * (DK + 2*DV + CACHE_LINE_SIZE_F32);
    float * k_row = wdata;
    float * v_row = k_row + DK;
    float * acc   = v_row + DV;

    const int64_t nr  = n_tok * n_head * n_seq;
    const int64_t dr  = (nr + params->nth - 1) / params->nth;
    const int64_t ir0 = dr * params->ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iq3 = ir / (n_tok * n_head);
        const int64_t iq1 = (ir / n_head) % n_tok;
        const int64_t iq2 = ir % n_head;

        const float * q_row = (const float *) ((const char *) q->data + iq1*q->nb[1] + iq2*q->nb[2] + iq3*q->nb[3]);
        const int64_t is3   = iq3 % n_vseq;
        const int32_t     * list = (const int32_t *)     ((const char *) idx->data  + iq1*idx->nb[1]  + is3*idx->nb[2]);
        const ggml_fp16_t * mval = (const ggml_fp16_t *) ((const char *) vals->data + iq1*vals->nb[1] + is3*vals->nb[3]);

        float M = -INFINITY;
        float S = 0.0f;
        for (int64_t d = 0; d < DV; ++d) {
            acc[d] = 0.0f;
        }

        for (int64_t s = 0; s < n_list; ++s) {
            const int32_t c = list[s];
            const float   m = GGML_CPU_FP16_TO_FP32(mval[s]);
            if (c < 0 || c >= n_kv || m == -INFINITY) {
                continue;
            }
            k_to_float((const char *) k->data + c*k->nb[1] + (iq2/rk2)*k->nb[2] + (iq3/rk3)*k->nb[3], k_row, DK);

            float dot = 0.0f;
            for (int64_t d = 0; d < DK; ++d) {
                dot += q_row[d]*k_row[d];
            }
            const float score = dot*scale + m;

            float ms = 1.0f;
            float vs = 1.0f;
            if (score > M) {
                ms = expf(M - score);
                M  = score;
                for (int64_t d = 0; d < DV; ++d) {
                    acc[d] *= ms;
                }
            } else {
                vs = expf(score - M);
            }
            S = S*ms + vs;

            v_to_float((const char *) v->data + c*v->nb[1] + (iq2/rv2)*v->nb[2] + (iq3/rv3)*v->nb[3], v_row, DV);
            for (int64_t d = 0; d < DV; ++d) {
                acc[d] += vs*v_row[d];
            }
        }

        const float inv = S > 0.0f ? 1.0f/S : 0.0f;
        float * out = (float *) ((char *) dst->data + iq2*dst->nb[1] + iq1*dst->nb[2] + iq3*dst->nb[3]);
        for (int64_t d = 0; d < DV; ++d) {
            out[d] = acc[d]*inv;
        }
    }
}
