// pipeline choices and paths of single ops in the fork (see ggml-vulkan-fork.h)

#include "ggml-vulkan-fork.h"

// GET_ROWS: rows shorter than the 512-wide workgroup take the flat pipeline, which spreads the elements of
// all rows over the workgroup; one row per workgroup leaves it mostly idle (one-element rows: 1 of 512)
bool ggml_vk_get_rows_flat(const ggml_tensor * src0, const ggml_tensor * src1) {
    // GGML_VK_DISABLE_GET_ROWS_FLAT (experiment): the old dispatch, for the comparison
    static const bool disabled = getenv("GGML_VK_DISABLE_GET_ROWS_FLAT") != nullptr;
    return !disabled &&
           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16 || src0->type == GGML_TYPE_I32) &&
           src0->ne[0] < 512 && (uint64_t)src0->ne[0] * src1->ne[0] <= UINT32_MAX / 2;
}

vk_pipeline ggml_vk_get_rows_flat_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    if (src0->type == GGML_TYPE_I32) {
        // i32 src only supports i32 result
        GGML_ASSERT(dst->type == GGML_TYPE_I32);
        return ctx->device->pipeline_get_rows_flat[src0->type];
    }
    if (dst->type == GGML_TYPE_F16) {
        return ctx->device->pipeline_get_rows_flat[src0->type];
    }
    if (dst->type == GGML_TYPE_F32) {
        return ctx->device->pipeline_get_rows_flat_f32[src0->type];
    }
    return nullptr;
}

vk_pipeline ggml_vk_concat_transpose_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1) {
    // a source with dim 1 innermost (a transposed view) goes through tiles, so its reads are contiguous
    auto transposed = [](const ggml_tensor * t) { return t->ne[0] > 1 && t->ne[1] > 1 && t->nb[1] < t->nb[0]; };
    if (!ggml_is_quantized(src0->type) && (transposed(src0) || transposed(src1))) {
        if (ggml_type_size(src0->type) == 4) {
            return ctx->device->pipeline_concat_transpose_32;
        }
        if (ggml_type_size(src0->type) == 2) {
            return ctx->device->pipeline_concat_transpose_16;
        }
    }
    return nullptr;
}

bool ggml_vk_topk_use_radix(uint32_t k, uint32_t ncols, uint32_t nrows, uint32_t tournament_cols) {
    // radix-select (one workgroup per row) for a large k over many long rows: the qwen4exp block top-k in prefill, k 513
    // of 31744 x 4096, takes 5.0 ms against 19.3 for the tournament on Strix Halo (8192 x 16: 19 against 32 us). the
    // tournament spreads a row over workgroups and stays faster for a few rows (decode, draft checks: 1 to 4 rows) and
    // has the short rows of MoE routing. a row longer than its widest workgroup (1024 elements) takes more passes, each
    // keeping k of every workgroup, so for k 513 a pass only halves the row: on the 395 (13.10, a 118K prompt, 12 calls
    // per ubatch of 4096) 8 ms at 960 columns (one pass), 20-25 at 1088 (two), 37-43 at 2112 (three), 64 at 3776
    // (four), radix 15 at 4160. GGML_VK_TOPK_RADIX: 1 = radix-select for every top-k, 0 = only past the k limit of the
    // tournament
    static const int radix_mode = [] {
        const char * env = getenv("GGML_VK_TOPK_RADIX");
        return env ? atoi(env) : -1;
    }();
    const bool radix_wins = k >= 256 && ncols > tournament_cols && nrows >= 8;
    return radix_mode == 1 || (radix_mode != 0 && radix_wins);
}

bool ggml_vk_mul_mat_int8_off(const ggml_backend_vk_context * ctx, const ggml_tensor * dst, uint64_t ne10) {
    // GGML_VK_INT_COOPMAT=3: int8 for MUL_MAT_ID only; with coopmat the q8_1 pipelines are the int8 coopmat ones.
    // a graph output (the logits) stays off int8 too: the error of the q8_1 activations goes straight into the
    // result. it is a matmul only when many rows need logits (perplexity), a prompt or a decode step reads one row.
    // so does K under 512: on Strix Halo the qwen4exp hc up projection (K 320) is 41% slower on int8 and takes
    // the KLD against exact sums from 0.020 to 0.044
    return ctx->device->coopmat_int_support && (!(ctx->device->coopmat_int_ops & 1u) || (dst->flags & GGML_TENSOR_FLAG_OUTPUT) || ne10 < 512);
}

bool ggml_vk_set_tensor_mapped(vk_context & cpy_ctx, const ggml_tensor * tensor, vk_buffer & buf, uint64_t dst_offset, const void * data,
                               size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    // on UMA the weights are mapped, so write them in place instead of staging plus a queue copy
    // weights only: nothing queued writes them during the load, and they are read only afterwards
    if (tensor->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        (buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostVisible)) {
        GGML_ASSERT(buf->memory_property_flags & vk::MemoryPropertyFlagBits::eHostCoherent);

        if (size == stride_data && size == stride_tensor) {
            deferred_memcpy((uint8_t *)buf->ptr + dst_offset, data, size * n_copies, &cpy_ctx->in_memcpys);
        } else {
            for (size_t i = 0; i < n_copies; i++) {
                deferred_memcpy((uint8_t *)buf->ptr + dst_offset + i * stride_tensor, (const uint8_t *)data + i * stride_data, size, &cpy_ctx->in_memcpys);
            }
        }
        return true;
    }
    return false;
}

// MUL_MAT_ID on mul_mmid_f16reg.comp (GGML_VK_MMID_F16REG): q4_K, q5_K (also the fused gate/up), q5_1 and q8_0 experts
// with f32 activations read in place, the row map of count_experts.comp, rows in whole 128-row tiles and K in
// whole 64-column stages of whole blocks, 32-bit offsets. from GGML_VK_MMID_F16REG_MIN_ROWS rows per expert on average
// (16): decode batches take the mat-vec path before they get here
vk_pipeline ggml_vk_fork_mmid_f16reg(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * glu_up,
                                     bool y_f32_kernel, uint64_t n_per_expert) {
    if (ctx->device->mmid_f16reg_mode == 0 || !y_f32_kernel || src1->type != GGML_TYPE_F32) {
        return nullptr;
    }
    static const uint64_t min_rows = [] { const char * env = getenv("GGML_VK_MMID_F16REG_MIN_ROWS"); return env ? (uint64_t) atoi(env) : 16u; }();
    if (n_per_expert < min_rows) {
        return nullptr;
    }
    int t = -1;
    switch (src0->type) {
        case GGML_TYPE_Q4_K: t = 0; break;
        case GGML_TYPE_Q5_K: t = 1; break;
        case GGML_TYPE_Q5_1: t = 2; break;
        case GGML_TYPE_Q8_0: t = 3; break;
        default: return nullptr;
    }
    const int64_t qk = ggml_blck_size(src0->type);
    const uint64_t range = ctx->device->properties.limits.maxStorageBufferRange;
    if (src0->ne[1] % 128 != 0 || src0->ne[0] % 64 != 0 || src0->ne[0] % qk != 0 || !ggml_vk_dim01_contiguous(src0) ||
        ggml_nbytes(src0) > range || ggml_nbytes(src1) > range) {
        return nullptr;
    }
    if (glu_up != nullptr && (glu_up->type != src0->type || !ggml_are_same_shape(glu_up, src0) || ggml_nbytes(glu_up) > range)) {
        return nullptr;
    }
    const int g = glu_up != nullptr ? 1 : 0;
    vk_pipeline pipeline = ctx->device->pipeline_mmid_f16reg[t][g];
    // the first choice of each pipeline, every one with GGML_VK_MMID_F16REG_LOG=1; on stderr as well with
    // GGML_SCHED_LOG_REALLOC, which 02 sets (llama-bench drops the ggml log)
    static const bool log_all = getenv("GGML_VK_MMID_F16REG_LOG") != nullptr;
    static const bool log_stderr = getenv("GGML_SCHED_LOG_REALLOC") != nullptr;
    static bool logged[4][2] = {};
    if (pipeline && (!logged[t][g] || log_all)) {
        logged[t][g] = true;
        GGML_LOG_INFO("ggml_vulkan: MUL_MAT_ID %s on %s (%s, %ld rows per expert)\n", ggml_type_name(src0->type), pipeline->name.c_str(),
                      g ? "fused gate/up" : "single", (long) n_per_expert);
        if (log_stderr) {
            fprintf(stderr, "ggml_vulkan: MUL_MAT_ID %s on %s (%s, %ld rows per expert)\n", ggml_type_name(src0->type), pipeline->name.c_str(),
                    g ? "fused gate/up" : "single", (long) n_per_expert);
        }
    }
    return pipeline;
}

// GATED_DELTA_NET_LAZY: one workgroup row per head of each info row, laid out as the plain net (gated_delta_net.comp);
// the replay alone (op_params[0] 2) has no q, k, v, g and beta and binds the state in their place
static int ggml_vk_gdn_size_idx(uint32_t S_v) {
    switch (S_v) {
        case 16:  return 0;
        case 32:  return 1;
        case 64:  return 2;
        case 128: return 3;
        default:  return -1;
    }
}

bool ggml_vk_gated_delta_net_lazy_supported(const ggml_tensor * op) {
    const int      mode = ggml_get_op_params_i32(op, 0);
    const uint32_t S_v  = mode == 2 ? (uint32_t) ggml_get_op_params_i32(op, 1) : (uint32_t) op->src[2]->ne[0];
    if (ggml_vk_gdn_size_idx(S_v) < 0 || op->type != GGML_TYPE_F32) {
        return false;
    }
    if (op->src[5]->type != GGML_TYPE_F32 || op->src[6]->type != GGML_TYPE_F32 || op->src[7]->type != GGML_TYPE_I32) {
        return false;
    }
    for (int i = 0; mode == 1 && i < 5; i++) {
        if (op->src[i]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    return true;
}

void ggml_vk_gated_delta_net_lazy(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 0);

    const ggml_tensor * src_q    = dst->src[0];
    const ggml_tensor * src_k    = dst->src[1];
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_beta = dst->src[4];
    const ggml_tensor * state    = dst->src[5];
    const ggml_tensor * rec      = dst->src[6];
    const ggml_tensor * info     = dst->src[7];

    const uint32_t S_v = mode == 2 ? (uint32_t) ggml_get_op_params_i32(dst, 1) : (uint32_t) src_v->ne[0];
    const uint32_t H   = mode == 2 ? (uint32_t) ggml_get_op_params_i32(dst, 2) : (uint32_t) src_v->ne[1];
    const uint32_t H_k = mode == 2 ? (uint32_t) ggml_get_op_params_i32(dst, 3) : (uint32_t) src_k->ne[1];

    const int si = ggml_vk_gdn_size_idx(S_v);
    GGML_ASSERT(si >= 0);
    vk_pipeline pipeline = ctx->device->pipeline_gated_delta_net_lazy[si][mode - 1];
    GGML_ASSERT(pipeline != nullptr);

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const vk_subbuffer state_buf = ggml_vk_tensor_subbuffer(ctx, state);
    const vk_subbuffer rec_buf   = ggml_vk_tensor_subbuffer(ctx, rec);
    const vk_subbuffer info_buf  = ggml_vk_tensor_subbuffer(ctx, info);
    const vk_subbuffer dst_buf   = mode == 2 ? state_buf : ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer src_buf[5];
    for (int i = 0; i < 5; i++) {
        src_buf[i] = mode == 2 ? state_buf : ggml_vk_tensor_subbuffer(ctx, dst->src[i]);
    }

    vk_op_gated_delta_net_lazy_push_constants pc = {};
    pc.H          = H;
    pc.H_k        = H_k;
    pc.scale      = 1.0f / sqrtf((float) S_v);
    pc.rec_floats = (uint32_t) rec->ne[0];
    pc.rec_set    = (uint32_t) (rec->nb[2] / sizeof(float));
    if (mode == 1) {
        pc.n_tokens = (uint32_t) src_v->ne[2];
        pc.sq1 = (uint32_t) (src_q->nb[1] / sizeof(float));
        pc.sq2 = (uint32_t) (src_q->nb[2] / sizeof(float));
        pc.sq3 = (uint32_t) (src_q->nb[3] / sizeof(float));
        pc.sk1 = (uint32_t) (src_k->nb[1] / sizeof(float));
        pc.sk2 = (uint32_t) (src_k->nb[2] / sizeof(float));
        pc.sk3 = (uint32_t) (src_k->nb[3] / sizeof(float));
        pc.sv1 = (uint32_t) (src_v->nb[1] / sizeof(float));
        pc.sv2 = (uint32_t) (src_v->nb[2] / sizeof(float));
        pc.sv3 = (uint32_t) (src_v->nb[3] / sizeof(float));
        pc.sb1 = (uint32_t) (src_beta->nb[1] / sizeof(float));
        pc.sb2 = (uint32_t) (src_beta->nb[2] / sizeof(float));
        pc.sb3 = (uint32_t) (src_beta->nb[3] / sizeof(float));
        pc.rq3 = (uint32_t) (src_v->ne[3] / src_q->ne[3]);
    } else {
        pc.rq3 = 1;
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], state_buf, rec_buf, info_buf, dst_buf},
        pc, { H, (uint32_t) info->ne[1], S_v });
}
