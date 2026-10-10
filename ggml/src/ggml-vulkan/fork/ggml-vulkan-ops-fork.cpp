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

bool ggml_vk_topk_use_radix(uint32_t k, uint32_t ncols, uint32_t nrows) {
    // radix-select (one workgroup per row) for a large k over many long rows: the qwen4exp block top-k in prefill, k 513
    // of 31744 x 4096, takes 5.0 ms against 19.3 for the tournament on Strix Halo (8192 x 16: 19 against 32 us). the
    // tournament spreads a row over workgroups and stays faster for a few rows (decode, draft checks: 1 to 4 rows) and
    // has the short rows of MoE routing. GGML_VK_TOPK_RADIX: 1 = radix-select for every top-k, 0 = only past the k
    // limit of the tournament
    static const int radix_mode = [] {
        const char * env = getenv("GGML_VK_TOPK_RADIX");
        return env ? atoi(env) : -1;
    }();
    const bool radix_wins = k >= 256 && ncols >= 4096 && nrows >= 8;
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
// with f32 activations read in place, the hoisted row ids of count_experts.comp, rows in whole 128-row tiles and K in
// whole 64-column stages of whole blocks, 32-bit offsets. from GGML_VK_MMID_F16REG_MIN_ROWS rows per expert on average
// (16): decode batches take the mat-vec path before they get here
vk_pipeline ggml_vk_fork_mmid_f16reg(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * glu_up,
                                     bool hoist_row_ids, bool y_f32_kernel, uint64_t n_per_expert) {
    if (ctx->device->mmid_f16reg_mode == 0 || !hoist_row_ids || !y_f32_kernel || src1->type != GGML_TYPE_F32) {
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
    vk_pipeline pipeline = ctx->device->pipeline_mmid_f16reg[t][glu_up != nullptr ? 1 : 0];
    // GGML_VK_MMID_F16REG_LOG=1: every choice, else the first
    static const bool log_all = getenv("GGML_VK_MMID_F16REG_LOG") != nullptr;
    static bool logged = false;
    if (pipeline && (!logged || log_all)) {
        logged = true;
        GGML_LOG_INFO("ggml_vulkan: MUL_MAT_ID %s on %s (%s, %ld rows per expert)\n", ggml_type_name(src0->type), pipeline->name.c_str(),
                      glu_up ? "fused gate/up" : "single", (long) n_per_expert);
    }
    return pipeline;
}
