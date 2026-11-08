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

int ggml_vk_topk_radix_stable() {
    // the upstream select compacts its picks with an atomic counter: the order of the output and, at ties, the elements it
    // takes changed from run to run. the qwen4exp block top-k of a prompt past 4096 cells (k 513, rows longer than the
    // 1024 of the tournament) runs here, and with another order the attention adds the chosen blocks in another order:
    // the logits of two equal runs on the 395 differed from the first ubatch of 4096 tokens on (01.11). default 1: with
    // subgroups (5500M: 0 to 6 percent over upstream), on a device without full subgroups upstream, since the select
    // without them reads a contiguous run per invocation (5500M: +40 percent at 32768 columns, 3-11x at 131072);
    // GGML_VK_TOPK_RADIX_STABLE=0: upstream, 2: the stable select without subgroups
    static const int mode = [] {
        const char * env = getenv("GGML_VK_TOPK_RADIX_STABLE");
        const int v = env ? atoi(env) : 1;
        return v == 0 || v == 2 ? v : 1;
    }();
    return mode;
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
// (16): decode batches take the mat-vec path before they get here.
// the tiles: GGML_VK_MMID_F16REG_BN (the single matmuls, the down projection) and GGML_VK_MMID_F16REG_GLU_BN (the fused
// gate/up) take 64 or 128 tokens a tile (64). 128 lost on the 395 at -ub 4096 (08.11, profile-20261108-170847, ms of a
// 4010-token ubatch): gate/up 664.9 -> 743.7, q5_1 down 445.4 -> 587.0 (221-255 VGPR against 150-171). the single
// matmuls of 256-row multiples take 256 rows a workgroup in two A tiles that share each B fragment, as the fused gate/up
// does (GGML_VK_MMID_F16REG_ROWS2): unset - q5_1 only (q5_1 down 445.4 -> 398.2, the q8_0 down lost, 52.6 -> 63.0:
// 144 VGPR against 92 for a cheap decode), 0 - none, 1 - every type. without a pipeline (the shared memory of the
// device) the 64-token tile and the plain one
vk_pipeline ggml_vk_fork_mmid_f16reg(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * glu_up,
                                     bool y_in_place, uint64_t n_per_expert, bool * b16, bool d16) {
    if (b16 != nullptr) {
        *b16 = false;
    }
    // f32 activations read in place or through an f16 copy; f16 ones (the down projection after the f16 result of the
    // fused gate/up) read in place
    if (ctx->device->mmid_f16reg_mode == 0 || !y_in_place || (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
        (d16 && glu_up == nullptr)) {
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
    // the shader reads b as vec4 from element offsets divided by 4: the rows are contiguous (y_f32_kernel) and K is a
    // multiple of 64, the strides of the batch dimensions must keep the 16-byte steps too
    if (src1->nb[2] % 16 != 0 || src1->nb[3] % 16 != 0) {
        return nullptr;
    }
    if (glu_up != nullptr && (glu_up->type != src0->type || !ggml_are_same_shape(glu_up, src0) || ggml_nbytes(glu_up) > range)) {
        return nullptr;
    }
    // the variants under test (08.11): GGML_VK_MMID_F16REG_GLU_SPLIT=1 - the fused gate/up with gate and up in separate
    // subgroups (64-row multiples); GGML_VK_MMID_F16REG_B16 / _GLU_B16 = 1 - the single matmuls / the fused gate/up read
    // an f16 copy of the activations that the caller makes once (32 also for the tiles)
    static const int bn_single = [] { const char * e = getenv("GGML_VK_MMID_F16REG_BN"); return e ? atoi(e) : 64; }();
    static const int bn_glu    = [] { const char * e = getenv("GGML_VK_MMID_F16REG_GLU_BN"); return e ? atoi(e) : 64; }();
    static const int rows2_env = [] { const char * e = getenv("GGML_VK_MMID_F16REG_ROWS2"); return e ? atoi(e) : -1; }();
    static const bool glu_split = [] { const char * e = getenv("GGML_VK_MMID_F16REG_GLU_SPLIT"); return e != nullptr && atoi(e) != 0; }();
    // the fused gate/up reads the f16 copy by default since 08.11 (the 395, profile-20261108-181557: 664.3 -> 612.6 ms of
    // a 4010-token ubatch with the copy); the single matmuls not (the down projection 400.8 -> 412.9: its 103 MB input
    // costs more to copy than its kernel gains)
    static const bool b16_single = [] { const char * e = getenv("GGML_VK_MMID_F16REG_B16"); return e != nullptr && atoi(e) != 0; }();
    static const bool b16_glu    = [] { const char * e = getenv("GGML_VK_MMID_F16REG_GLU_B16"); return e == nullptr || atoi(e) != 0; }();
    const bool rows2 = rows2_env > 0 || (rows2_env < 0 && src0->type == GGML_TYPE_Q5_1);
    const int g = glu_up != nullptr ? 1 : 0;
    int v = g ? (glu_split && src0->ne[1] % 64 == 0 ? 3 : 1) : (rows2 && src0->ne[1] % 256 == 0 ? 2 : 0);
    const int bn_want = g ? bn_glu : bn_single;
    int b = bn_want == 128 ? 1 : bn_want == 32 ? 2 : 0;
    int h = src1->type == GGML_TYPE_F16 || (g ? b16_glu : b16_single) ? 1 : 0;
    const int o = d16 ? 1 : 0;
    auto & pipes = ctx->device->pipeline_mmid_f16reg;
    if (!pipes[t][v][b][h][o] && v == 3) {
        v = 1;
    }
    if (!pipes[t][v][b][h][o] && v == 2) {
        v = 0;
    }
    if (!pipes[t][v][b][h][o] && b != 0) {
        b = 0;
    }
    vk_pipeline pipeline = pipes[t][v][b][h][o];
    if (b16 != nullptr) {
        *b16 = pipeline != nullptr && h != 0;
    }
    // the first choice of each pipeline, every one with GGML_VK_MMID_F16REG_LOG=1; on stderr as well with
    // GGML_SCHED_LOG_REALLOC, which 02 sets (llama-bench drops the ggml log)
    static const bool log_all = getenv("GGML_VK_MMID_F16REG_LOG") != nullptr;
    static const bool log_stderr = getenv("GGML_SCHED_LOG_REALLOC") != nullptr;
    static bool logged[4][4][3][2][2] = {};
    if (pipeline && (!logged[t][v][b][h][o] || log_all)) {
        logged[t][v][b][h][o] = true;
        GGML_LOG_INFO("ggml_vulkan: MUL_MAT_ID %s on %s (%s, %ld rows per expert)\n", ggml_type_name(src0->type), pipeline->name.c_str(),
                      g ? "fused gate/up" : "single", (long) n_per_expert);
        if (log_stderr) {
            fprintf(stderr, "ggml_vulkan: MUL_MAT_ID %s on %s (%s, %ld rows per expert)\n", ggml_type_name(src0->type), pipeline->name.c_str(),
                    g ? "fused gate/up" : "single", (long) n_per_expert);
        }
    }
    return pipeline;
}

// the buffer of a tensor read or written in place, false when its offset is off the alignment the shader needs
static bool ggml_vk_fork_tensor_buffer(const ggml_backend_vk_context * ctx, const ggml_tensor * t, uint64_t align, vk_subbuffer & out) {
    vk_buffer buffer = nullptr;
    size_t offset = 0;
    if (ctx->device->uma) {
        ggml_vk_host_get(ctx->device, t->data, buffer, offset);
    }
    if (!buffer) {
        const auto * buf_ctx = (const ggml_backend_vk_buffer_context *) t->buffer->context;
        buffer = buf_ctx->dev_buffer;
        offset = vk_tensor_offset(t) + t->view_offs;
    }
    align = std::max<uint64_t>(align, ctx->device->properties.limits.minStorageBufferOffsetAlignment);
    if (buffer == nullptr || offset % align != 0) {
        return false;
    }
    out = vk_subbuffer{ buffer, offset, ggml_nbytes(t) };
    return true;
}

// MUL_MAT of q8_0 weights on mul_mm_f16reg.comp (GGML_VK_MM_F16REG): f16 activations, f32 ones of K <= 4096 through an
// f16 copy, longer ones read in place; from GGML_VK_MM_F16REG_MIN_N tokens (64) and GGML_VK_MM_F16REG_MIN_M rows (512)
// (and K 1024 for f32 read in place), 2D, K a multiple of 64 (a block pair of a row on a word), the activations on 16
// bytes, nothing fused into the matmul
// hc_pre (MUL_MAT_DSV4_HC_PRE): the gated DSV4_HC_PRE that reads dst (through a reshape) takes the store, the result goes
// to its tensor and dst is not written. check_only: the answer without a dispatch (the fusion check)
bool ggml_vk_fork_mm_f16reg(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            const ggml_tensor * dst, const ggml_tensor * fused_dst, uint32_t fusion_flags, const ggml_tensor * hc_pre,
                            bool check_only) {
    const vk_device & device = ctx->device;
    if (device->mm_f16reg_mode == 0 || fused_dst != nullptr || fusion_flags != 0 || src0->type != GGML_TYPE_Q8_0 ||
        (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }
    static const int64_t min_n = [] { const char * e = getenv("GGML_VK_MM_F16REG_MIN_N"); return e ? (int64_t) atoll(e) : (int64_t) 64; }();
    // few rows over a long K (the hc down projection of qwen4exp, 320 x 10240) leave few workgroups, each a long walk
    // over K: there the int8 path was faster on the 395 (07.11, 05-mm: 17.2 against 15.8 TFLOP/s with f16 activations)
    static const int64_t min_m = [] { const char * e = getenv("GGML_VK_MM_F16REG_MIN_M"); return e ? (int64_t) atoll(e) : (int64_t) 512; }();
    // the f32 activations of K <= GGML_VK_MM_F16REG_B16_MAX_K (4096) go to an f16 copy in prealloc_y first, once for the
    // matmuls in a row that read the same tensor, and the f16 pipeline reads that (GGML_VK_MM_F16REG_B16=0: B read in
    // place as f32). on the 395 (08.11, profile-20261108-181557, ms of a 4010-token ubatch) the f32 B read in place was
    // what lost in the model on a short K: 10240 x 320 232.8 in place, 101.8 from the copy, 143.9 on the int8 path; the
    // big shapes 10240 x 2560 334.6 -> 299.6, 6144 x 2560 186.4 -> 161.4; 2560 x 6144 233.2 -> 250.8 (a copy of 98 MB of
    // one consumer costs more than the kernel gains), hence the bound on K
    static const bool b16_copy = [] { const char * e = getenv("GGML_VK_MM_F16REG_B16"); return e == nullptr || atoi(e) != 0; }();
    static const int64_t b16_max_k = [] { const char * e = getenv("GGML_VK_MM_F16REG_B16_MAX_K"); return e ? (int64_t) atoll(e) : (int64_t) 4096; }();
    // a short K read in place as f32: in the model the hc up projection (10240 x 320) took 2.43 ms a call against 1.49 on
    // the int8 path (profile-20261107-232325); with the f16 copy no bound (GGML_VK_MM_F16REG_MIN_K: for both)
    static const int64_t min_k_env = [] { const char * e = getenv("GGML_VK_MM_F16REG_MIN_K"); return e ? (int64_t) atoll(e) : (int64_t) -1; }();
    const int64_t K = src0->ne[0];
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const bool to_f16 = b16_copy && src1->type == GGML_TYPE_F32 && K <= b16_max_k && device->pipeline_mm_f16reg[1] && ggml_is_contiguous(src1);
    const int64_t min_k = min_k_env >= 0 ? min_k_env : to_f16 ? 0 : 1024;
    if (N < min_n || M < std::max<int64_t>(16, min_m) || K < min_k || K % 64 != 0 || src1->ne[0] != K || dst->ne[0] != M || dst->ne[1] != N ||
        src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    // A: rows of whole blocks, a row stride of whole block pairs; B: contiguous columns on 16-byte steps; D: f32 rows
    const size_t qs = ggml_type_size(src0->type);
    if (src0->nb[0] != qs || src0->nb[1] % (2 * qs) != 0 || src1->nb[0] != ggml_type_size(src1->type) || src1->nb[1] % 16 != 0 ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }
    const uint64_t range = device->properties.limits.maxStorageBufferRange;
    if (ggml_nbytes(src0) > range || ggml_nbytes(src1) > range || ggml_nbytes(dst) > range) {
        return false;
    }
    const int bi = src1->type == GGML_TYPE_F16 || to_f16 ? 1 : 0;
    vk_pipeline pipeline = device->pipeline_mm_f16reg[bi];
    vk_subbuffer a, b, d, x;
    // the DSV4_HC_PRE: x [E, 4, N] (f32 or f16, values contiguous) with M = 4 E, its result [E, N] f32
    const ggml_tensor * hc_x = hc_pre ? hc_pre->src[0] : nullptr;
    if (hc_pre) {
        const int64_t E = M / 4;
        if (M % 4 != 0 || hc_x->ne[0] != E || hc_x->ne[1] != 4 || hc_x->ne[2] != N || hc_x->ne[3] != 1 ||
            (hc_x->type != GGML_TYPE_F32 && hc_x->type != GGML_TYPE_F16) || hc_x->nb[0] != ggml_type_size(hc_x->type) ||
            hc_pre->type != GGML_TYPE_F32 || hc_pre->ne[0] != E || hc_pre->ne[1] != N || hc_pre->nb[0] != sizeof(float) ||
            ggml_nbytes(hc_x) > range || ggml_nbytes(hc_pre) > range) {
            return false;
        }
        pipeline = device->pipeline_mm_f16reg_hcpre[bi][hc_x->type == GGML_TYPE_F16 ? 1 : 0];
        if (!pipeline || !ggml_vk_fork_tensor_buffer(ctx, hc_x, 4, x)) {
            return false;
        }
    }
    if (!pipeline || !ggml_vk_fork_tensor_buffer(ctx, src0, 4, a) || !ggml_vk_fork_tensor_buffer(ctx, src1, to_f16 ? 4 : 16, b) ||
        !ggml_vk_fork_tensor_buffer(ctx, hc_pre ? hc_pre : dst, 4, d)) {
        return false;
    }
    if (check_only) {
        return true;
    }
    if (to_f16) {
        vk_pipeline cpy = ggml_vk_get_cpy_pipeline(ctx, src1, nullptr, GGML_TYPE_F16);
        if (!cpy) {
            return false;
        }
        const size_t y_sz = sizeof(ggml_fp16_t) * (size_t) ggml_nelements(src1);
        if (ctx->prealloc_size_y < y_sz) {
            ctx->prealloc_size_y = y_sz;
            ggml_vk_preallocate_buffers(ctx, subctx);
        }
        ggml_pipeline_request_descriptor_sets(ctx, cpy, 1);
        if (ctx->prealloc_y_last_pipeline_used != cpy.get() || ctx->prealloc_y_last_tensor_used != src1 || ctx->prealloc_y_last_k_padded) {
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
            ggml_vk_cpy_to_contiguous(ctx, subctx, cpy, src1, b, ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0));
            ctx->prealloc_y_last_pipeline_used = cpy.get();
            ctx->prealloc_y_last_tensor_used   = src1;
            ctx->prealloc_y_last_k_padded      = false;
        }
        ggml_vk_sync_buffers(ctx, subctx);
        b = vk_subbuffer{ ctx->prealloc_y, 0, y_sz };
        ctx->prealloc_y_need_sync = true;
    }

    vk_mat_mat_push_constants pc = {};
    pc.M           = (uint32_t) M;
    pc.N           = (uint32_t) N;
    pc.K           = (uint32_t) K;
    pc.stride_a    = (uint32_t) (src0->nb[1] / qs * ggml_blck_size(src0->type));
    pc.stride_b    = to_f16 ? (uint32_t) K : (uint32_t) (src1->nb[1] / ggml_type_size(src1->type));
    pc.stride_d    = (uint32_t) (dst->nb[1] / sizeof(float));
    pc.num_batches = 1;
    pc.k_split     = (uint32_t) K;
    pc.ne02        = 1;
    pc.ne12        = 1;
    pc.broadcast2  = 1;
    pc.broadcast3  = 1;
    pc.padded_N    = (uint32_t) N;

    // the first use of each pipeline, every one with GGML_VK_MM_F16REG_LOG=1; on stderr as well with GGML_SCHED_LOG_REALLOC
    static const bool log_all    = getenv("GGML_VK_MM_F16REG_LOG") != nullptr;
    static const bool log_stderr = getenv("GGML_SCHED_LOG_REALLOC") != nullptr;
    static bool logged[2] = {};
    if (!logged[bi] || log_all) {
        logged[bi] = true;
        GGML_LOG_INFO("ggml_vulkan: MUL_MAT q8_0 %ld x %ld x %ld on %s\n", (long) M, (long) N, (long) K, pipeline->name.c_str());
        if (log_stderr) {
            fprintf(stderr, "ggml_vulkan: MUL_MAT q8_0 %ld x %ld x %ld on %s\n", (long) M, (long) N, (long) K, pipeline->name.c_str());
        }
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);
    if (hc_pre) {
        vk_fork_mm_hc_pre_push_constants hpc = {};
        hpc.mm       = pc;
        hpc.hc_e     = (uint32_t) (M / 4);
        hpc.hc_x1    = (uint32_t) (hc_x->nb[1] / ggml_type_size(hc_x->type));
        hpc.hc_x2    = (uint32_t) (hc_x->nb[2] / ggml_type_size(hc_x->type));
        hpc.hc_d1    = (uint32_t) (hc_pre->nb[1] / sizeof(float));
        hpc.hc_scale = ggml_get_op_params_f32(hc_pre, 0);
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d, x }, hpc, { (uint32_t) M, (uint32_t) N, 1 });
        return true;
    }
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { a, b, d }, pc, { (uint32_t) M, (uint32_t) N, 1 });
    return true;
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
