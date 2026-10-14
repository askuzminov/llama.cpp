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

int ggml_vk_fork_dmmv_wg() {
    // GGML_VK_DMMV_WG=large|subgroup: the workgroup of every dequant mat-vec (and its MUL_MAT_ID form) instead of the
    // upstream choice, which gives the large one (4 subgroups, hybrid reduction) only on NVIDIA and Intel for a small M.
    // the expert mat-vec of qwen4exp is small M (640 rows of gate/up, 2560 of down with 640 columns) and runs at
    // 164-171 GB/s on the 395 against ~230 of the dense ones (05 of 14.10, one token)
    static const int mode = [] {
        const char * env = getenv("GGML_VK_DMMV_WG");
        if (env == nullptr) {
            return -1;
        }
        return strcmp(env, "large") == 0 ? 1 : strcmp(env, "subgroup") == 0 ? 0 : -1;
    }();
    return mode;
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

uint32_t ggml_vk_fork_fa_sparse_split(uint32_t base_wgs, uint32_t shader_core_count, bool & kv_range) {
    // the sparse prefill with the heads folded is one workgroup per token and KV head, each walking its own list of
    // 2052 cells from the start to the end. on the 395 at depth 122880 (12.10, 4096 tokens) the 12 qwen4exp attention
    // layers take 896 ms against 302 at depth 0, about 17 GB of K/V per layer at 224 GB/s, the DRAM rate: neighbouring
    // tokens pick almost the same cells, but the workgroups running at a time are at different points of their walks
    // and the cache does not hold the stretch between them. split into parts and run part 0 of every token first,
    // the walks of the running workgroups cover one part of KV. costs the split_k partials (101 MB per part at 4096
    // tokens, written and read once). GGML_VK_FA_SPARSE_SPLIT: the number of parts, 0 = upstream (one per token)
    static const uint32_t parts = [] {
        const char * env = getenv("GGML_VK_FA_SPARSE_SPLIT");
        return env ? (uint32_t) std::max(0, atoi(env)) : 0u;
    }();
    // 12.10 night on the 395: no gain at 4 and 8 parts, the parts of the lists of neighbouring tokens do not cover
    // one stretch of KV. GGML_VK_FA_SPARSE_KVSPLIT: the number of KV ranges instead, part s of every token takes the
    // listed cells of range s (the slots found by binary search), so all tokens walk one range at a time: at 122880
    // cells and 8 ranges 16 MB of K/V per head, against 17 GB of gathers per layer and ubatch that read each cell 66
    // times. the partials cost as above. takes precedence over GGML_VK_FA_SPARSE_SPLIT
    static const uint32_t ranges = [] {
        const char * env = getenv("GGML_VK_FA_SPARSE_KVSPLIT");
        return env ? (uint32_t) std::max(0, atoi(env)) : 0u;
    }();
    // fewer workgroups than that: upstream splits for occupancy already (decode, short batches)
    if (base_wgs < shader_core_count * 2) {
        return 0;
    }
    kv_range = ranges > 1;
    return ranges > 1 ? ranges : parts > 1 ? parts : 0u;
}

uint32_t ggml_vk_fork_fa_ablate() {
    // the sparse attention at depth on the 395 (13.10) did not move with the K staging (wide), the order of the
    // parts (split-major) or ranges of KV (kvsplit), so these leave parts out of flash_attn_cm1.comp and the profile
    // shows what each costs: 1 no block loop (the pre-pass, Q and the stores remain), 2 no K/V reads, 4 no mask, 8 no
    // matmuls, 16 no softmax. the results are wrong: profiling only. GGML_VK_FA_ABLATE=<sum of these>
    static const uint32_t bits = [] {
        const char * env = getenv("GGML_VK_FA_ABLATE");
        return env ? (uint32_t) atoi(env) & 31u : 0u;
    }();
    return bits;
}

bool ggml_vk_fork_fa_sparse_wide(const vk_device & device, const vk_fa_tuning_params & params, uint32_t hsk, uint32_t hsv,
                                 bool f32acc, ggml_type k_type, ggml_type v_type) {
    // the sparse folded tile of flash_attn_cm1.comp stages K 16 dims at a time, with two barriers and a lookup of the
    // cell in the list per staged element: 32 barriers per block of 64 cells at head size 256. on the 395 (12.10 night)
    // it computes the attention of a 4096 prompt at 122880 in 898 ms against 302 for the dense one at depth 0, about the
    // same work, and neither the split-major order nor the union tile gained (the reads are not what holds it).
    // SPARSE_WIDE: the cells of the block go to shared memory once, K in chunks of 64 dims (8 barriers per block).
    // GGML_VK_FA_SPARSE_WIDE: 1 = on, unset or 0 = off (until measured)
    static const int mode = [] {
        const char * env = getenv("GGML_VK_FA_SPARSE_WIDE");
        return env ? atoi(env) : 0;
    }();
    if (mode != 1 || params.path != FA_COOPMAT1 || params.shmem_staging || hsk % 64 != 0 ||
        k_type != GGML_TYPE_F16 || v_type != GGML_TYPE_F16) {
        return false;
    }
    // the shared memory of the coopmat1 tile as ggml_vk_flash_attn_coopmat_shmem_support counts it (kept in step with
    // it), with the wider K rows and the cells
    const uint32_t Br = params.block_rows;
    const uint32_t Bc = params.block_cols;
    const uint32_t row_split = Bc / 16;
    const uint32_t hsk_pad = (hsk + 15) & ~15u;
    const uint32_t acctype = f32acc ? 4 : 2;
    const uint32_t f16vec4 = 8;
    const uint32_t tmpsh = row_split * sizeof(float);
    const uint32_t iq_shmem = 16 * sizeof(ggml_fp16_t);
    const uint32_t Qf = Br * (hsk_pad / 4 + 2) * f16vec4;
    const uint32_t Psh = Bc * (Br / 4 + 2) * f16vec4;
    const uint32_t sfsh = Bc * ((hsk <= 128) ? (Br + 8) : Br) * acctype;
    const uint32_t ksh = std::max<uint32_t>(Bc * std::max<uint32_t>(16 / 4 + 2, 16 / 4 * row_split), Bc * (64 / 4 + 2)) * f16vec4;
    const uint32_t pvsh = 16 * (params.row_split * 16 / 4) * f16vec4;
    const uint32_t slope = Br * acctype;
    const uint32_t cellsh = Bc * sizeof(int32_t);
    GGML_UNUSED(hsv);
    return tmpsh + iq_shmem + Qf + Psh + sfsh + ksh + pvsh + slope + cellsh <= device->properties.limits.maxComputeSharedMemorySize;
}

bool ggml_vk_fork_fa_union(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * q, const ggml_tensor * k,
                           const ggml_tensor * v, const ggml_tensor * mask, const ggml_tensor * sinks, ggml_tensor * dst) {
    // a tile of R neighbouring tokens x G heads of one KV head walks the union of the token selections, see
    // flash_attn_union.comp. it takes the case of the folded sparse tile (flash_attn_cm1.comp): f16 K/V, a mask with the
    // n_kv_max hint, KV past the sparse threshold; and only where the tiles alone fill the device (prompts), the folded
    // tile keeps decode, draft checks and short batches
    const vk_device & device = ctx->device;
    if (device->fa_union_mode == 0 || mask == nullptr || sinks != nullptr || dst->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }
    const int32_t n_kv_max = ggml_get_op_params_i32(dst, 4);
    float scale, max_bias, softcap;
    memcpy(&scale,    (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&softcap,  (const float *) dst->op_params + 2, sizeof(float));
    if (n_kv_max <= 0 || max_bias != 0.0f || softcap != 0.0f) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || mask->type != GGML_TYPE_F16 ||
        dst->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t HSK = q->ne[0];
    const int64_t KV  = k->ne[1];
    const int64_t N   = q->ne[1];
    const int64_t n_head = q->ne[2];
    const int64_t nek2 = k->ne[2];
    if (k->ne[0] != HSK || v->ne[0] != HSK || HSK % 64 != 0 || HSK > 256 || v->ne[1] != KV || mask->ne[0] != KV ||
        v->ne[2] != nek2 || n_head % nek2 != 0 || mask->ne[2] != 1 || mask->ne[1] < N) {
        return false;
    }
    // 48 rows: R tokens x G heads, 2 to 8 tokens
    const int64_t G = n_head / nek2;
    if (48 % G != 0 || 48 / G < 2 || 48 / G > 8) {
        return false;
    }
    const uint32_t R = (uint32_t) (48 / G);
    // as the sparse path of flash_attn_cm1.comp with f16 K/V
    if (KV < std::max<int64_t>(4096, 2 * (int64_t) n_kv_max)) {
        return false;
    }
    const int64_t neq3 = q->ne[3];
    if (neq3 % k->ne[3] != 0 || neq3 % v->ne[3] != 0 || neq3 % mask->ne[3] != 0) {
        return false;
    }
    // element strides, rows on 16-byte boundaries for the vec4 loads
    if (q->nb[0] != sizeof(float) || k->nb[0] != sizeof(ggml_fp16_t) || v->nb[0] != sizeof(ggml_fp16_t) ||
        mask->nb[0] != sizeof(ggml_fp16_t) || (q->nb[1] | q->nb[2] | q->nb[3]) % 16 != 0 ||
        (k->nb[1] | k->nb[2] | k->nb[3] | v->nb[1] | v->nb[2] | v->nb[3]) % 8 != 0 || !ggml_is_contiguous(dst) ||
        dst->ne[0] != HSK || dst->ne[1] != n_head || dst->ne[2] != N) {
        return false;
    }
    const uint32_t n_tiles = (uint32_t) ((N + R - 1) / R);
    const uint32_t cores = device->shader_core_count ? device->shader_core_count : 16;
    if ((uint64_t) n_tiles * nek2 * neq3 < 2ull * cores) {
        return false;
    }
    // GGML_VK_FA_UNION_BC: list slots per step, 64 (default) or 32
    static const uint32_t Bc = [] {
        const char * env = getenv("GGML_VK_FA_UNION_BC");
        return env != nullptr && atoi(env) == 32 ? 32u : 64u;
    }();
    const uint32_t sg = device->subgroup_size;
    const uint64_t shmem = std::max<uint64_t>(Bc, 48) * (64 + 8) * 2 + 48ull * (std::max<uint64_t>(Bc, 64) + 4) * 4 +
                           48ull * (Bc + 8) * 2 + 8ull * Bc * 2 + Bc * 4ull + 3ull * sg * 4 +
                           (device->fa_union_mode == 2 ? 48ull * (64 + 8) * 2 : 0); // emulated: Q staged too
    if (shmem > device->properties.limits.maxComputeSharedMemorySize) {
        return false;
    }

    vk_pipeline pipeline;
    {
        std::lock_guard<std::mutex> guard(device->compile_mutex);
        vk_pipeline & p = device->pipeline_fa_union[{ (uint32_t) HSK, Bc }];
        if (!p) {
            p = std::make_shared<vk_pipeline_struct>();
        }
        pipeline = p;
    }
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    // the union lists: the sparse compact pass with R mask rows per group, ascending, -1 padded
    const uint32_t nem1 = (uint32_t) mask->ne[1];
    const uint32_t nem3 = (uint32_t) mask->ne[3];
    const uint32_t n_groups = (nem1 + R - 1) / R;
    const uint32_t capacity = (uint32_t) std::min<int64_t>((int64_t) R * n_kv_max, KV);
    const uint64_t list_size = sizeof(int32_t) * (uint64_t) capacity * n_groups * nem3;
    vk_pipeline compact = device->fa_sparse_compact_use_subgroups ? device->pipeline_fa_sparse_compact_subgroup
                                                                  : device->pipeline_fa_sparse_compact;
    ggml_pipeline_request_descriptor_sets(ctx, compact, 1);
    if (ctx->prealloc_size_y < list_size) {
        ctx->prealloc_size_y = list_size;
        ggml_vk_preallocate_buffers(ctx, subctx);
    }
    if (ctx->prealloc_y_need_sync) {
        ggml_vk_sync_buffers(ctx, subctx);
    }

    vk_subbuffer q_buf    = ggml_vk_tensor_subbuffer(ctx, q);
    vk_subbuffer k_buf    = ggml_vk_tensor_subbuffer(ctx, k);
    vk_subbuffer v_buf    = ggml_vk_tensor_subbuffer(ctx, v);
    vk_subbuffer mask_buf = ggml_vk_tensor_subbuffer(ctx, mask);
    vk_subbuffer dst_buf  = ggml_vk_tensor_subbuffer(ctx, dst);
    vk_subbuffer list_buf = ggml_vk_subbuffer(ctx, ctx->prealloc_y, 0);
    // the lists overwrite a matmul input converted into prealloc_y
    ctx->prealloc_y_last_pipeline_used = nullptr;
    ctx->prealloc_y_last_tensor_used = nullptr;

    const vk_op_flash_attn_sparse_compact_push_constants sc_pc = {
        (uint32_t) KV, nem1, 1,
        (uint32_t) (mask->nb[1] / sizeof(ggml_fp16_t)),
        (uint32_t) (mask->nb[2] / sizeof(ggml_fp16_t)),
        (uint32_t) (mask->nb[3] / sizeof(ggml_fp16_t)),
        (uint32_t) n_kv_max, R, capacity,
    };
    ggml_vk_dispatch_pipeline(ctx, subctx, compact, { mask_buf, list_buf }, sc_pc, { n_groups, 1, nem3 });
    ggml_vk_sync_buffers(ctx, subctx);

    const vk_op_fa_union_push_constants pc = {
        (uint32_t) N, (uint32_t) KV, (uint32_t) n_head, (uint32_t) G, R,
        (uint32_t) (q->nb[1] / sizeof(float)), (uint32_t) (q->nb[2] / sizeof(float)), (uint32_t) (q->nb[3] / sizeof(float)),
        (uint32_t) (k->nb[1] / sizeof(ggml_fp16_t)), (uint32_t) (k->nb[2] / sizeof(ggml_fp16_t)), (uint32_t) (k->nb[3] / sizeof(ggml_fp16_t)),
        (uint32_t) (v->nb[1] / sizeof(ggml_fp16_t)), (uint32_t) (v->nb[2] / sizeof(ggml_fp16_t)), (uint32_t) (v->nb[3] / sizeof(ggml_fp16_t)),
        (uint32_t) (mask->nb[1] / sizeof(ggml_fp16_t)), (uint32_t) (mask->nb[3] / sizeof(ggml_fp16_t)), nem3,
        (uint32_t) (neq3 / k->ne[3]), (uint32_t) (neq3 / v->ne[3]),
        capacity, n_groups, scale,
    };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { q_buf, k_buf, v_buf, mask_buf, list_buf, dst_buf }, pc,
                              { n_tiles, (uint32_t) nek2, (uint32_t) neq3 });
    ctx->prealloc_y_need_sync = true;
    return true;
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
