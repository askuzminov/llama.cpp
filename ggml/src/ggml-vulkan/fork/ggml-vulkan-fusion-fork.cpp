// the fusions of the fork: checks, dispatchers, the hooks of the graph_compute fusion chain and the node order of
// ggml_vk_graph_optimize (see ggml-vulkan-fork.h)

#include "ggml-vulkan-fork.h"

// as in ggml-vulkan.cpp
static uint32_t ggml_vk_nb_elem(const ggml_tensor * t, int i) {
    return (uint32_t)(t->nb[i] / ggml_type_size(t->type));
}

// the empty nodes of ggml_vk_graph_optimize (is_empty there)
static bool ggml_vk_graph_node_is_empty(const ggml_tensor * node) {
    return node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
}

// nodes: scale, sigmoid, scale, dsv4_hc_post, rms_norm, mul (hc_post_norm_pattern), then cpy to f16 when
// fused_hc_post_norm_f16 (hc_post_norm_f16_pattern): the normalized streams go to the cpy in f16
void ggml_vk_dsv4_hc_post_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * scale_in = cgraph->nodes[node_idx];
    const ggml_tensor * hc_post  = cgraph->nodes[node_idx + 3];
    const ggml_tensor * norm     = cgraph->nodes[node_idx + 4];
    const ggml_tensor * mul      = ctx->fused_hc_post_norm_f16 ? cgraph->nodes[node_idx + 6] : cgraph->nodes[node_idx + 5];

    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * p_src    = scale_in->src[0];

    // the gamma is the second source of the mul, whether the cpy follows it or not
    const ggml_tensor * w        = cgraph->nodes[node_idx + 5]->src[1];

    vk_pipeline pipeline = ctx->fused_hc_post_norm_mix ? ctx->device->pipeline_dsv4_hc_post_norm_mix :
                           ctx->fused_hc_post_norm_f16 ? ctx->device->pipeline_dsv4_hc_post_norm_f16n : ctx->device->pipeline_dsv4_hc_post_norm_f32;
    GGML_ASSERT(pipeline != nullptr);

    const uint32_t n_embd   = (uint32_t)x->ne[0];
    const uint32_t n_tokens = (uint32_t)x->ne[1];

    // HC_POST_NORM_MIX: the hc down that reads the f16 streams on the int8 path finds its q8_1 blocks in prealloc_y
    // (the cache of ggml_vk_mul_mat_q_f16 by quantize pipeline and tensor), the shader writes them.
    // GGML_VK_HC_MIX_Q8=0: the matmul quantizes them itself
    const ggml_tensor * xn   = ctx->fused_hc_post_norm_mix ? cgraph->nodes[node_idx + 7] : nullptr;
    const ggml_tensor * down = nullptr;
    uint64_t q_sz = 0;
    if (xn) {
        static const bool q8_on = [] { const char * e = getenv("GGML_VK_HC_MIX_Q8"); return e == nullptr || atoi(e) != 0; }();
        if (q8_on && n_embd % 128 == 0) {
            const int end = std::min(cgraph->n_nodes, node_idx + (int) hc_post_norm_mix_pattern.size() + 24);
            for (int j = node_idx + (int) hc_post_norm_mix_pattern.size(); j < end && down == nullptr; ++j) {
                const ggml_tensor * n = cgraph->nodes[j];
                if (n->op == GGML_OP_MUL_MAT && n->src[1] == xn && ggml_vk_fork_mul_mat_quantizes_y_x4(ctx, n)) {
                    down = n;
                }
            }
        }
        if (down) {
            // the size ggml_vk_mul_mat_q_f16 takes for the blocks: 144 bytes per 128 values
            q_sz = (uint64_t) ((ggml_nelements(xn) + 127) / 128) * 144;
            if (ctx->prealloc_size_y < q_sz) {
                ctx->prealloc_size_y = q_sz;
                ggml_vk_preallocate_buffers(ctx, subctx);
            }
            if (ctx->prealloc_y_need_sync) {
                ggml_vk_sync_buffers(ctx, subctx);
            }
        }
    }

    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const vk_subbuffer x_buf = ggml_vk_tensor_subbuffer(ctx, x,        true);
    const vk_subbuffer r_buf = ggml_vk_tensor_subbuffer(ctx, residual, true);
    const vk_subbuffer p_buf = ggml_vk_tensor_subbuffer(ctx, p_src,    true);
    const vk_subbuffer w_buf = ggml_vk_tensor_subbuffer(ctx, w,        true);
    const vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, hc_post,  true);
    const vk_subbuffer n_buf = ggml_vk_tensor_subbuffer(ctx, mul,      true);

    vk_op_dsv4_hc_post_norm_push_constants pc = {
        n_embd, n_tokens,
        ggml_vk_nb_elem(x, 1),
        ggml_vk_nb_elem(residual, 1), ggml_vk_nb_elem(residual, 2),
        ggml_vk_nb_elem(p_src, 0), ggml_vk_nb_elem(p_src, 1),
        w->ne[1] == 1 ? 0 : ggml_vk_nb_elem(w, 1), w->ne[2] == 1 ? 0 : ggml_vk_nb_elem(w, 2),
        ggml_vk_nb_elem(hc_post, 1), ggml_vk_nb_elem(hc_post, 2),
        ggml_vk_nb_elem(mul, 1), ggml_vk_nb_elem(mul, 2),
        get_misalign_bytes(ctx, x)        / (uint32_t) sizeof(float),
        get_misalign_bytes(ctx, residual) / (uint32_t) sizeof(float),
        get_misalign_bytes(ctx, p_src)    / (uint32_t) sizeof(float),
        get_misalign_bytes(ctx, w)        / (uint32_t) sizeof(float),
        get_misalign_bytes(ctx, hc_post)  / (uint32_t) sizeof(float),
        get_misalign_bytes(ctx, mul)      / (uint32_t) ggml_type_size(mul->type),
        1u,
        ggml_get_op_params_f32(scale_in, 0),
        ggml_get_op_params_f32(hc_post->src[2], 0),
        ggml_get_op_params_f32(norm, 0),
    };

    if (xn) {
        const ggml_tensor * mm   = cgraph->nodes[node_idx + 8];
        const ggml_tensor * cont = cgraph->nodes[node_idx + 10];
        const ggml_tensor * wi   = mm->src[1];
        const vk_subbuffer wi_buf  = ggml_vk_tensor_subbuffer(ctx, wi,   true);
        const vk_subbuffer inj_buf = ggml_vk_tensor_subbuffer(ctx, cont, true);
        const vk_subbuffer q_buf   = down ? vk_subbuffer{ ctx->prealloc_y, 0, q_sz } : n_buf;
        const vk_fork_hc_post_norm_mix_push_constants mpc = {
            pc,
            ggml_vk_nb_elem(wi, 1),
            get_misalign_bytes(ctx, wi)   / (uint32_t) sizeof(float),
            get_misalign_bytes(ctx, cont) / (uint32_t) sizeof(float),
            down ? 1u : 0u,
        };
        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, w_buf, d_buf, n_buf, wi_buf, inj_buf, q_buf }, mpc, { n_tokens, 1, 1 });
        if (down) {
            ctx->prealloc_y_last_pipeline_used = ctx->device->pipeline_quantize_q8_1_x4_f16.get();
            ctx->prealloc_y_last_tensor_used   = xn;
            ctx->prealloc_y_last_k_padded      = false;
            ctx->prealloc_y_need_sync          = true;
        }
        return;
    }

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { x_buf, r_buf, p_buf, w_buf, d_buf, n_buf }, pc, { 4, n_tokens, 1 });
}

static bool ggml_vk_can_fuse_hc_post_norm(const struct ggml_cgraph * cgraph, int node_idx) {
    if (!ggml_vk_can_fuse_hc_post_gate(cgraph, node_idx)) {
        return false;
    }
    const ggml_tensor * hc_post  = cgraph->nodes[node_idx + 3];
    const ggml_tensor * norm     = cgraph->nodes[node_idx + 4];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 5];
    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * w        = mul->src[1];

    // GGML_VK_HC_POST_NORM=0 keeps HC_POST_GATE and RMS_NORM_MUL apart, for A/B runs
    static const bool enabled = [] { const char * env = getenv("GGML_VK_HC_POST_NORM"); return env == nullptr || atoi(env) != 0; }();
    // the shader runs a workgroup per token and stream: on a decode batch it took TG down 1.6 percent (07.10, the
    // 395), the two shaders apart are faster there
    static const int64_t min_tokens = [] { const char * env = getenv("GGML_VK_HC_POST_NORM_MIN_TOKENS"); return env ? (int64_t) atoi(env) : 32; }();
    if (!enabled || hc_post->ne[2] < min_tokens) {
        return false;
    }

    // the shader keeps 8 columns per invocation (512 invocations) and reads rows as contiguous
    const auto rows = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float); };
    return hc_post->src[3] == nullptr && hc_post->ne[1] == 4 && hc_post->ne[3] == 1 &&
           hc_post->ne[0] <= 8*512 &&
           rows(x) && rows(residual) && rows(hc_post) && rows(norm) && rows(mul) && rows(w) &&
           ggml_are_same_shape(mul, hc_post) &&
           w->ne[0] == hc_post->ne[0] && (w->ne[1] == 1 || w->ne[1] == 4) &&
           (w->ne[2] == 1 || w->ne[2] == hc_post->ne[2]) && w->ne[3] == 1;
}

// MUL_MAT_ID, MUL_MAT_ID, GLU: the experts' up and gate projections (either first) and swiglu over both, run as
// one int8 coopmat matmul_id with two A matrices and silu(gate) * up as its output (FUSED_GLU in mul_mmq_cm1.comp).
// it needs the int8 path of the experts, the same input and ids for both, weights of one type and layout, and a
// batch that goes to matmul_id (decode takes mat-vec)
// MUL_MAT + RELU: mul_mm.comp rectifies the result at the store (fusion_flags bit 0), for example the per-head
// indexer scores of qwen4exp. only where ggml_vk_mul_mat takes the matrix path through mul_mm.comp: a float A (the
// coopmat2 shader has no such store) and more columns than the mat-vec takes. GGML_VK_DISABLE_MM_RELU turns it off
static bool ggml_vk_can_fuse_mm_relu(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    static const bool disabled = getenv("GGML_VK_DISABLE_MM_RELU") != nullptr;
    if (disabled || ctx->device->coopmat2) {
        return false;
    }
    if (!ggml_can_fuse(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_UNARY })) {
        return false;
    }
    const ggml_tensor * mm   = cgraph->nodes[node_idx];
    const ggml_tensor * relu = cgraph->nodes[node_idx + 1];
    if (ggml_get_unary_op(relu) != GGML_UNARY_OP_RELU || relu->src[0] != mm) {
        return false;
    }
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) ||
        (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_F16) ||
        mm->type != GGML_TYPE_F32 || relu->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(mm) || !ggml_is_contiguous(relu) || !ggml_are_same_shape(mm, relu)) {
        return false;
    }
    // the other paths of ggml_vk_mul_mat: mat-vec, one output row, permuted operands, a split A, the FWHT
    if (mm->ne[0] == 1 || mm->ne[1] <= mul_mat_vec_max_cols || ggml_is_permuted(src0) || ggml_is_permuted(src1) ||
        ggml_nbytes(src0) > ctx->device->properties.limits.maxStorageBufferRange || ggml_vk_can_use_fwht(ctx, src1, mm)) {
        return false;
    }
    return true;
}

// MUL_MAT_HEADSUM: the scores of the qwen4exp indexer heads (src/models/qwen4exp.cpp, build_qsa_top_k) in one matmul. for
// each head a MUL_MAT of the same A by a view of the rows of that head in q (the heads of a token are adjacent rows) and
// a RELU, then the adds that sum the heads in head order, and the add of a bias of the same shape if there is one.
// mul_mm.comp (MUL_MAT_HEADSUM) multiplies A by all rows of q and sums the rectified heads of a token at the store
static constexpr int HEADSUM_HEADS = 4;

// the key pool of QSA is f16 by default, f32 with LLAMA_QSA_POOL_F32 (src/fork/llama-memory-hybrid-idx-fork.cpp)
static const std::vector<vk_matmul_pipeline_pair> * ggml_vk_headsum_pipelines(const ggml_backend_vk_context * ctx, ggml_type type_a) {
    auto it = ctx->device->pipeline_matmul.find(vk_matmul_pipeline_key{type_a, GGML_TYPE_F32, false, false, false, true});
    return it == ctx->device->pipeline_matmul.end() || it->second.empty() ? nullptr : &it->second;
}

// finds the nodes of the pattern in [start, end), the first matmul at start, nodes marked in skip left out. idxs gets
// them in this order: the matmul and relu of each head, the adds of the sum, the bias add. returns the node count
static int ggml_vk_find_headsum(const ggml_cgraph * cgraph, int start, int end, const std::vector<uint8_t> * skip, int * idxs) {
    const ggml_tensor * mm0 = cgraph->nodes[start];
    if (mm0->op != GGML_OP_MUL_MAT || (mm0->src[0]->type != GGML_TYPE_F32 && mm0->src[0]->type != GGML_TYPE_F16) ||
        mm0->src[1]->type != GGML_TYPE_F32 ||
        mm0->src[1]->view_src == nullptr) {
        return 0;
    }
    end = std::min(end, cgraph->n_nodes);

    auto const &find = [&](const auto & pred) -> int {
        for (int j = start + 1; j < end; ++j) {
            if ((!skip || !(*skip)[j]) && pred(cgraph->nodes[j])) {
                return j;
            }
        }
        return -1;
    };

    const ggml_tensor * q0 = mm0->src[1];
    const size_t row = q0->ne[0] * ggml_element_size(q0);

    int n = 0;
    for (int h = 0; h < HEADSUM_HEADS; ++h) {
        const int mi = h == 0 ? start : find([&](const ggml_tensor * t) {
            return t->op == GGML_OP_MUL_MAT && t->src[0] == mm0->src[0] && t->src[1]->view_src == q0->view_src &&
                   t->src[1]->view_offs == q0->view_offs + h*row && ggml_are_same_shape(t->src[1], q0) && ggml_are_same_stride(t->src[1], q0);
        });
        if (mi < 0) {
            return 0;
        }
        const ggml_tensor * mm = cgraph->nodes[mi];
        const int ri = find([&](const ggml_tensor * t) {
            return t->op == GGML_OP_UNARY && t->src[0] == mm && ggml_get_unary_op(t) == GGML_UNARY_OP_RELU;
        });
        if (ri < 0) {
            return 0;
        }
        idxs[n++] = mi;
        idxs[n++] = ri;
    }

    const ggml_tensor * sum = cgraph->nodes[idxs[1]];
    for (int h = 1; h < HEADSUM_HEADS; ++h) {
        const ggml_tensor * relu = cgraph->nodes[idxs[2*h + 1]];
        const int ai = find([&](const ggml_tensor * t) {
            return t->op == GGML_OP_ADD && ((t->src[0] == sum && t->src[1] == relu) || (t->src[0] == relu && t->src[1] == sum));
        });
        if (ai < 0) {
            return 0;
        }
        idxs[n++] = ai;
        sum = cgraph->nodes[ai];
    }

    const int bi = find([&](const ggml_tensor * t) {
        return t->op == GGML_OP_ADD && (t->src[0] == sum) != (t->src[1] == sum);
    });
    if (bi >= 0) {
        idxs[n++] = bi;
    }
    return n;
}

// fused when the pattern nodes and empty nodes (views) between them make a run that starts at node_idx. returns the
// length of the run (0: not fused) and the bias tensor (nullptr: no bias add). GGML_VK_DISABLE_MM_HEADSUM turns it off
static int ggml_vk_can_fuse_mul_mat_headsum(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, const ggml_tensor ** bias) {
    static const bool disabled = getenv("GGML_VK_DISABLE_MM_HEADSUM") != nullptr;
    // tokens per stream below which the heads stay apart: a few tokens take the matrix-vector kernel per head, which
    // the fused matmul replaces with a tile of 4 columns. GGML_VK_HEADSUM_MIN_TOKENS, 0 fuses at any size
    static const int64_t min_tokens = [] {
        const char * env = getenv("GGML_VK_HEADSUM_MIN_TOKENS");
        return env ? (int64_t) atoi(env) : 0;
    }();
    if (disabled || ctx->device->coopmat2 || cgraph->nodes[node_idx]->op != GGML_OP_MUL_MAT ||
        !ggml_vk_headsum_pipelines(ctx, cgraph->nodes[node_idx]->src[0]->type)) {
        return 0;
    }
    if (cgraph->nodes[node_idx]->src[1]->ne[1] < min_tokens) {
        return 0;
    }

    int idxs[3*HEADSUM_HEADS];
    const int n = ggml_vk_find_headsum(cgraph, node_idx, node_idx + 32, nullptr, idxs);
    if (n == 0) {
        return 0;
    }

    // the output is the last node of the run, every other node of the run is a pattern node or empty
    const int last = idxs[n - 1];
    for (int j = node_idx; j <= last; ++j) {
        if (std::find(idxs, idxs + n, j) == idxs + n && !ggml_op_is_empty(cgraph->nodes[j]->op)) {
            return 0;
        }
    }
    for (int k = 0; k < n - 1; ++k) {
        if (idxs[k] > last) {
            return 0;
        }
        // the inner nodes are read only by the next node of the pattern
        const ggml_tensor * t = cgraph->nodes[idxs[k]];
        if ((t->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_node_has_n_uses(cgraph, idxs[k], 1)) {
            return 0;
        }
    }

    const ggml_tensor * mm0 = cgraph->nodes[node_idx];
    const ggml_tensor * a   = mm0->src[0];
    const ggml_tensor * q0  = mm0->src[1];
    const ggml_tensor * dst = cgraph->nodes[last];

    for (int k = 0; k < n; ++k) {
        const ggml_tensor * t = cgraph->nodes[idxs[k]];
        if (t->type != GGML_TYPE_F32 || !ggml_are_same_shape(t, dst)) {
            return 0;
        }
    }
    // A: one matrix of contiguous rows; q: the heads of a token are adjacent rows, one matrix
    if ((a->type != GGML_TYPE_F32 && a->type != GGML_TYPE_F16) || q0->type != GGML_TYPE_F32 || !ggml_is_contiguous(dst) ||
        a->nb[0] != ggml_type_size(a->type) || a->nb[1] % ggml_type_size(a->type) != 0 || a->ne[2] != 1 || a->ne[3] != 1 ||
        q0->nb[0] != sizeof(float) || q0->nb[1] != HEADSUM_HEADS * q0->ne[0] * sizeof(float) || q0->ne[2] != 1 || q0->ne[3] != 1 ||
        q0->view_offs + q0->ne[1] * q0->nb[1] > ggml_nbytes(q0->view_src)) {
        return 0;
    }

    const ggml_tensor * b = nullptr;
    if (n == 3*HEADSUM_HEADS) {
        const ggml_tensor * add = cgraph->nodes[last];
        b = add->src[0] == cgraph->nodes[idxs[n - 2]] ? add->src[1] : add->src[0];
        if (b->type != GGML_TYPE_F32 || !ggml_are_same_shape(b, dst) || !ggml_are_same_stride(b, dst) || get_misalign_bytes(ctx, b)) {
            return 0;
        }
    }

    const uint64_t max_range = ctx->device->properties.limits.maxStorageBufferRange;
    if (get_misalign_bytes(ctx, a) || get_misalign_bytes(ctx, q0) || get_misalign_bytes(ctx, dst) ||
        ggml_nbytes(a) > max_range || q0->ne[1] * q0->nb[1] > max_range || ggml_nbytes(dst) > max_range) {
        return 0;
    }

    *bias = b;
    return last - node_idx + 1;
}

void ggml_vk_mul_mat_headsum(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * mm0  = cgraph->nodes[node_idx];
    const ggml_tensor * a    = mm0->src[0];
    const ggml_tensor * q0   = mm0->src[1];
    const ggml_tensor * dst  = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];
    const ggml_tensor * bias = ctx->fused_mm_headsum_bias;

    const uint32_t m        = (uint32_t) a->ne[1];
    const uint32_t k        = (uint32_t) a->ne[0];
    const uint32_t n        = (uint32_t) q0->ne[1] * HEADSUM_HEADS;
    const uint32_t stride_a = (uint32_t) (a->nb[1] / ggml_type_size(a->type));
    const uint32_t stride_d = (uint32_t) (dst->nb[1] / sizeof(float));

    const std::vector<vk_matmul_pipeline_pair> & mmp = *ggml_vk_headsum_pipelines(ctx, a->type);
    const uint32_t kpad = ggml_vk_align_size(k, ggml_vk_guess_matmul_pipeline_align_map(ctx, mmp, m, n, false));
    // the aligned variant loads rows of A in vectors of up to 8
    const bool aligned = k == kpad && stride_a % 8 == 0 && m > 8 && n > 8;
    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, mmp, m, n, aligned, false);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer b_buf = ggml_vk_tensor_subbuffer(ctx, q0);
    // the rows of all heads
    b_buf.size = q0->ne[1] * q0->nb[1];
    vk_subbuffer d_buf = ggml_vk_tensor_subbuffer(ctx, dst);

    const vk_mat_mat_push_constants pc = {
        m, n, k, stride_a, k, stride_d, stride_a * m, k * n, stride_d * (n / HEADSUM_HEADS),
        0, 1, k, 1, 1, 1, 1, n, 1u | (bias ? 2u : 0u),
    };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { ggml_vk_tensor_subbuffer(ctx, a), b_buf, d_buf, bias ? ggml_vk_tensor_subbuffer(ctx, bias) : d_buf }, pc, { m, n, 1 });
}

// LIGHTNING_INDEXER on the MUL_MAT_HEADSUM matmul: sum_h relu(q_h . k) + mask is what the fused heads compute, with the
// f16 mask as the bias. the graph then holds one score and an f16 mask, where the matmul path held the head scores (two
// of 1 GiB at n_kv 262144 and a 4096-token ubatch on the 395, reserved even when the heads fuse) and an f32 bias. at every
// token count: the head matmul of the graph is this kernel at decode too, and test-micro-fork (31.10) gave the same time
// and the same bits for both, where lightning_indexer.comp took 115 against 65 us for one token at 65536 blocks on the
// 395 and 62 against 19 ms for 4096 tokens. lightning_indexer.comp stays for every case below: 4 heads, unit weights (a
// FILL of 1, as the qwen4exp graph builds them), one stream, an f32 or f16 key pool. GGML_VK_LID_HEADSUM=0:
// lightning_indexer.comp always (upstream). Since 01.11 the qwen4exp graph takes the op on Vulkan where it runs here
// (qsa_use_lid)
bool ggml_vk_fork_lid_headsum(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * dst) {
    static const bool disabled = [] { const char * env = getenv("GGML_VK_LID_HEADSUM"); return env && atoi(env) == 0; }();
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * w = dst->src[2];
    const ggml_tensor * m = dst->src[3];

    if (disabled || ctx->device->coopmat2 || ctx->device->disable_fusion) {
        return false;
    }
    if (q->type != GGML_TYPE_F32 || (k->type != GGML_TYPE_F32 && k->type != GGML_TYPE_F16) || m == nullptr ||
        m->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 || !ggml_vk_headsum_pipelines(ctx, k->type)) {
        return false;
    }
    if (q->ne[1] != HEADSUM_HEADS || q->ne[3] != 1 || k->ne[1] != 1 || k->ne[3] != 1 || m->ne[3] != 1 ||
        !ggml_is_contiguous(q) || !ggml_is_contiguous(dst) || k->nb[0] != ggml_type_size(k->type) ||
        k->nb[2] % ggml_type_size(k->type) != 0) {
        return false;
    }
    if (w->op != GGML_OP_FILL || ggml_get_op_params_f32(w, 0) != 1.0f) {
        return false;
    }
    // the mask is read at the index of dst
    if (m->ne[0] != dst->ne[0] || m->ne[1] != dst->ne[1] || m->nb[0] != sizeof(ggml_fp16_t) ||
        m->nb[1] / sizeof(ggml_fp16_t) != dst->nb[1] / sizeof(float) || m->nb[1] % sizeof(ggml_fp16_t) != 0) {
        return false;
    }
    const uint64_t max_range = ctx->device->properties.limits.maxStorageBufferRange;
    if (get_misalign_bytes(ctx, k) || get_misalign_bytes(ctx, q) || get_misalign_bytes(ctx, dst) || get_misalign_bytes(ctx, m) ||
        k->ne[2] * k->nb[2] > max_range || ggml_nbytes(q) > max_range || ggml_nbytes(dst) > max_range || ggml_nbytes(m) > max_range) {
        return false;
    }

    const uint32_t mr       = (uint32_t) k->ne[2];
    const uint32_t kd       = (uint32_t) k->ne[0];
    const uint32_t n        = (uint32_t) (q->ne[2] * HEADSUM_HEADS);
    const uint32_t stride_a = (uint32_t) (k->nb[2] / ggml_type_size(k->type));
    const uint32_t stride_d = (uint32_t) (dst->nb[1] / sizeof(float));

    const std::vector<vk_matmul_pipeline_pair> & mmp = *ggml_vk_headsum_pipelines(ctx, k->type);
    const uint32_t kpad = ggml_vk_align_size(kd, ggml_vk_guess_matmul_pipeline_align_map(ctx, mmp, mr, n, false));
    const bool aligned = kd == kpad && stride_a % 8 == 0 && mr > 8 && n > 8;
    vk_pipeline pipeline = ggml_vk_guess_matmul_pipeline_map(ctx, mmp, mr, n, aligned, false);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    vk_subbuffer a_buf = ggml_vk_tensor_subbuffer(ctx, k);
    a_buf.size = k->ne[2] * k->nb[2];

    // fusion_flags: 1 the heads of a token summed, 2 a bias, 4 the bias is f16
    const vk_mat_mat_push_constants pc = {
        mr, n, kd, stride_a, kd, stride_d, stride_a * mr, kd * n, stride_d * (n / HEADSUM_HEADS),
        0, 1, kd, 1, 1, 1, 1, n, 1u | 2u | 4u,
    };
    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,
        { a_buf, ggml_vk_tensor_subbuffer(ctx, q), ggml_vk_tensor_subbuffer(ctx, dst), ggml_vk_tensor_subbuffer(ctx, m) }, pc, { mr, n, 1 });

    // the first time, also on stderr with GGML_SCHED_LOG_REALLOC (llama-bench drops the ggml log)
    static bool logged = false;
    if (!logged) {
        logged = true;
        GGML_LOG_INFO("ggml_vulkan: LIGHTNING_INDEXER of %ld tokens on %s\n", (long) q->ne[2], pipeline->name.c_str());
        if (getenv("GGML_SCHED_LOG_REALLOC") != nullptr) {
            fprintf(stderr, "ggml_vulkan: LIGHTNING_INDEXER of %ld tokens on %s\n", (long) q->ne[2], pipeline->name.c_str());
        }
    }
    return true;
}

static bool ggml_vk_can_fuse_mmid_glu(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    // GGML_VK_DISABLE_MMID_GLU: off, for comparisons
    static const bool disabled = getenv("GGML_VK_DISABLE_MMID_GLU") != nullptr;
    const bool int8 = ctx->device->coopmat_int_support && (ctx->device->coopmat_int_ops & 2u) && ctx->device->integer_dot_product;
    if (disabled || (!int8 && ctx->device->mmid_f16reg_mode == 0)) {
        return false;
    }
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }, { node_idx + 2 })) {
        return false;
    }
    const ggml_tensor * a   = cgraph->nodes[node_idx];
    const ggml_tensor * b   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * glu = cgraph->nodes[node_idx + 2];
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[1] == nullptr ||
        !((glu->src[0] == a && glu->src[1] == b) || (glu->src[0] == b && glu->src[1] == a))) {
        return false;
    }
    if (a->src[1] != b->src[1] || a->src[2] != b->src[2] ||
        ggml_get_op_params_i32(a, 1) != ggml_get_op_params_i32(b, 1)) {
        return false;
    }
    const ggml_tensor * w0 = a->src[0];
    const ggml_tensor * w1 = b->src[0];
    if (w0->type != w1->type || !ggml_are_same_shape(w0, w1) || !ggml_are_same_stride(w0, w1) ||
        !ggml_is_contiguous(w0) || !ggml_is_contiguous(w1) ||
        ggml_nbytes(w0) > ctx->device->properties.limits.maxStorageBufferRange) {
        return false;
    }
    const ggml_tensor * x = a->src[1];
    if (x->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || (x->ne[0] * x->ne[1]) % 4 != 0) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(glu) || !ggml_are_same_shape(glu, a)) {
        return false;
    }
    if (ggml_vk_use_mul_mat_vec_id(cgraph, node_idx)) {
        return false;
    }
    // tokens below which the gate and up matmuls stay apart: with few rows per expert the fused shader is slower. on
    // the 395 (06.10, 512 experts, 10 per token) gate/up took 880 ms fused against 323 ms apart at 512 tokens, the
    // same at 2541 and less fused at 4096. GGML_VK_MMID_GLU_MIN_TOKENS, 0 fuses at any size
    static const int64_t min_tokens = [] {
        const char * env = getenv("GGML_VK_MMID_GLU_MIN_TOKENS");
        return env ? (int64_t) atoi(env) : 2048;
    }();
    if (a->src[2]->ne[1] < min_tokens) {
        return false;
    }
    // mul_mmid_f16reg.comp, when ggml_vk_mul_mat_id_q_f16 takes it for these operands (B read in place as f32, as it
    // decides it: not on coopmat2, not where Intel coopmat1 prefers f16 activations)
    const ggml_tensor * ids = a->src[2];
    const uint64_t n_as = w0->ne[2];
    const bool y_f32_in_place = ggml_vk_dim01_contiguous(x) && !ctx->device->coopmat2 &&
        !(ctx->device->coopmat_support && ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ggml_is_quantized(w0->type));
    if (ggml_vk_fork_mmid_f16reg(ctx, w0, x, w1, y_f32_in_place, CEIL_DIV(ids->ne[0] * ids->ne[1], n_as)) != nullptr) {
        return true;
    }
    return int8 && ggml_vk_get_mul_mat_mat_pipeline_map(ctx, w0->type, GGML_TYPE_Q8_1, (ggml_prec) a->op_params[0], true, true) != nullptr;
}

// the fused gate/up and the f16 cast of its GLU (qwen4exp on Vulkan casts it for the down projection, LLAMA_MOE_DOWN_F16):
// mul_mmid_f16reg.comp writes the result straight into the cast's tensor as f16, the GLU itself is not written. only on
// that shader (the int8 one writes f32), and the GLU must have no other reader
static bool ggml_vk_can_fuse_mmid_glu_cpy(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    if (node_idx + 3 >= cgraph->n_nodes || !ggml_vk_can_fuse_mmid_glu(ctx, cgraph, node_idx)) {
        return false;
    }
    const ggml_tensor * glu = cgraph->nodes[node_idx + 2];
    const ggml_tensor * cpy = cgraph->nodes[node_idx + 3];
    if (cpy->op != GGML_OP_CPY || cpy->src[0] != glu || cpy->type != GGML_TYPE_F16 || !ggml_are_same_shape(cpy, glu) ||
        !ggml_is_contiguous(cpy) || glu->ne[0] % 2 != 0) {
        return false;
    }
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU, GGML_OP_CPY }, { node_idx + 3 })) {
        return false;
    }
    const ggml_tensor * a   = cgraph->nodes[node_idx];
    const ggml_tensor * b   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * x   = a->src[1];
    const ggml_tensor * ids = a->src[2];
    const bool y_in_place = ggml_vk_dim01_contiguous(x) && !ctx->device->coopmat2 &&
        !(ctx->device->coopmat_support && ctx->device->vendor_id == VK_VENDOR_ID_INTEL && ggml_is_quantized(a->src[0]->type));
    return ggml_vk_fork_mmid_f16reg(ctx, a->src[0], x, b->src[0], y_in_place, CEIL_DIV(ids->ne[0] * ids->ne[1], a->src[0]->ne[2]),
                                    nullptr, true) != nullptr;
}

// the up projection of a hyper-connection mix and the gated DSV4_HC_PRE that reads it (qwen4exp): MUL_MAT, the reshape of
// its result to [E, 4, N], the DSV4_HC_PRE, with the reshape of the streams (a view) in between or not. mul_mm_f16reg.comp
// does the DSV4_HC_PRE in its store, the gates never go to memory. returns the nodes after the matmul (2 or 3), 0 if it
// does not apply. GGML_VK_DISABLE_MM_HC_PRE: off
static int ggml_vk_can_fuse_mm_hc_pre(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    static const bool disabled = getenv("GGML_VK_DISABLE_MM_HC_PRE") != nullptr;
    if (disabled || ctx->device->mm_f16reg_mode == 0) {
        return 0;
    }
    const ggml_tensor * mm = cgraph->nodes[node_idx];
    for (int extra : { 2, 3 }) {
        if (node_idx + extra >= cgraph->n_nodes) {
            break;
        }
        const ggml_tensor * hc = cgraph->nodes[node_idx + extra];
        const ggml_tensor * rg = cgraph->nodes[node_idx + extra - 1];
        if (hc->op != GGML_OP_DSV4_HC_PRE || ggml_get_op_params_i32(hc, 1) == 0 || hc->src[1] != rg ||
            rg->op != GGML_OP_RESHAPE || rg->src[0] != mm) {
            continue;
        }
        bool ok;
        if (extra == 2) {
            ok = ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE }, { node_idx + 2 });
        } else {
            const ggml_tensor * rx = cgraph->nodes[node_idx + 1];
            ok = rx->op == GGML_OP_RESHAPE && hc->src[0] == rx &&
                 ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_RESHAPE, GGML_OP_DSV4_HC_PRE },
                                        { node_idx + 1, node_idx + 3 });
        }
        if (!ok) {
            return 0;
        }
        vk_context none;
        return ggml_vk_fork_mm_f16reg(ctx, none, mm->src[0], mm->src[1], mm, nullptr, 0, hc, true) ? extra : 0;
    }
    return 0;
}

// gated_delta_net + the strided cpy that scatters its state snapshots into the recurrent cache (slot i -> rollback
// group i, slot 0 newest): the op writes the snapshots there and the cpy is skipped. Returns the number of nodes
// after the op that the fusion covers (views and the cpy), 0 if it does not apply
static int ggml_vk_can_fuse_gdn_cache(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx,
                                      const ggml_tensor ** cache) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the op skips its snapshot tail, so its output must not be read as a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0);
    const int64_t       n_written = std::min<int64_t>(n_tokens, K);
    const size_t        tail_off  = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // the cpy is the first node after the op that is not a view
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_PERMUTE ||
            n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_NONE) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the op writes to

    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off || !ggml_is_contiguous(src)) {
        return 0;
    }

    // [D, n_seqs, n_written] with a per-seq stride of D, as the shader writes it, at an offset the binding can take
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->buffer == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != sizeof(float) || dst->nb[1] != ggml_row_size(GGML_TYPE_F32, D) ||
        dst->nb[2] % sizeof(float) != 0 || dst->nb[2] / sizeof(float) > UINT32_MAX ||
        get_misalign_bytes(ctx, dst) != 0) {
        return 0;
    }

    // The fused op does not write the snapshot tail. Only the cache copy may read it.
    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (ggml_op_is_empty(node->op)) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = node->src[s];
            if (t == nullptr || (node == cpy && s == 0)) {
                continue;
            }
            if (t == gdn || (t->view_src == gdn &&
                (t->view_offs >= tail_off || ggml_nbytes(t) > tail_off - t->view_offs))) {
                return 0;
            }
        }
    }

    *cache = dst;
    return skip;
}

// GGML_VK_RMS_NORM_ROWS=0: rms_norm.comp for every RMS_NORM, as upstream, and none of the fusions of rms_norm_rows.comp
static bool ggml_vk_rms_norm_rows_enabled() {
    static const bool enabled = [] { const char * env = getenv("GGML_VK_RMS_NORM_ROWS"); return env == nullptr || atoi(env) != 0; }();
    return enabled;
}

// what rms_norm_rows.comp takes: f32 rows of up to 512 columns read with unit stride, a dst of the rms_norm shape with
// unit-stride rows, a weight of one row repeated over all rows, a gate of the rms_norm shape
static bool ggml_vk_rms_norm_rows_ok(const ggml_tensor * rms, const ggml_tensor * w, const ggml_tensor * gate, const ggml_tensor * dst) {
    const ggml_tensor * a = rms->src[0];
    const auto f32_rows = [](const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && t->nb[0] == sizeof(float); };
    if (!f32_rows(a) || !f32_rows(rms) || !f32_rows(dst) || a->ne[0] == 0 || a->ne[0] > 32*16 || ggml_nrows(a) == 0 ||
        !ggml_are_same_shape(dst, rms)) {
        return false;
    }
    if (w != nullptr && (!f32_rows(w) || w->ne[1] != 1 || w->ne[2] != 1 || w->ne[3] != 1 || a->ne[0] % w->ne[0] != 0)) {
        return false;
    }
    if (gate != nullptr && (!f32_rows(gate) || !ggml_are_same_shape(gate, rms))) {
        return false;
    }
    return true;
}

bool ggml_vk_fork_rms_norm_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * rms  = cgraph->nodes[node_idx];
    const ggml_tensor * w    = nullptr;
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * dst  = rms;
    float scale = 1.0f;

    if (ctx->fused_rms_norm_rows == 1) {
        dst   = cgraph->nodes[node_idx + 1];
        scale = ggml_get_op_params_f32(dst, 0);
    } else if (ctx->fused_rms_norm_rows == 2) {
        w    = cgraph->nodes[node_idx + 1]->src[1];
        gate = cgraph->nodes[node_idx + 2]->src[0];
        dst  = cgraph->nodes[node_idx + 3];
    } else {
        if (!ggml_vk_rms_norm_rows_enabled() || ctx->do_add_rms_partials) {
            return false;
        }
        if (ctx->fused_rms_norm_mode == RMS_NORM_MUL) {
            const ggml_tensor * mul = cgraph->nodes[node_idx + 1];
            w   = mul->src[0] == rms ? mul->src[1] : mul->src[0];
            dst = mul;
        } else if (ctx->fused_rms_norm_mode != RMS_NORM_COUNT) {
            return false;
        }
        if (!ggml_vk_rms_norm_rows_ok(rms, w, nullptr, dst)) {
            return false;
        }
    }

    const ggml_tensor * a = rms->src[0];
    const ggml_tensor * g = gate ? gate : a;

    vk_pipeline pipeline = ctx->device->pipeline_rms_norm_rows_f32[w != nullptr][gate != nullptr];
    GGML_ASSERT(pipeline != nullptr);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const uint32_t nrows = (uint32_t) ggml_nrows(a);
    const uint32_t f     = (uint32_t) sizeof(float);

    vk_op_rms_norm_rows_push_constants pc = {
        (uint32_t) a->ne[0], (uint32_t) a->ne[1], (uint32_t) a->ne[2], nrows,
        ggml_vk_nb_elem(a, 1),   ggml_vk_nb_elem(a, 2),   ggml_vk_nb_elem(a, 3),
        ggml_vk_nb_elem(g, 1),   ggml_vk_nb_elem(g, 2),   ggml_vk_nb_elem(g, 3),
        ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2), ggml_vk_nb_elem(dst, 3),
        w ? (uint32_t) w->ne[0] : 1u,
        get_misalign_bytes(ctx, a) / f,
        w ? get_misalign_bytes(ctx, w) / f : 0u,
        get_misalign_bytes(ctx, g) / f,
        get_misalign_bytes(ctx, dst) / f,
        ggml_get_op_params_f32(rms, 0),
        scale,
    };

    // 8 rows per workgroup, the workgroups in rows of at most 65535
    const uint32_t n_wg = (nrows + 7) / 8;
    const uint32_t wg_x = std::min<uint32_t>(n_wg, 65535u);
    const uint32_t wg_y = (n_wg + wg_x - 1) / wg_x;

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {
        ggml_vk_tensor_subbuffer(ctx, a, true),
        ggml_vk_tensor_subbuffer(ctx, w ? w : a, true),
        ggml_vk_tensor_subbuffer(ctx, g, true),
        ggml_vk_tensor_subbuffer(ctx, dst, true),
    }, pc, { wg_x, wg_y, 1 });

    return true;
}

bool ggml_vk_fork_fuse_rms_norm_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int i,
                                     bool * op_srcs_fused_elementwise, const char *& fusion_string) {
    const ggml_tensor * rms = cgraph->nodes[i];
    if (!ggml_vk_rms_norm_rows_enabled() || rms->op != GGML_OP_RMS_NORM || ctx->do_add_rms_partials) {
        return false;
    }

    // a row is read into registers before any of it is written, and a dst of the same layout as a source is written
    // where it was read, as rms_norm.comp with its fused MUL
    if (ggml_can_fuse_subgraph(cgraph, i, rms_norm_mul_sigmoid_mul_pattern, { i + 3 }) &&
        ggml_check_edges(cgraph, i, rms_norm_mul_sigmoid_mul_edges)) {
        const ggml_tensor * mul  = cgraph->nodes[i + 1];
        const ggml_tensor * sig  = cgraph->nodes[i + 2];
        const ggml_tensor * mul2 = cgraph->nodes[i + 3];
        if (ggml_get_unary_op(sig) == GGML_UNARY_OP_SIGMOID && sig->type == GGML_TYPE_F32 && mul->type == GGML_TYPE_F32 &&
            ggml_are_same_shape(mul, rms) && ggml_are_same_shape(sig, rms) &&
            ggml_vk_rms_norm_rows_ok(rms, mul->src[1], sig->src[0], mul2)) {
            ctx->num_additional_fused_ops = 3;
            ctx->fused_rms_norm_rows = 2;
            fusion_string = "RMS_NORM_MUL_SIGMOID_MUL";
            std::fill_n(op_srcs_fused_elementwise, 4, true);
            return true;
        }
    }

    if (ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE })) {
        const ggml_tensor * scale = cgraph->nodes[i + 1];
        if (ggml_get_op_params_f32(scale, 1) == 0.0f && ggml_vk_rms_norm_rows_ok(rms, nullptr, nullptr, scale)) {
            ctx->num_additional_fused_ops = 1;
            ctx->fused_rms_norm_rows = 1;
            fusion_string = "RMS_NORM_SCALE";
            std::fill_n(op_srcs_fused_elementwise, 2, true);
            return true;
        }
    }

    return false;
}

// CONCAT(state, x) along dim 0 read only by SSM_CONV (+ SILU): the conv reads the two sources. f32 everywhere, the
// channels and sequences of both sources match the conv, channel-contiguous conv rows. GGML_VK_SSM_CONV_CONCAT=0 runs
// the concat (upstream)
static bool ggml_vk_can_fuse_concat_ssm_conv(const struct ggml_cgraph * cgraph, int node_idx, bool & silu) {
    static const bool enabled = [] { const char * env = getenv("GGML_VK_SSM_CONV_CONCAT"); return env == nullptr || atoi(env) != 0; }();
    const ggml_tensor * concat = cgraph->nodes[node_idx];
    if (!enabled || concat->op != GGML_OP_CONCAT || node_idx + 1 >= cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * conv = cgraph->nodes[node_idx + 1];
    if (conv->op != GGML_OP_SSM_CONV || conv->src[0] != concat) {
        return false;
    }

    silu = node_idx + 2 < cgraph->n_nodes && cgraph->nodes[node_idx + 2]->op == GGML_OP_UNARY &&
           ggml_get_unary_op(cgraph->nodes[node_idx + 2]) == GGML_UNARY_OP_SILU && cgraph->nodes[node_idx + 2]->src[0] == conv &&
           ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_CONCAT, GGML_OP_SSM_CONV, GGML_OP_UNARY }, { node_idx + 2 });
    if (!silu && !ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_CONCAT, GGML_OP_SSM_CONV }, { node_idx + 1 })) {
        return false;
    }

    const ggml_tensor * st  = concat->src[0];
    const ggml_tensor * x   = concat->src[1];
    const ggml_tensor * w   = conv->src[1];
    const ggml_tensor * dst = silu ? cgraph->nodes[node_idx + 2] : conv;

    const auto f32 = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->nb[0] % sizeof(float) == 0 && t->nb[1] % sizeof(float) == 0 && t->nb[2] % sizeof(float) == 0;
    };
    if (ggml_get_op_params_i32(concat, 0) != 0 || !f32(st) || !f32(x) || !f32(w) || !f32(concat) || conv->type != GGML_TYPE_F32 ||
        dst->type != GGML_TYPE_F32 || dst->nb[0] != sizeof(float) || w->nb[0] != sizeof(float)) {
        return false;
    }
    const int64_t nr = conv->ne[0];
    return st->ne[1] == nr && x->ne[1] == nr && w->ne[1] == nr &&
           st->ne[2] == conv->ne[2] && x->ne[2] == conv->ne[2] &&
           st->ne[3] == 1 && x->ne[3] == 1 && conv->ne[3] == 1 &&
           ggml_are_same_shape(dst, conv);
}

void ggml_vk_fork_ssm_conv_concat(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * concat = cgraph->nodes[node_idx];
    const ggml_tensor * conv   = cgraph->nodes[node_idx + 1];
    const ggml_tensor * dst    = cgraph->nodes[node_idx + ctx->num_additional_fused_ops];
    const ggml_tensor * st     = concat->src[0];
    const ggml_tensor * x      = concat->src[1];
    const ggml_tensor * w      = conv->src[1];

    vk_pipeline pipeline = ctx->device->pipeline_ssm_conv_concat_f32[ctx->num_additional_fused_ops == 2];
    GGML_ASSERT(pipeline != nullptr);
    ggml_pipeline_request_descriptor_sets(ctx, pipeline, 1);

    const uint32_t f = (uint32_t) sizeof(float);

    vk_op_ssm_conv_concat_push_constants pc = {
        ggml_vk_nb_elem(st, 0), ggml_vk_nb_elem(st, 1), ggml_vk_nb_elem(st, 2),
        ggml_vk_nb_elem(x, 0),  ggml_vk_nb_elem(x, 1),  ggml_vk_nb_elem(x, 2),
        ggml_vk_nb_elem(w, 1),
        ggml_vk_nb_elem(dst, 1), ggml_vk_nb_elem(dst, 2),
        get_misalign_bytes(ctx, st) / f, get_misalign_bytes(ctx, x) / f, get_misalign_bytes(ctx, w) / f, get_misalign_bytes(ctx, dst) / f,
        (uint32_t) w->ne[0],
        (uint32_t) st->ne[0],
        (uint32_t) conv->ne[0],
        (uint32_t) conv->ne[1],
        (uint32_t) conv->ne[2],
    };

    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, {
        ggml_vk_tensor_subbuffer(ctx, st, true),
        ggml_vk_tensor_subbuffer(ctx, x, true),
        ggml_vk_tensor_subbuffer(ctx, w, true),
        ggml_vk_tensor_subbuffer(ctx, dst, true),
    }, pc, { (uint32_t) conv->ne[0], (uint32_t) conv->ne[1], (uint32_t) conv->ne[2] });
}

bool ggml_vk_fork_fuse_concat_ssm_conv(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int i,
                                       bool * op_srcs_fused_elementwise, const char *& fusion_string) {
    bool silu = false;
    if (!ggml_vk_can_fuse_concat_ssm_conv(cgraph, i, silu)) {
        return false;
    }
    ctx->num_additional_fused_ops = silu ? 2 : 1;
    ctx->fused_concat_ssm_conv = true;
    fusion_string = silu ? "CONCAT_SSM_CONV_SILU" : "CONCAT_SSM_CONV";
    // an output reads several input columns: no overlap with the sources
    std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
    return true;
}

void ggml_vk_fork_fusion_reset(ggml_backend_vk_context * ctx) {
    ctx->fused_hc_post_norm = false;
    ctx->fused_hc_post_norm_f16 = false;
    ctx->fused_hc_post_norm_mix = false;
    ctx->fused_mm_relu = false;
    ctx->fused_mm_hc_pre = false;
    ctx->fused_mm_headsum = false;
    ctx->fused_mm_headsum_bias = nullptr;
    ctx->fused_gdn_cache = nullptr;
    ctx->fused_rms_norm_rows = 0;
    ctx->fused_concat_ssm_conv = false;
}

bool ggml_vk_fork_fuse_mul_mat(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int i,
                               bool * op_srcs_fused_elementwise, const char *& fusion_string) {
    int headsum_run = 0;
    const ggml_tensor * headsum_bias = nullptr;
    if ((headsum_run = ggml_vk_can_fuse_mul_mat_headsum(ctx, cgraph, i, &headsum_bias)) > 0) {
        ctx->num_additional_fused_ops = headsum_run - 1;
        ctx->fused_mm_headsum = true;
        ctx->fused_mm_headsum_bias = headsum_bias;
        fusion_string = "MUL_MAT_HEADSUM";
        for (int k = 0; k < headsum_run; ++k) {
            op_srcs_fused_elementwise[k] = cgraph->nodes[i + k]->op != GGML_OP_MUL_MAT;
        }
    } else if (int hc_run = ggml_vk_can_fuse_mm_hc_pre(ctx, cgraph, i); hc_run > 0) {
        ctx->num_additional_fused_ops = hc_run;
        ctx->fused_mm_hc_pre = true;
        fusion_string = "MUL_MAT_DSV4_HC_PRE";
        for (int k = 0; k <= hc_run; ++k) {
            op_srcs_fused_elementwise[k] = false;
        }
    } else if (ggml_vk_can_fuse_mm_relu(ctx, cgraph, i)) {
        ctx->num_additional_fused_ops = 1;
        ctx->fused_mm_relu = true;
        fusion_string = "MUL_MAT_RELU";
        op_srcs_fused_elementwise[0] = false;
        op_srcs_fused_elementwise[1] = true;
    } else if (ggml_vk_can_fuse_mmid_glu_cpy(ctx, cgraph, i)) {
        ctx->num_additional_fused_ops = 3;
        fusion_string = "MUL_MAT_ID_MUL_MAT_ID_GLU_CPY";
        op_srcs_fused_elementwise[0] = false;
        op_srcs_fused_elementwise[1] = false;
        op_srcs_fused_elementwise[2] = false;
        op_srcs_fused_elementwise[3] = false;
    } else if (ggml_vk_can_fuse_mmid_glu(ctx, cgraph, i)) {
        ctx->num_additional_fused_ops = 2;
        fusion_string = "MUL_MAT_ID_MUL_MAT_ID_GLU";
        op_srcs_fused_elementwise[0] = false;
        op_srcs_fused_elementwise[1] = false;
        op_srcs_fused_elementwise[2] = false;
    } else {
        return false;
    }
    return true;
}

// HC_POST_NORM_MIX (hc_post_norm_mix_pattern): hc_post_norm_f16 and the injection product of the next mix, the f16
// streams read through their reshape, the injection weight f32 [hc*n_embd, hc], the cont [hc, n_tokens].
// GGML_VK_DISABLE_HC_POST_NORM_MIX: hc_post_norm_f16 and the injection apart
static bool ggml_vk_hc_post_norm_mix_off() {
    static const bool disabled = getenv("GGML_VK_DISABLE_HC_POST_NORM_MIX") != nullptr;
    return disabled;
}

static bool ggml_vk_can_fuse_hc_post_norm_mix(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int i) {
    if (ggml_vk_hc_post_norm_mix_off() || ctx->device->pipeline_dsv4_hc_post_norm_mix == nullptr ||
        !ggml_can_fuse_subgraph(cgraph, i, hc_post_norm_mix_pattern, { i + 3, i + 6, i + 7, i + 10 }) ||
        !ggml_check_edges(cgraph, i, hc_post_norm_mix_edges) || !ggml_vk_can_fuse_hc_post_norm(cgraph, i)) {
        return false;
    }
    const ggml_tensor * hc_post = cgraph->nodes[i + 3];
    const ggml_tensor * mul     = cgraph->nodes[i + 5];
    const ggml_tensor * cpy     = cgraph->nodes[i + 6];
    const ggml_tensor * xn      = cgraph->nodes[i + 7];
    const ggml_tensor * mm      = cgraph->nodes[i + 8];
    const ggml_tensor * cont    = cgraph->nodes[i + 10];
    const ggml_tensor * wi      = mm->src[1];
    const int64_t n_embd = hc_post->ne[0];
    const int64_t hc     = hc_post->ne[1];
    const int64_t nt     = hc_post->ne[2];
    return cpy->type == GGML_TYPE_F16 && ggml_is_contiguous(cpy) && ggml_are_same_shape(mul, cpy) &&
           xn->ne[0] == n_embd * hc && xn->ne[1] == nt && ggml_is_contiguous(xn) &&
           wi->type == GGML_TYPE_F32 && ggml_is_contiguous(wi) && wi->ne[0] == n_embd * hc && wi->ne[1] == hc &&
           wi->ne[2] == 1 && wi->ne[3] == 1 &&
           mm->type == GGML_TYPE_F32 && mm->ne[0] == nt && mm->ne[1] == hc &&
           cont->type == GGML_TYPE_F32 && ggml_is_contiguous(cont) && cont->ne[0] == hc && cont->ne[1] == nt &&
           cont->ne[2] == 1 && cont->ne[3] == 1;
}

bool ggml_vk_fork_fuse_hc_post_norm(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int i,
                                    bool * op_srcs_fused_elementwise, const char *& fusion_string) {
    if (ggml_vk_can_fuse_hc_post_norm_mix(ctx, cgraph, i)) {
        ctx->num_additional_fused_ops = hc_post_norm_mix_pattern.size() - 1;
        // written: the hc_post output (the residual of the next combine), the f16 streams and (the last node) the cont
        ctx->fused_ops_write_mask |= (1 << 3) | (1 << 6);
        ctx->fused_hc_post_norm = true;
        ctx->fused_hc_post_norm_f16 = true;
        ctx->fused_hc_post_norm_mix = true;
        fusion_string = "HC_POST_NORM_MIX";
        std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
        return true;
    }
    // the f16 copy of the normalized streams (qwen4exp, LLAMA_HC_F16) goes into the shader's store
    if (ggml_can_fuse_subgraph(cgraph, i, hc_post_norm_f16_pattern, { i + 3, i + 6 }) &&
        ggml_check_edges(cgraph, i, hc_post_norm_f16_edges) &&
        ggml_vk_can_fuse_hc_post_norm(cgraph, i)) {
        const ggml_tensor * mul = cgraph->nodes[i + 5];
        const ggml_tensor * cpy = cgraph->nodes[i + 6];
        if (cpy->type == GGML_TYPE_F16 && cpy->nb[0] == sizeof(ggml_fp16_t) && ggml_is_contiguous(cpy) &&
            ggml_are_same_shape(mul, cpy) && ctx->device->pipeline_dsv4_hc_post_norm_f16n) {
            ctx->num_additional_fused_ops = hc_post_norm_f16_pattern.size() - 1;
            ctx->fused_ops_write_mask |= 1 << 3;
            ctx->fused_hc_post_norm = true;
            ctx->fused_hc_post_norm_f16 = true;
            fusion_string = "HC_POST_NORM_F16";
            std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
            return true;
        }
    }
    if (!(ggml_can_fuse_subgraph(cgraph, i, hc_post_norm_pattern, { i + 3, i + 5 }) &&
          ggml_check_edges(cgraph, i, hc_post_norm_edges) &&
          ggml_vk_can_fuse_hc_post_norm(cgraph, i))) {
        return false;
    }
    ctx->num_additional_fused_ops = hc_post_norm_pattern.size() - 1;
    // the hc_post output is the residual of the next combine
    ctx->fused_ops_write_mask |= 1 << 3;
    ctx->fused_hc_post_norm = true;
    fusion_string = "HC_POST_NORM";
    std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
    return true;
}

bool ggml_vk_fork_fuse_gdn_cache(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int i,
                                 bool * op_srcs_fused_elementwise, const char *& fusion_string) {
    int n_gdn = ggml_vk_can_fuse_gdn_cache(ctx, cgraph, i, &ctx->fused_gdn_cache);
    if (!n_gdn) {
        return false;
    }
    ctx->num_additional_fused_ops = n_gdn;
    // the op writes its attention output, the cpy (the last node) the cache
    ctx->fused_ops_write_mask |= 1 << 0;
    fusion_string = "GATED_DELTA_NET_CACHE";
    std::fill_n(op_srcs_fused_elementwise, ctx->num_additional_fused_ops + 1, false);
    return true;
}

std::vector<uint8_t> ggml_vk_fork_headsum_nodes(ggml_backend_vk_context * ctx, const struct ggml_cgraph * graph) {
    // the nodes of a MUL_MAT_HEADSUM pattern (ggml_vk_find_headsum) are not pulled forward one by one: when its first
    // matmul comes up, ggml_vk_fork_keep_headsum puts all of them in a row
    std::vector<uint8_t> headsum_node(graph->n_nodes, false);
    if (!ctx->device->disable_fusion && (ggml_vk_headsum_pipelines(ctx, GGML_TYPE_F32) || ggml_vk_headsum_pipelines(ctx, GGML_TYPE_F16))) {
        int idxs[3*HEADSUM_HEADS];
        for (int i = 0; i < graph->n_nodes; ++i) {
            if (graph->nodes[i]->op == GGML_OP_MUL_MAT && !headsum_node[i]) {
                const int n = ggml_vk_find_headsum(graph, i, i + 48, nullptr, idxs);
                for (int k = 0; k < n; ++k) {
                    headsum_node[idxs[k]] = true;
                }
            }
        }
    }
    return headsum_node;
}

// MUL_MAT_HEADSUM: the views the pattern reads that are not scheduled yet, then the pattern nodes in their order
bool ggml_vk_fork_keep_headsum(struct ggml_cgraph * graph, struct ggml_backend_graph_optimize_params * params, int & first_unused,
                               std::vector<uint8_t> & used, std::vector<ggml_tensor *> & new_order, std::set<ggml_tensor *> & used_node_set) {
    int idxs[3*HEADSUM_HEADS];
    const int n = ggml_vk_find_headsum(graph, first_unused, first_unused + 48, &used, idxs);
    if (n == 0) {
        return false;
    }
    std::vector<int> nodes(idxs, idxs + n);
    std::sort(nodes.begin(), nodes.end());

    auto const &listed = [](const std::vector<int> & v, int x) {
        return std::find(v.begin(), v.end(), x) != v.end();
    };
    // the index of src in [first_unused, hi), -1 if it is scheduled already or not a node
    auto const &unscheduled = [&](const ggml_tensor * src, int hi) -> int {
        for (int t = first_unused; t < hi; ++t) {
            if (graph->nodes[t] == src) {
                return used[t] ? -1 : t;
            }
        }
        return -1;
    };

    std::vector<int> views;
    for (int p : nodes) {
        for (const ggml_tensor * src : graph->nodes[p]->src) {
            const int t = src ? unscheduled(src, p) : -1;
            if (t < 0 || listed(nodes, t) || listed(views, t)) {
                continue;
            }
            if (!ggml_vk_graph_node_is_empty(graph->nodes[t])) {
                return false;
            }
            for (const ggml_tensor * src2 : graph->nodes[t]->src) {
                const int t2 = src2 ? unscheduled(src2, t) : -1;
                if (t2 >= 0 && !listed(views, t2)) {
                    return false;
                }
            }
            views.push_back(t);
        }
    }
    std::sort(views.begin(), views.end());

    // the matmul reads A, q and the bias while it writes the output, so they live until the output
    if (params && params->add_alloc_dep) {
        for (int p : nodes) {
            for (ggml_tensor * src : graph->nodes[p]->src) {
                if (src && !listed(nodes, unscheduled(src, graph->n_nodes))) {
                    params->add_alloc_dep(params->user_data, src, graph->nodes[nodes.back()]);
                }
            }
        }
    }

    for (const auto * list : { &views, &nodes }) {
        for (int t : *list) {
            new_order.push_back(graph->nodes[t]);
            used_node_set.insert(graph->nodes[t]);
            used[t] = true;
        }
    }
    while (first_unused < graph->n_nodes && used[first_unused]) {
        first_unused++;
    }
    return true;
}

std::vector<uint8_t> ggml_vk_fork_protected_nodes(const struct ggml_cgraph * graph) {
    // the window would pull an independent node out of these patterns (the SIGMOID of a gate computed long before),
    // and only the first node of a pattern is kept from it
    std::vector<uint8_t> protect(graph->n_nodes, false);
    for (int i = 0; i < graph->n_nodes; ++i) {
        const ggml_tensor * node = graph->nodes[i];
        if (node->op == GGML_OP_RMS_NORM && i + 3 < graph->n_nodes &&
            graph->nodes[i + 1]->op == GGML_OP_MUL && graph->nodes[i + 2]->op == GGML_OP_UNARY && graph->nodes[i + 3]->op == GGML_OP_MUL &&
            ggml_get_unary_op(graph->nodes[i + 2]) == GGML_UNARY_OP_SIGMOID && ggml_check_edges(graph, i, rms_norm_mul_sigmoid_mul_edges)) {
            protect[i + 1] = protect[i + 2] = protect[i + 3] = true;
        }
        if (node->op == GGML_OP_CONCAT && i + 1 < graph->n_nodes &&
            graph->nodes[i + 1]->op == GGML_OP_SSM_CONV && graph->nodes[i + 1]->src[0] == node) {
            protect[i + 1] = true;
            if (i + 2 < graph->n_nodes && graph->nodes[i + 2]->op == GGML_OP_UNARY && graph->nodes[i + 2]->src[0] == graph->nodes[i + 1]) {
                protect[i + 2] = true;
            }
        }
    }
    return protect;
}

void ggml_vk_fork_pull_nodes(ggml_backend_vk_context * ctx, const struct ggml_cgraph * graph, int first_unused, int num_to_check,
                             std::vector<uint8_t> & used, std::vector<int> & current_set, const std::function<bool(int, int)> & is_src_of,
                             struct ggml_backend_graph_optimize_params * params) {
    // CONCAT read only by an SSM_CONV: pull the conv (and a SILU of it) right after the concat for
    // CONCAT_SSM_CONV(_SILU). the nodes in between (the conv state copies, the recurrent state gather) do not feed it.
    // the fused conv reads the concat sources while it writes, so they stay allocated until its last node
    if (graph->nodes[first_unused]->op == GGML_OP_CONCAT && !ctx->device->disable_fusion) {
        const ggml_tensor * concat = graph->nodes[first_unused];
        for (int k = first_unused + 1; k < std::min(first_unused + num_to_check, graph->n_nodes); ++k) {
            const ggml_tensor * n = graph->nodes[k];
            if (used[k] || n->op != GGML_OP_SSM_CONV || n->src[0] != concat) {
                continue;
            }
            bool can_pull = true;
            for (int c = first_unused + 1; can_pull && c < k; ++c) {
                if (!used[c] && is_src_of(k, c)) {
                    can_pull = false;
                }
            }
            if (!can_pull) {
                break;
            }
            current_set.push_back(k);
            used[k] = true;
            int last = k;
            for (int q = k + 1; q < std::min(first_unused + num_to_check, graph->n_nodes); ++q) {
                const ggml_tensor * u = graph->nodes[q];
                if (!used[q] && u->op == GGML_OP_UNARY && ggml_get_unary_op(u) == GGML_UNARY_OP_SILU && u->src[0] == n) {
                    bool ok = true;
                    for (int c = k + 1; ok && c < q; ++c) {
                        if (!used[c] && is_src_of(q, c)) {
                            ok = false;
                        }
                    }
                    if (ok) {
                        current_set.push_back(q);
                        used[q] = true;
                        last = q;
                    }
                    break;
                }
            }
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (concat->src[s]) {
                    params->add_alloc_dep(params->user_data, concat->src[s], graph->nodes[last]);
                }
            }
            break;
        }
    }

    // GATED_DELTA_NET: pull the cpy of its state snapshots into the cache (and the two views it reads) right
    // after it, so the cpy can be fused into the op
    if (graph->nodes[first_unused]->op == GGML_OP_GATED_DELTA_NET) {
        const int gdn_idx = first_unused;
        int cpy_idx = -1;
        for (int k = gdn_idx + 1; k < std::min(gdn_idx + num_to_check, graph->n_nodes); ++k) {
            const ggml_tensor * n = graph->nodes[k];
            if (!used[k] && n->op == GGML_OP_CPY && n->src[0]->op == GGML_OP_VIEW && n->src[0]->view_src == graph->nodes[gdn_idx]) {
                cpy_idx = k;
                break;
            }
        }
        if (cpy_idx != -1) {
            std::vector<int> views;
            for (int k = gdn_idx + 1; k < cpy_idx; ++k) {
                if (!used[k] && (graph->nodes[k] == graph->nodes[cpy_idx]->src[0] || graph->nodes[k] == graph->nodes[cpy_idx]->src[1])) {
                    views.push_back(k);
                }
            }
            // no other node before the cpy may feed it
            bool can_pull = true;
            for (int c = gdn_idx + 1; can_pull && c < cpy_idx; ++c) {
                if (!used[c] && std::find(views.begin(), views.end(), c) == views.end() && is_src_of(cpy_idx, c)) {
                    can_pull = false;
                }
            }
            if (can_pull) {
                views.push_back(cpy_idx);
                for (int k : views) {
                    current_set.push_back(k);
                    used[k] = true;
                }
            }
        }
    }

    // HC_POST_NORM_MIX: the reshape of the f16 streams, the injection product over it, its transpose and cont right
    // after the cpy that closes hc_post_norm_f16 (the generic pass puts the product next to the hc down). the block
    // output, the residual and the previous injection stay allocated until the cont: the shader reads them while it
    // writes the injection, and the backend drops a fusion whose output overlaps a source
    if (graph->nodes[first_unused]->op == GGML_OP_CPY && graph->nodes[first_unused]->type == GGML_TYPE_F16 &&
        ctx->device->pipeline_dsv4_hc_post_norm_mix != nullptr && !ctx->device->disable_fusion && !ggml_vk_hc_post_norm_mix_off()) {
        ggml_tensor * cpy  = graph->nodes[first_unused];
        ggml_tensor * mul  = cpy->src[0];
        ggml_tensor * norm = mul && mul->op == GGML_OP_MUL ? mul->src[0] : nullptr;
        ggml_tensor * post = norm && norm->op == GGML_OP_RMS_NORM ? norm->src[0] : nullptr;
        if (post && post->op == GGML_OP_DSV4_HC_POST) {
            const int end = std::min(first_unused + num_to_check, graph->n_nodes);
            int idx[4] = { -1, -1, -1, -1 }; // reshape, mul_mat, transpose, cont
            for (int k = first_unused + 1; k < end && idx[3] == -1; ++k) {
                const ggml_tensor * n = graph->nodes[k];
                if (used[k]) {
                    continue;
                }
                if (idx[0] == -1 && n->op == GGML_OP_RESHAPE && n->src[0] == cpy) {
                    idx[0] = k;
                } else if (idx[0] != -1 && idx[1] == -1 && n->op == GGML_OP_MUL_MAT && n->src[0] == graph->nodes[idx[0]] &&
                           n->src[1]->type == GGML_TYPE_F32) {
                    idx[1] = k;
                } else if (idx[1] != -1 && idx[2] == -1 && n->op == GGML_OP_TRANSPOSE && n->src[0] == graph->nodes[idx[1]]) {
                    idx[2] = k;
                } else if (idx[2] != -1 && n->op == GGML_OP_CONT && n->src[0] == graph->nodes[idx[2]]) {
                    idx[3] = k;
                }
            }
            // a view in between (the reshape of the same streams for the DSV4_HC_PRE) writes nothing: it holds the pull
            // back only as a source of a pulled node, is_src_of ties every two views of one tensor together
            bool can_pull = idx[3] != -1;
            for (int c = first_unused + 1; can_pull && c < idx[3]; ++c) {
                if (used[c] || c == idx[0] || c == idx[1] || c == idx[2]) {
                    continue;
                }
                const bool view = ggml_vk_graph_node_is_empty(graph->nodes[c]);
                for (int q = 0; q < 4 && can_pull; ++q) {
                    bool dep = false;
                    if (view) {
                        for (const ggml_tensor * src : graph->nodes[idx[q]]->src) {
                            dep = dep || src == graph->nodes[c];
                        }
                    } else {
                        dep = is_src_of(idx[q], c);
                    }
                    if (dep) {
                        can_pull = false;
                    }
                }
            }
            if (can_pull) {
                for (int q = 0; q < 4; ++q) {
                    current_set.push_back(idx[q]);
                    used[idx[q]] = true;
                }
                ggml_tensor * cont = graph->nodes[idx[3]];
                // post = 2*sigmoid(inject/hc): scale <- sigmoid <- scale <- the previous injection
                ggml_tensor * w   = post->src[2];
                ggml_tensor * inj = w && w->src[0] && w->src[0]->src[0] ? w->src[0]->src[0]->src[0] : nullptr;
                for (ggml_tensor * t : { post->src[0], post->src[1], inj }) {
                    if (t != nullptr) {
                        params->add_alloc_dep(params->user_data, t, cont);
                    }
                }
            }
        }
    }

    // MUL_MAT_DSV4_HC_PRE: the reshape of the matmul and the gated DSV4_HC_PRE that reads it right after it (and the
    // reshape of the streams, a view, before them when it is not placed yet), so that ggml_vk_can_fuse_mm_hc_pre finds
    // them in a row: the generic pass would pull the injection matmul of the same streams in between
    if (graph->nodes[first_unused]->op == GGML_OP_MUL_MAT && ctx->device->mm_f16reg_mode != 0 && !ctx->device->disable_fusion) {
        const ggml_tensor * mm = graph->nodes[first_unused];
        const int end = std::min(first_unused + num_to_check, graph->n_nodes);
        int rg_idx = -1;
        int hc_idx = -1;
        for (int k = first_unused + 1; k < end; ++k) {
            const ggml_tensor * n = graph->nodes[k];
            if (used[k]) {
                continue;
            }
            if (rg_idx == -1 && n->op == GGML_OP_RESHAPE && n->src[0] == mm) {
                rg_idx = k;
                continue;
            }
            if (rg_idx != -1 && n->op == GGML_OP_DSV4_HC_PRE && n->src[1] == graph->nodes[rg_idx] && ggml_get_op_params_i32(n, 1) != 0) {
                hc_idx = k;
                break;
            }
        }
        if (hc_idx != -1) {
            const ggml_tensor * hc = graph->nodes[hc_idx];
            int rx_idx = -1;
            for (int k = first_unused + 1; k < hc_idx; ++k) {
                if (!used[k] && graph->nodes[k] == hc->src[0] && graph->nodes[k]->op == GGML_OP_RESHAPE) {
                    rx_idx = k;
                }
            }
            // nothing else unplaced in between may feed them
            bool can_pull = true;
            for (int c = first_unused + 1; can_pull && c < hc_idx; ++c) {
                if (used[c] || c == rg_idx || c == rx_idx) {
                    continue;
                }
                if (is_src_of(hc_idx, c) || is_src_of(rg_idx, c) || (rx_idx != -1 && is_src_of(rx_idx, c))) {
                    can_pull = false;
                }
            }
            if (can_pull) {
                if (rx_idx != -1) {
                    current_set.push_back(rx_idx);
                    used[rx_idx] = true;
                }
                current_set.push_back(rg_idx);
                used[rg_idx] = true;
                current_set.push_back(hc_idx);
                used[hc_idx] = true;
                // the activations of the matmul stay allocated until the DSV4_HC_PRE: freed after the matmul, the
                // allocator could place the result there, and the backend drops a fusion whose output overlaps one of
                // its sources (395, 08.11, profile-20261108-194131: 22 of 96 fused before this dependency)
                params->add_alloc_dep(params->user_data, mm->src[1], graph->nodes[hc_idx]);
            }
        }
    }

    // Pull the MoE gate, up and swiglu together. Keep weight producers and alias dependencies before them.
    if (ctx->device->coopmat_int_support && graph->nodes[first_unused]->op == GGML_OP_MUL_MAT_ID) {
        const ggml_tensor * mm = graph->nodes[first_unused];
        for (int k = first_unused + 1; k < std::min(first_unused + num_to_check, graph->n_nodes); ++k) {
            const ggml_tensor * n = graph->nodes[k];
            if (used[k] || n->op != GGML_OP_GLU || ggml_get_glu_op(n) != GGML_GLU_OP_SWIGLU || n->src[1] == nullptr) {
                continue;
            }
            const ggml_tensor * other = n->src[0] == mm ? n->src[1] : n->src[1] == mm ? n->src[0] : nullptr;
            if (other == nullptr) {
                continue;
            }
            if (other->op == GGML_OP_MUL_MAT_ID && other->src[1] == mm->src[1] && other->src[2] == mm->src[2]) {
                for (int q = first_unused + 1; q < k; ++q) {
                    if (graph->nodes[q] == other && !used[q]) {
                        bool can_pull = true;
                        for (int c = first_unused + 1; can_pull && c < k; ++c) {
                            if (!used[c] && c != q && (is_src_of(q, c) || is_src_of(k, c))) {
                                can_pull = false;
                            }
                        }
                        if (!can_pull) {
                            break;
                        }
                        current_set.push_back(q);
                        used[q] = true;
                        current_set.push_back(k);
                        used[k] = true;
                        // the f16 cast of the GLU right after it (MUL_MAT_ID_MUL_MAT_ID_GLU_CPY); it reads the GLU only
                        for (int c2 = k + 1; c2 < std::min(first_unused + num_to_check, graph->n_nodes); ++c2) {
                            const ggml_tensor * cp = graph->nodes[c2];
                            if (used[c2] || cp->op != GGML_OP_CPY || cp->src[0] != n || cp->type != GGML_TYPE_F16) {
                                continue;
                            }
                            bool cp_ok = true;
                            for (int c = first_unused + 1; cp_ok && c < c2; ++c) {
                                if (!used[c] && is_src_of(c2, c)) {
                                    cp_ok = false;
                                }
                            }
                            if (cp_ok) {
                                current_set.push_back(c2);
                                used[c2] = true;
                            }
                            break;
                        }
                        break;
                    }
                }
            }
            break;
        }
    }
}
