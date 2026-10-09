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

// nodes: scale, sigmoid, scale, dsv4_hc_post, rms_norm, mul (hc_post_norm_pattern)
void ggml_vk_dsv4_hc_post_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx) {
    const ggml_tensor * scale_in = cgraph->nodes[node_idx];
    const ggml_tensor * hc_post  = cgraph->nodes[node_idx + 3];
    const ggml_tensor * norm     = cgraph->nodes[node_idx + 4];
    const ggml_tensor * mul      = cgraph->nodes[node_idx + 5];

    const ggml_tensor * x        = hc_post->src[0];
    const ggml_tensor * residual = hc_post->src[1];
    const ggml_tensor * p_src    = scale_in->src[0];
    const ggml_tensor * w        = mul->src[1];

    vk_pipeline pipeline = ctx->device->pipeline_dsv4_hc_post_norm_f32;
    GGML_ASSERT(pipeline != nullptr);

    const uint32_t n_embd   = (uint32_t)x->ne[0];
    const uint32_t n_tokens = (uint32_t)x->ne[1];

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
        get_misalign_bytes(ctx, mul)      / (uint32_t) sizeof(float),
        1u,
        ggml_get_op_params_f32(scale_in, 0),
        ggml_get_op_params_f32(hc_post->src[2], 0),
        ggml_get_op_params_f32(norm, 0),
    };

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

static bool ggml_vk_can_fuse_mmid_glu(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {
    // GGML_VK_DISABLE_MMID_GLU: off, for comparisons
    static const bool disabled = getenv("GGML_VK_DISABLE_MMID_GLU") != nullptr;
    if (disabled || !ctx->device->coopmat_int_support || !(ctx->device->coopmat_int_ops & 2u) || !ctx->device->integer_dot_product) {
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
    return ggml_vk_get_mul_mat_mat_pipeline_map(ctx, w0->type, GGML_TYPE_Q8_1, (ggml_prec) a->op_params[0], true, true) != nullptr;
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

void ggml_vk_fork_fusion_reset(ggml_backend_vk_context * ctx) {
    ctx->fused_hc_post_norm = false;
    ctx->fused_mm_relu = false;
    ctx->fused_mm_headsum = false;
    ctx->fused_mm_headsum_bias = nullptr;
    ctx->fused_gdn_cache = nullptr;
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
    } else if (ggml_vk_can_fuse_mm_relu(ctx, cgraph, i)) {
        ctx->num_additional_fused_ops = 1;
        ctx->fused_mm_relu = true;
        fusion_string = "MUL_MAT_RELU";
        op_srcs_fused_elementwise[0] = false;
        op_srcs_fused_elementwise[1] = true;
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

bool ggml_vk_fork_fuse_hc_post_norm(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int i,
                                    bool * op_srcs_fused_elementwise, const char *& fusion_string) {
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

void ggml_vk_fork_pull_nodes(ggml_backend_vk_context * ctx, const struct ggml_cgraph * graph, int first_unused, int num_to_check,
                             std::vector<uint8_t> & used, std::vector<int> & current_set, const std::function<bool(int, int)> & is_src_of) {
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
                        break;
                    }
                }
            }
            break;
        }
    }
}
