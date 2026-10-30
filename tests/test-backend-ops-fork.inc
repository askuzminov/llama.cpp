// the fork's tests, included by test-backend-ops.cpp after the upstream test classes: its own test classes and the
// cases make_test_cases_eval and make_test_cases_perf append at their end

// a row of ids uses the last expert more than once, as the MoE cache does with its zero slot
struct test_mul_mat_id_repeat : public test_mul_mat_id {
    test_mul_mat_id_repeat(ggml_type type_a, ggml_type type_b, int n_mats, int n_used, bool b, int64_t m, int64_t n, int64_t k)
        : test_mul_mat_id(type_a, type_b, n_mats, n_used, b, m, n, k) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * out = test_mul_mat_id::build_graph(ctx);
        ggml_mul_mat_set_hint(out, GGML_HINT_IDS_REPEAT);
        return out;
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_ID_REPEAT";
    }

    void init_ids(ggml_context * ctx) {
        std::random_device rd;
        std::default_random_engine rng(rd());
        std::vector<int32_t> experts(n_mats - 1);
        for (int i = 0; i < n_mats - 1; i++) {
            experts[i] = i;
        }
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (t->type != GGML_TYPE_I32 || ggml_is_view_op(t->op)) {
                continue;
            }
            for (int64_t r = 0; r < ggml_nrows(t); r++) {
                // a random number of distinct experts, the other uses go to the last expert
                std::shuffle(experts.begin(), experts.end(), rng);
                const int n_distinct = std::min<int>(rng() % (n_used + 1), n_mats - 1);
                std::vector<int32_t> data(t->ne[0], n_mats - 1);
                std::copy(experts.begin(), experts.begin() + n_distinct, data.begin());
                std::shuffle(data.begin(), data.begin() + n_used, rng);
                ggml_backend_tensor_set(t, data.data(), r * t->nb[1], t->ne[0] * sizeof(int32_t));
            }
        }
    }

    void initialize_tensors(ggml_context * ctx) override {
        test_mul_mat_id::initialize_tensors(ctx);
        init_ids(ctx);
    }

    void reinit_perf_iter(ggml_context * ctx) override {
        init_ids(ctx);
    }
};

// MUL_MAT followed by RELU, as the per-head indexer scores of qwen4exp. heads > 1 reads b as every heads-th row of
// a larger matrix (the scores of one head). backends may rectify the result at the matmul store
struct test_mul_mat_relu : public test_case {
    const ggml_type type_a;
    const ggml_type type_b;
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const int64_t heads;

    std::string vars() override {
        return VARS_TO_STR6(type_a, type_b, m, n, k, heads);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_RELU";
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    bool run_whole_graph() override { return true; }

    test_mul_mat_relu(ggml_type type_a = GGML_TYPE_F32, ggml_type type_b = GGML_TYPE_F32,
            int64_t m = 64, int64_t n = 32, int64_t k = 128, int64_t heads = 1)
        : type_a(type_a), type_b(type_b), m(m), n(n), k(k), heads(heads) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, type_a, k, m);
        ggml_set_name(a, "a");

        ggml_tensor * b = ggml_new_tensor_2d(ctx, type_b, k, n*heads);
        ggml_set_name(b, "b");
        if (heads > 1) {
            b = ggml_view_2d(ctx, b, k, n, heads*b->nb[1], (heads - 1)*b->nb[1]);
            ggml_set_name(b, "b_head");
        }

        ggml_tensor * out = ggml_relu(ctx, ggml_mul_mat(ctx, a, b));
        ggml_set_name(out, "out");

        return out;
    }
};

// the scores of the qwen4exp indexer heads: for each head a MUL_MAT of the same a by the rows of that head in q (the heads
// of a token are adjacent rows), a RELU, the sum over the heads and the add of a bias. backends may do it in one matmul.
// no perf cases: perf mode repeats only the last node
struct test_mul_mat_headsum : public test_case {
    const int64_t m;
    const int64_t n;
    const int64_t k;
    const int64_t heads;
    const bool bias;
    const ggml_type type_a; // the key pool: f16 by default since 09.10, f32 with LLAMA_QSA_POOL_F32

    std::string vars() override {
        return VARS_TO_STR6(type_a, m, n, k, heads, bias);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "MUL_MAT_HEADSUM";
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    bool run_whole_graph() override { return true; }

    test_mul_mat_headsum(int64_t m = 64, int64_t n = 32, int64_t k = 128, int64_t heads = 4, bool bias = true,
                         ggml_type type_a = GGML_TYPE_F32)
        : m(m), n(n), k(k), heads(heads), bias(bias), type_a(type_a) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_2d(ctx, type_a, k, m);
        ggml_set_name(a, "a");

        ggml_tensor * q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, heads*n);
        ggml_set_name(q, "q");

        ggml_tensor * out = nullptr;
        for (int64_t h = 0; h < heads; ++h) {
            ggml_tensor * qh = ggml_view_2d(ctx, q, k, n, heads*q->nb[1], h*q->nb[1]);
            ggml_tensor * score = ggml_relu(ctx, ggml_mul_mat(ctx, a, qh));
            out = out ? ggml_add(ctx, out, score) : score;
        }

        if (bias) {
            ggml_tensor * b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, m, n);
            ggml_set_name(b, "bias");
            out = ggml_add(ctx, out, b);
        }
        ggml_set_name(out, "out");

        return out;
    }
};

// LIGHTNING_INDEXER as the qwen4exp graph builds it: unit head weights made by a FILL of 1
struct test_lightning_indexer_unit : public test_case {
    const int64_t nh;
    const int64_t kv;
    const int64_t nb;
    const ggml_type type_K;

    std::string vars() override {
        return VARS_TO_STR4(type_K, nh, kv, nb);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "LIGHTNING_INDEXER_UNIT";
    }

    double max_nmse_err() override {
        return 5e-4;
    }

    bool run_whole_graph() override { return true; }

    test_lightning_indexer_unit(int64_t nh = 4, int64_t kv = 1000, int64_t nb = 300, ggml_type type_K = GGML_TYPE_F16)
        : nh(nh), kv(kv), nb(nb), type_K(type_K) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 128, nh, nb, 1);
        ggml_set_name(q, "q");

        ggml_tensor * k = ggml_new_tensor_4d(ctx, type_K, 128, 1, kv, 1);
        ggml_set_name(k, "k");

        ggml_tensor * w = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, nh, nb, 1, 1), 1.0f);
        ggml_set_name(w, "w");

        ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, kv, nb, 1, 1);
        ggml_set_name(m, "m");

        ggml_tensor * out = ggml_lightning_indexer(ctx, q, k, w, m);
        ggml_set_name(out, "out");

        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t)) {
            if (strcmp(t->name, "m") == 0) {
                init_tensor_kq_mask(t);
            } else if (t->op == GGML_OP_NONE) {
                init_tensor_uniform(t);
            }
        }
    }
};

// the qwen4exp hc combine: SCALE, SIGMOID, SCALE of the injection and DSV4_HC_POST with it as the post weights.
// Vulkan and CUDA run the four as one HC_POST (HC_POST_GATE)
struct test_hc_post_gate : public test_case {
    const int64_t n_embd;
    const int64_t n_tokens;
    const bool    comb;

    std::string vars() override {
        return VARS_TO_STR3(n_embd, n_tokens, comb);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "HC_POST_GATE";
    }

    bool run_whole_graph() override { return true; }

    test_hc_post_gate(int64_t n_embd = 2560, int64_t n_tokens = 3, bool comb = false)
        : n_embd(n_embd), n_tokens(n_tokens), comb(comb) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t hc = 4;
        ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd, n_tokens);
        ggml_set_name(x, "x");
        ggml_tensor * residual = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_embd, hc, n_tokens);
        ggml_set_name(residual, "residual");
        ggml_tensor * inject = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hc, n_tokens);
        ggml_set_name(inject, "inject");
        ggml_tensor * c = nullptr;
        if (comb) {
            c = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hc, hc, n_tokens);
            ggml_set_name(c, "comb");
        }
        ggml_tensor * w = ggml_scale(ctx, ggml_sigmoid(ctx, ggml_scale(ctx, inject, 1.0f / (float) hc)), 2.0f);
        ggml_tensor * out = ggml_dsv4_hc_post(ctx, x, residual, w, c);
        ggml_set_name(out, "out");
        return out;
    }
};

// the gated norm of the qwen4exp GDN output: RMS_NORM over rows of ne[0], MUL by a weight row, SIGMOID of the gate, MUL.
// pad > 0 reads a through a view with longer rows. backends may run it as one shader (Vulkan: RMS_NORM_MUL_SIGMOID_MUL)
struct test_rms_norm_gated : public test_case {
    const std::array<int64_t, 4> ne;
    const int64_t pad;

    std::string vars() override {
        return VARS_TO_STR2(ne, pad);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RMS_NORM_MUL_SIGMOID_MUL";
    }

    bool run_whole_graph() override { return true; }

    test_rms_norm_gated(std::array<int64_t, 4> ne = {128, 48, 33, 1}, int64_t pad = 0) : ne(ne), pad(pad) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * a = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0] + pad, ne[1], ne[2], ne[3]);
        ggml_set_name(a, "a");
        if (pad > 0) {
            a = ggml_view_4d(ctx, a, ne[0], ne[1], ne[2], ne[3], a->nb[1], a->nb[2], a->nb[3], 0);
            ggml_set_name(a, "view of a");
        }
        ggml_tensor * w = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne[0]);
        ggml_set_name(w, "w");
        ggml_tensor * z = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ne[0], ne[1], ne[2], ne[3]);
        ggml_set_name(z, "z");

        ggml_tensor * out = ggml_mul(ctx, ggml_mul(ctx, ggml_rms_norm(ctx, a, 1e-6f), w), ggml_sigmoid(ctx, z));
        ggml_set_name(out, "out");

        return out;
    }
};

// the l2 norm of the qwen4exp GDN q and k (build_gdn_l2_norm): RMS_NORM with eps/n, then SCALE by 1/sqrt(n), over heads
// of ne[0] read from rows of row_len (a view of the conv output). backends may fuse the scale (Vulkan: RMS_NORM_SCALE)
struct test_rms_norm_scale : public test_case {
    const std::array<int64_t, 3> ne;
    const int64_t row_len;

    std::string vars() override {
        return VARS_TO_STR2(ne, row_len);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "RMS_NORM_SCALE";
    }

    bool run_whole_graph() override { return true; }

    test_rms_norm_scale(std::array<int64_t, 3> ne = {128, 16, 33}, int64_t row_len = 10240) : ne(ne), row_len(row_len) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * src = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, row_len, ne[2]);
        ggml_set_name(src, "src");
        ggml_tensor * a = ggml_view_3d(ctx, src, ne[0], ne[1], ne[2], ne[0]*ggml_element_size(src), src->nb[1], ne[0]*ggml_element_size(src));
        ggml_set_name(a, "a");

        const float n = (float) ne[0];
        ggml_tensor * out = ggml_scale(ctx, ggml_rms_norm(ctx, a, 1e-6f/n), 1.0f/sqrtf(n));
        ggml_set_name(out, "out");

        return out;
    }
};

// the conv of a GDN layer: CONCAT of the conv state [ncs, nr, ns] and the transposed projection x [nr, nt, ns] along
// time, SSM_CONV with a [ncs + 1, nr] kernel, SILU. backends may read the state and x in the conv (Vulkan:
// CONCAT_SSM_CONV_SILU)
struct test_concat_ssm_conv : public test_case {
    const int64_t nr;
    const int64_t nt;
    const int64_t ns;
    const int64_t ncs;
    const bool silu;

    std::string vars() override {
        return VARS_TO_STR5(nr, nt, ns, ncs, silu);
    }

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "CONCAT_SSM_CONV";
    }

    bool run_whole_graph() override { return true; }

    test_concat_ssm_conv(int64_t nr = 1024, int64_t nt = 33, int64_t ns = 1, int64_t ncs = 3, bool silu = true)
        : nr(nr), nt(nt), ns(ns), ncs(ncs), silu(silu) {}

    ggml_tensor * build_graph(ggml_context * ctx) override {
        ggml_tensor * st = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ncs, nr, ns);
        ggml_set_name(st, "state");
        ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, nr, nt, ns);
        ggml_set_name(x, "x");
        ggml_tensor * w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncs + 1, nr);
        ggml_set_name(w, "w");

        ggml_tensor * out = ggml_ssm_conv(ctx, ggml_concat(ctx, st, ggml_transpose(ctx, x), 0), w);
        if (silu) {
            out = ggml_silu(ctx, out);
        }
        ggml_set_name(out, "out");

        return out;
    }
};

// GATED_DELTA_NET_LAZY, the qwen4exp delta net with deferred rollback (llama_memory_recurrent::rec_l). mode 1 runs the
// net from the rows of its cells with their records and writes the records of its tokens, mode 2 replays records into
// the rows the plain net reads. The info rows take the cases llama_memory_recurrent_context::lazy_info makes: a base
// with records, a rolled-back plane without them, a state left as it is, a base copied to a plane. The state rows and
// the records are read back after the op (each node runs alone, in graph order) and checked with its output
struct test_gated_delta_net_lazy : public test_case {
    const int     mode;
    const int64_t head_size;
    const int64_t H_v;
    const int64_t H_k;
    const int64_t n_seq_tokens;
    const int64_t n_seqs; // the info rows: the sequences of mode 1
    const int64_t R;      // the planes of a cell and the records of a set (1 + n_rs_seq)

    ggml_tensor * info = nullptr;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GATED_DELTA_NET_LAZY";
    }

    std::string vars() override {
        return VARS_TO_STR7(mode, head_size, H_v, H_k, n_seq_tokens, n_seqs, R);
    }

    test_gated_delta_net_lazy(int mode, int64_t head_size, int64_t H_v, int64_t H_k, int64_t n_seq_tokens, int64_t n_seqs,
                              int64_t R = 4)
        : mode(mode), head_size(head_size), H_v(H_v), H_k(H_k), n_seq_tokens(n_seq_tokens), n_seqs(n_seqs), R(R) {}

    // one cell per info row and a spare one
    int64_t n_cells() const { return n_seqs + 1; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t S          = head_size;
        const int64_t rec_floats = H_v * S + H_k * S + H_v;

        ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S * H_v, n_cells() * R);
        ggml_tensor * rec   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rec_floats, R, 2 * n_cells());
        info = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 5, n_seqs);
        ggml_set_name(state, "state");
        ggml_set_name(rec,   "rec");
        ggml_set_name(info,  "info");

        ggml_tensor * res = nullptr;
        if (mode == 1) {
            ggml_tensor * q    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, n_seq_tokens, n_seqs);
            ggml_tensor * k    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_k, n_seq_tokens, n_seqs);
            ggml_tensor * v    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, S, H_v, n_seq_tokens, n_seqs);
            ggml_tensor * g    = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, n_seq_tokens, n_seqs);
            ggml_tensor * beta = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, H_v, n_seq_tokens, n_seqs);
            ggml_set_name(q, "q");
            ggml_set_name(k, "k");
            ggml_set_name(v, "v");
            ggml_set_name(g, "g");
            ggml_set_name(beta, "beta");
            q = ggml_l2_norm(ctx, q, 1e-6f);
            k = ggml_l2_norm(ctx, k, 1e-6f);
            res = ggml_gated_delta_net_lazy(ctx, q, k, v, g, beta, state, rec, info);
        } else {
            res = ggml_gated_delta_net_replay(ctx, state, rec, info, (int) S, (int) H_v, (int) H_k);
        }
        ggml_set_name(res, "res");

        // the add visits its first source first, so the reads come after the op
        ggml_tensor * st = ggml_cont(ctx, ggml_view_2d(ctx, state, state->ne[0], state->ne[1], state->nb[1], 0));
        ggml_tensor * rc = ggml_cont(ctx, ggml_view_3d(ctx, rec, rec->ne[0], rec->ne[1], rec->ne[2], rec->nb[1], rec->nb[2], 0));
        ggml_set_name(st, "state_after");
        ggml_set_name(rc, "rec_after");
        return ggml_add(ctx, ggml_sum(ctx, res), ggml_add(ctx, ggml_sum(ctx, st), ggml_sum(ctx, rc)));
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (ggml_is_view_op(t->op) || t->op != GGML_OP_NONE) {
                continue;
            }
            if (t == info) {
                const int64_t   C = n_cells();
                std::vector<int32_t> inf(5 * n_seqs);
                for (int64_t i = 0; i < n_seqs; i++) {
                    const int32_t c   = (int32_t) i;
                    const int32_t rho = (int32_t) (1 + i % (R - 1));
                    const int32_t a   = (int32_t) (i & 1);
                    int32_t *     r   = inf.data() + 5 * i;
                    r[3] = 2 * c + a;
                    r[4] = 2 * c + (a ^ 1);
                    if (mode == 1) {
                        // a base with records, the rolled-back plane of the plain net, a state without records
                        r[1] = c;
                        switch (i % 3) {
                            case 0:  r[0] = c;                 r[2] = (int32_t) (1 + i % R); break;
                            case 1:  r[0] = rho * (int32_t) C + c; r[2] = 0;                   break;
                            default: r[0] = c;                 r[2] = 0;                   break;
                        }
                    } else {
                        // records into a plane, records in place, nothing, a base copied to a plane
                        r[0] = c;
                        r[4] = r[3];
                        switch (i % 4) {
                            case 0:  r[1] = rho * (int32_t) C + c; r[2] = (int32_t) (1 + i % R); break;
                            case 1:  r[1] = c;                 r[2] = (int32_t) (1 + i % R); break;
                            case 2:  r[1] = c;                 r[2] = 0;                   break;
                            default: r[1] = rho * (int32_t) C + c; r[2] = 0;                   break;
                        }
                    }
                }
                ggml_backend_tensor_set(t, inf.data(), 0, ggml_nbytes(t));
            } else if (strcmp(t->name, "g") == 0) {
                init_tensor_uniform(t, -20.0f, -1e-4f);
            } else if (strcmp(t->name, "beta") == 0) {
                init_tensor_uniform(t, 0.0f, 1.0f);
            } else if (strcmp(t->name, "v") == 0) {
                init_tensor_uniform(t, -0.3f, 5.0f);
            } else if (strcmp(t->name, "rec") == 0) {
                // decays in (0, 1], keys and deltas of either sign do no harm over a few records
                init_tensor_uniform(t, 0.0f, 1.0f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

// the replay of llama_memory_recurrent::rec_flatten: one graph, a replay per layer (two states, one info row), run as a
// whole on the device of the layers
struct test_gated_delta_net_replay_layers : public test_case {
    const int64_t head_size;
    const int64_t H_v;
    const int64_t H_k;
    const int64_t R = 4;

    ggml_tensor * out1 = nullptr;
    ggml_tensor * out2 = nullptr;

    std::string op_desc(ggml_tensor * t) override {
        GGML_UNUSED(t);
        return "GATED_DELTA_NET_REPLAY_LAYERS";
    }

    std::string vars() override {
        return VARS_TO_STR3(head_size, H_v, H_k);
    }

    test_gated_delta_net_replay_layers(int64_t head_size, int64_t H_v, int64_t H_k) : head_size(head_size), H_v(H_v), H_k(H_k) {}

    bool run_whole_graph() override { return true; }

    std::vector<ggml_tensor *> fusion_test_nodes() override {
        return { out1, out2 };
    }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        const int64_t S          = head_size;
        const int64_t rec_floats = H_v * S + H_k * S + H_v;
        ggml_tensor * info = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 5, 1);
        ggml_set_name(info, "info");
        ggml_tensor * outs[2];
        for (int l = 0; l < 2; l++) {
            ggml_tensor * state = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, S * S * H_v, 2 * R);
            ggml_tensor * rec   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, rec_floats, R, 4);
            ggml_set_name(rec, "rec");
            outs[l] = ggml_gated_delta_net_replay(ctx, state, rec, info, (int) S, (int) H_v, (int) H_k);
        }
        out1 = outs[0];
        out2 = outs[1];
        return ggml_add(ctx, ggml_sum(ctx, out1), ggml_sum(ctx, out2));
    }

    void initialize_tensors(ggml_context * ctx) override {
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t != nullptr; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE) {
                continue;
            }
            if (t->type == GGML_TYPE_I32) {
                const int32_t inf[5] = { 1, 1, 3, 3, 3 };
                ggml_backend_tensor_set(t, inf, 0, sizeof(inf));
            } else if (strcmp(t->name, "rec") == 0) {
                init_tensor_uniform(t, 0.0f, 1.0f);
            } else {
                init_tensor_uniform(t);
            }
        }
    }
};

static void make_test_cases_eval_fork(std::vector<std::unique_ptr<test_case>> & test_cases) {
    // the delta net with deferred rollback: the qwen4exp head sizes, grouped keys and not, a decode step and a verify
    for (int64_t S : {16, 32, 64, 128}) {
        for (auto hk : std::vector<std::array<int64_t, 2>>{{4, 2}, {2, 2}}) {
            for (int64_t nt : {1, 2, 4}) {
                for (int64_t ns : {1, 3, 7}) {
                    test_cases.emplace_back(new test_gated_delta_net_lazy(1, S, hk[0], hk[1], nt, ns));
                }
            }
            test_cases.emplace_back(new test_gated_delta_net_lazy(2, S, hk[0], hk[1], 1, 4));
            test_cases.emplace_back(new test_gated_delta_net_lazy(2, S, hk[0], hk[1], 1, 7));
        }
        test_cases.emplace_back(new test_gated_delta_net_replay_layers(S, 4, 2));
    }

    // GET_ROWS of short rows (CUDA: one thread per element, fork/getrows-fork.inc): the qwen4exp attention gathers of
    // a 64-token ubatch, the positions of the selected cells (rows of 1, flat), the mask values (rows of 1, per token)
    // and the cells of the selected blocks (rows of 4); a viewed source and index with a column offset
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 1, 4096, 2052*64, 1, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F16, 1, 4096, 2052, 64, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 4, 1024, 513*64, 1, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 100, 30, 70, 3, 2, true, true, 2));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F16, 128, 30, 70, 3, 2, true, true, 1));

    // Vulkan fuses the norm from 32 tokens on (GGML_VK_HC_POST_NORM_MIN_TOKENS)
    test_cases.emplace_back(new test_dsv4_hc_post(1, 1, true, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(31, 17, true, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(31, 33, true, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(257, 40, true, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(2560, 64, true, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(4096, 32, true, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(4097, 32, true, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(31, 33, false, true, true));

    // one- and four-element rows (the qwen4exp sparse attention gathers) and an i32 row wider than 512
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F16, 1, 64, 33, 16, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 4, 32, 100000, 1, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 1000, 5, 4, 2, 1, false));

    // large tile coverage: m and n above 64 select the 128x128 tile on the non-coopmat2 path. n = 65
    // and 129 leave a partial last tile, m = 200 a partial row tile, k = 1056 the unaligned pipeline
    for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0}) {
        for (int n : {65, 129, 512}) {
            test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 256, n, 1024, {1, 1}, {1, 1}));
        }
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 200, 129, 1024, {1, 1}, {1, 1}));
    }
    for (ggml_type type_a : {GGML_TYPE_F16, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
        test_cases.emplace_back(new test_mul_mat(type_a, GGML_TYPE_F32, 256, 129, 1056, {1, 1}, {1, 1}));
    }
    for (ggml_type type_a : {GGML_TYPE_F16, GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
        for (int n : {65, 129, 512}) {
            test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 4, 4, false, 256, n, 1024));
        }
        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 16, 2, false, 256, 256, 1024));
    }

    // 512 experts at prefill ubatch sizes (about 40 and 80 rows per expert)
    for (int n : {2048, 4096}) {
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q4_0, GGML_TYPE_F32, 512, 10, false, 64, n, 256));
    }
    for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (int64_t heads : {1, 4}) {
            test_cases.emplace_back(new test_mul_mat_relu(type_a, GGML_TYPE_F32, 64, 32, 128, heads));
            test_cases.emplace_back(new test_mul_mat_relu(type_a, GGML_TYPE_F32, 1000, 300, 128, heads));
            // a mat-vec size: no fusion, the same result
            test_cases.emplace_back(new test_mul_mat_relu(type_a, GGML_TYPE_F32, 64, 4, 128, heads));
        }
    }
    // the qwen4exp indexer has 4 heads; 1 token is a decode step, 3 a draft check, 2 and 8 heads are not fused
    for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (bool bias : {false, true}) {
            for (auto mn : std::vector<std::array<int64_t, 2>>{{64, 32}, {1000, 300}, {513, 77}, {1000, 1}, {1000, 3}}) {
                test_cases.emplace_back(new test_mul_mat_headsum(mn[0], mn[1], 128, 4, bias, type_a));
            }
            test_cases.emplace_back(new test_mul_mat_headsum(1000, 300, 128, 2, bias, type_a));
            test_cases.emplace_back(new test_mul_mat_headsum(1000, 300, 128, 8, bias, type_a));
        }
    }
    // prompt and decode sizes, 4 and 64 heads
    for (ggml_type type_k : {GGML_TYPE_F32, GGML_TYPE_F16}) {
        for (auto kn : std::vector<std::array<int64_t, 2>>{{1000, 65}, {1000, 300}, {4097, 512}, {513, 77}, {1000, 3}}) {
            test_cases.emplace_back(new test_lightning_indexer_unit(4, kn[0], kn[1], type_k));
        }
        test_cases.emplace_back(new test_lightning_indexer_unit(64, 1000, 300, type_k));
    }
    for (int64_t nt : {1, 3, 77}) {
        for (bool comb : {false, true}) {
            test_cases.emplace_back(new test_hc_post_gate(2560, nt, comb));
        }
    }

    // REPEAT of one element (the -inf the qwen4exp QSA mask starts from): Vulkan and CUDA run it as a fill, CUDA 16
    // bytes a thread, so sizes with a tail past a whole number of 16-byte stores
    for (ggml_type type : {GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I32}) {
        test_cases.emplace_back(new test_repeat(type, {1, 1, 1, 1}, {1, 4096, 33, 1}));
        test_cases.emplace_back(new test_repeat(type, {1, 1, 1, 1}, {7, 13, 3, 2}));
        test_cases.emplace_back(new test_repeat(type, {1, 1, 1, 1}, {1, 1, 1, 1}));
    }

    // MoE cache: 37 slots and the zero slot, n_used 10 uses the templated ids helper and n_used 5 the generic one
    for (ggml_type type_a : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_IQ4_XS, GGML_TYPE_IQ1_M}) {
        for (bool b : {false, true}) {
            for (int n : {1, 8, 9, 24, 31, 128}) {
                test_cases.emplace_back(new test_mul_mat_id_repeat(type_a, GGML_TYPE_F32, 38, 10, b, 256, n, 256));
            }
        }
    }
    for (int n : {9, 64}) {
        test_cases.emplace_back(new test_mul_mat_id_repeat(GGML_TYPE_Q4_0, GGML_TYPE_F32, 17, 5, false, 256, n, 256));
    }

    // transposed sources; no i64, the CPU concat of it needs contiguous rows
    for (int v : { 16, 32, 48 }) {
        for (int dim : { 0, 1, 2, 3, }) {
            for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16, GGML_TYPE_I8, GGML_TYPE_I16, GGML_TYPE_I32 }) {
                test_cases.emplace_back(new test_concat(type, {11, 12, 13, 14}, 7, dim, v));
            }
        }
    }

    // conv state + transposed tokens, as before ssm_conv: many tiles, with partial tiles at the edges
    for (ggml_type type : { GGML_TYPE_F32, GGML_TYPE_F16 }) {
        test_cases.emplace_back(new test_concat(type, {3, 1100, 2, 1}, 67, 0, 16));
        test_cases.emplace_back(new test_concat(type, {45, 70, 3, 1}, 37, 1, 32));
    }

    // sparse mask, qwen4exp QSA shape: 24 q heads over 2 kv heads, head_dim 256
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 4096,  1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 4096, 64, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048));

    // same, with the finite entries in aligned runs of 4 cells as the QSA block compression leaves them.
    // the last one has whole groups of query rows share a selection, which is what leaves a mask tile
    // entirely -inf and lets a backend skip it - that skip stays untested otherwise
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 4096,   1, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048, 4));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 4096,  64, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048, 4));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 4096, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048, 4));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 4096, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048, 4, 64));

    // sparse prefill with few query rows: the heads of a group share the exact list of each row,
    // and the device is not full, so the list is also split over several workgroups
    test_cases.emplace_back(new test_flash_attn_ext(128, 128, 1, { 8, 1}, 8192, 12, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false,  512));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 8192, 16, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2052, 4));

    // sparse mask where the rows of a tile pick different cells, so a tile cannot share one list
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16384,  64, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 1024, 4,  8));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 16384, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 1024, 4,  8));
    test_cases.emplace_back(new test_flash_attn_ext(128, 128, 1, { 8, 1}, 16384, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false,  512, 1, 16));

    // the depth the model is measured at: a 64k cache and a per-token selection of 2048 cells.
    // The Vulkan backend folds the 12 heads of a group into one tile here; without the fold the tile
    // walks the union of its rows, and without the union it re-tunes to a one-row tile
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 65536,  64, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048, 4));
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 65536, 512, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048, 4));
    // the budget the model really asks for: top_k 2048 plus the tail of the last block, so 2052
    // cells, which is not a multiple of the 64-cell tile the backends walk
    test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 65536,  64, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2052, 4));

    // prefill-sized batches go to matmul_id; on the Vulkan int8 coopmat path the experts' gate and up projections
    // and the swiglu run as one shader there (FUSED_GLU). q6_K has no fused shader and checks the plain path
    for (ggml_type type : {GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL,
                           GGML_TYPE_IQ4_XS, GGML_TYPE_MXFP4, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}) {
        for (bool b : {false, true}) {
            for (int64_t m_batch : {16, 64, 200}) {
                test_cases.emplace_back(new test_mul_mat_vec_fusion(type, GGML_GLU_OP_SWIGLU, m_batch, 96, 256,
                    true, 16, 8, b, false, true, false, {1, 1}));
            }
        }
    }

    for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q4_K}) {
        test_cases.emplace_back(new test_mul_mat_vec_fusion(type, GGML_GLU_OP_SWIGLU, 32, 32, 256,
            true, 8, 2, false, false, true, false, {1, 1}, true));
    }

    test_cases.emplace_back(new test_gated_delta_net_cache_fusion(GGML_TYPE_F32, 4, 32,   2, 1, 2, 1));
    test_cases.emplace_back(new test_gated_delta_net_cache_fusion(GGML_TYPE_F32, 4, 32,   2, 1, 2, 2));

    // rms_norm_rows.comp on Vulkan: rows up to 512 columns, a row count that does not fill the last workgroup
    for (int64_t n : {1, 7, 33}) {
        test_cases.emplace_back(new test_rms_norm_gated({128, 48, n, 1}, 0));
        test_cases.emplace_back(new test_rms_norm_gated({128, 48, n, 1}, 64));
        test_cases.emplace_back(new test_rms_norm_scale({128, 16, n}, 10240));
    }
    test_cases.emplace_back(new test_rms_norm_gated({256, 3, 5, 2}, 0));
    test_cases.emplace_back(new test_rms_norm_gated({512, 4, 3, 1}, 0));
    test_cases.emplace_back(new test_rms_norm_gated({100, 5, 3, 1}, 7));
    test_cases.emplace_back(new test_rms_norm_scale({64, 4, 9}, 1000));
    test_cases.emplace_back(new test_rms_norm_scale({513, 2, 3}, 2000));
    for (int64_t nt : {1, 4, 33, 512}) {
        test_cases.emplace_back(new test_concat_ssm_conv(1024, nt, 1, 3, true));
    }
    test_cases.emplace_back(new test_concat_ssm_conv(100, 17, 2, 3, true));
    test_cases.emplace_back(new test_concat_ssm_conv(64, 9, 1, 3, false));

    // mul_mmid_f16reg.comp on Vulkan (GGML_VK_MMID_F16REG): rows in 128-row tiles, K in 64-column stages, from 16 rows per
    // expert on average; an expert with more rows than a 64-token tile, the router weight scale, the fused gate/up
    for (ggml_type type_a : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0}) {
        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 16, 4, false, 256, 64, 512));
        test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 8, 4, false, 384, 200, 768));
        test_cases.emplace_back(new test_mul_mat_id_fusion(type_a, GGML_TYPE_F32, 16, 4, false, 256, 64, 512, 1, true));
    }
    for (ggml_type type_a : {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K}) {
        test_cases.emplace_back(new test_mul_mat_vec_fusion(type_a, GGML_GLU_OP_SWIGLU, 200, 256, 512,
            true, 8, 4, true, false, true, false, {1, 1}));
    }

    // the qwen4exp block top-k (513) of a 4096-token ubatch at n_kv 4352, 8448 and 15104, and one column past the
    // widest workgroup of the tournament (1024): radix-select on Vulkan there from 8 rows (ggml_vk_topk_use_radix)
    for (int64_t n_blocks : {1088, 2112, 3776}) {
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {n_blocks, 4096, 1, 1}, 513));
    }
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {1025, 16, 1, 1}, 513));
    test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {1025, 16, 1, 1}, 513, true));
}

static void make_test_cases_perf_fork(std::vector<std::unique_ptr<test_case>> & test_cases) {
    // qwen4exp prefill: the hyper-connection down projection [hc*n_embd] -> [hc_lr] (m 320, k 10240) at four
    // ubatch sizes, and q8_0 neighbours of the same graph to tell which of m, k and n makes it slow
    for (int64_t n : {512, 1024, 2048, 4096}) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 320, n, 10240, {1, 1}, {1, 1}));
    }
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32,  320, 2048,  2560, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32,  640, 2048, 10240, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, 2560, 2048, 10240, {1, 1}, {1, 1}));
    test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32,  640, 2048,  2560, {1, 1}, {1, 1}));

    // qwen4exp prefill at -ub 4096: the other q8_0 projections of the graph by time (attention and GDN in and out,
    // the shared expert, the hyper-connection up projection), for tile sweeps
    for (auto [m, k] : std::vector<std::pair<int64_t, int64_t>>{{10240, 2560}, {2560, 6144}, {6144, 2560}, {12288, 2560}, {10240, 320}, {640, 2560}}) {
        test_cases.emplace_back(new test_mul_mat(GGML_TYPE_Q8_0, GGML_TYPE_F32, m, 4096, k, {1, 1}, {1, 1}));
    }

    // qwen4exp: 512 experts, 10 per token. gate/up is q4_K [640, 2560] and down is q5_1 [2560, 640],
    // and at prefill q4_K reads its weights at about half the rate of q5_1. q4_0 runs the same shape
    // at the same 0.5625 bytes per weight, so it tells the k-quant path apart from the bit width.
    // n = 2048, 4096 and 8192 are prefill at those -ub, about 40, 80 and 160 rows per expert
    for (int bs : {512, 2048, 4096, 8192}) {
        for (ggml_type type_a : {GGML_TYPE_Q4_K, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1}) {
            test_cases.emplace_back(new test_mul_mat_id(type_a, GGML_TYPE_F32, 512, 10, false, 640, bs, 2560));
        }
        test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_Q5_1, GGML_TYPE_F32, 512, 10, false, 2560, bs, 640));
    }
    // the same gate and up with the swiglu as one graph, which Vulkan fuses into one shader (FUSED_GLU) from
    // GGML_VK_MMID_GLU_MIN_TOKENS tokens on: a short prompt and -ub 4096
    for (int64_t bs : {512, 4096}) {
        test_cases.emplace_back(new test_mul_mat_vec_fusion(GGML_TYPE_Q4_K, GGML_GLU_OP_SWIGLU, bs, 640, 2560,
            true, 512, 10, true, false, true, false, {1, 1}));
    }

    // qwen4exp hc_combine at -ub 4096, alone and with the RMSNorm of the next hc mix (HC_POST_NORM on Vulkan)
    test_cases.emplace_back(new test_dsv4_hc_post(2560, 4096, true, true));
    test_cases.emplace_back(new test_dsv4_hc_post(2560, 4096, true, true, true));

    // sparse prefill, qwen4exp QSA shape and 2048-cell budget over a 32k cache.
    // n_kv_max=0 is the dense causal mask, the baseline the sparse cases have to beat.
    // blk is the run length of the selection, grp how many query rows share one selection:
    // a backend that only skips whole mask tiles needs both to be large, a row-granular
    // gather would run at the same speed for every combination.
    for (int64_t nb : {512, 2048}) {
        test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 32768, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 0));
        for (int64_t grp : {1, 16, 64}) {
            for (int64_t blk : {1, 4}) {
                test_cases.emplace_back(new test_flash_attn_ext(256, 256, 2, {12, 1}, 32768, nb, true, false, 0, 0, GGML_PREC_F32, GGML_TYPE_F16, GGML_TYPE_F16, {0, 1, 2, 3}, true, false, 2048, blk, grp));
            }
        }
    }

    // the same for the block top-k (513 blocks of 4 cells), and a 512-token ubatch
    for (auto cols : {8192, 32768}) {
        for (auto nrows : {1, 4, 16, 64, 512}) {
            test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {cols, nrows, 1, 1}, 513));
        }
    }

    test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {3,  3328, 1, 1},  512, 0, 16)); // prefill conv input: state + transposed x
    test_cases.emplace_back(new test_concat(GGML_TYPE_F32, {3, 10240, 1, 1}, 2048, 0, 16)); // same, qwen4exp with a 2048 ubatch
    // qwen4exp GDN at a 4096-token ubatch: the conv over the concat sources, the gated output norm, the q/k l2 norm
    test_cases.emplace_back(new test_concat_ssm_conv(10240, 4096, 1, 3, true));
    test_cases.emplace_back(new test_rms_norm_gated({128, 48, 4096, 1}, 0));
    test_cases.emplace_back(new test_rms_norm_scale({128, 16, 4096}, 10240));

    // qwen4exp sparse attention gathers at a 4096-token ubatch: the KQ mask value of each selected cell
    // (one-element rows) and the cells of each selected 4-cell block
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_F16, 1, 4096, 2052, 4096, 1, false));
    test_cases.emplace_back(new test_get_rows(GGML_TYPE_I32, 4, 1024, 513*4096, 1, 1, false));

    // qwen4exp indexer at a 4096-token ubatch over 4-cell blocks, n_kv 32768 and 126976: the top-k of 513 blocks
    for (int64_t n_blocks : {8192, 31744}) {
        test_cases.emplace_back(new test_top_k(GGML_TYPE_F32, {n_blocks, 4096, 1, 1}, 513));
    }
}
