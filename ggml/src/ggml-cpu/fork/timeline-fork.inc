// fork: GGML_TIMELINE on the CPU backend (see ggml_timeline_event in ggml-backend.h). Thread 0 marks the end of each
// node of a graph after the barrier that waits for the slowest thread: a span per node on track "cpu nodes", named by
// the node, with its op and, for the matmuls, the weight bytes it read (a MUL_MAT_ID counts each expert it computes
// once, without the ones its skip list leaves to the MoE cache). "cpu wake" spans the time from the compute call to the
// start of the last thread. Included by ggml-cpu.c after struct ggml_threadpool

struct ggml_cpu_timeline_fork {
    bool    on;
    bool    woke;    // the wake span is written
    int64_t t_prev;  // end of the last node
    const struct ggml_tensor * last; // the last node of the graph, its barrier comes after the loop
};

// the weight bytes a node read: src0 of a matmul, the selected experts of a MUL_MAT_ID
static int64_t ggml_cpu_timeline_fork_bytes(const struct ggml_tensor * node) {
    const struct ggml_tensor * src0 = node->src[0];
    if (node->op == GGML_OP_MUL_MAT && src0 != NULL) {
        return (int64_t) ggml_nbytes(src0);
    }
    if (node->op != GGML_OP_MUL_MAT_ID || src0 == NULL || node->src[2] == NULL) {
        return 0;
    }
    const struct ggml_tensor * ids  = node->src[2];
    const int32_t            * skip = node->src[3] ? (const int32_t *) node->src[3]->data : NULL;

    enum { MAX_EXPERTS = 4096 };
    uint64_t seen[MAX_EXPERTS / 64];
    const int64_t n_as = src0->ne[2];
    if (n_as > MAX_EXPERTS) {
        return 0;
    }
    memset(seen, 0, sizeof(uint64_t) * (size_t) ((n_as + 63) / 64));

    int64_t n = 0;
    for (int64_t i1 = 0; i1 < ids->ne[1]; ++i1) {
        for (int64_t i0 = 0; i0 < ids->ne[0]; ++i0) {
            const int32_t e = *(const int32_t *) ((const char *) ids->data + i1*ids->nb[1] + i0*ids->nb[0]);
            if (e < 0 || e >= n_as || (skip && skip[e]) || (seen[e / 64] >> (e % 64)) & 1) {
                continue;
            }
            seen[e / 64] |= 1ull << (e % 64);
            n++;
        }
    }
    return n * (int64_t) src0->nb[2];
}

static void ggml_cpu_timeline_fork_emit(struct ggml_cpu_timeline_fork * tl, struct ggml_threadpool * tp, const struct ggml_tensor * node) {
    const int64_t t_now = ggml_time_us();

    // every thread has started once thread 0 is past a barrier. no call time: the timeline came on during the graph
    if (!tl->woke && tp->fork_t_call > 0) {
        tl->woke = true;
        const int n_threads = atomic_load_explicit(&tp->n_graph, memory_order_relaxed) & GGML_THREADPOOL_N_THREADS_MASK;
        int64_t t_last = tp->fork_t_call;
        for (int i = 0; i < n_threads && i < GGML_MAX_N_THREADS; ++i) {
            t_last = MAX(t_last, tp->fork_t_enter[i]);
        }
        char args[64];
        snprintf(args, sizeof(args), "\"threads\":%d", n_threads);
        ggml_timeline_event_args("cpu wake", "wake", tp->fork_t_call, t_last, args);
    }

    char name[GGML_MAX_NAME];
    snprintf(name, sizeof(name), "%s", node->name);
    for (char * c = name; *c; ++c) {
        if (*c == '"' || *c == '\\' || (unsigned char) *c < 0x20) {
            *c = '_';
        }
    }
    char args[64 + GGML_MAX_NAME];
    snprintf(args, sizeof(args), "\"op\":\"%s\",\"bytes\":%lld", ggml_op_name(node->op), (long long) ggml_cpu_timeline_fork_bytes(node));
    ggml_timeline_event_args("cpu nodes", name, tl->t_prev, t_now, args);
    tl->t_prev = t_now;
}

// every thread at the start of the graph
static void ggml_cpu_timeline_fork_begin(struct ggml_cpu_timeline_fork * tl, struct ggml_compute_state * state) {
    tl->on   = false;
    tl->woke = false;
    tl->last = NULL;
    if (!ggml_timeline_on()) {
        return;
    }
    const int64_t t_now = ggml_time_us();
    if (state->ith < GGML_MAX_N_THREADS) {
        state->threadpool->fork_t_enter[state->ith] = t_now;
    }
    tl->on     = state->ith == 0;
    tl->t_prev = t_now;
}

// thread 0 after a node and its barrier, the last node of the graph waits for the barrier after the loop
static void ggml_cpu_timeline_fork_node(struct ggml_cpu_timeline_fork * tl, struct ggml_compute_state * state,
        const struct ggml_tensor * node, bool last) {
    if (!tl->on) {
        return;
    }
    if (last) {
        tl->last = node;
        return;
    }
    ggml_cpu_timeline_fork_emit(tl, state->threadpool, node);
}

static void ggml_cpu_timeline_fork_end(struct ggml_cpu_timeline_fork * tl, struct ggml_compute_state * state) {
    if (tl->on && tl->last != NULL) {
        ggml_cpu_timeline_fork_emit(tl, state->threadpool, tl->last);
    }
}
