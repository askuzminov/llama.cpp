// the fork's additions to the scheduler, included at the end of ggml-backend.cpp: they use its internal structs.
// ggml-backend-fork-decl.inc declares them where ggml-backend.cpp uses them first

// GGML_SCHED_LOG_REALLOC: the splits of the graph, with the nodes of every split that is not on the first backend
static void ggml_backend_sched_log_splits(ggml_backend_sched_t sched, const char * what) {
    fprintf(stderr, "sched %s: %d nodes, %d splits\n", what, sched->graph.n_nodes, sched->n_splits);
    for (int i = 0; i < sched->n_splits; i++) {
        const struct ggml_backend_sched_split * split = &sched->splits[i];
        fprintf(stderr, "  split %d: %s, nodes %d to %d, %d inputs\n", i, ggml_backend_name(sched->backends[split->backend_id]),
                split->i_start, split->i_end, split->n_inputs);
        for (int j = 0; split->backend_id != 0 && j < split->graph.n_nodes && j < 8; j++) {
            const struct ggml_tensor * t = split->graph.nodes[j];
            fprintf(stderr, "    %s %s [%" PRId64 ", %" PRId64 ", %" PRId64 ", %" PRId64 "] %zu bytes\n", t->name, ggml_op_desc(t),
                    t->ne[0], t->ne[1], t->ne[2], t->ne[3], ggml_nbytes(t));
        }
    }
}

// GGML_SCHED_PROF_GPU=1 with GGML_SCHED_PROF: the device time of the compute calls of each backend that reports it
// (CUDA: timing events around each call), "device busy" per graph next to the host times
static void ggml_backend_sched_prof_gpu_init(ggml_backend_sched_t sched) {
    const char * env = getenv("GGML_SCHED_PROF_GPU");
    if (sched->prof_period <= 0 || env == nullptr || atoi(env) == 0) {
        return;
    }
    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(sched->prof_devs[b]);
        if (reg == nullptr) {
            continue;
        }
        auto enable = (ggml_backend_timing_enable_fork_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_timing_enable_fork");
        auto take   = (ggml_backend_timing_take_fork_t)   ggml_backend_reg_get_proc_address(reg, "ggml_backend_timing_take_fork");
        if (enable != nullptr && take != nullptr) {
            enable(sched->backends[b], true);
            sched->prof_gpu[b] = take;
        }
    }
}

// at the start of a graph: the calls that ended since the last take, those of the previous graph once it synced
static void ggml_backend_sched_prof_gpu_take(ggml_backend_sched_t sched, int bucket) {
    for (int b = 0; b < sched->n_backends; b++) {
        if (sched->prof_gpu[b] != nullptr) {
            double  busy_us = 0.0;
            int64_t n_calls = 0;
            sched->prof_gpu[b](sched->backends[b], &busy_us, &n_calls);
            sched->prof[bucket].t_gpu[b] += busy_us;
        }
    }
}

static void ggml_backend_sched_prof_print(ggml_backend_sched_t sched, int bucket) {
    struct ggml_backend_sched_prof * p = &sched->prof[bucket];

    const double n = (double) p->n_graphs;

    char   buf[1024];
    int    len  = 0;
    double busy = 0.0;
    for (int b = 0; b < sched->n_backends && len >= 0 && len < (int) sizeof(buf); b++) {
        if (p->n_split[b] == 0) {
            continue;
        }
        const double t_in   = 1e-3*(double) p->t_in[b]/n;
        const double t_wait = 1e-3*(double) p->t_wait[b]/n;
        const double t_run  = 1e-3*(double) p->t_run[b]/n;
        busy += t_in + t_run;
        len += snprintf(buf + len, sizeof(buf) - len, "%s %s %.0f splits, wait %.2f, copy %.2f, run %.2f ms",
                len > 0 ? ";" : "", ggml_backend_dev_name(sched->prof_devs[b]), (double) p->n_split[b]/n, t_wait, t_in - t_wait, t_run);
        if (sched->prof_gpu[b] != nullptr && p->n_wall > 0 && len >= 0 && len < (int) sizeof(buf)) {
            len += snprintf(buf + len, sizeof(buf) - len, ", device busy %.2f ms", 1e-3*p->t_gpu[b]/(double) p->n_wall);
        }
        if (p->n_overlap[b] > 0 && len >= 0 && len < (int) sizeof(buf)) {
            len += snprintf(buf + len, sizeof(buf) - len, ", %.0f CPU waits avoided", (double) p->n_overlap[b]/n);
        }
    }

    char splits[32];
    if (p->n_splits > 0) {
        snprintf(splits, sizeof(splits), "%d", p->n_splits);
    } else {
        snprintf(splits, sizeof(splits), "other");
    }

    if (p->n_wall > 0) {
        const double t_wall = 1e-3*(double) p->t_wall/(double) p->n_wall;
        GGML_LOG_WARN("sched_prof %d: %" PRId64 " graphs of %s splits, %.2f ms per graph:%s; outside %.2f ms\n",
                sched->prof_id, p->n_graphs, splits, t_wall, buf, t_wall - busy);
    } else {
        GGML_LOG_WARN("sched_prof %d: %" PRId64 " graphs of %s splits:%s\n", sched->prof_id, p->n_graphs, splits, buf);
    }

    const int n_splits = p->n_splits;
    memset(p, 0, sizeof(*p));
    p->n_splits = n_splits;
}

static bool ggml_backend_sched_split_uses_host_buffer(const ggml_backend_sched_split * split) {
    auto is_host = [](const ggml_tensor * t) {
        if (t == nullptr) {
            return false;
        }
        const ggml_tensor * base = t->view_src ? t->view_src : t;
        return base->buffer == nullptr || ggml_backend_buffer_is_host(base->buffer);
    };

    for (int i = 0; i < split->n_inputs; ++i) {
        if (is_host(split->inputs[i])) {
            return true;
        }
    }
    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const ggml_tensor * t = split->graph.nodes[i];
        if (is_host(t)) {
            return true;
        }
        for (const ggml_tensor * src : t->src) {
            if (is_host(src)) {
                return true;
            }
        }
    }
    return false;
}

// graph state after the split and the allocation of a graph
struct ggml_backend_sched_snapshot {
    ggml_backend_sched_t sched;
    size_t n_realloc; // buffer reallocations of the scheduler at the save

    std::vector<ggml_backend_sched_split> splits; // the inputs of each split point into inputs
    std::vector<ggml_tensor *> inputs;            // inputs of all splits
    std::vector<ggml_tensor>   copies;            // copy of each input in the backend of its split, the nodes use these copies

    // backends of the graph tensors
    struct ggml_hash_set hash_set;
    std::vector<int>     backend_ids; // [hash_set.size]
};

static int ggml_backend_sched_snapshot_backend_id(const ggml_backend_sched_snapshot * snapshot, const struct ggml_tensor * tensor) {
    const size_t id = ggml_hash_find(&snapshot->hash_set, tensor);
    if (id == GGML_HASHSET_FULL || !ggml_bitset_get(snapshot->hash_set.used, id)) {
        return -1;
    }
    return snapshot->backend_ids[id];
}

ggml_backend_sched_snapshot_t ggml_backend_sched_snapshot_save(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    GGML_ASSERT(sched->is_alloc);

    // pipeline parallelism uses more than one copy of each input
    if (sched->n_copies > 1) {
        return NULL;
    }

    // a buffer reset releases the tensor extras, so the allocation of the next graph makes the tensors of this graph not valid
    auto has_reset = [](const struct ggml_tensor * t) {
        return t->buffer != NULL && t->buffer->usage == GGML_BACKEND_BUFFER_USAGE_COMPUTE && t->buffer->iface.reset != NULL;
    };

    int n_inputs  = 0;
    int n_tensors = sched->graph.n_leafs;
    for (int i = 0; i < sched->graph.n_leafs; i++) {
        if (has_reset(sched->graph.leafs[i])) {
            return NULL;
        }
    }
    for (int i = 0; i < sched->n_splits; i++) {
        const struct ggml_backend_sched_split * split = &sched->splits[i];
        for (int j = 0; j < split->graph.n_nodes; j++) {
            if (has_reset(split->graph.nodes[j])) {
                return NULL;
            }
        }
        for (int j = 0; j < split->n_inputs; j++) {
            if (has_reset(tensor_copy(split->inputs[j], split->backend_id, 0))) {
                return NULL;
            }
        }
        n_inputs  += split->n_inputs;
        n_tensors += split->graph.n_nodes + split->n_inputs;
    }

    ggml_backend_sched_snapshot * snapshot = new ggml_backend_sched_snapshot;
    snapshot->sched     = sched;
    snapshot->n_realloc = ggml_gallocr_get_n_realloc(sched->galloc);
    snapshot->hash_set  = ggml_hash_set_new(2*n_tensors);
    snapshot->backend_ids.assign(snapshot->hash_set.size, -1);

    auto add = [&](struct ggml_tensor * t) {
        snapshot->backend_ids[ggml_hash_find_or_insert(&snapshot->hash_set, t)] = tensor_backend_id(t);
    };

    // with one copy, the leafs of the graph copy are the leafs of the graph
    for (int i = 0; i < sched->graph.n_leafs; i++) {
        add(sched->graph.leafs[i]);
    }

    snapshot->splits.assign(sched->splits, sched->splits + sched->n_splits);
    snapshot->inputs.resize(n_inputs);
    snapshot->copies.resize(n_inputs);

    // the copies of the scheduler are in its context, the next split reuses that memory
    std::unordered_map<const struct ggml_tensor *, struct ggml_tensor *> copies;
    copies.reserve(n_inputs);

    int k = 0;
    for (int i = 0; i < sched->n_splits; i++) {
        const struct ggml_backend_sched_split * split = &sched->splits[i];
        struct ggml_backend_sched_split * split_dst = &snapshot->splits[i];

        split_dst->inputs          = snapshot->inputs.data() + k;
        split_dst->inputs_capacity = split->n_inputs;

        for (int j = 0; j < split->graph.n_nodes; j++) {
            add(split->graph.nodes[j]);
        }
        for (int j = 0; j < split->n_inputs; j++, k++) {
            struct ggml_tensor * input = split->inputs[j];
            add(input);

            struct ggml_tensor ** cpy = &tensor_copy(input, split->backend_id, 0);
            snapshot->inputs[k] = input;
            snapshot->copies[k] = **cpy;
            copies[*cpy] = &snapshot->copies[k];
            *cpy = &snapshot->copies[k];
        }
    }

    // a copy can be the source of nodes in later splits of the same backend
    const uintptr_t ctx_begin = (uintptr_t) sched->context_buffer;
    const uintptr_t ctx_end   = ctx_begin + sched->context_buffer_size;
    auto remap = [&](struct ggml_tensor ** t) {
        if ((uintptr_t) *t >= ctx_begin && (uintptr_t) *t < ctx_end) {
            auto it = copies.find(*t);
            GGML_ASSERT(it != copies.end());
            *t = it->second;
        }
    };
    for (int i = 0; i < sched->n_splits; i++) {
        const struct ggml_backend_sched_split * split = &sched->splits[i];
        for (int j = 0; j < split->graph.n_nodes; j++) {
            struct ggml_tensor * node = split->graph.nodes[j];
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                remap(&node->src[s]);
            }
            remap(&node->view_src);
        }
    }

    return snapshot;
}

bool ggml_backend_sched_snapshot_restore(ggml_backend_sched_t sched, ggml_backend_sched_snapshot_t snapshot) {
    GGML_ASSERT(sched);
    GGML_ASSERT(snapshot && snapshot->sched == sched);

    if (ggml_gallocr_get_n_realloc(sched->galloc) != snapshot->n_realloc) {
        return false;
    }

    const int n_splits = (int) snapshot->splits.size();
    if (n_splits > sched->splits_capacity) {
        const int old_cap = sched->splits_capacity;
        sched->splits_capacity = n_splits;
        sched->splits = (ggml_backend_sched_split *) realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
        GGML_ASSERT(sched->splits != NULL);
        memset(&sched->splits[old_cap], 0, (sched->splits_capacity - old_cap) * sizeof(struct ggml_backend_sched_split));
    }

    // compute_splits finds the input copies in the hash set
    ggml_hash_set_reset(&sched->hash_set);

    int k = 0;
    for (int i = 0; i < n_splits; i++) {
        const struct ggml_backend_sched_split * src = &snapshot->splits[i];
        struct ggml_backend_sched_split * dst = &sched->splits[i];

        while (dst->inputs_capacity < src->n_inputs) {
            ggml_backend_sched_split_inputs_grow(dst);
        }
        for (int j = 0; j < src->n_inputs; j++, k++) {
            struct ggml_tensor * input = src->inputs[j];
            dst->inputs[j] = input;

            size_t id = ggml_hash_insert(&sched->hash_set, input);
            if (id == GGML_HASHSET_ALREADY_EXISTS) {
                id = ggml_hash_find(&sched->hash_set, input);
            } else {
                sched->hv_tensor_backend_ids[id] = ggml_backend_sched_snapshot_backend_id(snapshot, input);
                memset(&tensor_id_copy(id, 0, 0), 0, sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
            }
            tensor_id_copy(id, src->backend_id, 0) = &snapshot->copies[k];
        }
        dst->backend_id = src->backend_id;
        dst->i_start    = src->i_start;
        dst->i_end      = src->i_end;
        dst->n_inputs   = src->n_inputs;
        dst->graph      = src->graph;
    }

    sched->n_splits       = n_splits;
    sched->n_graph_inputs = 0;
    sched->snapshot       = snapshot;
    sched->is_reset       = false;
    sched->is_alloc       = true;

    return true;
}

size_t ggml_backend_sched_get_n_reserve(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_reserve;
}

void ggml_backend_sched_snapshot_free(ggml_backend_sched_snapshot_t snapshot) {
    if (snapshot == NULL) {
        return;
    }
    if (snapshot->sched->snapshot == snapshot) {
        snapshot->sched->snapshot = NULL;
    }
    ggml_hash_set_free(&snapshot->hash_set);
    delete snapshot;
}
