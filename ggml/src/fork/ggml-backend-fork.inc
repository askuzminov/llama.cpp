// the fork's additions to the scheduler, included at the end of ggml-backend.cpp: they use its internal structs.
// ggml-backend-fork-decl.inc declares them where ggml-backend.cpp uses them first

#include <array>
#include <mutex>
#include <string>

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
        dst->backend_id  = src->backend_id;
        dst->i_start     = src->i_start;
        dst->i_end       = src->i_end;
        dst->n_inputs    = src->n_inputs;
        dst->graph       = src->graph;
        dst->prefetch_at = src->prefetch_at;
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

// GGML_SCHED_PREFETCH=<depth> (3; 0 off; at most GGML_SCHED_PREFETCH_MAX): the weights in host memory that a split copies to
// its device are uploaded while the split <depth> before it computes, on a second instance of the device's backend,
// instead of at the start of the split after a sync of the device. With the experts in RAM (llama.cpp -ncmoe) and the op
// offload of a prompt ubatch, each split of a MoE layer starts with such a copy: on the 3090 (14.10) ~3 s of PCIe per
// ubatch of 3707 tokens in series with ~1.9 s of kernels. The copies are allocated at the start of the split that uploads
// them, so nothing computed in between takes their memory, and an upload waits for the work before that split, whose
// memory it may reuse. ggml-alloc gives a graph the offsets of the reserved graph by node index, so the order of the
// nodes must not depend on the batch: which splits upload ahead is fixed by the splits, and a split that starts with a
// MUL_MAT_ID whose ids select fewer than GGML_SCHED_PREFETCH_LOAD (20) per expert of its weight is copied as before, at
// its start, only the experts its ids name (the copy callback)

// a weight in host memory or a view of one: a view gets its buffer only when the graph is allocated, after the split,
// and the plan must find the same inputs as the compute (the reshaped scale of qwen4exp's experts was left in place
// and its early upload was overwritten)
static bool ggml_backend_sched_prefetch_weight(const struct ggml_tensor * t) {
    return ggml_backend_sched_is_host_weight(t->view_src ? t->view_src : t);
}

static bool ggml_backend_sched_prefetch_full(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split) {
    if (sched->callback_copy == nullptr || split->graph.n_nodes == 0) {
        return true;
    }
    const struct ggml_tensor * node = split->graph.nodes[0];
    if (node->op != GGML_OP_MUL_MAT_ID || node->src[0] == nullptr || node->src[2] == nullptr) {
        return true;
    }
    for (int j = 0; j < split->n_inputs; j++) {
        struct ggml_tensor * input = split->inputs[j];
        if (ggml_backend_sched_prefetch_weight(input) && tensor_copy(input, split->backend_id, sched->cur_copy) == node->src[0]) {
            return ggml_nelements(node->src[2]) >= (int64_t) sched->prefetch.min_load * input->ne[2];
        }
    }
    return true;
}

// the second instance of backend b and its events, false if the device does not give them
static bool ggml_backend_sched_prefetch_ensure(ggml_backend_sched_t sched, int b) {
    struct ggml_backend_sched_prefetch * pf = &sched->prefetch;
    if (pf->copy[b] != nullptr) {
        return true;
    }
    if (pf->failed[b]) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(sched->backends[b]);
    ggml_backend_t copy = ggml_backend_dev_init(dev, nullptr);
    bool ok = copy != nullptr;
    ggml_backend_event_t before = ok ? ggml_backend_event_new(dev) : nullptr;
    ok = ok && before != nullptr;
    ggml_backend_event_t done[GGML_SCHED_PREFETCH_MAX + 1] = {};
    for (int k = 0; k <= GGML_SCHED_PREFETCH_MAX && ok; k++) {
        done[k] = ggml_backend_event_new(dev);
        ok = done[k] != nullptr;
    }
    if (!ok) {
        // the splits keep the allocation of the plan and take the copy at their start
        GGML_LOG_WARN("%s: no second instance or events for %s, the host weights are copied at the start of each split\n",
                __func__, ggml_backend_name(sched->backends[b]));
        for (int k = 0; k <= GGML_SCHED_PREFETCH_MAX; k++) {
            if (done[k] != nullptr) {
                ggml_backend_event_free(done[k]);
            }
        }
        if (before != nullptr) {
            ggml_backend_event_free(before);
        }
        if (copy != nullptr) {
            ggml_backend_free(copy);
        }
        pf->failed[b] = true;
        return false;
    }

    pf->copy[b]   = copy;
    pf->before[b] = before;
    for (int k = 0; k <= GGML_SCHED_PREFETCH_MAX; k++) {
        pf->done[b][k] = done[k];
    }
    return true;
}

// split is uploaded during split s
static bool ggml_backend_sched_prefetch_on(ggml_backend_sched_t sched, const struct ggml_backend_sched_split * split, int s) {
    return split->prefetch_at == s && sched->prefetch.copy[split->backend_id] != nullptr && ggml_backend_sched_prefetch_full(sched, split);
}

static void ggml_backend_sched_prefetch_init(ggml_backend_sched_t sched) {
    struct ggml_backend_sched_prefetch * pf = &sched->prefetch;

    // 14.10, 3090: a 485-token prompt ubatch took 2.39 s with the experts its ids name (about 70 percent of them, the
    // routing is uneven) against 3.20 s uploading them all ahead, a 3761-token one 3.52 s ahead against 4.83 s; so the
    // whole tensor goes ahead from 20 selections per expert (1024 tokens of 10 of 512 experts)
    const char * load = getenv("GGML_SCHED_PREFETCH_LOAD");
    pf->min_load = load ? std::max(0, atoi(load)) : 20;

    for (int b = 0; b < sched->n_backends; b++) {
        ggml_backend_dev_t dev = ggml_backend_get_device(sched->backends[b]);
        if (dev == nullptr || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ||
                sched->backends[b]->iface.set_tensor_async == nullptr) {
            continue;
        }
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        pf->supported[b] = props.caps.async && props.caps.events;
    }

    // on by default since 14.10 (3090 chats, depth 3 with the threshold of 20 per expert): the ~3.8K turns 686.1 t/s
    // against 547.9, the turns of 160-540 tokens and TG the same. GGML_SCHED_PREFETCH=0: off (upstream)
    const char * depth = getenv("GGML_SCHED_PREFETCH");
    ggml_backend_sched_set_prefetch(sched, depth ? atoi(depth) : 3);
}

void ggml_backend_sched_set_prefetch(ggml_backend_sched_t sched, int depth) {
    GGML_ASSERT(sched);
    // pipeline parallelism keeps every input copy alive anyway
    depth = sched->n_copies > 1 ? 0 : std::max(0, std::min(depth, GGML_SCHED_PREFETCH_MAX));
    if (depth != sched->prefetch.depth) {
        // ggml-alloc would give the order of nodes of the new depth the offsets reserved for the old one
        GGML_ASSERT(sched->n_reserve == 0 && "set the prefetch depth before the first reserve");
        sched->prefetch.depth = depth;
    }
}

// the uploads are done and their events can be reused (Vulkan takes back the events of a record only at a sync)
static void ggml_backend_sched_prefetch_sync(ggml_backend_sched_t sched) {
    struct ggml_backend_sched_prefetch * pf = &sched->prefetch;
    for (int b = 0; b < sched->n_backends; b++) {
        if (pf->copy[b] == nullptr) {
            continue;
        }
        ggml_backend_synchronize(pf->copy[b]);
        ggml_backend_event_synchronize(pf->before[b]);
        for (int k = 0; k <= GGML_SCHED_PREFETCH_MAX; k++) {
            ggml_backend_event_synchronize(pf->done[b][k]);
        }
    }
}

static void ggml_backend_sched_prefetch_free(ggml_backend_sched_t sched) {
    struct ggml_backend_sched_prefetch * pf = &sched->prefetch;
    ggml_backend_sched_prefetch_sync(sched);
    for (int b = 0; b < sched->n_backends; b++) {
        if (pf->copy[b] == nullptr) {
            continue;
        }
        ggml_backend_event_free(pf->before[b]);
        for (int k = 0; k <= GGML_SCHED_PREFETCH_MAX; k++) {
            ggml_backend_event_free(pf->done[b][k]);
        }
        ggml_backend_free(pf->copy[b]);
        pf->copy[b] = nullptr;
    }
}

static void ggml_backend_sched_prefetch_plan(ggml_backend_sched_t sched) {
    const int depth = sched->prefetch.depth;
    for (int t = 0; t < sched->n_splits; t++) {
        struct ggml_backend_sched_split * split = &sched->splits[t];
        split->prefetch_at = -1;
        if (depth == 0 || t == 0 || !sched->prefetch.supported[split->backend_id]) {
            continue;
        }
        for (int j = 0; j < split->n_inputs; j++) {
            if (ggml_backend_sched_prefetch_weight(split->inputs[j])) {
                split->prefetch_at = std::max(0, t - depth);
                break;
            }
        }
    }
}

static bool ggml_backend_sched_prefetch_input(const struct ggml_backend_sched_split * split, const struct ggml_tensor * input) {
    return split->prefetch_at >= 0 && ggml_backend_sched_prefetch_weight(input);
}

// the copies of the host weights of the splits uploaded during split i, allocated at its start after its own inputs
static void ggml_backend_sched_prefetch_add_inputs(ggml_backend_sched_t sched, int i, struct ggml_cgraph * graph_copy) {
    if (sched->prefetch.depth == 0) {
        return;
    }
    for (int t = i + 1; t <= i + GGML_SCHED_PREFETCH_MAX && t < sched->n_splits; t++) {
        const struct ggml_backend_sched_split * split = &sched->splits[t];
        if (split->prefetch_at != i) {
            continue;
        }
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            if (!ggml_backend_sched_prefetch_weight(input)) {
                continue;
            }
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }
    }
}

// before split s takes its inputs: an event after the work of the splits before it, on each backend that uploads the
// weights of a later split during s
static void ggml_backend_sched_prefetch_begin(ggml_backend_sched_t sched, int s) {
    struct ggml_backend_sched_prefetch * pf = &sched->prefetch;
    if (pf->depth == 0) {
        return;
    }
    bool recorded[GGML_SCHED_MAX_BACKENDS] = {};
    for (int t = s + 1; t <= s + GGML_SCHED_PREFETCH_MAX && t < sched->n_splits; t++) {
        const struct ggml_backend_sched_split * split = &sched->splits[t];
        const int b = split->backend_id;
        if (split->prefetch_at != s || recorded[b] || !ggml_backend_sched_prefetch_full(sched, split) ||
                !ggml_backend_sched_prefetch_ensure(sched, b)) {
            continue;
        }
        ggml_backend_event_record(pf->before[b], sched->backends[b]);
        recorded[b] = true;
    }
}

// after split s is submitted: the uploads of the later splits planned during s, on the second instance
static void ggml_backend_sched_prefetch_issue(ggml_backend_sched_t sched, int s) {
    struct ggml_backend_sched_prefetch * pf = &sched->prefetch;
    if (pf->depth == 0) {
        return;
    }
    for (int t = s + 1; t <= s + GGML_SCHED_PREFETCH_MAX && t < sched->n_splits; t++) {
        const struct ggml_backend_sched_split * split = &sched->splits[t];
        const int b = split->backend_id;
        if (!ggml_backend_sched_prefetch_on(sched, split, s)) {
            continue;
        }
        // a wait for each upload: Vulkan submits its command buffer at the record of an event and records the next
        // upload into a new one, and a wait recorded in the buffer before need not order it
        ggml_backend_t copy = pf->copy[b];
        ggml_backend_event_wait(copy, pf->before[b]);
        for (int j = 0; j < split->n_inputs; j++) {
            struct ggml_tensor * input = split->inputs[j];
            if (!ggml_backend_sched_prefetch_weight(input)) {
                continue;
            }
            ggml_backend_tensor_set_async(copy, tensor_copy(input, b, sched->cur_copy), input->data, 0, ggml_nbytes(input));
        }
        ggml_backend_event_record(pf->done[b][t % (GGML_SCHED_PREFETCH_MAX + 1)], copy);
        if (pf->n_uploads++ == 0) {
            GGML_LOG_INFO("%s: the host weights of split %d are uploaded during split %d on a second %s\n", __func__, t, s,
                    ggml_backend_name(copy));
        }
    }
}

// true when the host weights of split t were uploaded during an earlier split: its backend waits for the upload
static bool ggml_backend_sched_prefetch_wait(ggml_backend_sched_t sched, int t) {
    const struct ggml_backend_sched_split * split = &sched->splits[t];
    if (split->prefetch_at < 0 || !ggml_backend_sched_prefetch_on(sched, split, split->prefetch_at)) {
        return false;
    }
    const int b = split->backend_id;
    ggml_backend_event_wait(sched->backends[b], sched->prefetch.done[b][t % (GGML_SCHED_PREFETCH_MAX + 1)]);
    return true;
}


// GGML_TIMELINE=<file>: the spans of ggml_timeline_event while ggml_timeline_enable(true), one trace event per line, the
// array left open so that a process killed after the capture leaves a file Perfetto and chrome://tracing read. A track
// is a thread of the trace (its name in a metadata event at its first span)
static std::mutex             ggml_timeline_mutex;
static std::atomic<bool>      ggml_timeline_flag { false };
static FILE *                 ggml_timeline_file = nullptr;
static bool                   ggml_timeline_opened = false;
static std::vector<std::string> ggml_timeline_tracks;

bool ggml_timeline_on(void) {
    return ggml_timeline_flag.load(std::memory_order_relaxed);
}

void ggml_timeline_enable(bool enable) {
    std::lock_guard<std::mutex> lock(ggml_timeline_mutex);
    if (!ggml_timeline_opened) {
        ggml_timeline_opened = true;
        const char * path = getenv("GGML_TIMELINE");
        if (path != nullptr && *path != '\0') {
            ggml_timeline_file = fopen(path, "w");
            if (ggml_timeline_file == nullptr) {
                GGML_LOG_WARN("%s: cannot write %s\n", __func__, path);
            } else {
                fputs("[\n", ggml_timeline_file);
            }
        }
    }
    ggml_timeline_flag = enable && ggml_timeline_file != nullptr;
    if (ggml_timeline_file != nullptr) {
        fflush(ggml_timeline_file);
    }
}

void ggml_timeline_event(const char * track, const char * name, int64_t t_start_us, int64_t t_end_us) {
    ggml_timeline_event_args(track, name, t_start_us, t_end_us, nullptr);
}

void ggml_timeline_event_args(const char * track, const char * name, int64_t t_start_us, int64_t t_end_us, const char * args) {
    if (!ggml_timeline_on()) {
        return;
    }
    // the names are ours, but a quote or a backslash would break the line
    auto clean = [](const char * text) {
        std::string out(text);
        for (char & c : out) {
            if (c == '"' || c == '\\' || (unsigned char) c < 0x20) {
                c = '_';
            }
        }
        return out;
    };
    std::lock_guard<std::mutex> lock(ggml_timeline_mutex);
    if (ggml_timeline_file == nullptr) {
        return;
    }
    size_t tid = 0;
    while (tid < ggml_timeline_tracks.size() && ggml_timeline_tracks[tid] != track) {
        tid++;
    }
    if (tid == ggml_timeline_tracks.size()) {
        ggml_timeline_tracks.push_back(track);
        fprintf(ggml_timeline_file, "{\"ph\":\"M\",\"pid\":1,\"tid\":%zu,\"name\":\"thread_name\",\"args\":{\"name\":\"%s\"}},\n",
                tid + 1, clean(track).c_str());
    }
    // the args are built by our callers from names and numbers, the name inside them is cleaned there
    fprintf(ggml_timeline_file, "{\"ph\":\"X\",\"pid\":1,\"tid\":%zu,\"ts\":%lld,\"dur\":%lld,\"name\":\"%s\"%s%s%s},\n", tid + 1,
            (long long) t_start_us, (long long) std::max<int64_t>(0, t_end_us - t_start_us), clean(name).c_str(),
            args ? ",\"args\":{" : "", args ? args : "", args ? "}" : "");
}

// the shift from a device clock to ggml_time_us() along the capture: each span gives a lower bound, the host time it
// was issued at - its device start, tight for a span the device began at once, and the shift is the upper hull of those
// points, linear between them, so that no span starts before it was issued. the clocks run at slightly different rates
// (MoltenVK: 0.4 percent, 5 ms in 1.2 s); a hull of the largest bound per chunk of spans left the spans before its first
// point up to 8.5 ms early
void ggml_timeline_device_spans(const char * track, size_t n, const double * start_us, const double * end_us, const double * issued_us) {
    if (!ggml_timeline_on() || n == 0) {
        return;
    }
    std::vector<std::array<double, 2>> pts(n);
    for (size_t i = 0; i < n; i++) {
        pts[i] = { start_us[i], issued_us[i] - start_us[i] };
    }
    std::sort(pts.begin(), pts.end());
    std::vector<std::array<double, 2>> hull;
    for (const auto & p : pts) {
        // drop the last point while it is on or below the line from the one before it to p
        while (hull.size() >= 2) {
            const auto & a = hull[hull.size() - 2];
            const auto & b = hull.back();
            if ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0) {
                break;
            }
            hull.pop_back();
        }
        hull.push_back(p);
    }
    auto shift = [&](double x) {
        if (hull.size() == 1) {
            return hull[0][1];
        }
        size_t i = 1;
        while (i + 1 < hull.size() && hull[i][0] < x) {
            i++;
        }
        const auto & a = hull[i - 1];
        const auto & b = hull[i];
        return b[0] > a[0] ? a[1] + (b[1] - a[1]) * (x - a[0]) / (b[0] - a[0]) : std::max(a[1], b[1]);
    };
    for (size_t i = 0; i < n; i++) {
        const double d = shift(start_us[i]);
        ggml_timeline_event(track, "gpu", (int64_t) (start_us[i] + d), (int64_t) (end_us[i] + d));
    }
}

static void ggml_backend_sched_split_done_fork(ggml_backend_sched_t sched, ggml_backend_t backend, struct ggml_backend_sched_split * split) {
    if (sched->split_callback_fork == nullptr) {
        return;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }
    sched->split_callback_fork(backend, &split->graph, sched->split_callback_fork_user_data);
}

static void ggml_backend_sched_timeline_split(ggml_backend_t backend, const struct ggml_backend_sched_split * split, int64_t t_start_us) {
    ggml_backend_dev_t dev = ggml_backend_get_device(backend);
    if (dev == nullptr || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        return;
    }
    const char * name = split->graph.n_nodes > 0 ? split->graph.nodes[0]->name : "cpu";
    ggml_timeline_event("sched cpu", name[0] != '\0' ? name : "cpu", t_start_us, ggml_time_us());
}
