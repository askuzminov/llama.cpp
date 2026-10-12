// fork: what fills a compute buffer at its peak, included by ggml-alloc.c. GGML_ALLOC_PEAK=1: every graph reserve logs,
// per buffer of 64 MiB or more, the size of the peak, the tensor whose allocation set it and the largest tensors alive at
// that moment (name, op, MiB). the allocator only knows offsets and sizes; the tensors are tracked by their address, so a
// node that took over its parent's memory in place shows under the parent's name

#define GGML_ALLOC_PEAK_TOP 16

struct ggml_alloc_peak_entry {
    const struct ggml_tensor * tensor;
    int    chunk;
    size_t offset;
    size_t size;
};

struct ggml_alloc_peak_fork {
    struct ggml_alloc_peak_entry * live;
    int    n_live;
    int    cap_live;

    size_t peak;            // the highest end of an allocation, chunk 0 and up together
    char   at[128];         // the tensor that set it
    size_t live_bytes;      // bytes alive at the peak
    int    live_count;
    struct ggml_alloc_peak_entry top[GGML_ALLOC_PEAK_TOP];
    int    n_top;
    char   top_names[GGML_ALLOC_PEAK_TOP][96];
};

static bool ggml_alloc_peak_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char * env = getenv("GGML_ALLOC_PEAK");
        enabled = env != NULL && atoi(env) != 0;
    }
    return enabled != 0;
}

static struct ggml_alloc_peak_fork * ggml_alloc_peak_new(void) {
    if (!ggml_alloc_peak_enabled()) {
        return NULL;
    }
    struct ggml_alloc_peak_fork * p = (struct ggml_alloc_peak_fork *) calloc(1, sizeof(struct ggml_alloc_peak_fork));
    return p;
}

static void ggml_alloc_peak_free(struct ggml_alloc_peak_fork * p) {
    if (p) {
        free(p->live);
        free(p);
    }
}

static void ggml_alloc_peak_reset(struct ggml_alloc_peak_fork * p) {
    if (p) {
        p->n_live = 0;
        p->peak = 0;
        p->at[0] = '\0';
        p->live_bytes = 0;
        p->live_count = 0;
        p->n_top = 0;
    }
}

static void ggml_alloc_peak_name(char * dst, size_t n, const struct ggml_tensor * t) {
    if (t == NULL) {
        snprintf(dst, n, "?");
    } else if (t->name[0] != '\0') {
        snprintf(dst, n, "%s (%s)", t->name, ggml_op_desc(t));
    } else {
        snprintf(dst, n, "(%s)", ggml_op_desc(t));
    }
}

static void ggml_alloc_peak_on_alloc(struct ggml_alloc_peak_fork * p, const struct ggml_tensor * tensor, int chunk, size_t offset, size_t size,
                                     size_t chunk_max_before, size_t chunk_base) {
    if (p == NULL) {
        return;
    }
    if (p->n_live == p->cap_live) {
        p->cap_live = p->cap_live ? 2*p->cap_live : 256;
        p->live = (struct ggml_alloc_peak_entry *) realloc(p->live, p->cap_live * sizeof(struct ggml_alloc_peak_entry));
    }
    p->live[p->n_live++] = (struct ggml_alloc_peak_entry) { tensor, chunk, offset, size };

    const size_t end = chunk_base + offset + size;
    if (offset + size <= chunk_max_before || end <= p->peak) {
        return;
    }
    p->peak = end;
    ggml_alloc_peak_name(p->at, sizeof(p->at), tensor);
    p->live_bytes = 0;
    p->live_count = p->n_live;
    p->n_top = 0;
    for (int i = 0; i < p->n_live; ++i) {
        const struct ggml_alloc_peak_entry e = p->live[i];
        p->live_bytes += e.size;
        // keep the largest GGML_ALLOC_PEAK_TOP, sorted by size
        int k = p->n_top < GGML_ALLOC_PEAK_TOP ? p->n_top++ : GGML_ALLOC_PEAK_TOP - 1;
        if (k == GGML_ALLOC_PEAK_TOP - 1 && p->n_top == GGML_ALLOC_PEAK_TOP && p->top[k].size >= e.size) {
            continue;
        }
        while (k > 0 && p->top[k - 1].size < e.size) {
            p->top[k] = p->top[k - 1];
            --k;
        }
        p->top[k] = e;
    }
    for (int i = 0; i < p->n_top; ++i) {
        ggml_alloc_peak_name(p->top_names[i], sizeof(p->top_names[i]), p->top[i].tensor);
    }
}

static void ggml_alloc_peak_on_free(struct ggml_alloc_peak_fork * p, int chunk, size_t offset) {
    if (p == NULL) {
        return;
    }
    for (int i = p->n_live - 1; i >= 0; --i) {
        if (p->live[i].chunk == chunk && p->live[i].offset == offset) {
            p->live[i] = p->live[--p->n_live];
            return;
        }
    }
}

static void ggml_alloc_peak_report(struct ggml_alloc_peak_fork * p, const char * buft_name) {
    if (p == NULL || p->peak < ((size_t) 64 << 20)) {
        return;
    }
    char line[4096];
    int len = snprintf(line, sizeof(line), "alloc peak %s: %.1f MiB, set by %s; %d tensors alive, %.1f MiB:",
            buft_name, p->peak/1048576.0, p->at, p->live_count, p->live_bytes/1048576.0);
    for (int i = 0; i < p->n_top && len < (int) sizeof(line) - 160; ++i) {
        len += snprintf(line + len, sizeof(line) - len, " %s %.1f;", p->top_names[i], p->top[i].size/1048576.0);
    }
    GGML_LOG_INFO("%s\n", line);
}
