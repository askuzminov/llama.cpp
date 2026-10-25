// the fork's declarations for ggml-backend.cpp, included before struct ggml_backend_sched: the types the scheduler
// holds and the functions of ggml-backend-fork.inc that the upstream code calls

// GGML_SCHED_LOG_REALLOC: on stderr, why the scheduler reserves its buffers again while computing (each time a sync of
// all backends and a new plan) and the splits of every reserved graph
static bool ggml_backend_sched_log_realloc() {
    static const bool enabled = getenv("GGML_SCHED_LOG_REALLOC") != nullptr;
    return enabled;
}

// kinds of graphs, by split count, that GGML_SCHED_PROF reports apart
#define GGML_SCHED_PROF_BUCKETS 4

// host time of the splits of one kind of graph [GGML_SCHED_PROF]
struct ggml_backend_sched_prof {
    int     n_splits; // splits of the graph, 0 when the bucket is free, -1 for the graphs that have no bucket
    int64_t n_graphs;
    int64_t n_wall;   // graphs with a known wall time
    int64_t t_wall;   // from the start of a graph to the start of the next graph
    int64_t n_split[GGML_SCHED_MAX_BACKENDS];
    int64_t t_in   [GGML_SCHED_MAX_BACKENDS]; // input copies, with the waits
    int64_t t_wait [GGML_SCHED_MAX_BACKENDS]; // waits for a backend in the input copies
    int64_t t_run  [GGML_SCHED_MAX_BACKENDS]; // graph compute call, the full compute for the CPU
    int64_t n_overlap[GGML_SCHED_MAX_BACKENDS]; // CPU splits submitted without waiting for device-only work
    double  t_gpu  [GGML_SCHED_MAX_BACKENDS]; // device time of the compute calls [GGML_SCHED_PROF_GPU], of n_wall graphs
};

// the device time of the compute calls, from the backend's proc addresses (the CUDA backend has them)
typedef void (*ggml_backend_timing_enable_fork_t)(ggml_backend_t backend, bool enable);
typedef void (*ggml_backend_timing_take_fork_t)(ggml_backend_t backend, double * busy_us, int64_t * n_calls);

// GGML_SCHED_PREFETCH=<depth> (3): the weights in host memory of a split are uploaded while the split <depth> before it
// computes, on a second instance of its backend (see ggml-backend-fork.inc)
#define GGML_SCHED_PREFETCH_MAX 4

struct ggml_backend_sched_prefetch {
    int depth;    // 0: off
    int min_load; // a MUL_MAT_ID at the start of a split is uploaded whole when its ids select at least this many per expert

    bool                 supported[GGML_SCHED_MAX_BACKENDS]; // a device with async copies and events, fixed at the start
    bool                 failed   [GGML_SCHED_MAX_BACKENDS]; // no second instance or events: copies at the start of the split
    ggml_backend_t       copy     [GGML_SCHED_MAX_BACKENDS]; // the second instance, created at the first upload
    ggml_backend_event_t before   [GGML_SCHED_MAX_BACKENDS]; // on the backend, before the split that uploads
    ggml_backend_event_t done     [GGML_SCHED_MAX_BACKENDS][GGML_SCHED_PREFETCH_MAX + 1]; // on the copy, by split % (MAX + 1)

    int64_t n_uploads; // the first one is logged

    ggml_backend_sched_copy_callback upload_cb; // ggml_backend_sched_set_upload_callback
    void *                           upload_cb_data;
};

static void ggml_backend_sched_prefetch_init(ggml_backend_sched_t sched);
static void ggml_backend_sched_prefetch_free(ggml_backend_sched_t sched);
static void ggml_backend_sched_prefetch_sync(ggml_backend_sched_t sched);
static void ggml_backend_sched_prefetch_plan(ggml_backend_sched_t sched);
static bool ggml_backend_sched_prefetch_input(const struct ggml_backend_sched_split * split, const struct ggml_tensor * input);
static void ggml_backend_sched_prefetch_add_inputs(ggml_backend_sched_t sched, int i_split, struct ggml_cgraph * graph_copy);
static void ggml_backend_sched_prefetch_begin(ggml_backend_sched_t sched, int i_split);
static bool ggml_backend_sched_prefetch_wait(ggml_backend_sched_t sched, int i_split);
static void ggml_backend_sched_prefetch_issue(ggml_backend_sched_t sched, int i_split);

// GGML_TIMELINE: a CPU split is computed in its call, its span on track "sched cpu" named by its first node (the
// experts of a layer: the layer number is in the name)
static void ggml_backend_sched_timeline_split(ggml_backend_t backend, const struct ggml_backend_sched_split * split, int64_t t_start_us);

static void ggml_backend_sched_log_splits(ggml_backend_sched_t sched, const char * what);
static void ggml_backend_sched_prof_print(ggml_backend_sched_t sched, int bucket);
static void ggml_backend_sched_prof_gpu_init(ggml_backend_sched_t sched);
static void ggml_backend_sched_prof_gpu_take(ggml_backend_sched_t sched, int bucket);
static bool ggml_backend_sched_split_uses_host_buffer(const struct ggml_backend_sched_split * split);
static int  ggml_backend_sched_snapshot_backend_id(const struct ggml_backend_sched_snapshot * snapshot, const struct ggml_tensor * tensor);
