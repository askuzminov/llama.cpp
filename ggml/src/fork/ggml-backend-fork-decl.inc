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

static void ggml_backend_sched_log_splits(ggml_backend_sched_t sched, const char * what);
static void ggml_backend_sched_prof_print(ggml_backend_sched_t sched, int bucket);
static void ggml_backend_sched_prof_gpu_init(ggml_backend_sched_t sched);
static void ggml_backend_sched_prof_gpu_take(ggml_backend_sched_t sched, int bucket);
static bool ggml_backend_sched_split_uses_host_buffer(const struct ggml_backend_sched_split * split);
static int  ggml_backend_sched_snapshot_backend_id(const struct ggml_backend_sched_snapshot * snapshot, const struct ggml_tensor * tensor);
