// the fork's GPU time of the graph compute calls, included in ggml-cuda.cu. GGML_SCHED_PROF_GPU=1 (with
// GGML_SCHED_PROF) turns it on through the proc address ggml_backend_timing_enable_fork and reads it through
// ggml_backend_timing_take_fork: a pair of timing events around each compute call, read once the end has passed.
// the span starts when the stream reaches the first event, so it also holds the host launch of the call

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

#include <cuda_profiler_api.h>
#include <deque>

struct ggml_cuda_timing_fork {
    struct span {
        cudaEvent_t start;
        cudaEvent_t end;
    };
    std::deque<span>         pending; // recorded, not read yet
    std::vector<cudaEvent_t> free_events;
    double  busy_us = 0.0;            // of the spans read since the last take
    int64_t n_calls = 0;
};

static std::mutex ggml_cuda_timing_fork_mutex;
static std::atomic<int> ggml_cuda_timing_fork_n { 0 };
static std::unordered_map<const ggml_backend_cuda_context *, std::unique_ptr<ggml_cuda_timing_fork>> ggml_cuda_timing_fork_ctx;

static ggml_cuda_timing_fork * ggml_cuda_timing_fork_get(const ggml_backend_cuda_context * ctx) {
    if (ggml_cuda_timing_fork_n.load(std::memory_order_relaxed) == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ggml_cuda_timing_fork_mutex);
    auto it = ggml_cuda_timing_fork_ctx.find(ctx);
    return it == ggml_cuda_timing_fork_ctx.end() ? nullptr : it->second.get();
}

static void ggml_cuda_timing_fork_destroy(ggml_cuda_timing_fork * t) {
    for (const auto & s : t->pending) {
        CUDA_CHECK(cudaEventDestroy(s.start));
        CUDA_CHECK(cudaEventDestroy(s.end));
    }
    for (cudaEvent_t e : t->free_events) {
        CUDA_CHECK(cudaEventDestroy(e));
    }
}

// the spans whose end has passed, all of them with wait
static void ggml_cuda_timing_fork_read(ggml_cuda_timing_fork * t, bool wait) {
    while (!t->pending.empty()) {
        const ggml_cuda_timing_fork::span s = t->pending.front();
        if (wait) {
            CUDA_CHECK(cudaEventSynchronize(s.end));
        } else {
            const cudaError_t err = cudaEventQuery(s.end);
            if (err == cudaErrorNotReady) {
                // not an error: keep it out of the CUDA_CHECK(cudaGetLastError()) after the next kernel launch
                if (cudaPeekAtLastError() == cudaErrorNotReady) {
                    (void) cudaGetLastError();
                }
                break;
            }
            CUDA_CHECK(err);
        }
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s.start, s.end));
        t->busy_us += 1e3*(double) ms;
        t->n_calls++;
        t->free_events.push_back(s.start);
        t->free_events.push_back(s.end);
        t->pending.pop_front();
    }
}

static cudaEvent_t ggml_cuda_timing_fork_event(ggml_cuda_timing_fork * t) {
    if (!t->free_events.empty()) {
        cudaEvent_t e = t->free_events.back();
        t->free_events.pop_back();
        return e;
    }
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDefault));
    return e;
}

// before the compute call, outside a stream capture
static void ggml_cuda_timing_fork_begin(ggml_backend_cuda_context * ctx) {
    ggml_cuda_timing_fork * t = ggml_cuda_timing_fork_get(ctx);
    if (t == nullptr) {
        return;
    }
    // the scheduler takes the time at each graph start, a long run of calls without a take waits for the oldest
    ggml_cuda_timing_fork_read(t, t->pending.size() >= 4096);
    const ggml_cuda_timing_fork::span s = { ggml_cuda_timing_fork_event(t), ggml_cuda_timing_fork_event(t) };
    CUDA_CHECK(cudaEventRecord(s.start, ctx->stream()));
    t->pending.push_back(s);
}

// after the compute call: the graph is launched and the capture is over
static void ggml_cuda_timing_fork_end(ggml_backend_cuda_context * ctx) {
    ggml_cuda_timing_fork * t = ggml_cuda_timing_fork_get(ctx);
    if (t == nullptr || t->pending.empty()) {
        return;
    }
    CUDA_CHECK(cudaEventRecord(t->pending.back().end, ctx->stream()));
}

static void ggml_cuda_timing_fork_free(const ggml_backend_cuda_context * ctx) {
    if (ggml_cuda_timing_fork_n.load(std::memory_order_relaxed) == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ggml_cuda_timing_fork_mutex);
    auto it = ggml_cuda_timing_fork_ctx.find(ctx);
    if (it != ggml_cuda_timing_fork_ctx.end()) {
        ggml_cuda_timing_fork_destroy(it->second.get());
        ggml_cuda_timing_fork_ctx.erase(it);
        ggml_cuda_timing_fork_n--;
    }
}

static void ggml_backend_cuda_timing_enable_fork(ggml_backend_t backend, bool enable) {
    const ggml_backend_cuda_context * ctx = (const ggml_backend_cuda_context *) backend->context;
    if (!enable) {
        ggml_cuda_timing_fork_free(ctx);
        return;
    }
    std::lock_guard<std::mutex> lock(ggml_cuda_timing_fork_mutex);
    if (ggml_cuda_timing_fork_ctx.find(ctx) == ggml_cuda_timing_fork_ctx.end()) {
        ggml_cuda_timing_fork_ctx.emplace(ctx, std::make_unique<ggml_cuda_timing_fork>());
        ggml_cuda_timing_fork_n++;
    }
}

// the GPU time and the count of the calls that ended since the last take, without waiting
static void ggml_backend_cuda_timing_take_fork(ggml_backend_t backend, double * busy_us, int64_t * n_calls) {
    ggml_cuda_timing_fork * t = ggml_cuda_timing_fork_get((const ggml_backend_cuda_context *) backend->context);
    *busy_us = 0.0;
    *n_calls = 0;
    if (t == nullptr) {
        return;
    }
    ggml_cuda_timing_fork_read(t, false);
    *busy_us = t->busy_us;
    *n_calls = t->n_calls;
    t->busy_us = 0.0;
    t->n_calls = 0;
}

// ---- GGML_CUDA_OP_TIMING=<report period in compute calls>: the GPU time of every op (a fused group counts as one), a
// pair of timing events around each, read once passed. the CUDA graphs are off meanwhile (a capture cannot hold the
// events), so the time between the ops is not the one of a normal run; the time of each kernel is. every period the
// log gets the ops by total time, with their count and mean, labelled by op, type and shape. GGML_CUDA_OP_TIMING=idle:
// the report comes when a compute call starts after the GPU was idle for 300 ms or more, so it holds one request of a
// server (scripts/win-qsa 05: the report of a request is printed at the start of the next one)

#include <map>
#include <string>
#include <algorithm>
#include <chrono>
#include <cstring>

struct ggml_cuda_op_timing_fork {
    struct span {
        cudaEvent_t start;
        cudaEvent_t end;
        std::string label;
    };
    std::deque<span>         pending;
    std::vector<cudaEvent_t> free_events;
    std::map<std::string, std::pair<int64_t, double>> agg; // label -> (count, us)
    int64_t n_calls = 0;
    int64_t n_ops   = 0;
    double  op_us   = 0.0;
    std::chrono::steady_clock::time_point t_last {}; // the start of the last compute call (idle mode)
};

// the report period in compute calls, -1 for idle, 0 off
static int ggml_cuda_op_timing_fork_period() {
    static const int period = [] {
        const char * env = getenv("GGML_CUDA_OP_TIMING");
        if (env == nullptr) {
            return 0;
        }
        return strcmp(env, "idle") == 0 ? -1 : atoi(env);
    }();
    return period;
}

static std::unordered_map<const ggml_backend_cuda_context *, std::unique_ptr<ggml_cuda_op_timing_fork>> ggml_cuda_op_timing_fork_ctx;

static ggml_cuda_op_timing_fork * ggml_cuda_op_timing_fork_get(const ggml_backend_cuda_context * ctx) {
    if (ggml_cuda_op_timing_fork_period() == 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(ggml_cuda_timing_fork_mutex);
    auto & t = ggml_cuda_op_timing_fork_ctx[ctx];
    if (!t) {
        t = std::make_unique<ggml_cuda_op_timing_fork>();
    }
    return t.get();
}

static cudaEvent_t ggml_cuda_op_timing_fork_event(ggml_cuda_op_timing_fork * t) {
    if (!t->free_events.empty()) {
        cudaEvent_t e = t->free_events.back();
        t->free_events.pop_back();
        return e;
    }
    cudaEvent_t e;
    CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDefault));
    return e;
}

static bool ggml_cuda_op_timing_fork_on() {
    return ggml_cuda_op_timing_fork_period() != 0;
}

// before the op (or the fused group) at node i
static void ggml_cuda_op_timing_fork_begin(ggml_backend_cuda_context * ctx) {
    ggml_cuda_op_timing_fork * t = ggml_cuda_op_timing_fork_get(ctx);
    if (t == nullptr) {
        return;
    }
    ggml_cuda_op_timing_fork::span s { ggml_cuda_op_timing_fork_event(t), ggml_cuda_op_timing_fork_event(t), {} };
    CUDA_CHECK(cudaEventRecord(s.start, ctx->stream()));
    t->pending.push_back(std::move(s));
}

static bool ggml_cuda_op_timing_fork_empty_node(const ggml_tensor * x) {
    return ggml_is_empty(x) || x->op == GGML_OP_NONE || x->op == GGML_OP_RESHAPE || x->op == GGML_OP_VIEW ||
           x->op == GGML_OP_PERMUTE || x->op == GGML_OP_TRANSPOSE;
}

// after it: skip > 0 when nodes i .. i + skip ran as one fused group
static void ggml_cuda_op_timing_fork_end(ggml_backend_cuda_context * ctx, const ggml_cgraph * g, int i, int skip) {
    ggml_cuda_op_timing_fork * t = ggml_cuda_op_timing_fork_get(ctx);
    if (t == nullptr || t->pending.empty()) {
        return;
    }
    ggml_cuda_op_timing_fork::span & s = t->pending.back();
    CUDA_CHECK(cudaEventRecord(s.end, ctx->stream()));

    const ggml_tensor * n = g->nodes[i];
    std::string label;
    if (skip > 0) {
        label = "FUSED";
        for (int k = 0; k <= skip; ++k) {
            const ggml_tensor * x = g->nodes[i + k];
            if (!ggml_cuda_op_timing_fork_empty_node(x)) {
                label += (label.size() == 5 ? " " : "+");
                label += ggml_op_desc(x);
            }
        }
    } else {
        label = ggml_op_desc(n);
    }
    char buf[192];
    const ggml_tensor * a = n->src[0];
    const ggml_tensor * b = n->src[1];
    if ((n->op == GGML_OP_MUL_MAT || n->op == GGML_OP_MUL_MAT_ID) && a != nullptr && b != nullptr) {
        snprintf(buf, sizeof(buf), " %s m=%lld k=%lld n=%lld%s", ggml_type_name(a->type), (long long) a->ne[1], (long long) a->ne[0],
                 (long long) (b->ne[1] * b->ne[2]), n->op == GGML_OP_MUL_MAT_ID ? " (id)" : "");
    } else {
        const ggml_tensor * d = g->nodes[i + skip];
        snprintf(buf, sizeof(buf), " [%lld,%lld,%lld,%lld]", (long long) d->ne[0], (long long) d->ne[1], (long long) d->ne[2], (long long) d->ne[3]);
    }
    s.label = label + buf;
}

// at the start of a compute call: the spans that ended, and the report every period
// reads the spans that have passed (all of them with wait)
static void ggml_cuda_op_timing_fork_drain(ggml_cuda_op_timing_fork * t, bool wait) {
    while (!t->pending.empty()) {
        ggml_cuda_op_timing_fork::span & s = t->pending.front();
        if (wait) {
            CUDA_CHECK(cudaEventSynchronize(s.end));
        }
        const cudaError_t err = cudaEventQuery(s.end);
        if (err == cudaErrorNotReady) {
            if (cudaPeekAtLastError() == cudaErrorNotReady) {
                (void) cudaGetLastError();
            }
            break;
        }
        CUDA_CHECK(err);
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, s.start, s.end));
        auto & a = t->agg[s.label];
        a.first++;
        a.second += 1e3 * (double) ms;
        t->n_ops++;
        t->op_us += 1e3 * (double) ms;
        t->free_events.push_back(s.start);
        t->free_events.push_back(s.end);
        t->pending.pop_front();
    }
}

// the ops by total time; to stderr, as llama-bench (05) drops the log
static void ggml_cuda_op_timing_fork_report(const char * name, ggml_cuda_op_timing_fork * t) {
    std::vector<std::pair<std::string, std::pair<int64_t, double>>> rows(t->agg.begin(), t->agg.end());
    std::sort(rows.begin(), rows.end(), [](const auto & x, const auto & y) { return x.second.second > y.second.second; });
    const double n_calls = (double) std::max<int64_t>(t->n_calls, 1);
    fprintf(stderr, "cuda_op_timing %s: %lld compute calls, %lld ops, %.2f ms of kernels: %.1f ops and %.3f ms per call\n",
            name, (long long) t->n_calls, (long long) t->n_ops, 1e-3 * t->op_us, (double) t->n_ops / n_calls, 1e-3 * t->op_us / n_calls);
    for (size_t r = 0; r < rows.size() && r < 40; ++r) {
        fprintf(stderr, "cuda_op_timing %s: %9.3f ms %8lld x %8.2f us  %s\n", name, 1e-3 * rows[r].second.second,
                (long long) rows[r].second.first, rows[r].second.second / (double) rows[r].second.first, rows[r].first.c_str());
    }
    fflush(stderr);
    t->agg.clear();
    t->n_calls = 0;
    t->n_ops   = 0;
    t->op_us   = 0.0;
}

static void ggml_cuda_op_timing_fork_call(ggml_backend_cuda_context * ctx) {
    ggml_cuda_op_timing_fork * t = ggml_cuda_op_timing_fork_get(ctx);
    if (t == nullptr) {
        return;
    }
    if (ggml_cuda_op_timing_fork_period() < 0) {
        // idle mode: the calls before a gap are one request, report them before this call adds to them
        const auto now = std::chrono::steady_clock::now();
        const bool gap = t->n_ops > 0 && std::chrono::duration<double>(now - t->t_last).count() >= 0.3;
        t->t_last = now;
        if (gap) {
            ggml_cuda_op_timing_fork_drain(t, true);
            ggml_cuda_op_timing_fork_report(ctx->name.c_str(), t);
        } else {
            ggml_cuda_op_timing_fork_drain(t, false);
        }
        ++t->n_calls;
        return;
    }
    ggml_cuda_op_timing_fork_drain(t, false);
    if (++t->n_calls < ggml_cuda_op_timing_fork_period()) {
        return;
    }
    ggml_cuda_op_timing_fork_report(ctx->name.c_str(), t);
}

static void ggml_cuda_op_timing_fork_free(const ggml_backend_cuda_context * ctx) {
    if (ggml_cuda_op_timing_fork_period() == 0) {
        return;
    }
    std::lock_guard<std::mutex> lock(ggml_cuda_timing_fork_mutex);
    auto it = ggml_cuda_op_timing_fork_ctx.find(ctx);
    if (it == ggml_cuda_op_timing_fork_ctx.end()) {
        return;
    }
    // what the last period did not report yet (all of a llama-bench run with a period above its compute calls)
    ggml_cuda_op_timing_fork_drain(it->second.get(), true);
    if (it->second->n_ops > 0) {
        ggml_cuda_op_timing_fork_report(ctx->name.c_str(), it->second.get());
    }
    for (const auto & s : it->second->pending) {
        CUDA_CHECK(cudaEventDestroy(s.start));
        CUDA_CHECK(cudaEventDestroy(s.end));
    }
    for (cudaEvent_t e : it->second->free_events) {
        CUDA_CHECK(cudaEventDestroy(e));
    }
    ggml_cuda_op_timing_fork_ctx.erase(it);
}

// the capture range of nsys --capture-range=cudaProfilerApi, set by llama-server (LLAMA_TRACE_REQUESTS) through the proc
// address ggml_backend_profiler_fork. The stop waits for the devices: the work of the traced requests is in the trace
// only once it ends
static void ggml_backend_cuda_profiler_fork(bool start) {
    if (start) {
        CUDA_CHECK(cudaProfilerStart());
        return;
    }
    for (int d = 0; d < ggml_backend_cuda_get_device_count(); d++) {
        ggml_cuda_set_device(d);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    CUDA_CHECK(cudaProfilerStop());
}

#define GGML_CUDA_TIMING_FORK 1

#else

static void ggml_cuda_timing_fork_begin(ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }
static void ggml_cuda_timing_fork_end(ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }
static void ggml_cuda_timing_fork_free(const ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }
static bool ggml_cuda_op_timing_fork_on() { return false; }
static void ggml_cuda_op_timing_fork_begin(ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }
static void ggml_cuda_op_timing_fork_end(ggml_backend_cuda_context * ctx, const ggml_cgraph * g, int i, int skip) {
    GGML_UNUSED(ctx); GGML_UNUSED(g); GGML_UNUSED(i); GGML_UNUSED(skip);
}
static void ggml_cuda_op_timing_fork_call(ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }
static void ggml_cuda_op_timing_fork_free(const ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
