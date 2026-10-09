// the fork's GPU time of the graph compute calls, included in ggml-cuda.cu. GGML_SCHED_PROF_GPU=1 (with
// GGML_SCHED_PROF) turns it on through the proc address ggml_backend_timing_enable_fork and reads it through
// ggml_backend_timing_take_fork: a pair of timing events around each compute call, read once the end has passed.
// the span starts when the stream reaches the first event, so it also holds the host launch of the call

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)

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

#define GGML_CUDA_TIMING_FORK 1

#else

static void ggml_cuda_timing_fork_begin(ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }
static void ggml_cuda_timing_fork_end(ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }
static void ggml_cuda_timing_fork_free(const ggml_backend_cuda_context * ctx) { GGML_UNUSED(ctx); }

#endif // !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
