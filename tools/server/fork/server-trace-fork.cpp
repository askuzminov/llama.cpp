#include "server-trace-fork.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "log.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

struct trace_range {
    int first = 0;
    int last  = 0;
};

trace_range trace_range_from_env() {
    trace_range r;
    const char * env = getenv("LLAMA_TRACE_REQUESTS");
    if (env == nullptr || *env == '\0') {
        return r;
    }
    const int n = sscanf(env, "%d:%d", &r.first, &r.last);
    if (n < 1) {
        LOG_WRN("%s: LLAMA_TRACE_REQUESTS=\"%s\" is not first[:last], no trace\n", __func__, env);
        return trace_range{};
    }
    if (n < 2) {
        r.last = r.first;
    }
    return r;
}

// every backend that has one (CUDA: the capture range of nsys; Vulkan: its gpu spans on the timeline), and the timeline
// of GGML_TIMELINE: on before the backends start, off after they wrote their spans
void trace_profiler(bool start) {
    typedef void (*profiler_t)(bool);
    if (start) {
        ggml_timeline_enable(true);
    }
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto fn = (profiler_t) ggml_backend_reg_get_proc_address(ggml_backend_reg_get(i), "ggml_backend_profiler_fork");
        if (fn != nullptr) {
            fn(start);
        }
    }
    if (!start) {
        ggml_timeline_enable(false);
    }
}

// one count for every route the wrapper is on
std::atomic<int> n_requests { 0 };

} // namespace

server_http_context::handler_t server_trace_fork(server_http_context::handler_t func) {
    static const trace_range range = trace_range_from_env();
    if (range.first <= 0) {
        return func;
    }
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        const int n = ++n_requests;
        if (n == range.first) {
            LOG_INF("server_trace_fork: profiler on at request %d (LLAMA_TRACE_REQUESTS)\n", n);
            trace_profiler(true);
        }
        const int64_t t_start_us = ggml_time_us();
        server_http_res_ptr res;
        try {
            res = func(req);
        } catch (...) {
            if (n == range.last) {
                trace_profiler(false);
            }
            throw;
        }
        if (n >= range.first && n <= range.last) {
            const std::string name = "request " + std::to_string(n);
            ggml_timeline_event("server", name.c_str(), t_start_us, ggml_time_us());
        }
        if (n == range.last) {
            trace_profiler(false);
            LOG_INF("server_trace_fork: profiler off after request %d\n", n);
        }
        return res;
    };
}
