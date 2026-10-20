#pragma once

// fork: LLAMA_TRACE_REQUESTS=<first>[:<last>]: the backends' profiler runs from the start of chat completion request
// <first> to the end of request <last>, counted from 1 (scripts/win-qsa 07-trace): on CUDA cudaProfilerStart/Stop, so
// that nsys --capture-range=cudaProfilerApi traces those requests only; with GGML_TIMELINE=<file> the timeline of
// ggml_timeline_event (the requests, the ubatches of llama, the syncs, the gpu spans of Vulkan) for the same requests.
// The end is when the handler returns, so the traced requests must not stream. Unset: the handler as it is

#include "server-http.h"

server_http_context::handler_t server_trace_fork(server_http_context::handler_t func);
