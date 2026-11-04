// the fork's helpers declared in llama-impl.h

#include "llama-impl.h"
#include "llama-fork.h"

#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <thread>
#include <vector>

void llama_parallel_for(int64_t n, int64_t work, const std::function<void(int64_t, int64_t)> & fn) {
    static const int64_t n_threads = [] {
        // set: at least one thread
        const int64_t n = llama_fork_env_int("LLAMA_INPUT_THREADS", -1);
        if (n >= 0) {
            return std::max<int64_t>(1, n);
        }
        return (int64_t) std::clamp(std::thread::hardware_concurrency() / 2, 1u, 8u);
    }();

    // a part below this many elements does not pay for its thread
    constexpr int64_t min_work = 1 << 20;

    const int64_t n_parts = std::min({ n_threads, n, std::max<int64_t>(1, work / min_work) });
    if (n_parts <= 1) {
        fn(0, n);
        return;
    }

    const int64_t chunk = (n + n_parts - 1) / n_parts;

    // what a part throws (no host memory for its bookkeeping) goes to the caller once every thread is joined: a thread
    // that ends with an exception, or a vector of joinable threads destroyed on the way out, ends the process. a part
    // that gets no thread runs here
    std::vector<std::exception_ptr> errors((size_t) n_parts);
    std::vector<std::thread> threads;
    threads.reserve(n_parts - 1);
    int64_t i_part = 1;
    for (int64_t b = chunk; b < n; b += chunk, ++i_part) {
        const int64_t e = std::min(n, b + chunk);
        std::exception_ptr & err = errors[(size_t) i_part];
        try {
            threads.emplace_back([&fn, &err, b, e] {
                try {
                    fn(b, e);
                } catch (...) {
                    err = std::current_exception();
                }
            });
        } catch (...) {
            try {
                fn(b, e);
            } catch (...) {
                err = std::current_exception();
            }
        }
    }
    try {
        fn(0, std::min(n, chunk));
    } catch (...) {
        errors[0] = std::current_exception();
    }
    for (auto & t : threads) {
        t.join();
    }
    for (const auto & err : errors) {
        if (err) {
            std::rethrow_exception(err);
        }
    }
}
