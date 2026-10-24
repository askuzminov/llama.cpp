// the fork's helpers declared in llama-impl.h

#include "llama-impl.h"
#include "llama-fork.h"

#include <algorithm>
#include <cstdlib>
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

    std::vector<std::thread> threads;
    threads.reserve(n_parts - 1);
    for (int64_t b = chunk; b < n; b += chunk) {
        threads.emplace_back(fn, b, std::min(n, b + chunk));
    }
    fn(0, std::min(n, chunk));
    for (auto & t : threads) {
        t.join();
    }
}
