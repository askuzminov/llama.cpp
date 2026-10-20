#include "llama-load-watch-fork.h"

#include "llama-impl.h"

#include <cstdio>
#include <cstdlib>

llama_load_watch_fork::llama_load_watch_fork() {
    static const int limit = [] {
        const char * env = getenv("LLAMA_LOAD_WATCH");
        return env ? atoi(env) : 15;
    }();
    static const int exit_after = [] {
        const char * env = getenv("LLAMA_LOAD_WATCH_EXIT");
        return env ? atoi(env) : 180;
    }();
    limit_s = limit;
    exit_s  = exit_after;
    t_step  = clock::now();
    if (limit_s > 0 || exit_s > 0) {
        thread = std::thread([this] { run(); });
    }
}

llama_load_watch_fork::~llama_load_watch_fork() {
    if (!thread.joinable()) {
        return;
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cv.notify_all();
    thread.join();
}

std::string llama_load_watch_fork::describe() const {
    std::string s = name;
    if (!tensor.empty()) {
        s += " of " + tensor;
    }
    if (chunk >= 0) {
        s += ", chunk " + std::to_string(chunk);
    }
    return s;
}

void llama_load_watch_fork::step(const char * step_name, const char * step_tensor, int64_t step_chunk) {
    if (!thread.joinable()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex);
    const auto now = clock::now();
    if (reported) {
        LLAMA_LOG_WARN("load_all_data: %s took %.1f s\n", describe().c_str(),
                       std::chrono::duration<double>(now - t_step).count());
        reported = false;
    }
    name   = step_name;
    tensor = step_tensor ? step_tensor : "";
    chunk  = step_chunk;
    t_step = now;
}

void llama_load_watch_fork::run() {
    std::unique_lock<std::mutex> lock(mutex);
    while (!stop) {
        cv.wait_for(lock, std::chrono::seconds(1), [this] { return stop; });
        if (stop) {
            continue;
        }
        const double s = std::chrono::duration<double>(clock::now() - t_step).count();
        if (exit_s > 0 && s >= exit_s) {
            // stderr directly as well: the log may go through a queue that the exit does not drain
            LLAMA_LOG_ERROR("load_all_data: no progress for %.0f s in %s, exiting with code 75\n", s, describe().c_str());
            fprintf(stderr, "load_all_data: no progress for %.0f s in %s, exiting with code 75\n", s, describe().c_str());
            fflush(stderr);
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            std::_Exit(75);
        }
        if (limit_s > 0 && !reported && s >= limit_s) {
            LLAMA_LOG_WARN("load_all_data: no progress for %.0f s in %s\n", s, describe().c_str());
            reported = true;
        }
    }
}
