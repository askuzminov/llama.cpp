// fork: router child side, see server-role-fork.h

#include "server-role-fork.h"

#include "server-common.h"

#include "common.h"
#include "log.h"
#include "fork/env-fork.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#else
#   include <cerrno>
#   include <unistd.h>
#endif

int server_child_stop_timeout() {
    return (int) std::clamp<int64_t>(common_fork_env_int("LLAMA_SERVER_STOP_TIMEOUT", 10), 0, INT_MAX);
}

bool server_child_read_stdin(std::string & buf) {
    char chunk[256];
#if defined(_WIN32)
    DWORD n = 0;
    if (!ReadFile(GetStdHandle(STD_INPUT_HANDLE), chunk, (DWORD) sizeof(chunk), &n, NULL) || n == 0) {
        return false;
    }
#else
    ssize_t n;
    do {
        n = read(STDIN_FILENO, chunk, sizeof(chunk));
    } while (n < 0 && errno == EINTR);
    if (n <= 0) {
        return false;
    }
#endif
    buf.append(chunk, (size_t) n);
    return true;
}

[[noreturn]] static void terminate_self() {
#if defined(_WIN32)
    // also skips the DLL detach routines, which can hang while another thread is inside a backend call
    TerminateProcess(GetCurrentProcess(), 1);
#endif
    std::_Exit(1);
}

void server_child_watch_router_after_exit(int stop_timeout) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(stop_timeout);
    std::thread([deadline]() {
        std::string buf;
        while (server_child_read_stdin(buf)) {
            buf.clear();
        }
        std::this_thread::sleep_until(deadline);
        // no log: the main thread can be in the static destructors already
        terminate_self();
    }).detach();
}

void server_role::exit_now() {
    common_log_pause(common_log_main()); // write out the log
    terminate_self();
}

// devices the model is loaded on by default: the --device list, else the discrete gpus, else the integrated ones
static std::vector<ggml_backend_dev_t> mem_devices(const common_params & params) {
    std::vector<ggml_backend_dev_t> res;
    if (!params.devices.empty()) {
        for (auto * dev : params.devices) {
            if (dev != nullptr) {
                res.push_back(dev);
            }
        }
        return res;
    }
    for (auto type : { GGML_BACKEND_DEVICE_TYPE_GPU, GGML_BACKEND_DEVICE_TYPE_IGPU }) {
        for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == type) {
                res.push_back(dev);
            }
        }
        if (!res.empty()) {
            break;
        }
    }
    return res;
}

// device name -> free and total bytes, "host" is the system memory
static std::map<std::string, std::pair<size_t, size_t>> mem_query(const std::vector<ggml_backend_dev_t> & devs) {
    std::map<std::string, std::pair<size_t, size_t>> res;
    for (auto * dev : devs) {
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        res[ggml_backend_dev_name(dev)] = { free, total };
    }
    size_t free  = 0;
    size_t total = 0;
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (cpu != nullptr) {
        ggml_backend_dev_memory(cpu, &free, &total);
    }
    // outside windows the cpu device reports all memory as free
    const size_t avail = common_host_mem_available();
    res["host"] = { avail > 0 ? avail : free, total };
    return res;
}

void server_role::wait_mem(const common_params & params) {
    mem_devs  = mem_devices(params);
    mem_start = mem_query(mem_devs);

    const char * env = std::getenv("LLAMA_SERVER_MEM_WAIT");
    if (env == nullptr) {
        return;
    }
    int64_t wait_ms = 0;
    std::map<std::string, size_t> need;
    try {
        const json cfg = json::parse(env);
        wait_ms = cfg.at("ms").get<int64_t>();
        for (const auto & [dev, n] : cfg.at("need").items()) {
            need[dev] = n.get<size_t>();
        }
    } catch (const std::exception & e) {
        SRV_WRN("ignoring LLAMA_SERVER_MEM_WAIT: %s\n", e.what());
        return;
    }

    const int64_t t_start = ggml_time_ms();
    bool waited = false;
    while (true) {
        std::string lack;
        for (const auto & [dev, n] : need) {
            auto it = mem_start.find(dev);
            if (it != mem_start.end() && it->second.first < n) {
                lack += string_format(" %s %zu/%zu MiB", dev.c_str(), it->second.first / (1024 * 1024), n / (1024 * 1024));
            }
        }
        const int64_t t = ggml_time_ms() - t_start;
        if (lack.empty()) {
            if (waited) {
                SRV_INF("previous instance released its memory after %.1f s\n", t / 1000.0);
            }
            return;
        }
        if (t >= wait_ms) {
            SRV_WRN("previous instance still holds memory after %.1f s, loading anyway, free/needed:%s\n", t / 1000.0, lack.c_str());
            return;
        }
        if (!waited) {
            SRV_INF("waiting up to %.1f s until the previous instance releases its memory, free/needed:%s\n", wait_ms / 1000.0, lack.c_str());
            waited = true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        mem_start = mem_query(mem_devs);
    }
}

json server_role::mem_report() {
    json res = json::object();
    for (const auto & [dev, cur] : mem_query(mem_devs)) {
        auto it = mem_start.find(dev);
        if (it == mem_start.end()) {
            continue;
        }
        const size_t free = it->second.first;
        res[dev] = {
            {"free",  free},
            {"used",  free > cur.first ? free - cur.first : 0},
            {"total", cur.second},
        };
    }
    return res;
}
