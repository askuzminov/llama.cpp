// fork: router memory wait and child exit handling, see server-models-fork.h

#include "server-models-fork.h"

#include "server-common.h"
#include "server-models.h"

#include "common.h"
#include "log.h"

#include <algorithm>
#include <chrono>
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

// the driver can release the memory of an exited child some time after the process ended.
// a child spawned in this time after an exit first waits for that memory, see mem_wait_env
#define MEM_WAIT_MS 120000

//
// server_models
//

std::string server_models::mem_wait_env() const {
    const int64_t left = t_last_exit + MEM_WAIT_MS - ggml_time_ms();
    if (t_last_exit == 0 || left <= 0) {
        return "";
    }
    json need = json::object();
    for (const auto & [dev, m] : mem_last.items()) {
        const int64_t free  = json_value(m, "free",  (int64_t) 0);
        const int64_t used  = json_value(m, "used",  (int64_t) 0);
        const int64_t total = json_value(m, "total", (int64_t) 0);
        // other programs also change the free memory, so only a device the last child filled noticeably is waited for
        const int64_t slack = std::max<int64_t>(1ll << 30, total / 20);
        if (used >= slack) {
            // room for the same load, but not more than was free before it
            need[dev] = std::min(free - slack, used + slack);
        }
    }
    if (need.empty()) {
        return "";
    }
    return safe_json_to_str({{"ms", left}, {"need", need}});
}

//
// server_child
//

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

void server_child::exit_now() {
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

void server_child::wait_mem(const common_params & params) {
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

json server_child::mem_report() {
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
