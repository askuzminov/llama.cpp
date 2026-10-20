#pragma once

// the fork's watchdog of llama_model_loader::load_all_data. a model load that stops names the step it stopped in: on
// the 395 (Vulkan, UMA) the draft model load of llama-server stopped after its buffer allocation in 12 of 43 starts
// (06.10-11.10), the log ending there. the loader sets the step before every call that can block; a step lasting
// LLAMA_LOAD_WATCH seconds (15, 0 = off) is logged once, and again with its time when it ends. a step lasting
// LLAMA_LOAD_WATCH_EXIT seconds (180, 0 = never) ends the process with exit code 75 (EX_TEMPFAIL), so that whoever
// started it can start it again: the stops seen on the 395 never ended on their own

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

struct llama_load_watch_fork {
    llama_load_watch_fork();
    ~llama_load_watch_fork();

    // tensor and chunk only describe the step, chunk < 0 for none
    void step(const char * name, const char * tensor = nullptr, int64_t chunk = -1);

private:
    void run();
    std::string describe() const;

    using clock = std::chrono::steady_clock;

    std::mutex              mutex;
    std::condition_variable cv;
    std::thread             thread;
    bool                    stop     = false;
    bool                    reported = false;
    const char *            name     = "start";
    std::string             tensor;
    int64_t                 chunk    = -1;
    clock::time_point       t_step;
    int                     limit_s  = 0;
    int                     exit_s   = 0;
};
