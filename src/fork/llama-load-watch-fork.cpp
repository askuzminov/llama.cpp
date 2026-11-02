#include "llama-load-watch-fork.h"

#include "llama-fork.h"
#include "llama-impl.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <csignal>
#include <cstring>
#include <exception>
#endif

// the load in progress, for the crash report
static std::atomic<llama_load_watch_fork *> g_load_watch{nullptr};

llama_load_watch_fork::llama_load_watch_fork() {
    static const int limit      = (int) llama_fork_env_int("LLAMA_LOAD_WATCH", 15);
    static const int exit_after = (int) llama_fork_env_int("LLAMA_LOAD_WATCH_EXIT", 180);
    limit_s = limit;
    exit_s  = exit_after;
    t_step  = clock::now();
    if (limit_s > 0 || exit_s > 0) {
        thread = std::thread([this] { run(); });
    }
    g_load_watch.store(this);
}

llama_load_watch_fork::~llama_load_watch_fork() {
    llama_load_watch_fork * self = this;
    g_load_watch.compare_exchange_strong(self, nullptr);
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

bool llama_load_watch_fork::current(std::string & out, double & seconds) {
    // without the watchdog step() records nothing
    if (!thread.joinable()) {
        return false;
    }
    std::unique_lock<std::mutex> lock(mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        return false;
    }
    out     = describe();
    seconds = std::chrono::duration<double>(clock::now() - t_step).count();
    return true;
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

#ifdef _WIN32

// the crash report. a release build of MSVC ends the process on abort() with 0xc0000409 and no word, and so it does on
// an exception that leaves a noexcept frame in any thread but the main one: the terminate handler of ggml.cpp is per
// thread there. on the 395 eight model loads ended so after 'mapMemory: ErrorMemoryMapFailed' (31.10-01.11), with no
// event in the Windows log. SIGABRT and the invalid parameter handler are per process: they print the thread, the last
// C++ exception thrown in it (a vectored handler notes every throw), the stack as module+offset and the step of a model
// load, on stderr directly: the log goes through a queue that the end of the process does not drain.
// LLAMA_CRASH_REPORT=0: off

namespace {

DWORD g_main_thread = 0;

// the last C++ exception thrown in this thread: its type, what() of a std::exception, when
thread_local char      t_exception[384];
thread_local ULONGLONG t_exception_ms = 0;

// the throw information of MSVC on x64 (ehdata.h): offsets from the base of the module that threw
struct eh_type_desc  { const void * vftable; void * spare; char name[1]; };
struct eh_catchable  { unsigned int properties; int type; int mdisp; int pdisp; int vdisp; int size; int copy; };
struct eh_catchables { int n; int types[1]; };
struct eh_throw_info { unsigned int attributes; int unwind; int forward_compat; int catchables; };

LONG WINAPI note_cpp_exception(EXCEPTION_POINTERS * ep) {
    const EXCEPTION_RECORD * er = ep->ExceptionRecord;
    // 0xE06D7363: a C++ throw, its parameters the magic number, the object, the throw information, the module base
    if (er->ExceptionCode != 0xE06D7363 || er->NumberParameters < 4 ||
        er->ExceptionInformation[0] < 0x19930520 || er->ExceptionInformation[0] > 0x19930522 ||
        er->ExceptionInformation[2] == 0 || er->ExceptionInformation[3] == 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const char * base = (const char *) er->ExceptionInformation[3];
    const auto * info = (const eh_throw_info *) er->ExceptionInformation[2];
    const auto * list = (const eh_catchables *) (base + info->catchables);
    const char * type = nullptr;
    const char * what = nullptr;
    for (int i = 0; i < list->n; ++i) {
        const auto * ct   = (const eh_catchable *) (base + list->types[i]);
        const char * name = ((const eh_type_desc *) (base + ct->type))->name;
        if (i == 0) {
            type = name;
        }
        // std::exception as a plain base: at mdisp in the object
        if (ct->pdisp == -1 && er->ExceptionInformation[1] != 0 && strcmp(name, ".?AVexception@std@@") == 0) {
            what = ((const std::exception *) ((const char *) er->ExceptionInformation[1] + ct->mdisp))->what();
        }
    }
    snprintf(t_exception, sizeof(t_exception), "%s%s%s", type ? type : "?", what ? ": " : "", what ? what : "");
    t_exception_ms = GetTickCount64();
    return EXCEPTION_CONTINUE_SEARCH;
}

void crash_report(const char * event) {
    const DWORD tid = GetCurrentThreadId();
    fprintf(stderr, "%s in thread %lu%s\n", event, (unsigned long) tid, tid == g_main_thread ? " (the main thread)" : "");
    if (t_exception_ms != 0) {
        fprintf(stderr, "  the last C++ exception of this thread, %.1f s before: %s\n",
                (GetTickCount64() - t_exception_ms) / 1000.0, t_exception);
    }
    void * frames[62];
    const USHORT n = RtlCaptureStackBackTrace(1, 62, frames, nullptr);
    for (USHORT i = 0; i < n; ++i) {
        HMODULE module = nullptr;
        char    path[MAX_PATH] = "?";
        if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                               (LPCSTR) frames[i], &module)) {
            GetModuleFileNameA(module, path, sizeof(path));
        }
        const char * file = strrchr(path, '\\');
        fprintf(stderr, "  %s+0x%llx\n", file ? file + 1 : path,
                (unsigned long long) ((const char *) frames[i] - (const char *) module));
    }
    fflush(stderr);
    // last: it allocates, and the heap may be what failed
    if (llama_load_watch_fork * watch = g_load_watch.load()) {
        std::string step;
        double      seconds = 0.0;
        if (watch->current(step, seconds)) {
            fprintf(stderr, "  a model load is at: %s, for %.1f s\n", step.c_str(), seconds);
            fflush(stderr);
        }
    }
    // for the lines still in the queue of the log
    Sleep(200);
}

void on_abort(int) {
    // abort() ends the process after this, as before
    crash_report("abort()");
}

void on_invalid_parameter(const wchar_t *, const wchar_t *, const wchar_t *, unsigned int, uintptr_t) {
    crash_report("an invalid parameter of a C runtime function");
    // what the default handler of a release build does
    __fastfail(FAST_FAIL_INVALID_ARG);
}

bool g_crash_report = [] {
    if (llama_fork_env_int("LLAMA_CRASH_REPORT", 1) == 0) {
        return false;
    }
    g_main_thread = GetCurrentThreadId();
    AddVectoredExceptionHandler(0, note_cpp_exception);
    signal(SIGABRT, on_abort);
    _set_invalid_parameter_handler(on_invalid_parameter);
    return true;
}();

} // namespace

#endif
