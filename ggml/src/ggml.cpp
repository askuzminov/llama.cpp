#include "ggml-impl.h"

#include <cstdio>
#include <cstdlib>
#include <exception>

static std::terminate_handler previous_terminate_handler;

GGML_NORETURN static void ggml_uncaught_exception() {
    // the message of the exception: the backtrace prints nothing on some platforms (Windows)
    if (std::exception_ptr e = std::current_exception()) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception & ex) {
            fprintf(stderr, "%s: %s\n", __func__, ex.what());
        } catch (...) {
            fprintf(stderr, "%s: an exception that is not a std::exception\n", __func__);
        }
    }
    ggml_print_backtrace();
    if (previous_terminate_handler) {
        previous_terminate_handler();
    }
    abort(); // unreachable unless previous_terminate_handler was nullptr
}

static bool ggml_uncaught_exception_init = []{
    const char * GGML_NO_BACKTRACE = getenv("GGML_NO_BACKTRACE");
    if (GGML_NO_BACKTRACE) {
        return false;
    }
    const auto prev{std::get_terminate()};
    GGML_ASSERT(prev != ggml_uncaught_exception);
    previous_terminate_handler = prev;
    std::set_terminate(ggml_uncaught_exception);
    return true;
}();
