// fork: the two helpers of src/llama-impl.cpp that the row cache and llama_file use, for test-row-cache built from
// their sources (tests/tests-fork.cmake, LLAMA_FORK_ROW_CACHE_STANDALONE): a Windows DLL of libllama exports
// LLAMA_API only, so the test cannot link them from it. the same code as in llama-impl.cpp, the log goes to stderr

#include "llama-impl.h"

#include <climits>
#include <cstdarg>
#include <cstdio>
#include <string>
#include <vector>

void llama_log_internal(ggml_log_level level, const char * fmt, ...) {
    (void) level;
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fflush(stderr);
}

std::string format(const char * fmt, ...) {
    va_list ap;
    va_list ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX); // NOLINT
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}
