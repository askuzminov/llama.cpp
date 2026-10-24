#pragma once

// a whole number of the environment, the fork's knobs in common and the server: def when the variable is unset or
// empty, and def with a warning (once per variable) when its value is not a whole number. atoi read such a value as 0,
// which for most knobs means off or never: a typo in an arm changed the layout without a word. libllama has the same
// in llama_fork_env_int (src/fork/llama-fork.h), with an error line: common_fit_params hides its warnings

#include "log.h"

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <set>
#include <string>

inline int64_t common_fork_env_int(const char * name, int64_t def) {
    const char * env = getenv(name);
    if (env == nullptr) {
        return def;
    }
    const char * p = env;
    while (isspace((unsigned char) *p)) {
        p++;
    }
    if (*p == '\0') {
        return def;
    }
    errno = 0;
    char * end = nullptr;
    const long long v = strtoll(p, &end, 10);
    while (end != nullptr && isspace((unsigned char) *end)) {
        end++;
    }
    if (end == p || end == nullptr || *end != '\0' || errno == ERANGE) {
        static std::mutex            mutex;
        static std::set<std::string> warned;
        std::lock_guard<std::mutex> lock(mutex);
        if (warned.insert(name).second) {
            LOG_WRN("%s: %s=\"%s\" is not a whole number, using %lld\n", __func__, name, env, (long long) def);
        }
        return def;
    }
    return v;
}
