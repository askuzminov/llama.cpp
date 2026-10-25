#include "llama-fork.h"
#include "llama-env-parse-fork.h"

#include "llama-impl.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <mutex>
#include <set>
#include <sstream>
#include <string>

int64_t llama_fork_env_int(const char * name, int64_t def) {
    const char * env = getenv(name);
    int64_t v = def;
    const llama_fork_env_parse_result res = llama_fork_env_parse(env, v);
    if (res == LLAMA_FORK_ENV_UNSET) {
        return def;
    }
    if (res == LLAMA_FORK_ENV_BAD) {
        static std::mutex            mutex;
        static std::set<std::string> warned;
        std::lock_guard<std::mutex> lock(mutex);
        // an error, not a warning: the first read of a knob is often in a context of common_fit_params, which turns
        // whatever is below an error into a debug line, and the value is cached after it
        if (warned.insert(name).second) {
            LLAMA_LOG_ERROR("%s: %s=\"%s\" is not a whole number, using %lld\n", __func__, name, env, (long long) def);
        }
        return def;
    }
    return v;
}

bool llama_fork_upstream(const char * component) {
    const char * env = getenv("LLAMA_UPSTREAM");
    if (env == nullptr) {
        return false;
    }
    std::stringstream ss(env);
    for (std::string item; std::getline(ss, item, ',');) {
        if (item == component || item == "all") {
            return true;
        }
    }
    return false;
}
