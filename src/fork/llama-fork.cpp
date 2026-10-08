#include "llama-fork.h"

#include <cstdlib>
#include <sstream>
#include <string>

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
