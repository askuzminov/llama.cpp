#pragma once

// fork: the parse of a whole number knob of the environment, the one core of llama_fork_env_int (libllama,
// src/fork/llama-fork.cpp) and common_fork_env_int (common and the server, common/fork/env-fork.h). header-only, so a
// test checks the parse both read their knobs with, on a Windows DLL build too (test-env-fork)

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>

enum llama_fork_env_parse_result {
    LLAMA_FORK_ENV_UNSET, // no value, or only blanks: the default, without a word
    LLAMA_FORK_ENV_OK,
    LLAMA_FORK_ENV_BAD,   // not a whole number, or out of range: the default, and the caller says so once
};

// a decimal whole number with optional sign and surrounding blanks. atoi read anything else as 0, which for most knobs
// means off or never
static inline llama_fork_env_parse_result llama_fork_env_parse(const char * env, int64_t & value) {
    if (env == nullptr) {
        return LLAMA_FORK_ENV_UNSET;
    }
    const char * p = env;
    while (isspace((unsigned char) *p)) {
        p++;
    }
    if (*p == '\0') {
        return LLAMA_FORK_ENV_UNSET;
    }
    errno = 0;
    char * end = nullptr;
    const long long v = strtoll(p, &end, 10);
    while (end != nullptr && isspace((unsigned char) *end)) {
        end++;
    }
    if (end == p || end == nullptr || *end != '\0' || errno == ERANGE) {
        return LLAMA_FORK_ENV_BAD;
    }
    value = (int64_t) v;
    return LLAMA_FORK_ENV_OK;
}
