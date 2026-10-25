// the fork's knobs of the environment: a value that is not a whole number gives the default and a warning, not the 0 of
// atoi, which for most knobs means off or never. both readers parse with llama_fork_env_parse
// (src/fork/llama-env-parse-fork.h): common_fork_env_int (common/fork/env-fork.h, common and the server) through the
// environment, and the core itself for llama_fork_env_int (libllama: LLAMA_MTP_WINDOW*, the QSA knobs), which a
// Windows DLL build does not export

#include "fork/env-fork.h"

#include <cstdio>
#include <cstdlib>

static int n_fail = 0;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "line %d: %s\n", line, what);
        n_fail++;
    }
}

#define CHECK(x) check((x), #x, __LINE__)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// a value through common_fork_env_int, and the same value through the core libllama reads with: both give the same
static int64_t with(const char * value, int64_t def) {
    set_env("LLAMA_TEST_ENV_FORK", value);
    const int64_t res = common_fork_env_int("LLAMA_TEST_ENV_FORK", def);

    int64_t core = def;
    if (llama_fork_env_parse(value, core) == LLAMA_FORK_ENV_BAD) {
        core = def;
    }
    check(core == res, "llama_fork_env_parse agrees with common_fork_env_int", __LINE__);
    return res;
}

int main() {
    // whole numbers as atoi read them
    CHECK(with("0", 7) == 0);
    CHECK(with("1", 7) == 1);
    CHECK(with("-1", 7) == -1);
    CHECK(with("+12", 7) == 12);
    CHECK(with("16384", 7) == 16384);
    CHECK(with("  42  ", 7) == 42);

    // empty: as unset (on Windows an empty value unsets the variable anyway)
    CHECK(with("", 7) == 7);
    CHECK(with("   ", 7) == 7);

    // not a whole number: the default, where atoi gave 0
    CHECK(with("abc", 7) == 7);
    CHECK(with("/abc", 512) == 512);
    CHECK(with("ture", -1) == -1);
    CHECK(with("1.5", 7) == 7);
    CHECK(with("12abc", 7) == 7);
    CHECK(with("0x10", 7) == 7);
    CHECK(with("99999999999999999999999", 7) == 7);

    // the core: no value at all, and the kinds of result the readers act on
    int64_t v = 5;
    CHECK(llama_fork_env_parse(nullptr, v) == LLAMA_FORK_ENV_UNSET && v == 5);
    CHECK(llama_fork_env_parse(" \t", v) == LLAMA_FORK_ENV_UNSET && v == 5);
    CHECK(llama_fork_env_parse("-9223372036854775808", v) == LLAMA_FORK_ENV_OK && v == INT64_MIN);
    CHECK(llama_fork_env_parse("16384x", v) == LLAMA_FORK_ENV_BAD);

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }

    printf("OK\n");

    return 0;
}
