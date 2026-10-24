// the fork's knobs of the environment (common_fork_env_int, common/fork/env-fork.h; llama_fork_env_int in libllama is
// the same code): a value that is not a whole number gives the default and a warning, not the 0 of atoi, which for most
// knobs means off or never

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

static int64_t with(const char * value, int64_t def) {
    set_env("LLAMA_TEST_ENV_FORK", value);
    return common_fork_env_int("LLAMA_TEST_ENV_FORK", def);
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

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }

    printf("OK\n");

    return 0;
}
