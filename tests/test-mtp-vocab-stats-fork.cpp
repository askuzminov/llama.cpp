// the counts file of the vocabulary subset of the MTP head (LLAMA_MTP_VOCAB_STATS, src/fork/llama-mtp-vocab-stats-fork.h):
// the read, the save that adds the tokens of a run to the file as it is at that moment (another server may have saved
// since the load), the files the save leaves alone, the default place in the llama.cpp cache and the values of the
// variable. header-only, so this runs on a Windows DLL build too

#include "fork/llama-mtp-vocab-stats-fork.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static int n_fail = 0;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "line %d: %s\n", line, what);
        n_fail++;
    }
}

#define CHECK(x) check((x), #x, __LINE__)

static std::string slurp(const fs::path & p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

static void spit(const fs::path & p, const std::string & s) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f << s;
}

// null unsets
static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value ? value : "");
    const std::wstring wname(name, name + strlen(name));
    const std::string  v = value ? value : "";
    const std::wstring wvalue(v.begin(), v.end());
    _wputenv_s(wname.c_str(), wvalue.c_str());
#else
    if (value) {
        setenv(name, value, 1);
    } else {
        unsetenv(name);
    }
#endif
}

struct saved_env {
    const char * name;
    bool         had;
    std::string  value;
    explicit saved_env(const char * name) : name(name) {
        const char * v = getenv(name);
        had   = v != nullptr;
        value = v ? v : "";
    }
    ~saved_env() { set_env(name, had ? value.c_str() : nullptr); }
};

static std::vector<uint64_t> counts_of(const fs::path & p, int64_t n_vocab, int64_t n_tok, llama_mtp_vocab_stats_status_fork * st = nullptr) {
    std::vector<uint64_t> c((size_t) n_vocab, 0);
    const auto s = llama_mtp_vocab_stats_read_fork(p, n_vocab, n_tok, c);
    if (st) {
        *st = s;
    }
    return c;
}

int main() {
    const fs::path dir = fs::temp_directory_path() /
        ("test-mtp-vocab-stats-fork-" + std::to_string((unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(dir);

    const int64_t n_vocab = 10; // rows of the LM head
    const int64_t n_tok   = 8;  // tokens of the vocabulary
    std::string   err;

    // a missing file: nothing read; the save creates it with the counts of the run, through no leftover temporary
    {
        const fs::path p = dir / "a.txt";
        llama_mtp_vocab_stats_status_fork st;
        const auto c = counts_of(p, n_vocab, n_tok, &st);
        CHECK(st == LLAMA_MTP_VOCAB_STATS_MISSING_FORK);
        CHECK(c == std::vector<uint64_t>(10, 0));

        std::vector<uint64_t> run(10, 0);
        run[1] = 3;
        run[5] = 7;
        CHECK(llama_mtp_vocab_stats_save_fork(p, n_vocab, n_tok, run, err));
        CHECK(slurp(p) == "llama-mtp-vocab-stats 1 10\n1 3\n5 7\n");
        CHECK(!fs::exists(dir / "a.txt.tmp"));

        // the next run adds to it
        std::vector<uint64_t> run2(10, 0);
        run2[5] = 1;
        run2[7] = 2;
        CHECK(llama_mtp_vocab_stats_save_fork(p, n_vocab, n_tok, run2, err));
        CHECK(slurp(p) == "llama-mtp-vocab-stats 1 10\n1 3\n5 8\n7 2\n");

        // another server saved after this one loaded: the save adds to what the file holds now
        spit(p, "llama-mtp-vocab-stats 1 10\n2 100\n");
        std::vector<uint64_t> run3(10, 0);
        run3[2] = 1;
        run3[9] = 4;
        CHECK(llama_mtp_vocab_stats_save_fork(p, n_vocab, n_tok, run3, err));
        CHECK(slurp(p) == "llama-mtp-vocab-stats 1 10\n2 101\n9 4\n");

        st = LLAMA_MTP_VOCAB_STATS_MISSING_FORK;
        const auto c2 = counts_of(p, n_vocab, n_tok, &st);
        CHECK(st == LLAMA_MTP_VOCAB_STATS_READ_FORK);
        CHECK(c2[2] == 101 && c2[9] == 4 && c2[0] == 0);
    }

    // a file of the tokens of the vocabulary (n_tok) is read; the save writes the rows of the head; ids outside the
    // head are skipped
    {
        const fs::path p = dir / "b.txt";
        spit(p, "llama-mtp-vocab-stats 1 8\n3 4\n-1 5\n10 6\n9 1\n");
        llama_mtp_vocab_stats_status_fork st;
        const auto c = counts_of(p, n_vocab, n_tok, &st);
        CHECK(st == LLAMA_MTP_VOCAB_STATS_READ_FORK);
        std::vector<uint64_t> want(10, 0);
        want[3] = 4;
        want[9] = 1;
        CHECK(c == want);
        std::vector<uint64_t> run(10, 0);
        run[3] = 1;
        CHECK(llama_mtp_vocab_stats_save_fork(p, n_vocab, n_tok, run, err));
        CHECK(slurp(p) == "llama-mtp-vocab-stats 1 10\n3 5\n9 1\n");
    }

    // what the save leaves alone: not a counts file, the counts of another vocabulary, a directory
    {
        std::vector<uint64_t> run(10, 0);
        run[1] = 1;

        const fs::path foreign = dir / "foreign.gguf";
        spit(foreign, "GGUF\x03 not counts\n");
        llama_mtp_vocab_stats_status_fork st;
        counts_of(foreign, n_vocab, n_tok, &st);
        CHECK(st == LLAMA_MTP_VOCAB_STATS_FOREIGN_FORK);
        err.clear();
        CHECK(!llama_mtp_vocab_stats_save_fork(foreign, n_vocab, n_tok, run, err));
        CHECK(!err.empty());
        CHECK(slurp(foreign) == "GGUF\x03 not counts\n");

        const fs::path other = dir / "other.txt";
        spit(other, "llama-mtp-vocab-stats 1 12345\n1 1\n");
        const auto c = counts_of(other, n_vocab, n_tok, &st);
        CHECK(st == LLAMA_MTP_VOCAB_STATS_OTHER_FORK);
        CHECK(c == std::vector<uint64_t>(10, 0));
        CHECK(!llama_mtp_vocab_stats_save_fork(other, n_vocab, n_tok, run, err));
        CHECK(slurp(other) == "llama-mtp-vocab-stats 1 12345\n1 1\n");

        const fs::path version2 = dir / "v2.txt";
        spit(version2, "llama-mtp-vocab-stats 2 10\n1 1\n");
        counts_of(version2, n_vocab, n_tok, &st);
        CHECK(st == LLAMA_MTP_VOCAB_STATS_FOREIGN_FORK);

        const fs::path sub = dir / "a-directory";
        fs::create_directories(sub);
        counts_of(sub, n_vocab, n_tok, &st);
        CHECK(st == LLAMA_MTP_VOCAB_STATS_FOREIGN_FORK);
        CHECK(!llama_mtp_vocab_stats_save_fork(sub, n_vocab, n_tok, run, err));
        CHECK(fs::is_directory(sub));
    }

    // the save makes the directories of the path
    {
        const fs::path p = dir / "new" / "deeper" / "c.txt";
        std::vector<uint64_t> run(10, 0);
        run[0] = 2;
        CHECK(llama_mtp_vocab_stats_save_fork(p, n_vocab, n_tok, run, err));
        CHECK(slurp(p) == "llama-mtp-vocab-stats 1 10\n0 2\n");
    }

    // the name hash (FNV-1a test vectors)
    CHECK(llama_mtp_vocab_stats_hash_fork(LLAMA_MTP_VOCAB_STATS_HASH0_FORK, "", 0) == 0xcbf29ce484222325ULL);
    CHECK(llama_mtp_vocab_stats_hash_fork(LLAMA_MTP_VOCAB_STATS_HASH0_FORK, "a", 1) == 0xaf63dc4c8601ec8cULL);
    CHECK(llama_mtp_vocab_stats_hash_fork(LLAMA_MTP_VOCAB_STATS_HASH0_FORK, "foobar", 6) == 0x85944171f73967e8ULL);

    // the values of LLAMA_MTP_VOCAB_STATS and the default place
    {
        saved_env e1("LLAMA_CACHE");
        saved_env e2("LOCALAPPDATA");
        saved_env e3("HOME");
        saved_env e4("XDG_CACHE_HOME");

        CHECK(llama_mtp_vocab_stats_path_fork("none", 1).empty());
        CHECK(llama_mtp_vocab_stats_path_fork("0", 1).empty());
        CHECK(llama_mtp_vocab_stats_path_fork("", 1).empty());
        CHECK(llama_mtp_vocab_stats_path_fork("some/where.txt", 1) == fs::path("some/where.txt"));

        const std::string name = "mtp-vocab-stats-00000000000000ab.txt";
        const fs::path    cache = dir / "cache";
        set_env("LLAMA_CACHE", cache.string().c_str());
        CHECK(llama_mtp_vocab_stats_path_fork(nullptr, 0xab) == cache / name);

        set_env("LLAMA_CACHE", nullptr);
#if defined(_WIN32)
        const fs::path local = dir / "local";
        set_env("LOCALAPPDATA", local.string().c_str());
        CHECK(llama_mtp_vocab_stats_path_fork(nullptr, 0xab) == local / "llama.cpp" / name);
#else
        const fs::path home = dir / "home";
        set_env("HOME", home.string().c_str());
        set_env("XDG_CACHE_HOME", nullptr);
#    if defined(__APPLE__)
        CHECK(llama_mtp_vocab_stats_path_fork(nullptr, 0xab) == home / "Library" / "Caches" / "llama.cpp" / name);
#    else
        CHECK(llama_mtp_vocab_stats_path_fork(nullptr, 0xab) == home / ".cache" / "llama.cpp" / name);
        const fs::path xdg = dir / "xdg";
        set_env("XDG_CACHE_HOME", xdg.string().c_str());
        CHECK(llama_mtp_vocab_stats_path_fork(nullptr, 0xab) == xdg / "llama.cpp" / name);
#    endif
#endif
    }

    std::error_code ec;
    fs::remove_all(dir, ec);

    if (n_fail) {
        fprintf(stderr, "test-mtp-vocab-stats-fork: %d failed\n", n_fail);
        return 1;
    }
    printf("test-mtp-vocab-stats-fork: ok\n");
    return 0;
}
