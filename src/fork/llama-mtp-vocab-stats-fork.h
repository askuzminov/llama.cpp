#pragma once

// the token counts of the vocabulary subset of the MTP head (LLAMA_MTP_VOCAB_STATS, src/fork/qwen4exp-fork.cpp): the
// file, its default place in the llama.cpp cache (one file per vocabulary), and the save that adds the tokens of a run
// to what the file holds at that moment. header-only, so that tests/test-mtp-vocab-stats-fork.cpp checks it on any
// build

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#if !defined(_WIN32)
#    include <pwd.h>
#    include <unistd.h>
#endif

enum llama_mtp_vocab_stats_status_fork {
    LLAMA_MTP_VOCAB_STATS_READ_FORK,    // the counts of the file were added
    LLAMA_MTP_VOCAB_STATS_MISSING_FORK, // no file
    LLAMA_MTP_VOCAB_STATS_FOREIGN_FORK, // something that is not a counts file
    LLAMA_MTP_VOCAB_STATS_OTHER_FORK,   // the counts of another vocabulary size
};

// a path as UTF-8, for the log
static inline std::string llama_mtp_vocab_stats_text_fork(const std::filesystem::path & p) {
#if defined(__cpp_lib_char8_t)
    const auto s = p.u8string();
    return std::string(s.begin(), s.end());
#else
    return p.u8string();
#endif
}

// adds the counts of path ("llama-mtp-vocab-stats 1 <n>", then "<id> <count>" lines) to counts (n_vocab entries); n is
// the rows of the LM head (n_vocab) or the tokens of the vocabulary (n_tok)
static inline llama_mtp_vocab_stats_status_fork llama_mtp_vocab_stats_read_fork(
        const std::filesystem::path & path, int64_t n_vocab, int64_t n_tok, std::vector<uint64_t> & counts) {
    std::error_code ec;
    const auto st = std::filesystem::status(path, ec);
    if (st.type() == std::filesystem::file_type::not_found) {
        return LLAMA_MTP_VOCAB_STATS_MISSING_FORK;
    }
    if (ec || !std::filesystem::is_regular_file(st)) {
        return LLAMA_MTP_VOCAB_STATS_FOREIGN_FORK;
    }
    std::ifstream f(path);
    std::string   magic;
    int           version = 0;
    int64_t       n       = 0;
    if (!f || !(f >> magic >> version >> n) || magic != "llama-mtp-vocab-stats" || version != 1) {
        return LLAMA_MTP_VOCAB_STATS_FOREIGN_FORK;
    }
    if (n != n_vocab && n != n_tok) {
        return LLAMA_MTP_VOCAB_STATS_OTHER_FORK;
    }
    int64_t  id    = 0;
    uint64_t count = 0;
    while (f >> id >> count) {
        if (id >= 0 && id < n_vocab && id < (int64_t) counts.size()) {
            counts[(size_t) id] += count;
        }
    }
    return LLAMA_MTP_VOCAB_STATS_READ_FORK;
}

// writes the counts path holds now plus run (the tokens counted since the load; another server may have saved its own
// in between), through a temporary file next to it. a file that is not a counts file or counts another vocabulary is
// left alone. false with the reason in err when nothing was written
static inline bool llama_mtp_vocab_stats_save_fork(const std::filesystem::path & path, int64_t n_vocab, int64_t n_tok,
        const std::vector<uint64_t> & run, std::string & err) {
    std::vector<uint64_t> total((size_t) n_vocab, 0);
    switch (llama_mtp_vocab_stats_read_fork(path, n_vocab, n_tok, total)) {
        case LLAMA_MTP_VOCAB_STATS_FOREIGN_FORK:
            err = "not a vocabulary statistics file, left as it is";
            return false;
        case LLAMA_MTP_VOCAB_STATS_OTHER_FORK:
            err = "the counts of another vocabulary, left as they are";
            return false;
        default:
            break;
    }
    for (size_t t = 0; t < total.size() && t < run.size(); ++t) {
        total[t] += run[t];
    }
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::filesystem::path tmp = path;
    tmp += ".tmp";
    {
        std::ofstream f(tmp, std::ios::trunc);
        if (f) {
            f << "llama-mtp-vocab-stats 1 " << n_vocab << "\n";
            for (size_t t = 0; t < total.size(); ++t) {
                if (total[t]) {
                    f << t << " " << total[t] << "\n";
                }
            }
            f.flush();
        }
        if (!f) {
            err = "cannot write " + llama_mtp_vocab_stats_text_fork(tmp);
            f.close();
            std::filesystem::remove(tmp, ec);
            return false;
        }
    }
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        err = "cannot replace it: " + ec.message();
        std::error_code ec2;
        std::filesystem::remove(tmp, ec2);
        return false;
    }
    return true;
}

// a variable of the environment as a path (the wide form on Windows), empty when unset or empty
static inline std::filesystem::path llama_mtp_vocab_stats_env_path_fork(const char * name) {
#if defined(_WIN32)
    const std::wstring wname(name, name + strlen(name));
    const wchar_t *    v = _wgetenv(wname.c_str());
    return v && *v ? std::filesystem::path(v) : std::filesystem::path();
#else
    const char * v = getenv(name);
    return v && *v ? std::filesystem::path(v) : std::filesystem::path();
#endif
}

// the directory of the llama.cpp cache as fs_get_cache_directory of common finds it: LLAMA_CACHE, else
// %LOCALAPPDATA%\llama.cpp, ~/Library/Caches/llama.cpp (macOS), $XDG_CACHE_HOME/llama.cpp or ~/.cache/llama.cpp;
// empty when there is none
static inline std::filesystem::path llama_mtp_vocab_stats_cache_dir_fork() {
    std::filesystem::path dir = llama_mtp_vocab_stats_env_path_fork("LLAMA_CACHE");
    if (!dir.empty()) {
        return dir;
    }
#if defined(_WIN32)
    dir = llama_mtp_vocab_stats_env_path_fork("LOCALAPPDATA");
#else
    std::filesystem::path home = llama_mtp_vocab_stats_env_path_fork("HOME");
    if (home.empty()) {
        const struct passwd * pw = getpwuid(getuid());
        if (pw && pw->pw_dir && *pw->pw_dir) {
            home = pw->pw_dir;
        }
    }
#    if defined(__APPLE__)
    if (!home.empty()) {
        dir = home / "Library" / "Caches";
    }
#    else
    dir = llama_mtp_vocab_stats_env_path_fork("XDG_CACHE_HOME");
    if (dir.empty() && !home.empty()) {
        dir = home / ".cache";
    }
#    endif
#endif
    return dir.empty() ? dir : dir / "llama.cpp";
}

// FNV-1a, for the name of the default file: the vocabularies of models with the same tokenizer give the same name
static const uint64_t LLAMA_MTP_VOCAB_STATS_HASH0_FORK = 0xcbf29ce484222325ULL;

static inline uint64_t llama_mtp_vocab_stats_hash_fork(uint64_t h, const void * data, size_t n) {
    const unsigned char * p = (const unsigned char *) data;
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

// the file of LLAMA_MTP_VOCAB_STATS (value, null when unset): unset - mtp-vocab-stats-<vocab_hash>.txt in the llama.cpp
// cache (empty without a cache directory), empty, "none" or "0" - no file, anything else - that path
static inline std::filesystem::path llama_mtp_vocab_stats_path_fork(const char * value, uint64_t vocab_hash) {
    if (value == nullptr) {
        const std::filesystem::path dir = llama_mtp_vocab_stats_cache_dir_fork();
        if (dir.empty()) {
            return {};
        }
        char name[64];
        snprintf(name, sizeof(name), "mtp-vocab-stats-%016" PRIx64 ".txt", vocab_hash);
        return dir / name;
    }
    if (*value == '\0' || strcmp(value, "none") == 0 || strcmp(value, "0") == 0) {
        return {};
    }
    return std::filesystem::path(value);
}
