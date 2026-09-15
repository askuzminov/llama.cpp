// Round-trip coverage for the on-disk prompt-cache spill files (--cache-spill-dir).
//
// The spill path bypasses the OS file cache, so every transfer is block-aligned and the logical
// payload sizes live in the file header. These tests drive server_prompt_cache directly - no model
// is involved - over payload sizes that are deliberately awkward for an aligned layout (empty,
// one byte, exactly one block, one below a block), plus the cold-start adoption, signature
// mismatch and truncation paths.

#include "server-task.h"

#include <filesystem>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#endif

#undef NDEBUG
#include <cassert>

static const uint64_t SIG = 0xabcdef0123456789ull;

// the spill directory arrives from the CLI as UTF-8; on Windows a narrow filesystem path is read in
// the ANSI codepage instead, so both the server and this test have to convert explicitly. The test
// directories carry a non-ASCII suffix ("-cyrillic" in UTF-8 bytes) to exercise that conversion.
static const char * DIR_SUFFIX = "-\xd0\xbf\xd1\x83\xd1\x82\xd1\x8c";

static std::filesystem::path fs_path(const std::string & utf8) {
#if defined(_WIN32)
    if (utf8.empty()) {
        return {};
    }
    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int) utf8.size(), nullptr, 0);
    assert(n > 0);
    std::wstring wide((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int) utf8.size(), &wide[0], n);
    return std::filesystem::path(wide);
#else
    return std::filesystem::path(utf8);
#endif
}

static std::string fs_utf8(const std::filesystem::path & path) {
#if defined(_WIN32)
    const std::wstring wide = path.wstring();
    if (wide.empty()) {
        return {};
    }
    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), nullptr, 0, nullptr, nullptr);
    assert(n > 0);
    std::string utf8((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), &utf8[0], n, nullptr, nullptr);
    return utf8;
#else
    return path.string();
#endif
}

static std::vector<uint8_t> make_blob(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    std::mt19937 rng(seed);
    for (size_t i = 0; i < n; ++i) {
        v[i] = (uint8_t) (rng() & 0xff);
    }
    return v;
}

static llama_tokens make_tokens(size_t n, int32_t base) {
    llama_tokens t(n);
    for (size_t i = 0; i < n; ++i) {
        t[i] = base + (int32_t) i;
    }
    return t;
}

static server_prompt_cache_state make_state(size_t n_tokens, int32_t base, size_t n_main, size_t n_drft) {
    server_prompt_cache_state st;
    st.prompt.tokens = server_tokens(make_tokens(n_tokens, base), false);
    st.prompt.n_used = (uint32_t) base;
    st.data.main     = make_blob(n_main, (uint32_t) base + 1);
    st.data.drft     = make_blob(n_drft, (uint32_t) base + 2);
    return st;
}

struct spill_case {
    size_t  n_tokens;
    int32_t base;
    size_t  n_main;
    size_t  n_drft;
};

// sizes chosen around the 4096-byte block boundary
static const spill_case CASES[] = {
    {  16, 100,       1,    0 },
    {   0, 200,    4096,    7 },
    { 333, 300, 1234567,   89 },
    {   5, 400,       0, 4095 },
};

static void fill(server_prompt_cache & cache) {
    for (const spill_case & c : CASES) {
        cache.states.push_back(make_state(c.n_tokens, c.base, c.n_main, c.n_drft));
    }
}

static bool dir_empty(const std::string & dir) {
    return std::filesystem::directory_iterator(fs_path(dir)) == std::filesystem::directory_iterator();
}

// spill -> unspill within one run
static void test_roundtrip(const std::string & dir) {
    server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
    assert(cache.states.empty());

    fill(cache);

    std::vector<server_prompt_data> ref;
    for (const auto & st : cache.states) {
        ref.push_back(st.data);
    }

    for (auto & st : cache.states) {
        assert(cache.spill_state(st));
        assert(st.on_disk);
        assert(st.data.main.empty() && st.data.drft.empty());

        const uintmax_t n_file = std::filesystem::file_size(cache.spill_path(st.uid));
        assert(n_file % 4096 == 0);        // unbuffered I/O writes whole blocks only
        assert(n_file == st.disk_bytes);   // the disk budget must see the padded size
    }

    size_t i = 0;
    for (auto & st : cache.states) {
        assert(cache.unspill_state(st));
        assert(!st.on_disk);
        assert(st.data.main == ref[i].main);
        assert(st.data.drft == ref[i].drft);
        assert(!std::filesystem::exists(cache.spill_path(st.uid)));
        ++i;
    }
    assert(i == ref.size());

    cache.states.clear();
    assert(dir_empty(dir));
}

// shutdown -> restart: the files are adopted by a run with the same signature
static void test_cold_start(const std::string & dir) {
    std::vector<server_prompt_data> ref;
    std::vector<llama_tokens>       ref_tokens;

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
        fill(cache);
        for (const auto & st : cache.states) {
            ref.push_back(st.data);
            ref_tokens.push_back(st.prompt.tokens.get_text_tokens());
        }
    } // the destructor persists everything

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
        assert(cache.states.size() == ref.size());

        size_t i = 0;
        for (auto & st : cache.states) {
            assert(st.on_disk);
            assert(st.prompt.tokens.get_text_tokens() == ref_tokens[i]);
            assert(st.prompt.n_used == (uint32_t) CASES[i].base); // the use counter survives a restart

            assert(cache.unspill_state(st));
            assert(st.data.main == ref[i].main);
            assert(st.data.drft == ref[i].drft);
            ++i;
        }

        cache.states.clear();
    }

    assert(dir_empty(dir));
}

// a run with a different model / KV layout must discard the files instead of restoring them
static void test_signature_mismatch(const std::string & dir) {
    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
        fill(cache);
    }
    assert(!dir_empty(dir));

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG ^ 1ull);
        assert(cache.states.empty());
    }
    assert(dir_empty(dir));
}

// a short file must be rejected before anything is allocated from its header
static void test_truncated(const std::string & dir) {
    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
        cache.states.push_back(make_state(8, 600, 100000, 0));
    }

    for (const auto & entry : std::filesystem::directory_iterator(fs_path(dir))) {
        const uintmax_t n_file = std::filesystem::file_size(entry.path());
        assert(n_file > 4096);
        std::filesystem::resize_file(entry.path(), n_file - 4096);
    }

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
        assert(cache.states.empty());
    }
    assert(dir_empty(dir));
}

int main() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / "llama-spill-test";

    std::filesystem::remove_all(root);

    const struct {
        const char * name;
        void (*fn)(const std::string &);
    } tests[] = {
        { "roundtrip",          test_roundtrip          },
        { "cold-start",         test_cold_start         },
        { "signature-mismatch", test_signature_mismatch },
        { "truncated",          test_truncated          },
    };

    for (const auto & t : tests) {
        // no create_directories here: the cache must create the directory from the UTF-8 name itself
        const std::string dir = fs_utf8(root / fs_path(std::string(t.name) + DIR_SUFFIX));
        t.fn(dir);
        printf("%s: ok\n", t.name);
    }

    std::filesystem::remove_all(root);

    printf("all tests passed\n");

    return 0;
}
