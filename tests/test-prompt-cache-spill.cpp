// Round-trip coverage for the on-disk prompt-cache spill files (--cache-spill-dir).
//
// The spill path bypasses the OS file cache, so every transfer is block-aligned and the logical
// payload sizes live in the file header. These tests drive server_prompt_cache directly - no model
// is involved - over payload sizes that are deliberately awkward for an aligned layout (empty,
// one byte, exactly one block, one below a block, one block past the direct-transfer threshold),
// plus the cold-start adoption, signature mismatch and truncation paths.

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

static server_state_buf make_blob(size_t n, uint32_t seed) {
    server_state_buf v(n);
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

// sizes chosen around the 4096-byte block boundary, plus one payload past the 16 MiB point where
// the spill stops using its bounce buffer and hands the blob to the device directly
static const spill_case CASES[] = {
    {  16, 100,             1,        0 },
    {   0, 200,          4096,        7 },
    { 333, 300,       1234567,       89 },
    {   5, 400,             0,     4095 },
    {   7, 500, 16*1024*1024 + 4095, 16*1024*1024 },
};

static void fill(server_prompt_cache & cache) {
    for (const spill_case & c : CASES) {
        cache.states.push_back(make_state(c.n_tokens, c.base, c.n_main, c.n_drft));
    }
}

static bool dir_empty(const std::string & dir) {
    return std::filesystem::directory_iterator(fs_path(dir)) == std::filesystem::directory_iterator();
}

// the cache never drops a state without erasing its file first, so a test must not either
static void clear_cache(server_prompt_cache & cache) {
    for (const auto & st : cache.states) {
        cache.erase_spill(st);
    }
    cache.states.clear();
}

// a state whose front is already on disk waits until it grows a full minimum past it, and the
// rewrite then takes the shorter file with it
static void test_rewrite_step(const std::string & dir) {
    server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
    cache.min_tokens = 64;

    // turn 1: nothing on disk yet, so the state goes out whatever its length
    cache.states.push_back(make_state(100, 1, 40960, 0));
    assert(cache.write_behind(nullptr) == 1);

    const uint64_t uid_100 = cache.states.front().uid;

    cache.evict_state(cache.states.front()); // what alloc() does to a prefix of the next prompt

    // turn 2: 32 tokens past the prefix is not worth moving the blob again
    cache.states.push_back(make_state(132, 1, 40960, 0));
    assert(cache.write_behind(nullptr) == 0);
    assert(!cache.states.back().on_disk);
    assert(std::filesystem::exists(cache.spill_path(uid_100)));

    // turn 3: a full minimum past the prefix, so it is written and the prefix file is now redundant
    cache.states.push_back(make_state(164, 1, 40960, 0));
    assert(cache.write_behind(nullptr) == 1);
    assert(cache.states.back().is_clean());
    assert(!std::filesystem::exists(cache.spill_path(uid_100)));
    assert(cache.states.size() == 2);

    // without a minimum nothing is held back
    cache.min_tokens = 0;
    assert(cache.write_behind(nullptr) == 1);
    for (const auto & st : cache.states) {
        assert(st.is_clean());
    }

    clear_cache(cache);
    assert(dir_empty(dir));
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
        assert(st.is_clean());               // read back, and the file stays for the next cold start
        assert(st.data.main == ref[i].main);
        assert(st.data.drft == ref[i].drft);
        assert(std::filesystem::file_size(cache.spill_path(st.uid)) == st.disk_bytes);
        ++i;
    }
    assert(i == ref.size());

    clear_cache(cache);
    assert(dir_empty(dir));
}

// write-behind: the copy reaches the disk while the blobs stay in RAM
static void test_write_behind(const std::string & dir) {
    server_prompt_cache cache(0, 0, 0, dir, 0, SIG);

    fill(cache);

    std::vector<server_prompt_data> ref;
    for (const auto & st : cache.states) {
        ref.push_back(st.data);
    }

    assert(cache.write_behind(nullptr) == ref.size());
    assert(cache.write_behind(nullptr) == 0); // nothing is dirty any more

    size_t i = 0;
    for (auto & st : cache.states) {
        assert(st.is_clean());
        assert(st.data.main == ref[i].main); // the blobs are left untouched
        assert(st.data.drft == ref[i].drft);
        assert(std::filesystem::file_size(cache.spill_path(st.uid)) == st.disk_bytes);
        ++i;
    }

    // a hit on a clean state needs no read, and it does not cost the file either
    auto & first = cache.states.front();
    assert(cache.unspill_state(first));
    assert(first.is_clean());
    assert(first.data.main == ref[0].main);
    assert(std::filesystem::exists(cache.spill_path(first.uid)));

    // dropping the RAM of the rest costs no I/O and they still read back byte for byte
    i = 1;
    for (auto it = std::next(cache.states.begin()); it != cache.states.end(); ++it, ++i) {
        cache.evict_state(*it);
        assert(it->on_disk && it->data.size() == 0);

        assert(cache.unspill_state(*it));
        assert(it->is_clean());
        assert(it->data.main == ref[i].main);
        assert(it->data.drft == ref[i].drft);
    }

    clear_cache(cache);
    assert(dir_empty(dir));
}

// a write stopped part way through leaves nothing behind and the state stays in RAM
static void test_write_behind_cancel(const std::string & dir) {
    server_prompt_cache cache(0, 0, 0, dir, 0, SIG);

    // big enough to take the direct path, where the cancel is polled every 16 MiB
    cache.states.push_back(make_state(7, 700, 16*1024*1024 + 4095, 0));

    const server_state_buf ref = cache.states.front().data.main;

    // false on the first poll (so the write starts), true on every one after
    int n_polls = 0;
    const std::function<bool()> stop_after_start = [&n_polls]() { return ++n_polls > 1; };

    assert(cache.write_behind(stop_after_start) == 0);
    assert(n_polls > 1);                       // the write was stopped, not skipped
    assert(!cache.states.front().on_disk);
    assert(cache.states.front().data.main == ref);
    assert(dir_empty(dir));                    // no half-written file

    // and the next idle writes it out for real
    assert(cache.write_behind(nullptr) == 1);
    assert(cache.states.front().is_clean());

    clear_cache(cache);
    assert(dir_empty(dir));
}

// the minimum prompt length keeps short prompts out of the cache
static void test_min_tokens(const std::string & dir) {
    server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
    cache.min_tokens = 64;

    server_prompt prompt;

    prompt.tokens = server_tokens(make_tokens(63, 1), false);
    assert(cache.alloc(prompt, 1024, 0) == nullptr);
    assert(cache.states.empty());

    prompt.tokens = server_tokens(make_tokens(64, 1), false);
    assert(cache.alloc(prompt, 1024, 0) != nullptr);
    assert(cache.states.size() == 1);

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

        clear_cache(cache);
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
        { "write-behind",       test_write_behind       },
        { "write-behind-cancel", test_write_behind_cancel },
        { "min-tokens",         test_min_tokens         },
        { "rewrite-step",       test_rewrite_step       },
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
