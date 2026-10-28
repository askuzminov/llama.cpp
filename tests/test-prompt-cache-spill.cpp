// Round-trip coverage for the on-disk prompt-cache spill files (--cache-spill-dir).
//
// The spill path bypasses the OS file cache, so every transfer is block-aligned and the logical
// payload sizes live in the file header. These tests drive server_prompt_cache directly - no model
// is involved - over payload sizes that are deliberately awkward for an aligned layout (empty,
// one byte, exactly one block, one below a block, one block past the direct-transfer threshold),
// plus the cold-start adoption, signature mismatch, legacy names and truncation paths.

#include "server-task.h"
#include "fork/server-context-fork.h"
#include "common.h"
#include "file-lock.h"

#include <chrono>
#include <filesystem>
#include <fstream>
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
    for (const auto & entry : std::filesystem::directory_iterator(fs_path(dir))) {
        if (entry.path().filename() != ".prompt-cache.lock") {
            return false;
        }
    }
    return true;
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

// fork: the RAM limit counts the RAM the server holds outside the cache (ram_outside: the states of slots paused for
// room). the cache gives way: with no disk write while one may not happen (a decode would wait for it), the dirty
// states then wait above the limit; with one when it may; and with none for the states write_behind put on disk
static void test_ram_outside(const std::string & dir) {
    server_prompt_cache cache(1, 0, 0, dir, 0, SIG);
    cache.states.push_back(make_state(16, 100, 300*1024, 0));
    cache.states.push_back(make_state(16, 200, 300*1024, 0));

    size_t outside = 0;
    cache.ram_outside = [&]() { return outside; };

    cache.update(true);
    assert(cache.states.size() == 2 && dir_empty(dir));

    // 600 KiB of the cache and 600 KiB outside it: over the limit of 1 MiB
    outside = 600*1024;
    cache.update(false);
    assert(cache.states.size() == 2 && dir_empty(dir));
    for (const auto & st : cache.states) {
        assert(!st.on_disk && st.data.size() > 0);
    }

    cache.update(true);
    assert(cache.states.front().on_disk && cache.states.front().data.size() == 0);
    assert(cache.size() + outside <= cache.limit_size);

    // the rest written out while idle: update(false) frees its RAM, with no write
    outside = 1024*1024;
    assert(cache.write_behind(nullptr) == 1);
    cache.update(false);
    for (const auto & st : cache.states) {
        assert(st.on_disk && st.data.size() == 0);
    }

    clear_cache(cache);
    assert(dir_empty(dir));
}

// fork: alloc() makes room for the new state under the same limit, ram_outside counted: with no disk write while the
// writes are deferred (the new state then comes above the limit), and with one when they are not
static void test_ram_outside_alloc(const std::string & dir) {
    server_prompt_cache cache(1, 0, 0, dir, 0, SIG);
    cache.states.push_back(make_state(16, 100, 300*1024, 0));

    const size_t outside = 600*1024;
    cache.ram_outside = [&]() { return outside; };

    server_prompt prompt;

    // 300 KiB of the cache, 600 KiB outside it and 300 KiB new: over the limit of 1 MiB
    cache.defer_writes = true;
    prompt.tokens = server_tokens(make_tokens(16, 200), false);
    assert(cache.alloc(prompt, 300*1024, 0) != nullptr);
    assert(cache.states.size() == 2 && dir_empty(dir));

    cache.defer_writes = false;
    prompt.tokens = server_tokens(make_tokens(16, 300), false);
    assert(cache.alloc(prompt, 300*1024, 0) != nullptr);
    assert(cache.states.size() == 3);
    assert(cache.size() + outside <= cache.limit_size);

    clear_cache(cache);
    assert(dir_empty(dir));
}

// fork: host RAM below the reserve while a decode may not wait for the disk (may_write = false): the cache drops what
// it would have written, and says whether that covered the deficit
static void test_reserve_no_write(const std::string & dir) {
    if (common_host_mem_available() == 0) {
        return; // the reserve does nothing where the free host RAM is unknown
    }

    // a reserve no host has
    server_prompt_cache cache(0, 0, SIZE_MAX / 4, dir, 0, SIG);
    cache.states.push_back(make_state(16, 100, 4096, 0));
    cache.states.push_back(make_state(16, 200, 4096, 0));

    assert(!cache.evict_for_reserve(0, false));
    assert(cache.states.empty());
    assert(dir_empty(dir));
    server_prompt prompt;
    prompt.tokens = server_tokens(make_tokens(16, 300), false);
    assert(cache.alloc(prompt, 4096, 0) == nullptr);
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

// a run of the same model with another KV layout cannot restore the files, so it removes them.
// A run of another model spilling into the same directory leaves them alone
static void test_signature_mismatch(const std::string & dir) {
    const uint64_t owner_a = 0x1111111111111111ull;
    const uint64_t owner_b = 0x2222222222222222ull;

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG, owner_a);
        fill(cache);
    }
    assert(!dir_empty(dir));

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG, owner_b);
        assert(cache.states.empty());
    }
    assert(!dir_empty(dir));

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG ^ 1ull, owner_a);
        assert(cache.states.empty());
    }
    assert(dir_empty(dir));
}

// files named before the owner was added: this configuration's ones are renamed and kept, the
// others cannot be told apart by model any more and are removed
static void test_legacy_names(const std::string & dir) {
    const uint64_t owner = 0x3333333333333333ull;

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG, owner);
        fill(cache);
    }

    // the uids are given when the files are written, so take the names from the directory
    const std::filesystem::path root = fs_path(dir);
    const std::string           pre  = "state-3333333333333333-";

    std::vector<std::filesystem::path> files;
    for (const auto & entry : std::filesystem::directory_iterator(root)) {
        if (entry.path().filename() != ".prompt-cache.lock") {
            files.push_back(entry.path());
        }
    }
    assert(files.size() == sizeof(CASES)/sizeof(CASES[0]));
    for (const auto & path : files) {
        const std::string n = fs_utf8(path.filename());
        assert(n.compare(0, pre.size(), pre) == 0);
        std::filesystem::rename(path, root / fs_path("state-" + n.substr(pre.size())));
    }

    char name[96];

    // through the path, not fopen of its UTF-8 bytes: Windows reads a narrow name in the ANSI codepage
    const auto touch = [&](const char * n) {
        std::ofstream f(root / n, std::ios::binary);
        assert(f);
        f << "x";
    };

    snprintf(name, sizeof(name), "state-%016llx-7.bin", (unsigned long long) (SIG ^ 1ull));
    const std::string legacy_other = name;
    snprintf(name, sizeof(name), "state-%016llx-8.bin.tmp", (unsigned long long) SIG);
    const std::string legacy_tmp = name;
    snprintf(name, sizeof(name), "state-4444444444444444-%016llx-9.bin", (unsigned long long) SIG);
    const std::string other_owner = name;

    touch(legacy_other.c_str());
    touch(legacy_tmp.c_str());
    touch(other_owner.c_str());
    touch("notes.txt");

    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG, owner);
        assert(cache.states.size() == files.size());
        for (auto & st : cache.states) {
            assert(std::filesystem::exists(cache.spill_path(st.uid)));
            assert(cache.unspill_state(st));
        }
        clear_cache(cache);
    }

    assert(!std::filesystem::exists(root / legacy_other));
    assert(!std::filesystem::exists(root / legacy_tmp));
    assert( std::filesystem::exists(root / other_owner));
    assert( std::filesystem::exists(root / "notes.txt"));

    std::filesystem::remove(root / other_owner);
    std::filesystem::remove(root / "notes.txt");
    assert(dir_empty(dir));
}

// a short file must be rejected before anything is allocated from its header
static void test_truncated(const std::string & dir) {
    {
        server_prompt_cache cache(0, 0, 0, dir, 0, SIG);
        cache.states.push_back(make_state(8, 600, 100000, 0));
    }

    for (const auto & entry : std::filesystem::directory_iterator(fs_path(dir))) {
        if (entry.path().filename() == ".prompt-cache.lock") {
            continue;
        }
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

static void test_shared_budget(const std::string & dir) {
    server_prompt_cache a(0, 0, 0, dir, 60000, SIG, 1);
    server_prompt_cache b(0, 0, 0, dir, 60000, SIG ^ 1, 2);
    a.states.push_back(make_state(100, 1, 40960, 0));
    const auto ref = a.states.front().data;
    assert(a.write_behind(nullptr) == 1);
    const auto old = a.spill_path(a.states.front().uid);
    auto orphan = old;
    orphan += ".tmp";
    std::filesystem::copy_file(old, orphan);
    std::filesystem::last_write_time(old, std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
    const auto unrelated = fs_path(dir) / "notes.bin";
    { std::ofstream f(unrelated); f << "keep"; }
    b.states.push_back(make_state(100, 2, 40960, 0));
    assert(b.write_behind(nullptr) == 1);
    assert(a.disk_size() <= 60000 && b.disk_size() == a.disk_size());
    assert(!std::filesystem::exists(old));
    assert(!std::filesystem::exists(orphan));
    assert(std::filesystem::exists(unrelated));
    // Eviction by another model must not discard the remaining RAM copy.
    a.evict_state(a.states.front());
    assert(!a.states.front().on_disk && a.states.front().data.main == ref.main);
    assert(a.unspill_state(a.states.front()));
    b.evict_state(b.states.front());
    const auto b_path = b.spill_path(b.states.front().uid);
    std::filesystem::last_write_time(b_path, std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
    assert(a.write_behind(nullptr) == 1);
    assert(!std::filesystem::exists(b_path));
    assert(!b.unspill_state(b.states.front()));
    b.enforce_disk_limit();
    assert(b.states.empty());
    clear_cache(a);
    std::filesystem::remove(unrelated);
    assert(dir_empty(dir));
}

static void test_shared_writers(const std::string & dir) {
    server_prompt_cache a(0, 0, 0, dir, 0, SIG, 1);
    server_prompt_cache b(0, 0, 0, dir, 0, SIG, 1);
    a.states.push_back(make_state(100, 1, 40960, 0));
    b.states.push_back(make_state(100, 2, 40960, 0));
    {
        common_file_lock lock(fs_path(dir) / ".prompt-cache.lock");
        assert(lock.acquire());
        assert(a.write_behind(nullptr) == 0);
        assert(!a.states.front().on_disk);
    }
    assert(a.write_behind(nullptr) == 1);
    assert(b.write_behind(nullptr) == 1);
    assert(a.states.front().uid != b.states.front().uid);
    a.evict_state(a.states.front());
    assert(a.unspill_state(a.states.front()));
    assert(a.states.front().data.main == make_blob(40960, 2));
    clear_cache(a);
    clear_cache(b);
    assert(dir_empty(dir));
}

static void test_shared_lru(const std::string & dir) {
    server_prompt_cache a(0, 0, 0, dir, 100000, SIG, 1);
    server_prompt_cache b(0, 0, 0, dir, 100000, SIG ^ 1, 2);
    a.states.push_back(make_state(100, 1, 40960, 0));
    b.states.push_back(make_state(100, 2, 40960, 0));
    assert(a.write_behind(nullptr) == 1);
    assert(b.write_behind(nullptr) == 1);
    const auto old_a = a.spill_path(a.states.front().uid);
    const auto old_b = b.spill_path(b.states.front().uid);
    const auto now = std::filesystem::file_time_type::clock::now();
    std::filesystem::last_write_time(old_a, now - std::chrono::hours(2));
    std::filesystem::last_write_time(old_b, now - std::chrono::hours(1));
    assert(a.unspill_state(a.states.front()));
    b.states.push_back(make_state(100, 3, 40960, 0));
    assert(b.write_behind(nullptr) == 1);
    assert(std::filesystem::exists(old_a));
    assert(!std::filesystem::exists(old_b));
    assert(a.disk_size() <= 100000);
    clear_cache(a);
    clear_cache(b);
    assert(dir_empty(dir));
}

static void test_no_rewrite_after_eviction(const std::string & dir) {
    server_prompt_cache a(0, 0, 0, dir, 60000, SIG, 1);
    server_prompt_cache b(0, 0, 0, dir, 60000, SIG ^ 1, 2);
    a.states.push_back(make_state(100, 1, 40960, 0));
    b.states.push_back(make_state(100, 2, 40960, 0));
    assert(a.write_behind(nullptr) == 1);
    const auto old = a.spill_path(a.states.front().uid);
    std::filesystem::last_write_time(old, std::filesystem::file_time_type::clock::now() - std::chrono::hours(1));
    assert(b.write_behind(nullptr) == 1);
    const auto keep = b.spill_path(b.states.front().uid);
    for (int i = 0; i < 4; ++i) {
        assert(a.write_behind(nullptr) == 0);
        assert(b.write_behind(nullptr) == 0);
        assert(std::filesystem::exists(keep));
    }
    a.persist();
    assert(std::filesystem::exists(keep));
    assert(!a.states.front().data.main.empty());
    // Reusing the RAM copy makes it eligible for persistence again.
    assert(a.unspill_state(a.states.front()));
    assert(a.write_behind(nullptr) == 1);
    clear_cache(a);
    clear_cache(b);
    assert(dir_empty(dir));
}

static void test_ram_limit_spill_failure(const std::string & dir) {
    server_prompt_cache cache(1, 262144, 0, dir, 65536, SIG);
    cache.defer_writes = true;
    for (int base : {100, 200}) {
        server_prompt prompt;
        prompt.tokens = server_tokens(make_tokens(100, base), false);
        assert(cache.alloc(prompt, 700*1024, 0));
    }
    assert(cache.size() > cache.limit_size);
    cache.defer_writes = false;
    cache.update(true);
    assert(cache.size() <= cache.limit_size);
    assert(cache.states.size() == 1);
    assert(cache.states.front().prompt.tokens[0] == 200);
    clear_cache(cache);

    // A busy directory is also a spill failure, even when the file fits its budget.
    cache.spill_limit = 0;
    cache.defer_writes = true;
    for (int base : {300, 400}) {
        server_prompt prompt;
        prompt.tokens = server_tokens(make_tokens(100, base), false);
        assert(cache.alloc(prompt, 700*1024, 0));
    }
    cache.defer_writes = false;
    {
        common_file_lock lock(fs_path(dir) / ".prompt-cache.lock");
        assert(lock.acquire());
        cache.update(true);
        assert(cache.size() <= cache.limit_size);
    }
    assert(cache.states.size() == 1);
    assert(cache.states.front().prompt.tokens[0] == 400);
    clear_cache(cache);
    cache.spill_limit = 65536;
    for (int base : {500, 600}) {
        server_prompt prompt;
        prompt.tokens = server_tokens(make_tokens(100, base), false);
        assert(cache.alloc(prompt, 700*1024, 0));
        assert(cache.size() <= cache.limit_size);
    }
    assert(cache.states.size() == 1);
    assert(cache.states.front().prompt.tokens[0] == 600);
    clear_cache(cache);
    cache.defer_writes = true;
    for (int base : {700, 800}) {
        server_prompt prompt;
        prompt.tokens = server_tokens(make_tokens(100, base), false);
        assert(cache.alloc(prompt, 700*1024, 0));
    }
    cache.defer_writes = false;
    assert(cache.write_behind(nullptr) == 0);
    assert(cache.size() <= cache.limit_size);
    assert(cache.states.size() == 1);
    assert(cache.states.front().prompt.tokens[0] == 800);
    clear_cache(cache);
    assert(dir_empty(dir));
}

static void test_cache_compatibility(const char * model_path) {
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model_ptr model(llama_model_load_from_file(model_path, mp));
    assert(model);
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = cp.n_ubatch = 32;
    cp.n_threads = cp.n_threads_batch = 2;
    llama_context_ptr ctx(llama_init_from_model(model.get(), cp));
    assert(ctx);
    common_params params;
    params.model.path = model_path;
    const auto signature = [&](const common_params & p) { return server_prompt_cache_signature(p, model.get(), ctx.get(), nullptr); };
    const uint64_t base = signature(params);
    auto changed = params;
    changed.rope_freq_scale = 0.5f;
    assert(signature(changed) != base);
    changed = params;
    changed.yarn_attn_factor = 1.0f;
    const auto yarn = signature(changed);
    changed.yarn_attn_factor = 1.0000001f;
    assert(signature(changed) != yarn);
    const auto file = std::filesystem::temp_directory_path() / "llama-cache-identity-test.bin";
    { std::ofstream f(file); f << "weights"; }
    changed.model.path = fs_utf8(file);
    const auto original_time = std::filesystem::last_write_time(file);
    const auto first = signature(changed);
    std::filesystem::last_write_time(file, original_time - std::chrono::seconds(10));
    assert(signature(changed) != first);
    auto draft_signature = [&](const common_params & p) { return server_prompt_cache_signature(p, model.get(), ctx.get(), ctx.get()); };
    changed.speculative.draft.mparams.path = changed.model.path;
    const auto draft = draft_signature(changed);
    changed.speculative.draft.cache_type_k = GGML_TYPE_Q8_0;
    assert(draft_signature(changed) != draft);
    changed = params;
    changed.mmproj.path = fs_utf8(file);
    assert(signature(changed) != base);
    const auto media = signature(changed);
    changed.image_max_tokens = 4096;
    assert(signature(changed) != media);
    std::filesystem::remove(file);
    ctx.reset();
    model.reset();
    llama_backend_free();
}

int main(int argc, char ** argv) {
    if (argc == 3 && std::string(argv[1]) == "-m") {
        test_cache_compatibility(argv[2]);
        return 0;
    }
    // write_behind times itself: on Windows ggml_time_us divides by the counter frequency that ggml_time_init
    // sets, and no model load does it here (0xc0000094 on the 395, 22.10)
    ggml_time_init();

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
        { "ram-outside",        test_ram_outside        },
        { "ram-outside-alloc",  test_ram_outside_alloc  },
        { "reserve-no-write",   test_reserve_no_write   },
        { "rewrite-step",       test_rewrite_step       },
        { "cold-start",         test_cold_start         },
        { "signature-mismatch", test_signature_mismatch },
        { "legacy-names",       test_legacy_names       },
        { "truncated",          test_truncated          },
        { "shared-budget",      test_shared_budget      },
        { "shared-writers",     test_shared_writers     },
        { "shared-lru",         test_shared_lru         },
        { "no-rewrite-evicted", test_no_rewrite_after_eviction },
        { "ram-spill-failure",  test_ram_limit_spill_failure },
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
