// fork: server_prompt_cache members for the disk spill and the host-RAM reserve

#include "server-task.h"

#include "common.h"
#include "file-lock.h"
#include "llama.h"
#include "server-common.h"

#include "hash/xxhash/xxhash.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#   include <malloc.h>
#else
#   include <cerrno>
#   include <fcntl.h>
#   include <sys/stat.h>
#   include <unistd.h>
#endif

//
// server_prompt_cache
//
// on-disk layout of a spilled prompt-cache state: header, then the serialized prompt, then the two
// state blobs, then the context checkpoints (an index of fixed records, then their blobs).
// Self-describing so that a spill dir can be picked up again after a restart.
namespace {

struct spill_header {
    char     magic[8];
    uint32_t version;
    uint32_t n_used;
    uint64_t uid;
    uint64_t signature;
    uint64_t n_prompt; // server_tokens::serialize() output, in llama_token units
    uint64_t n_main;
    uint64_t n_drft;
    uint64_t n_ckpt;       // context checkpoints, i.e. spill_ckpt records in the index
    uint64_t n_ckpt_bytes; // their blobs, padding included
    uint64_t sum;          // XXH3-64 of the prompt, main and drft sections, without the padding (spill_sum)
};

static_assert(sizeof(spill_header) == 80, "unexpected spill_header layout");

// one context checkpoint in the index; its blobs follow in the same order, each padded
struct spill_ckpt {
    int64_t  n_tokens;
    int32_t  id_task;
    int32_t  pos_min;
    int32_t  pos_max;
    int32_t  base_pos;
    uint64_t n_tgt;
    uint64_t n_dft;
    uint64_t n_spec;
    uint64_t sum; // XXH3-64 of the fields above and of the tgt, dft and spec blobs (spill_ckpt_sum)
};

static_assert(sizeof(spill_ckpt) == 56, "unexpected spill_ckpt layout");

constexpr char     SPILL_MAGIC[8] = { 'L', 'C', 'P', 'C', 'A', 'C', 'H', 'E' };
// 5 (11.11): the sums of the sections, a damaged file (the disk, a copy, the network of a pool) is not loaded as a state
constexpr uint32_t SPILL_VERSION  = 5;

// the sums are per section, so that the export of a pool, which sends the header with a part of the checkpoints, keeps
// the sums of the file. XXH3 over the bytes a state holds: 12.5 GB/s on one core of the i9-9980HK (the vendored build is
// SSE2), the disk gives 2-7 GB/s. a write sums first (it runs in the background), a read sums each piece while it reads
// the next one (spill_file::read_summed)
struct spill_sum {
    XXH3_state_t * state = XXH3_createState();

    spill_sum() {
        GGML_ASSERT(state != nullptr);
        XXH3_64bits_reset(state);
    }
    ~spill_sum() {
        XXH3_freeState(state);
    }
    spill_sum(const spill_sum &) = delete;
    spill_sum & operator=(const spill_sum &) = delete;

    spill_sum & add(const void * data, size_t n) {
        if (n > 0) {
            XXH3_64bits_update(state, data, n);
        }
        return *this;
    }
    uint64_t get() const {
        return XXH3_64bits_digest(state);
    }
};

uint64_t spill_head_sum(const void * prompt, size_t n_prompt, const void * main, size_t n_main, const void * drft, size_t n_drft) {
    return spill_sum().add(prompt, n_prompt).add(main, n_main).add(drft, n_drft).get();
}

uint64_t spill_ckpt_sum(const spill_ckpt & e, const void * tgt, const void * dft, const void * spec) {
    return spill_sum().add(&e, offsetof(spill_ckpt, sum)).add(tgt, e.n_tgt).add(dft, e.n_dft).add(spec, e.n_spec).get();
}

spill_ckpt spill_ckpt_of(const common_prompt_checkpoint & cp) {
    spill_ckpt e = {
        cp.n_tokens, cp.id_task, cp.pos_min, cp.pos_max, cp.base_pos,
        cp.data_tgt.size(), cp.data_dft.size(), cp.data_spec.size(), 0,
    };
    e.sum = spill_ckpt_sum(e, cp.data_tgt.data(), cp.data_dft.data(), cp.data_spec.data());
    return e;
}

// Spill files bypass the OS file cache. A spilled state is written once and read back at most
// once, so caching it buys nothing and costs memory - on a host sized for a large model those
// pages compete with the model itself, and on Windows they are reported as available memory,
// which feeds straight back into the context-checkpoint budget.
//
// Bypassing the cache constrains the layout: file offset, transfer length and buffer address must
// all be multiples of the device block size. 4096 covers both 512e and 4Kn media. Every section is
// padded up to it, so the file is larger than its payload and the logical sizes live in the header.
constexpr uint64_t SPILL_ALIGN   = SERVER_STATE_ALIGN;
constexpr size_t   SPILL_CHUNK   = 4*1024*1024;  // bounce buffer, for sections that are not aligned
constexpr size_t   SPILL_REQUEST = 16*1024*1024; // direct transfer size, same as the model loader

constexpr uint64_t spill_align_up(uint64_t n) {
    return (n + SPILL_ALIGN - 1) & ~(SPILL_ALIGN - 1);
}

// an aligned buffer can be handed to the device as is, with no copy in between
bool spill_is_aligned(const void * p) {
    return ((uintptr_t) p % SPILL_ALIGN) == 0;
}

// the sizes come from a file that may have been truncated or tampered with
constexpr uint64_t SPILL_MAX_BYTES = 1ull << 40; // far above any real state
constexpr uint64_t SPILL_MAX_CKPT  = 4096;       // far above any real checkpoint count

bool spill_header_sane(const spill_header & h) {
    return h.n_prompt < SPILL_MAX_BYTES/sizeof(llama_token) && h.n_main < SPILL_MAX_BYTES &&
           h.n_drft < SPILL_MAX_BYTES && h.n_ckpt < SPILL_MAX_CKPT && h.n_ckpt_bytes < SPILL_MAX_BYTES;
}

bool spill_ckpt_sane(const spill_ckpt & c) {
    return c.n_tgt < SPILL_MAX_BYTES && c.n_dft < SPILL_MAX_BYTES && c.n_spec < SPILL_MAX_BYTES;
}

// UTF-8 -> filesystem path. On Windows a narrow string is taken to be in the ANSI codepage, which
// mangles everything outside it (a Cyrillic user directory, say), so convert explicitly.
std::filesystem::path spill_fs_path(const std::string & utf8) {
#if defined(_WIN32)
    if (utf8.empty()) {
        return {};
    }

    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int) utf8.size(), nullptr, 0);
    if (n <= 0) {
        return {};
    }

    std::wstring wide((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int) utf8.size(), &wide[0], n);

    return std::filesystem::path(wide);
#else
    return std::filesystem::path(utf8);
#endif
}

// filesystem path -> UTF-8, for log lines (path::string() would go through the ANSI codepage)
std::string spill_fs_utf8(const std::filesystem::path & path) {
#if defined(_WIN32)
    const std::wstring wide = path.wstring();
    if (wide.empty()) {
        return {};
    }

    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
        return {};
    }

    std::string utf8((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), &utf8[0], n, nullptr, nullptr);

    return utf8;
#else
    return path.string();
#endif
}

// file name of a spilled state. The owner (the model file) keeps the caches of different models apart
// in a shared spill dir, the signature tells a model's current configuration from an old one
std::string spill_name_of(uint64_t owner, uint64_t signature, uint64_t uid) {
    char buf[80];
    snprintf(buf, sizeof(buf), "state-%016llx-%016llx-%llu.bin", (unsigned long long) owner, (unsigned long long) signature, (unsigned long long) uid);
    return buf;
}

enum spill_name_kind {
    SPILL_NAME_OTHER,      // another model's file, or not one of ours at all
    SPILL_NAME_MINE,
    SPILL_NAME_TMP,        // ours, from a write that did not get to the rename
    SPILL_NAME_STALE,      // this model under an old configuration: it cannot be restored
    SPILL_NAME_LEGACY,     // the name had no owner yet, this configuration
    SPILL_NAME_LEGACY_OLD, // the name had no owner yet, another configuration or model, or unfinished
};

// a state is written under its name plus this suffix and renamed when complete
constexpr char SPILL_TMP_EXT[] = ".tmp";

static bool spill_is_hex16(const std::string & s, size_t pos) {
    return s.size() >= pos + 16 && s.find_first_not_of("0123456789abcdef", pos) >= pos + 16;
}

spill_name_kind spill_name_classify(const std::string & name, uint64_t owner, uint64_t signature) {
    static const std::string pre = "state-";
    static const std::string ext = ".bin";
    static const std::string tmp = SPILL_TMP_EXT;

    if (name.size() > tmp.size() && name.compare(name.size() - tmp.size(), tmp.size(), tmp) == 0) {
        switch (spill_name_classify(name.substr(0, name.size() - tmp.size()), owner, signature)) {
            case SPILL_NAME_MINE:       return SPILL_NAME_TMP;
            case SPILL_NAME_STALE:      return SPILL_NAME_STALE;
            case SPILL_NAME_LEGACY:
            case SPILL_NAME_LEGACY_OLD: return SPILL_NAME_LEGACY_OLD;
            default:                    return SPILL_NAME_OTHER;
        }
    }

    if (name.size() <= pre.size() + ext.size() ||
        name.compare(0, pre.size(), pre) != 0 ||
        name.compare(name.size() - ext.size(), ext.size(), ext) != 0) {
        return SPILL_NAME_OTHER;
    }

    // <owner>-<signature>-<uid>, or <signature>-<uid> as written before the owner was added
    const std::string mid = name.substr(pre.size(), name.size() - pre.size() - ext.size());

    const size_t sep = mid.rfind('-');
    if (sep == std::string::npos || sep + 1 >= mid.size() ||
        mid.find_first_not_of("0123456789", sep + 1) != std::string::npos) {
        return SPILL_NAME_OTHER;
    }

    char hex[20];
    if (sep == 16 && spill_is_hex16(mid, 0)) {
        snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) signature);
        return mid.compare(0, 16, hex) == 0 ? SPILL_NAME_LEGACY : SPILL_NAME_LEGACY_OLD;
    }
    if (sep != 33 || mid[16] != '-' || !spill_is_hex16(mid, 0) || !spill_is_hex16(mid, 17)) {
        return SPILL_NAME_OTHER;
    }

    snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) owner);
    if (mid.compare(0, 16, hex) != 0) {
        return SPILL_NAME_OTHER;
    }
    snprintf(hex, sizeof(hex), "%016llx", (unsigned long long) signature);

    return mid.compare(17, 16, hex) == 0 ? SPILL_NAME_MINE : SPILL_NAME_STALE;
}

// section offsets implied by a header
struct spill_layout {
    uint64_t off_prompt;
    uint64_t off_main;
    uint64_t off_drft;
    uint64_t off_index;
    uint64_t off_ckpt;
    uint64_t size;
};

spill_layout spill_layout_of(const spill_header & h) {
    spill_layout l = {};
    l.off_prompt = SPILL_ALIGN;
    l.off_main   = l.off_prompt + spill_align_up(h.n_prompt*sizeof(llama_token));
    l.off_drft   = l.off_main   + spill_align_up(h.n_main);
    l.off_index  = l.off_drft   + spill_align_up(h.n_drft);
    l.off_ckpt   = l.off_index  + spill_align_up(h.n_ckpt*sizeof(spill_ckpt));
    l.size       = l.off_ckpt   + h.n_ckpt_bytes; // already a sum of padded sections
    return l;
}

struct spill_entry {
    std::filesystem::path path;
    std::filesystem::file_time_type modified;
    size_t bytes;
};

// Only completed cache files participate. Ignore temporary files, symlinks and unrelated files.
static std::vector<spill_entry> spill_entries(const std::filesystem::path & dir, bool remove_tmp = false) {
    std::vector<spill_entry> entries;
    std::error_code ec;
    for (std::filesystem::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const auto & entry = *it;
        if (entry.symlink_status(ec).type() != std::filesystem::file_type::regular || ec) {
            ec.clear();
            continue;
        }
        std::string name = spill_fs_utf8(entry.path().filename());
        const bool temporary = name.size() > 4 && name.compare(name.size() - 4, 4, SPILL_TMP_EXT) == 0;
        if (temporary) {
            name.resize(name.size() - 4);
        }
        if (name.compare(0, 6, "state-") != 0 || !spill_is_hex16(name, 6) || !spill_is_hex16(name, 23)) {
            continue;
        }
        const uint64_t owner = std::strtoull(name.substr(6, 16).c_str(), nullptr, 16);
        const uint64_t signature = std::strtoull(name.substr(23, 16).c_str(), nullptr, 16);
        if (spill_name_classify(name, owner, signature) != SPILL_NAME_MINE) {
            continue;
        }
        if (temporary) {
            // All writers hold the directory lock, so any remaining temporary file is abandoned.
            if (remove_tmp) {
                std::filesystem::remove(entry.path(), ec);
                ec.clear();
            }
            continue;
        }
        spill_header h = {};
        std::ifstream f(entry.path(), std::ios::binary);
        if (!f.read((char *) &h, sizeof(h)) || memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) != 0 ||
                h.version != SPILL_VERSION || !spill_header_sane(h) || h.signature != signature ||
                spill_name_of(owner, signature, h.uid) != name) {
            continue;
        }
        const auto bytes = entry.file_size(ec);
        if (ec || bytes != spill_layout_of(h).size || bytes > SIZE_MAX) {
            ec.clear();
            continue;
        }
        const auto modified = entry.last_write_time(ec);
        if (!ec) {
            entries.push_back({entry.path(), modified, (size_t) bytes});
        }
        ec.clear();
    }
    return entries;
}

// Caller holds the directory lock. A cache hit updates mtime, so eviction is global LRU.
static bool spill_prune(const std::filesystem::path & dir, size_t limit) {
    auto entries = spill_entries(dir, true);
    uint64_t used = 0;
    for (const auto & entry : entries) {
        used += entry.bytes;
    }
    std::sort(entries.begin(), entries.end(), [](const spill_entry & a, const spill_entry & b) {
        return a.modified == b.modified ? a.path < b.path : a.modified < b.modified;
    });
    for (const auto & entry : entries) {
        if (used <= limit) {
            break;
        }
        std::error_code ec;
        if (std::filesystem::remove(entry.path, ec)) {
            used -= entry.bytes;
            SRV_TRC(" - prompt-cache directory budget: removed '%s' (%.3f MiB)\n", spill_fs_utf8(entry.path).c_str(), entry.bytes / (1024.0 * 1024.0));
        }
    }
    return used <= limit;
}

// One spill file, read or written with the cache bypassed. An aligned buffer goes to the device as
// is; anything else passes through an aligned bounce buffer, so callers may hand over ordinary
// unaligned pointers and sizes. Falls back to cached I/O when the platform or filesystem refuses
// the unbuffered open - the layout stays valid either way.
class spill_file {
public:
    spill_file() = default;

    ~spill_file() {
        close();
    }

    spill_file(const spill_file &)             = delete;
    spill_file & operator=(const spill_file &) = delete;

    bool open_read (const std::filesystem::path & path) { return open_impl(path, false); }
    bool open_write(const std::filesystem::path & path) { return open_impl(path, true);  }

    // a whole state can be gigabytes, so a write is stopped between device transfers when the
    // caller says the machine is needed elsewhere. The half-written file is then thrown away
    void set_cancel(const std::function<bool()> * fn) { cancel = fn; }

    bool was_aborted() const { return aborted; }

    void close() {
#if defined(_WIN32)
        if (fh != INVALID_HANDLE_VALUE) {
            CloseHandle(fh);
            fh = INVALID_HANDLE_VALUE;
        }
        if (buf) {
            _aligned_free(buf);
            buf = nullptr;
        }
#else
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (buf) {
            free(buf);
            buf = nullptr;
        }
#endif
        used = 0;
    }

    uint64_t size() const {
#if defined(_WIN32)
        LARGE_INTEGER li;
        return GetFileSizeEx(fh, &li) ? (uint64_t) li.QuadPart : 0;
#else
        struct stat st = {};
        return fstat(fd, &st) == 0 ? (uint64_t) st.st_size : 0;
#endif
    }

    // append `n` bytes, then zero-pad up to the next SPILL_ALIGN boundary
    bool write_section(const void * src, size_t n) {
        const uint8_t * p = (const uint8_t *) src;

        // a large aligned source goes straight to the file; only the tail that is shorter than one
        // block still needs the bounce buffer, to be zero-padded. `used` is a whole number of
        // blocks here, so flushing it keeps the file offset aligned
        if (n >= SPILL_REQUEST && spill_is_aligned(p)) {
            if (used > 0 && !flush()) {
                return false;
            }

            const size_t direct = n & ~(size_t) (SPILL_ALIGN - 1);
            if (!write_raw(p, direct)) {
                return false;
            }

            p += direct;
            n -= direct;
        }

        while (n > 0) {
            const size_t k = std::min(n, SPILL_CHUNK - used);
            memcpy(buf + used, p, k);
            used += k;
            p    += k;
            n    -= k;

            if (used == SPILL_CHUNK && (check_cancel() || !flush())) {
                return false;
            }
        }

        const size_t pad = (size_t) (spill_align_up(used) - used);
        if (pad > 0) {
            memset(buf + used, 0, pad);
            used += pad;
        }

        return used < SPILL_CHUNK || flush();
    }

    // write out whatever is still buffered (always a whole number of blocks)
    bool finish() {
        return used == 0 || flush();
    }

    // put the written data and the file size on the device, past its write cache
    bool sync() {
#if defined(_WIN32)
        return FlushFileBuffers(fh) != 0;
#else
#if defined(F_FULLFSYNC)
        if (fcntl(fd, F_FULLFSYNC) == 0) {
            return true; // macOS: fsync() does not flush the drive cache
        }
#endif
        return fsync(fd) == 0;
#endif
    }

    // read `n` bytes starting at a block-aligned file offset
    bool read_at(uint64_t offset, void * dst, size_t n) {
        if (offset % SPILL_ALIGN != 0) {
            return false;
        }

        uint8_t * p = (uint8_t *) dst;

        // a large aligned destination takes the whole blocks straight from the file; the tail that
        // is shorter than one block still comes through the bounce buffer
        if (n >= SPILL_REQUEST && spill_is_aligned(p)) {
            const size_t direct = n & ~(size_t) (SPILL_ALIGN - 1);
            if (!read_raw_into(offset, p, direct)) {
                return false;
            }

            p      += direct;
            n      -= direct;
            offset += direct;
        }

        while (n > 0) {
            const size_t want = (size_t) std::min<uint64_t>(SPILL_CHUNK, spill_align_up(n));
            const size_t got  = read_raw(offset, want);
            if (got == 0) {
                return false;
            }

            // an unbuffered handle only accepts aligned offsets, so a short read is cut back to a
            // whole number of blocks - unless what came back already covers the rest of the request
            const size_t k = got >= n ? n : got - got % (size_t) SPILL_ALIGN;
            if (k == 0) {
                return false; // less than one block: no way to continue from an aligned offset
            }

            memcpy(p, buf, k);

            p      += k;
            n      -= k;
            offset += k;
        }

        return true;
    }

    // read_at with the sum of the bytes: the sum of a piece runs on another thread while the next piece is read, so the
    // check costs the sum of the last piece, not a second pass after the read
    bool read_summed(uint64_t offset, void * dst, size_t n, spill_sum & sum) {
        constexpr size_t PIECE = 4*SPILL_REQUEST; // aligned: a whole piece into an aligned buffer goes straight there
        uint8_t * p = (uint8_t *) dst;
        std::thread th;
        bool ok = true;
        for (size_t done = 0; done < n && ok; done += PIECE) {
            const size_t len = std::min(PIECE, n - done);
            ok = read_at(offset + done, p + done, len);
            if (th.joinable()) {
                th.join();
            }
            if (ok) {
                th = std::thread([&sum, q = p + done, len] { sum.add(q, len); });
            }
        }
        if (th.joinable()) {
            th.join();
        }
        return ok;
    }

private:
    bool check_cancel() {
        if (cancel && *cancel && (*cancel)()) {
            aborted = true;
        }

        return aborted;
    }

    // the fallback is a property of the volume, not of one file - say it once
    static void warn_buffered() {
        static bool once = false;
        if (!once) {
            once = true;
            SRV_WRN("%s", " - prompt-cache spill: unbuffered I/O is not available, using the OS cache\n");
        }
    }

    bool open_impl(const std::filesystem::path & path, bool write) {
        close();

        aborted = false;

#if defined(_WIN32)
        buf = (uint8_t *) _aligned_malloc(SPILL_CHUNK, (size_t) SPILL_ALIGN);
        if (!buf) {
            return false;
        }

        const std::wstring name = path.wstring();

        const DWORD access = write ? GENERIC_WRITE : GENERIC_READ;
        const DWORD share  = write ? 0 : (FILE_SHARE_READ | FILE_SHARE_DELETE);
        const DWORD disp   = write ? CREATE_ALWAYS : OPEN_EXISTING;

        fh = CreateFileW(name.c_str(), access, share, nullptr, disp,
                FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (fh == INVALID_HANDLE_VALUE) {
            // the volume may not support unbuffered access
            fh = CreateFileW(name.c_str(), access, share, nullptr, disp, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (fh != INVALID_HANDLE_VALUE) {
                warn_buffered();
            }
        }
        if (fh == INVALID_HANDLE_VALUE) {
            close();
            return false;
        }
#else
        const std::string name = path.string();
        if (posix_memalign((void **) &buf, (size_t) SPILL_ALIGN, SPILL_CHUNK) != 0) {
            buf = nullptr;
            return false;
        }

        const int flags = write ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;

#if defined(O_DIRECT)
        fd = ::open(name.c_str(), flags | O_DIRECT, 0644);
        if (fd < 0) {
            fd = ::open(name.c_str(), flags, 0644); // filesystem without O_DIRECT support
            if (fd >= 0) {
                warn_buffered();
            }
        }
#else
        fd = ::open(name.c_str(), flags, 0644);
#endif
        if (fd < 0) {
            close();
            return false;
        }

#if defined(F_NOCACHE)
        fcntl(fd, F_NOCACHE, 1); // macOS has no O_DIRECT; this is the equivalent
#endif
#endif
        used = 0;
        return true;
    }

    bool flush() {
        const size_t n = used;
        used = 0;

#if defined(_WIN32)
        DWORD written = 0;
        return WriteFile(fh, buf, (DWORD) n, &written, nullptr) && written == n;
#else
        size_t off = 0;
        while (off < n) {
            const ssize_t k = ::write(fd, buf + off, n - off);
            if (k <= 0) {
                if (k < 0 && errno == EINTR) {
                    continue;
                }
                return false;
            }
            off += (size_t) k;
        }
        return true;
#endif
    }

    // write `n` bytes from the caller's buffer at the current offset; `n` is a whole number of
    // blocks and the buffer is aligned, so the bounce buffer is not needed
    bool write_raw(const void * src, size_t n) {
        const uint8_t * p = (const uint8_t *) src;

        while (n > 0) {
            if (check_cancel()) {
                return false;
            }

            const size_t want = std::min<size_t>(n, SPILL_REQUEST);

#if defined(_WIN32)
            DWORD k = 0;
            if (!WriteFile(fh, p, (DWORD) want, &k, nullptr)) {
                return false;
            }
#else
            const ssize_t r = ::write(fd, p, want);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            const size_t k = (size_t) r;
#endif
            // an unbuffered handle can only go on from a block boundary
            if (k == 0 || (k != n && k % SPILL_ALIGN != 0)) {
                return false;
            }

            p += k;
            n -= k;
        }

        return true;
    }

    // read `len` bytes into the caller's buffer; offset, length and buffer are all aligned
    bool read_raw_into(uint64_t offset, void * dst, size_t len) {
        uint8_t * p = (uint8_t *) dst;

#if defined(_WIN32)
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG) offset;
        if (!SetFilePointerEx(fh, li, nullptr, FILE_BEGIN)) {
            return false;
        }
#endif

        while (len > 0) {
            const size_t want = std::min<size_t>(len, SPILL_REQUEST);

#if defined(_WIN32)
            DWORD k = 0;
            if (!ReadFile(fh, p, (DWORD) want, &k, nullptr)) {
                return false;
            }
#else
            const ssize_t r = ::pread(fd, p, want, (off_t) offset);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            const size_t k = (size_t) r;
#endif
            if (k == 0 || (k != len && k % SPILL_ALIGN != 0)) {
                return false;
            }

            p      += k;
            len    -= k;
            offset += k;
        }

        return true;
    }

    size_t read_raw(uint64_t offset, size_t len) {
#if defined(_WIN32)
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG) offset;
        if (!SetFilePointerEx(fh, li, nullptr, FILE_BEGIN)) {
            return 0;
        }
        DWORD got = 0;
        if (!ReadFile(fh, buf, (DWORD) len, &got, nullptr)) {
            return 0;
        }
        return (size_t) got;
#else
        const ssize_t got = ::pread(fd, buf, len, (off_t) offset);
        return got > 0 ? (size_t) got : 0;
#endif
    }

#if defined(_WIN32)
    HANDLE fh = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif

    uint8_t * buf  = nullptr; // aligned bounce buffer, SPILL_CHUNK bytes
    size_t    used = 0;       // bytes buffered on the write path

    const std::function<bool()> * cancel = nullptr; // polled between device transfers, may be null
    bool aborted = false;                           // the cancel callback stopped this write
};

} // namespace

std::filesystem::path server_fs_path_fork(const std::string & utf8) {
    return spill_fs_path(utf8);
}

server_prompt_cache::server_prompt_cache(size_t limit_size_mib, size_t limit_tokens, size_t reserve_bytes,
                                         const std::string & spill_dir, size_t spill_limit_bytes,
                                         uint64_t signature, uint64_t owner, bool has_mtmd) {
    this->limit_size   = 1024ull*1024ull*limit_size_mib;
    this->limit_tokens = limit_tokens;
    this->reserve_size = reserve_bytes;
    this->spill_dir    = spill_dir;
    this->spill_limit  = spill_limit_bytes;
    this->signature    = signature;
    this->owner        = owner;
    this->has_mtmd     = has_mtmd;
    next_uid = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

    if (!this->spill_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(spill_dir_path(), ec);
        if (ec) {
            SRV_WRN(" - could not create prompt-cache spill dir '%s' (%s) - disabling spill\n",
                    this->spill_dir.c_str(), ec.message().c_str());
            this->spill_dir.clear();
        } else {
            const size_t n = restore_spilled();
            if (n > 0) {
                SRV_INF(" - adopted %zu prompt-cache state(s) from '%s' (%.3f MiB on disk)\n",
                        n, this->spill_dir.c_str(), disk_size() / (1024.0 * 1024.0));
            }
        }
    }
}

server_prompt_cache::~server_prompt_cache() {
    // a destructor must not propagate: a failed persist only costs the spilled states
    try {
        persist();
    } catch (const std::exception & e) {
        SRV_ERR(" - failed to persist the prompt cache: %s\n", e.what());
    } catch (...) {
        SRV_ERR("%s", " - failed to persist the prompt cache\n");
    }
}

std::filesystem::path server_prompt_cache::spill_dir_path() const {
    return spill_fs_path(spill_dir);
}

std::filesystem::path server_prompt_cache::spill_path(uint64_t uid) const {
    return spill_dir_path() / spill_name_of(owner, signature, uid);
}

size_t server_prompt_cache::restore_spilled() {
    common_file_lock lock(spill_dir_path() / ".prompt-cache.lock");
    if (!lock.acquire([] { return false; })) {
        return 0;
    }
    std::error_code ec;
    std::filesystem::directory_iterator it(spill_dir_path(), ec);
    if (ec) {
        return 0;
    }

    // adopt in a stable order so that the list stays oldest-first by uid
    std::vector<server_prompt_cache_state> found;

    // list first: a renamed legacy file must not show up again later in the same walk
    std::vector<std::filesystem::path> paths;
    for (const auto & entry : it) {
        if (entry.is_regular_file(ec) && !ec) {
            paths.push_back(entry.path());
        }
    }

    size_t n_stale = 0;
    size_t n_stale_bytes = 0;

    for (std::filesystem::path path : paths) {
        const spill_name_kind kind = spill_name_classify(spill_fs_utf8(path.filename()), owner, signature);
        if (kind == SPILL_NAME_OTHER) {
            continue; // another model spills here too - leave its files alone
        }
        if (kind == SPILL_NAME_TMP) {
            SRV_WRN(" - removing unfinished prompt-cache spill file '%s'\n", spill_fs_utf8(path).c_str());
            std::filesystem::remove(path, ec);
            continue;
        }
        if (kind == SPILL_NAME_STALE || kind == SPILL_NAME_LEGACY_OLD) {
            // nothing can restore it: this model under an old configuration, or a name without owner
            // that is not this configuration's. Left in place, it would take disk space forever
            const uintmax_t n = std::filesystem::file_size(path, ec);
            if (std::filesystem::remove(path, ec)) {
                n_stale++;
                n_stale_bytes += ec ? 0 : (size_t) n;
            }
            continue;
        }
        if (kind == SPILL_NAME_LEGACY) {
            // this configuration's state under the old name: give it the owner and keep it
            spill_file f;
            spill_header h = {};
            const bool ok = f.open_read(path) && f.read_at(0, &h, sizeof(h)) && memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) == 0;
            f.close();
            const std::filesystem::path to = spill_path(h.uid);
            if (!ok || std::filesystem::exists(to, ec)) {
                std::filesystem::remove(path, ec);
                continue;
            }
            std::filesystem::rename(path, to, ec);
            if (ec) {
                std::filesystem::remove(path, ec);
                continue;
            }
            path = to;
        }

        spill_file f;
        if (!f.open_read(path)) {
            continue;
        }

        spill_header h = {};
        if (!f.read_at(0, &h, sizeof(h)) || memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) != 0) {
            // our name, but no header: an unfinished write of a build that wrote in place
            SRV_WRN(" - removing unfinished prompt-cache spill file '%s'\n", spill_fs_utf8(path).c_str());
            f.close();
            std::filesystem::remove(path, ec);
            continue;
        }
        if (h.version != SPILL_VERSION) {
            // our file, written by a build with another layout: the state is gone
            f.close();
            std::filesystem::remove(path, ec);
            continue;
        }

        // the header disagrees with the name, or the file is truncated - the state cannot be restored. a uid of the
        // header that is not the one of the name (a copied or edited directory) would make the hit of this entry
        // open, and drop, the file of that other uid: spill_entries binds the two the same way
        const spill_layout lay = spill_header_sane(h) ? spill_layout_of(h) : spill_layout{};
        if (h.signature != signature || lay.size == 0 || lay.size != f.size() ||
                spill_name_of(owner, signature, h.uid) != spill_fs_utf8(path.filename())) {
            SRV_WRN(" - removing broken prompt-cache spill file '%s'\n", spill_fs_utf8(path).c_str());
            f.close();
            std::filesystem::remove(path, ec);
            continue;
        }

        llama_tokens packed(h.n_prompt);
        if (h.n_prompt && !f.read_at(lay.off_prompt, packed.data(), h.n_prompt*sizeof(llama_token))) {
            f.close();
            std::filesystem::remove(path, ec);
            continue;
        }
        f.close();

        server_prompt_cache_state st;
        try {
            st.prompt.tokens = server_tokens::deserialize(packed, has_mtmd);
        } catch (const std::exception & e) {
            // the media chunks cannot be rebuilt: the server runs without an mmproj now, or the
            // chunk format moved on. Either way the state is unusable
            SRV_WRN(" - discarding prompt-cache spill file '%s': %s\n", spill_fs_utf8(path).c_str(), e.what());
            std::filesystem::remove(path, ec);
            continue;
        }
        st.uid           = h.uid;
        st.prompt.n_used = h.n_used;
        st.on_disk       = true;
        st.ckpt_on_disk  = h.n_ckpt;
        st.disk_bytes    = lay.size;

        next_uid = std::max(next_uid, h.uid + 1);

        found.push_back(std::move(st));
    }

    std::sort(found.begin(), found.end(),
            [](const server_prompt_cache_state & a, const server_prompt_cache_state & b) { return a.uid < b.uid; });

    for (auto & st : found) {
        states.push_back(std::move(st));
    }

    if (n_stale > 0) {
        SRV_INF(" - removed %zu prompt-cache spill file(s) that cannot be restored (%.3f MiB)\n",
                n_stale, n_stale_bytes / (1024.0 * 1024.0));
    }

    lock.close();
    enforce_disk_limit();

    return found.size();
}

void server_prompt_cache::persist() {
    if (spill_dir.empty()) {
        // no spill dir: nothing was written, nothing to clean up
        return;
    }

    for (auto & st : states) {
        if (spill_exists(st) || !st.disk_evicted) {
            spill_state(st);
        }
    }

    enforce_disk_limit();

    size_t n_disk = 0;
    for (const auto & st : states) {
        n_disk += st.on_disk ? 1 : 0;
    }

    SRV_INF(" - prompt cache: %zu of %zu states kept on disk, %.3f MiB\n",
            n_disk, states.size(), disk_size() / (1024.0 * 1024.0));
}

// how many leading tokens of `st` an entry on disk gives back: the prefix they share, when a slot would restore that
// entry for it (find_best: the entry keeps at least a quarter of itself). Writing `st` out buys only the tokens past
// that point. Not only an entry that is a whole prefix of `st`: the state of a chat ends with the answer it generated,
// which the next request renders again, often otherwise (a template that drops the reasoning of the last answer), and
// a hybrid model goes back to the checkpoint of the entry at the end of the prompt before that answer
static size_t spill_prefix_on_disk(const std::list<server_prompt_cache_state> & states,
        const server_prompt_cache_state & st) {
    size_t res = 0;

    for (const auto & other : states) {
        if (&other == &st || !other.on_disk) {
            continue;
        }

        const size_t len = other.prompt.tokens.get_common_prefix(st.prompt.tokens);
        if (len > 0 && 4*len >= other.prompt.tokens.size()) {
            res = std::max(res, len);
        }
    }

    return res;
}

// the smallest piece of work worth a disk write (--cache-min-tokens): a shorter state stays in RAM only, and one whose
// front is on disk already waits until it grew that much past it - a rewrite moves gigabytes to win back only the tokens
// past that point. What leaving RAM meanwhile costs is that much recompute, not the whole prompt
bool server_prompt_cache::below_step(const server_prompt_cache_state & st) const {
    if (min_tokens == 0) {
        return false;
    }
    const size_t n_tok  = st.prompt.tokens.size();
    const size_t n_have = spill_prefix_on_disk(states, st);
    return n_tok < min_tokens || (n_have > 0 && n_tok - n_have < min_tokens);
}

bool server_prompt_cache::write_state(server_prompt_cache_state & st, const std::function<bool()> * cancel, size_t disk_free) {
    if (spill_dir.empty() || spill_exists(st)) {
        return false;
    }
    if (st.data.main.empty() && st.data.drft.empty()) {
        return false; // nothing big to write out
    }
    if (below_step(st)) {
        SRV_TRC(" - a state of %d tokens is below the %zu token step, not written\n", st.prompt.n_tokens(), min_tokens);
        return false;
    }
    common_file_lock lock(spill_dir_path() / ".prompt-cache.lock");
    // Skip a busy directory instead of delaying a request behind another model's write.
    if (!lock.acquire([] { return false; }) || (cancel && *cancel && (*cancel)())) {
        return false;
    }
    if (spill_limit > 0) {
        disk_free = std::min(disk_free, spill_limit);
    }

    // the prompt is stored alongside the state so the file can be picked up after a restart. Media
    // chunks come with it, without their pixels: the image is already encoded into the state blob,
    // and matching the prompt again only needs the chunk id and its token count
    std::vector<char> prompt;
    try {
        prompt = st.prompt.tokens.serialize();
    } catch (const std::exception & e) {
        SRV_ERR(" - failed to serialize a prompt-cache state: %s\n", e.what());
        return false;
    }

    GGML_ASSERT(prompt.size() % sizeof(llama_token) == 0);

    spill_header h = {};
    memcpy(h.magic, SPILL_MAGIC, sizeof(h.magic));
    h.version   = SPILL_VERSION;
    h.n_used    = st.prompt.n_used;
    h.signature = signature;
    h.n_prompt  = prompt.size()/sizeof(llama_token);
    h.n_main    = st.data.main.size();
    h.n_drft    = st.data.drft.size();
    h.sum       = spill_head_sum(prompt.data(), prompt.size(), st.data.main.data(), h.n_main, st.data.drft.data(), h.n_drft);

    // the context checkpoints go out with the state: after a restart they are what lets a prompt
    // that matches only a prefix roll back, instead of being processed from the start again
    std::vector<spill_ckpt> index;
    index.reserve(st.prompt.checkpoints.size());

    for (const auto & cp_ptr : st.prompt.checkpoints) {
        const common_prompt_checkpoint & cp = *cp_ptr;
        index.push_back(spill_ckpt_of(cp));

        h.n_ckpt_bytes += spill_align_up(cp.data_tgt.size())
                        + spill_align_up(cp.data_dft.size())
                        + spill_align_up(cp.data_spec.size());
    }

    h.n_ckpt = index.size();

    // a checkpoint only saves recomputation, so when the disk budget cannot take both, the
    // checkpoints go and the state is still written out
    if (h.n_ckpt > 0 && spill_layout_of(h).size > disk_free) {
        h.n_ckpt       = 0;
        h.n_ckpt_bytes = 0;
        index.clear();
    }

    // what does not fit the disk budget is skipped rather than written and deleted again right after
    if (spill_layout_of(h).size > disk_free) {
        return false;
    }
    // Reserve room for the temporary file too; the directory lock excludes other writers.
    if (spill_limit > 0 && !spill_prune(spill_dir_path(), spill_limit - spill_layout_of(h).size)) {
        return false;
    }

    // Another server using the same model may have assigned the next UID already.
    std::error_code exists_ec;
    do {
        st.uid = next_uid++;
    } while (std::filesystem::exists(spill_path(st.uid), exists_ec) && !exists_ec);
    h.uid = st.uid;

    const std::filesystem::path path = spill_path(st.uid);

    // the final name only ever holds a complete file: a write that the process does not finish (a
    // kill, a power loss) leaves the temporary file, and the next start removes it
    std::filesystem::path tmp = path;
    tmp += SPILL_TMP_EXT;

    spill_file f;
    f.set_cancel(cancel);

    if (!f.open_write(tmp)) {
        SRV_ERR(" - failed to create prompt-cache spill file '%s'\n", spill_fs_utf8(tmp).c_str());
        return false;
    }

    // sections are written in the order spill_layout_of() describes; an empty one writes nothing
    // and contributes no padding, so the offsets still line up
    bool ok =
        f.write_section(&h, sizeof(h)) &&
        (h.n_prompt == 0 || f.write_section(prompt.data(), prompt.size())) &&
        (h.n_main   == 0 || f.write_section(st.data.main.data(), h.n_main)) &&
        (h.n_drft   == 0 || f.write_section(st.data.drft.data(), h.n_drft)) &&
        (h.n_ckpt   == 0 || f.write_section(index.data(), index.size()*sizeof(spill_ckpt)));

    if (ok && h.n_ckpt > 0) {
        for (const auto & cp_ptr : st.prompt.checkpoints) {
            const common_prompt_checkpoint & cp = *cp_ptr;
            ok = (cp.data_tgt.empty()  || f.write_section(cp.data_tgt.data(),  cp.data_tgt.size())) &&
                 (cp.data_dft.empty()  || f.write_section(cp.data_dft.data(),  cp.data_dft.size())) &&
                 (cp.data_spec.empty() || f.write_section(cp.data_spec.data(), cp.data_spec.size()));
            if (!ok) {
                break;
            }
        }
    }

    // without the sync, a power loss can keep the rename and lose the data
    ok = ok && f.finish() && f.sync();

    f.close(); // before the rename: on Windows the handle does not share the file

    std::error_code ec;
    if (ok) {
        std::filesystem::rename(tmp, path, ec);
        ok = !ec;
    }

    if (!ok) {
        // a cancelled write is not an error: the state stays in RAM and goes out at the next idle
        if (!f.was_aborted()) {
            SRV_ERR(" - failed to write prompt-cache spill file '%s' (%.3f MiB)\n",
                    spill_fs_utf8(path).c_str(), spill_layout_of(h).size / (1024.0 * 1024.0));
        }

        std::filesystem::remove(tmp, ec);
        return false;
    }

    st.disk_bytes   = spill_layout_of(h).size;
    st.ckpt_on_disk = h.n_ckpt;
    st.on_disk      = true;
    st.disk_evicted = false;
    if (!spill_exists(st)) {
        return false;
    }

    // the new file repeats what the shorter disk-only entries in front of it hold, so those go. This is deduplication,
    // not eviction: nothing that is still the only copy of something is touched. Only other entries are erased, so the
    // callers that walk `states` keep their place
    for (auto it = states.begin(); it != states.end();) {
        if (&*it == &st || !it->on_disk || it->data.size() > 0 || it->prompt.tokens.empty() ||
                it->prompt.tokens.get_common_prefix(st.prompt.tokens) != it->prompt.tokens.size()) {
            ++it;
            continue;
        }

        SRV_TRC(" - dropping the disk copy of a %d token prefix, the %d token state covers it\n",
                it->prompt.n_tokens(), st.prompt.n_tokens());

        erase_spill(*it);
        it = states.erase(it);
    }
    return true;
}

// fork: --pool. a state leaves for another node in the layout of a spill file, with nothing written for it: a state on
// disk goes as pieces of its file, a state only in RAM as a copy of its blobs. of the checkpoints only the one the
// request goes back to goes along, after the base of its chain: none when the request goes on from the end of the state
// or the memory rolls back by itself, else the one the restore of a slot picks (the newest within the prefix the
// request shares with the state) - where a template that drops the reasoning of the last answer goes back to
bool server_prompt_cache::pool_export(std::list<server_prompt_cache_state>::iterator it, const std::filesystem::path & link,
                                      size_t n_match, size_t n_req, bool need_ckpt, int32_t n_swa, size_t min_reuse,
                                      server_pool_plan_fork & plan, std::string & error) {
    server_prompt_cache_state & st = *it;
    plan = {};

    const bool resident = st.data.size() > 0;
    const bool on_disk  = spill_exists(st);
    if (!resident && !on_disk) {
        error = "the state holds no data";
        return false;
    }

    // a state on disk: the header and the index of the checkpoints of its file
    spill_header fh = {};
    std::vector<spill_ckpt> findex;
    if (on_disk) {
        std::ifstream f(spill_path(st.uid), std::ios::binary);
        // the checks of unspill_state: a file of another layout or another state is not this state
        if (!f.read((char *) &fh, sizeof(fh)) || memcmp(fh.magic, SPILL_MAGIC, sizeof(fh.magic)) != 0 ||
                fh.version != SPILL_VERSION || !spill_header_sane(fh) || fh.uid != st.uid || fh.signature != signature) {
            error = "cannot read the state file";
            return false;
        }
        findex.resize(fh.n_ckpt);
        f.seekg((std::streamoff) spill_layout_of(fh).off_index);
        if (fh.n_ckpt > 0 && !f.read((char *) findex.data(), findex.size()*sizeof(spill_ckpt))) {
            error = "cannot read the checkpoints of the state";
            return false;
        }
    }

    // the checkpoints, oldest first: those of the file, unless the disk budget left some only in RAM
    const bool ckpt_file = on_disk && fh.n_ckpt >= st.prompt.checkpoints.size();
    struct ckpt_ref {
        int64_t   n_tokens;
        llama_pos pos_min;
        llama_pos pos_max;
        bool      delta; // applied on top of the ones before it
        bool      data;  // not a failed capture
    };
    std::vector<ckpt_ref> ckpts;
    if (ckpt_file) {
        for (const auto & e : findex) {
            ckpts.push_back({e.n_tokens, e.pos_min, e.pos_max, e.base_pos >= 0, e.n_tgt > 0});
        }
    } else {
        for (const auto & cp_ptr : st.prompt.checkpoints) {
            const common_prompt_checkpoint & cp = *cp_ptr;
            ckpts.push_back({cp.n_tokens, cp.pos_min, cp.pos_max, cp.is_delta(), !cp.data_tgt.empty()});
        }
    }

    // the checkpoints that go along: [k_first, k_end). the one a slot restores, as server_context_impl searches for it:
    // it ends within the shared prefix and starts before the first position computed again (minus the window of a SWA
    // memory), and the slot goes on from its last position
    const size_t n_state = st.prompt.tokens.size();
    size_t k_first = 0;
    size_t k_end   = 0;
    plan.n_reuse = n_match;
    if (need_ckpt && n_match >= n_state && n_req > n_state) {
        plan.n_reuse = n_state;
    } else if (need_ckpt) {
        plan.n_reuse = 0;
        const llama_pos pos_next = st.prompt.tokens.pos_next((int64_t) n_match);
        const llama_pos pos_eval = pos_next - (n_match < n_req ? 0 : 1);
        const llama_pos thold    = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);
        for (size_t k = ckpts.size(); k-- > 0; ) {
            const ckpt_ref & c = ckpts[k];
            if (!c.data || c.pos_max > pos_next || !(c.pos_min < thold || c.pos_min == 0)) {
                continue;
            }
            size_t b = k;
            while (b > 0 && (ckpts[b].delta || !ckpts[b].data)) {
                --b;
            }
            if (!ckpts[b].delta && ckpts[b].data) {
                k_first      = b;
                k_end        = k + 1;
                plan.n_reuse = std::min(st.prompt.tokens.size_up_to_pos(std::min(pos_next, std::max(c.pos_min + 1, c.pos_max))),
                                        (size_t) c.n_tokens);
            }
            break;
        }
    }
    plan.n_ckpt     = k_end - k_first;
    plan.n_ckpt_all = ckpts.size();
    if (plan.n_reuse < min_reuse) {
        error = string_format("the request takes %zu tokens from the state, fewer than %zu", (size_t) plan.n_reuse, min_reuse);
        return false;
    }

    std::vector<char> prompt;
    spill_header h = fh;
    if (!on_disk) {
        try {
            prompt = st.prompt.tokens.serialize();
        } catch (const std::exception & e) {
            error = std::string("cannot serialize the prompt of the state: ") + e.what();
            return false;
        }
        memcpy(h.magic, SPILL_MAGIC, sizeof(h.magic));
        h.version   = SPILL_VERSION;
        h.n_used    = st.prompt.n_used;
        h.uid       = st.uid;
        h.signature = signature;
        h.n_prompt  = prompt.size()/sizeof(llama_token);
        h.n_main    = st.data.main.size();
        h.n_drft    = st.data.drft.size();
        h.sum       = spill_head_sum(prompt.data(), prompt.size(), st.data.main.data(), h.n_main, st.data.drft.data(), h.n_drft);
    }
    // the records of the file keep their sums, those of RAM get them here
    std::vector<spill_ckpt> index;
    if (ckpt_file) {
        index.assign(findex.begin() + k_first, findex.begin() + k_end);
    } else {
        size_t k = 0;
        for (const auto & cp_ptr : st.prompt.checkpoints) {
            const common_prompt_checkpoint & cp = *cp_ptr;
            if (k >= k_first && k < k_end) {
                index.push_back(spill_ckpt_of(cp));
            }
            ++k;
        }
    }
    h.n_ckpt       = index.size();
    h.n_ckpt_bytes = 0;
    for (const auto & e : index) {
        h.n_ckpt_bytes += spill_align_up(e.n_tgt) + spill_align_up(e.n_dft) + spill_align_up(e.n_spec);
    }

    // the sections in the order of the file, each padded as the file pads it
    try {
        const auto add_mem = [&](const void * p, size_t n) {
            if (n > 0) {
                std::string s(spill_align_up(n), '\0');
                memcpy(&s[0], p, n);
                plan.pieces.push_back({0, s.size(), std::move(s)});
            }
        };
        const spill_layout fl = spill_layout_of(fh);
        add_mem(&h, sizeof(h));
        if (on_disk) {
            plan.pieces.push_back({fl.off_prompt, fl.off_index - fl.off_prompt, {}});
        } else {
            add_mem(prompt.data(), prompt.size());
            add_mem(st.data.main.data(), st.data.main.size());
            add_mem(st.data.drft.data(), st.data.drft.size());
        }
        add_mem(index.data(), index.size()*sizeof(spill_ckpt));
        if (ckpt_file) {
            uint64_t off = fl.off_ckpt;
            for (size_t k = 0; k < findex.size(); ++k) {
                const spill_ckpt & e = findex[k];
                const uint64_t len = spill_align_up(e.n_tgt) + spill_align_up(e.n_dft) + spill_align_up(e.n_spec);
                if (k >= k_first && k < k_end) {
                    plan.pieces.push_back({off, len, {}});
                }
                off += len;
            }
            if (off != fl.size) {
                error = "the checkpoints of the state do not add up to its file";
                return false;
            }
        } else {
            size_t k = 0;
            for (const auto & cp_ptr : st.prompt.checkpoints) {
                const common_prompt_checkpoint & cp = *cp_ptr;
                if (k >= k_first && k < k_end) {
                    add_mem(cp.data_tgt.data(),  cp.data_tgt.size());
                    add_mem(cp.data_dft.data(),  cp.data_dft.size());
                    add_mem(cp.data_spec.data(), cp.data_spec.size());
                }
                ++k;
            }
        }
    } catch (const std::bad_alloc &) {
        error = "not enough memory for a copy of the state";
        return false;
    }

    // the file is read through a link, so that the disk budget can drop the state meanwhile; the file itself where the
    // file system has no links
    if (on_disk) {
        std::error_code ec;
        if (!link.empty()) {
            std::filesystem::remove(link, ec);
            ec.clear();
            std::filesystem::create_hard_link(spill_path(st.uid), link, ec);
            // a link shares the time of the state it links, which can be older than the idle time of the transfer
            // cleaner (server_pool_xfer_clean_fork): until the plan is registered, it would remove the new link
            if (!ec) {
                std::error_code ec_time;
                std::filesystem::last_write_time(link, std::filesystem::file_time_type::clock::now(), ec_time);
            }
        }
        plan.file      = link.empty() || ec ? spill_path(st.uid) : link;
        plan.file_link = !link.empty() && !ec;
    }

    for (const auto & pc : plan.pieces) {
        plan.size += pc.len;
    }
    GGML_ASSERT(plan.size == spill_layout_of(h).size);
    return true;
}

// fork: --pool, a state that comes in chunks: each section is written into the buffer it ends up in, those of the
// checkpoints once the index before them is in
struct server_pool_inbox_fork::impl {
    struct target {
        uint64_t  off;
        uint64_t  len;
        uint8_t * dst;
    };

    uint64_t     received  = 0;
    bool         has_head  = false;
    bool         has_index = false;
    spill_header h   = {};
    spill_layout lay = {};

    std::vector<char>                   prompt;
    server_state_buf                    main;
    server_state_buf                    drft;
    std::vector<spill_ckpt>             index;
    std::list<common_prompt_checkpoint> ckpts; // a list: the targets point into its elements
    std::vector<target>                 targets;

    impl() {
        add(0, sizeof(h), &h);
    }

    void add(uint64_t off, uint64_t len, void * dst) {
        if (len > 0) {
            targets.push_back({off, len, (uint8_t *) dst});
        }
    }

    bool parse_head(uint64_t max_bytes, std::string & error) {
        if (memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) != 0 || h.version != SPILL_VERSION || !spill_header_sane(h)) {
            error = "not a prompt-cache state of this build";
            return false;
        }
        lay = spill_layout_of(h);
        if (max_bytes > 0 && lay.size > max_bytes) {
            error = string_format("the state (%.0f MiB) is larger than this node takes (%.0f MiB: the RAM limit of its "
                                  "prompt cache, or without one its host RAM above the reserve)",
                                  lay.size / (1024.0 * 1024.0), max_bytes / (1024.0 * 1024.0));
            return false;
        }
        try {
            prompt.resize(h.n_prompt*sizeof(llama_token));
            main.resize(h.n_main);
            drft.resize(h.n_drft);
            index.resize(h.n_ckpt);
        } catch (const std::bad_alloc &) {
            error = "not enough memory for the state";
            return false;
        }
        add(lay.off_prompt, prompt.size(), prompt.data());
        add(lay.off_main,   main.size(),   main.data());
        add(lay.off_drft,   drft.size(),   drft.data());
        add(lay.off_index,  index.size()*sizeof(spill_ckpt), index.data());
        has_head = true;
        return true;
    }

    bool parse_index(std::string & error) {
        uint64_t n_bytes = 0;
        for (const auto & e : index) {
            if (!spill_ckpt_sane(e)) {
                n_bytes = h.n_ckpt_bytes + 1;
                break;
            }
            n_bytes += spill_align_up(e.n_tgt) + spill_align_up(e.n_dft) + spill_align_up(e.n_spec);
        }
        if (n_bytes != h.n_ckpt_bytes) {
            error = "the checkpoints of the state do not add up";
            return false;
        }
        uint64_t off = lay.off_ckpt;
        try {
            for (const auto & e : index) {
                common_prompt_checkpoint & cp = ckpts.emplace_back();
                cp.n_tokens = e.n_tokens;
                cp.id_task  = -1; // not a task of this node
                cp.pos_min  = e.pos_min;
                cp.pos_max  = e.pos_max;
                cp.base_pos = e.base_pos;
                cp.data_tgt.resize(e.n_tgt);
                cp.data_dft.resize(e.n_dft);
                cp.data_spec.resize(e.n_spec);
                add(off, e.n_tgt,  cp.data_tgt.data());  off += spill_align_up(e.n_tgt);
                add(off, e.n_dft,  cp.data_dft.data());  off += spill_align_up(e.n_dft);
                add(off, e.n_spec, cp.data_spec.data()); off += spill_align_up(e.n_spec);
            }
        } catch (const std::bad_alloc &) {
            error = "not enough memory for the state";
            return false;
        }
        has_index = true;
        return true;
    }
};

server_pool_inbox_fork::server_pool_inbox_fork() : pimpl(std::make_unique<impl>()) {}

server_pool_inbox_fork::~server_pool_inbox_fork() = default;

bool server_pool_inbox_fork::write(uint64_t offset, const char * data, size_t n, uint64_t max_bytes, std::string & error) {
    impl & in = *pimpl;
    if (offset != in.received) {
        error = "a chunk out of order: " + std::to_string(in.received) + " bytes came before it";
        return false;
    }
    if (in.has_head && n > in.lay.size - in.received) {
        error = "more bytes than the state holds";
        return false;
    }
    in.received += n;

    // a section becomes known once the bytes before it are in, and takes its part of this chunk too: the header gives
    // the sections up to the index, the index (filled in the round after) those of the checkpoints
    size_t from = 0;
    while (from < in.targets.size()) {
        const size_t to = in.targets.size();
        for (size_t i = from; i < to; ++i) {
            const auto & t = in.targets[i];
            const uint64_t lo = std::max(offset, t.off);
            const uint64_t hi = std::min(offset + n, t.off + t.len);
            if (lo < hi) {
                memcpy(t.dst + (lo - t.off), data + (lo - offset), hi - lo);
            }
        }
        from = to;
        if (!in.has_head) {
            if (in.received >= sizeof(in.h) && !in.parse_head(max_bytes, error)) {
                return false;
            }
        } else if (!in.has_index && in.received >= in.lay.off_index + in.h.n_ckpt*sizeof(spill_ckpt)) {
            if (!in.parse_index(error)) {
                return false;
            }
        }
    }
    if (in.has_head && in.received > in.lay.size) {
        error = "more bytes than the state holds";
        return false;
    }
    return true;
}

// fork: --pool. a state another node sent joins this cache in RAM, as the save of a slot does: alloc() drops the prompts
// it holds whole and makes room, then the blobs the inbox filled move in without a copy. the portable signatures (the
// model and the configuration, not the machine) must match
bool server_prompt_cache::pool_import(server_pool_inbox_fork & inbox, uint64_t signature_from, size_t & n_tokens, std::string & error) {
    if (signature_portable == 0 || signature_from != signature_portable) {
        error = "the node runs another model or configuration (the portable cache signatures differ)";
        return false;
    }
    server_pool_inbox_fork::impl & in = *inbox.pimpl;
    if (!in.has_index || in.received != in.lay.size) {
        error = "the state came short";
        return false;
    }
    if (spill_head_sum(in.prompt.data(), in.prompt.size(), in.main.data(), in.main.size(), in.drft.data(), in.drft.size()) != in.h.sum) {
        error = "the state came damaged (its sum differs)";
        return false;
    }
    {
        size_t k = 0;
        for (const auto & cp : in.ckpts) {
            if (spill_ckpt_sum(in.index[k], cp.data_tgt.data(), cp.data_dft.data(), cp.data_spec.data()) != in.index[k].sum) {
                error = "a checkpoint of the state came damaged (its sum differs)";
                return false;
            }
            ++k;
        }
    }

    llama_tokens packed(in.h.n_prompt);
    if (!packed.empty()) {
        memcpy(packed.data(), in.prompt.data(), in.prompt.size());
    }
    server_prompt prompt;
    try {
        prompt.tokens = server_tokens::deserialize(packed, has_mtmd);
    } catch (const std::exception & e) {
        error = std::string("the prompt of the state cannot be restored: ") + e.what();
        return false;
    }
    prompt.n_used = in.h.n_used;
    n_tokens = prompt.tokens.size();

    // the conversation was here before: a state holds the whole prompt already
    for (const auto & st : states) {
        if (st.prompt.tokens.get_common_prefix(prompt.tokens) == prompt.tokens.size()) {
            return true;
        }
    }

    size_t size = in.main.size() + in.drft.size();
    for (const auto & cp : in.ckpts) {
        size += cp.size();
    }
    if (limit_size > 0 && size > limit_size) {
        error = string_format("the state (%.0f MiB) is larger than the prompt cache of this node (%.0f MiB)",
                              size / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return false;
    }

    // the request it came for takes it, whatever --cache-ram-min-tokens says
    const size_t ram_min = ram_min_tokens;
    ram_min_tokens = 0;
    server_prompt_cache_state * st = alloc(prompt, 0, 0);
    ram_min_tokens = ram_min;
    if (st == nullptr) {
        error = "the prompt cache cannot take the state";
        return false;
    }
    st->data.main          = std::move(in.main);
    st->data.drft          = std::move(in.drft);
    st->prompt.checkpoints.clear();
    for (auto & cp : in.ckpts) {
        st->prompt.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(std::move(cp)));
    }
    // the request it came for takes it into a slot right away: write_behind does not write it out meanwhile
    st->disk_evicted = true;
    return true;
}

bool server_prompt_cache::spill_exists(server_prompt_cache_state & st) {
    if (st.on_disk) {
        std::error_code ec;
        if (!std::filesystem::exists(spill_path(st.uid), ec) && !ec) {
            st.on_disk = false;
            st.disk_bytes = 0;
            st.ckpt_on_disk = 0;
            st.disk_evicted = true;
        }
    }
    return st.on_disk;
}

void server_prompt_cache::evict_state(server_prompt_cache_state & st) {
    if (!spill_exists(st)) {
        return; // without a disk copy the blobs are the only copy
    }

    st.data.main.clear();  st.data.main.shrink_to_fit();
    st.data.drft.clear();  st.data.drft.shrink_to_fit();

    // the checkpoints are in the file too, unless the disk budget refused them
    if (st.ckpt_on_disk == st.prompt.checkpoints.size()) {
        st.prompt.checkpoints.clear();
    }
}

bool server_prompt_cache::spill_state(server_prompt_cache_state & st, size_t disk_free) {
    if (!spill_exists(st)) {
        if (!write_state(st, nullptr, disk_free)) {
            return false;
        }
        SRV_TRC(" - spilled a prompt-cache state of %d tokens to disk (%.3f MiB)\n", st.prompt.n_tokens(), st.disk_bytes / (1024.0 * 1024.0));
    }

    evict_state(st);
    return true;
}

bool server_prompt_cache::unspill_state(server_prompt_cache_state & st) {
    if (!spill_exists(st)) {
        if (st.data.size() > 0) {
            st.disk_evicted = false; // a real cache hit makes the state worth persisting again
        }
        return st.data.size() > 0;
    }
    std::error_code touch_ec;
    std::filesystem::last_write_time(spill_path(st.uid), std::filesystem::file_time_type::clock::now(), touch_ec);

    // the file stays. It holds exactly the bytes the state is handing back, so taking it away only
    // buys a full rewrite at the next idle. enforce_disk_limit decides when a file goes
    if (st.data.size() > 0 && st.prompt.checkpoints.size() >= st.ckpt_on_disk) {
        return true; // clean: already in RAM, no read needed
    }

    const std::filesystem::path path = spill_path(st.uid);

    spill_file f;

    // A read failure discards the file, but an intact resident state remains usable.
    const auto drop = [&]() {
        f.close();

        std::error_code ec;
        std::filesystem::remove(path, ec);

        st.prompt.checkpoints.clear();
        st.on_disk      = false;
        st.disk_bytes   = 0;
        st.ckpt_on_disk = 0;

        return st.data.size() > 0;
    };

    if (!f.open_read(path)) {
        SRV_ERR(" - failed to read spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
    }

    spill_header h = {};
    if (!f.read_at(0, &h, sizeof(h)) ||
            memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) != 0 ||
            h.version != SPILL_VERSION || h.uid != st.uid ||
            h.signature != signature ||
            !spill_header_sane(h) ||
            spill_layout_of(h).size != f.size()) {
        SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
    }

    const spill_layout lay = spill_layout_of(h);

    const auto packed = st.prompt.tokens.serialize();
    std::vector<char> stored(packed.size());
    if (h.n_prompt != packed.size()/sizeof(llama_token) ||
            (!stored.empty() && !f.read_at(lay.off_prompt, stored.data(), stored.size())) || stored != packed) {
        return drop();
    }

    // a resident state keeps its blobs, the file only gives the checkpoints then
    const auto rd = [&](server_state_buf & v, uint64_t off, uint64_t n, spill_sum & sum) -> bool {
        v.resize(n);
        return n == 0 || f.read_summed(off, v.data(), n, sum);
    };
    server_prompt_data loaded;
    const bool resident = st.data.size() > 0;
    if (!resident) {
        spill_sum sum;
        sum.add(stored.data(), stored.size());
        if (!rd(loaded.main, lay.off_main, h.n_main, sum) || !rd(loaded.drft, lay.off_drft, h.n_drft, sum)) {
            SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
            return drop();
        }
        if (sum.get() != h.sum) {
            SRV_ERR(" - damaged spilled prompt-cache state '%s': the sum of its blobs differs\n", spill_fs_utf8(path).c_str());
            return drop();
        }
    }

    if (h.n_ckpt > 0) {
        // checkpoints the disk budget refused are only in RAM, so the list is replaced by the file
        // only when the file actually holds one
        st.prompt.checkpoints.clear();

        std::vector<spill_ckpt> index(h.n_ckpt);

        if (!f.read_at(lay.off_index, index.data(), index.size()*sizeof(spill_ckpt))) {
            SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
            return drop();
        }

        // the index must describe exactly the bytes the header reserved, or the walk below would
        // read past the sections it owns
        uint64_t n_bytes = 0;
        for (const auto & e : index) {
            if (!spill_ckpt_sane(e)) {
                n_bytes = h.n_ckpt_bytes + 1;
                break;
            }
            n_bytes += spill_align_up(e.n_tgt) + spill_align_up(e.n_dft) + spill_align_up(e.n_spec);
        }

        if (n_bytes != h.n_ckpt_bytes) {
            SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
            return drop();
        }

        uint64_t off = lay.off_ckpt;

        const auto rd_ckpt = [&](auto & v, uint64_t n, spill_sum & sum) -> bool {
            v.resize(n);
            if (n > 0 && !f.read_summed(off, v.data(), n, sum)) {
                return false;
            }
            off += spill_align_up(n);
            return true;
        };

        for (const auto & e : index) {
            common_prompt_checkpoint cp;
            cp.n_tokens = e.n_tokens;
            cp.id_task  = e.id_task;
            cp.pos_min  = e.pos_min;
            cp.pos_max  = e.pos_max;
            cp.base_pos = e.base_pos;

            // the sum of spill_ckpt_sum: the fields of the record, then its blobs
            spill_sum sum;
            sum.add(&e, offsetof(spill_ckpt, sum));
            if (!rd_ckpt(cp.data_tgt, e.n_tgt, sum) || !rd_ckpt(cp.data_dft, e.n_dft, sum) || !rd_ckpt(cp.data_spec, e.n_spec, sum)) {
                SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
                return drop();
            }
            if (sum.get() != e.sum) {
                SRV_ERR(" - damaged spilled prompt-cache state '%s': the sum of a checkpoint differs\n", spill_fs_utf8(path).c_str());
                return drop();
            }

            st.prompt.checkpoints.push_back(std::make_shared<common_prompt_checkpoint>(std::move(cp)));
        }
    }

    f.close();

    if (!resident) {
        st.data = std::move(loaded);
    }
    return true; // the state is clean now: same bytes in RAM and on disk
}

void server_prompt_cache::erase_spill(const server_prompt_cache_state & st) {
    if (st.on_disk && !spill_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove(spill_path(st.uid), ec);
    }
}

void server_prompt_cache::enforce_disk_limit() {
    if (spill_dir.empty()) {
        return;
    }
    common_file_lock lock(spill_dir_path() / ".prompt-cache.lock");
    if (spill_limit > 0 && lock.acquire([] { return false; })) {
        spill_prune(spill_dir_path(), spill_limit);
    }
    for (auto it = states.begin(); it != states.end();) {
        spill_exists(*it);
        if (!it->on_disk && it->data.size() == 0) {
            it = states.erase(it);
        } else {
            ++it;
        }
    }
}

size_t server_prompt_cache::free_ram(size_t need, bool may_write) {
    size_t freed = 0;

    if (!spill_dir.empty()) {
        const size_t disk_free = spill_limit == 0 ? SIZE_MAX : spill_limit;

        // a clean state already has its copy on disk (a cache hit read it back, or write_behind put it there while
        // idle), so releasing its RAM costs no I/O at all. Take those first
        for (auto & st : states) {
            if (freed >= need) {
                break;
            }
            if (!st.is_clean()) {
                continue;
            }
            const size_t big = st.size();
            evict_state(st);
            freed += big - st.size();
        }

        // then oldest first: a state below the --cache-min-tokens step is not written (write_state), so it leaves RAM
        // as it is and costs that much recompute; a longer one is written out first, when a write may run now. The
        // writes erase only other entries (the prefixes they cover), so the walk keeps its place
        for (auto it = states.begin(); it != states.end() && freed < need; ) {
            server_prompt_cache_state & st = *it;
            if (st.on_disk || st.data.size() == 0) {
                ++it;
                continue;
            }
            if (below_step(st)) {
                SRV_TRC(" - a state of %d tokens below the %zu token step leaves RAM without a write\n",
                        st.prompt.n_tokens(), min_tokens);
                freed += st.size();
                it = states.erase(it);
                continue;
            }
            if (may_write) {
                const size_t big = st.size();
                if (spill_state(st, disk_free)) {
                    freed += big - st.size();
                }
            }
            ++it;
        }

        // checkpoints the disk budget refused, or ones read back by a cache hit, are still in RAM.
        // they only save recomputation, so let them go when spilling did not cover the deficit
        for (auto & st : states) {
            if (freed >= need) {
                break;
            }
            if (!st.on_disk || st.prompt.checkpoints.empty()) {
                continue;
            }
            size_t big = 0;
            for (const auto & ckpt_ptr : st.prompt.checkpoints) {
                const common_prompt_checkpoint & ckpt = *ckpt_ptr;
                big += ckpt.size();
            }
            st.prompt.checkpoints.clear();
            freed += big;
        }

        if (may_write) {
            enforce_disk_limit();
            if (freed < need) {
                freed += drop_resident(need - freed);
            }
        }
        return freed;
    }

    // no spill dir: fall back to dropping oldest states outright (front = oldest)
    while (freed < need && !states.empty()) {
        SRV_WRN(" - dropping oldest prompt-cache entry (%.3f MiB)\n",
                states.front().size() / (1024.0 * 1024.0));
        freed += states.front().size();
        states.pop_front();
    }

    return freed;
}

size_t server_prompt_cache::drop_resident(size_t need) {
    size_t freed = 0;
    for (auto it = states.begin(); it != states.end() && freed < need;) {
        const size_t before = it->size();
        if (before == 0) {
            ++it;
            continue;
        }
        if (spill_exists(*it)) {
            evict_state(*it);
        }
        if (it->on_disk) {
            SRV_WRN(" - releasing %.3f MiB of prompt-cache RAM after spill could not cover the deficit\n", before / (1024.0 * 1024.0));
            it->prompt.checkpoints.clear();
            freed += before - it->size();
            ++it;
        } else {
            // the only copy: the conversation is computed again from the front when it comes back
            SRV_WRN(" - dropping a prompt-cache state of %d tokens (%.3f MiB) that has no copy on disk: host memory pressure\n",
                    it->prompt.n_tokens(), before / (1024.0 * 1024.0));
            freed += before;
            it = states.erase(it);
        }
    }
    return freed;
}

size_t server_prompt_cache::write_behind(const std::function<bool()> & interrupted, bool all) {
    if (spill_dir.empty() || (interrupted && interrupted())) {
        return 0;
    }

    enforce_disk_limit();
    const size_t disk_free = spill_limit == 0 ? SIZE_MAX : spill_limit;

    // the RAM the cache holds over its limit: what the saves of a busy server left for later (defer_writes)
    const auto over_limit = [this]() -> size_t {
        if (limit_size == 0) {
            return 0;
        }
        const size_t resident = size();
        const size_t outside  = ram_outside ? ram_outside() : 0;
        const size_t budget   = limit_size - std::min(limit_size, outside);
        return resident > budget ? resident - budget : 0;
    };

    size_t n_written = 0;
    size_t n_bytes   = 0;

    const int64_t t_start = ggml_time_us();

    // oldest first, the same order free_ram spills in: those are the states it will evict next, and evicting a state that
    // is already on disk is free. all: every changed state; else only as many as the RAM limit pushes out
    size_t need = all ? 0 : over_limit();
    for (auto it = states.begin(); it != states.end() && (all || need > 0); ++it) {
        auto & st = *it;

        if (st.data.size() == 0) {
            continue;
        }
        // leaves RAM with no write in the eviction below: a clean state, one below the --cache-min-tokens step
        if (!all && (st.is_clean() || below_step(st))) {
            need -= std::min(need, st.size());
            continue;
        }
        if (spill_exists(st) || st.disk_evicted) {
            continue;
        }

        // a request is waiting - the rest goes out at the next idle
        if (interrupted && interrupted()) {
            break;
        }

        const size_t ram = st.size();
        if (!write_state(st, &interrupted, disk_free)) {
            if (interrupted && interrupted()) {
                break; // stopped part way through, the file is gone and the state is still in RAM
            }
            continue;
        }

        n_bytes   += st.disk_bytes;
        n_written += 1;
        need      -= std::min(need, ram);
    }

    // Writes that failed or were skipped must not leave the idle cache above its RAM budget.
    if (const size_t need_now = over_limit(); need_now > 0) {
        const size_t freed = free_ram(need_now, false);
        if (freed < need_now) {
            drop_resident(need_now - freed);
        }
    }

    if (n_written > 0) {
        const double t_ms = (ggml_time_us() - t_start) / 1000.0;

        SRV_INF(" - prompt cache: wrote %zu state(s) to disk while idle%s, %.3f MiB in %.0f ms (%.2f GB/s)\n",
                n_written, all ? "" : " (over the RAM limit)", n_bytes / (1024.0 * 1024.0), t_ms, n_bytes / (t_ms * 1e6));
    }

    return n_written;
}

size_t server_prompt_cache::disk_size() const {
    size_t res = 0;
    if (!spill_dir.empty()) {
        for (const auto & entry : spill_entries(spill_dir_path())) {
            res += entry.bytes;
        }
    }
    return res;
}

bool server_prompt_cache::make_room(size_t bytes) {
    // 07.11: a state that comes from another node takes its whole size in RAM as its chunks arrive. on a 395 the RAM is
    // what the 96 GiB of the GPU leave, and the cache holds all of it down to the reserve: the room is made before the
    // transfer, by the writes a transfer can wait for, not after it in the pages of the system
    const size_t avail = common_host_mem_available();
    if (avail == 0) {
        return true;
    }
    const size_t want = reserve_size + bytes;
    if (avail >= want) {
        return true;
    }
    const size_t need = want - avail;
    SRV_INF(" - room for a state of %.3f MiB from another node: relieving %.3f MiB from the prompt cache (available %.3f MiB, reserve %.3f MiB)\n",
            bytes / (1024.0 * 1024.0), need / (1024.0 * 1024.0), avail / (1024.0 * 1024.0), reserve_size / (1024.0 * 1024.0));
    size_t freed = free_ram(need, true);
    if (freed < need) {
        freed += drop_resident(need - freed);
    }
    return freed >= need;
}

bool server_prompt_cache::evict_for_reserve(size_t extra, bool may_write) {
    if (reserve_size == 0) {
        return true;
    }

    const size_t avail = common_host_mem_available();
    if (avail == 0) {
        return true; // could not determine available RAM -> don't act on it
    }

    const size_t want = reserve_size + extra;
    if (avail >= want) {
        reserve_short = false;
        return true;
    }

    // free the deficit only (freed heap may not return to the OS immediately, so re-querying
    // available RAM here would over-evict). free_ram spills cold states to disk when a spill dir
    // is configured, and only drops them outright as a last resort.
    const size_t need = want - avail;

    // 12.11: with nothing in RAM to give back the memory is taken elsewhere (the KV cells an elastic cache took from the
    // system memory, another program), and this ran on every update, twice a second, each line saying it relieves 2 GB
    // it did not have. said once when that starts and then once a minute
    const size_t held = size();
    if (held == 0) {
        const int64_t t_us = ggml_time_us();
        if (!reserve_short || t_us - t_reserve_log_us >= 60ll*1000*1000) {
            SRV_WRN(" - host memory is short (available %.3f MiB < reserve %.3f MiB) and the prompt cache holds nothing in RAM: "
                    "the memory is taken outside the cache\n", avail / (1024.0 * 1024.0), reserve_size / (1024.0 * 1024.0));
            t_reserve_log_us = t_us;
        }
        reserve_short = true;
        return false;
    }
    reserve_short = false;

    size_t freed = free_ram(need, may_write);

    // 07.11: the writes wait while other slots run (defer_writes), and with requests in the queue they always run, so the
    // deferred write never came and drop_resident below threw the long states away: their conversations were computed
    // from the start when they came back. a long state pushed out of RAM is written out now instead (the --cache-min-tokens
    // step still holds): the decodes of the other slots wait for the write, a second or so per few GB, while computing the
    // state again takes minutes of the GPU and pauses them too (LLAMA_PREFILL_EXCLUSIVE)
    if (freed < need && !may_write && !spill_dir.empty()) {
        freed += free_ram(need - freed, true);
    }

    if (freed < need) {
        freed += drop_resident(need - freed);
    }

    SRV_WRN(" - host memory pressure: %.3f of %.3f MiB given back by the prompt cache, which held %.3f MiB (available %.3f "
            "MiB < reserve %.3f MiB)\n", freed / (1024.0 * 1024.0), need / (1024.0 * 1024.0), held / (1024.0 * 1024.0),
            avail / (1024.0 * 1024.0), reserve_size / (1024.0 * 1024.0));

    return freed >= need;
}
