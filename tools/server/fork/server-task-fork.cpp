// fork: server_prompt_cache members for the disk spill and the host-RAM reserve

#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "server-common.h"

#include <cstdio>
#include <filesystem>

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
};

static_assert(sizeof(spill_header) == 72, "unexpected spill_header layout");

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
};

static_assert(sizeof(spill_ckpt) == 48, "unexpected spill_ckpt layout");

constexpr char     SPILL_MAGIC[8] = { 'L', 'C', 'P', 'C', 'A', 'C', 'H', 'E' };
constexpr uint32_t SPILL_VERSION  = 4;

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
        const DWORD share  = write ? 0 : FILE_SHARE_READ;
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

        // the header disagrees with the name, or the file is truncated - the state cannot be restored
        const spill_layout lay = spill_header_sane(h) ? spill_layout_of(h) : spill_layout{};
        if (h.signature != signature || lay.size == 0 || lay.size != f.size()) {
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

    enforce_disk_limit();

    return found.size();
}

void server_prompt_cache::persist() {
    if (spill_dir.empty()) {
        // no spill dir: nothing was written, nothing to clean up
        return;
    }

    for (auto & st : states) {
        spill_state(st);
    }

    enforce_disk_limit();

    size_t n_disk = 0;
    for (const auto & st : states) {
        n_disk += st.on_disk ? 1 : 0;
    }

    SRV_INF(" - prompt cache: %zu of %zu states kept on disk, %.3f MiB\n",
            n_disk, states.size(), disk_size() / (1024.0 * 1024.0));
}

bool server_prompt_cache::write_state(server_prompt_cache_state & st, const std::function<bool()> * cancel, size_t disk_free) {
    if (spill_dir.empty() || st.on_disk) {
        return false;
    }
    if (st.data.main.empty() && st.data.drft.empty()) {
        return false; // nothing big to write out
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

    // the context checkpoints go out with the state: after a restart they are what lets a prompt
    // that matches only a prefix roll back, instead of being processed from the start again
    std::vector<spill_ckpt> index;
    index.reserve(st.prompt.checkpoints.size());

    for (const auto & cp : st.prompt.checkpoints) {
        index.push_back({
            cp.n_tokens, cp.id_task, cp.pos_min, cp.pos_max, cp.base_pos,
            cp.data_tgt.size(), cp.data_dft.size(), cp.data_spec.size(),
        });

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

    if (st.uid == 0) {
        st.uid = next_uid++;
    }
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
        for (const auto & cp : st.prompt.checkpoints) {
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
    return true;
}

void server_prompt_cache::evict_state(server_prompt_cache_state & st) {
    if (!st.on_disk) {
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
    if (!st.on_disk && !write_state(st, nullptr, disk_free)) {
        return false;
    }

    evict_state(st);
    return true;
}

bool server_prompt_cache::unspill_state(server_prompt_cache_state & st) {
    if (!st.on_disk) {
        return true;
    }

    // the file stays. It holds exactly the bytes the state is handing back, so taking it away only
    // buys a full rewrite at the next idle. enforce_disk_limit decides when a file goes
    if (st.data.size() > 0 && st.prompt.checkpoints.size() >= st.ckpt_on_disk) {
        return true; // clean: already in RAM, no read needed
    }

    const std::filesystem::path path = spill_path(st.uid);

    spill_file f;

    // a file that cannot be read back is of no use to anyone: drop it and clear the on-disk flag,
    // so the state is not retried on every load and stops counting against the disk budget.
    // the entry is left with no data at all, which is what the caller must react to.
    const auto drop = [&]() {
        f.close();

        std::error_code ec;
        std::filesystem::remove(path, ec);

        st.data.main.clear(); st.data.main.shrink_to_fit();
        st.data.drft.clear(); st.data.drft.shrink_to_fit();
        st.prompt.checkpoints.clear();
        st.on_disk      = false;
        st.disk_bytes   = 0;
        st.ckpt_on_disk = 0;

        return false;
    };

    if (!f.open_read(path)) {
        SRV_ERR(" - failed to read spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
    }

    spill_header h = {};
    if (!f.read_at(0, &h, sizeof(h)) ||
            memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) != 0 ||
            h.version != SPILL_VERSION ||
            h.signature != signature ||
            !spill_header_sane(h) ||
            spill_layout_of(h).size != f.size()) {
        SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
    }

    const spill_layout lay = spill_layout_of(h);

    const auto rd = [&](server_state_buf & v, uint64_t off, uint64_t n) -> bool {
        v.resize(n);
        return n == 0 || f.read_at(off, v.data(), n);
    };
    if (!rd(st.data.main, lay.off_main, h.n_main) || !rd(st.data.drft, lay.off_drft, h.n_drft)) {
        SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
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

        const auto rd_ckpt = [&](std::vector<uint8_t> & v, uint64_t n) -> bool {
            v.resize(n);
            if (n > 0 && !f.read_at(off, v.data(), n)) {
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

            if (!rd_ckpt(cp.data_tgt, e.n_tgt) || !rd_ckpt(cp.data_dft, e.n_dft) || !rd_ckpt(cp.data_spec, e.n_spec)) {
                SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
                return drop();
            }

            st.prompt.checkpoints.push_back(std::move(cp));
        }
    }

    f.close();

    return true; // the state is clean now: same bytes in RAM and on disk
}

void server_prompt_cache::erase_spill(const server_prompt_cache_state & st) {
    if (st.on_disk && !spill_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove(spill_path(st.uid), ec);
    }
}

void server_prompt_cache::enforce_disk_limit() {
    if (spill_dir.empty() || spill_limit == 0) {
        return;
    }
    size_t used = disk_size();

    // drop the least valuable on-disk state (oldest first on a tie) until within budget. Value is
    // how often the state was reused times how much work it saves, so a rarely used or a short and
    // cheap to redo prompt goes first and a cold start keeps the long ones that pay off
    while (used > spill_limit) {
        auto   sel     = states.end();
        size_t sel_val = 0;

        for (auto it = states.begin(); it != states.end(); ++it) {
            if (!it->on_disk) {
                continue;
            }
            const size_t val = (size_t) (it->prompt.n_used + 1) * it->prompt.tokens.size();

            if (sel == states.end() || val < sel_val) {
                sel     = it;
                sel_val = val;
            }
        }
        if (sel == states.end()) {
            break; // nothing on disk left to drop
        }
        const bool resident = sel->data.size() > 0;

        SRV_WRN(" - prompt-cache disk budget exceeded, dropping %s state (%.3f MiB, %d tokens, used %u time(s))\n",
                resident ? "the disk copy of a resident" : "spilled", sel->disk_bytes / (1024.0 * 1024.0),
                sel->prompt.n_tokens(), sel->prompt.n_used);

        used -= std::min(used, sel->disk_bytes);
        erase_spill(*sel);

        if (resident) {
            // the blobs are still in RAM, so only the copy goes - the entry stays usable
            sel->on_disk    = false;
            sel->disk_bytes = 0;
        } else {
            states.erase(sel);
        }
    }
}

size_t server_prompt_cache::free_ram(size_t need) {
    size_t freed = 0;

    if (!spill_dir.empty()) {
        // spill oldest resident states to disk: this frees their RAM but keeps them restorable, so
        // we never drop a state just to satisfy a RAM deficit. Spilling the whole (bounded) cache
        // is the most we can do for host-RAM pressure; the disk budget alone decides what is
        // dropped for good (enforce_disk_limit)
        size_t disk_free = spill_limit == 0 ? SIZE_MAX : spill_limit - std::min(spill_limit, disk_size());

        // a clean state already has its copy on disk (write_behind put it there while idle), so
        // releasing its RAM costs no I/O at all. Take those first
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

        for (auto & st : states) {
            if (freed >= need) {
                break;
            }
            if (st.on_disk || st.data.size() == 0) {
                continue;
            }
            const size_t big = st.size();
            if (spill_state(st, disk_free)) {
                freed     += big - st.size();
                disk_free -= std::min(disk_free, st.disk_bytes);
            }
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
            for (const auto & ckpt : st.prompt.checkpoints) {
                big += ckpt.size();
            }
            st.prompt.checkpoints.clear();
            freed += big;
        }

        enforce_disk_limit();
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

// how many leading tokens of `st` some other entry already holds on disk. Writing `st` out buys
// only the tokens past that point
static size_t spill_prefix_on_disk(const std::list<server_prompt_cache_state> & states,
        const server_prompt_cache_state & st) {
    size_t res = 0;

    for (const auto & other : states) {
        if (&other == &st || !other.on_disk) {
            continue;
        }

        const size_t len = other.prompt.tokens.get_common_prefix(st.prompt.tokens);
        if (len == other.prompt.tokens.size()) {
            res = std::max(res, len);
        }
    }

    return res;
}

size_t server_prompt_cache::write_behind(const std::function<bool()> & interrupted) {
    if (spill_dir.empty()) {
        return 0;
    }

    size_t disk_free = spill_limit == 0 ? SIZE_MAX : spill_limit - std::min(spill_limit, disk_size());

    size_t n_written = 0;
    size_t n_bytes   = 0;

    const int64_t t_start = ggml_time_us();

    // oldest first, the same order free_ram spills in: those are the states it will evict next, and
    // evicting a state that is already on disk is free
    for (auto it = states.begin(); it != states.end(); ++it) {
        auto & st = *it;

        if (st.on_disk || st.data.size() == 0) {
            continue;
        }

        // an older file already holds the front of this state. A rewrite would move gigabytes to
        // win back only the tokens past that point, so let them pile up to the minimum first. What
        // a crash costs is then that much recompute, not the whole prompt
        const size_t n_have = spill_prefix_on_disk(states, st);
        if (n_have > 0 && st.prompt.tokens.size() - n_have < min_tokens) {
            SRV_TRC(" - %zu of %zu tokens are on disk already, below the %zu token rewrite step, skipping\n",
                    n_have, st.prompt.tokens.size(), min_tokens);
            continue;
        }

        // a request is waiting - the rest goes out at the next idle
        if (interrupted && interrupted()) {
            break;
        }

        if (!write_state(st, &interrupted, disk_free)) {
            if (interrupted && interrupted()) {
                break; // stopped part way through, the file is gone and the state is still in RAM
            }
            continue;
        }

        disk_free -= std::min(disk_free, st.disk_bytes);
        n_bytes   += st.disk_bytes;
        n_written += 1;

        // the new file repeats what the shorter ones in front of it hold, so those go. This is
        // deduplication, not eviction: nothing that is still the only copy of something is touched
        for (auto it2 = states.begin(); it2 != states.end();) {
            if (it2 == it || !it2->on_disk || it2->data.size() > 0 ||
                    it2->prompt.tokens.get_common_prefix(st.prompt.tokens) != it2->prompt.tokens.size()) {
                ++it2;
                continue;
            }

            SRV_TRC(" - dropping the disk copy of a %d token prefix, the %d token state covers it\n",
                    it2->prompt.n_tokens(), st.prompt.n_tokens());

            if (disk_free != SIZE_MAX) {
                disk_free += it2->disk_bytes;
            }

            erase_spill(*it2);
            it2 = states.erase(it2);
        }
    }

    if (n_written > 0) {
        const double t_ms = (ggml_time_us() - t_start) / 1000.0;

        SRV_INF(" - prompt cache: wrote %zu state(s) to disk while idle, %.3f MiB in %.0f ms (%.2f GB/s)\n",
                n_written, n_bytes / (1024.0 * 1024.0), t_ms, n_bytes / (t_ms * 1e6));
    }

    return n_written;
}

size_t server_prompt_cache::disk_size() const {
    size_t res = 0;

    for (const auto & state : states) {
        if (state.on_disk) {
            res += state.disk_bytes;
        }
    }

    return res;
}

void server_prompt_cache::evict_for_reserve(size_t extra) {
    if (reserve_size == 0) {
        return;
    }

    const size_t avail = common_host_mem_available();
    if (avail == 0) {
        return; // could not determine available RAM -> don't act on it
    }

    const size_t want = reserve_size + extra;
    if (avail >= want) {
        return;
    }

    // free the deficit only (freed heap may not return to the OS immediately, so re-querying
    // available RAM here would over-evict). free_ram spills cold states to disk when a spill dir
    // is configured, and only drops them outright as a last resort.
    const size_t need = want - avail;
    SRV_WRN(" - host memory pressure, relieving %.3f MiB from the prompt cache (available %.3f MiB < reserve %.3f MiB)\n",
            need / (1024.0 * 1024.0), avail / (1024.0 * 1024.0), reserve_size / (1024.0 * 1024.0));
    free_ram(need);
}
