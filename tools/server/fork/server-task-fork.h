#pragma once

// fork: types for the prompt cache disk spill, used by server-task.h. The spill itself is in server-task-fork.cpp

#include <cstddef>
#include <cstdint>
#include <filesystem> // spill paths in server_prompt_cache
#include <memory>
#include <new>
#include <string>
#include <vector>

// a UTF-8 path as the file system takes it (a narrow path on Windows is in the ANSI codepage)
std::filesystem::path server_fs_path_fork(const std::string & utf8);

// fork: --pool, the bytes of a state that moves to another node, in the layout of a spill file, with nothing written
// to disk for it: a state on disk goes as pieces of its file, a state only in RAM as a copy of its blobs. the header
// and the index of the checkpoints that go along are new
struct server_pool_plan_fork {
    struct piece {
        uint64_t    off;  // in `file`, when `data` is empty
        uint64_t    len;
        std::string data; // the bytes themselves, padding included
    };
    std::filesystem::path file; // the spill file the pieces without data read, empty = none
    bool file_link = false;     // `file` is a hard link made for the move, removed with it
    std::vector<piece> pieces;
    uint64_t size    = 0;
    uint64_t n_ckpt  = 0; // the checkpoints that go along, of n_ckpt_all
    uint64_t n_ckpt_all = 0;
    uint64_t n_reuse = 0; // the tokens of the request the receiving node takes from the state
};

// fork: --pool, a state that comes from another node in chunks, in the layout of a spill file, parsed into the buffers
// of a cache state as it arrives: nothing goes to disk, and the blobs are not copied again when the state joins the
// cache (server_prompt_cache::pool_import)
struct server_pool_inbox_fork {
    server_pool_inbox_fork();
    ~server_pool_inbox_fork();

    // the chunks come in order. max_bytes: the largest state this node takes, 0 = any
    bool write(uint64_t offset, const char * data, size_t n, uint64_t max_bytes, std::string & error);

    struct impl;
    std::unique_ptr<impl> pimpl;
};

// the spill writes the state blobs with the OS cache bypassed, which needs a sector-aligned source.
// allocating them aligned lets the spill move them without a bounce buffer
constexpr size_t SERVER_STATE_ALIGN = 4096;

template <typename T>
struct server_state_alloc {
    using value_type = T;

    server_state_alloc() = default;

    template <typename U> server_state_alloc(const server_state_alloc<U> &) {}

    T * allocate(size_t n) {
        return (T *) ::operator new(n*sizeof(T), std::align_val_t(SERVER_STATE_ALIGN));
    }

    void deallocate(T * p, size_t) noexcept {
        ::operator delete(p, std::align_val_t(SERVER_STATE_ALIGN));
    }

    template <typename U> bool operator==(const server_state_alloc<U> &) const { return true;  }
    template <typename U> bool operator!=(const server_state_alloc<U> &) const { return false; }
};

using server_state_buf = std::vector<uint8_t, server_state_alloc<uint8_t>>;
