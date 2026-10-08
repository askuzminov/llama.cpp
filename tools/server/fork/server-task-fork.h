#pragma once

// fork: types for the prompt cache disk spill, used by server-task.h. The spill itself is in server-task-fork.cpp

#include <cstddef>
#include <cstdint>
#include <filesystem> // spill paths in server_prompt_cache
#include <new>
#include <vector>

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
