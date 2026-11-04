#pragma once

// the fork's load of a tensor whose buffer has no async uploads (llama_model_loader::load_all_data): an integrated GPU
// since 04.11, whose weight buffers Vulkan maps into the host, so a tensor set is a memcpy with no queue submit. the
// tensor is read in chunks into two buffers in turn, and while one chunk is read the one before it is copied into the
// tensor on another thread, so the load takes as long as the reads, as with the async uploads. one copy runs at a
// time. the path before read the whole tensor into one buffer and copied it after, the disk idle meanwhile.
// LLAMA_LOAD_CHUNK_MB: the chunk (64), 0 for that path

#include "llama-impl.h"

#include <cstddef>
#include <cstdint>
#include <future>
#include <vector>

struct ggml_tensor;
struct llama_file;
struct llama_load_watch_fork;

struct llama_load_chunked_fork {
    // the chunk in bytes, 0 for the whole-tensor path
    static size_t chunk_size();

    // reads [offs, offs + size) of file into cur, from the start of cur. the reads count in t_read_us and n_read, the
    // waits for a copy in t_wait_us; an error of an earlier copy is thrown here
    void load(llama_file & file, ggml_tensor * cur, size_t offs, size_t size, llama_load_watch_fork & watch,
              int64_t & t_read_us, size_t & n_read, int64_t & t_wait_us);

    // waits for the copies, throws the error of one. before the loaded weights are used, and before another tensor set
    // of the same loader: two sets of one backend at once are not safe for every backend
    void finish(llama_load_watch_fork & watch, int64_t & t_wait_us);

    // waits for the copies without a throw: on an error the buffers and tensors go away after this
    ~llama_load_chunked_fork();

private:
    void wait(int i, llama_load_watch_fork & watch, int64_t & t_wait_us);

    std::vector<no_init<uint8_t>> bufs[2];
    std::future<void>             done[2]; // after the buffers: the copies out of them end first
    int                           cur_buf = 0;
};
