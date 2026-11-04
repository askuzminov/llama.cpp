#include "llama-load-chunked-fork.h"

#include "llama-fork.h"
#include "llama-load-watch-fork.h"
#include "llama-mmap.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>

size_t llama_load_chunked_fork::chunk_size() {
    static const size_t bytes = [] {
        const int64_t mb = llama_fork_env_int("LLAMA_LOAD_CHUNK_MB", 64);
        return mb > 0 ? (size_t) mb * 1024 * 1024 : 0;
    }();
    return bytes;
}

void llama_load_chunked_fork::wait(int i, llama_load_watch_fork & watch, int64_t & t_wait_us) {
    if (!done[i].valid()) {
        return;
    }
    watch.step("copy wait");
    const int64_t t_start = ggml_time_us();
    // get() throws what the copy threw, and leaves the future empty
    done[i].get();
    t_wait_us += ggml_time_us() - t_start;
}

void llama_load_chunked_fork::load(llama_file & file, ggml_tensor * cur, size_t offs, size_t size,
                                   llama_load_watch_fork & watch, int64_t & t_read_us, size_t & n_read,
                                   int64_t & t_wait_us) {
    // with direct IO the offset, the size and the destination of a read are multiples of the alignment (a power of 2,
    // 1 without direct IO). the loader takes this path only with a chunk size, a chunk of 0 would never advance
    const size_t align      = std::max<size_t>(file.read_alignment(), 1);
    const size_t chunk      = GGML_PAD(chunk_size(), align);
    GGML_ASSERT((align & (align - 1)) == 0 && chunk > 0);
    const size_t read_start = offs & ~(align - 1);
    const size_t read_end   = (offs + size + align - 1) & ~(align - 1);

    file.seek(read_start, SEEK_SET);

    int64_t i_chunk = 0;
    for (size_t pos = read_start; pos < read_end; i_chunk++) {
        const size_t read_size = std::min(chunk, read_end - pos);

        // the copy out of this buffer was waited for when the next one started; this only makes it certain
        wait(cur_buf, watch, t_wait_us);

        // as large as the largest read so far: the tensors come biggest-first, and a small model does not page in two
        // whole chunks
        auto & buf = bufs[cur_buf];
        if (buf.size() < read_size + align) {
            buf = std::vector<no_init<uint8_t>>();
            buf.resize(read_size + align);
        }
        uint8_t * dst = (uint8_t *) GGML_PAD((uintptr_t) buf.data(), align);

        watch.step("read", ggml_get_name(cur), i_chunk);
        const int64_t t_start = ggml_time_us();
        file.read_raw_unsafe(dst, read_size);
        t_read_us += ggml_time_us() - t_start;
        n_read    += read_size;

        // the part of the chunk inside the tensor: the alignment pads it at both ends of the tensor
        const size_t begin = std::max(pos, offs);
        const size_t end   = std::min(pos + read_size, offs + size);

        const uint8_t * src     = dst + (begin - pos);
        const size_t    dst_off = begin - offs;
        const size_t    len     = end - begin;

        // one copy at a time: the one before has to land first
        wait(cur_buf ^ 1, watch, t_wait_us);
        done[cur_buf] = std::async(std::launch::async, [cur, src, dst_off, len] {
            ggml_backend_tensor_set(cur, src, dst_off, len);
        });

        pos     += read_size;
        cur_buf ^= 1;
    }
}

void llama_load_chunked_fork::finish(llama_load_watch_fork & watch, int64_t & t_wait_us) {
    // the older copy first: cur_buf is the buffer the next read goes to
    wait(cur_buf,     watch, t_wait_us);
    wait(cur_buf ^ 1, watch, t_wait_us);
}

llama_load_chunked_fork::~llama_load_chunked_fork() {
    for (auto & d : done) {
        if (d.valid()) {
            d.wait();
        }
    }
}
