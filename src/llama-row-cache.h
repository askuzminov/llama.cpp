#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// On-demand row reader for a tensor that is never loaded: the rows asked for are gathered
// through a fixed host budget of aligned file blocks, read with direct I/O. Used for tensors
// marked TENSOR_READ_LAZY when the lazy mode is LLAMA_LAZY_MODE_DIRECT_IO, where the mmap path
// would instead leave the table to the page cache. See llama_model_loader::lazy_read.
struct llama_row_cache {
    // n_cols/n_rows are the tensor dimensions, offs the start of its data in the file
    // n_threads is the read queue depth, 0 lets the cache size it from the machine
    llama_row_cache(const std::string & path, size_t offs, ggml_type type,
                    int64_t n_cols, int64_t n_rows, size_t budget, int n_threads);
    ~llama_row_cache();

    llama_row_cache(const llama_row_cache &) = delete;
    llama_row_cache & operator=(const llama_row_cache &) = delete;

    // gathers n rows into dst, dequantized: dst holds n*n_cols floats
    void get_rows(const int32_t * rows, int64_t n, float * dst);

    size_t budget() const;      // bytes actually reserved, rounded to whole blocks
    size_t block_size() const;

    void print_stats(const char * tag) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
