// llama_row_cache reads rows of a tensor straight out of the gguf, so the tests write a file that
// looks like one: a header of an odd length, then the rows. F32 covers the plain copy, Q8_0 the
// dequantizing path.

#include "llama-row-cache.h"

#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const size_t HEADER = 1234;   // not a multiple of anything, like a real gguf

static float ref_value(int64_t row, int64_t col) {
    return (float) (row % 97) + 0.125f*(float) (col % 13);
}

static std::string write_table(const char * name, ggml_type type, int64_t n_cols, int64_t n_rows) {
    std::string path = std::string(name);

    const auto * traits = ggml_get_type_traits(type);

    std::vector<float>   row_f32((size_t) n_cols);
    std::vector<uint8_t> row_enc(ggml_row_size(type, n_cols));

    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot create %s\n", path.c_str());
        exit(1);
    }

    std::vector<uint8_t> header(HEADER, 0xAB);
    fwrite(header.data(), 1, header.size(), f);

    for (int64_t r = 0; r < n_rows; r++) {
        for (int64_t c = 0; c < n_cols; c++) {
            row_f32[c] = ref_value(r, c);
        }

        if (type == GGML_TYPE_F32) {
            memcpy(row_enc.data(), row_f32.data(), row_enc.size());
        } else {
            traits->from_float_ref(row_f32.data(), row_enc.data(), n_cols);
        }

        fwrite(row_enc.data(), 1, row_enc.size(), f);
    }

    fclose(f);

    return path;
}

// the same rows read back through ggml, so the check does not depend on the quantization error
static std::vector<float> expected(ggml_type type, int64_t n_cols, const std::vector<int32_t> & rows) {
    const auto * traits = ggml_get_type_traits(type);

    std::vector<float>   row_f32((size_t) n_cols);
    std::vector<uint8_t> row_enc(ggml_row_size(type, n_cols));
    std::vector<float>   out;

    out.reserve(rows.size()*n_cols);

    for (int32_t r : rows) {
        for (int64_t c = 0; c < n_cols; c++) {
            row_f32[c] = ref_value(r, c);
        }

        if (type == GGML_TYPE_F32) {
            out.insert(out.end(), row_f32.begin(), row_f32.end());
            continue;
        }

        traits->from_float_ref(row_f32.data(), row_enc.data(), n_cols);
        traits->to_float(row_enc.data(), row_f32.data(), n_cols);

        out.insert(out.end(), row_f32.begin(), row_f32.end());
    }

    return out;
}

static int n_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        n_fail++;
    }
}

static void check_gather(llama_row_cache & cache, ggml_type type, int64_t n_cols,
                         const std::vector<int32_t> & rows, const char * what) {
    std::vector<float> got(rows.size()*n_cols, -1.0f);
    cache.get_rows(rows.data(), (int64_t) rows.size(), got.data());

    const std::vector<float> want = expected(type, n_cols, rows);

    for (size_t i = 0; i < want.size(); i++) {
        if (got[i] != want[i]) {
            fprintf(stderr, "FAIL: %s: element %zu (row %d) is %f, expected %f\n",
                    what, i, rows[i/n_cols], got[i], want[i]);
            n_fail++;
            return;
        }
    }
}

static void test_type(ggml_type type, int64_t n_cols, int64_t n_rows, int n_threads) {
    const std::string path = write_table("test-row-cache.bin", type, n_cols, n_rows);

    {
        llama_row_cache cache(path, HEADER, type, n_cols, n_rows, n_threads);

        check(cache.block_size() > 0, "the block size is set");

        // every row once, in order: the sequential case, blocks are shared by neighbours
        {
            std::vector<int32_t> rows;
            for (int64_t r = 0; r < n_rows; r++) {
                rows.push_back((int32_t) r);
            }
            check_gather(cache, type, n_cols, rows, "sequential");
        }

        // the first and the last row, which cover the two partial blocks at the ends
        check_gather(cache, type, n_cols, { 0, (int32_t) (n_rows - 1) }, "edges");

        // the same row many times: everything after the first is a hit
        check_gather(cache, type, n_cols, std::vector<int32_t>(64, (int32_t) (n_rows/3)), "repeated");

        // random access, twice, so the second pass runs against a warm cache
        std::mt19937 rng(1234);
        std::vector<int32_t> rows(512);
        for (auto & r : rows) {
            r = (int32_t) (rng() % (uint32_t) n_rows);
        }
        check_gather(cache, type, n_cols, rows, "random, cold");
        check_gather(cache, type, n_cols, rows, "random, warm");

        // a gather longer than the cache: it has to be split into chunks
        std::vector<int32_t> many(4096);
        for (size_t i = 0; i < many.size(); i++) {
            many[i] = (int32_t) (i % (size_t) n_rows);
        }
        check_gather(cache, type, n_cols, many, "longer than the cache");

        cache.print_stats("test");
    }

    remove(path.c_str());
}

static void set_env(const char * name, const char * value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

int main() {
    // ggml_time_us() divides by a frequency that ggml_time_init() sets, and the cache times its
    // reads. In a real run llama_backend_init() does this, the test has to do it itself
    ggml_time_init();

    // every shape is run twice: once with each block read on its own, once with adjacent blocks
    // merged into one request. both paths must return exactly the same rows
    for (const char * run : { "1", "64" }) {
        set_env("LLAMA_ROW_CACHE_RUN", run);

        // rows shorter than a block, so several rows share one
        test_type(GGML_TYPE_F32, 128, 1000, 4);

        // rows longer than a block, so every row straddles block boundaries
        test_type(GGML_TYPE_F32, 4096, 300, 2);

        // dequantization, with a row size that is not a power of two
        test_type(GGML_TYPE_Q8_0, 1536, 500, 3);

        // one reader
        test_type(GGML_TYPE_F32, 2048, 100, 1);

        // reader count picked by the cache itself
        test_type(GGML_TYPE_Q8_0, 4096, 400, 0);
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }

    printf("OK\n");

    return 0;
}
