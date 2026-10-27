#include "llama-row-cache-fork.h"
#include "llama-env-parse-fork.h"

#include "llama-impl.h"
#include "llama-mmap.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

// a whole number, as the fork's other knobs (llama_fork_env_parse; the standalone build of test-row-cache has no
// llama_fork_env_int): atoll read "12abc" as 12 without a word. not above 0 is the default
size_t row_cache_env(const char * name, size_t def) {
    const char * val = getenv(name);
    int64_t parsed = 0;
    switch (llama_fork_env_parse(val, parsed)) {
        case LLAMA_FORK_ENV_UNSET:
            return def;
        case LLAMA_FORK_ENV_BAD:
            LLAMA_LOG_ERROR("%s: %s=\"%s\" is not a whole number, using %zu\n", __func__, name, val, def);
            return def;
        case LLAMA_FORK_ENV_OK:
            break;
    }

    return parsed > 0 ? (size_t) parsed : def;
}

// a switch: 0 is off, any other whole number on
bool row_cache_env_flag(const char * name, bool def) {
    const char * val = getenv(name);
    int64_t parsed = 0;
    switch (llama_fork_env_parse(val, parsed)) {
        case LLAMA_FORK_ENV_UNSET:
            return def;
        case LLAMA_FORK_ENV_BAD:
            LLAMA_LOG_ERROR("%s: %s=\"%s\" is not a whole number, using %d\n", __func__, name, val, (int) def);
            return def;
        case LLAMA_FORK_ENV_OK:
            break;
    }

    return parsed != 0;
}

} // namespace

struct llama_row_cache::impl {
    // one read request: n consecutive blocks of the file, read into pool slots i_first .. i_first + n
    struct run {
        uint64_t block;
        uint32_t n;
        uint32_t i_first;
    };

    size_t    offs;
    ggml_type type;
    int64_t   n_cols;
    int64_t   n_rows;
    size_t    row_size;
    size_t    blk_size;
    size_t    run_max;         // most blocks one read request may cover
    size_t    file_size = 0;

    ggml_to_float_t to_float = nullptr;

    std::vector<uint8_t> pool;       // one slot per block the chunk in flight holds
    size_t               n_slots;

    std::vector<uint8_t> row_buf;    // staging for rows that straddle a block boundary
    std::mutex           mutex;      // one cache is shared by every context of the model

    // readers: llama_file carries a position and a single event, so one handle per worker
    std::vector<std::unique_ptr<llama_file>> files;

    std::vector<std::thread> workers;
    std::mutex               job_mutex;
    std::condition_variable  job_cv;
    std::condition_variable  done_cv;
    const std::vector<run> * job_runs  = nullptr;
    size_t                   job_next  = 0;
    size_t                   job_done  = 0;
    bool                     job_stop  = false;
    std::string              job_error;

    int64_t n_hit       = 0;
    int64_t n_miss      = 0;
    int64_t n_read      = 0;   // blocks pulled from the file
    int64_t n_req       = 0;   // read requests those blocks were merged into
    int64_t n_gather    = 0;
    int64_t n_row_ask   = 0;
    int64_t t_wait_us   = 0;   // wall time the gather spends waiting for a batch of requests
    int64_t t_gather_us = 0;

    // measured inside the readers: t_io_us over t_wait_us is how many requests the drive really
    // had in flight, which is the only way to tell a slow drive from readers that serialize
    std::atomic<int64_t> t_io_us{0};
    std::atomic<int32_t> n_inflight{0};
    std::atomic<int32_t> n_inflight_max{0};

    // blocks per point of the read path sweep, 0 = do not run it
    size_t bench_n = 0;

    // the slice path (LLAMA_ROW_CACHE_SLICES, on by default): the distinct rows of a gather in file order, a job a slice
    // of them, which its reader reads into a buffer of its own and dequantizes straight into dst, once per row however
    // many entries of the list ask for it. the block path above has a job per request and a barrier per pool of
    // 2*run_max*3 blocks, every job takes job_mutex twice, and a repeat is found only within a pool: on 27.10 a prompt
    // ubatch of 3923 tokens (62768 rows) took 205 ms of inputs on the 3090 and 130-240 ms on the 395. the PLE n-grams
    // of qwen4exp repeat: in 3923 words of English text 53% of the bigrams and 39% of the trigrams came before, and a
    // repeat is 8 rows. a run of r.n distinct rows from r.block, r.i_first unused
    bool                              slices        = true;
    size_t                            slice_run_max = 1;    // most rows one request of a slice covers
    const int32_t *                   job_rows      = nullptr;   // the distinct rows, ascending
    const uint32_t *                  job_first     = nullptr;   // distinct row j goes to the entries job_ent[job_first[j] ..
    const uint32_t *                  job_ent       = nullptr;   // job_first[j + 1]) of the list
    float *                           job_dst       = nullptr;
    std::vector<std::vector<uint8_t>> slice_bufs;           // one per reader
    std::atomic<int64_t>              slice_req{0};
    std::vector<uint64_t>             plan_keys;            // the scratch of a gather: row << 32 | entry, sorted
    std::vector<int32_t>              plan_rows;
    std::vector<uint32_t>             plan_first;
    std::vector<uint32_t>             plan_ent;

    impl(const std::string & path, size_t offs, ggml_type type, int64_t n_cols, int64_t n_rows,
         int n_threads);
    ~impl();

    void stop_workers();

    void read_run(size_t i_worker, const run & r);
    void read_slice(size_t i_worker, const run & r);
    void run_jobs(const std::vector<run> & runs);
    void bench(size_t n_bench);

    void gather(const int32_t * rows, int64_t n, float * dst);
    void gather_slices(const int32_t * rows, int64_t n, float * dst);
};

llama_row_cache::impl::impl(const std::string & path, size_t offs, ggml_type type,
                            int64_t n_cols, int64_t n_rows, int n_threads) :
        offs(offs), type(type), n_cols(n_cols), n_rows(n_rows) {
    // the reads are timed with ggml_time_us(), which needs ggml_time_init() before it on Windows.
    // llama_backend_init() calls it, so every caller that loads a model is covered
    row_size = ggml_row_size(type, n_cols);
    to_float = ggml_get_type_traits(type)->to_float;

    if (type != GGML_TYPE_F32 && to_float == nullptr) {
        throw std::runtime_error(format("row cache: type %s cannot be dequantized", ggml_type_name(type)));
    }

    // buffered on purpose. The page cache then holds every row this file gave out, in whatever
    // memory is free, and it survives the process: a second run starts warm. Unbuffered reads of
    // this file also get no concurrency at all, 32 readers get what one gets
    files.emplace_back(new llama_file(path.c_str(), "rb", false));
    file_size = files[0]->size();

    // one block is one row: the rows a gather asks for are scattered, so a bigger block reads
    // bytes nothing wants. 4 KiB used to read 4.47 GiB where the rows themselves are 0.25 GiB
    blk_size = row_size;

    // blocks that turn out to be adjacent are read in one request, which is what makes a gather
    // over real text cheap: neighbouring tokens ask for neighbouring rows
    run_max = std::max<size_t>(row_cache_env("LLAMA_ROW_CACHE_RUN", 64), 1);

    bench_n = row_cache_env("LLAMA_ROW_CACHE_BENCH", 0);

    // the pool holds the chunk in flight and nothing else: keeping blocks between gathers was
    // measured and gave 0.6 t/s of 342, because the page cache of the system holds the same
    // pages anyway. Wide enough for a chunk to merge run_max adjacent blocks into one request
    const size_t blocks_per_row = row_size/blk_size + 2;

    n_slots = 2*run_max*blocks_per_row;

    pool.resize(n_slots*blk_size);
    row_buf.resize(row_size);

    // random block reads are latency bound, so the reader count is really the queue depth the
    // drive gets to see: a handful of threads leaves an nvme almost idle
    if (n_threads <= 0) {
        const unsigned hw = std::thread::hardware_concurrency();
        n_threads = (int) std::min<unsigned>(std::max<unsigned>(hw, 8), 32);
    }
    // one file handle per reader, and a cast that wrapped to 0 would leave the pool with no
    // worker at all, so the request is clamped before it becomes an int
    n_threads = (int) std::min<size_t>(row_cache_env("LLAMA_ROW_CACHE_THREADS", (size_t) n_threads), 1024);

    for (int i = 1; i < n_threads; i++) {
        files.emplace_back(new llama_file(path.c_str(), "rb", false));
    }

    // a request of the slice path stays under 256 KiB, a reader's buffer holds one
    slices = row_cache_env_flag("LLAMA_ROW_CACHE_SLICES", true);
    if (slices) {
        slice_run_max = std::max<size_t>(1, std::min<size_t>(run_max, 256*1024/row_size));
        slice_bufs.assign((size_t) n_threads, std::vector<uint8_t>(slice_run_max*row_size));
    }

    // ~impl does not run for a constructor that throws, and destroying a joinable thread calls
    // std::terminate. The guard joins the pool on the way out until the constructor disarms it
    struct pool_guard {
        impl * self;
        ~pool_guard() { if (self) { self->stop_workers(); } }
    } guard { this };

    for (int i = 0; i < n_threads; i++) {
        workers.emplace_back([this, i] {
            while (true) {
                size_t i_job = 0;
                bool   slice = false;
                {
                    std::unique_lock<std::mutex> lock(job_mutex);
                    job_cv.wait(lock, [this] {
                        return job_stop || (job_runs != nullptr && job_next < job_runs->size());
                    });
                    if (job_stop) {
                        return;
                    }
                    i_job = job_next++;
                    slice = job_rows != nullptr;
                }

                const size_t n_jobs = job_runs->size();
                const run &  r      = (*job_runs)[i_job];

                const int32_t in_flight = n_inflight.fetch_add(1) + 1;
                int32_t       peak      = n_inflight_max.load();
                while (in_flight > peak && !n_inflight_max.compare_exchange_weak(peak, in_flight)) {}

                const int64_t t_io_start = ggml_time_us();

                try {
                    if (slice) {
                        read_slice(i, r);
                    } else {
                        read_run(i, r);
                    }
                } catch (const std::exception & err) {
                    std::lock_guard<std::mutex> lock(job_mutex);
                    if (job_error.empty()) {
                        job_error = err.what();
                    }
                }

                t_io_us.fetch_add(ggml_time_us() - t_io_start);
                n_inflight.fetch_sub(1);

                {
                    std::lock_guard<std::mutex> lock(job_mutex);
                    if (++job_done == n_jobs) {
                        done_cv.notify_one();
                    }
                }
            }
        });
    }

    // the file name matters for a read path measurement: a split model keeps the table in one
    // part, and a raw drive benchmark has to be pointed at that same part
    const size_t i_name = path.find_last_of("/\\");

    LLAMA_LOG_INFO("%s: %.2f GiB table stays on disk in %s, blocks of %zu B, %d readers, "
            "up to %zu blocks per request, %s\n",
            __func__, ggml_row_size(type, n_cols)*n_rows/1024.0/1024.0/1024.0,
            path.c_str() + (i_name == std::string::npos ? 0 : i_name + 1),
            blk_size, n_threads, slices ? slice_run_max : run_max,
            slices ? "slices of the row list" : "chunks of the block pool (LLAMA_ROW_CACHE_SLICES=0)");

    if (bench_n > 0) {
        bench(bench_n);
    }

    guard.self = nullptr;
}

// Read path self test: random reads of the file, through the same handles, workers and pool
// slots a gather uses. It sweeps the request size and the queue depth in one model load, so
// "do more readers help" and "do bigger requests help" are answered in seconds instead of a
// full run. Compare it against a raw drive benchmark of the same size and depth
void llama_row_cache::impl::bench(size_t n_bench) {
    const uint64_t n_file_blocks = file_size/blk_size;
    if (n_file_blocks == 0) {
        return;
    }

    std::vector<run> runs;

    uint64_t seed = 0x853c49e6748fea9bULL;

    for (size_t n_blk = 1; n_blk <= run_max; n_blk *= 4) {
        for (size_t depth = 1; depth <= files.size(); depth *= 2) {
            const size_t batch = std::min(depth, n_slots/n_blk);
            if (batch == 0) {
                break;
            }

            const size_t n_batches = std::max<size_t>(n_bench/(n_blk*batch), 1);

            runs.resize(batch);

            const int64_t t_io_before = t_io_us.load();
            const int64_t t_start     = ggml_time_us();

            n_inflight_max.store(0);

            // a run of n_blk blocks has to start early enough to fit, and a file shorter than one
            // run leaves only block 0
            const uint64_t n_starts = n_file_blocks > n_blk ? n_file_blocks - n_blk : 1;

            for (size_t i = 0; i < n_batches; i++) {
                // the runs take consecutive slots from the start of the pool, as in a gather, and
                // batch*n_blk <= n_slots keeps them apart: two readers on one slot would be a data race
                for (size_t k = 0; k < batch; k++) {
                    seed = seed*6364136223846793005ULL + 1442695040888963407ULL;

                    runs[k] = { (seed >> 16) % n_starts, (uint32_t) n_blk, (uint32_t) (k*n_blk) };
                }

                run_jobs(runs);
            }

            const int64_t n_reqs  = (int64_t) n_batches*batch;
            const int64_t n_bytes = n_reqs*n_blk*blk_size;
            const int64_t t_wall  = std::max<int64_t>(ggml_time_us() - t_start, 1);
            const int64_t t_io    = t_io_us.load() - t_io_before;

            LLAMA_LOG_INFO("%s: read path: %7zu B x depth %2zu, %5lld requests, %7.1f MiB/s, "
                    "%6.0f requests/s, %5.0f us per request, %6.0f us as seen by a reader, "
                    "%d readers at once\n",
                    __func__, n_blk*blk_size, batch, (long long) n_reqs,
                    n_bytes/1024.0/1024.0/(t_wall/1e6), n_reqs/(t_wall/1e6),
                    (double) t_wall/n_reqs, (double) t_io/n_reqs, n_inflight_max.load());
        }
    }

    // the sweep is not a gather, keep it out of the run stats
    n_read    = 0;
    n_req     = 0;
    t_wait_us = 0;
    t_io_us.store(0);
    n_inflight_max.store(0);
}

llama_row_cache::impl::~impl() {
    stop_workers();
}

void llama_row_cache::impl::stop_workers() {
    {
        std::lock_guard<std::mutex> lock(job_mutex);
        job_stop = true;
    }
    job_cv.notify_all();

    for (auto & worker : workers) {
        worker.join();
    }
}

void llama_row_cache::impl::read_run(size_t i_worker, const run & r) {
    llama_file * file = files[i_worker].get();

    const size_t off  = (size_t) r.block*blk_size;
    const size_t len  = (size_t) r.n*blk_size;

    uint8_t * dst = pool.data() + (size_t) r.i_first*blk_size;

    file->seek(off, SEEK_SET);

    // a read past the end of the file would throw, the tail of the last block is zero filled
    const size_t want = off < file_size ? std::min(len, file_size - off) : 0;
    if (want > 0) {
        file->read_raw_unsafe(dst, want);
    }
    if (want < len) {
        memset(dst + want, 0, len - want);
    }
}

// distinct rows r.block .. r.block + r.n of the gather: a row, with the rows right after it in the file when the gather
// asks for them too, in one request; dequantized into the first entry that asks for it and copied to the others
void llama_row_cache::impl::read_slice(size_t i_worker, const run & r) {
    llama_file * file = files[i_worker].get();
    uint8_t *    buf  = slice_bufs[i_worker].data();

    const int64_t k_end = (int64_t) r.block + r.n;

    for (int64_t k = (int64_t) r.block; k < k_end; ) {
        const int64_t row = job_rows[k];

        int64_t len = 1;
        while (k + len < k_end && (size_t) len < slice_run_max && job_rows[k + len] == row + len) {
            len++;
        }

        const size_t off  = offs + (size_t) row*row_size;
        const size_t size = (size_t) len*row_size;

        // a table cut short by the end of the file reads zeros there, as the block path does
        const size_t want = off < file_size ? std::min(size, file_size - off) : 0;
        if (want > 0) {
            file->seek(off, SEEK_SET);
            file->read_raw_unsafe(buf, want);
        }
        if (want < size) {
            memset(buf + want, 0, size - want);
        }

        for (int64_t j = 0; j < len; j++) {
            const uint8_t * src = buf + (size_t) j*row_size;
            const uint32_t  e0  = job_first[k + j];
            const uint32_t  e1  = job_first[k + j + 1];
            float *         out = job_dst + (size_t) job_ent[e0]*n_cols;
            if (type == GGML_TYPE_F32) {
                memcpy(out, src, row_size);
            } else {
                to_float(src, out, n_cols);
            }
            for (uint32_t e = e0 + 1; e < e1; e++) {
                memcpy(job_dst + (size_t) job_ent[e]*n_cols, out, n_cols*sizeof(float));
            }
        }

        slice_req.fetch_add(1, std::memory_order_relaxed);
        k += len;
    }
}

void llama_row_cache::impl::run_jobs(const std::vector<run> & runs) {
    if (runs.empty()) {
        return;
    }

    const int64_t t_start = ggml_time_us();

    {
        std::lock_guard<std::mutex> lock(job_mutex);
        job_runs  = &runs;
        job_next  = 0;
        job_done  = 0;
        job_error.clear();
    }
    job_cv.notify_all();

    std::string err;
    {
        std::unique_lock<std::mutex> lock(job_mutex);
        done_cv.wait(lock, [&] { return job_done == runs.size(); });
        job_runs  = nullptr;
        err       = job_error;
    }

    t_wait_us += ggml_time_us() - t_start;

    if (!err.empty()) {
        throw std::runtime_error(format("row cache: %s", err.c_str()));
    }
}

void llama_row_cache::impl::gather(const int32_t * rows, int64_t n, float * dst) {
    std::unordered_map<uint64_t, uint32_t> pinned;   // blocks this chunk holds -> slot
    std::vector<uint64_t>                  want;     // of those, the ones still to be read
    std::vector<run>                       runs;

    // the rows are walked in chunks that fit the pool, so nothing a chunk reads is overwritten
    // before its rows are copied out. The next chunk starts empty
    int64_t i = 0;
    while (i < n) {
        pinned.clear();
        want.clear();
        runs.clear();

        int64_t i_end = i;
        for (; i_end < n; i_end++) {
            const int64_t row = rows[i_end];
            if (row < 0 || row >= n_rows) {
                throw std::runtime_error(format("row cache: row %" PRId64 " out of range", row));
            }

            const uint64_t b0 = (offs + (size_t) row*row_size)/blk_size;
            const uint64_t b1 = (offs + (size_t) (row + 1)*row_size - 1)/blk_size;

            size_t n_new = 0;
            for (uint64_t b = b0; b <= b1; b++) {
                n_new += pinned.count(b) == 0;
            }

            // one row always goes in, however many blocks it needs
            if (i_end > i && pinned.size() + n_new > n_slots) {
                break;
            }

            for (uint64_t b = b0; b <= b1; b++) {
                if (pinned.count(b) > 0) {
                    n_hit++;   // two rows of the chunk share the block
                    continue;
                }

                pinned[b] = 0;
                want.push_back(b);
                n_miss++;
            }
        }

        // in file order, so that blocks which happen to be adjacent merge into one request. Block
        // k of the order goes to slot k, so the blocks of a run sit next to each other in the pool
        std::sort(want.begin(), want.end());

        for (size_t k = 0; k < want.size(); k++) {
            pinned[want[k]] = (uint32_t) k;

            if (runs.empty() || want[k] != runs.back().block + runs.back().n || runs.back().n >= run_max) {
                runs.push_back({ want[k], 1, (uint32_t) k });
            } else {
                runs.back().n++;
            }
        }

        run_jobs(runs);

        n_read += want.size();
        n_req  += runs.size();

        for (int64_t k = i; k < i_end; k++) {
            const size_t   off = offs + (size_t) rows[k]*row_size;
            const uint64_t b0  = off/blk_size;
            const uint64_t b1  = (off + row_size - 1)/blk_size;

            const uint8_t * src = nullptr;

            if (b0 == b1) {
                src = pool.data() + (size_t) pinned[b0]*blk_size + (off - b0*blk_size);
            } else {
                // the row straddles blocks, put it together first
                size_t done = 0;
                for (uint64_t b = b0; b <= b1; b++) {
                    const size_t from = b == b0 ? off - b0*blk_size : 0;
                    const size_t take = std::min(blk_size - from, row_size - done);

                    memcpy(row_buf.data() + done, pool.data() + (size_t) pinned[b]*blk_size + from, take);
                    done += take;
                }
                src = row_buf.data();
            }

            float * out = dst + k*n_cols;
            if (type == GGML_TYPE_F32) {
                memcpy(out, src, row_size);
            } else {
                to_float(src, out, n_cols);
            }
        }

        i = i_end;
    }
}

void llama_row_cache::impl::gather_slices(const int32_t * rows, int64_t n, float * dst) {
    if (n <= 0) {
        return;
    }
    if (n > (int64_t) UINT32_MAX) {
        throw std::runtime_error(format("row cache: a gather of %" PRId64 " rows", n));
    }

    // the distinct rows in file order, and for each the entries of the list that ask for it
    plan_keys.resize((size_t) n);
    for (int64_t k = 0; k < n; k++) {
        if (rows[k] < 0 || rows[k] >= n_rows) {
            throw std::runtime_error(format("row cache: row %" PRId64 " out of range", (int64_t) rows[k]));
        }
        plan_keys[k] = (uint64_t) rows[k] << 32 | (uint64_t) k;
    }
    std::sort(plan_keys.begin(), plan_keys.end());

    plan_rows.clear();
    plan_first.clear();
    plan_ent.resize((size_t) n);
    for (int64_t k = 0; k < n; k++) {
        const int32_t row = (int32_t) (plan_keys[k] >> 32);
        if (plan_rows.empty() || plan_rows.back() != row) {
            plan_rows.push_back(row);
            plan_first.push_back((uint32_t) k);
        }
        plan_ent[k] = (uint32_t) plan_keys[k];
    }
    plan_first.push_back((uint32_t) n);

    const int64_t n_uniq = (int64_t) plan_rows.size();

    // up to 4 slices per reader, taken as the readers come free: a row the page cache does not hold is a read of the
    // drive, and the reader that meets a few of them must not hold up the gather. a gather of up to that many rows,
    // as at a decode step, has a slice per row: the requests of the block path, without its pool and map
    const int64_t n_slices = std::min<int64_t>(n_uniq, 4*(int64_t) files.size());
    const int64_t per      = (n_uniq + n_slices - 1)/n_slices;

    std::vector<run> runs;
    runs.reserve((size_t) n_slices);
    for (int64_t k = 0; k < n_uniq; k += per) {
        runs.push_back({ (uint64_t) k, (uint32_t) std::min<int64_t>(per, n_uniq - k), 0 });
    }

    // published to the readers by the lock run_jobs takes, and theirs until the last job is done
    job_rows  = plan_rows.data();
    job_first = plan_first.data();
    job_ent   = plan_ent.data();
    job_dst   = dst;
    slice_req.store(0);

    struct reset {
        impl * self;
        ~reset() { self->job_rows = nullptr; self->job_first = nullptr; self->job_ent = nullptr; self->job_dst = nullptr; }
    } on_exit { this };

    run_jobs(runs);

    n_hit  += n - n_uniq;
    n_miss += n_uniq;
    n_read += n_uniq;
    n_req  += slice_req.load();
}

llama_row_cache::llama_row_cache(const std::string & path, size_t offs, ggml_type type,
                                 int64_t n_cols, int64_t n_rows, int n_threads) :
    pimpl(new impl(path, offs, type, n_cols, n_rows, n_threads)) {}

llama_row_cache::~llama_row_cache() = default;

void llama_row_cache::get_rows(const int32_t * rows, int64_t n, float * dst) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    const int64_t t_start = ggml_time_us();

    if (pimpl->slices) {
        pimpl->gather_slices(rows, n, dst);
    } else {
        pimpl->gather(rows, n, dst);
    }

    pimpl->n_gather++;
    pimpl->n_row_ask   += n;
    pimpl->t_gather_us += ggml_time_us() - t_start;
}

void llama_row_cache::print_stats(const char * tag) const {
    const impl *  p     = pimpl.get();
    const int64_t n_ask = p->n_hit + p->n_miss;
    if (n_ask == 0) {
        return;
    }

    if (p->slices) {
        LLAMA_LOG_INFO("%s: %s: %.1f%% of %" PRId64 " rows asked repeat a row of the same gather, %" PRId64 " rows read in %"
                PRId64 " requests (%.2f GiB, %.0f B each)\n",
                __func__, tag, 100.0*p->n_hit/n_ask, n_ask, p->n_read, p->n_req,
                p->n_read*p->row_size/1024.0/1024.0/1024.0,
                p->n_req > 0 ? (double) p->n_read*p->row_size/p->n_req : 0.0);
    } else {
        LLAMA_LOG_INFO("%s: %s: %.1f%% of %" PRId64 " block lookups hit, %" PRId64 " blocks read "
                "in %" PRId64 " requests (%.2f GiB, %.0f B each)\n",
                __func__, tag, 100.0*p->n_hit/n_ask, n_ask, p->n_read, p->n_req,
                p->n_read*p->blk_size/1024.0/1024.0/1024.0,
                p->n_req > 0 ? (double) p->n_read*p->blk_size/p->n_req : 0.0);
    }

    const int64_t t_io = p->t_io_us.load();

    // t_io is summed over the readers and t_wait is the wall time they were waited on, so their
    // ratio is how many requests were really in flight. Close to 1 with many readers means the
    // requests are serialized somewhere, not that the drive is slow
    LLAMA_LOG_INFO("%s: %s: %.2f s in %" PRId64 " gathers of %.0f rows, %.2f s waiting on reads, "
            "%.2f s of reader time (%.1f in flight on average, peak %d)\n",
            __func__, tag, p->t_gather_us/1e6, p->n_gather,
            p->n_gather > 0 ? (double) p->n_row_ask/p->n_gather : 0.0, p->t_wait_us/1e6, t_io/1e6,
            p->t_wait_us > 0 ? (double) t_io/p->t_wait_us : 0.0, p->n_inflight_max.load());

    LLAMA_LOG_INFO("%s: %s: %.0f us per request as seen by a reader, %.0f us of wall time per "
            "request, %.0f MiB/s\n", __func__, tag,
            p->n_req > 0 ? (double) t_io/p->n_req : 0.0,
            p->n_req > 0 ? (double) p->t_wait_us/p->n_req : 0.0,
            p->t_wait_us > 0 ? p->n_read*p->blk_size/1024.0/1024.0/(p->t_wait_us/1e6) : 0.0);
}
