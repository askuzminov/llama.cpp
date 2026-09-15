#include "llama-row-cache.h"

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
#include <unordered_set>
#include <vector>

namespace {

size_t row_cache_env(const char * name, size_t def) {
    const char * val = getenv(name);
    if (val == nullptr) {
        return def;
    }

    const long long parsed = atoll(val);

    return parsed > 0 ? (size_t) parsed : def;
}

} // namespace

struct llama_row_cache::impl {
    // one read request: n consecutive blocks of the file, landing in slots[i_first .. i_first + n)
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
    std::vector<std::vector<uint8_t>>        stage;   // per worker, holds one multi-block run

    std::vector<std::thread> workers;
    std::mutex               job_mutex;
    std::condition_variable  job_cv;
    std::condition_variable  done_cv;
    const std::vector<run>      * job_runs  = nullptr;
    const std::vector<uint32_t> * job_slots = nullptr;
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

    // only tracked under LLAMA_ROW_CACHE_STATS, the set costs about 48 bytes per distinct block
    bool                        track_ws = false;
    std::unordered_set<uint64_t> seen;

    impl(const std::string & path, size_t offs, ggml_type type, int64_t n_cols, int64_t n_rows,
         int n_threads);
    ~impl();

    void read_run(size_t i_worker, const run & r, const uint32_t * to_slots);
    void run_jobs(const std::vector<run> & runs, const std::vector<uint32_t> & to_slots, size_t n_blocks);
    void bench(size_t n_bench);

    void gather(const int32_t * rows, int64_t n, float * dst);
};

llama_row_cache::impl::impl(const std::string & path, size_t offs, ggml_type type,
                            int64_t n_cols, int64_t n_rows, int n_threads) :
        offs(offs), type(type), n_cols(n_cols), n_rows(n_rows) {
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
    blk_size = row_cache_env("LLAMA_ROW_CACHE_BLOCK", row_size);

    // blocks that turn out to be adjacent are read in one request, which is what makes a gather
    // over real text cheap: neighbouring tokens ask for neighbouring rows
    run_max = std::max<size_t>(row_cache_env("LLAMA_ROW_CACHE_RUN", 64), 1);

    track_ws = row_cache_env("LLAMA_ROW_CACHE_STATS", 0) > 0;
    bench_n  = row_cache_env("LLAMA_ROW_CACHE_BENCH", 0);

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
    n_threads = (int) row_cache_env("LLAMA_ROW_CACHE_THREADS", (size_t) n_threads);

    for (int i = 1; i < n_threads; i++) {
        files.emplace_back(new llama_file(path.c_str(), "rb", false));
    }

    stage.resize(n_threads);
    if (run_max > 1 || bench_n > 0) {
        for (int i = 0; i < n_threads; i++) {
            stage[i].resize(run_max*blk_size);
        }
    }

    for (int i = 0; i < n_threads; i++) {
        workers.emplace_back([this, i] {
            while (true) {
                size_t i_job = 0;
                {
                    std::unique_lock<std::mutex> lock(job_mutex);
                    job_cv.wait(lock, [this] {
                        return job_stop || (job_runs != nullptr && job_next < job_runs->size());
                    });
                    if (job_stop) {
                        return;
                    }
                    i_job = job_next++;
                }

                const size_t n_jobs = job_runs->size();
                const run &  r      = (*job_runs)[i_job];

                const int32_t in_flight = n_inflight.fetch_add(1) + 1;
                int32_t       peak      = n_inflight_max.load();
                while (in_flight > peak && !n_inflight_max.compare_exchange_weak(peak, in_flight)) {}

                const int64_t t_io_start = ggml_time_us();

                try {
                    read_run(i, r, job_slots->data() + r.i_first);
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
            "up to %zu blocks per request\n",
            __func__, ggml_row_size(type, n_cols)*n_rows/1024.0/1024.0/1024.0,
            path.c_str() + (i_name == std::string::npos ? 0 : i_name + 1),
            blk_size, n_threads, run_max);

    if (bench_n > 0) {
        bench(bench_n);
    }
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

    std::vector<uint32_t> to_slots;
    std::vector<run>      runs;

    uint64_t seed = 0x853c49e6748fea9bULL;

    for (size_t n_blk = 1; n_blk <= run_max; n_blk *= 4) {
        for (size_t depth = 1; depth <= files.size(); depth *= 2) {
            const size_t batch = std::min(depth, n_slots/n_blk);
            if (batch == 0) {
                break;
            }

            const size_t n_batches   = std::max<size_t>(n_bench/(n_blk*batch), 1);
            const size_t n_slot_base = n_slots/n_blk;

            to_slots.resize(batch*n_blk);
            runs.resize(batch);

            const int64_t t_io_before = t_io_us.load();
            const int64_t t_start     = ggml_time_us();

            n_inflight_max.store(0);

            for (size_t i = 0; i < n_batches; i++) {
                for (size_t k = 0; k < batch; k++) {
                    seed = seed*6364136223846793005ULL + 1442695040888963407ULL;

                    runs[k] = { (seed >> 16) % std::max<uint64_t>(n_file_blocks - n_blk, 1),
                                (uint32_t) n_blk, (uint32_t) (k*n_blk) };

                    // a gather lands its reads on slots scattered over the whole pool. Reusing
                    // the first few slots would keep the same pages hot and hide that cost
                    seed = seed*6364136223846793005ULL + 1442695040888963407ULL;

                    const size_t base = (size_t) ((seed >> 16) % std::max<size_t>(n_slot_base, 1))*n_blk;

                    for (size_t b = 0; b < n_blk; b++) {
                        to_slots[k*n_blk + b] = (uint32_t) (base + b);
                    }
                }

                run_jobs(runs, to_slots, batch*n_blk);
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
    {
        std::lock_guard<std::mutex> lock(job_mutex);
        job_stop = true;
    }
    job_cv.notify_all();

    for (auto & worker : workers) {
        worker.join();
    }
}

void llama_row_cache::impl::read_run(size_t i_worker, const run & r, const uint32_t * to_slots) {
    llama_file * file = files[i_worker].get();

    const size_t off  = (size_t) r.block*blk_size;
    const size_t len  = (size_t) r.n*blk_size;

    // one block goes straight into its slot, several land in the staging buffer first because
    // the slots they belong to are scattered over the pool
    const bool via_stage = r.n > 1;

    uint8_t * dst = via_stage ? stage[i_worker].data() : pool.data() + (size_t) to_slots[0]*blk_size;

    file->seek(off, SEEK_SET);

    // a read past the end of the file would throw, the tail of the last block is zero filled
    const size_t want = off < file_size ? std::min(len, file_size - off) : 0;
    if (want > 0) {
        file->read_raw_unsafe(dst, want);
    }
    if (want < len) {
        memset(dst + want, 0, len - want);
    }

    if (via_stage) {
        for (uint32_t k = 0; k < r.n; k++) {
            memcpy(pool.data() + (size_t) to_slots[k]*blk_size, dst + (size_t) k*blk_size, blk_size);
        }
    }
}

void llama_row_cache::impl::run_jobs(const std::vector<run> & runs, const std::vector<uint32_t> & to_slots,
                                     size_t n_blocks) {
    if (runs.empty()) {
        return;
    }

    const int64_t t_start = ggml_time_us();

    {
        std::lock_guard<std::mutex> lock(job_mutex);
        job_runs  = &runs;
        job_slots = &to_slots;
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
        job_slots = nullptr;
        err       = job_error;
    }

    n_read    += n_blocks;
    n_req     += runs.size();
    t_wait_us += ggml_time_us() - t_start;

    if (!err.empty()) {
        throw std::runtime_error(format("row cache: %s", err.c_str()));
    }
}

void llama_row_cache::impl::gather(const int32_t * rows, int64_t n, float * dst) {
    std::unordered_map<uint64_t, uint32_t> pinned;   // blocks this chunk holds -> slot
    std::vector<uint64_t>                  want;     // of those, the ones still to be read
    std::vector<uint32_t>                  want_slot;
    std::vector<run>                       runs;

    // the rows are walked in chunks that fit the pool, so nothing a chunk reads is overwritten
    // before its rows are copied out. The next chunk starts empty
    int64_t i = 0;
    while (i < n) {
        pinned.clear();
        want.clear();
        want_slot.clear();
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

        // in file order, so that blocks which happen to be adjacent merge into one request
        std::sort(want.begin(), want.end());

        want_slot.resize(want.size());

        for (size_t k = 0; k < want.size(); k++) {
            const uint32_t i_slot = (uint32_t) k;

            pinned[want[k]] = i_slot;
            want_slot[k]    = i_slot;

            if (runs.empty() || want[k] != runs.back().block + runs.back().n || runs.back().n >= run_max) {
                runs.push_back({ want[k], 1, (uint32_t) k });
            } else {
                runs.back().n++;
            }
        }

        run_jobs(runs, want_slot, want.size());

        if (track_ws) {
            seen.insert(want.begin(), want.end());
        }

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

llama_row_cache::llama_row_cache(const std::string & path, size_t offs, ggml_type type,
                                 int64_t n_cols, int64_t n_rows, int n_threads) :
    pimpl(new impl(path, offs, type, n_cols, n_rows, n_threads)) {}

llama_row_cache::~llama_row_cache() = default;

void llama_row_cache::get_rows(const int32_t * rows, int64_t n, float * dst) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);

    const int64_t t_start = ggml_time_us();

    pimpl->gather(rows, n, dst);

    pimpl->n_gather++;
    pimpl->n_row_ask   += n;
    pimpl->t_gather_us += ggml_time_us() - t_start;
}

size_t llama_row_cache::block_size() const {
    return pimpl->blk_size;
}

void llama_row_cache::print_stats(const char * tag) const {
    const impl *  p     = pimpl.get();
    const int64_t n_ask = p->n_hit + p->n_miss;
    if (n_ask == 0) {
        return;
    }

    LLAMA_LOG_INFO("%s: %s: %.1f%% of %" PRId64 " block lookups hit, %" PRId64 " blocks read "
            "in %" PRId64 " requests (%.2f GiB, %.0f B each)\n",
            __func__, tag, 100.0*p->n_hit/n_ask, n_ask, p->n_read, p->n_req,
            p->n_read*p->blk_size/1024.0/1024.0/1024.0,
            p->n_req > 0 ? (double) p->n_read*p->blk_size/p->n_req : 0.0);

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

    if (p->track_ws) {
        const size_t ws = p->seen.size()*p->blk_size;
        LLAMA_LOG_INFO("%s: %s: %zu distinct blocks touched (%.2f GiB), a cache of that size would "
                "hold the whole working set\n", __func__, tag, p->seen.size(), ws/1024.0/1024.0/1024.0);
    }
}
