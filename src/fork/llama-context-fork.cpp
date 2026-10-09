// the fork's members of llama_context: the phase memory (--phase-mem), the ubatch split it sets, the graph cache
// and the delta state of a sequence. llama-context.h declares them in its fork block

#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "llama-ext.h"
#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

// phase_mem: device memory that the prompt compute buffers leave free
static const size_t PHASE_MEM_MARGIN = 1024ull*1024*1024;

uint32_t llama_context::n_ubatch_split() const {
    return phase_n_tokens > 0 ? std::min(phase_n_tokens, cparams.n_ubatch) : cparams.n_ubatch;
}

void llama_context::sched_free() {
    synchronize();

    const size_t max_nodes = graph_max_nodes(std::min(cparams.n_ctx, cparams.n_ubatch));

    gf_res_prev.clear();
    gf_res_reserve.reset(new llm_graph_result(max_nodes));
    gf_res_prev_active = nullptr;

    // the old scheduler frees its buffers, the new one allocates them only in a reserve
    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));
    ggml_backend_sched_set_copy_callback(sched.get(), sched_copy_experts, this);

    sched_need_reserve = true;
}

uint32_t llama_context::phase_kv_end(uint32_t n_tokens, uint32_t n_pad) const {
    return memory ? std::min(cparams.n_ctx, GGML_PAD(memory->get_n_kv_used() + n_tokens, 256) + n_pad) : 0;
}

int llama_context::phase_size(uint32_t n_ub, uint32_t n_tokens, uint32_t n_pad, std::map<ggml_backend_dev_t, size_t> & sizes) {
    if (!memory) {
        return 0;
    }

    memory->set_n_kv_full(phase_kv_end(n_tokens, n_pad));
    const auto mctx = memory->init_full();
    memory->set_n_kv_full(0);

    std::vector<size_t> s(backend_ptrs.size(), 0);

    // the scheduler has no buffers, so the query does not change them
    graph_reserve(n_ub, cparams.n_seq_max, std::min(n_ub, cparams.n_outputs_max), mctx.get(), true, s.data());

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        sizes[ggml_backend_get_device(backend_ptrs[i])] += s[i];
    }

    return ggml_backend_sched_get_n_splits(sched.get());
}

void llama_context::phase_update(uint32_t n_tokens) {
    // a non-causal batch must be one ubatch, it uses the static layout
    if (!cparams.causal_attn) {
        if (phase != LLAMA_PHASE_NONE) {
            if (moe_cache_fork) {
                moe_cache_fork->release();
            }
            phase_cache        = false;
            phase              = LLAMA_PHASE_NONE;
            phase_n_tokens     = 0;
            phase_n_kv         = 0;
            sched_need_reserve = true;
        }
        return;
    }

    // a short prompt runs in the generation layout, in ubatches of its workspace: the moe-cache stays, and the host
    // experts are not streamed to the device for a few tokens. on the 3090 (06.10) an 82-token prompt took 1.6 s so
    // (about 0.53 s per 31-token ubatch) against 2.7 s in the prompt layout and 0.73 s to fill the moe-cache again
    // after it; the prompt layout costs about 2.5 s + 1.8 ms per token, so the two meet near 200 tokens.
    // LLAMA_PHASE_SMALL_PROMPT sets the limit, 0 turns it off
    static const uint32_t n_small = [] {
        const char * s = getenv("LLAMA_PHASE_SMALL_PROMPT");
        return s ? (uint32_t) std::max(0, atoi(s)) : 192u;
    }();
    const bool small = n_tokens > phase_n_gen && n_tokens <= n_small;
    const bool gen   = n_tokens <= phase_n_gen || small;

    if (gen && (phase != LLAMA_PHASE_GEN || (!small && n_tokens > phase_n_reserve))) {
        if (!small && n_tokens > phase_n_reserve) {
            phase_n_reserve = n_tokens;
        }
        if (moe_cache_fork) {
            moe_cache_fork->release();
        }
        phase_cache = false;
        phase_gen();
    }

    if (!gen && (phase != LLAMA_PHASE_PROMPT || phase_kv_end(n_tokens) > phase_n_kv)) {
        phase_prompt(n_tokens);
    }

    // as in the static layout: the warmup of common and the first batch of a prompt start at position 0, the draft contexts
    // are created after the warmup
    if (gen && moe_cache_fork && !phase_cache && !cparams.warmup && balloc->get_batch().pos[0] > 0) {
        phase_cache = true;

        if (!moe_cache_fork->is_init()) {
            moe_cache_fork->init(sched.get());
        }
        moe_cache_fork->alloc();

        if (moe_stats && moe_cache_fork->ready()) {
            moe_stats->set_cache(moe_cache_fork->get_n_slots(), moe_cache_fork->get_max_ins(), moe_cache_fork->uses_frequency_admission());
        }

        // the graphs keep the tensors of the old slots
        gf_res_prev_reset();
        gf_res_prev_active = nullptr;
    }
}

void llama_context::phase_gen() {
    const int64_t t_start_us = ggml_time_us();

    // the followers first, so that the moe-cache gets all the memory that is left
    for (auto * f : phase_followers) {
        f->phase_n_reserve     = std::max(f->phase_n_reserve, std::min(phase_n_reserve, f->phase_n_gen));
        f->phase              = LLAMA_PHASE_GEN;
        f->phase_n_tokens     = f->phase_n_reserve;
        f->phase_n_kv         = 0;
        f->sched_need_reserve = true;
        f->sched_reserve();
    }

    phase              = LLAMA_PHASE_GEN;
    phase_n_tokens     = phase_n_reserve;
    phase_n_kv         = 0;
    sched_need_reserve = true;
    sched_reserve();

    LLAMA_LOG_INFO("%s: phase_mem: generation, ubatch %u, %.2f s\n", __func__, phase_n_tokens, 1e-6*(ggml_time_us() - t_start_us));
}

void llama_context::phase_prompt(uint32_t n_tokens) {
    const int64_t t_start_us = ggml_time_us();

    if (moe_cache_fork) {
        moe_cache_fork->release();
    }
    phase_cache = false;

    // the size queries need schedulers without buffers, and the free memory must not count the old buffers
    for (auto * f : phase_followers) {
        f->sched_free();
    }
    sched_free();

    const int64_t t_free_us = ggml_time_us();

    // free memory of each device that runs the graphs
    std::map<ggml_backend_dev_t, size_t> budget;
    size_t free_all = 0;

    auto add_devs = [&](const llama_context * c) {
        for (auto * b : c->backend_ptrs) {
            ggml_backend_dev_t dev = ggml_backend_get_device(b);
            const auto type = ggml_backend_dev_type(dev);
            if ((type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) || budget.count(dev) > 0) {
                continue;
            }
            size_t free  = 0;
            size_t total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            if (total > 0) {
                budget[dev] = free > PHASE_MEM_MARGIN ? free - PHASE_MEM_MARGIN : 0;
                free_all += free;
            }
        }
    };
    add_devs(this);
    for (auto * f : phase_followers) {
        add_devs(f);
    }

    const uint32_t n_step = 256;
    const uint32_t n_max  = std::min(cparams.n_ctx, cparams.n_ubatch);

    // KV headroom of the plan: the next batches of a long prompt fit it and keep the buffers, without a new plan per
    // batch (on the 3090, 09.10: 0.19 s at 25K cells to 0.48 s at 120K per 4096-token batch, about 10 s of a 118K
    // prompt). taken only when it leaves the ubatch as large. LLAMA_PHASE_KV_STEP=<cells>, 0 (default): no headroom
    static const uint32_t kv_step = [] {
        const char * s = getenv("LLAMA_PHASE_KV_STEP");
        return s ? (uint32_t) std::max(0, atoi(s)) : 0u;
    }();
    uint32_t n_pad = kv_step;

    int n_query = 0;

    int64_t e_last = 0; // bytes over the budget on the worst device
    int     s_last = 0; // graph splits

    const char * func = __func__;

    auto query = [&](uint32_t n_ub) {
        std::map<ggml_backend_dev_t, size_t> sizes;
        s_last = phase_size(n_ub, n_tokens, n_pad, sizes);
        for (auto * f : phase_followers) {
            s_last += f->phase_size(n_ub, n_tokens, n_pad, sizes);
        }
        e_last = std::numeric_limits<int64_t>::min();
        for (const auto & [dev, b] : budget) {
            e_last = std::max(e_last, (int64_t) sizes[dev] - (int64_t) b);
        }
        n_query++;
        LLAMA_LOG_DEBUG("%s: phase_mem: ubatch %u, %.0f MiB over the free memory, %d graph splits\n", func, n_ub, e_last/1024.0/1024.0, s_last);
    };

    // a device can reject an op with a large tensor (Vulkan buffer limits), then the op runs on the CPU in an extra split
    int s_max = std::numeric_limits<int>::max();
    if (n_max > n_step) {
        query(n_step);
        s_max = s_last;
    }

    auto fits = [&](uint32_t n_ub) {
        query(n_ub);
        return e_last <= 0 && s_last <= s_max;
    };

    uint32_t n_lo = n_max;
    bool     ok   = fits(n_max);

    if (!ok && n_pad > 0) {
        n_pad = 0;
        ok    = fits(n_max);
    }

    if (!ok && n_max > n_step) {
        uint32_t n_hi = n_max;

        // halve until a size fits
        while (!ok && n_lo > n_step) {
            n_hi = n_lo;
            n_lo = std::max(n_step, n_lo/2/n_step*n_step);
            ok   = fits(n_lo);
        }

        // then bisect between the size that fits and the size that does not
        while (ok && n_hi - n_lo > n_step) {
            const uint32_t n = (n_lo + n_hi)/2/n_step*n_step;
            if (n <= n_lo) {
                break;
            }
            if (fits(n)) {
                n_lo = n;
            } else {
                n_hi = n;
            }
        }
    }

    if (!ok) {
        LLAMA_LOG_WARN("%s: phase_mem: a %u-token ubatch does not fit: %.0f MiB over the free memory, %d graph splits of %d\n",
                __func__, n_lo, e_last/1024.0/1024.0, s_last, s_max);
    }

    const int64_t t_query_us = ggml_time_us();

    // the allocation can fail although the sizes fit, e.g. when a driver refuses a large buffer
    while (true) {
        try {
            for (auto * f : phase_followers) {
                f->phase          = LLAMA_PHASE_PROMPT;
                f->phase_n_tokens = n_lo;
                f->phase_n_kv     = f->phase_kv_end(n_tokens, n_pad);
                f->sched_reserve();
            }

            phase          = LLAMA_PHASE_PROMPT;
            phase_n_tokens = n_lo;
            phase_n_kv     = phase_kv_end(n_tokens, n_pad);
            sched_reserve();
            break;
        } catch (const std::exception & err) {
            if (n_lo <= n_step) {
                throw;
            }
            LLAMA_LOG_WARN("%s: phase_mem: %s at ubatch %u, trying a smaller one\n", __func__, err.what(), n_lo);
            n_lo = std::max(n_step, n_lo/2/n_step*n_step);
            for (auto * f : phase_followers) {
                f->sched_free();
            }
            sched_free();
        }
    }

    // compute buffers on the devices
    auto dev_size = [](const llama_context * c) {
        size_t res = 0;
        for (auto * b : c->backend_ptrs) {
            const auto type = ggml_backend_dev_type(ggml_backend_get_device(b));
            if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                res += ggml_backend_sched_get_buffer_size(c->sched.get(), b);
            }
        }
        return res;
    };
    size_t size_f = 0;
    for (auto * f : phase_followers) {
        size_f += dev_size(f);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: phase_mem: prompt, batch %u tokens, KV cells %u, ubatch %u of %u, compute %.0f + %.0f MiB (drafts), %.0f MiB were free, %d queries, %.2f s (free %.2f, queries %.2f, reserve %.2f)\n",
            __func__, n_tokens, phase_n_kv, n_lo, n_max, dev_size(this)/1024.0/1024.0, size_f/1024.0/1024.0, free_all/1024.0/1024.0,
            n_query, 1e-6*(t_end_us - t_start_us), 1e-6*(t_free_us - t_start_us), 1e-6*(t_query_us - t_free_us), 1e-6*(t_end_us - t_query_us));
}

llm_graph_result * llama_context::get_gf_res_reuse(const llm_graph_params & gparams) {
    if (graph_reuse_disable) {
        return nullptr;
    }

    for (size_t i = 0; i < gf_res_prev.size(); ++i) {
        auto & entry = gf_res_prev[i];
        auto * res   = entry.res.get();

        const bool active = res == gf_res_prev_active;
        if (!active && !entry.snapshot) {
            continue;
        }
        if (!res->can_reuse(gparams)) {
            continue;
        }
        if (!active) {
            if (!ggml_backend_sched_snapshot_restore(sched.get(), entry.snapshot.get())) {
                // the compute buffers were reallocated, the tensors of the graph are not valid
                entry.snapshot.reset();
                res->reset();
                continue;
            }
            gf_res_prev_active = res;
        }

        std::rotate(gf_res_prev.begin(), gf_res_prev.begin() + i, gf_res_prev.begin() + i + 1);

        return res;
    }

    return nullptr;
}

void llama_context::gf_res_prev_reset() {
    for (auto & entry : gf_res_prev) {
        entry.snapshot.reset();
        entry.res->reset();
    }
}

size_t llama_context::state_seq_get_delta(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos base_pos) {
    if (memory) {
        return memory->state_write_delta(io, seq_id, flags, base_pos);
    }

    // no memory, no delta: 0 makes the caller take a full checkpoint
    return 0;
}

int32_t llama_context::state_seq_apply_delta(
        const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags, llama_pos base_pos) {
    // Create a temporary io_read wrapper for the delta data
    struct memory_buffer_io : llama_io_read_i {
        const uint8_t * data;
        size_t pos = 0;
        size_t len;

        memory_buffer_io(const uint8_t * src, size_t size) : data(src), len(size) {}

        void read(void * dst, size_t n) override {
            if (n > len - pos) {
                throw std::runtime_error("buffer overflow in memory_buffer_io::read");
            }
            memcpy(dst, data + pos, n);
            pos += n;
        }

        void read_tensor(ggml_tensor * tensor, size_t offset, size_t size_read) override {
            if (size_read > len - pos) {
                throw std::runtime_error("buffer overflow in memory_buffer_io::read_tensor");
            }
            ggml_backend_tensor_set(tensor, data + pos, offset, size_read);
            pos += size_read;
        }

        size_t n_bytes() override {
            return pos;
        }
    };

    memory_buffer_io io(src, size);

    if (memory) {
        bool success = memory->state_read_delta(io, seq_id, flags, base_pos);
        return success && io.n_bytes() == size ? 0 : -1;
    }

    return -1;
}
