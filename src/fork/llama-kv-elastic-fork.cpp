// the elastic KV cache of the fork (LLAMA_KV_ELASTIC, llama_context::kv_elastic_init_fork): the cells of each stream are
// used from the front up to its capacity, and the device maps memory to the cells of the capacities only. The tensors
// keep their shape and their addresses, a reserved range holds the whole cache (ggml-cuda/fork/elastic-buffer-fork.inc);
// the memory of the cells beyond the capacity stays free for the moe-cache, which takes its slots from the free memory
// of the device. Each tensor starts on a granule boundary, so the tail of one never shares a granule with the next one.
// Several streams (a stream per slot without --kv-unified, 03.11): the streams of a tensor lie one after the other
// (nb[2]), each maps the front of its own part; a granule across the border of two streams stays mapped once mapped
// (the commit rounds out, the decommit in). The streams of a ubatch read the same n_kv cells (get_n_kv): past the
// capacity of a stream they are empty cells, masked, which a device that reads unmapped memory as zeros (a Vulkan sparse
// buffer with residencyNonResidentStrict) may leave unmapped; elsewhere (CUDA VMM faults there) find_slot maps them

#include "llama-kv-cache.h"

#include "llama-cparams.h"
#include "llama-impl.h"
#include "llama-fork.h"
#include "llama-model.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>

typedef ggml_backend_buffer_t (*llama_buft_alloc_elastic_t)(ggml_backend_buffer_type_t buft, size_t size);
typedef bool (*llama_buffer_commit_t)(ggml_backend_buffer_t buffer, size_t offset, size_t size);
typedef void (*llama_buffer_decommit_t)(ggml_backend_buffer_t buffer, size_t offset, size_t size);
typedef bool (*llama_buffer_zero_reads_t)(ggml_backend_buffer_t buffer);

// the VMM granule of CUDA; a coarser one only maps a little more than asked
static constexpr size_t LLAMA_KV_ELASTIC_ALIGN = 2u*1024*1024;

extern "C" {
// ggml-backend-impl.h (exported from ggml-base, not in the public headers): one buffer over several, which frees them
GGML_API ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers);
}

struct llama_kv_elastic_fork {
    struct range {
        ggml_tensor *           t;
        ggml_backend_buffer_t   buf;
        size_t                  offs;   // of the tensor in the buffer
        size_t                  row;    // bytes of a cell
        size_t                  stride; // bytes of a stream (nb[2]; the whole tensor with one)
        size_t                  bytes;  // of the tensor
        size_t                  span;   // up to the next granule boundary after it
        llama_buffer_commit_t   commit;
        llama_buffer_decommit_t decommit;
    };

    std::vector<range> ranges;

    // per stream: the cells the memory is mapped for, which an emulated cache keeps as the poisoned bound
    std::vector<uint32_t> n_mapped;

    // LLAMA_KV_ELASTIC_POISON=1 on an emulated cache: the cells beyond the capacity hold NaN, so that a read of them
    // shows in the output; they are zeroed again when the capacity takes them, as a mapped granule is. a cache of
    // several streams whose reads past a capacity give zeros holds zeros there instead (those reads are expected)
    bool poison = false;

    // reads past the mapped cells of a stream give zeros: the device (llama_buffer_zero_reads_t), or an emulated cache
    // (LLAMA_KV_ELASTIC_ZERO_READS, 1 by default; 0 tests the other way)
    bool zero_reads = false;

    // the value the cells past a capacity hold in an emulated cache, -1: left as they are
    int fill(uint32_t n_stream) const {
        return poison ? (n_stream == 1 || !zero_reads ? 0xFF : 0x00) : -1;
    }
};

int llama_kv_elastic_mode_fork() {
    static const int mode = (int) llama_fork_env_int("LLAMA_KV_ELASTIC", -1);
    return mode;
}

// a device of the model whose backend gives elastic buffers (CUDA, Vulkan); whether the device itself takes them (VMM,
// sparse buffers) shows only when the cache asks for one
static bool llama_kv_elastic_device_fork(const llama_model & model) {
    for (const auto & d : model.devices) {
        if (d.is_meta || d.dev == nullptr) {
            continue;
        }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(d.dev);
        if (reg != nullptr && ggml_backend_reg_get_proc_address(reg, "ggml_backend_buft_alloc_elastic_fork") != nullptr) {
            return true;
        }
    }
    return false;
}

bool llama_kv_elastic_no_cache_fork(const llama_model & model, const llama_cparams & cparams) {
    return llama_kv_elastic_mode_fork() < 0 && cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
           cparams.n_moe_cache == 0 && cparams.moe_cache_size == 0 && llama_kv_elastic_device_fork(model);
}

bool llama_kv_elastic_defer_fork(const llama_model & model, const llama_cparams & cparams) {
    return cparams.ctx_type == LLAMA_CONTEXT_TYPE_DEFAULT &&
           (llama_kv_elastic_mode_fork() > 0 || llama_kv_elastic_no_cache_fork(model, cparams));
}

static thread_local bool llama_kv_defer_fork = false;

llama_kv_defer_scope_fork::llama_kv_defer_scope_fork(bool on) : prev(llama_kv_defer_fork) {
    llama_kv_defer_fork = on;
}

llama_kv_defer_scope_fork::~llama_kv_defer_scope_fork() {
    llama_kv_defer_fork = prev;
}

bool llama_kv_defer_alloc_fork() {
    return llama_kv_defer_fork;
}

void llama_kv_cache::alloc_deferred_fork() {
    if (ctxs_bufs.no_alloc) {
        return;
    }
    for (auto & [ctx, buf] : ctxs_bufs) {
        // the buffer of size 0 of the constructor (an elastic cache has replaced it)
        if (ggml_backend_buffer_get_size(buf.get()) != 0) {
            continue;
        }
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            t->buffer = nullptr;
            t->data   = nullptr;
        }
        ggml_backend_buffer_t res = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft);
        if (!res) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }
        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(res), ggml_backend_buffer_get_size(res)/1024.0/1024.0);
        ggml_backend_buffer_clear(res, 0);
        buf.reset(res);
    }
}

bool llama_kv_cache::elastic_init_fork(bool emulate) {
    // rows per cell (no transposed V), the own tensors of the cache. quantized types pad their tensors in init_tensor,
    // which would write into memory not mapped yet
    if (elastic_fork || v_trans || other || ctxs_bufs.no_alloc || layers.empty()) {
        return false;
    }
    for (const auto & layer : layers) {
        if ((layer.k && ggml_is_quantized(layer.k->type)) || (layer.v && ggml_is_quantized(layer.v->type))) {
            return false;
        }
    }

    // the tensors of a buffer type in chunks of at most its largest buffer (Vulkan: one allocation, 2 GiB on the 395),
    // a chunk per elastic buffer and one buffer over the chunks for the cache
    struct chunk {
        std::vector<std::pair<ggml_tensor *, size_t>> placed;
        size_t                total = 0;
        ggml_backend_buffer_t buf   = nullptr;
    };
    struct plan {
        ggml_backend_buffer_type_t buft;
        llama_buft_alloc_elastic_t alloc;
        llama_buffer_commit_t      commit;
        llama_buffer_decommit_t    decommit;
        std::vector<chunk>         chunks;
    };
    std::vector<plan> plans;

    auto free_all = [&plans]() {
        for (auto & q : plans) {
            for (auto & c : q.chunks) {
                if (c.buf) {
                    ggml_backend_buffer_free(c.buf);
                    c.buf = nullptr;
                }
            }
        }
    };

    // all buffers or none: a cache half elastic would map memory for some tensors only
    for (auto & [ctx, buf] : ctxs_bufs) {
        plan p = {};
        p.buft = ggml_backend_buffer_get_type(buf.get());

        ggml_backend_dev_t dev = ggml_backend_buft_get_device(p.buft);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            p.alloc    = (llama_buft_alloc_elastic_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buft_alloc_elastic_fork");
            p.commit   = (llama_buffer_commit_t)      ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_commit_fork");
            p.decommit = (llama_buffer_decommit_t)    ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_decommit_fork");
        }
        if (!p.alloc || !p.commit || !p.decommit) {
            if (!emulate) {
                return false;
            }
            p.alloc    = nullptr;
            p.commit   = nullptr;
            p.decommit = nullptr;
        }

        // whole granules: a chunk padded to its granule stays within the largest buffer
        const size_t max_size = ggml_backend_buft_get_max_size(p.buft)/LLAMA_KV_ELASTIC_ALIGN*LLAMA_KV_ELASTIC_ALIGN;

        p.chunks.emplace_back();
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->view_src != nullptr) {
                continue;
            }
            const size_t bytes = ggml_backend_buft_get_alloc_size(p.buft, t);
            if (bytes > max_size) {
                return false;
            }
            size_t offs = GGML_PAD(p.chunks.back().total, LLAMA_KV_ELASTIC_ALIGN);
            if (offs + bytes > max_size && !p.chunks.back().placed.empty()) {
                p.chunks.emplace_back();
                offs = 0;
            }
            p.chunks.back().placed.push_back({ t, offs });
            p.chunks.back().total = offs + bytes;
        }
        for (auto & c : p.chunks) {
            c.total = std::max<size_t>(GGML_PAD(c.total, LLAMA_KV_ELASTIC_ALIGN), LLAMA_KV_ELASTIC_ALIGN);
        }

        plans.push_back(std::move(p));
    }

    // the first range decides how the capacity counts as mapped, so the cache is all elastic or all emulated: a
    // backend without elastic buffers (the CPU part of a partial offload) makes the others emulated too
    if (emulate && std::any_of(plans.begin(), plans.end(), [](const plan & p) { return p.alloc == nullptr; })) {
        for (auto & p : plans) {
            p.alloc    = nullptr;
            p.commit   = nullptr;
            p.decommit = nullptr;
        }
    }

    auto alloc_all = [&plans]() {
        for (auto & p : plans) {
            for (auto & c : p.chunks) {
                c.buf = p.alloc ? p.alloc(p.buft, c.total) : ggml_backend_buft_alloc_buffer(p.buft, c.total);
                if (c.buf == nullptr) {
                    return false;
                }
            }
        }
        return true;
    };

    // a backend with elastic buffers whose device gives none (Vulkan without sparse buffers: MoltenVK) emulates the
    // whole cache, as a backend without them
    if (!alloc_all()) {
        free_all();
        const bool elastic = std::any_of(plans.begin(), plans.end(), [](const plan & p) { return p.alloc != nullptr; });
        if (!emulate || !elastic) {
            return false;
        }
        for (auto & p : plans) {
            p.alloc    = nullptr;
            p.commit   = nullptr;
            p.decommit = nullptr;
        }
        if (!alloc_all()) {
            free_all();
            return false;
        }
        LLAMA_LOG_INFO("%s: kv_elastic: the device gives no elastic buffer, emulated in ordinary buffers\n", __func__);
    }

    auto el = std::make_shared<llama_kv_elastic_fork>();
    el->poison = emulate && llama_fork_env_int("LLAMA_KV_ELASTIC_POISON", 0) != 0;

    // reads past a capacity: every buffer of the cache must give zeros there
    el->zero_reads = true;
    for (const auto & p : plans) {
        llama_buffer_zero_reads_t zero = nullptr;
        if (p.alloc) {
            ggml_backend_dev_t dev = ggml_backend_buft_get_device(p.buft);
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            zero = reg ? (llama_buffer_zero_reads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_buffer_elastic_zero_reads_fork") : nullptr;
        }
        for (const auto & c : p.chunks) {
            el->zero_reads = el->zero_reads &&
                (p.alloc ? zero != nullptr && zero(c.buf) : llama_fork_env_int("LLAMA_KV_ELASTIC_ZERO_READS", 1) != 0);
        }
    }

    for (size_t i = 0; i < plans.size(); ++i) {
        auto & p   = plans[i];
        auto & ctx = ctxs_bufs.bufs[i].first;

        for (auto & c : p.chunks) {
            char * base = (char *) ggml_backend_buffer_get_base(c.buf);

            for (size_t j = 0; j < c.placed.size(); ++j) {
                ggml_tensor * t    = c.placed[j].first;
                const size_t  offs = c.placed[j].second;
                const size_t  end  = j + 1 < c.placed.size() ? c.placed[j + 1].second : c.total;

                t->buffer = nullptr;
                t->data   = nullptr;
                GGML_ASSERT(ggml_backend_tensor_alloc(c.buf, t, base + offs) == GGML_STATUS_SUCCESS);

                el->ranges.push_back({ t, c.buf, offs, t->nb[1], t->nb[2], ggml_nbytes(t), end - offs, p.commit, p.decommit });
            }
        }

        // the views of the streams follow their tensors
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->view_src != nullptr) {
                t->buffer = nullptr;
                GGML_ASSERT(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS);
            }
        }

        // an emulated chunk is cleared as the cache cleared its own buffer
        std::vector<ggml_backend_buffer_t> bufs;
        for (auto & c : p.chunks) {
            if (!p.alloc) {
                ggml_backend_buffer_clear(c.buf, 0);
            }
            bufs.push_back(c.buf);
        }

        // the old buffer goes, the new one owns its chunks
        ctxs_bufs.bufs[i].second.reset(bufs.size() == 1 ? bufs[0] : ggml_backend_multi_buffer_alloc_buffer(bufs.data(), bufs.size()));
    }

    // an emulated cache has its memory, all of it counts as mapped until the first capacity is set
    el->n_mapped.assign(n_stream, el->ranges.empty() || el->ranges[0].commit ? 0 : get_size());

    elastic_fork = el;
    v_cap_fork.assign(n_stream, 0);

    if (n_stream > 1) {
        LLAMA_LOG_INFO("%s: kv_elastic: %u streams, each maps its own cells; %s\n", __func__, n_stream, el->zero_reads ?
                "the cells a ubatch reads past them read as zeros (unmapped)" :
                "the streams of a ubatch map the cells it reads (the device does not read unmapped memory as zeros)");
    }

    return true;
}

size_t llama_kv_cache::elastic_cell_bytes_fork() const {
    size_t res = 0;
    if (elastic_fork) {
        for (const auto & r : elastic_fork->ranges) {
            res += r.row;
        }
    }
    return res;
}

bool llama_kv_cache::elastic_commit_fork(uint32_t strm, uint32_t n_cap) {
    if (!elastic_fork || strm >= n_stream) {
        return false;
    }
    n_cap = std::min(n_cap, get_size());

    auto &     el     = *elastic_fork;
    uint32_t & mapped = el.n_mapped[strm];

    // the poison test: a cell in use that the capacity takes only now held data a device would not have kept (the
    // writes past the mapped memory go nowhere there), and the zeroing below loses it here as well
    if (el.poison && n_cap > mapped) {
        const auto & cells = v_cells[strm];
        for (uint32_t i = mapped; i < std::min(n_cap, cells.size()); ++i) {
            if (!cells.is_empty(i)) {
                LLAMA_LOG_ERROR("%s: kv_elastic: cell %u of stream %u is in use past the %u mapped cells, its data is lost\n",
                        __func__, i, strm, mapped);
                break;
            }
        }
    }

    for (const auto & r : el.ranges) {
        const size_t need = std::min(r.stride, (size_t) n_cap*r.row);
        if (r.commit) {
            if (!r.commit(r.buf, r.offs + strm*r.stride, need)) {
                return false;
            }
        } else if (el.poison && n_cap > mapped) {
            // the cells the capacity takes back: zeroed, as a fresh granule
            const size_t from = std::min(r.stride, (size_t) mapped*r.row);
            if (need > from) {
                ggml_backend_tensor_memset(r.t, 0, strm*r.stride + from, need - from);
            }
        }
    }
    mapped = std::max(mapped, n_cap);

    return true;
}

uint32_t llama_kv_cache::elastic_set_capacity_fork(uint32_t strm, uint32_t n_cap) {
    if (!elastic_fork || strm >= n_stream) {
        return 0;
    }

    // never below the cells in use, nor below what the shift and the padding of n_kv read
    const uint32_t n_used = v_cells[strm].used_max_p1();
    n_cap = std::min(std::max({ n_cap, n_used, 256u }), get_size());
    n_cap = std::min(GGML_PAD(n_cap, 256u), get_size());

    auto &     el     = *elastic_fork;
    uint32_t & mapped = el.n_mapped[strm];

    // more cells: their memory first, a capacity the device cannot map stays where it is
    if (n_cap > mapped && !elastic_commit_fork(strm, n_cap)) {
        return v_cap_fork[strm];
    }

    if (n_cap < mapped) {
        const int fill = el.fill(n_stream);
        for (const auto & r : el.ranges) {
            const size_t keep = std::min(r.stride, (size_t) n_cap*r.row);
            if (r.decommit) {
                // the granules whole beyond the kept cells, up to the next stream (the last one: up to the next tensor)
                const size_t from = r.offs + strm*r.stride + keep;
                const size_t to   = strm + 1 < n_stream ? r.offs + (strm + 1)*r.stride : r.offs + r.span;
                r.decommit(r.buf, from, to - from);
            } else if (fill >= 0 && r.stride > keep) {
                ggml_backend_tensor_memset(r.t, (uint8_t) fill, strm*r.stride + keep, r.stride - keep);
            }
        }
        mapped = n_cap;
    }

    v_cap_fork[strm] = n_cap;
    if (v_heads[strm] >= n_cap) {
        v_heads[strm] = 0;
    }

    return n_cap;
}

bool llama_kv_cache::elastic_zero_reads_fork() const {
    return elastic_fork && elastic_fork->zero_reads;
}

uint32_t llama_kv_cache::elastic_cells_fork(uint32_t strm) const {
    const uint32_t cap = elastic_capacity_fork(strm);
    return cap > 0 ? std::min(cap, get_size()) : get_size();
}

uint32_t llama_kv_cache::elastic_shift_base_fork(uint32_t strm) const {
    if (!elastic_fork) {
        return strm*get_size();
    }
    uint32_t base = 0;
    for (uint32_t s = 0; s < strm; ++s) {
        base += elastic_cells_fork(s);
    }
    return base;
}

bool llama_kv_cache::elastic_cover_fork(const slot_info & sinfo) const {
    if (!elastic_fork || elastic_fork->zero_reads || sinfo.n_stream() < 2) {
        return true;
    }

    // the n_kv of the ubatch once its cells are placed, as get_n_kv will count it
    uint32_t n_kv = 0;
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        uint32_t m = v_cells[sinfo.strm[s]].used_max_p1();
        for (const uint32_t idx : sinfo.idxs[s]) {
            m = std::max(m, idx + 1);
        }
        n_kv = std::max(n_kv, std::min(get_size(), std::max(256u, GGML_PAD(m, 256u))));
    }

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const uint32_t strm = sinfo.strm[s];
        const uint32_t cap  = elastic_capacity_fork(strm);
        if (cap > 0 && cap < n_kv && !(elastic_grow_fork && elastic_grow_fork(strm, n_kv) && elastic_capacity_fork(strm) >= n_kv)) {
            LLAMA_LOG_ERROR("%s: kv_elastic: stream %u maps %u cells, the ubatch reads %u\n", __func__, strm,
                    elastic_capacity_fork(strm), n_kv);
            return false;
        }
    }
    return true;
}

void llama_kv_cache::elastic_copy_stream_fork(uint32_t ssrc, uint32_t sdst) const {
    // the cells the destination holds now (seq_cp put the ones of the source there), in the memory of both
    const uint32_t n = std::min({ v_cells[sdst].used_max_p1(), elastic_cells_fork(ssrc), elastic_cells_fork(sdst) });
    if (n == 0) {
        return;
    }

    ggml_init_params params = { 4*ggml_tensor_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    GGML_ASSERT(ctx != nullptr);

    auto rows = [&](ggml_tensor * t, uint32_t strm) {
        ggml_tensor * v = ggml_view_2d(ctx, t, t->ne[0], n, t->nb[1], strm*t->nb[2]);
        v->buffer = nullptr;
        GGML_ASSERT(ggml_backend_view_init(v) == GGML_STATUS_SUCCESS);
        return v;
    };

    for (const auto & layer : layers) {
        if (layer.k) {
            ggml_backend_tensor_copy(rows(layer.k, ssrc), rows(layer.k, sdst));
        }
        if (layer.v) {
            ggml_backend_tensor_copy(rows(layer.v, ssrc), rows(layer.v, sdst));
        }
        ggml_reset(ctx);
    }

    ggml_free(ctx);
}
