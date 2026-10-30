// the elastic KV cache of the fork (LLAMA_KV_ELASTIC, llama_context::kv_elastic_init_fork): the cells of the cache are
// used from the front up to a capacity, and the device maps memory to the cells of the capacity only. The tensors keep
// their shape and their addresses, a reserved range holds the whole cache (ggml-cuda/fork/elastic-buffer-fork.inc);
// the memory of the cells beyond the capacity stays free for the moe-cache, which takes its slots from the free memory
// of the device. Each tensor starts on a granule boundary, so the tail of one never shares a granule with the next one

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
        size_t                  offs;  // of the tensor in the buffer
        size_t                  row;   // bytes of a cell
        size_t                  bytes; // of the tensor
        size_t                  span;  // up to the next granule boundary after it
        llama_buffer_commit_t   commit;
        llama_buffer_decommit_t decommit;
    };

    std::vector<range> ranges;

    // the cells the memory is mapped for, which an emulated cache keeps as the poisoned bound
    uint32_t n_mapped = 0;

    // LLAMA_KV_ELASTIC_POISON=1 on an emulated cache: the cells beyond the capacity hold NaN, so that a read of them
    // shows in the output; they are zeroed again when the capacity takes them, as a mapped granule is
    bool poison = false;
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
    // one stream, rows per cell (no transposed V), the own tensors of the cache. quantized types pad their tensors in
    // init_tensor, which would write into memory not mapped yet
    if (elastic_fork || n_stream != 1 || v_trans || other || ctxs_bufs.no_alloc || layers.empty()) {
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

                el->ranges.push_back({ t, c.buf, offs, t->nb[1], ggml_nbytes(t), end - offs, p.commit, p.decommit });
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
    el->n_mapped = el->ranges.empty() || el->ranges[0].commit ? 0 : get_size();

    elastic_fork = el;
    n_cap_fork   = 0;

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

bool llama_kv_cache::elastic_commit_fork(uint32_t n_cap) {
    if (!elastic_fork) {
        return false;
    }
    n_cap = std::min(n_cap, get_size());

    auto & el = *elastic_fork;

    // the poison test: a cell in use that the capacity takes only now held data a device would not have kept (the
    // writes past the mapped memory go nowhere there), and the zeroing below loses it here as well
    if (el.poison && n_cap > el.n_mapped) {
        const auto & cells = v_cells[0];
        for (uint32_t i = el.n_mapped; i < std::min(n_cap, cells.size()); ++i) {
            if (!cells.is_empty(i)) {
                LLAMA_LOG_ERROR("%s: kv_elastic: cell %u is in use past the %u mapped cells, its data is lost\n", __func__, i, el.n_mapped);
                break;
            }
        }
    }

    for (const auto & r : el.ranges) {
        const size_t need = std::min(r.bytes, (size_t) n_cap*r.row);
        if (r.commit) {
            if (!r.commit(r.buf, r.offs, need)) {
                return false;
            }
        } else if (el.poison && n_cap > el.n_mapped) {
            // the cells the capacity takes back: zeroed, as a fresh granule
            const size_t from = std::min(r.bytes, (size_t) el.n_mapped*r.row);
            if (need > from) {
                ggml_backend_tensor_memset(r.t, 0, from, need - from);
            }
        }
    }
    el.n_mapped = std::max(el.n_mapped, n_cap);

    return true;
}

uint32_t llama_kv_cache::elastic_set_capacity_fork(uint32_t n_cap) {
    if (!elastic_fork) {
        return 0;
    }

    // never below the cells in use, nor below what the shift and the padding of n_kv read
    const uint32_t n_used = v_cells[0].used_max_p1();
    n_cap = std::min(std::max({ n_cap, n_used, 256u }), get_size());
    n_cap = std::min(GGML_PAD(n_cap, 256u), get_size());

    auto & el = *elastic_fork;

    // more cells: their memory first, a capacity the device cannot map stays where it is
    if (n_cap > el.n_mapped && !elastic_commit_fork(n_cap)) {
        return n_cap_fork;
    }

    if (n_cap < el.n_mapped) {
        for (const auto & r : el.ranges) {
            const size_t keep = std::min(r.bytes, (size_t) n_cap*r.row);
            if (r.decommit) {
                // the granules whole beyond the kept cells, up to the next tensor
                r.decommit(r.buf, r.offs + keep, r.span - keep);
            } else if (el.poison && r.bytes > keep) {
                ggml_backend_tensor_memset(r.t, 0xFF, keep, r.bytes - keep);
            }
        }
        el.n_mapped = n_cap;
    }

    n_cap_fork = n_cap;
    if (v_heads[0] >= n_cap) {
        v_heads[0] = 0;
    }

    return n_cap;
}
