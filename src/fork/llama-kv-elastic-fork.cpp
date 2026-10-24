// the elastic KV cache of the fork (LLAMA_KV_ELASTIC, llama_context::kv_elastic_init_fork): the cells of the cache are
// used from the front up to a capacity, and the device maps memory to the cells of the capacity only. The tensors keep
// their shape and their addresses, a reserved range holds the whole cache (ggml-cuda/fork/elastic-buffer-fork.inc);
// the memory of the cells beyond the capacity stays free for the moe-cache, which takes its slots from the free memory
// of the device. Each tensor starts on a granule boundary, so the tail of one never shares a granule with the next one

#include "llama-kv-cache.h"

#include "llama-impl.h"

#include "ggml-backend.h"

#include <algorithm>
#include <cstdlib>

typedef ggml_backend_buffer_t (*llama_buft_alloc_elastic_t)(ggml_backend_buffer_type_t buft, size_t size);
typedef bool (*llama_buffer_commit_t)(ggml_backend_buffer_t buffer, size_t offset, size_t size);
typedef void (*llama_buffer_decommit_t)(ggml_backend_buffer_t buffer, size_t offset, size_t size);

// the VMM granule of CUDA; a coarser one only maps a little more than asked
static constexpr size_t LLAMA_KV_ELASTIC_ALIGN = 2u*1024*1024;

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

bool llama_kv_cache::elastic_init_fork(bool emulate) {
    // one stream, rows per cell (no transposed V), the own tensors of the cache. quantized types pad their tensors in
    // init_tensor, which would write into memory not mapped yet
    if (elastic_fork || n_stream != 1 || v_trans || other || hparams.no_alloc || layers.empty()) {
        return false;
    }
    for (const auto & layer : layers) {
        if ((layer.k && ggml_is_quantized(layer.k->type)) || (layer.v && ggml_is_quantized(layer.v->type))) {
            return false;
        }
    }

    struct plan {
        ggml_backend_buffer_type_t buft;
        llama_buft_alloc_elastic_t alloc;
        llama_buffer_commit_t      commit;
        llama_buffer_decommit_t    decommit;
        std::vector<std::pair<ggml_tensor *, size_t>> placed;
        size_t                     total = 0;
        ggml_backend_buffer_t      buf   = nullptr;
    };
    std::vector<plan> plans;

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

        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->view_src != nullptr) {
                continue;
            }
            p.total = GGML_PAD(p.total, LLAMA_KV_ELASTIC_ALIGN);
            p.placed.push_back({ t, p.total });
            p.total += ggml_backend_buft_get_alloc_size(p.buft, t);
        }
        p.total = GGML_PAD(p.total, LLAMA_KV_ELASTIC_ALIGN);

        plans.push_back(std::move(p));
    }

    for (auto & p : plans) {
        p.buf = p.alloc ? p.alloc(p.buft, p.total) : ggml_backend_buft_alloc_buffer(p.buft, p.total);
        if (p.buf == nullptr) {
            for (auto & q : plans) {
                if (q.buf) {
                    ggml_backend_buffer_free(q.buf);
                }
            }
            return false;
        }
    }

    auto el = std::make_shared<llama_kv_elastic_fork>();
    {
        const char * env = getenv("LLAMA_KV_ELASTIC_POISON");
        el->poison = emulate && env != nullptr && atoi(env) != 0;
    }

    for (size_t i = 0; i < plans.size(); ++i) {
        auto & p   = plans[i];
        auto & ctx = ctxs_bufs[i].first;

        char * base = (char *) ggml_backend_buffer_get_base(p.buf);

        for (size_t j = 0; j < p.placed.size(); ++j) {
            ggml_tensor * t    = p.placed[j].first;
            const size_t  offs = p.placed[j].second;
            const size_t  end  = j + 1 < p.placed.size() ? p.placed[j + 1].second : p.total;

            t->buffer = nullptr;
            t->data   = nullptr;
            GGML_ASSERT(ggml_backend_tensor_alloc(p.buf, t, base + offs) == GGML_STATUS_SUCCESS);

            el->ranges.push_back({ t, p.buf, offs, t->nb[1], ggml_nbytes(t), end - offs, p.commit, p.decommit });
        }

        // the views of the streams follow their tensors
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            if (t->view_src != nullptr) {
                t->buffer = nullptr;
                GGML_ASSERT(ggml_backend_view_init(t) == GGML_STATUS_SUCCESS);
            }
        }

        // the old buffer goes, an emulated one is cleared as the cache cleared its own
        ctxs_bufs[i].second.reset(p.buf);
        if (!p.alloc) {
            ggml_backend_buffer_clear(p.buf, 0);
        }
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
