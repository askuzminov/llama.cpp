#include "llama-moecache.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <functional>

// the report is opt-in, so it goes out at warn level to survive the default log verbosity
#define MOE_LOG "moe_cache_stats"

// cache sizes to simulate, in slots per layer
static const int32_t MOE_STATS_SLOTS[] = { 8, 16, 32, 48, 64, 96, 128, 192, 256 };

// max expert uploads per layer per decode step
static const int32_t MOE_STATS_INS[] = { 1, 2, 4, 8 };

// tokens replayed from the end of a prompt step, the whole prompt is too slow to replay
static const int64_t MOE_STATS_PREFILL_TAIL = 128;

// assumed bandwidths used only for the net column of the report
static const double MOE_STATS_BW_HOST = 35.0; // GB/s, host RAM read
static const double MOE_STATS_BW_LINK = 20.0; // GB/s, host to device upload

static int moe_stats_period() {
    const char * s = getenv("LLAMA_MOE_CACHE_STATS");
    if (s == nullptr || s[0] == '\0' || s[0] == '0') {
        return 0;
    }
    const int v = atoi(s);
    return v > 1 ? v : 2048;
}

bool llama_moe_stats_enabled() {
    static const bool enabled = moe_stats_period() > 0;
    return enabled;
}

llama_moe_stats::llama_moe_stats(const llama_model & model) : model(model) {
    n_expert     = model.hparams.n_expert;
    period       = moe_stats_period();
    n_tokens_dec = 16; // MTP and two slots still stay far below the op-offload batch of 32

    il2idx.resize(model.layers.size(), -1);
}

int llama_moe_stats::layer_idx(int il) {
    if (il < 0 || il >= (int) model.layers.size()) {
        return -1;
    }
    if (il2idx[il] >= 0) {
        return il2idx[il];
    }

    const auto & layer = model.layers[il];

    moe_layer ml;
    ml.il = il;
    ml.freq.resize(n_expert, 0);

    for (const ggml_tensor * t : { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps }) {
        if (t == nullptr || t->ne[2] <= 0) {
            continue;
        }
        ml.bytes += ggml_nbytes(t) / t->ne[2];
        if (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) {
            ml.on_host = true;
        }
    }

    const int idx = (int) mlayers.size();
    il2idx[il] = idx;
    mlayers.push_back(std::move(ml));

    for (auto & s : sims) {
        sim_layer sl;
        sl.prev.resize(n_expert, -1);
        sl.next.resize(n_expert, -1);
        sl.cached.resize(n_expert, 0);
        s.layers.push_back(std::move(sl));
    }

    return idx;
}

void llama_moe_stats::access(int idx, int32_t e, bool is_dec) {
    moe_layer & ml = mlayers[idx];
    if (!ml.on_host) {
        return;
    }
    ml.freq[e] += is_dec ? 1 : 0;
    ml.n_acc   += is_dec ? 1 : 0;

    for (auto & s : sims) {
        sim_layer & l = s.layers[idx];

        s.n_acc += is_dec ? 1 : 0;

        if (l.cached[e]) {
            if (is_dec) {
                s.n_hit++;
                s.b_hit += ml.bytes;
            }

            // move to front
            if (l.head != e) {
                const int32_t p = l.prev[e];
                const int32_t n = l.next[e];
                l.next[p] = n;
                if (n >= 0) {
                    l.prev[n] = p;
                } else {
                    l.tail = p;
                }
                l.prev[e] = -1;
                l.next[e] = l.head;
                l.prev[l.head] = e;
                l.head = e;
            }
            continue;
        }

        if (l.n_ins >= s.max_ins) {
            continue;
        }

        if (l.n_cached >= s.n_slots) {
            const int32_t v = l.tail;
            const int32_t p = l.prev[v];
            if (p >= 0) {
                l.next[p] = -1;
            } else {
                l.head = -1;
            }
            l.tail = p;
            l.cached[v] = 0;
            l.n_cached--;
        }

        l.prev[e] = -1;
        l.next[e] = l.head;
        if (l.head >= 0) {
            l.prev[l.head] = e;
        }
        l.head = e;
        if (l.tail < 0) {
            l.tail = e;
        }
        l.cached[e] = 1;
        l.n_cached++;
        l.n_ins++;
        if (is_dec) {
            s.b_ins += ml.bytes;
        }
    }
}

void llama_moe_stats::add_ubatch(const llm_graph_result * res, ggml_backend_sched_t sched, uint32_t n_tokens) {
    if (res->t_moe_topk.empty()) {
        return;
    }

    if (sims.empty()) {
        for (int32_t n_slots : MOE_STATS_SLOTS) {
            if (n_slots > n_expert) {
                break;
            }
            for (int32_t max_ins : MOE_STATS_INS) {
                sim s;
                s.n_slots = n_slots;
                s.max_ins = max_ins;
                sims.push_back(std::move(s));
            }
        }
    }

    const bool is_dec = n_tokens <= n_tokens_dec;

    n_steps_all++;
    if (is_dec) {
        n_steps_dec++;
        n_tok_dec += n_tokens;
    }

    // one ubatch is one cache update step
    for (auto & s : sims) {
        for (auto & l : s.layers) {
            l.n_ins = 0;
        }
    }

    // a long prompt costs far more to replay than it teaches, so keep only its tail
    const int64_t n_take = is_dec ? n_tokens : std::min<int64_t>(n_tokens, MOE_STATS_PREFILL_TAIL);

    size_t total = 0;
    for (ggml_tensor * t : res->t_moe_topk) {
        total += t->ne[0]*n_take;
    }
    buf.resize(total);

    backends.clear();

    size_t off = 0;
    for (ggml_tensor * t : res->t_moe_topk) {
        const int64_t n_used = t->ne[0];
        const int64_t n_skip = t->ne[1] - n_take;

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);
        if (backend != nullptr) {
            if (std::find(backends.begin(), backends.end(), backend) == backends.end()) {
                backends.push_back(backend);
            }
            ggml_backend_tensor_get_async(backend, t, buf.data() + off,
                    n_skip*n_used*sizeof(int32_t), n_take*n_used*sizeof(int32_t));
        }

        off += n_used*n_take;
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }

    off = 0;
    for (size_t i = 0; i < res->t_moe_topk.size(); ++i) {
        ggml_tensor * t = res->t_moe_topk[i];

        const int64_t n_used = t->ne[0];
        const int     idx    = layer_idx(res->t_moe_topk_il[i]);

        if (idx < 0) {
            off += n_used*n_take;
            continue;
        }

        n_expert_used = n_used;

        for (int64_t j = 0; j < n_used*n_take; ++j) {
            const int32_t e = buf[off + j];
            if (e >= 0 && e < n_expert) {
                access(idx, e, is_dec);
            }
        }
        off += n_used*n_take;
    }

    if (period > 0 && n_steps_dec > 0 && n_steps_dec % period == 0) {
        report();
    }
}

void llama_moe_stats::report() const {
    if (mlayers.empty() || n_steps_dec == 0) {
        return;
    }

    int      n_host  = 0;
    uint64_t b_host  = 0; // bytes of one slot summed over host-resident MoE layers
    uint64_t b_slot  = 0; // bytes of one slot, max over host-resident layers
    for (const auto & ml : mlayers) {
        if (ml.on_host) {
            n_host++;
            b_host += ml.bytes;
            b_slot = std::max(b_slot, ml.bytes);
        }
    }

    if (n_host == 0) {
        LLAMA_LOG_WARN(MOE_LOG ": all %d MoE layers keep their experts on the device, a cache would add nothing\n",
                (int) mlayers.size());
        return;
    }

    const double mib   = 1024.0*1024.0;
    const double gib   = 1024.0*1024.0*1024.0;
    const double n_tok = (double) n_tok_dec;

    // host expert bytes read per decode token with no cache
    const double read_mib = (double) n_expert_used * (double) b_host / mib;

    LLAMA_LOG_WARN(MOE_LOG ": ---- MoE expert cache potential ----\n");
    LLAMA_LOG_WARN(MOE_LOG ": experts %" PRId64 ", used per token %" PRId64 ", MoE layers %d of which %d keep experts in host memory, the tables below cover those %d\n",
            n_expert, n_expert_used, (int) mlayers.size(), n_host, n_host);
    LLAMA_LOG_WARN(MOE_LOG ": one slot (up+gate+down of one expert) %.2f MiB, one slot on every host layer %.2f MiB\n",
            (double) b_slot / mib, (double) b_host / mib);
    LLAMA_LOG_WARN(MOE_LOG ": decode steps %" PRIu64 ", decode tokens %" PRIu64 ", other steps %" PRIu64 "\n",
            n_steps_dec, n_tok_dec, n_steps_all - n_steps_dec);
    LLAMA_LOG_WARN(MOE_LOG ": host expert reads without a cache %.1f MiB per token, %.1f ms at %.0f GB/s\n",
            read_mib, read_mib*mib/(MOE_STATS_BW_HOST*1e9)*1e3, MOE_STATS_BW_HOST);
    LLAMA_LOG_WARN(MOE_LOG ": net column assumes %.0f GB/s host read and %.0f GB/s upload\n",
            MOE_STATS_BW_HOST, MOE_STATS_BW_LINK);
    LLAMA_LOG_WARN(MOE_LOG ": a step of more than %u tokens counts as prompt, only its last %d tokens warm the cache\n",
            n_tokens_dec, (int) MOE_STATS_PREFILL_TAIL);

    // best case for a cache that is filled once and never changes
    LLAMA_LOG_WARN(MOE_LOG ":  slots | VRAM total |  static hit\n");
    for (int32_t n_slots : MOE_STATS_SLOTS) {
        if (n_slots > n_expert) {
            break;
        }
        uint64_t hit = 0;
        uint64_t acc = 0;
        std::vector<uint32_t> f;
        for (const auto & ml : mlayers) {
            if (!ml.on_host) {
                continue;
            }
            f = ml.freq;
            const size_t k = std::min((size_t) n_slots, f.size());
            std::partial_sort(f.begin(), f.begin() + k, f.end(), std::greater<uint32_t>());
            for (size_t i = 0; i < k; ++i) {
                hit += f[i];
            }
            acc += ml.n_acc;
        }
        LLAMA_LOG_WARN(MOE_LOG ": %6d | %7.2f GiB | %9.1f%%\n",
                n_slots, (double) n_slots * (double) b_host / gib,
                acc ? 100.0*(double) hit/(double) acc : 0.0);
    }

    LLAMA_LOG_WARN(MOE_LOG ":  slots | VRAM total | maxins |  LRU hit | upload MiB/tok | saved MiB/tok |   net ms/tok\n");
    for (const auto & s : sims) {
        const double hit = s.n_acc ? (double) s.n_hit/(double) s.n_acc : 0.0;

        // only the inserts made on decode steps cost decode time
        const double up_mib   = (double) s.b_ins / mib / n_tok;
        const double save_mib = (double) s.b_hit / mib / n_tok;
        const double net_ms   = (save_mib/MOE_STATS_BW_HOST - up_mib/MOE_STATS_BW_LINK)*mib/1e9*1e3;

        LLAMA_LOG_WARN(MOE_LOG ": %6d | %7.2f GiB | %6d | %7.1f%% | %14.1f | %13.1f | %+12.1f\n",
                s.n_slots, (double) s.n_slots * (double) b_host / gib, s.max_ins,
                100.0*hit, up_mib, save_mib, net_ms);
    }

    for (const auto & d : model.devices) {
        size_t free = 0;
        size_t total = 0;
        ggml_backend_dev_memory(d.dev, &free, &total);
        LLAMA_LOG_WARN(MOE_LOG ": %s free %.2f GiB of %.2f GiB, fits %d slots on every host layer\n",
                ggml_backend_dev_name(d.dev), (double) free/gib, (double) total/gib,
                b_host ? (int) (free / b_host) : 0);
    }
}

//
// llama_moe_cache
//

#define MOE_CACHE_LOG "moe_cache"

// device memory that the automatic slot count leaves free
static const size_t MOE_CACHE_MARGIN = 1024ull*1024*1024;

// max expert uploads per layer per step
static const int MOE_CACHE_MAX_INS = 1;

llama_moe_cache::llama_moe_cache(const llama_model & model, int32_t n_slots_req) : model(model), n_slots_req(n_slots_req) {
}

llama_moe_cache::~llama_moe_cache() {
    // the uploads read the host weights and write the slots, finish them before the buffers go away
    if (backend_up != nullptr) {
        ggml_backend_synchronize(backend_up);
    }
    if (ev_up != nullptr) {
        ggml_backend_event_free(ev_up);
    }
    if (backend_up != nullptr) {
        ggml_backend_free(backend_up);
    }
}

void llama_moe_cache::init(ggml_backend_sched_t sched) {
    initialized = true;

    const auto & hparams = model.hparams;

    n_expert = hparams.n_expert;
    if (n_expert <= 0) {
        return;
    }

    // the last backend of the scheduler is the CPU
    const int n_dev_backends = ggml_backend_sched_get_n_backends(sched) - 1;

    int n_moe  = 0; // MoE layers
    int n_dev  = 0; // MoE layers with the experts in device memory
    int n_skip = 0; // MoE layers with host experts that the cache cannot use

    bool pageable = false;

    std::vector<layer> sel;

    for (int il = 0; il < (int) hparams.n_layer(); ++il) {
        const auto & ml = model.layers[il];

        layer l;
        l.il     = il;
        l.merged = ml.ffn_gate_up_exps != nullptr;
        l.w_up   = l.merged ? ml.ffn_gate_up_exps : ml.ffn_up_exps;
        l.w_gate = l.merged ? nullptr : ml.ffn_gate_exps;
        l.w_down = ml.ffn_down_exps;

        if (l.w_up == nullptr || l.w_down == nullptr) {
            continue;
        }

        n_moe++;

        bool on_dev = false;
        bool ok     = true;

        for (const ggml_tensor * w : { l.w_up, l.w_gate, l.w_down }) {
            if (w == nullptr) {
                continue;
            }
            if (w->buffer == nullptr) {
                ok = false;
                continue;
            }
            for (int i = 0; i < n_dev_backends; ++i) {
                // accelerators such as BLAS also use host buffers, but they do not run mul_mat_id
                ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
                const auto type = ggml_backend_dev_type(ggml_backend_get_device(b));
                if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                    on_dev = on_dev || ggml_backend_supports_buft(b, ggml_backend_buffer_get_type(w->buffer));
                }
            }
            // repacked weights are not host buffers
            ok = ok && ggml_backend_buffer_is_host(w->buffer) && ggml_is_contiguous(w) && w->ne[2] == n_expert;
        }

        if (on_dev) {
            n_dev++;
            continue;
        }

        ok = ok && !ml.ffn_up_exps_b && !ml.ffn_gate_exps_b && !ml.ffn_down_exps_b && !ml.ffn_gate_up_exps_b;
        ok = ok && !ml.ffn_up_exps_s && !ml.ffn_gate_exps_s && !ml.ffn_down_exps_s;

        // the rest of the layer has to run on a device of the scheduler, all cached layers share one device
        ggml_backend_dev_t d = model.dev_layer(il);
        ggml_backend_t     b = nullptr;
        for (int i = 0; i < n_dev_backends; ++i) {
            if (ggml_backend_get_device(ggml_backend_sched_get_backend(sched, i)) == d) {
                b = ggml_backend_sched_get_backend(sched, i);
            }
        }
        ok = ok && b != nullptr && (dev == nullptr || d == dev);

        if (!ok) {
            n_skip++;
            continue;
        }

        if (dev == nullptr) {
            dev     = d;
            backend = b;
        }

        ggml_backend_dev_t src = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(l.w_up->buffer));
        pageable = pageable || src == nullptr || ggml_backend_dev_type(src) == GGML_BACKEND_DEVICE_TYPE_CPU;

        sel.push_back(l);
    }

    if (sel.empty()) {
        LLAMA_LOG_WARN("%s: no MoE layer has host experts that the cache can use (MoE layers %d, on the device %d, not supported %d), the cache is off\n",
                MOE_CACHE_LOG, n_moe, n_dev, n_skip);
        return;
    }

    const int n_layers = (int) sel.size();

    b_slot = 0;
    for (const auto & l : sel) {
        for (const ggml_tensor * w : { l.w_up, l.w_gate, l.w_down }) {
            b_slot += w ? w->nb[2] : 0;
        }
    }

    const double mib = 1024.0*1024.0;

    size_t free  = 0;
    size_t total = 0;
    ggml_backend_dev_memory(dev, &free, &total);

    int64_t n = n_slots_req;
    if (n < 0) {
        // one more slot covers the zero slot and the padding
        const size_t need = MOE_CACHE_MARGIN + b_slot;
        n = free > need ? (int64_t) ((free - need) / b_slot) : 0;
    }
    n = std::min<int64_t>(n, n_expert);

    if (n <= 0) {
        LLAMA_LOG_WARN("%s: %s has %.0f MiB free, one slot on %d layers takes %.2f MiB and %.0f MiB stay free, the cache is off\n",
                MOE_CACHE_LOG, ggml_backend_dev_name(dev), free/mib, n_layers, b_slot/mib, MOE_CACHE_MARGIN/mib);
        return;
    }

    {
        ggml_init_params params = {
            /*.mem_size   =*/ (4*n_layers + 1)*ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ctx_dev.reset(ggml_init(params));
        params.mem_size = 2*n_layers*ggml_tensor_overhead();
        ctx_host.reset(ggml_init(params));
    }

    table_all = ggml_new_tensor_2d(ctx_dev.get(), GGML_TYPE_I32, n_expert, n_layers);
    ggml_set_name(table_all, "moe_cache.table");

    const int64_t n_used_max = hparams.n_expert_used_max();

    for (int idx = 0; idx < n_layers; ++idx) {
        layer & l = sel[idx];

        auto new_slots = [&](const ggml_tensor * w, const char * name) -> ggml_tensor * {
            if (w == nullptr) {
                return nullptr;
            }
            ggml_tensor * t = ggml_new_tensor_3d(ctx_dev.get(), w->type, w->ne[0], w->ne[1], n + 1);
            ggml_format_name(t, "moe_cache.%d.%s", l.il, name);
            return t;
        };

        l.s_up   = new_slots(l.w_up, l.merged ? "gate_up" : "up");
        l.s_gate = new_slots(l.w_gate, "gate");
        l.s_down = new_slots(l.w_down, "down");

        l.table = ggml_view_2d(ctx_dev.get(), table_all, 1, n_expert, ggml_element_size(table_all), idx*table_all->nb[1]);
        ggml_format_name(l.table, "moe_cache.%d.table", l.il);

        l.skip = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, n_expert);
        ggml_format_name(l.skip, "moe_cache.%d.skip", l.il);

        l.ids = ggml_new_tensor_1d(ctx_host.get(), GGML_TYPE_I32, n_used_max*N_TOKENS_MAX);
        ggml_format_name(l.ids, "moe_cache.%d.ids", l.il);
    }

    buf_dev.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_dev.get(), ggml_backend_dev_buffer_type(dev)));
    if (!buf_dev) {
        LLAMA_LOG_ERROR("%s: failed to allocate %d slots on %s, the cache is off\n", MOE_CACHE_LOG, (int) n, ggml_backend_dev_name(dev));
        return;
    }

    buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU))));
    if (!buf_host) {
        LLAMA_LOG_ERROR("%s: failed to allocate the host buffer, the cache is off\n", MOE_CACHE_LOG);
        buf_dev.reset();
        return;
    }

    backend_up = ggml_backend_dev_init(dev, nullptr);
    if (backend_up == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to create the upload backend on %s, the cache is off\n", MOE_CACHE_LOG, ggml_backend_dev_name(dev));
        buf_dev.reset();
        buf_host.reset();
        return;
    }

    ev_up = ggml_backend_event_new(dev);

    // the zero slot must stay zero, the other slots start empty
    ggml_backend_buffer_set_usage(buf_dev.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(buf_dev.get(), 0);
    ggml_backend_buffer_clear(buf_host.get(), 0);

    n_slots = (int32_t) n;

    table_host.assign(n_expert*n_layers, n_slots);
    ggml_backend_tensor_set(table_all, table_host.data(), 0, ggml_nbytes(table_all));

    il2idx.assign(hparams.n_layer(), -1);
    lrus.resize(n_layers);
    for (int idx = 0; idx < n_layers; ++idx) {
        lru & c = lrus[idx];
        c.prev.assign(n_expert, -1);
        c.next.assign(n_expert, -1);
        c.slot.assign(n_expert, -1);
        c.expert.assign(n_slots, -1);
        c.live.assign(n_slots, 0);

        il2idx[sel[idx].il] = idx;
    }
    layers = std::move(sel);

    // warn: the library INFO level is hidden at the default verbosity
    LLAMA_LOG_WARN("%s: %s, %d of %d MoE layers, %d slots per layer, %.2f MiB per slot, %.2f MiB total, %.0f MiB were free\n",
            MOE_CACHE_LOG, ggml_backend_dev_name(dev), n_layers, n_moe, n_slots, b_slot/mib,
            ggml_backend_buffer_get_size(buf_dev.get())/mib, free/mib);

    if (n_skip > 0) {
        LLAMA_LOG_WARN("%s: %d MoE layers with host experts are not cached: repacked weights (use --no-repack), biases, scales or another device\n",
                MOE_CACHE_LOG, n_skip);
    }

    if (pageable) {
        LLAMA_LOG_WARN("%s: the host experts are in pageable memory, the uploads can block the host, load the model without mmap (-lm none or -lm dio) to use pinned memory\n",
                MOE_CACHE_LOG);
    }
}

const llama_moe_cache::layer * llama_moe_cache::get_layer(int il) const {
    if (!ready() || il < 0 || il >= (int) il2idx.size() || il2idx[il] < 0) {
        return nullptr;
    }
    return &layers[il2idx[il]];
}

void llama_moe_cache::lru_unlink(lru & c, int32_t e) {
    const int32_t p = c.prev[e];
    const int32_t n = c.next[e];
    if (p >= 0) {
        c.next[p] = n;
    } else {
        c.head = n;
    }
    if (n >= 0) {
        c.prev[n] = p;
    } else {
        c.tail = p;
    }
    c.prev[e] = -1;
    c.next[e] = -1;
}

void llama_moe_cache::lru_push(lru & c, int32_t e) {
    c.prev[e] = -1;
    c.next[e] = c.head;
    if (c.head >= 0) {
        c.prev[c.head] = e;
    }
    c.head = e;
    if (c.tail < 0) {
        c.tail = e;
    }
}

void llama_moe_cache::insert(size_t idx, int32_t e) {
    const layer & l = layers[idx];
    lru         & c = lrus[idx];

    int32_t s = -1;

    if (c.n_fill < n_slots) {
        s = c.n_fill++;
    } else {
        // take the slot of the least recently used expert, but not before its own upload is published
        const int32_t v = c.tail;
        s = c.slot[v];
        if (!c.live[s]) {
            return;
        }
        lru_unlink(c, v);
        c.slot[v] = -1;
        c.live[s] = 0;
        c.evicted.push_back(v);
    }

    c.expert[s] = e;
    c.slot[e]   = s;
    lru_push(c, e);
    c.uploads_new.push_back(s);

    auto upload = [&](const ggml_tensor * w, ggml_tensor * t) {
        if (w != nullptr) {
            ggml_backend_tensor_set_async(backend_up, t, (const char *) w->data + e*w->nb[2], s*t->nb[2], w->nb[2]);
            b_ins += w->nb[2];
        }
    };
    upload(l.w_up,   l.s_up);
    upload(l.w_gate, l.s_gate);
    upload(l.w_down, l.s_down);

    n_ins++;
}

void llama_moe_cache::update(const llm_graph_result * res) {
    if (!ready() || res->moe_cache_il.empty()) {
        return;
    }

    // the device can still read the slots of this graph
    ggml_backend_synchronize(backend);

    // the uploads of the last step ran during this graph
    {
        const int64_t t_start_us = ggml_time_us();
        if (ev_up_recorded) {
            ggml_backend_event_synchronize(ev_up);
            ev_up_recorded = false;
        }
        ggml_backend_synchronize(backend_up);
        t_wait_us += ggml_time_us() - t_start_us;
    }

    n_steps++;

    const int64_t n_tokens = res->moe_cache_n_tokens;

    for (size_t i = 0; i < res->moe_cache_il.size(); ++i) {
        const int idx = il2idx[res->moe_cache_il[i]];

        const layer & l = layers[idx];
        lru         & c = lrus[idx];

        const int32_t * ids  = (const int32_t *) l.ids->data;
        const int32_t * skip = (const int32_t *) l.skip->data;

        int n_new = 0;

        for (int64_t j = 0; j < res->moe_cache_n_used[i]*n_tokens; ++j) {
            const int32_t e = ids[j];
            if (e < 0 || e >= n_expert) {
                continue;
            }

            // the hit counts what the graph computed on the device
            n_acc++;
            n_hit += skip[e] != 0;

            if (c.slot[e] >= 0) {
                lru_unlink(c, e);
                lru_push(c, e);
            } else if (n_new < MOE_CACHE_MAX_INS) {
                n_new++;
                insert(idx, e);
            }
        }
    }

    bool dirty  = false;
    bool issued = false;

    for (size_t idx = 0; idx < layers.size(); ++idx) {
        lru & c = lrus[idx];

        int32_t * skip  = (int32_t *) layers[idx].skip->data;
        int32_t * table = table_host.data() + idx*n_expert;

        // the next graph must not read the slots that receive new uploads
        for (int32_t v : c.evicted) {
            skip[v]  = 0;
            table[v] = n_slots;
        }
        for (int32_t s : c.uploads_done) {
            const int32_t e = c.expert[s];
            skip[e]   = 1;
            table[e]  = s;
            c.live[s] = 1;
        }

        dirty  = dirty  || !c.evicted.empty() || !c.uploads_done.empty();
        issued = issued || !c.uploads_new.empty();

        c.evicted.clear();
        c.uploads_done.clear();
        std::swap(c.uploads_done, c.uploads_new);
    }

    if (dirty) {
        ggml_backend_tensor_set(table_all, table_host.data(), 0, ggml_nbytes(table_all));
    }

    if (issued && ev_up != nullptr) {
        ggml_backend_event_record(ev_up, backend_up);
        ev_up_recorded = true;
    }

    const int period = moe_stats_period();
    if (period > 0 && n_steps % period == 0) {
        report();
    }
}

void llama_moe_cache::report() const {
    if (!ready() || n_steps == 0) {
        return;
    }

    const double mib = 1024.0*1024.0;

    LLAMA_LOG_WARN("%s: %d layers, %d slots, steps %" PRIu64 ", hit %.1f%% of %" PRIu64 ", uploads %" PRIu64 " (%.1f MiB per step), upload wait %.2f ms per step\n",
            MOE_CACHE_LOG, (int) layers.size(), n_slots, n_steps,
            n_acc ? 100.0*(double) n_hit/(double) n_acc : 0.0, n_acc,
            n_ins, (double) b_ins/mib/(double) n_steps, 1e-3*(double) t_wait_us/(double) n_steps);
}
