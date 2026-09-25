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
