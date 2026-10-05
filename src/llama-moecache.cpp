#include "llama-moecache.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <queue>
#include <thread>

// the report is opt-in, so it goes out at warn level to survive the default log verbosity
#define MOE_LOG "moe_cache_stats"

// cache sizes to simulate, in slots per layer
static const int32_t MOE_STATS_SLOTS[] = { 8, 16, 32, 48, 64, 96, 128, 192, 256 };

// max expert uploads per layer per decode step
static const int32_t MOE_STATS_INS[] = { 1, 2, 4, 8 };

// steps from a miss to the first step that reads the new slot, as in llama_moe_cache::update()
static const int32_t MOE_STATS_DELAY = 2;

// the LFU counts halve after this many accesses per slot, as in TinyLFU
static const int32_t MOE_STATS_LFU_WINDOW = 10;

// assumed host RAM read bandwidth, used only for the time estimate of the report
static const double MOE_STATS_BW_HOST = 35.0; // GB/s

// read once: llama_moe_cache::update() asks for it on every step
static int moe_stats_period() {
    static const int period = [] {
        const char * s = getenv("LLAMA_MOE_CACHE_STATS");
        if (s == nullptr || s[0] == '\0' || s[0] == '0') {
            return 0;
        }
        const int v = atoi(s);
        return v > 1 ? v : 2048;
    }();
    return period;
}

bool llama_moe_stats_enabled() {
    return moe_stats_period() > 0;
}

llama_moe_stats::llama_moe_stats(const llama_model & model) : model(model) {
    n_expert     = model.hparams.n_expert;
    period       = moe_stats_period();
    n_tokens_dec = llama_moe_cache::N_TOKENS_MAX;

    il2idx.resize(model.layers.size(), -1);
}

void llama_moe_stats::set_cache(int32_t n_slots, int32_t max_ins, bool lfu) {
    cache_slots = n_slots;
    cache_ins   = max_ins;
    cache_lfu   = lfu;
}

std::vector<int32_t> llama_moe_stats::table_slots() const {
    std::vector<int32_t> v;
    for (int32_t n : MOE_STATS_SLOTS) {
        if (n <= n_expert) {
            v.push_back(n);
        }
    }
    if (cache_slots > 0 && cache_slots <= n_expert && std::find(v.begin(), v.end(), cache_slots) == v.end()) {
        v.push_back(cache_slots);
        std::sort(v.begin(), v.end());
    }
    return v;
}

std::vector<int32_t> llama_moe_stats::table_ins() const {
    std::vector<int32_t> v(std::begin(MOE_STATS_INS), std::end(MOE_STATS_INS));
    if (cache_ins > 0 && std::find(v.begin(), v.end(), cache_ins) == v.end()) {
        v.push_back(cache_ins);
        std::sort(v.begin(), v.end());
    }
    return v;
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

    for (const ggml_tensor * t : { layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_gate_up_exps, layer.ffn_down_exps }) {
        if (t == nullptr || t->ne[2] <= 0) {
            continue;
        }
        ml.bytes += ggml_nbytes(t) / t->ne[2];
        if (t->buffer != nullptr && ggml_backend_buffer_is_host(t->buffer)) {
            ml.on_host = true;
        }
    }

    // the steps in the trace had no routing for this layer
    ml.tr_off.assign(n_steps_dec - tr_step0 + 1, 0);

    const int idx = (int) mlayers.size();
    il2idx[il] = idx;
    mlayers.push_back(std::move(ml));

    return idx;
}

void llama_moe_stats::add_ubatch(const llm_graph_result * res, ggml_backend_sched_t sched, uint32_t n_tokens) {
    if (res->t_moe_topk.empty()) {
        return;
    }

    n_steps_all++;

    // only a step that computes the host experts on the CPU can use the cache, as in build_moe_cache()
    bool is_dec = n_tokens <= n_tokens_dec;
    for (size_t i = 0; is_dec && i < res->t_moe_mm.size() && i < res->t_moe_topk_il.size(); ++i) {
        const int idx = layer_idx(res->t_moe_topk_il[i]);
        if (idx < 0 || !mlayers[idx].on_host) {
            continue;
        }
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, res->t_moe_mm[i]);
        is_dec = b != nullptr && ggml_backend_dev_type(ggml_backend_get_device(b)) == GGML_BACKEND_DEVICE_TYPE_CPU;
        break;
    }
    if (!is_dec) {
        return;
    }

    size_t total = 0;
    for (ggml_tensor * t : res->t_moe_topk) {
        total += ggml_nelements(t);
    }
    buf.resize(total);

    backends.clear();

    size_t off = 0;
    for (ggml_tensor * t : res->t_moe_topk) {
        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, t);
        if (backend != nullptr) {
            if (std::find(backends.begin(), backends.end(), backend) == backends.end()) {
                backends.push_back(backend);
            }
            ggml_backend_tensor_get_async(backend, t, buf.data() + off, 0, ggml_nbytes(t));
        } else {
            std::fill(buf.begin() + off, buf.begin() + off + ggml_nelements(t), -1);
        }
        off += ggml_nelements(t);
    }

    for (ggml_backend_t backend : backends) {
        ggml_backend_synchronize(backend);
    }

    // add the step to the trace, the last layer can have fewer tokens than the others
    off = 0;
    for (size_t i = 0; i < res->t_moe_topk.size(); ++i) {
        ggml_tensor * t = res->t_moe_topk[i];

        const int64_t n   = ggml_nelements(t);
        const int     idx = layer_idx(res->t_moe_topk_il[i]);

        if (idx >= 0) {
            n_expert_used = t->ne[0];
        }

        if (idx >= 0 && mlayers[idx].on_host) {
            moe_layer & ml = mlayers[idx];
            for (int64_t j = 0; j < n; ++j) {
                const int32_t e = buf[off + j];
                if (e >= 0 && e < n_expert) {
                    ml.tr_ids.push_back(e);
                    ml.freq[e]++;
                    ml.n_acc++;
                }
            }
        }
        off += n;
    }

    for (auto & ml : mlayers) {
        ml.tr_off.push_back((uint32_t) ml.tr_ids.size());
    }

    n_steps_dec++;
    n_tok_dec += n_tokens;

    if (period > 0 && n_steps_dec % period == 0) {
        report();
    }
}

void llama_moe_stats::sim_layer::unlink(int32_t e) {
    const int     li = list[e];
    const int32_t p  = prev[e];
    const int32_t n  = next[e];
    if (p >= 0) {
        next[p] = n;
    } else {
        head[li] = n;
    }
    if (n >= 0) {
        prev[n] = p;
    } else {
        tail[li] = p;
    }
    prev[e] = -1;
    next[e] = -1;
    list[e] = LIST_NONE;
    size[li]--;
}

void llama_moe_stats::sim_layer::push(int li, int32_t e) {
    prev[e] = -1;
    next[e] = head[li];
    if (head[li] >= 0) {
        prev[head[li]] = e;
    }
    head[li] = e;
    if (tail[li] < 0) {
        tail[li] = e;
    }
    list[e] = (uint8_t) li;
    size[li]++;
}

void llama_moe_stats::sim_layer::evict(int32_t e, int32_t step) {
    // the graph of this step read the slot before the eviction
    if (ready[e] >= 0 && ready[e] <= step) {
        gone[e] = step;
    }
    ready[e] = -1;
    unlink(e);
}

void llama_moe_stats::sim_init(sim & s) const {
    while (s.layers.size() < mlayers.size()) {
        sim_layer l;
        l.ready.assign(n_expert, -1);
        l.gone.assign(n_expert, -1);
        if (s.pol == POLICY_OPT) {
            l.nu.assign(n_expert, INT32_MAX);
        } else {
            l.prev.assign(n_expert, -1);
            l.next.assign(n_expert, -1);
            l.list.assign(n_expert, LIST_NONE);
        }
        if (s.pol == POLICY_LFU) {
            l.freq.assign(n_expert, 0);
        }
        s.layers.push_back(std::move(l));
    }
}

// LRU, LRU/1, 2Q and LFU, the order of the updates is the order of update() in llama_moe_cache
void llama_moe_stats::replay_layer(sim & s, int idx) const {
    const moe_layer & ml = mlayers[idx];
    sim_layer       & l  = s.layers[idx];

    const int32_t kin  = std::max(1, s.n_slots/4); // 2Q: max slots of A1in before it gives a slot
    const int32_t kout = std::max(1, s.n_slots/2); // 2Q: length of A1out

    const size_t n_win = ml.tr_off.size() - 1;

    for (size_t k = 0; k < n_win; ++k) {
        const int32_t g = (int32_t) (tr_step0 + k);

        l.n_ins = 0;

        for (uint32_t j = ml.tr_off[k]; j < ml.tr_off[k + 1]; ++j) {
            const int32_t e = ml.tr_ids[j];

            // the graph of the step read the slots as they were before the step
            const bool hit = (l.ready[e] >= 0 && l.ready[e] <= g) || l.gone[e] == g;

            s.n_acc++;
            s.n_hit  += hit;
            s.b_host += !hit && ml.tr_first[j] ? ml.bytes : 0;

            if (s.pol == POLICY_LFU) {
                l.freq[e] += l.freq[e] < UINT16_MAX;
                if (++l.n_seen >= MOE_STATS_LFU_WINDOW*s.n_slots) {
                    l.n_seen = 0;
                    for (auto & f : l.freq) {
                        f /= 2;
                    }
                }
            }

            // the expert has a slot, with or without a finished upload
            if (l.ready[e] >= 0) {
                // 2Q: A1in is a FIFO
                if (s.pol != POLICY_2Q || l.list[e] == LIST_MAIN) {
                    l.unlink(e);
                    l.push(LIST_MAIN, e);
                }
                continue;
            }

            if (l.n_ins >= s.max_ins) {
                continue;
            }

            // 2Q: an expert from A1out goes to Am, a new one to A1in
            int li = LIST_MAIN;
            if (s.pol == POLICY_2Q && l.list[e] != LIST_OUT) {
                li = LIST_IN;
            }

            if (l.size[LIST_MAIN] + l.size[LIST_IN] >= s.n_slots) {
                int lv = LIST_MAIN;
                if (s.pol == POLICY_2Q && (l.size[LIST_IN] > kin || l.size[LIST_MAIN] == 0)) {
                    lv = LIST_IN;
                }
                const int32_t v = l.tail[lv];

                // LFU: a candidate with fewer accesses than the victim does not enter and uses no upload
                if (s.pol == POLICY_LFU && l.freq[e] <= l.freq[v]) {
                    continue;
                }
                // the slot of an expert with an upload that is not published yet is not free
                if (l.ready[v] > g) {
                    l.n_ins++;
                    continue;
                }

                l.evict(v, g);
                if (lv == LIST_IN) {
                    l.push(LIST_OUT, v);
                    if (l.size[LIST_OUT] > kout) {
                        l.unlink(l.tail[LIST_OUT]);
                    }
                }
            }

            if (l.list[e] == LIST_OUT) {
                l.unlink(e);
            }
            l.push(li, e);
            l.ready[e] = g + s.delay;
            l.n_ins++;
            s.b_ins += ml.bytes;
        }
    }
}

// Belady with the budget and the delay of the other policies: in each step it takes the misses with
// the nearest next use and evicts the cached expert with the farthest next use, while that is farther
void llama_moe_stats::replay_layer_opt(sim & s, int idx) const {
    const moe_layer & ml = mlayers[idx];
    sim_layer       & l  = s.layers[idx];

    const int32_t n_win = (int32_t) ml.tr_off.size() - 1;
    const int32_t never = INT32_MAX;
    const int32_t g0    = (int32_t) tr_step0;

    // the distinct experts of each step with their access count and the next entry of the same expert
    std::vector<int32_t> ent_e;
    std::vector<int32_t> ent_n;
    std::vector<int32_t> ent_k;
    std::vector<int32_t> ent_next;
    std::vector<int32_t> ent_off(n_win + 1, 0);
    std::vector<int32_t> pos(n_expert, -1);

    for (int32_t k = 0; k < n_win; ++k) {
        ent_off[k] = (int32_t) ent_e.size();
        for (uint32_t j = ml.tr_off[k]; j < ml.tr_off[k + 1]; ++j) {
            const int32_t e = ml.tr_ids[j];
            if (ml.tr_first[j]) {
                pos[e] = (int32_t) ent_e.size();
                ent_e.push_back(e);
                ent_n.push_back(1);
                ent_k.push_back(k);
            } else {
                ent_n[pos[e]]++;
            }
        }
    }
    ent_off[n_win] = (int32_t) ent_e.size();

    // after this loop pos holds the first entry of each expert
    ent_next.resize(ent_e.size());
    std::fill(pos.begin(), pos.end(), -1);
    for (int32_t i = (int32_t) ent_e.size() - 1; i >= 0; --i) {
        ent_next[i] = pos[ent_e[i]];
        pos[ent_e[i]] = i;
    }

    // first step at or after k_min among entry i and the later entries of its expert
    auto next_use = [&](int32_t i, int32_t k_min) {
        for (; i >= 0; i = ent_next[i]) {
            if (ent_k[i] >= k_min) {
                return ent_k[i];
            }
        }
        return never;
    };

    // the cached experts by their next use, the farthest on top, stale entries go when they come up
    std::priority_queue<std::pair<int32_t, int32_t>> heap;

    for (int32_t e = 0; e < n_expert; ++e) {
        if (l.ready[e] >= 0) {
            l.nu[e] = next_use(pos[e], std::max(0, l.ready[e] - g0));
            heap.push({ l.nu[e], e });
        }
    }

    std::vector<std::pair<int32_t, int32_t>> cand; // next use, expert

    for (int32_t k = 0; k < n_win; ++k) {
        const int32_t g = g0 + k;

        cand.clear();

        for (int32_t i = ent_off[k]; i < ent_off[k + 1]; ++i) {
            const int32_t e = ent_e[i];

            const bool hit = l.ready[e] >= 0 && l.ready[e] <= g;

            s.n_acc  += ent_n[i];
            s.n_hit  += hit ? ent_n[i] : 0;
            s.b_host += hit ? 0 : ml.bytes;

            if (l.ready[e] >= 0) {
                l.nu[e] = next_use(ent_next[i], std::max(k + 1, l.ready[e] - g0));
                heap.push({ l.nu[e], e });
            } else {
                const int32_t nu = next_use(ent_next[i], k + s.delay);
                if (nu != never) {
                    cand.push_back({ nu, e });
                }
            }
        }

        std::sort(cand.begin(), cand.end());

        int32_t n_ins = 0;
        for (const auto & c : cand) {
            if (n_ins >= s.max_ins) {
                break;
            }
            if (l.size[LIST_MAIN] >= s.n_slots) {
                while (!heap.empty() && (l.ready[heap.top().second] < 0 || l.nu[heap.top().second] != heap.top().first)) {
                    heap.pop();
                }
                if (heap.empty() || heap.top().first <= c.first) {
                    break;
                }
                l.ready[heap.top().second] = -1;
                heap.pop();
                l.size[LIST_MAIN]--;
            }
            const int32_t e = c.second;
            l.ready[e] = g + s.delay;
            l.nu[e]    = c.first;
            heap.push({ c.first, e });
            l.size[LIST_MAIN]++;
            n_ins++;
            s.b_ins += ml.bytes;
        }
    }
}

void llama_moe_stats::replay() {
    const int64_t t_start_us = ggml_time_us();

    // the cells of one table row are next to each other
    if (sims.empty()) {
        for (int32_t n_slots : table_slots()) {
            for (int32_t max_ins : table_ins()) {
                for (int pol = 0; pol < POLICY_COUNT; ++pol) {
                    sim s;
                    s.pol     = (policy) pol;
                    s.n_slots = n_slots;
                    s.max_ins = max_ins;
                    s.delay   = pol == POLICY_LRU1 ? 1 : MOE_STATS_DELAY;
                    sims.push_back(std::move(s));
                }
            }
        }
    }

    // mark the first access of each expert in its step and count the host reads with no cache
    std::vector<int32_t> stamp(n_expert);
    for (auto & ml : mlayers) {
        if (!ml.on_host) {
            continue;
        }
        std::fill(stamp.begin(), stamp.end(), -1);
        ml.tr_first.resize(ml.tr_ids.size());
        for (size_t k = 0; k + 1 < ml.tr_off.size(); ++k) {
            for (uint32_t j = ml.tr_off[k]; j < ml.tr_off[k + 1]; ++j) {
                const int32_t e = ml.tr_ids[j];
                ml.tr_first[j] = stamp[e] != (int32_t) k;
                stamp[e] = (int32_t) k;
                ml.b_host += ml.tr_first[j] ? ml.bytes : 0;
            }
        }
    }

    // the simulations are independent, the replay runs between two decode steps
    std::atomic<size_t> next { 0 };
    auto worker = [&]() {
        for (size_t i = next++; i < sims.size(); i = next++) {
            sim & s = sims[i];
            sim_init(s);
            for (int idx = 0; idx < (int) mlayers.size(); ++idx) {
                if (!mlayers[idx].on_host) {
                    continue;
                }
                if (s.pol == POLICY_OPT) {
                    replay_layer_opt(s, idx);
                } else {
                    replay_layer(s, idx);
                }
            }
        }
    };

    const size_t n_threads = std::max<size_t>(1, std::min<size_t>(std::thread::hardware_concurrency(), sims.size()));

    std::vector<std::thread> threads;
    for (size_t i = 1; i < n_threads; ++i) {
        threads.emplace_back(worker);
    }
    worker();
    for (auto & t : threads) {
        t.join();
    }

    for (auto & ml : mlayers) {
        ml.tr_ids.clear();
        ml.tr_first.clear();
        ml.tr_off.assign(1, 0);
    }
    tr_step0 = n_steps_dec;

    t_replay_us = ggml_time_us() - t_start_us;
}

void llama_moe_stats::report() {
    if (n_steps_dec == 0) {
        if (n_steps_all > 0) {
            LLAMA_LOG_WARN(MOE_LOG ": none of the %" PRIu64 " steps can use a cache, it needs at most %u tokens and the host experts computed on the CPU\n",
                    n_steps_all, n_tokens_dec);
        }
        return;
    }
    if (mlayers.empty() || n_steps_dec == n_steps_rep) {
        return;
    }

    int      n_host = 0;
    uint64_t b_host = 0; // bytes of one slot summed over host-resident MoE layers
    uint64_t b_slot = 0; // bytes of one slot, max over host-resident layers
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
        n_steps_rep = n_steps_dec;
        return;
    }

    const uint64_t n_replay = n_steps_dec - tr_step0;

    replay();

    n_steps_rep = n_steps_dec;

    uint64_t b_read = 0; // host expert bytes read with no cache over all decode steps
    for (const auto & ml : mlayers) {
        b_read += ml.b_host;
    }

    const double mib     = 1024.0*1024.0;
    const double gib     = 1024.0*1024.0*1024.0;
    const double n_steps = (double) n_steps_dec;

    const double read_mib = (double) b_read / mib / n_steps;

    LLAMA_LOG_WARN(MOE_LOG ": ---- MoE expert cache potential ----\n");
    LLAMA_LOG_WARN(MOE_LOG ": experts %" PRId64 ", used per token %" PRId64 ", MoE layers %d of which %d keep experts in host memory, the tables below cover those %d\n",
            n_expert, n_expert_used, (int) mlayers.size(), n_host, n_host);
    LLAMA_LOG_WARN(MOE_LOG ": one slot (up+gate+down of one expert) %.2f MiB, one slot on every host layer %.2f MiB\n",
            (double) b_slot / mib, (double) b_host / mib);
    LLAMA_LOG_WARN(MOE_LOG ": decode steps %" PRIu64 " with %" PRIu64 " tokens (%.2f per step), other steps %" PRIu64 "\n",
            n_steps_dec, n_tok_dec, (double) n_tok_dec / n_steps, n_steps_all - n_steps_dec);
    LLAMA_LOG_WARN(MOE_LOG ": a decode step has at most %u tokens and computes the host experts on the CPU, the steps that --moe-cache serves\n",
            n_tokens_dec);
    LLAMA_LOG_WARN(MOE_LOG ": host expert reads without a cache %.1f MiB per step, %.1f ms at %.0f GB/s\n",
            read_mib, read_mib*mib/(MOE_STATS_BW_HOST*1e9)*1e3, MOE_STATS_BW_HOST);
    LLAMA_LOG_WARN(MOE_LOG ": replayed %" PRIu64 " steps through %d simulations in %.2f s\n",
            n_replay, (int) sims.size(), 1e-6*(double) t_replay_us);

    // best case for a cache that is filled once and never changes
    LLAMA_LOG_WARN(MOE_LOG ":  slots | VRAM total |  static hit\n");
    for (int32_t n_slots : table_slots()) {
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
        LLAMA_LOG_WARN(MOE_LOG ": %5d%c | %7.2f GiB | %9.1f%%\n",
                n_slots, n_slots == cache_slots ? '*' : ' ', (double) n_slots * (double) b_host / gib,
                acc ? 100.0*(double) hit/(double) acc : 0.0);
    }

    LLAMA_LOG_WARN(MOE_LOG ": policies, all with the same routing and at most ins uploads per layer per step:\n");
    LLAMA_LOG_WARN(MOE_LOG ":   LRU   as --moe-cache: a new expert serves from the second step after its miss, a slot with an unpublished upload is not taken\n");
    LLAMA_LOG_WARN(MOE_LOG ":   LRU/1 the same LRU if a new expert served from the next step\n");
    LLAMA_LOG_WARN(MOE_LOG ":   2Q    Johnson and Shasha 2Q: a new expert goes to a FIFO (A1in, 25%% of the slots), it goes to the LRU part when it comes back from the ghost list (A1out, 50%% of the slots)\n");
    LLAMA_LOG_WARN(MOE_LOG ":   LFU   LRU that takes a new expert only if it has more accesses than the LRU tail, the counts halve every %d x slots accesses (TinyLFU)\n",
            MOE_STATS_LFU_WINDOW);
    LLAMA_LOG_WARN(MOE_LOG ":   OPT   Belady with the future of the replayed steps, a limit for the other policies\n");
    LLAMA_LOG_WARN(MOE_LOG ": each cell: decode hit, upload MiB per step, host read MiB per step\n");
    if (cache_slots > 0) {
        LLAMA_LOG_WARN(MOE_LOG ": * the slots and the upload limit of --moe-cache, its %s cell must match the moe_cache line with the same steps\n", cache_lfu ? "LFU" : "LRU");
    }
    LLAMA_LOG_WARN(MOE_LOG ":  slots | VRAM total | ins |         LRU         |        LRU/1        |         2Q          |         LFU         |         OPT\n");

    for (size_t i = 0; i + POLICY_COUNT <= sims.size(); i += POLICY_COUNT) {
        const bool is_cache = sims[i].n_slots == cache_slots && sims[i].max_ins == cache_ins;
        char line[512];
        int  n = snprintf(line, sizeof(line), "%5d%c | %7.2f GiB | %3d",
                sims[i].n_slots, is_cache ? '*' : ' ', (double) sims[i].n_slots * (double) b_host / gib, sims[i].max_ins);
        for (int p = 0; p < POLICY_COUNT && n > 0 && n < (int) sizeof(line); ++p) {
            const sim & s = sims[i + p];
            n += snprintf(line + n, sizeof(line) - n, " | %5.1f%% %5.1f %6.1f",
                    s.n_acc ? 100.0*(double) s.n_hit/(double) s.n_acc : 0.0,
                    (double) s.b_ins / mib / n_steps, (double) s.b_host / mib / n_steps);
        }
        LLAMA_LOG_WARN(MOE_LOG ": %s\n", line);
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

llama_moe_cache::llama_moe_cache(const llama_model & model, int32_t n_slots_req) : model(model), n_slots_req(n_slots_req) {
    const char * policy = getenv("LLAMA_MOE_CACHE_POLICY");
    frequency_admission = policy && strcmp(policy, "lfu") == 0;
    const char * async = getenv("LLAMA_MOE_CACHE_ASYNC");
    async_table = async && atoi(async) != 0;
}

llama_moe_cache::~llama_moe_cache() {
    join_uploads();
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

    backend_up = ggml_backend_dev_init(dev, nullptr);
    if (backend_up == nullptr) {
        LLAMA_LOG_ERROR("%s: failed to create the upload backend on %s, the cache is off\n", MOE_CACHE_LOG, ggml_backend_dev_name(dev));
        return;
    }

    ev_up = ggml_backend_event_new(dev);
    async_upload = async_table && strncmp(ggml_backend_dev_name(dev), "CUDA", 4) == 0;
    LLAMA_LOG_INFO("%s: policy %s, async uploads %d, async table %d\n", MOE_CACHE_LOG,
            frequency_admission ? "lfu" : "lru", async_upload, async_table);

    b_slot = 0;
    for (const auto & l : sel) {
        for (const ggml_tensor * w : { l.w_up, l.w_gate, l.w_down }) {
            b_slot += w ? w->nb[2] : 0;
        }
    }

    il2idx.assign(hparams.n_layer(), -1);
    for (int idx = 0; idx < (int) sel.size(); ++idx) {
        il2idx[sel[idx].il] = idx;
    }
    hot.resize(sel.size());
    layers = std::move(sel);

    if (n_skip > 0) {
        LLAMA_LOG_WARN("%s: %d MoE layers with host experts are not cached: repacked weights (use --no-repack), biases, scales or another device\n",
                MOE_CACHE_LOG, n_skip);
    }

    if (pageable) {
        LLAMA_LOG_WARN("%s: the host experts are in pageable memory, the uploads can block the host, load the model without mmap (-lm none or -lm dio) to use pinned memory\n",
                MOE_CACHE_LOG);
    }
}

void llama_moe_cache::alloc() {
    if (layers.empty() || backend_up == nullptr || ready()) {
        return;
    }

    const int64_t t_start_us = ggml_time_us();

    const int n_layers = (int) layers.size();

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

    // warn: the library INFO level is hidden at the default verbosity, later calls come from the phase switches of --phase-mem
    const auto log_level = alloc_logged ? GGML_LOG_LEVEL_INFO : GGML_LOG_LEVEL_WARN;
    alloc_logged = true;

    if (n <= 0) {
        llama_log_internal(log_level, "%s: %s has %.0f MiB free, one slot on %d layers takes %.2f MiB and %.0f MiB stay free, no slots\n",
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
        params.mem_size = (2*n_layers + (async_table ? 1 : 0))*ggml_tensor_overhead();
        ctx_host.reset(ggml_init(params));
    }

    table_all = ggml_new_tensor_2d(ctx_dev.get(), GGML_TYPE_I32, n_expert, n_layers);
    ggml_set_name(table_all, "moe_cache.table");
    if (async_table) {
        table_staging = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, n_expert, n_layers);
        ggml_set_name(table_staging, "moe_cache.table_staging");
    }

    const int64_t n_used_max = model.hparams.n_expert_used_max();

    for (int idx = 0; idx < n_layers; ++idx) {
        layer & l = layers[idx];

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
    if (buf_dev) {
        ggml_backend_buffer_type_t buft_host = async_table ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
        if (!buft_host) {
            buft_host = ggml_backend_dev_buffer_type(ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU));
        }
        buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
    }
    if (!buf_dev || !buf_host) {
        LLAMA_LOG_ERROR("%s: failed to allocate %d slots on %s, no slots\n", MOE_CACHE_LOG, (int) n, ggml_backend_dev_name(dev));
        buf_dev.reset();
        for (auto & l : layers) {
            l.s_up = l.s_gate = l.s_down = l.table = l.skip = l.ids = nullptr;
        }
        table_all = nullptr;
        table_staging = nullptr;
        ctx_dev.reset();
        ctx_host.reset();
        return;
    }

    // the zero slot must stay zero, the other slots start empty
    ggml_backend_buffer_set_usage(buf_dev.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ggml_backend_buffer_clear(buf_dev.get(), 0);
    ggml_backend_buffer_clear(buf_host.get(), 0);

    n_slots   = (int32_t) n;
    n_slots_l = n_slots;

    table_host.assign(n_expert*n_layers, n_slots);

    lrus.assign(n_layers, lru());

    uint64_t n_up = 0;
    uint64_t b_up = 0;

    for (int idx = 0; idx < n_layers; ++idx) {
        lru & c = lrus[idx];
        c.prev.assign(n_expert, -1);
        c.next.assign(n_expert, -1);
        c.slot.assign(n_expert, -1);
        c.expert.assign(n_slots, -1);
        c.live.assign(n_slots, 0);
        c.seen.assign(n_expert, UINT64_MAX);
        if (frequency_admission) {
            c.freq.assign(n_expert, 0);
        }

        int32_t * skip  = (int32_t *) layers[idx].skip->data;
        int32_t * table = table_host.data() + idx*n_expert;

        // upload the experts of the last release() again, the least recent first so that the LRU keeps its order
        const auto & h = hot[idx];
        for (int32_t i = std::min<int32_t>(n_slots, (int32_t) h.size()) - 1; i >= 0; --i) {
            const int32_t e = h[i];
            const int32_t s = c.n_fill++;

            c.expert[s] = e;
            c.slot[e]   = s;
            c.live[s]   = 1;
            lru_push(c, e);

            skip[e]  = 1;
            table[e] = s;

            b_up += upload(idx, e, s);
            n_up++;
        }
    }

    // the next graph reads the new slots at once
    ggml_backend_synchronize(backend_up);
    ggml_backend_tensor_set(table_all, table_host.data(), 0, ggml_nbytes(table_all));

    const int64_t t_us = ggml_time_us() - t_start_us;

    n_alloc++;
    n_refill   += n_up;
    b_refill   += b_up;
    t_alloc_us += t_us;

    llama_log_internal(log_level, "%s: %s, %d of %d MoE layers, %d slots per layer, %.2f MiB per slot, %.2f MiB total, %.0f MiB were free, %d uploads per layer per step, "
            "refilled %" PRIu64 " experts (%.1f MiB) in %.1f ms\n",
            MOE_CACHE_LOG, ggml_backend_dev_name(dev), n_layers, n_moe, n_slots, b_slot/mib,
            ggml_backend_buffer_get_size(buf_dev.get())/mib, free/mib, max_ins, n_up, b_up/mib, 1e-3*t_us);
}

void llama_moe_cache::release() {
    if (!ready()) {
        return;
    }

    const int64_t t_start_us = ggml_time_us();
    join_uploads();

    // the last graph and the uploads can still use the slots
    ggml_backend_synchronize(backend);
    if (ev_up_recorded) {
        ggml_backend_event_synchronize(ev_up);
        ev_up_recorded = false;
    }
    ggml_backend_synchronize(backend_up);

    for (size_t idx = 0; idx < layers.size(); ++idx) {
        const lru & c = lrus[idx];

        auto & h = hot[idx];
        h.clear();
        for (int32_t e = c.head; e >= 0; e = c.next[e]) {
            h.push_back(e);
        }

        layer & l = layers[idx];
        l.s_up = l.s_gate = l.s_down = l.table = l.skip = l.ids = nullptr;
    }

    table_all = nullptr;
    table_staging = nullptr;
    buf_dev.reset();
    buf_host.reset();
    ctx_dev.reset();
    ctx_host.reset();

    lrus.clear();
    table_host.clear();

    n_slots = 0;

    t_alloc_us += ggml_time_us() - t_start_us;
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
    lru & c = lrus[idx];

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

    if (async_upload) {
        pending_uploads.push_back({ idx, e, s });
        const layer & l = layers[idx];
        b_ins += l.w_up->nb[2] + (l.w_gate ? l.w_gate->nb[2] : 0) + l.w_down->nb[2];
    } else {
        b_ins += upload(idx, e, s);
    }
    n_ins++;
}

void llama_moe_cache::join_uploads() {
    if (upload_thread.joinable()) {
        upload_thread.join();
        t_submit_us += upload_submit_us;
    }
    pending_uploads.clear();
}

size_t llama_moe_cache::upload(size_t idx, int32_t e, int32_t s) {
    const layer & l = layers[idx];

    size_t b = 0;

    auto up = [&](const ggml_tensor * w, ggml_tensor * t) {
        if (w != nullptr) {
            ggml_backend_tensor_set_async(backend_up, t, (const char *) w->data + e*w->nb[2], s*t->nb[2], w->nb[2]);
            b += w->nb[2];
        }
    };
    up(l.w_up,   l.s_up);
    up(l.w_gate, l.s_gate);
    up(l.w_down, l.s_down);

    return b;
}

void llama_moe_cache::update(const llm_graph_result * res) {
    if (!ready() || res->moe_cache_il.empty()) {
        return;
    }

    const int64_t t_start_us = ggml_time_us();

    // the device can still read the slots of this graph
    ggml_backend_synchronize(backend);

    const int64_t t_graph_end_us = ggml_time_us();
    t_graph_us += t_graph_end_us - t_start_us;

    // the uploads of the last step ran during this graph
    join_uploads();
    if (ev_up_recorded) {
        ggml_backend_event_synchronize(ev_up);
        ev_up_recorded = false;
    }
    ggml_backend_synchronize(backend_up);

    const int64_t t_wait_end_us = ggml_time_us();
    t_wait_us += t_wait_end_us - t_graph_end_us;

    n_steps++;

    for (size_t i = 0; i < res->moe_cache_il.size(); ++i) {
        const int idx = il2idx[res->moe_cache_il[i]];

        const layer & l = layers[idx];
        lru         & c = lrus[idx];

        const int32_t * ids  = (const int32_t *) l.ids->data;
        const int32_t * skip = (const int32_t *) l.skip->data;

        const size_t b_exp = l.w_up->nb[2] + (l.w_gate ? l.w_gate->nb[2] : 0) + l.w_down->nb[2];

        int n_new = 0;

        for (int64_t j = 0; j < res->moe_cache_n_ids[i]; ++j) {
            const int32_t e = ids[j];
            if (e < 0 || e >= n_expert) {
                continue;
            }

            // the hit counts what the graph computed on the device
            n_acc++;
            n_hit += skip[e] != 0;

            if (c.seen[e] != n_steps) {
                c.seen[e] = n_steps;
                b_host += skip[e] != 0 ? 0 : b_exp;
            }

            if (frequency_admission) {
                c.freq[e] += c.freq[e] < UINT16_MAX;
                if (++c.n_seen >= MOE_STATS_LFU_WINDOW*n_slots) {
                    c.n_seen = 0;
                    for (auto & count : c.freq) {
                        count >>= 1;
                    }
                }
            }

            if (c.slot[e] >= 0) {
                lru_unlink(c, e);
                lru_push(c, e);
            } else if (n_new < max_ins) {
                if (frequency_admission && c.n_fill == n_slots && c.freq[e] <= c.freq[c.tail]) {
                    continue;
                }
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
        if (async_table) {
            memcpy(table_staging->data, table_host.data(), ggml_nbytes(table_all));
            ggml_backend_tensor_set_async(backend, table_all, table_staging->data, 0, ggml_nbytes(table_all));
        } else {
            ggml_backend_tensor_set(table_all, table_host.data(), 0, ggml_nbytes(table_all));
        }
    }

    auto submit = [this] {
        const int64_t start = ggml_time_us();
        for (const auto & up : pending_uploads) {
            upload(up.idx, up.expert, up.slot);
        }
        if (ev_up != nullptr) {
            ggml_backend_event_record(ev_up, backend_up);
            ev_up_recorded = true;
        }
        upload_submit_us = ggml_time_us() - start;
    };
    if (issued) {
        if (async_upload) {
            upload_thread = std::thread(submit);
        } else {
            submit();
        }
    }

    t_upd_us += ggml_time_us() - t_wait_end_us;

    const int period = moe_stats_period();
    if (period > 0 && n_steps % period == 0) {
        report();
    }
}

void llama_moe_cache::report() const {
    if (n_alloc == 0 || n_steps == 0) {
        return;
    }

    const double mib = 1024.0*1024.0;

    const double ms = 1e-3/(double) n_steps;

    LLAMA_LOG_WARN("%s: %d layers, %d slots, %d uploads per layer per step, steps %" PRIu64 ", hit %.1f%% of %" PRIu64 ", uploads %" PRIu64 " (%.1f MiB per step), host experts %.1f MiB per step\n",
            MOE_CACHE_LOG, (int) layers.size(), n_slots_l, max_ins, n_steps,
            n_acc ? 100.0*(double) n_hit/(double) n_acc : 0.0, n_acc,
            n_ins, (double) b_ins/mib/(double) n_steps, (double) b_host/mib/(double) n_steps);
    LLAMA_LOG_WARN("%s: host time per step: graph end wait %.2f ms, upload wait %.2f ms, update %.2f ms\n",
            MOE_CACHE_LOG, ms*(double) t_graph_us, ms*(double) t_wait_us, ms*(double) t_upd_us);
    if (async_upload) {
        LLAMA_LOG_WARN("%s: worker upload submission %.2f ms per step, join time is included in upload wait\n", MOE_CACHE_LOG, ms*(double) t_submit_us);
    }
    if (n_alloc > 1) {
        LLAMA_LOG_WARN("%s: slots allocated %" PRIu64 " times, refilled %" PRIu64 " experts (%.1f MiB), alloc and release %.1f ms per call\n",
                MOE_CACHE_LOG, n_alloc, n_refill, (double) b_refill/mib, 1e-3*(double) t_alloc_us/(double) n_alloc);
    }
}
