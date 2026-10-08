// fork: helpers of server_context, see server-context-fork.h

#include "server-context-fork.h"

#include "server-common.h"
#include "server-task.h"

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <functional>
#include <string>

// count cap used when the host cannot be probed, so no byte budget can be measured
static constexpr int32_t CKPT_COUNT_CAP_FALLBACK = 32;

// smallest checkpoint pool handed out over all slots
static constexpr size_t CKPT_BUDGET_FLOOR = 512ull*1024*1024;

//
// server_ckpt_limits
//

void server_ckpt_limits::init_reserve(const common_params & params) {
    if (params.cache_ram_reserve_mib < 0) {
        // auto: a small, roughly-constant anti-swap headroom - the reserve only needs to keep
        // the OS/other processes from swapping, which does not scale with total RAM. Keep
        // ~1/16 of total, clamped to [512 MiB, 2 GiB], so it rarely evicts a full chain.
        const size_t total = common_host_mem_total();
        if (total > 0) {
            reserve_bytes = std::min<size_t>(std::max<size_t>(total / 16, 512ull*1024*1024), 2ull*1024*1024*1024);
            SRV_TRC("context-checkpoint host-RAM reserve: auto = %zu MiB (total host RAM = %zu MiB)\n",
                    reserve_bytes >> 20, total >> 20);
        } else {
            reserve_bytes = 0;
            SRV_WRN("%s", "could not determine total host RAM; context-checkpoint memory guard disabled (use --cache-ram-reserve)\n");
        }
    } else {
        reserve_bytes = (size_t) params.cache_ram_reserve_mib * 1024 * 1024;
    }
}

void server_ckpt_limits::init_ckpt(const common_params & params, const llama_model * model_tgt, const llama_model * model_dft) {
    // Bound the checkpoint footprint in bytes, not in checkpoint count: one checkpoint is
    // a few KiB of recurrent state on a hybrid model and several GiB of attention KV on a
    // dense model at long context, so a count says nothing about the memory used. How many
    // bytes there are to spend is measured on every checkpoint creation - see budget().

    // Mapped weights sit in the page cache, which the OS reports as available: on Linux
    // MemAvailable counts reclaimable page cache, on Windows ullAvailPhys counts the
    // standby list. Handing that memory to checkpoints would page the weights back out.
    // Ask the model what it mapped - --load-mode does not say: a lazy tensor maps its file
    // on any load mode, and `auto` drops mmap on a device without support for it.
    if (model_tgt) {
        weights_bytes = llama_model_mapped_size(model_tgt);
    }

    if (model_dft && model_dft != model_tgt) {
        weights_bytes += llama_model_mapped_size(model_dft);
    }

    count_cap = std::max(0, params.n_ctx_checkpoints);

    if (common_host_mem_available() == 0) {
        SRV_WRN("%s", "could not determine available host RAM; context checkpoints cannot be bounded in bytes\n");

        if (count_cap == 0) {
            // the host-RAM reserve reads the same probe that just failed, so a count cap is
            // all that is left to bound the footprint
            count_cap = CKPT_COUNT_CAP_FALLBACK;

            SRV_WRN("context checkpoints fall back to a count cap of %d per slot; set --ctx-checkpoints to choose one\n",
                    count_cap);
        }
    } else {
        SRV_TRC("context-checkpoint budget: measured per checkpoint (host-RAM reserve %zu MiB, mapped weights %zu MiB, %d slots)\n",
                reserve_bytes >> 20, weights_bytes >> 20, params.n_parallel);
    }

    const std::string cap = count_cap > 0
        ? std::to_string(count_cap) : std::string("no count limit");

    SRV_TRC("context checkpoints enabled, count cap = %s, min spacing = %d\n",
            cap.c_str(), params.checkpoint_min_step);
}

size_t server_ckpt_limits::budget(size_t held, int n_parallel) const {
    const size_t avail = common_host_mem_available();
    if (avail == 0) {
        return 0;
    }

    // The prompt cache is not subtracted. What it holds is out of `avail` already, and --cache-ram
    // bounds the rest of it, the checkpoints it carries included. Both sides stop at the reserve.
    const size_t taken = reserve_bytes + weights_bytes;

    // floor: below this the checkpoints would be evicted as fast as they are made and buy
    // nothing. The host-RAM reserve still evicts if the host really is that tight.
    const size_t pool = std::max<size_t>(avail + held > taken ? avail + held - taken : 0, CKPT_BUDGET_FLOOR);

    return pool / std::max(1, n_parallel);
}

//
// prompt cache
//

void server_prompt_cache_init(
        std::unique_ptr<server_prompt_cache> & prompt_cache,
        const common_params & params,
        int32_t n_ctx,
        size_t reserve_bytes,
        const llama_model * model_tgt,
        const llama_context * ctx_tgt,
        const llama_context * ctx_dft,
        bool has_mtmd) {
    int32_t cache_ram_mib_eff = params.cache_ram_mib;
    if (cache_ram_mib_eff < 0) {
        // auto: cap the prompt-cache archive at ~1/4 of total host RAM (floor 1 GiB).
        // The host-RAM reserve (see server_ckpt_limits::reserve_bytes) is the real anti-swap backstop.
        const size_t total = common_host_mem_total();
        cache_ram_mib_eff = total > 0
            ? (int32_t) std::min<size_t>(std::max<size_t>((total / 4) >> 20, 1024), (size_t) INT32_MAX)
            : 8192;
        SRV_TRC("prompt cache size limit: auto = %d MiB (total host RAM = %zu MiB)\n", cache_ram_mib_eff, total >> 20);
    } else {
        SRV_TRC("prompt cache is enabled, size limit: %d MiB\n", cache_ram_mib_eff);
    }
    SRV_TRC("%s", "use `--cache-ram 0` to disable the prompt cache\n");

    const size_t spill_limit = (size_t) std::max(0, params.cache_disk_mib) * 1024ull * 1024ull;
    prompt_cache = std::make_unique<server_prompt_cache>(
            (size_t) cache_ram_mib_eff, n_ctx, reserve_bytes, params.cache_spill_dir, spill_limit,
            server_prompt_cache_signature(params, model_tgt, ctx_tgt, ctx_dft), (uint64_t) std::hash<std::string>{}(params.model.path), has_mtmd);
    prompt_cache->min_tokens = (size_t) std::max(0, params.cache_min_tokens);

    if (params.cache_min_tokens > 0) {
        SRV_INF("prompt cache minimum prompt length: %d tokens\n", params.cache_min_tokens);
    }

    if (!params.cache_spill_dir.empty()) {
        const std::string budget = spill_limit ? (std::to_string(spill_limit >> 20) + " MiB") : std::string("unlimited");
        SRV_INF("prompt cache disk spill enabled: dir = %s, disk budget = %s\n",
                params.cache_spill_dir.c_str(), budget.c_str());
    }
}

uint64_t server_prompt_cache_signature(
        const common_params & params,
        const llama_model * model_tgt,
        const llama_context * ctx_tgt,
        const llama_context * ctx_dft) {
    char desc[256] = {};
    llama_model_desc(model_tgt, desc, sizeof(desc));

    std::string s;
    s += desc;
    s += "|" + params.model.path;
    s += "|" + std::to_string(llama_model_size(model_tgt));
    s += "|" + std::to_string(llama_n_ctx(ctx_tgt));
    s += "|" + std::to_string(llama_n_ctx_seq(ctx_tgt));
    s += "|" + std::to_string((int) params.cache_type_k);
    s += "|" + std::to_string((int) params.cache_type_v);
    s += "|" + std::to_string(params.n_parallel);
    s += "|" + std::to_string((int) params.swa_full);
    s += "|" + std::to_string((int) params.kv_unified);
    s += "|" + std::to_string(ctx_dft != nullptr);
    s += "|" + std::to_string(LLAMA_STATE_SEQ_VERSION);

    return (uint64_t) std::hash<std::string>{}(s);
}

//
// context checkpoints
//

bool server_ckpt_has_dependent_delta(const server_ckpt_list & ckpts, server_ckpt_list::const_iterator it) {
    return std::next(it) != ckpts.end() && std::next(it)->is_delta();
}

size_t server_ckpt_evict_front_chain(server_ckpt_list & ckpts) {
    if (ckpts.empty()) {
        return 0;
    }
    size_t freed = ckpts.front().size();
    const bool erased_base = !ckpts.front().is_delta();
    ckpts.erase(ckpts.begin());
    if (erased_base) {
        while (!ckpts.empty() && ckpts.front().is_delta()) {
            freed += ckpts.front().size();
            ckpts.erase(ckpts.begin());
        }
    }
    return freed;
}

llama_pos server_ckpt_next_base_pos(const server_ckpt_list & ckpts, const llama_model * model) {
    // Walk back over the last chain: the deltas, then the base they hang off.
    bool   has_base          = false;
    size_t chain_base_bytes  = 0;
    size_t chain_delta_bytes = 0;

    for (auto it = ckpts.rbegin(); it != ckpts.rend(); ++it) {
        // a failed capture holds no data, so it cannot serve as a base
        if (!it->is_delta() && !it->data_tgt.empty()) {
            has_base         = true;
            chain_base_bytes = it->size();
            break;
        }

        chain_delta_bytes += it->size();
    }

    // Re-anchor once the deltas cost as much as the base they hang off. Eviction drops a chain
    // as a unit, so a chain that never ends is one the byte budget can never touch, and
    // the whole list is then only dropped at once by the host-RAM reserve. This bounds a chain
    // at about two full checkpoints and keeps the older chains evictable.
    const bool re_anchor = has_base && chain_delta_bytes >= chain_base_bytes;

    // The recurrent state is a running state that cannot be delta-encoded, so every checkpoint
    // of a recurrent or hybrid model is self-contained. With PARTIAL_ONLY a hybrid stores only
    // the recurrent state, which is small - the attention KV is rolled back from the live cache.
    const bool has_rs  = llama_model_is_recurrent(model) || llama_model_is_hybrid(model);
    const bool is_base = !has_base || re_anchor;

    if (is_base || has_rs) {
        return -1;
    }

    return ckpts.back().pos_max;
}

size_t server_ckpt_evict_over_budget(server_ckpt_list & ckpts, size_t budget) {
    size_t total = 0;
    for (const common_prompt_checkpoint & cp : ckpts) {
        total += cp.size();
    }

    size_t freed = 0;
    while (total > budget && ckpts.size() > 1) {
        // the front chain is a base plus its trailing deltas; if that is the whole list,
        // evicting it would take the checkpoint just created with it
        size_t n_chain = 1;
        for (auto it = std::next(ckpts.begin());
                it != ckpts.end() && it->is_delta(); ++it) {
            ++n_chain;
        }
        if (n_chain >= ckpts.size()) {
            break;
        }

        const size_t n = server_ckpt_evict_front_chain(ckpts);
        if (n == 0) {
            break;
        }
        freed += n;
        total -= std::min(total, n);
    }

    return freed;
}

const common_prompt_checkpoint * server_ckpt_restore(
        const server_ckpt_list & ckpts,
        server_ckpt_list::const_iterator it_cur,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        llama_seq_id seq_id) {
    auto it_base = it_cur;
    while (it_base != ckpts.begin() &&
            (it_base->is_delta() || it_base->data_tgt.empty())) {
        --it_base;
    }

    // a failed capture holds no data: restore the base before it and process the
    // tokens after that base again
    const common_prompt_checkpoint & cur_cp = *it_cur;
    const common_prompt_checkpoint & base_cp = *it_base;
    const bool not_captured = cur_cp.data_tgt.empty();
    const common_prompt_checkpoint & restored_cp = not_captured ? base_cp : cur_cp;

    bool ok = !base_cp.is_delta() && !base_cp.data_tgt.empty();
    if (ok) {
        ok = base_cp.apply(
                ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }
    if (ok && !not_captured) {
        for (auto it_delta = std::next(it_base); ok && it_delta != std::next(it_cur); ++it_delta) {
            ok = it_delta->apply(
                    ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
    }
    if (ok) {
        // no-op when the checkpoint holds no draft state: the caller trims the
        // draft context to the restored position right after this
        ok = restored_cp.apply_dft(ctx_dft, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
    }

    return ok ? &restored_cp : nullptr;
}
