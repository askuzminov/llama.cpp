#pragma once

// fork: helpers of server_context for the speculative replay, the prompt cache setup and the
// byte-bounded context checkpoints. server_context_impl and server_slot are local to
// server-context.cpp, so these take what they need as arguments

#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstdint>
#include <list>
#include <memory>
#include <vector>

struct server_prompt_cache;

// replay after a checkpoint restore: the target sampled these tokens before, so accept them as they are.
// a new check can reject them, because the logits can change with the batch (e.g. MoE routing), and the replay then does not end
std::vector<llama_token> server_sample_and_accept_replay(
        common_sampler * smpl,
        llama_context * ctx,
        const std::vector<int32_t> & idxs,
        const llama_tokens & draft);

// host-RAM bounds of the prompt cache and the context checkpoints
struct server_ckpt_limits {
    // resolved host-RAM reserve for context checkpoints, in bytes (0 = disabled)
    size_t reserve_bytes = 0;

    // mapped model bytes; the OS reports them as available, but they are not free for checkpoints
    size_t weights_bytes = 0;

    // resolved count cap on context checkpoints per slot (0 = no count limit)
    int32_t count_cap = 0;

    // resolve the host-RAM reserve that bounds context-checkpoint memory
    void init_reserve(const common_params & params);

    // resolve the checkpoint bounds when context checkpoints are enabled
    void init_ckpt(const common_params & params, const llama_model * model_tgt, const llama_model * model_dft);

    // Per-slot byte budget for context checkpoints, measured when a checkpoint is created and not
    // once at startup: host RAM comes and goes while the server runs. `held` is what the checkpoints
    // of all slots hold now. 0 = the host cannot be probed, then only the count cap bounds the footprint.
    size_t budget(size_t held, int n_parallel) const;
};

// create the prompt cache for --cache-ram != 0, with the disk spill when it is configured
void server_prompt_cache_init(
        std::unique_ptr<server_prompt_cache> & prompt_cache,
        const common_params & params,
        int32_t n_ctx,
        size_t reserve_bytes,
        const llama_model * model_tgt,
        const llama_context * ctx_tgt,
        const llama_context * ctx_dft,
        bool has_mtmd);

// Identifies the model and context configuration that the cached states belong to. Spill files
// of this model written under a different signature cannot be restored and are removed on startup.
uint64_t server_prompt_cache_signature(
        const common_params & params,
        const llama_model * model_tgt,
        const llama_context * ctx_tgt,
        const llama_context * ctx_dft);

// context checkpoints of a slot: a base and the deltas after it form a chain
using server_ckpt_list = std::list<common_prompt_checkpoint>;

// the checkpoint after `it` is a delta that depends on it, so `it` cannot be dropped
bool server_ckpt_has_dependent_delta(const server_ckpt_list & ckpts, server_ckpt_list::const_iterator it);

// Evict whole base->delta chains from the front. Deltas depend on all preceding
// checkpoints in their chain, so eviction can only drop a chain as a unit (never a
// checkpoint from the middle). Drops the oldest chain and its trailing deltas. Returns the bytes freed.
size_t server_ckpt_evict_front_chain(server_ckpt_list & ckpts);

// base_pos for the checkpoint about to be appended: -1 for a self-contained one, else the
// pos_max of the last checkpoint, which the new delta hangs off
llama_pos server_ckpt_next_base_pos(const server_ckpt_list & ckpts, const llama_model * model);

// evict chains from the front while the list is over `budget`, but never the newest chain.
// Returns the bytes freed.
size_t server_ckpt_evict_over_budget(server_ckpt_list & ckpts, size_t budget);

// restore the checkpoint at `it_cur`: its base, then the deltas up to it, then the draft state.
// Returns the checkpoint whose state is now in the context, nullptr on failure
const common_prompt_checkpoint * server_ckpt_restore(
        const server_ckpt_list & ckpts,
        server_ckpt_list::const_iterator it_cur,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        llama_seq_id seq_id);
