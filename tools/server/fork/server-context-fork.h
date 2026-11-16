#pragma once

// fork: helpers of server_context for the speculative replay, the prompt cache setup and the
// byte-bounded context checkpoints. server_context_impl and server_slot are local to
// server-context.cpp, so these take what they need as arguments

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <cstdint>
#include <list>
#include <memory>
#include <string>
#include <vector>

struct server_prompt_cache;
struct server_tokens;

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

// LLAMA_CKPT_USER_BREAK (0 by default since 16.10, 1: upstream): with 0 the prompt batch does not stop at the start of the last user message
// (or of a user message past checkpoint_min_step) for a checkpoint there. The new tokens of a chat turn (the last
// answer as the client sends it back, the template, the new message) then go in one batch; the checkpoints near the
// end of the prompt, which the next turn restores, stay. With the experts in RAM (-ncmoe) every prompt batch uploads
// them: the part before the last user message ran on its own (3090, 15.10: 8 tokens 0.18 s, 60 tokens 0.51 s in the
// generation layout; past 192 tokens it would be a second full upload of the experts, ~2.5 s). the 3090 (15.10 night,
// chat to 213K): the part ran on its own in 63 of 65 turns, 69 tokens and 0.66 s on average; without the break PP of
// the chat 756.1 against 647.4, turns of ~550 tokens after 29K 3.62 s against 4.41, TG the same
bool server_ckpt_user_break_fork();

// LLAMA_CKPT_SYSTEM_BREAK (03.11, default 1024 tokens, 0: upstream): a prompt that starts with a system message of at
// least this many tokens (the system block of the template, with the tools) is split at its end, so that a context
// checkpoint is made there, and of the checkpoints of a slot that one is evicted last (server_ckpt_evict_front_chain).
// a model that rolls back only to a checkpoint (hybrid: qwen4exp, SWA) then takes a long system prompt it shares with
// another conversation from that one's state: Qwen Code's Auto Mode asks the model about each gated tool call with
// ~12K tokens, ~2.9K of them the same system prompt of the classifier; new sessions and subagents of an agent share
// its system prompt and tools. a split adds a prompt batch where the system prompt is computed (with the experts in
// RAM, -ncmoe on the 3090, one more upload of them)
int32_t server_ckpt_system_break_fork();

// where the leading system block of a prompt ends (the start of the message after it, as the message delimiters of the
// chat template find it), -1: unknown
int32_t server_system_end_fork(const common_chat_msg_spans & spans);

// the same when the system break is on and that block is long enough, else -1
int32_t server_ckpt_system_end_fork(const common_chat_msg_spans & spans);

// LLAMA_CKPT_TAIL (16.11, 1 by default, 0: upstream): where the batch of a prompt stops for a context checkpoint near
// its end, as token counts, ascending. upstream stops 4 + n_ubatch and 4 tokens before the end while it adds text tokens,
// and makes no checkpoint after an image, so a prompt with an image over such a point gets none there. the 395, 14.11:
// an image over the first point left a prompt with no checkpoint near its end, and the next turn read 132669 tokens
// again (10 turns of 3.6 h went back further than the last prompt for it). with this:
// - a point inside or right after a run of images moves back to the start of the run: the checkpoint before an image
//   holds the text up to it, as any other one does
// - one more point n_tail tokens before the end: the tail of its last prompt that the request before this one rewrote
//   (a client that ends every request with the same images rewrites as much each turn), 0: none
std::vector<int32_t> server_ckpt_tail_fork(const server_tokens & tokens, int32_t n_ubatch, int32_t n_batch, int32_t n_tail);

// LLAMA_SLOT_ROOM (1 by default since 23.10, 0: upstream): the room of a unified KV cache (--kv-unified), where every
// slot may grow to the whole cache (agents, no --kv-unified-per-slot) but all of them together cannot. upstream
// clears the first idle slot when a decode finds no room and keeps nothing, and when the running slots alone do not
// fit, every one of them fails ("Context size has been exceeded") and loses its cache. with this:
// - the idle slot used least recently gives way, and the prompt cache keeps its state first (RAM, the disk with
//   --cache-spill-dir); a load from the prompt cache makes its room the same way before it restores
// - a new task whose prompt does not fit next to the running slots waits in the queue until one of them finishes
// - when the running slots outgrow the cache, the one whose task came last pauses: its state goes to RAM, its cells
//   are freed, and it continues once there is room again; nothing fails
// with --no-cache-idle-slots the idle conversations stay in the cache while they fit
bool server_slot_room_fork();

// LLAMA_PREFILL_BUDGET (23.10, default 2048, 0: n_batch as upstream): while slots generate, a step takes at most this
// many prompt tokens, so that the next token of the generating conversations does not wait for a whole batch of
// prompt (Rulith Inference, apply_agent_slots_027). with experts in RAM (the 3090) every prompt piece uploads them
int32_t server_prefill_budget_fork();

// LLAMA_PREFILL_EXCLUSIVE (default 1 since 03.11): generation pauses until all pending prompts finish, the prompts go in
// whole batches. 0 interleaves generation with prompt chunks limited by LLAMA_PREFILL_BUDGET: then a generating slot
// gets about a token per chunk (0.3 t/s on the 395 at depth) and the prompt runs slower too
bool server_prefill_exclusive_fork();

// LLAMA_PREFILL_GEN_WAIT, LLAMA_PREFILL_GEN_RUN (16.11, ms, 10000 and 3000 by default; wait 0: the prompts never give
// way): with LLAMA_PREFILL_EXCLUSIVE the generating slots wait for the prompts, but no longer than the first: then they
// generate for the second while the prompts wait, and so on in turns. a prompt shorter than the wait runs as before.
// without it an answer waited for the whole prompt of another slot, however long: on the Mac, 4 slots, a query of 3.5K
// tokens whose prompt was read at once got its answer after 80 s, when the prompts of 65K and 45K were done
int64_t server_prefill_gen_wait_us_fork();
int64_t server_prefill_gen_run_us_fork();

// LLAMA_PREFILL_ORDER (16.11, 1 by default, 0: the order of the slots): the prompts go into the batch in the order of
// the tokens they have left, the shortest first. in the order of the slots the first one filled every batch until its
// prompt was done: on the Mac, 4 slots, a query of 3.5K tokens waited 188 s behind a prompt of 125K
bool server_prefill_order_fork();

// LLAMA_SLOT_CONTINUE (23.10, 1 by default, 0: upstream's similarity): a slot takes a request for what it holds only
// when the request continues it, otherwise a free slot or the least recently used one
bool server_slot_continue_fork();

// whether a request of n_req tokens continues what a slot holds: it shares n_shared tokens with the slot, n_keep is the
// prompt of the slot's last task (the answer after it may come back rendered otherwise), sys_end where the system block
// of the request ends (server_system_end_fork, -1: not known). the request keeps that prompt less the 16 tokens a
// template may render otherwise once the answer is in the history, or (16.11) it rewrites a tail of it:
// LLAMA_SLOT_CONTINUE_TAIL (default 8192 tokens, 0: off), no more than half of what the prompt holds past the system
// block, and no more than the request brings after the shared part. the last two keep a new session with the same
// system prompt out of the slot of another one: where the template puts the system prompt into the first user message
// (gemma) the system block is not known, and such a session shares most of a short prompt (the pool test
// test_pool_measures_no_reuse_without_a_checkpoint). the 395, 14.11: a client that ends every request with the same
// three images (3259 tokens, the new text goes before them) went to the least recently used slot in 46 turns of
// 3.6 h; each of them saved the whole state first (median 0.73 s, the RAM cache gave back 272 GB in 51 evictions),
// and with several slots it would have left its slot
bool server_slot_continues_fork(size_t n_shared, size_t n_keep, size_t n_req, int32_t sys_end);

// LLAMA_SLOT_CONTINUE_TAIL: the longest tail of its last prompt a request may rewrite and still continue the slot
int32_t server_slot_tail_max_fork();

// LLAMA_SERVER_DEBUG_ROLLBACK_FAIL=N (tests, 0 by default): true at every Nth call. the server then leaves out the
// rollback of the rejected tokens of a draft, as if the memory had refused it (rollback_recover_fork)
bool server_debug_rollback_fail_fork();

// LLAMA_SLOT_ANSWER_ROOM (23.10, default 8192, 0: none): the room a running answer is counted at when a new task asks
// for room in a unified cache (server_slot_room_fork)
int32_t server_answer_room_fork();

// the room of a unified KV cache for the pool (GET /pool/node, server-pool-fork.cpp), set by the server loop at every
// step: the cells the running and paused slots hold with their answer reserve, counted as the admission of a new task
// counts them (room_for_task_fork), how many slots that is, the size of the cache and the margin of the admission.
// held -1: no room logic (no --kv-unified, LLAMA_SLOT_ROOM=0), a task never waits for room there
struct server_room_fork {
    int64_t held    = -1;
    int64_t n_slots = 0;
    int64_t n_ctx   = 0;
    int64_t margin  = 0;
};
void             server_room_report_fork(const server_room_fork & room);
server_room_fork server_room_get_fork();

// the work of the server since it started, for the pool (GET /pool/node "stats"), set by the server loop at every step
// from its metrics: the prompt tokens it computed and the time of it, the tokens it generated and the time, the prompt
// tokens it took from its cache, when it started (ggml_time_us)
struct server_stats_fork {
    uint64_t n_prompt    = 0;
    uint64_t t_prompt_us = 0;
    uint64_t n_gen       = 0;
    uint64_t t_gen_us    = 0;
    uint64_t n_cached    = 0;
    int64_t  t_start_us  = 0;
};
void              server_stats_report_fork(const server_stats_fork & stats);
server_stats_fork server_stats_get_fork();

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

// fork: --pool, the same without the machine (the model path and the file times): equal on another node of the same
// model and configuration, so that a cached state can move to it
uint64_t server_prompt_cache_signature_portable(
        const common_params & params,
        const llama_model * model_tgt,
        const llama_context * ctx_tgt,
        const llama_context * ctx_dft);

// context checkpoints of a slot: a base and the deltas after it form a chain
using server_ckpt_list = std::list<std::shared_ptr<const common_prompt_checkpoint>>; // shared since upstream e152332a1

// the checkpoint after `it` is a delta that depends on it, so `it` cannot be dropped
bool server_ckpt_has_dependent_delta(const server_ckpt_list & ckpts, server_ckpt_list::const_iterator it);

// Evict whole base->delta chains from the front. Deltas depend on all preceding
// checkpoints in their chain, so eviction can only drop a chain as a unit (never a
// checkpoint from the middle). Drops the oldest chain and its trailing deltas. Returns the bytes freed.
// fork: `keep` >= 0 (server_ckpt_system_end_fork): a front chain whose base is at `keep` tokens stays while another
// chain follows it, the next chain goes instead
size_t server_ckpt_evict_front_chain(server_ckpt_list & ckpts, int64_t keep = -1);

// base_pos for the checkpoint about to be appended: -1 for a self-contained one, else the
// pos_max of the last checkpoint, which the new delta hangs off. fork: -1 also when the last chain starts with the
// checkpoint at `keep` (server_ckpt_system_end_fork), which stays a chain of its own
llama_pos server_ckpt_next_base_pos(const server_ckpt_list & ckpts, const llama_model * model, int64_t keep = -1);

// evict chains from the front while the list is over `budget`, but never the newest chain.
// Returns the bytes freed. `keep` as in server_ckpt_evict_front_chain
size_t server_ckpt_evict_over_budget(server_ckpt_list & ckpts, size_t budget, int64_t keep = -1);

// restore the checkpoint at `it_cur`: its base, then the deltas up to it, then the draft state.
// Returns the checkpoint whose state is now in the context, nullptr on failure
const common_prompt_checkpoint * server_ckpt_restore(
        const server_ckpt_list & ckpts,
        server_ckpt_list::const_iterator it_cur,
        llama_context * ctx_tgt,
        llama_context * ctx_dft,
        llama_seq_id seq_id);

// LLAMA_MTMD_MIXED (16.11; -1 by default: on for qwen4exp, 1: every model, 0: upstream): the images of a prompt go into
// the batch as embeddings next to its text tokens, in place of a decode of their own each. a decode of a few tens of
// tokens or more costs a pass over all the experts whatever its size (the 395, 14.11, at 125K-220K: 29 tokens between
// two images 0.30 s, an image of 1058 tokens 1.49 s where as many text tokens take about 1.1 s), and a tail of three
// images with the text between them took six decodes. the logits agree with the decodes one by one as far as two
// splits of plain text into batches agree (Mac CPU: 1.2e-4 against 0.7e-4..2.8e-4); such a ubatch takes its mask from
// the KQ mask, as a ubatch of an image does (qsa_mask_pos_ubatch). an image that does not fit into the batch whole is
// decoded on its own as before
bool server_mtmd_mixed_fork(const llama_model * model);

// LLAMA_MTMD_EMBD_CACHE (16.11, MiB, default 256, 0: off): what the projector made of an image, kept by the hash of
// the image and its token count while it fits, the least recently used ones out first. an image that comes again at
// another place of the prompt (the 395, 14.11: the same three images end every request, the new text goes before
// them) was encoded again each time, 0.69 s for 1058 tokens there; its embeddings do not depend on where it stands
struct server_embd_cache_fork {
    server_embd_cache_fork();

    // the embeddings of `n` floats kept under the key, nullptr: none. they stay valid until the next put()
    const float * find(const std::string & key, size_t n);
    void put(const std::string & key, const float * embd, size_t n);

    size_t n_hit  = 0;
    size_t n_miss = 0;

private:
    struct entry {
        std::string        key;
        std::vector<float> embd;
    };
    std::list<entry> entries; // the most recently used first
    size_t           bytes = 0;
    size_t           limit = 0;
};

// --mmproj-on-demand: the device memory of an encode of the projector, from the moe-cache of the target context
// (llama_device_make_room_fork). user_data is the address of the server's llama_context pointer, read at the call
bool server_mmproj_device_room_fork(ggml_backend_dev_t dev, size_t bytes, void * user_data);
