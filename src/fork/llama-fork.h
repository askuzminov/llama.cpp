#pragma once

// The fork keeps its own versions of some upstream components in files of their own (name-fork.cpp, class name_fork),
// so that a merge of upstream touches the upstream files only. scripts/fork/copies.txt lists each copy with the
// upstream file and commit it was taken from.
//
// LLAMA_UPSTREAM picks the upstream version instead, per component, separated by commas: LLAMA_UPSTREAM=qwen4exp.
// "all" picks every upstream version. This is how a test compares the two.

// true if LLAMA_UPSTREAM lists the component (or "all")
bool llama_fork_upstream(const char * component);

#include <cstdint>

// a whole number of the environment, the fork's knobs: def when the variable is unset or empty, and def with an error
// line (once per variable) when its value is not a whole number. atoi read such a value as 0, which for most knobs means
// off or never: a typo in an arm changed the layout without a word
int64_t llama_fork_env_int(const char * name, int64_t def);

struct llama_model;
struct llama_memory_context_i;
struct llama_memory_i;
struct llama_ubatch;

// the fork's step before the graph of a ubatch runs (llama_context::process_ubatch), qwen4exp-fork.cpp: the PLE rows of
// the next ubatch are gathered while this one computes, for the last ubatch of a decode those of hint (the first
// ubatch of the next decode, llama_set_next_tokens) when given. true when the hint was taken
bool llama_fork_before_compute(const llama_model & model, llama_memory_context_i * mctx, const llama_ubatch * hint);

// the inputs of tokens a coming decode will hold are read from storage now (the PLE rows of qwen4exp,
// llama_prefetch_tokens): next holds the tokens from the first one not in the cache, only its last n_new are new
void llama_fork_prefetch_tokens(const llama_model & model, llama_memory_i * mem, const llama_ubatch & next, int64_t n_new);

// whether the graph of the model runs the deferred rollback of the gated delta net (llama_memory_recurrent::rec_l,
// LLAMA_GDN_LAZY): the fork's qwen4exp does, the upstream copy does not
bool llama_fork_gdn_lazy(const llama_model & model);

// LLAMA_KV_ELASTIC (llama_context::kv_elastic_init_fork): -1 (default) elastic with --phase-mem and the moe-cache, 1
// also elsewhere (emulated where the device has no elastic buffers), 0 off
int llama_kv_elastic_mode_fork();

// the KV caches built while one of these lives leave their tensors without memory, a buffer of size 0 marking them for
// the scheduler as with no_alloc. the context then places them in elastic buffers (llama_kv_cache::elastic_init_fork)
// or, where that does not take, allocates them in full (alloc_deferred_fork), so that the start does not hold the
// whole static cache next to the compute buffers before the elastic one replaces it (26.10). set with
// LLAMA_KV_ELASTIC=1 around the memory of the fork's qwen4exp, the one memory whose context does either
struct llama_kv_defer_scope_fork {
    explicit llama_kv_defer_scope_fork(bool on);
    ~llama_kv_defer_scope_fork();
    bool prev;
};
bool llama_kv_defer_alloc_fork();
