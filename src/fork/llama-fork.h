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
