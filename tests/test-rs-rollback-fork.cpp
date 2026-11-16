// fork: the cells of the recurrent state that a decode with several sequences moves (llama_memory_recurrent::find_slot).
//
// 1. a partial rollback after a decode that held sequences of unequal length (seq_rm). The memory answers the rollback
// of a sequence from the snapshots of its last batch (or the records of the deferred rollback, LLAMA_GDN_LAZY). A decode
// with several sequences runs as several ubatches, and a later one can move the cell of a sequence that is done, or
// copy it along: its snapshots are gone then. The rollback must be refused, or it must leave the state the sequence
// had alone: until 16.11 it was accepted and read the plane of another sequence (the verify of one server slot in a
// decode with the prompt pieces of two others). The state is compared, not the logits: the generated model moves them
// too little.
//
// 2. a ubatch that fails (ubatch_restore). find_slot moves the cells before the graph of the ubatch moves their
// states. Until 16.11 a ubatch whose graph did not run (no memory for it: the "Compute error" of the server) left the
// cells moved: a sequence that was not in the ubatch, its cell between those of two that were, went on with the state
// of another sequence. Now a sequence the memory still holds after a failed ubatch is as it was before the ubatch, byte
// for byte, and the others are gone whole:
// - the graph did not run: every sequence stays, and the same decode after it gives the states of a context that never
//   failed. two places: before the graph is built (LLAMA_DEBUG_UBATCH_FAIL=1, no memory for it), and with its inputs
//   set (LLAMA_DEBUG_UBATCH_FAIL_INPUTS=1: an input that cannot be read or placed), when the memory has answered the
//   rollback that waited and moved the records of the deferred one
// - the graph ran in part (an abort after some of its nodes) or in full (LLAMA_DEBUG_UBATCH_FAIL=-1): the sequences
//   of the ubatch and the one between them are gone, the one past them stays
// - the ubatch went on inside an image (its patches share one position): its sequences are gone, the others stay
// - a failure reported with no cells moved since the last ubatch that was done: nothing is put back; one reported
//   twice is put back once
// - a rollback that waited when the ubatch failed before its graph still waits: the sequence is where the rollback
//   left it, and the decode after it gives the states of a context that never failed. the ubatch is the one of two
//   sequences, or a short one of the sequence alone: the step of one server slot after a rejected draft, which the
//   deferred rollback answers without moving the state (its inputs flip the record set of the cell)
// - the ubatch held a sequence the memory had not seen (the first ubatch of a prompt next to a sequence that goes on):
//   the cell apply() gave it is free again
// - the ubatch held a sequence that shared its cell with the one it was copied from (the children of a server task):
//   they share it again, or, once the graph ran, the copy is gone and the other one is as it was
//
// usage: test-rs-rollback-fork -m <dir>/qwen4exp-moe.gguf
//        test-rs-rollback-fork -m <dir>/<model>.gguf --failed
// --failed: part 2 alone, without the image, for the model of any memory: what a plain recurrent memory, a hybrid one,
// one with a window of attention or a plain attention cache holds after a failed ubatch

#include "llama.h"

#include "../src/llama-memory-hybrid.h"
#include "../src/llama-memory-hybrid-iswa.h"
#include "../src/llama-memory-recurrent.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int  g_fail  = 0;
static bool g_quiet = false; // while a decode fails as the test asks: its error lines are not news

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static constexpr uint32_t N_RS_SEQ = 7;
static constexpr int      N_SEQ    = 3;
static constexpr int      N_PROMPT = 16; // a long batch
static constexpr int      N_SHORT  = 4;  // a short batch: it leaves snapshots
static constexpr int      N_BACK   = 2;  // positions taken back

// the tokens of a sequence do not depend on what runs next to it
static std::vector<llama_token> tokens_of(int seq, int n) {
    std::mt19937 rng(1000 + seq);
    std::vector<llama_token> res(n);
    for (auto & t : res) {
        t = 3 + (llama_token) (rng() % 100);
    }
    return res;
}

struct piece {
    int seq;
    int n;
};

// one decode: the pieces in the order given, each sequence goes on from its position
static int decode_ret(llama_context * ctx, const std::vector<piece> & pieces, std::vector<int> & pos) {
    int n_all = 0;
    for (const piece & p : pieces) {
        n_all += p.n;
    }
    llama_batch batch = llama_batch_init(n_all, 0, 1);
    int i = 0;
    for (const piece & p : pieces) {
        const std::vector<llama_token> toks = tokens_of(p.seq, pos[p.seq] + p.n);
        for (int k = 0; k < p.n; ++k, ++i) {
            batch.token   [i]    = toks[pos[p.seq] + k];
            batch.pos     [i]    = pos[p.seq] + k;
            batch.n_seq_id[i]    = 1;
            batch.seq_id  [i][0] = p.seq;
            batch.logits  [i]    = k + 1 == p.n;
        }
        pos[p.seq] += p.n;
    }
    batch.n_tokens = n_all;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret;
}

static bool decode(llama_context * ctx, const std::vector<piece> & pieces, std::vector<int> & pos) {
    return decode_ret(ctx, pieces, pos) == 0;
}

static llama_context * make_ctx(llama_model * model, int n_seq = N_SEQ) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 1024;
    cp.n_batch    = 256;
    cp.n_ubatch   = 256;
    cp.n_seq_max  = n_seq;
    cp.kv_unified = true;
    cp.n_rs_seq   = N_RS_SEQ;
    cp.no_perf    = true;
    return llama_init_from_model(model, cp);
}

// the part of the state that cannot roll back by itself, a rollback that waits applied; with no flags the whole
// state of the sequence, the attention cells too
static std::vector<uint8_t> state_of(llama_context * ctx, int seq, llama_state_seq_flags flags = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) {
    std::vector<uint8_t> res(llama_state_seq_get_size_ext(ctx, seq, flags));
    if (res.empty() || llama_state_seq_get_data_ext(ctx, res.data(), res.size(), seq, flags) != res.size()) {
        res.clear();
    }
    return res;
}

// the words that differ read as floats: the states of one sequence computed in two batch layouts differ by rounding
static bool same_state(const std::vector<uint8_t> & a, const std::vector<uint8_t> & b) {
    if (a.empty() || a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i + 4 <= a.size(); i += 4) {
        if (memcmp(a.data() + i, b.data() + i, 4) == 0) {
            continue;
        }
        float x;
        float y;
        memcpy(&x, a.data() + i, 4);
        memcpy(&y, b.data() + i, 4);
        if (!std::isfinite(x) || !std::isfinite(y) || std::fabs(x - y) > 1e-3f*std::max(1.0f, std::max(std::fabs(x), std::fabs(y)))) {
            return false;
        }
    }
    return true;
}

// the sequence `a` alone: its prompt, a short batch, the verify batch, the rollback
static std::vector<uint8_t> run_alone(llama_model * model, int a) {
    llama_context * ctx = make_ctx(model);
    if (ctx == nullptr) {
        return {};
    }
    std::vector<int> pos(N_SEQ, 0);
    bool ok = decode(ctx, {{a, N_PROMPT}}, pos) && decode(ctx, {{a, N_SHORT}}, pos) && decode(ctx, {{a, N_SHORT}}, pos);
    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx), a, pos[a] - N_BACK, -1);
    CHECK(ok); // one sequence: the rollback of a short batch works
    std::vector<uint8_t> res = ok ? state_of(ctx, a) : std::vector<uint8_t>();
    llama_free(ctx);
    return res;
}

// 0: the rollback was refused, 1: accepted and right, -1: accepted and wrong
static int run_joint(llama_model * model, int a, const std::vector<uint8_t> & ref) {
    llama_context * ctx = make_ctx(model);
    if (ctx == nullptr) {
        return -1;
    }
    std::vector<int> pos(N_SEQ, 0);
    bool ok = true;
    // every sequence in its cell, each with snapshots of a short batch
    for (int s = 0; s < N_SEQ && ok; ++s) {
        ok = decode(ctx, {{s, N_PROMPT}}, pos) && decode(ctx, {{s, N_SHORT}}, pos);
    }
    // the verify of `a` with the prompt pieces of the two others, the sequences in the order of their cells
    std::vector<piece> pieces;
    for (int s = 0; s < N_SEQ; ++s) {
        pieces.push_back({s, s == a ? N_SHORT : 12 + 8*((s + 1) % N_SEQ)});
    }
    ok = ok && decode(ctx, pieces, pos);
    CHECK(ok);

    int res = -1;
    if (ok) {
        if (!llama_memory_seq_rm(llama_get_memory(ctx), a, pos[a] - N_BACK, -1)) {
            res = 0;
        } else {
            res = same_state(state_of(ctx, a), ref) ? 1 : -1;
        }
    }
    llama_free(ctx);
    return res;
}

//
// a ubatch that fails
//

// the sequences 0 and 2 decode together: the cell of 1 lies between theirs and moves, the cell of 3 lies past them.
// a plain recurrent memory puts only sequences that follow each other into one ubatch (split_equal, sequential): for
// --failed the sequences take their cells in the order 0, 2, 1, 3, and 0 and 1 decode together around the cell of 2.
// either way the graph of the decode reaches the cells of the sequences 0, 1 and 2
static constexpr int N_SEQ_F = 4;

static int g_order[N_SEQ_F] = { 0, 1, 2, 3 }; // the order the sequences take their cells in
static int g_joint[2]       = { 0, 2 };       // the sequences of the decode that fails

// where a ubatch whose graph does not run fails: before the graph is built, or with its inputs set
static bool g_inputs = false;

static int ret_not_run() {
    return g_inputs ? -3 : -2;
}
static constexpr int N_PATCH = 6; // patches a batch: short, so its snapshots would let a rollback of one position through

struct abort_at {
    int n     = 0;       // nodes computed
    int limit = INT_MAX; // the abort comes after one more than this
};

static bool abort_cb(void * data) {
    abort_at * at = (abort_at *) data;
    return ++at->n > at->limit;
}

// what a context that never failed holds, per sequence: after the prompts and after the decode of 0 and 2 that follows,
// the whole state and the part that cannot roll back by itself
struct fail_ref {
    std::vector<std::vector<uint8_t>> before;
    std::vector<std::vector<uint8_t>> before_part;
    std::vector<std::vector<uint8_t>> after;
    std::vector<std::vector<uint8_t>> after_part;

    size_t size_empty = 0; // the whole state of a sequence that holds nothing
    int    n_nodes    = 0; // the nodes of the graph of that decode, as the abort callback counts them
};

// four sequences, each in a cell of its own with a short batch last: it leaves snapshots, or the records of the deferred
// rollback. knob: LLAMA_DEBUG_UBATCH_FAIL, read when a context is made. skip: a sequence the memory does not see.
// every: the knob counts every ubatch, not only those of two sequences
static llama_context * fail_ctx(llama_model * model, const char * knob, std::vector<int> & pos, int skip = -1, bool every = false) {
    set_env("LLAMA_DEBUG_UBATCH_FAIL", knob);
    set_env("LLAMA_DEBUG_UBATCH_FAIL_INPUTS", g_inputs ? "1" : "0");
    set_env("LLAMA_DEBUG_UBATCH_FAIL_SEQS", every ? "1" : "2");
    llama_context * ctx = make_ctx(model, N_SEQ_F);
    set_env("LLAMA_DEBUG_UBATCH_FAIL", "0");
    if (ctx == nullptr) {
        return nullptr;
    }
    pos.assign(N_SEQ_F, 0);
    for (const int s : g_order) {
        if (s == skip) {
            continue;
        }
        if (!decode(ctx, {{s, N_PROMPT}}, pos) || !decode(ctx, {{s, N_SHORT}}, pos)) {
            llama_free(ctx);
            return nullptr;
        }
    }
    return ctx;
}

// the recurrent memory of the context, alone or next to an attention cache; null: the memory has none
static const llama_memory_recurrent * recr_of(llama_context * ctx) {
    llama_memory_t mem = llama_get_memory(ctx);
    if (const auto * m = dynamic_cast<const llama_memory_recurrent *>(mem)) {
        return m;
    }
    if (const auto * m = dynamic_cast<const llama_memory_hybrid *>(mem)) {
        return m->get_mem_recr();
    }
    if (const auto * m = dynamic_cast<const llama_memory_hybrid_iswa *>(mem)) {
        return m->get_mem_recr();
    }
    return nullptr;
}

static uint32_t recr_used(const llama_memory_recurrent * recr) {
    return (uint32_t) std::count_if(recr->cells.begin(), recr->cells.end(), [](const auto & cell) { return !cell.is_empty(); });
}

static int decode_joint(llama_context * ctx, std::vector<int> & pos) {
    return decode_ret(ctx, {{g_joint[0], N_SHORT}, {g_joint[1], N_SHORT}}, pos);
}

// n patches of one image for the sequences of the decode that fails, from patch k0 on: they all share the position pos
static int decode_patches(llama_context * ctx, const llama_model * model, llama_pos pos, int k0, int n) {
    const int n_embd = llama_model_n_embd_inp(model);

    std::mt19937 rng(7 + k0);
    std::normal_distribution<float> nd(0.0f, 0.05f);
    std::vector<float> embd((size_t) n * n_embd);
    for (float & v : embd) {
        v = nd(rng);
    }

    llama_batch_ext * batch = llama_batch_ext_init(ctx);
    for (const int s : g_joint) {
        for (int k = 0; k < n; ++k) {
            const int32_t   idx  = llama_batch_ext_add_embd(batch, s, { embd.data() + (size_t) k * n_embd, 1, (size_t) n_embd });
            const llama_pos p[4] = { pos, pos + (k0 + k) / 8, pos + (k0 + k) % 8, 0 };
            llama_batch_ext_set_pos(batch, idx, p);
            llama_batch_ext_set_output_logits(batch, idx, false);
        }
    }
    const int ret = llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch);
    llama_batch_ext_free(batch);
    return ret;
}

static bool fail_ref_make(llama_model * model, fail_ref & ref) {
    llama_context * ctx = make_ctx(model, N_SEQ_F);
    if (ctx == nullptr) {
        return false;
    }
    ref.size_empty = llama_state_seq_get_size_ext(ctx, 0, LLAMA_STATE_SEQ_FLAGS_NONE);
    llama_free(ctx);

    std::vector<int> pos;
    ctx = fail_ctx(model, "0", pos);
    if (ctx == nullptr) {
        return false;
    }
    for (int s = 0; s < N_SEQ_F; ++s) {
        ref.before     .push_back(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE));
        ref.before_part.push_back(state_of(ctx, s));
    }
    llama_free(ctx);

    ctx = fail_ctx(model, "0", pos);
    if (ctx == nullptr) {
        return false;
    }
    abort_at at;
    llama_set_abort_callback(ctx, abort_cb, &at);
    bool ok = decode_joint(ctx, pos) == 0;
    llama_set_abort_callback(ctx, nullptr, nullptr);
    ref.n_nodes = at.n;
    for (int s = 0; s < N_SEQ_F; ++s) {
        ref.after     .push_back(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE));
        ref.after_part.push_back(state_of(ctx, s));
        ok = ok && !ref.before[s].empty() && !ref.before_part[s].empty() && !ref.after[s].empty() && !ref.after_part[s].empty();
    }
    llama_free(ctx);

    return ok && ref.n_nodes > 16;
}

// after a failed ubatch a sequence the memory still holds is as it was before the ubatch, byte for byte, and one it
// does not hold is gone whole. must: the sequences that have to be there. returns the ones that are
static std::vector<bool> fail_check(llama_context * ctx, const fail_ref & ref, const std::vector<int> & pos, const std::vector<bool> & must) {
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<bool> kept(N_SEQ_F);
    for (int s = 0; s < N_SEQ_F; ++s) {
        kept[s] = llama_memory_seq_pos_max(mem, s) >= 0;
        CHECK(kept[s] || !must[s]);
        if (kept[s]) {
            CHECK(llama_memory_seq_pos_max(mem, s) == pos[s] - 1);
            CHECK(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.before[s]);
        } else {
            CHECK(llama_state_seq_get_size_ext(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.size_empty);
        }
    }
    return kept;
}

// the memory takes the sequences it lost again
static bool fail_read_again(llama_context * ctx, const fail_ref & ref, std::vector<int> & pos, const std::vector<bool> & kept) {
    bool ok = true;
    for (int s = 0; s < N_SEQ_F; ++s) {
        if (!kept[s]) {
            pos[s] = 0;
            ok = ok && decode(ctx, {{s, N_PROMPT}}, pos) && decode(ctx, {{s, N_SHORT}}, pos);
            CHECK(ok && same_state(state_of(ctx, s), ref.before_part[s]));
        }
    }
    return ok;
}

// the graph of the ubatch does not run. retry: the same decode right after it, against the context that never failed
static void fail_before_graph(llama_model * model, const fail_ref & ref, bool retry) {
    std::vector<int> pos;
    llama_context * ctx = fail_ctx(model, "1", pos);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return;
    }

    // where the memory places the next ubatch, as the last ubatch that was done left it
    const llama_memory_recurrent * recr = recr_of(ctx);
    const uint32_t head = recr ? recr->head : 0;

    std::vector<int> pos_try = pos;
    g_quiet = true;
    CHECK(decode_joint(ctx, pos_try) == ret_not_run());
    g_quiet = false;

    CHECK(recr == nullptr || recr->head == head);

    if (!retry) {
        fail_check(ctx, ref, pos, std::vector<bool>(N_SEQ_F, true));

        // a failure reported twice is put back once
        llama_get_memory(ctx)->ubatch_failed(false);
        fail_check(ctx, ref, pos, std::vector<bool>(N_SEQ_F, true));
    } else {
        CHECK(decode_joint(ctx, pos) == 0);
        for (int s = 0; s < N_SEQ_F; ++s) {
            CHECK(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.after[s]);
        }
    }
    llama_free(ctx);
}

// the graph of the ubatch ran: in part (an abort after n_abort + 1 nodes) or in full (n_abort < 0). returns how many
// sequences the memory kept
static int fail_after_graph(llama_model * model, const fail_ref & ref, int n_abort) {
    std::vector<int> pos;
    llama_context * ctx = fail_ctx(model, n_abort < 0 ? "-1" : "0", pos);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return 0;
    }

    abort_at at;
    at.limit = n_abort < 0 ? INT_MAX : n_abort;
    llama_set_abort_callback(ctx, abort_cb, &at);
    std::vector<int> pos_try = pos;
    g_quiet = true;
    const int ret = decode_joint(ctx, pos_try);
    g_quiet = false;
    llama_set_abort_callback(ctx, nullptr, nullptr);
    CHECK(ret == (n_abort < 0 ? -3 : 2));

    // the graph does not reach the cell of sequence 3
    const std::vector<bool> kept = fail_check(ctx, ref, pos, { false, false, false, true });

    // with the lost ones read again the decode gives the states of the context that never failed
    bool ok = fail_read_again(ctx, ref, pos, kept);
    ok = ok && decode_joint(ctx, pos) == 0;
    CHECK(ok);
    for (int s = 0; s < N_SEQ_F && ok; ++s) {
        CHECK(same_state(state_of(ctx, s), ref.after_part[s]));
    }
    llama_free(ctx);

    return (int) std::count(kept.begin(), kept.end(), true);
}

// a ubatch that fails before the memory applies it reports its failure too: the cells are those of the last ubatch
// that was done, whose snapshot must not come back
static void fail_before_apply(llama_model * model, const fail_ref & ref) {
    std::vector<int> pos;
    llama_context * ctx = fail_ctx(model, "0", pos);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return;
    }

    llama_get_memory(ctx)->ubatch_failed(false);

    fail_check(ctx, ref, pos, std::vector<bool>(N_SEQ_F, true));
    llama_free(ctx);
}

// a rollback that waits when the ubatch fails before its graph: the tail of a short batch taken back is answered by the
// next decode of the sequence, from the snapshots or the records of that batch. apply() of the ubatch takes the
// rollback of the sequences it holds and the snapshots of a cell it moves, so its failure must give them back: the
// first sequence of the decode and the one between take two positions back first. single: the decode is a short batch
// of the first sequence alone, in its own cell: the ninth ubatch of the context, after the two of each sequence.
// -1: the memory does not take such a rollback, else the checks that failed
static int fail_pending(llama_model * model, bool single) {
    const int seqs[2] = { g_joint[0], g_order[1] };
    const int n_fail  = g_fail;

    const auto decode_it = [&](llama_context * ctx, std::vector<int> & pos) {
        return single ? decode_ret(ctx, {{g_joint[0], N_SHORT}}, pos) : decode_joint(ctx, pos);
    };

    std::vector<std::vector<uint8_t>> before;
    std::vector<std::vector<uint8_t>> after;
    std::vector<int>                  pos;

    // the context that never fails first, then the one whose decode fails once
    for (const char * knob : { "0", single ? "9" : "1" }) {
        llama_context * ctx = fail_ctx(model, knob, pos, -1, single);
        CHECK(ctx != nullptr);
        if (ctx == nullptr) {
            return g_fail - n_fail;
        }
        llama_memory_t mem = llama_get_memory(ctx);

        for (const int s : seqs) {
            if (!llama_memory_seq_rm(mem, s, pos[s] - N_BACK, -1)) {
                llama_free(ctx);
                return -1;
            }
            pos[s] -= N_BACK;
        }

        const bool fails = strcmp(knob, "0") != 0;
        if (fails) {
            std::vector<int> pos_try = pos;
            g_quiet = true;
            CHECK(decode_it(ctx, pos_try) == ret_not_run());
            g_quiet = false;
        }
        for (int s = 0; s < N_SEQ_F; ++s) {
            CHECK(llama_memory_seq_pos_max(mem, s) == pos[s] - 1);
            if (!fails) {
                before.push_back(state_of(ctx, s));
                CHECK(!before.back().empty());
            } else {
                CHECK(state_of(ctx, s) == before[s]);
            }
        }

        CHECK(decode_it(ctx, pos) == 0);
        for (int s = 0; s < N_SEQ_F; ++s) {
            if (!fails) {
                after.push_back(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE));
                CHECK(!after.back().empty());
            } else {
                CHECK(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == after[s]);
            }
        }
        llama_free(ctx);
    }

    return g_fail - n_fail;
}

// a short ubatch of one sequence in its own cell fails after its graph ran: the graph reaches that cell alone, the
// sequence is gone and the three others are as they were
static void fail_single_after(llama_model * model, const fail_ref & ref) {
    const int seq = g_joint[0];

    std::vector<std::vector<uint8_t>> after(N_SEQ_F);
    std::vector<int>                  pos;

    for (const bool fails : { false, true }) {
        llama_context * ctx = fail_ctx(model, fails ? "-9" : "0", pos, -1, true);
        CHECK(ctx != nullptr);
        if (ctx == nullptr) {
            return;
        }
        llama_memory_t mem = llama_get_memory(ctx);

        if (fails) {
            std::vector<int> pos_try = pos;
            g_quiet = true;
            CHECK(decode_ret(ctx, {{seq, N_SHORT}}, pos_try) == -3);
            g_quiet = false;

            const bool gone = recr_of(ctx) != nullptr;
            for (int s = 0; s < N_SEQ_F; ++s) {
                if (s == seq && gone) {
                    CHECK(llama_memory_seq_pos_max(mem, s) < 0);
                    CHECK(llama_state_seq_get_size_ext(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.size_empty);
                } else {
                    CHECK(llama_memory_seq_pos_max(mem, s) == pos[s] - 1);
                    CHECK(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.before[s]);
                }
            }
            if (gone) {
                pos[seq] = 0;
                CHECK(decode(ctx, {{seq, N_PROMPT}}, pos) && decode(ctx, {{seq, N_SHORT}}, pos));
            }
        }

        CHECK(decode(ctx, {{seq, N_SHORT}}, pos));
        for (int s = 0; s < N_SEQ_F; ++s) {
            if (!fails) {
                after[s] = state_of(ctx, s);
                CHECK(!after[s].empty());
            } else {
                CHECK(same_state(state_of(ctx, s), after[s]));
            }
        }
        llama_free(ctx);
    }
}

// the ubatch that fails holds a sequence the memory has not seen: the first ubatch of a prompt next to a sequence that
// goes on. apply() gives the new sequence a cell, the failure takes it back: the sequence is not there, the others are
// as they were or, once the graph ran, gone whole, and the cells in use are counted as before
static void fail_new(llama_model * model, const fail_ref & ref, bool started) {
    const int seq_new = g_joint[1];

    std::vector<std::vector<uint8_t>> before(N_SEQ_F);
    std::vector<std::vector<uint8_t>> after (N_SEQ_F);
    std::vector<int>                  pos;

    // the context that never fails first, then the one whose decode fails once
    for (const bool fails : { false, true }) {
        llama_context * ctx = fail_ctx(model, !fails ? "0" : started ? "-1" : "1", pos, seq_new);
        CHECK(ctx != nullptr);
        if (ctx == nullptr) {
            return;
        }
        llama_memory_t                 mem  = llama_get_memory(ctx);
        const llama_memory_recurrent * recr = recr_of(ctx);
        const uint32_t                 used = recr ? recr->used : 0;

        if (!fails) {
            for (int s = 0; s < N_SEQ_F; ++s) {
                if (s != seq_new) {
                    before[s] = state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE);
                    CHECK(!before[s].empty());
                }
            }
        } else {
            std::vector<int> pos_try = pos;
            g_quiet = true;
            CHECK(decode_joint(ctx, pos_try) == (started ? -3 : ret_not_run()));
            g_quiet = false;

            CHECK(llama_memory_seq_pos_max(mem, seq_new) < 0);
            CHECK(llama_state_seq_get_size_ext(ctx, seq_new, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.size_empty);

            uint32_t n_kept = 0;
            for (int s = 0; s < N_SEQ_F; ++s) {
                if (s == seq_new) {
                    continue;
                }
                if (llama_memory_seq_pos_max(mem, s) >= 0) {
                    n_kept++;
                    CHECK(llama_memory_seq_pos_max(mem, s) == pos[s] - 1);
                    CHECK(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == before[s]);
                } else {
                    // the graph ran: read again
                    CHECK(started);
                    CHECK(llama_state_seq_get_size_ext(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.size_empty);
                    pos[s] = 0;
                    CHECK(decode(ctx, {{s, N_PROMPT}}, pos) && decode(ctx, {{s, N_SHORT}}, pos));
                }
            }
            if (recr != nullptr) {
                CHECK(recr->used == recr_used(recr));
                CHECK(started || recr->used == used);
                CHECK(started || n_kept == N_SEQ_F - 1);
            }
        }

        // the decode that failed, or that never did: the new sequence takes its cell
        CHECK(decode_joint(ctx, pos) == 0);
        for (int s = 0; s < N_SEQ_F; ++s) {
            if (!fails) {
                after[s] = state_of(ctx, s);
                CHECK(!after[s].empty());
            } else {
                CHECK(same_state(state_of(ctx, s), after[s]));
            }
        }
        llama_free(ctx);
    }
}

// the ubatch that fails holds a sequence that shares its cell with the one it was copied from and has not decoded
// since (the children of a server task): apply() gives the copy a cell of its own, past the others, so the graph
// reaches that cell alone. the graph did not run: the two share the cell again. it ran: the copy is gone, as every
// sequence of such a ubatch, and the one it was copied from is as it was, its row was only read. a memory without a
// recurrent state keeps the copy either way
static void fail_shared(llama_model * model, const fail_ref & ref, bool started) {
    const int seq_cpy = g_joint[1];
    const int seq_src = g_order[N_SEQ_F - 1];

    std::vector<std::vector<uint8_t>> after(N_SEQ_F);
    std::vector<int>                  pos;

    // the context that never fails first, then the one whose decode fails once: the seventh ubatch, after the two of
    // each of the three sequences
    for (const bool fails : { false, true }) {
        llama_context * ctx = fail_ctx(model, !fails ? "0" : started ? "-7" : "7", pos, seq_cpy, true);
        CHECK(ctx != nullptr);
        if (ctx == nullptr) {
            return;
        }
        llama_memory_t mem = llama_get_memory(ctx);

        llama_memory_seq_cp(mem, seq_src, seq_cpy, -1, -1);
        pos[seq_cpy] = pos[seq_src];

        if (fails) {
            std::vector<std::vector<uint8_t>> before;
            for (int s = 0; s < N_SEQ_F; ++s) {
                before.push_back(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE));
                CHECK(!before.back().empty());
            }

            std::vector<int> pos_try = pos;
            g_quiet = true;
            CHECK(decode_ret(ctx, {{seq_cpy, N_SHORT}}, pos_try) == (started ? -3 : ret_not_run()));
            g_quiet = false;

            const bool gone = started && recr_of(ctx) != nullptr;
            for (int s = 0; s < N_SEQ_F; ++s) {
                if (s == seq_cpy && gone) {
                    CHECK(llama_memory_seq_pos_max(mem, s) < 0);
                    CHECK(llama_state_seq_get_size_ext(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == ref.size_empty);
                } else {
                    CHECK(llama_memory_seq_pos_max(mem, s) == pos[s] - 1);
                    CHECK(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == before[s]);
                }
            }
            if (gone) {
                llama_memory_seq_cp(mem, seq_src, seq_cpy, -1, -1);
            }
        }

        CHECK(decode(ctx, {{seq_cpy, N_SHORT}}, pos));
        for (int s = 0; s < N_SEQ_F; ++s) {
            if (!fails) {
                after[s] = state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE);
                CHECK(!after[s].empty());
            } else {
                CHECK(state_of(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == after[s]);
            }
        }
        llama_free(ctx);
    }
}

// the ubatch that fails goes on inside an image: its positions do not come after the state of its sequences, and the
// memory cannot take them back alone
static int fail_in_image(llama_model * model, const fail_ref & ref) {
    std::vector<int> pos;
    llama_context * ctx = fail_ctx(model, "2", pos);
    CHECK(ctx != nullptr);
    if (ctx == nullptr) {
        return 0;
    }

    CHECK(decode_patches(ctx, model, pos[0], 0,       N_PATCH) ==  0);
    g_quiet = true;
    CHECK(decode_patches(ctx, model, pos[0], N_PATCH, N_PATCH) == -2);
    g_quiet = false;

    // the first batch of patches moved the cell of sequence 1
    const std::vector<bool> kept = fail_check(ctx, ref, pos, { false, true, false, true });
    fail_read_again(ctx, ref, pos, kept);
    llama_free(ctx);

    return (int) std::count(kept.begin(), kept.end(), true);
}

int main(int argc, char ** argv) {
    const char * model_path  = nullptr;
    bool         failed_only = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--failed") == 0) {
            failed_only = true;
        }
    }
    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s -m <qwen4exp-moe.gguf> | -m <model.gguf> --failed\n", argv[0]);
        return 1;
    }
    if (failed_only) {
        const int order[N_SEQ_F] = { 0, 2, 1, 3 };
        std::copy(order, order + N_SEQ_F, g_order);
        g_joint[1] = 1;
    }

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR && !g_quiet) {
            fputs(text, stderr);
        }
    }, nullptr);
    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", model_path);
        llama_backend_free();
        return 1;
    }

    // the knob is read when a context makes its memory
    for (const char * lazy : { "0", "1" }) {
        set_env("LLAMA_GDN_LAZY", lazy);
        int n_right = 0;
        for (int a = 0; a < N_SEQ && !failed_only; ++a) {
            const std::vector<uint8_t> ref = run_alone(model, a);
            CHECK(!ref.empty());
            const int res = run_joint(model, a, ref);
            printf("LLAMA_GDN_LAZY=%s, the verify of sequence %d next to two prompts: the rollback %s\n", lazy, a,
                    res == 0 ? "refused" : res == 1 ? "accepted, the state as alone" : "accepted, ANOTHER STATE");
            CHECK(res >= 0);
            n_right += res == 1;
        }
        // not every layout loses the snapshots: the first cell is never moved or copied along
        CHECK(failed_only || n_right >= 1 || strcmp(lazy, "1") == 0);

        fail_ref ref;
        const bool ref_ok = fail_ref_make(model, ref);
        CHECK(ref_ok);
        if (!ref_ok) {
            continue;
        }

        int n_fail = g_fail;
        for (const bool inputs : { false, true }) {
            g_inputs = inputs;
            const char * stage = inputs ? "with its inputs set" : "before its graph";

            n_fail = g_fail;
            fail_before_graph(model, ref, false);
            fail_before_graph(model, ref, true);
            printf("LLAMA_GDN_LAZY=%s, the ubatch of sequences %d and %d fails %s: %s\n", lazy, g_joint[0], g_joint[1], stage,
                    g_fail == n_fail ? "the four sequences as they were, the decode after it as without a failure" : "WRONG");

            n_fail = g_fail;
            fail_new(model, ref, false);
            printf("LLAMA_GDN_LAZY=%s, it holds a sequence the memory has not seen (%d) and fails %s: %s\n", lazy, g_joint[1], stage,
                    g_fail == n_fail ? "the sequence is not there, the others as they were" : "WRONG");

            n_fail = g_fail;
            fail_shared(model, ref, false);
            printf("LLAMA_GDN_LAZY=%s, a ubatch of sequence %d, a copy of %d in its cell, fails %s: %s\n", lazy, g_joint[1], g_order[N_SEQ_F - 1], stage,
                    g_fail == n_fail ? "the two share the cell as before" : "WRONG");

            for (const bool single : { false, true }) {
                const int n_pending = fail_pending(model, single);
                printf("LLAMA_GDN_LAZY=%s, %s fails %s while the rollback of sequences %d and %d waits: %s\n", lazy,
                        single ? "a short ubatch of the first one alone" : "it", stage, g_joint[0], g_order[1],
                        n_pending < 0 ? "the memory takes no such rollback" : n_pending == 0 ? "the rollback still waits, the decode after it as without a failure" : "WRONG");
            }
        }
        g_inputs = false;

        n_fail = g_fail;
        fail_before_apply(model, ref);
        printf("LLAMA_GDN_LAZY=%s, a failure before the cells of a ubatch moved: %s\n", lazy,
                g_fail == n_fail ? "the four sequences as they were" : "WRONG");

        // the first node of the graph, the last, eight between; then the whole graph
        n_fail = g_fail;
        int n_kept = 0;
        int n_runs = 0;
        for (int i = 0; i <= 9; ++i, ++n_runs) {
            n_kept += fail_after_graph(model, ref, (int) ((int64_t) i * (ref.n_nodes - 1) / 9));
        }
        n_kept += fail_after_graph(model, ref, -1);
        n_runs++;
        printf("LLAMA_GDN_LAZY=%s, it fails after a part of its graph (%d aborts over %d nodes) and after all of it: %d of %d sequences kept%s\n",
                lazy, n_runs - 1, ref.n_nodes, n_kept, n_runs * N_SEQ_F, g_fail == n_fail ? ", each as it was, the others gone whole" : ", WRONG");

        n_fail = g_fail;
        fail_single_after(model, ref);
        fail_new(model, ref, true);
        fail_shared(model, ref, true);
        printf("LLAMA_GDN_LAZY=%s, a ubatch of one sequence, one with a sequence the memory has not seen and one of a copy in the cell of another fail after their graphs: %s\n", lazy,
                g_fail == n_fail ? "each sequence as it was or gone whole" : "WRONG");

        if (failed_only) {
            continue;
        }

        n_fail = g_fail;
        n_kept = fail_in_image(model, ref);
        printf("LLAMA_GDN_LAZY=%s, a ubatch of 0 and 2 that goes on inside an image fails before its graph: %d of %d sequences kept%s\n",
                lazy, n_kept, N_SEQ_F, g_fail == n_fail ? ", each as it was, the others gone whole" : ", WRONG");
    }

    llama_model_free(model);
    llama_backend_free();

    if (g_fail > 0) {
        fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
