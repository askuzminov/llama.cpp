// fork: a ubatch that fails on the DSV4 memory (llama_kv_cache_dsv4::ubatch_save, ubatch_failed). The compressor state
// of a sequence goes back over snapshots that the graph of its last ubatch writes, and the rollback bookkeeping moved
// on before any graph ran. Until 16.11 the positions of a failed ubatch were taken back as the rollback of a draft:
// the sequence then went on from the compressor state of an older ubatch (another state after the same tokens), a
// ubatch longer than the snapshots stayed in the cache with nothing computed, and a batch that stopped early took the
// pending rollback of the sequences of its later ubatches. Now:
// - the graph did not run (LLAMA_DEBUG_UBATCH_FAIL: before the graph is built, and with LLAMA_DEBUG_UBATCH_FAIL_INPUTS
//   with its inputs set): the sequence is where it was, and the same tokens decoded after it give the state of a
//   context that never failed, byte for byte; a rollback that waited before the batch still waits; the ubatches of the
//   batch that ran before stay
// - the graph ran in part (an abort after some of its nodes) or in full: the sequences of the ubatch are gone whole
// - a sequence that was not in the ubatch is as it was in every case
// - a sequence the memory had not seen (the first ubatch of a prompt) is not there after either failure
// - a failure reported with no ubatch applied since the last one that was done takes nothing back, and one reported
//   twice is taken back once
// - a rollback asked after the failure goes over the snapshots of the last ubatch that was done: it is taken or
//   refused as in a context that never failed
// The rollback that waits is one of a draft longer than the ring of the compressor state: the rejected tokens wrote
// over rows of tokens that stay, so a rollback that is lost shows in the next compressed row.
//
// usage: test-dsv4-failed-fork -m <dir>/deepseek4-moe.gguf

#include "llama.h"

#include "../src/llama-memory.h"

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int  g_fail  = 0;
static bool g_quiet = false; // while a decode fails as the test asks

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

static constexpr int N_SEQ    = 3;
static constexpr int N_PROMPT = 16;
static constexpr int N_SHORT  = 4; // leaves snapshots
static constexpr int N_DRAFT  = 7; // a draft that wraps the ring of the compressor state
static constexpr int N_BACK   = 6; // of which this many are taken back
static constexpr int N_TAIL   = 6; // tokens the sequence takes at the end

// the ubatches of before(): the failure is counted over every ubatch of the context. n_seq: the sequences with a prompt
static int n_before(int n_ubatch, bool draft, int n_seq) {
    return n_seq * ((N_PROMPT + n_ubatch - 1) / n_ubatch + (N_SHORT + n_ubatch - 1) / n_ubatch) + (draft ? 1 : 0);
}

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

// one decode: the pieces in the order given, each sequence goes on from its position. pos moves on only when it worked
static int decode(llama_context * ctx, const std::vector<piece> & pieces, std::vector<int> & pos) {
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
    }
    batch.n_tokens = n_all;
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (ret == 0) {
        for (const piece & p : pieces) {
            pos[p.seq] += p.n;
        }
    }
    return ret;
}

// fail: LLAMA_DEBUG_UBATCH_FAIL, counted over every ubatch of the context. inputs: a ubatch whose graph does not run
// fails with its inputs set, not before its graph is built. the knobs are read when the context is made
static llama_context * make_ctx(llama_model * model, int fail, int n_ubatch = 256, bool inputs = false) {
    set_env("LLAMA_DEBUG_UBATCH_FAIL", std::to_string(fail).c_str());
    set_env("LLAMA_DEBUG_UBATCH_FAIL_SEQS", "1");
    set_env("LLAMA_DEBUG_UBATCH_FAIL_INPUTS", inputs ? "1" : "0");

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx      = 256;
    cp.n_batch    = 256;
    cp.n_ubatch   = n_ubatch;
    cp.n_seq_max  = N_SEQ;
    cp.kv_unified = true;
    cp.n_rs_seq   = 7;
    cp.no_perf    = true;
    llama_context * ctx = llama_init_from_model(model, cp);

    set_env("LLAMA_DEBUG_UBATCH_FAIL", "0");
    return ctx;
}

// every sequence: a prompt, then a short batch. skip: a sequence the memory does not see
static bool setup(llama_context * ctx, std::vector<int> & pos, int skip = -1) {
    pos.assign(N_SEQ, 0);
    for (int s = 0; s < N_SEQ; ++s) {
        if (s == skip) {
            continue;
        }
        if (decode(ctx, {{s, N_PROMPT}}, pos) != 0 || decode(ctx, {{s, N_SHORT}}, pos) != 0) {
            return false;
        }
    }
    return true;
}

static std::vector<uint8_t> state_of(llama_context * ctx, int seq) {
    std::vector<uint8_t> res(llama_state_seq_get_size_ext(ctx, seq, LLAMA_STATE_SEQ_FLAGS_NONE));
    if (res.empty() || llama_state_seq_get_data_ext(ctx, res.data(), res.size(), seq, LLAMA_STATE_SEQ_FLAGS_NONE) != res.size()) {
        res.clear();
    }
    return res;
}

static std::vector<std::vector<uint8_t>> states_of(llama_context * ctx) {
    std::vector<std::vector<uint8_t>> res;
    for (int s = 0; s < N_SEQ; ++s) {
        res.push_back(state_of(ctx, s));
    }
    return res;
}

static bool rollback(llama_context * ctx, int seq, int n, std::vector<int> & pos) {
    if (!llama_memory_seq_rm(llama_get_memory(ctx), seq, pos[seq] - n, -1)) {
        return false;
    }
    pos[seq] -= n;
    return true;
}

static int decode_quiet(llama_context * ctx, const std::vector<piece> & pieces, std::vector<int> & pos) {
    g_quiet = true;
    const int ret = decode(ctx, pieces, pos);
    g_quiet = false;
    return ret;
}

// a scenario: what runs before the batch that fails, the batch, and what runs after it. the reference runs the same
// without a failure
struct scenario {
    const char *       name;
    int                n_ubatch;
    int                back;       // the sequence that drafts N_DRAFT tokens and takes N_BACK back before the batch: a
                                   // rollback that waits. -1: none
    std::vector<piece> batch;      // the batch that fails
    int                i_fail;     // its ubatch that fails, from 1
    std::vector<piece> rest;       // one decode after the failure: with the ubatches that ran, it is the batch
    std::vector<int>   pos_after;  // where the sequences of the batch are after the failure, relative to before it
    int                skip = -1;  // a sequence without a prompt: the batch brings its first tokens

    int n_before() const {
        return ::n_before(n_ubatch, back >= 0, N_SEQ - (skip >= 0 ? 1 : 0));
    }
};

static bool draft_back(llama_context * ctx, int seq, std::vector<int> & pos) {
    return decode(ctx, {{seq, N_DRAFT}}, pos) == 0 && rollback(ctx, seq, N_BACK, pos);
}

static bool before(llama_context * ctx, const scenario & sc, std::vector<int> & pos) {
    return setup(ctx, pos, sc.skip) && (sc.back < 0 || draft_back(ctx, sc.back, pos));
}

// the sequence with the rollback takes a few tokens more: the compressed row of the block the rollback was in
static bool tail(llama_context * ctx, const scenario & sc, std::vector<int> & pos) {
    return sc.back < 0 || decode(ctx, {{sc.back, N_TAIL}}, pos) == 0;
}

// the graph of the ubatch does not run: it fails before its graph is built, or with its inputs set
static void fail_before_graph(llama_model * model, const scenario & sc, bool inputs) {
    const int n_fail = g_fail;

    std::vector<int> pos;
    llama_context * ref = make_ctx(model, 0, sc.n_ubatch);
    bool ok = ref != nullptr && before(ref, sc, pos);
    const std::vector<std::vector<uint8_t>> ref_before = ok ? states_of(ref) : std::vector<std::vector<uint8_t>>();
    ok = ok && decode(ref, sc.batch, pos) == 0 && tail(ref, sc, pos);
    const std::vector<std::vector<uint8_t>> ref_after = ok ? states_of(ref) : std::vector<std::vector<uint8_t>>();
    CHECK(ok);
    if (ref) {
        llama_free(ref);
    }
    if (!ok) {
        return;
    }

    llama_context * ctx = make_ctx(model, sc.n_before() + sc.i_fail, sc.n_ubatch, inputs);
    ok = ctx != nullptr && before(ctx, sc, pos);
    CHECK(ok);
    if (ok) {
        const std::vector<int> pos0 = pos;
        CHECK(decode_quiet(ctx, sc.batch, pos) == (inputs ? -3 : -2));

        llama_memory_t mem = llama_get_memory(ctx);
        mem->ubatch_failed(false); // reported twice: nothing more to take back
        std::vector<bool> in_batch(N_SEQ, false);
        for (size_t i = 0; i < sc.batch.size(); ++i) {
            const int s = sc.batch[i].seq;
            in_batch[s] = true;
            // the ubatches that ran stay, the one that failed and the ones after it left nothing
            CHECK(llama_memory_seq_pos_max(mem, s) == pos0[s] + sc.pos_after[i] - 1);
            pos[s] = pos0[s] + sc.pos_after[i];
        }
        for (int s = 0; s < N_SEQ; ++s) {
            if (!in_batch[s]) {
                CHECK(state_of(ctx, s) == ref_before[s]);
            }
        }

        ok = decode(ctx, sc.rest, pos) == 0 && tail(ctx, sc, pos);
        CHECK(ok);
        if (ok) {
            CHECK(states_of(ctx) == ref_after);
        }
    }
    if (ctx) {
        llama_free(ctx);
    }

    printf("%s, the graph does not run (%s): %s\n", sc.name, inputs ? "its inputs set" : "not built", g_fail == n_fail ?
            "the tokens decoded after it give the states of a context that never failed" : "WRONG");
}

struct abort_at {
    int n     = 0;
    int limit = INT_MAX;
};

static bool abort_cb(void * data) {
    abort_at * at = (abort_at *) data;
    return ++at->n > at->limit;
}

// the graph of a ubatch of the batch ran: in part (an abort after n_abort + 1 nodes of the batch) or, the first one, in
// full (n_abort < 0)
static void fail_after_graph(llama_model * model, const scenario & sc, const std::vector<std::vector<uint8_t>> & ref_before,
                             const std::vector<std::vector<uint8_t>> & ref_after, size_t size_empty, int n_abort) {
    std::vector<int> pos;
    llama_context * ctx = make_ctx(model, n_abort < 0 ? -(sc.n_before() + 1) : 0, sc.n_ubatch);
    bool ok = ctx != nullptr && before(ctx, sc, pos);
    CHECK(ok);
    if (!ok) {
        if (ctx) {
            llama_free(ctx);
        }
        return;
    }

    abort_at at;
    at.limit = n_abort < 0 ? INT_MAX : n_abort;
    llama_set_abort_callback(ctx, abort_cb, &at);
    const int ret = decode_quiet(ctx, sc.batch, pos);
    llama_set_abort_callback(ctx, nullptr, nullptr);
    CHECK(ret == (n_abort < 0 ? -3 : 2));

    llama_memory_t mem = llama_get_memory(ctx);
    std::vector<bool> in_batch(N_SEQ, false);
    for (const piece & p : sc.batch) {
        in_batch[p.seq] = true;
    }
    for (int s = 0; s < N_SEQ; ++s) {
        if (!in_batch[s]) {
            CHECK(state_of(ctx, s) == ref_before[s]);
        }
    }

    // a sequence of the ubatch that failed is gone whole and is read again; one whose ubatch ran before keeps it; one
    // of a later ubatch is as it was. then the sequences take what is left of the batch, a sequence a call
    for (const piece & p : sc.batch) {
        const int s = p.seq;
        const llama_pos pos_max = llama_memory_seq_pos_max(mem, s);
        if (pos_max == pos[s] + p.n - 1) {
            pos[s] += p.n;
            continue;
        }
        if (pos_max < 0) {
            CHECK(llama_state_seq_get_size_ext(ctx, s, LLAMA_STATE_SEQ_FLAGS_NONE) == size_empty);
            pos[s] = 0;
            ok = ok && (sc.skip == s || (decode(ctx, {{s, N_PROMPT}}, pos) == 0 && decode(ctx, {{s, N_SHORT}}, pos) == 0)) &&
                 (sc.back != s || draft_back(ctx, s, pos));
        } else {
            CHECK(pos_max == pos[s] - 1);
            CHECK(state_of(ctx, s) == ref_before[s]);
        }
        ok = ok && decode(ctx, {p}, pos) == 0;
    }
    ok = ok && tail(ctx, sc, pos);
    CHECK(ok);
    if (ok) {
        CHECK(states_of(ctx) == ref_after);
    }
    llama_free(ctx);
}

static void fail_after_graph_all(llama_model * model, const scenario & sc) {
    const int n_fail = g_fail;

    // the reference decodes the batch a sequence a call, as the test does after the failure
    std::vector<int> pos;
    llama_context * ref = make_ctx(model, 0, sc.n_ubatch);
    bool ok = ref != nullptr;
    const size_t size_empty = ok ? llama_state_seq_get_size_ext(ref, 0, LLAMA_STATE_SEQ_FLAGS_NONE) : 0;
    ok = ok && before(ref, sc, pos);
    const std::vector<std::vector<uint8_t>> ref_before = ok ? states_of(ref) : std::vector<std::vector<uint8_t>>();
    for (const piece & p : sc.batch) {
        ok = ok && decode(ref, {p}, pos) == 0;
    }
    ok = ok && tail(ref, sc, pos);
    const std::vector<std::vector<uint8_t>> ref_after = ok ? states_of(ref) : std::vector<std::vector<uint8_t>>();
    if (ref) {
        llama_free(ref);
    }

    // the nodes of the graphs of the batch, as the abort callback counts them
    int n_nodes = 0;
    if (ok) {
        llama_context * ctx = make_ctx(model, 0, sc.n_ubatch);
        ok = ctx != nullptr && before(ctx, sc, pos);
        abort_at at;
        if (ok) {
            llama_set_abort_callback(ctx, abort_cb, &at);
            ok = decode(ctx, sc.batch, pos) == 0;
            llama_set_abort_callback(ctx, nullptr, nullptr);
        }
        n_nodes = at.n;
        if (ctx) {
            llama_free(ctx);
        }
    }
    CHECK(ok && n_nodes > 16);
    if (!ok || n_nodes <= 16) {
        return;
    }

    // after its first node, in the middle and after its last one
    for (int i = 0; i <= 2; ++i) {
        fail_after_graph(model, sc, ref_before, ref_after, size_empty, (int) ((int64_t) i * (n_nodes - 1) / 2));
    }
    fail_after_graph(model, sc, ref_before, ref_after, size_empty, -1);

    printf("%s, the graph runs in part (3 aborts over %d nodes) and in full: %s\n", sc.name, n_nodes, g_fail == n_fail ?
            "its sequences gone whole, read again they give the states of a context that never failed" : "WRONG");
}

// a rollback asked after a ubatch that failed before its graph: the snapshots are those of the last ubatch that was
// done (a short batch of N_SHORT tokens), the failed one of n_tokens left none. the rollback of n_back tokens is taken
// or refused as in a context that never failed, and the tokens after it give the same states
static void fail_then_rollback(llama_model * model, int n_tokens, int n_back) {
    const int n_fail = g_fail;

    bool                              ref_taken = false;
    std::vector<std::vector<uint8_t>> ref_after;

    for (const bool fails : { false, true }) {
        std::vector<int> pos;
        llama_context * ctx = make_ctx(model, fails ? n_before(256, false, N_SEQ) + 1 : 0);
        bool ok = ctx != nullptr && setup(ctx, pos);
        if (ok && fails) {
            CHECK(decode_quiet(ctx, {{0, n_tokens}}, pos) == -2);
        }
        const bool taken = ok && rollback(ctx, 0, n_back, pos);
        ok = ok && decode(ctx, {{0, N_TAIL}}, pos) == 0;
        CHECK(ok);
        if (ok && !fails) {
            ref_taken = taken;
            ref_after = states_of(ctx);
        } else if (ok) {
            CHECK(taken == ref_taken);
            CHECK(states_of(ctx) == ref_after);
        }
        if (ctx) {
            llama_free(ctx);
        }
    }

    printf("a rollback of %d tokens after a ubatch of %d that did not run: %s, %s\n", n_back, n_tokens, ref_taken ? "taken" : "refused",
            g_fail == n_fail ? "as in a context that never failed" : "WRONG");
}

// a ubatch that fails before the memory applies it reports its failure too: the last ubatch that was done stays
static void fail_before_apply(llama_model * model) {
    const int n_fail = g_fail;

    std::vector<int> pos;
    llama_context * ctx = make_ctx(model, 0);
    bool ok = ctx != nullptr && setup(ctx, pos);
    CHECK(ok);
    if (ok) {
        const std::vector<std::vector<uint8_t>> states = states_of(ctx);
        llama_get_memory(ctx)->ubatch_failed(false);
        for (int s = 0; s < N_SEQ; ++s) {
            CHECK(llama_memory_seq_pos_max(llama_get_memory(ctx), s) == pos[s] - 1);
        }
        CHECK(states_of(ctx) == states);
    }
    if (ctx) {
        llama_free(ctx);
    }

    printf("a failure before a ubatch is applied: %s\n", g_fail == n_fail ? "the sequences as they were" : "WRONG");
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
    }
    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s -m <deepseek4-moe.gguf>\n", argv[0]);
        return 1;
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

    const std::vector<scenario> scenarios = {
        // a ubatch within the snapshots: its positions were taken back as a rollback of that many tokens
        { "3 tokens of one sequence",              256, -1, {{0, 3}},          1, {{0, 3}},          {0} },
        { "4 tokens of one sequence",              256, -1, {{0, 4}},          1, {{0, 4}},          {0} },
        { "5 tokens of one sequence",              256, -1, {{0, 5}},          1, {{0, 5}},          {0} },
        // past the snapshots: the removal was refused
        { "12 tokens of one sequence",             256, -1, {{0, 12}},         1, {{0, 12}},         {0} },
        // the sequences 0 and 1 share a ubatch
        { "two sequences in one ubatch",           256, -1, {{0, 4}, {1, 4}},  1, {{0, 4}, {1, 4}},  {0, 0} },
        // the rollback of the sequence waits for its next ubatch, which is the one that fails
        { "1 token after a rollback",              256,  0, {{0, 1}},          1, {{0, 1}},          {0} },
        // three ubatches of 16 tokens, the second fails: the first stays
        { "the second of three ubatches",          16,  -1, {{0, 48}},         2, {{0, 32}},         {16} },
        // the sequences 0 and 2 never share a ubatch: the one of 0 fails, the one of 2 with its rollback does not run
        { "the ubatch before one with a rollback", 256,  2, {{0, 4}, {2, 1}},  1, {{0, 4}, {2, 1}},  {0, 0} },
        // the first ubatch of a prompt: the memory has not seen the sequence
        { "the first 12 tokens of a sequence",     256, -1, {{2, 12}},         1, {{2, 12}},         {0},    2 },
        // the same next to a sequence that goes on, in one ubatch
        { "a new sequence next to another one",    256, -1, {{0, 4}, {1, 4}},  1, {{0, 4}, {1, 4}},  {0, 0}, 1 },
    };

    for (const scenario & sc : scenarios) {
        fail_before_graph(model, sc, false);
    }
    // with the inputs set: a ubatch within the snapshots, one after a rollback, one with a sequence the memory has not seen
    for (const size_t i : { (size_t) 1, (size_t) 5, (size_t) 9 }) {
        fail_before_graph(model, scenarios[i], true);
    }
    for (const size_t i : { (size_t) 1, (size_t) 4, (size_t) 5, (size_t) 7, (size_t) 8, (size_t) 9 }) {
        fail_after_graph_all(model, scenarios[i]);
    }
    // a short ubatch that failed would leave fewer snapshots than the last one that was done, a long one more
    fail_then_rollback(model, 1,  N_SHORT - 1);
    fail_then_rollback(model, 12, N_SHORT + 2);
    fail_before_apply(model);

    llama_model_free(model);
    llama_backend_free();

    if (g_fail > 0) {
        fprintf(stderr, "%d checks failed\n", g_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
