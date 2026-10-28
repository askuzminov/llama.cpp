// QSA with several sequences in one unified KV cache, as the slots of a server hold them
// (src/fork/llama-memory-hybrid-idx-fork.cpp, src/fork/qwen4exp-fork.cpp). the model is the qwen4exp of
// test-llama-archs with an indexer top-k of 16, so that QSA selects (the 131072 of the file covers every cell):
// - the groupings carried from one ubatch to the next across the tail rollbacks of drafts (qsa_group_update_multi).
//   LLAMA_QSA_CHECK=1, which this test sets, compares each one with a full grouping and aborts on a difference. a
//   carry that finds its state does not add up falls back to a full grouping, right but slow, so every round of
//   several sequences must have carried. the rounds allocate more cells than the cache has, so that the cells freed by
//   rollbacks are taken again, by other sequences too
// - the mask values from positions with a range of its own for each sequence (qsa_pos_multi_ok, set_input_qsa_pos):
//   every round of several sequences must take them (the graph shows |cell - mid|, an ABS of a SUB; the gate of the
//   PLE takes the ABS of a SCALE), and the logits must equal bit for bit those of the same run with the values read
//   out of the dense mask. LLAMA_QSA_MASK_POS is read once per process, so that run is a second one: --dense-out
//   writes its logits, --dense-in compares with them (ctest runs both)
// - the logits of every sequence close to those of the same sequence decoded alone, with the same tokens, positions
//   and rollbacks (its cells lie elsewhere then, so not bit for bit; the attention of the model moves its logits little)
// - a sequence whose positions cross the stride of the ranges: its range would reach into the next rank, so a round
//   with a position at or past it in a cell or a token takes the dense mask, every other one the mask from positions,
//   both with the logits of the dense run. QSA keeps its positions below n_kv, so the 2^18 of the default takes a
//   cache of more than 262144 cells to cross: the test lowers the stride to 2^7 (LLAMA_QSA_POS_STRIDE_LOG2)
// - no repeated cells in a selection: Vulkan FLASH_ATTN_LIST counts every entry, including block padding
// - a prompt copied to other sequences: shared cells, so the dense mask and nothing carried, the logits as alone
//
// usage: test-qsa-multi-fork -m <dir>/qwen4exp-moe.gguf [--dense-out <file> | --dense-in <file>]

#include "llama.h"
#include "llama-ext.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "line %d: %s\n", line, what);
        n_fail++;
    }
}

#define CHECK(x) check((x), #x, __LINE__)

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// the log, for the count of LLAMA_QSA_CHECK; errors go on to stderr
static std::mutex  g_log_mutex;
static std::condition_variable g_log_cv;
static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log += text;
    g_log_cv.notify_all();
    if (level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

static size_t log_size() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    return g_log.size();
}

// the carried groupings LLAMA_QSA_CHECK compared, from the line a cache prints when it goes, in the log from `from` on
static long long qsa_checked_since(size_t from) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    const char * key = "LLAMA_QSA_CHECK: ";
    long long n = 0;
    for (size_t at = g_log.find(key, from); at != std::string::npos; at = g_log.find(key, at + 1)) {
        long long groups = 0;
        long long pools  = 0;
        if (sscanf(g_log.c_str() + at + strlen(key), "%lld carried groupings and %lld kept pool prefixes compared",
                    &groups, &pools) == 2) {
            n = groups;
        }
    }
    return n;
}

// the nodes |cell - mid| of the graphs a context computed, see the top
static int g_n_abs = 0;
static int g_n_selections = 0;
static int g_n_duplicate_rows = 0;

static bool eval_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask && t->op == GGML_OP_UNARY && ggml_get_unary_op(t) == GGML_UNARY_OP_ABS && t->src[0] != nullptr &&
            t->src[0]->op == GGML_OP_SUB) {
        g_n_abs++;
    }
    if (strncmp(t->name, "indexer_top_k-", 14) == 0 && strchr(t->name, ' ') == nullptr) {
        if (ask) {
            return true;
        }
        std::vector<int32_t> ids(t->ne[0]);
        for (int64_t row = 0; row < ggml_nrows(t); ++row) {
            const size_t offset = (row % t->ne[1])*t->nb[1] + (row/t->ne[1] % t->ne[2])*t->nb[2] + (row/(t->ne[1]*t->ne[2]))*t->nb[3];
            ggml_backend_tensor_get(t, ids.data(), offset, ids.size()*sizeof(int32_t));
            std::sort(ids.begin(), ids.end());
            g_n_duplicate_rows += std::adjacent_find(ids.begin(), ids.end()) != ids.end();
            g_n_selections++;
        }
        return true;
    }
    return false;
}

static const int      N_SEQ    = 4;
static const uint32_t N_DRAFT  = 4;   // the longest draft, the recurrent states kept for its rollback
static const uint32_t N_CELLS  = 256; // the cache of a run of several sequences
static const int      N_ROUNDS = 40;
static const int      STRIDE_LOG2 = 7; // LLAMA_QSA_POS_STRIDE_LOG2, above every position of the other scenarios
static const llama_pos STRIDE     = llama_pos(1) << STRIDE_LOG2;

// one step of a run: the tokens of every active sequence (the same count for each: an uneven batch splits into several
// ubatches, whose rollbacks llama_memory_recurrent does not support yet), then how many of them each one keeps
struct step {
    std::vector<int>                      seqs;
    std::vector<std::vector<llama_token>> tokens; // per entry of seqs
    std::vector<int>                      keep;   // per entry of seqs, 0 = all (a prompt)
};

enum scenario_kind {
    SC_INDEPENDENT, // every sequence from position 0
    SC_STRIDE,      // the last sequence from just below the stride, past it after some rounds
    SC_COPIES,      // sequence 0 decodes the prompt, sequences 1 and 2 copy its cells
};

struct scenario {
    const char *           name;
    scenario_kind          kind   = SC_INDEPENDENT;
    bool                   copies = false;
    int                    n_seq  = 0;
    std::vector<llama_pos> pos0; // the first position of each sequence
    std::vector<step>      steps;
};

static scenario make_scenario(const char * name, scenario_kind kind, int n_vocab, uint32_t seed) {
    std::mt19937 rng(seed);
    auto tok = [&]() { return (llama_token) (rng() % (uint32_t) n_vocab); };

    const bool copies = kind == SC_COPIES;

    scenario sc;
    sc.name   = name;
    sc.kind   = kind;
    sc.copies = copies;
    sc.n_seq  = copies ? 3 : N_SEQ;
    sc.pos0.assign(N_SEQ, 0);
    if (kind == SC_STRIDE) {
        // 20 below the stride after the prompt: the rounds cross it about halfway
        sc.pos0[N_SEQ - 1] = STRIDE - 36;
    }

    // the prompt: one batch of all the sequences, or of sequence 0 alone when the others copy it
    {
        step st;
        for (int s = 0; s < (copies ? 1 : sc.n_seq); ++s) {
            st.seqs.push_back(s);
            st.tokens.emplace_back();
            for (int i = 0; i < 16; ++i) {
                st.tokens.back().push_back(tok());
            }
            st.keep.push_back(0);
        }
        sc.steps.push_back(st);
    }

    // the rounds of drafts, most of them rejected after the first token: some sequences sit a round out, at least two
    // take part
    for (int r = 0; r < N_ROUNDS; ++r) {
        step st;
        const int n_tok = 1 + (int) (rng() % N_DRAFT);
        for (int s = 0; s < sc.n_seq; ++s) {
            if (rng() % 4 == 0 && (int) st.seqs.size() + (sc.n_seq - s) > 2) {
                continue;
            }
            st.seqs.push_back(s);
            st.tokens.emplace_back();
            for (int i = 0; i < n_tok; ++i) {
                st.tokens.back().push_back(tok());
            }
            st.keep.push_back(rng() % 3 == 0 ? 1 + (int) (rng() % n_tok) : 1);
        }
        sc.steps.push_back(st);
    }

    return sc;
}

static llama_context * make_ctx(llama_model * model, uint32_t n_seq_max, uint32_t n_ctx, bool count_abs) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = 256;
    cp.n_ubatch        = 256;
    cp.n_seq_max       = n_seq_max;
    cp.n_rs_seq        = N_DRAFT;
    cp.kv_unified      = true;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.no_perf         = true;
    if (count_abs) {
        cp.cb_eval           = eval_cb;
        cp.cb_eval_user_data = nullptr;
    }
    return llama_init_from_model(model, cp);
}

// the logits of a run: per step, per entry of its seqs, per token
using logits_t = std::vector<std::vector<std::vector<std::vector<float>>>>;

// runs the steps of the sequences in a context of several (only < 0), or of sequence `only` as the sequence 0 of a
// context of one. abs_steps gets the count of |cell - mid| of each step
static bool run(llama_context * ctx, const scenario & sc, int only, logits_t & out, std::vector<int> * abs_steps) {
    const int      n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(llama_get_model(ctx)));
    llama_memory_t mem     = llama_get_memory(ctx);

    auto seq_of = [&](int s) { return only < 0 ? s : 0; };

    std::vector<llama_pos> pos = sc.pos0;

    llama_batch batch = llama_batch_init(256, 0, 1);

    auto add = [&](llama_token t, llama_pos p, llama_seq_id s, bool logits) {
        const int b = batch.n_tokens++;
        batch.token[b]     = t;
        batch.pos[b]       = p;
        batch.n_seq_id[b]  = 1;
        batch.seq_id[b][0] = s;
        batch.logits[b]    = logits;
    };

    out.assign(sc.steps.size(), {});

    bool ok = true;
    for (size_t si = 0; si < sc.steps.size() && ok; ++si) {
        const step & st = sc.steps[si];

        batch.n_tokens = 0;
        std::vector<std::pair<size_t, int>> at; // entry, batch index of its first token
        for (size_t e = 0; e < st.seqs.size(); ++e) {
            const int s = st.seqs[e];
            if (only >= 0 && s != only) {
                continue;
            }
            at.push_back({ e, batch.n_tokens });
            for (size_t i = 0; i < st.tokens[e].size(); ++i) {
                // a prompt needs its last logits only
                add(st.tokens[e][i], pos[s] + (llama_pos) i, seq_of(s), st.keep[e] != 0 || i + 1 == st.tokens[e].size());
            }
        }

        // a sequence that copies the prompt of sequence 0, alone: it decodes the prompt itself
        if (only > 0 && si == 0 && sc.copies) {
            for (size_t i = 0; i < st.tokens[0].size(); ++i) {
                add(st.tokens[0][i], sc.pos0[0] + (llama_pos) i, 0, false);
            }
            pos[only] = sc.pos0[0] + (llama_pos) st.tokens[0].size();
        }

        const int n_abs0 = g_n_abs;
        if (batch.n_tokens > 0 && llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "%s: decode of step %zu failed\n", sc.name, si);
            ok = false;
            break;
        }
        if (abs_steps) {
            abs_steps->push_back(g_n_abs - n_abs0);
        }

        out[si].assign(st.seqs.size(), {});
        for (const auto & [e, b0] : at) {
            const int s = st.seqs[e];
            const int n = (int) st.tokens[e].size();
            for (int i = 0; i < n; ++i) {
                if (batch.logits[b0 + i]) {
                    const float * lg = llama_get_logits_ith(ctx, b0 + i);
                    out[si][e].emplace_back(lg, lg + n_vocab);
                }
            }
            // the rollback of the draft tokens past those kept, a prompt keeps all
            const int keep = st.keep[e] == 0 ? n : st.keep[e];
            if (keep < n && !llama_memory_seq_rm(mem, seq_of(s), pos[s] + keep, -1)) {
                fprintf(stderr, "%s: rollback of sequence %d at step %zu failed\n", sc.name, s, si);
                ok = false;
            }
            pos[s] += keep;
        }

        if (si == 0 && only < 0 && sc.copies) {
            for (int s = 1; s < sc.n_seq; ++s) {
                llama_memory_seq_cp(mem, 0, s, -1, -1);
                pos[s] = pos[0];
            }
        }
    }

    llama_batch_free(batch);
    return ok;
}

// the largest difference of a logit relative to the largest logit of its row, over the rows of sequence s (all for
// s < 0) that both runs have
static double compare(const logits_t & a, const logits_t & b, const scenario & sc, int s, int & n_rows) {
    double worst = 0.0;
    for (size_t si = 0; si < sc.steps.size(); ++si) {
        const step & st = sc.steps[si];
        for (size_t e = 0; e < st.seqs.size(); ++e) {
            if (s >= 0 && st.seqs[e] != s) {
                continue;
            }
            if (si >= a.size() || si >= b.size() || e >= a[si].size() || e >= b[si].size() ||
                    a[si][e].size() != b[si][e].size()) {
                fprintf(stderr, "%s: sequence %d step %zu: the runs have different rows\n", sc.name, st.seqs[e], si);
                return INFINITY;
            }
            const auto & ra = a[si][e];
            const auto & rb = b[si][e];
            for (size_t i = 0; i < ra.size(); ++i) {
                if (ra[i].size() != rb[i].size()) {
                    return INFINITY;
                }
                double scale = 1.0;
                double diff  = 0.0;
                for (size_t j = 0; j < ra[i].size(); ++j) {
                    scale = std::max(scale, (double) std::fabs(rb[i][j]));
                    diff  = std::max(diff, (double) std::fabs(ra[i][j] - rb[i][j]));
                }
                worst = std::max(worst, diff/scale);
                n_rows++;
            }
        }
    }
    return worst;
}

// the logits of the runs with the dense mask in a file: the count of runs, then per run, per step, per entry, the rows
// and their width, then the values
static bool write_runs(const char * path, const std::vector<logits_t> & runs) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    auto put = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, f); };
    put((uint32_t) runs.size());
    for (const auto & lg : runs) {
        put((uint32_t) lg.size());
        for (const auto & st : lg) {
            put((uint32_t) st.size());
            for (const auto & e : st) {
                put((uint32_t) e.size());
                for (const auto & row : e) {
                    put((uint32_t) row.size());
                    fwrite(row.data(), sizeof(float), row.size(), f);
                }
            }
        }
    }
    return fclose(f) == 0;
}

static bool read_runs(const char * path, std::vector<logits_t> & runs) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return false;
    }
    bool ok = true;
    auto get = [&]() {
        uint32_t v = 0;
        ok = ok && fread(&v, sizeof(v), 1, f) == 1 && v < (1u << 20);
        return ok ? v : 0u;
    };
    runs.assign(get(), {});
    for (auto & lg : runs) {
        lg.assign(get(), {});
        for (auto & st : lg) {
            st.assign(get(), {});
            for (auto & e : st) {
                e.assign(get(), {});
                for (auto & row : e) {
                    row.assign(get(), 0.0f);
                    ok = ok && fread(row.data(), sizeof(float), row.size(), f) == row.size();
                }
            }
        }
    }
    fclose(f);
    return ok;
}

// per step, whether every position the cells hold once its tokens are in, those of the tokens included, is below the
// stride: the mask from positions then, the dense mask otherwise (qsa_pos_multi_ok)
static std::vector<bool> below_stride(const scenario & sc) {
    std::vector<llama_pos> pos = sc.pos0;
    std::vector<bool>      held(N_SEQ, false);
    std::vector<bool>      res;
    for (const step & st : sc.steps) {
        llama_pos top = -1;
        for (int s = 0; s < N_SEQ; ++s) {
            if (held[s]) {
                top = std::max(top, pos[s] - 1);
            }
        }
        for (size_t e = 0; e < st.seqs.size(); ++e) {
            top = std::max(top, pos[st.seqs[e]] + (llama_pos) st.tokens[e].size() - 1);
        }
        res.push_back(top < STRIDE);
        for (size_t e = 0; e < st.seqs.size(); ++e) {
            const int s = st.seqs[e];
            pos[s] += st.keep[e] == 0 ? (llama_pos) st.tokens[e].size() : st.keep[e];
            held[s] = true;
        }
    }
    return res;
}

struct multi_run {
    logits_t         logits;
    std::vector<int> abs_steps; // the count of |cell - mid| of each step
    long long        n_checked = 0;
    int              n_multi   = 0; // rounds of several sequences
    int              n_abs     = 0; // of them, those with the mask from positions per sequence
};

static bool run_multi(llama_model * model, const scenario & sc, multi_run & res) {
    llama_context * ctx = make_ctx(model, N_SEQ, N_CELLS, true);
    if (ctx == nullptr) {
        return false;
    }
    const int n_selections = g_n_selections;
    const int n_duplicates = g_n_duplicate_rows;
    const bool ok = run(ctx, sc, -1, res.logits, &res.abs_steps);
    printf("%s: %d QSA selections, %d rows with repeated cells\n", sc.name, g_n_selections - n_selections, g_n_duplicate_rows - n_duplicates);
    CHECK(g_n_selections > n_selections);
    CHECK(g_n_duplicate_rows == n_duplicates);

    // the cache reports what LLAMA_QSA_CHECK compared as it goes
    const size_t from = log_size();
    llama_free(ctx);
    res.n_checked = qsa_checked_since(from);

    for (size_t si = 1; si < res.abs_steps.size(); ++si) {
        res.n_multi += sc.steps[si].seqs.size() > 1;
        res.n_abs   += sc.steps[si].seqs.size() > 1 && res.abs_steps[si] > 0;
    }
    printf("%s: LLAMA_QSA_CHECK compared %lld carried groupings, %d of %d rounds of several sequences took the mask "
           "from positions per sequence\n", sc.name, res.n_checked, res.n_abs, res.n_multi);
    return ok;
}

static void test_scenario(llama_model * model, const scenario & sc, const logits_t * dense) {
    multi_run multi;
    const bool ok = run_multi(model, sc, multi);
    CHECK(ok);
    if (!ok) {
        return;
    }
    CHECK(multi.n_multi > 0);

    // the mask of each round of several sequences: from positions below the stride, dense past it or with shared cells
    const std::vector<bool> below = below_stride(sc);
    int n_below = 0;
    int n_wrong = 0;
    for (size_t si = 1; si < sc.steps.size(); ++si) {
        if (sc.steps[si].seqs.size() < 2) {
            continue;
        }
        const bool want = below[si] && !sc.copies;
        n_below += want;
        n_wrong += want != (multi.abs_steps[si] > 0);
    }
    CHECK(n_wrong == 0);

    switch (sc.kind) {
        case SC_INDEPENDENT:
            CHECK(n_below == multi.n_multi);
            CHECK(multi.n_checked >= multi.n_multi);
            break;
        case SC_STRIDE:
            // rounds on both sides of the stride. the positions past the cells group by rank, which carries nothing
            printf("%s: %d of %d rounds below the stride\n", sc.name, n_below, multi.n_multi);
            CHECK(n_below > 0 && n_below < multi.n_multi);
            break;
        case SC_COPIES:
            // shared cells: every pass groups in full and the mask is the dense one
            CHECK(multi.n_checked == 0);
            CHECK(multi.n_abs == 0);
            break;
    }

    // the same run with the values out of the dense mask (--dense-out): the same logits, bit for bit
    if (dense) {
        int n_rows = 0;
        const double worst = compare(multi.logits, *dense, sc, -1, n_rows);
        printf("%s: against the dense mask, %d rows of logits, largest difference %.3g\n", sc.name, n_rows, worst);
        CHECK(n_rows > 0);
        CHECK(worst == 0.0);
    } else if (!sc.copies) {
        printf("%s: no run with the dense mask to compare with (--dense-in)\n", sc.name);
    }

    for (int s = 0; s < sc.n_seq; ++s) {
        llama_context * ctx = make_ctx(model, 1, N_CELLS, false);
        CHECK(ctx != nullptr);
        if (!ctx) {
            return;
        }
        logits_t   ref;
        const bool ok_ref = run(ctx, sc, s, ref, nullptr);
        CHECK(ok_ref);
        llama_free(ctx);
        if (!ok_ref) {
            continue;
        }

        int n_rows = 0;
        const double worst = compare(multi.logits, ref, sc, s, n_rows);
        printf("%s: sequence %d alone, %d rows of logits, largest difference %.3g of the largest logit\n", sc.name, s, n_rows, worst);
        CHECK(n_rows > 0);
        CHECK(worst < 1e-3);
    }
}

static void test_ple_read_error(const char * source) {
    const auto path = std::filesystem::temp_directory_path() / "llama-ple-read-error.gguf";
    std::filesystem::copy_file(std::filesystem::u8path(source), path, std::filesystem::copy_options::overwrite_existing);
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    mp.lazy_mode = LLAMA_LAZY_MODE_DIO;
    llama_model * model = llama_model_load_from_file(path.u8string().c_str(), mp);
    CHECK(model != nullptr);
    if (model) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 128;
        cp.n_batch = cp.n_ubatch = 8;
        cp.n_threads = cp.n_threads_batch = 2;
        llama_context * ctx = llama_init_from_model(model, cp);
        CHECK(ctx != nullptr);
        if (ctx) {
            const size_t from = log_size();
            std::filesystem::resize_file(path, 0);
            llama_token tokens[] = {1, 2, 3};
            llama_prefetch_tokens(ctx, tokens, 3, 3, 0, 0);
            {
                std::unique_lock<std::mutex> lock(g_log_mutex);
                CHECK(g_log_cv.wait_for(lock, std::chrono::seconds(30), [from] {
                    return g_log.find("PLE read-ahead disabled:", from) != std::string::npos;
                }));
            }
            // A disabled warmer must also tolerate another request and a normal shutdown.
            llama_prefetch_tokens(ctx, tokens, 3, 3, 0, 0);
            llama_free(ctx);
        }
        llama_model_free(model);
    }
    std::filesystem::remove(path);
}

int main(int argc, char ** argv) {
    const char * path      = nullptr;
    const char * dense_out = nullptr;
    const char * dense_in  = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[++i];
        } else if (strcmp(argv[i], "--dense-out") == 0) {
            dense_out = argv[++i];
        } else if (strcmp(argv[i], "--dense-in") == 0) {
            dense_in = argv[++i];
        }
    }
    if (path == nullptr) {
        fprintf(stderr, "usage: %s -m <dir>/qwen4exp-moe.gguf [--dense-out <file> | --dense-in <file>]\n", argv[0]);
        return 1;
    }

    // read once, before the first context
    set_env("LLAMA_QSA_CHECK", "1");
    set_env("LLAMA_QSA_MASK_POS", dense_out ? "0" : "1");
    set_env("LLAMA_QSA_POS_STRIDE_LOG2", std::to_string(STRIDE_LOG2).c_str());

    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    if (argc > 1 && strcmp(argv[1], "--ple-read-error") == 0) {
        test_ple_read_error(path);
        llama_backend_free();
        return n_fail == 0 ? 0 : 1;
    }

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    llama_model_kv_override kvo[2];
    memset(kvo, 0, sizeof(kvo));
    kvo[0].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(kvo[0].key, sizeof(kvo[0].key), "%s", "qwen4exp.attention.indexer.top_k");
    kvo[0].val_i64 = 16;
    mp.kv_overrides = kvo;

    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // the scenarios the run with the dense mask covers, in the order of its file
    const std::vector<scenario> compared = {
        make_scenario("independent",     SC_INDEPENDENT, n_vocab, 1234),
        make_scenario("past the stride", SC_STRIDE,      n_vocab, 5678),
    };

    if (dense_out) {
        // the runs of several sequences with the values out of the dense mask, for the other run to compare with: the
        // groupings carried as there, no mask from positions
        std::vector<logits_t> runs;
        for (const scenario & sc : compared) {
            multi_run dense;
            const bool ok = run_multi(model, sc, dense);
            CHECK(ok);
            CHECK(dense.n_abs == 0);
            if (sc.kind == SC_INDEPENDENT) {
                CHECK(dense.n_checked >= dense.n_multi);
            }
            runs.push_back(std::move(dense.logits));
        }
        CHECK(write_runs(dense_out, runs));
    } else {
        std::vector<logits_t> dense;
        if (dense_in) {
            const bool ok_read = read_runs(dense_in, dense);
            CHECK(ok_read && dense.size() == compared.size());
            if (!ok_read || dense.size() != compared.size()) {
                dense.clear();
            }
        }
        for (size_t i = 0; i < compared.size(); ++i) {
            test_scenario(model, compared[i], dense.empty() ? nullptr : &dense[i]);
        }
        test_scenario(model, make_scenario("copied prompt", SC_COPIES, n_vocab, 4321), nullptr);
    }

    llama_model_free(model);
    llama_backend_free();

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }

    printf("OK\n");

    return 0;
}
