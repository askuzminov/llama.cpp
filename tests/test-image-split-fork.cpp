// fork: an image cut by the ubatch size (llama_kv_cache::get_prev_tokens). The patches of an image share one position,
// so the n-gram history of an entry (the PLE rows of qwen4exp) is read by the order of the ubatch, and before its first
// entry by position. A ubatch that went on inside an image took the tokens before the image there, not the patches an
// earlier ubatch had stored: until 16.11 the logits depended on where n_ubatch cut a batch, by up to 1.3 on a generated
// model with a sensitive trunk. The server puts the images of a prompt into its batches next to the text
// (LLAMA_MTMD_MIXED), and several slots share a batch, so the cut falls anywhere.
// - the history of every entry, on the cells alone (no graph): the same tokens for every ubatch size
// - the logits after the prompt and after a few more tokens: one sequence cut at several places and piece by piece, two
//   sequences in one batch, against the prompt in one ubatch
//
// usage: test-image-split-fork -m <dir>/qwen4exp-moe.gguf

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "../src/llama-vocab.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static constexpr uint32_t N_POS     = 4; // M-RoPE: temporal, y, x, other
static constexpr int      N_PATCH_X = 8;
static constexpr int      N_STEPS   = 3; // single tokens after the prompt

struct item {
    bool img;
    int  n;
};

// text, image, text, image, text: 116 entries. the first image takes the entries 24..63, the second 72..111
static const std::vector<item> PROMPT = { {false, 24}, {true, 40}, {false, 8}, {true, 40}, {false, 4} };

// 25: one patch of the first image before the cut, 32 and 48: inside it, 63: one patch after the cut, 64: the cut at
// its end, 70: inside the text between the images, 80: inside the second image, 16: both images cut twice
static const std::vector<int> CUTS = { 25, 32, 48, 63, 64, 70, 80, 16 };

//
// the history on the cells
//

struct entry {
    llama_token id; // LLAMA_TOKEN_NULL: a patch
    llama_pos   t, y, x;
};

// no text token is the mark the cells keep for a patch
static std::vector<entry> entries_of(const item & it, llama_pos & pos, std::mt19937 & rng, llama_token patch_mark) {
    std::vector<entry> res;
    if (!it.img) {
        for (int k = 0; k < it.n; ++k, ++pos) {
            const llama_token id = (llama_token) (3 + rng() % 100);
            res.push_back({ id == patch_mark ? 103 : id, pos, pos, pos });
        }
        return res;
    }
    for (int k = 0; k < it.n; ++k) {
        res.push_back({ LLAMA_TOKEN_NULL, pos, pos + k / N_PATCH_X, pos + k % N_PATCH_X });
    }
    pos += std::max(N_PATCH_X, (it.n + N_PATCH_X - 1) / N_PATCH_X);
    return res;
}

// one batch into the cache, every ubatch placed and applied as a decode does it; the history of every entry is added
// to the list of its sequence
static bool place(llama_kv_cache & kv, const llama_model & model, const std::vector<std::pair<llama_seq_id, entry>> & batch,
                  uint32_t n_ubatch, uint32_t n_prev, std::vector<std::vector<llama_token>> & hist) {
    const uint32_t n_embd  = model.hparams.n_embd;
    const uint32_t n_vocab = model.vocab.n_tokens();

    llama_batch_ext b(batch.size(), n_embd, n_embd, (llama_seq_id) hist.size(), &kv, (llama_token) n_vocab, N_POS);
    std::vector<float> row(n_embd, 0.0f);
    for (const auto & [seq, e] : batch) {
        const int32_t idx = b.add_token(seq);
        if (idx < 0) {
            return false;
        }
        const llama_pos pos[N_POS] = { e.t, e.y, e.x, 0 };
        const bool ok = e.id == LLAMA_TOKEN_NULL ? b.set_token_embd(idx, { row.data(), 1, n_embd }) : b.set_token_id(idx, e.id);
        if (!ok || !b.set_token_pos(idx, pos) || !b.set_output(idx, false)) {
            return false;
        }
    }

    llama_batch_allocr balloc(N_POS, true);
    if (!balloc.init(b, model.vocab, false)) {
        fprintf(stderr, "the batch was rejected\n");
        return false;
    }

    auto mctx = kv.init_batch(balloc, n_ubatch, false);
    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        return false;
    }
    std::vector<llama_token> prev;
    do {
        if (!mctx->apply()) {
            return false;
        }
        const llama_ubatch & ub = mctx->get_ubatch();
        kv.get_prev_tokens(ub, n_prev, prev);
        for (uint32_t i = 0; i < ub.n_tokens; ++i) {
            auto & h = hist[ub.seq_id[i][0]];
            h.insert(h.end(), prev.begin() + i*n_prev, prev.begin() + (i + 1)*n_prev);
        }
    } while (mctx->next());
    return true;
}

// the history of the entries of n_seq sequences that hold the same prompt. pieces: every text and every image in a
// batch of its own, a sequence after the other
static bool history(const llama_model & model, int n_seq, uint32_t n_ubatch, bool pieces, uint32_t n_prev,
                    std::vector<std::vector<llama_token>> & hist) {
    const llama_hparams & hparams = model.hparams;

    int32_t il_attn = -1;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.is_recr(il)) {
            il_attn = (int32_t) il;
            break;
        }
    }
    if (il_attn < 0) {
        return false;
    }

    llama_kv_cache kv(model, false, hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true, 1024, n_seq, 0,
            LLAMA_SWA_TYPE_NONE, nullptr, [&](int32_t il) { return il == il_attn; }, nullptr, nullptr, "test");

    hist.assign(n_seq, {});

    std::vector<std::pair<llama_seq_id, entry>> batch;
    for (int s = 0; s < n_seq; ++s) {
        std::mt19937 rng(100);
        llama_pos    pos = 0;
        for (const item & it : PROMPT) {
            for (const entry & e : entries_of(it, pos, rng, hparams.ple_embd_token)) {
                batch.emplace_back(s, e);
            }
            if (pieces) {
                if (!place(kv, model, batch, n_ubatch, n_prev, hist)) {
                    return false;
                }
                batch.clear();
            }
        }
    }
    return pieces || place(kv, model, batch, n_ubatch, n_prev, hist);
}

//
// the logits
//

struct seq_state {
    llama_pos    pos = 0;
    std::mt19937 rng_tok;
    std::mt19937 rng_embd;
};

struct runner {
    llama_context *   ctx     = nullptr;
    llama_batch_ext * batch   = nullptr;
    int               n_embd  = 0;
    int               n_vocab = 0;

    std::vector<seq_state>          seqs;
    std::vector<std::vector<float>> embds; // alive until the decode

    runner(llama_model * model, int n_ubatch, int n_seq) {
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx      = 1024;
        cp.n_batch    = 512;
        cp.n_ubatch   = n_ubatch;
        cp.n_seq_max  = n_seq;
        cp.kv_unified = true;
        cp.no_perf    = true;
        ctx = llama_init_from_model(model, cp);
        if (ctx == nullptr) {
            return;
        }
        batch   = llama_batch_ext_init(ctx);
        n_embd  = llama_model_n_embd_inp(model);
        n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        seqs.resize(n_seq);
        for (int s = 0; s < n_seq; ++s) {
            seqs[s].rng_tok .seed(100 + s);
            seqs[s].rng_embd.seed(200 + s);
        }
    }

    ~runner() {
        if (batch) {
            llama_batch_ext_free(batch);
        }
        if (ctx) {
            llama_free(ctx);
        }
    }

    // the index of the last entry added
    int add(int s, const item & it, bool output) {
        seq_state & st = seqs[s];
        int idx = -1;
        if (!it.img) {
            for (int k = 0; k < it.n; ++k) {
                idx = llama_batch_ext_add_token(batch, s, (llama_token) (3 + st.rng_tok() % 100));
                const llama_pos p = st.pos++;
                llama_batch_ext_set_pos(batch, idx, &p);
                llama_batch_ext_set_output_logits(batch, idx, output && k + 1 == it.n);
            }
            return idx;
        }
        std::normal_distribution<float> nd(0.0f, 0.05f);
        embds.emplace_back((size_t) it.n * n_embd);
        for (float & v : embds.back()) {
            v = nd(st.rng_embd);
        }
        for (int k = 0; k < it.n; ++k) {
            idx = llama_batch_ext_add_embd(batch, s, { embds.back().data() + (size_t) k * n_embd, 1, (size_t) n_embd });
            const llama_pos p[N_POS] = { st.pos, st.pos + k / N_PATCH_X, st.pos + k % N_PATCH_X, 0 };
            llama_batch_ext_set_pos(batch, idx, p);
            llama_batch_ext_set_output_logits(batch, idx, false);
        }
        st.pos += std::max(N_PATCH_X, (it.n + N_PATCH_X - 1) / N_PATCH_X);
        return idx;
    }

    bool decode() {
        return llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) == 0;
    }

    void take(int idx, std::vector<float> & out) {
        const float * l = llama_get_logits_ith(ctx, idx);
        out.insert(out.end(), l, l + n_vocab);
    }

    void clear() {
        llama_batch_ext_clear(batch);
        embds.clear();
    }

    // a few single tokens of every sequence, one sequence per decode
    bool steps(std::vector<std::vector<float>> & out) {
        for (int k = 0; k < N_STEPS; ++k) {
            for (size_t s = 0; s < seqs.size(); ++s) {
                const int idx = add((int) s, {false, 1}, true);
                if (!decode()) {
                    return false;
                }
                take(idx, out[s]);
                clear();
            }
        }
        return true;
    }
};

// the prompts of n_seq sequences as one batch, cut by n_ubatch; pieces: every text and every image in a decode of its own
static bool run(llama_model * model, int n_ubatch, int n_seq, bool pieces, std::vector<std::vector<float>> & out) {
    runner r(model, n_ubatch, n_seq);
    if (r.ctx == nullptr) {
        return false;
    }
    out.assign(n_seq, {});
    std::vector<int> last(n_seq, -1);

    if (pieces) {
        for (int s = 0; s < n_seq; ++s) {
            for (size_t i = 0; i < PROMPT.size(); ++i) {
                last[s] = r.add(s, PROMPT[i], i + 1 == PROMPT.size());
                if (!r.decode()) {
                    return false;
                }
                if (i + 1 == PROMPT.size()) {
                    r.take(last[s], out[s]);
                }
                r.clear();
            }
        }
    } else {
        for (int s = 0; s < n_seq; ++s) {
            for (size_t i = 0; i < PROMPT.size(); ++i) {
                last[s] = r.add(s, PROMPT[i], i + 1 == PROMPT.size());
            }
        }
        if (!r.decode()) {
            return false;
        }
        for (int s = 0; s < n_seq; ++s) {
            r.take(last[s], out[s]);
        }
        r.clear();
    }

    return r.steps(out);
}

// the largest difference relative to the largest logit
static float diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return INFINITY;
    }
    float d   = 0.0f;
    float top = 1e-6f;
    for (size_t i = 0; i < a.size(); ++i) {
        d   = std::max(d, std::fabs(a[i] - b[i]));
        top = std::max(top, std::fabs(a[i]));
    }
    return d / top;
}

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
    }
    if (model_path == nullptr) {
        fprintf(stderr, "usage: %s -m <dir>/qwen4exp-moe.gguf\n", argv[0]);
        return 1;
    }

    llama_log_set([](ggml_log_level level, const char * text, void *) {
        if (level == GGML_LOG_LEVEL_ERROR) {
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

    // the history: at least 3 tokens back, the n-gram of the model if it is longer
    {
        const uint32_t n_prev = std::max<uint32_t>(3, model->hparams.ple_ngram_size > 0 ? model->hparams.ple_ngram_size - 1 : 0);

        std::vector<std::vector<llama_token>> ref;
        CHECK(history(*model, 1, 512, false, n_prev, ref));
        CHECK(ref.size() == 1 && ref[0].size() == 116 * n_prev);

        // the patch after a cut inside an image has a patch before it, as every patch but the first: with the mark the
        // cells keep for a patch, or with no token
        if (ref.size() == 1 && ref[0].size() == 116 * n_prev) {
            CHECK(ref[0][32*n_prev + n_prev - 1] == ref[0][33*n_prev + n_prev - 1]);
            CHECK(ref[0][24*n_prev + n_prev - 1] != ref[0][32*n_prev + n_prev - 1]);
        }

        int n_bad = 0;
        for (const int n_ubatch : CUTS) {
            std::vector<std::vector<llama_token>> hist;
            CHECK(history(*model, 1, (uint32_t) n_ubatch, false, n_prev, hist));
            n_bad += hist != ref;
            CHECK(hist == ref);
        }
        {
            std::vector<std::vector<llama_token>> hist;
            CHECK(history(*model, 1, 512, true, n_prev, hist));
            n_bad += hist != ref;
            CHECK(hist == ref);
        }
        // two sequences with the same prompt in one batch: each as the one alone
        for (const int n_ubatch : { 512, 100, 64, 50 }) {
            std::vector<std::vector<llama_token>> hist;
            CHECK(history(*model, 2, (uint32_t) n_ubatch, false, n_prev, hist));
            const bool same = hist.size() == 2 && ref.size() == 1 && hist[0] == ref[0] && hist[1] == ref[0];
            n_bad += !same;
            CHECK(same);
        }
        printf("the history of the entries (%u tokens back): %d of %d layouts differ from the prompt in one ubatch\n",
                n_prev, n_bad, (int) CUTS.size() + 5);
    }

    const float tol = 1e-3f;

    // one sequence: the whole prompt in one ubatch is the reference
    std::vector<std::vector<float>> ref;
    CHECK(run(model, 512, 1, false, ref));

    for (const int n_ubatch : CUTS) {
        std::vector<std::vector<float>> out;
        CHECK(run(model, n_ubatch, 1, false, out));
        const float d = out.empty() || ref.empty() ? INFINITY : diff(out[0], ref[0]);
        printf("one sequence, ubatch %3d against one ubatch: largest difference %.3g of the largest logit\n", n_ubatch, d);
        CHECK(d <= tol);
    }

    {
        std::vector<std::vector<float>> out;
        CHECK(run(model, 512, 1, true, out));
        const float d = out.empty() || ref.empty() ? INFINITY : diff(out[0], ref[0]);
        printf("one sequence, texts and images in decodes of their own: largest difference %.3g of the largest logit\n", d);
        CHECK(d <= tol);
    }

    // two sequences in one batch, as two slots of a server: each against itself piece by piece
    std::vector<std::vector<float>> ref2;
    CHECK(run(model, 512, 2, true, ref2));
    for (const int n_ubatch : { 512, 100, 64, 50 }) {
        std::vector<std::vector<float>> out;
        CHECK(run(model, n_ubatch, 2, false, out));
        for (int s = 0; s < 2; ++s) {
            const float d = out.size() < 2 || ref2.size() < 2 ? INFINITY : diff(out[s], ref2[s]);
            printf("two sequences in one batch, ubatch %3d, sequence %d against its pieces: largest difference %.3g of the largest logit\n", n_ubatch, s, d);
            CHECK(d <= tol);
        }
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
