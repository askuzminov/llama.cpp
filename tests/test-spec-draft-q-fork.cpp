// fork: a drafter that samples its drafts (probabilistic drafting: --spec-draft-sampling, the default of --spec-auto)
// hands the verifier q, the distribution each drafted token was drawn from. Rejection sampling returns the target
// distribution only when the token really comes from that q: a drafter that takes its argmax but reports its softmax, or
// that keeps a draft depending on the token it drew, makes the output lean towards its own picks. On draft-simple (the
// llama of the speculative fixture as the target and the draft):
// - the first drafted token over many drafts from one context has the q the drafter reported (chi-square, p < 1e-6)
// - the second, among the drafts whose first token was the most frequent one, has the q of its step
// - every drafted token is in its q with q > 0, and q is the same in every draft from the same context
// - a draft shorter than n_min is dropped together with its q (the rejection check asserts the same length)
// The MTP head and DFlash2 draw the same way (common/speculative.cpp); the gates of --spec-auto that may stop a draft
// look at the distribution and not at the drawn token, which test-spec-auto-fork checks without a model
//
// usage: test-spec-draft-q-fork -m <dir>/llama-dense.gguf

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "speculative.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static constexpr int N_TRIALS = 12000;

static double chi2_crit(int df) {
    const double z = 4.7534; // p = 1e-6
    const double a = 2.0/(9.0*df);
    return df*std::pow(1.0 - a + z*std::sqrt(a), 3.0);
}

// the drafted tokens against q: every token of q a cell (expected counts under 5 merged), a token outside q fails
static bool fits(const char * name, const std::map<llama_token, int> & counts, const std::vector<llama_token_data> & q, int n) {
    double chi2 = 0.0;
    int    df   = -1;
    double e_rest = 0.0;
    double o_rest = 0.0;
    int    seen   = 0;
    for (const auto & c : q) {
        const auto it = counts.find(c.id);
        const int  o  = it == counts.end() ? 0 : it->second;
        seen += o;
        const double e = n*(double) c.p;
        if (e < 5.0) {
            e_rest += e;
            o_rest += o;
            continue;
        }
        chi2 += (o - e)*(o - e)/e;
        df++;
    }
    if (e_rest > 0.0) {
        chi2 += (o_rest - e_rest)*(o_rest - e_rest)/e_rest;
        df++;
    }
    if (seen != n) {
        fprintf(stderr, "%s: %d of %d drafts are tokens outside q\n", name, n - seen, n);
        return false;
    }
    const double crit = chi2_crit(std::max(df, 1));
    fprintf(stderr, "%-7s: %d drafts, chi2 %.1f, df %d, bar %.1f %s\n", name, n, chi2, df, crit, chi2 <= crit ? "ok" : "DIFFERS");
    return chi2 <= crit;
}

static bool same_q(const std::vector<llama_token_data> & a, const std::vector<llama_token_data> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    std::map<llama_token, float> pa;
    for (const auto & e : a) {
        pa[e.id] = e.p;
    }
    for (const auto & e : b) {
        const auto it = pa.find(e.id);
        if (it == pa.end() || std::fabs(it->second - e.p) > 1e-6f) {
            return false;
        }
    }
    return true;
}

static float q_of(const std::vector<llama_token_data> & q, llama_token t) {
    for (const auto & e : q) {
        if (e.id == t) {
            return e.p;
        }
    }
    return 0.0f;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }
    params.n_ctx   = 256;
    params.warmup  = false;
    params.sampling.backend_sampling = false;
    params.speculative.types                 = { COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE };
    params.speculative.draft.mparams.path    = params.model.path;
    params.speculative.draft.probabilistic   = true;
    params.speculative.draft.n_max           = 2;
    params.speculative.draft.n_min           = 0;
    params.speculative.draft.p_min           = 0.0f;

    // the q of every trial has to be the same: Metal on an AMD GPU (the 5500M of a MacBook) computes run-to-run
    // different values with concurrent dispatch, upstream as well, which moved 2000-4000 of 12000 drafts out of q.
    // ordered dispatch there (no effect on other backends)
    if (getenv("GGML_METAL_CONCURRENCY_DISABLE") == nullptr) {
#ifdef _WIN32
        _putenv_s("GGML_METAL_CONCURRENCY_DISABLE", "1");
#else
        setenv("GGML_METAL_CONCURRENCY_DISABLE", "1", 1);
#endif
    }

    common_init();
    llama_backend_init();

    auto init = common_init_from_params(params);
    llama_model *   model_tgt = init->model();
    llama_context * ctx_tgt   = init->context();
    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        fprintf(stderr, "cannot load the target\n");
        return 1;
    }

    common_params params_dft = common_base_params_to_speculative(params);
    common_speculative_init_result_ptr spec_init = common_speculative_init_from_params(params_dft, model_tgt, ctx_tgt);
    llama_context * ctx_dft = spec_init ? spec_init->context() : nullptr;
    if (ctx_dft == nullptr) {
        fprintf(stderr, "cannot create the draft context\n");
        return 1;
    }
    params.speculative.draft.ctx_tgt = ctx_tgt;
    params.speculative.draft.ctx_dft = ctx_dft;

    common_speculative * spec = common_speculative_init(params.speculative, 1);
    if (spec == nullptr) {
        fprintf(stderr, "cannot init the speculative decoding\n");
        return 1;
    }

    const llama_seq_id seq = 0;
    const llama_tokens inp = { 1, 5, 9, 13, 17, 21, 25 };
    llama_tokens prompt(inp.begin(), inp.end() - 1);
    {
        common_batch batch(ctx_tgt);
        for (size_t i = 0; i < prompt.size(); ++i) {
            batch.add(prompt[i], (llama_pos) i, seq, false);
        }
        CHECK(llama_process(ctx_tgt, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0);
        CHECK(common_speculative_process(spec, batch));
    }
    common_speculative_begin(spec, seq, prompt);

    const llama_pos   n_past  = (llama_pos) prompt.size();
    const llama_token id_last = inp.back();

    std::vector<llama_token_data>                         q0;
    std::map<llama_token, int>                            first;
    std::map<llama_token, std::vector<llama_token_data>>  q1_of;    // the q of the second step after a first token
    std::map<llama_token, std::map<llama_token, int>>     second;
    int n_first = 0;
    int n_bad   = 0;

    llama_tokens                               draft;
    std::vector<std::vector<llama_token_data>> q;
    for (int trial = 0; trial < N_TRIALS; ++trial) {
        // the draft context back to the prompt: each draft starts from the same state, its sampler draws on
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
        draft.clear();
        q.clear();
        common_speculative_get_draft_params(spec, seq) = {
            /* .drafting = */ true,
            /* .n_max    = */ 2,
            /* .pos0     = */ n_past,
            /* .id_last  = */ id_last,
            /* .prompt   = */ &prompt,
            /* .result   = */ &draft,
            /* .result_q = */ &q,
            /* .temp     = */ 0.8f,
            /* .seed     = */ 42,
        };
        common_speculative_draft(spec);

        if (draft.empty()) {
            continue;
        }
        if (q.size() != draft.size()) {
            n_bad++;
            continue;
        }
        for (size_t i = 0; i < draft.size(); ++i) {
            n_bad += q_of(q[i], draft[i]) > 0.0f ? 0 : 1;
        }
        if (q0.empty()) {
            q0 = q[0];
        }
        CHECK(same_q(q0, q[0]) || (n_bad++, false));
        first[draft[0]]++;
        n_first++;
        if (draft.size() > 1) {
            auto & q1 = q1_of[draft[0]];
            if (q1.empty()) {
                q1 = q[1];
            }
            if (!same_q(q1, q[1])) {
                n_bad++;
            }
            second[draft[0]][draft[1]]++;
        }
    }

    CHECK(n_bad == 0);
    CHECK(n_first > N_TRIALS/2);
    CHECK(fits("first", first, q0, n_first));

    // the second token after the most frequent first one
    llama_token mode = -1;
    int         n_mode = 0;
    for (const auto & [t, n] : second) {
        int s = 0;
        for (const auto & [u, c] : n) {
            s += c;
        }
        if (s > n_mode) {
            mode   = t;
            n_mode = s;
        }
    }
    CHECK(mode >= 0 && n_mode > 200);
    if (mode >= 0) {
        CHECK(fits("second", second[mode], q1_of[mode], n_mode));
    }

    common_speculative_free(spec);

    // n_min above n_max: every draft is dropped, its q too
    {
        params.speculative.draft.n_min = params.speculative.draft.n_max + 1;
        common_speculative * spec_min = common_speculative_init(params.speculative, 1);
        CHECK(spec_min != nullptr);
        if (spec_min != nullptr) {
            common_speculative_begin(spec_min, seq, prompt);
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq, n_past, -1);
            draft.clear();
            q.clear();
            common_speculative_get_draft_params(spec_min, seq) = {
                /* .drafting = */ true,
                /* .n_max    = */ 2,
                /* .pos0     = */ n_past,
                /* .id_last  = */ id_last,
                /* .prompt   = */ &prompt,
                /* .result   = */ &draft,
                /* .result_q = */ &q,
                /* .temp     = */ 0.8f,
                /* .seed     = */ 42,
            };
            common_speculative_draft(spec_min);
            CHECK(draft.empty());
            CHECK(q.size() == draft.size());
            common_speculative_free(spec_min);
        }
    }

    llama_backend_free();

    if (n_fail == 0) {
        fprintf(stderr, "OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
