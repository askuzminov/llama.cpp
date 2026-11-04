// fork: the rejection sampling of speculative drafts (common_sampler_sample_and_accept_n_rejection, common/sampling.cpp)
// returns the target distribution whatever the draft distribution is. The upstream test-speculative-rejection-sampling
// checks it on the nearly flat distribution of its tiny model (p 0.005 - 0.012 over 128 tokens) with a tolerance of six
// standard errors per token (~0.009): a residual drawn from p instead of norm(p - q)+, or p_x and q_x swapped, pass it
// (04.11). Here the target is sharp (the test sets the logits: a few tokens hold most of it), the draft distributions are
// far from it in several ways, and the counts of all tokens go into one chi-square test against the target:
// - tail: q leans on the tail of p and on tokens p does not have, most drafts are rejected and the residual decides
// - greedy: q one-hot on the second token of p, as a greedy draft (accepted with p(x), else the rest of p)
// - close: q near p, most drafts accepted
// - random: a new random q every trial
// - second: two draft tokens, the second output token whenever the first was accepted
// a token outside the support of p fails at once. The draws of a scenario: 20000, the bar: p < 1e-6
// the verify callback (the trace of --spec-auto) must see each checked position once, in order, with the target p
//
// usage: test-spec-rejection-fork -m <a model> (any vocabulary of 32 tokens or more, the logits are the test's)

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static constexpr int N_SUPPORT = 16;    // the tokens the target has
static constexpr int N_TRIALS  = 20000;

// the chi-square critical value of df degrees of freedom at p = 1e-6 (Wilson-Hilferty)
static double chi2_crit(int df) {
    const double z = 4.7534; // the one-sided normal quantile of 1e-6
    const double a = 2.0/(9.0*df);
    return df*std::pow(1.0 - a + z*std::sqrt(a), 3.0);
}

// counts of the output tokens against the target p: cells of an expected count under 5 merge into one
static bool fits(const char * name, const std::vector<int> & counts, const std::vector<double> & p, int n) {
    double chi2 = 0.0;
    int    df   = -1;
    double e_rest = 0.0;
    double o_rest = 0.0;
    for (size_t t = 0; t < p.size(); ++t) {
        if (p[t] == 0.0) {
            if (counts[t] > 0) {
                fprintf(stderr, "%s: token %zu came out %d times, the target does not have it\n", name, t, counts[t]);
                return false;
            }
            continue;
        }
        const double e = n*p[t];
        if (e < 5.0) {
            e_rest += e;
            o_rest += counts[t];
            continue;
        }
        chi2 += (counts[t] - e)*(counts[t] - e)/e;
        df++;
    }
    if (e_rest > 0.0) {
        chi2 += (o_rest - e_rest)*(o_rest - e_rest)/e_rest;
        df++;
    }
    const double crit = chi2_crit(std::max(df, 1));
    fprintf(stderr, "%-8s: %d draws, chi2 %.1f, df %d, bar %.1f %s\n", name, n, chi2, df, crit, chi2 <= crit ? "ok" : "DIFFERS");
    return chi2 <= crit;
}

int main(int argc, char ** argv) {
    common_params params;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    params.sampling.backend_sampling = false;
    params.n_ctx   = 64;
    params.warmup  = false;

    common_init();
    llama_backend_init();

    auto init = common_init_from_params(params);
    llama_model *   model = init->model();
    llama_context * ctx   = init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "cannot load the model\n");
        return 1;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    if (n_vocab < 2*N_SUPPORT) {
        fprintf(stderr, "the vocabulary has %d tokens, the test needs %d\n", n_vocab, 2*N_SUPPORT);
        return 1;
    }

    // a prompt, so that the context has logits to overwrite
    {
        const llama_tokens prompt = { 1, 2, 3 };
        common_batch batch = common_batch_get_one(ctx, prompt);
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "cannot decode the prompt\n");
            return 1;
        }
    }

    // the target: logits 4, 3.4, 2.8, ... over the first N_SUPPORT tokens skipping the eog ones, -inf elsewhere
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<int> support;
    for (int t = 0; t < n_vocab && (int) support.size() < N_SUPPORT; ++t) {
        if (!llama_vocab_is_eog(vocab, t)) {
            support.push_back(t);
        }
    }
    float * logits = llama_get_logits_ith(ctx, -1);
    std::vector<double> p(n_vocab, 0.0);
    {
        double z = 0.0;
        for (int t = 0; t < n_vocab; ++t) {
            logits[t] = -INFINITY;
        }
        for (int i = 0; i < N_SUPPORT; ++i) {
            logits[support[i]] = 4.0f - 0.6f*i;
            z += std::exp(4.0 - 0.6*i);
        }
        for (int i = 0; i < N_SUPPORT; ++i) {
            p[support[i]] = std::exp(4.0 - 0.6*i)/z;
        }
    }

    // tokens p does not have, for the drafts
    std::vector<int> outside;
    for (int t = 0; t < n_vocab && (int) outside.size() < 4; ++t) {
        if (p[t] == 0.0 && !llama_vocab_is_eog(vocab, t)) {
            outside.push_back(t);
        }
    }

    common_params_sampling sampling;
    sampling.seed     = 1234;
    sampling.samplers = { COMMON_SAMPLER_TYPE_TEMPERATURE };
    sampling.temp     = 1.0f;

    std::mt19937 rng(4321);
    std::uniform_real_distribution<double> uni(0.0, 1.0);

    auto normalized = [](std::vector<llama_token_data> q) {
        double s = 0.0;
        for (const auto & e : q) {
            s += e.p;
        }
        for (auto & e : q) {
            e.p = (float) (e.p/s);
        }
        return q;
    };
    auto draw = [&](const std::vector<llama_token_data> & q) {
        double u = uni(rng);
        for (const auto & e : q) {
            u -= e.p;
            if (u <= 0.0) {
                return (llama_token) e.id;
            }
        }
        return (llama_token) q.back().id;
    };

    // q over the support (and the tokens outside it) with the weights of a scenario
    std::vector<llama_token_data> q_tail;
    std::vector<llama_token_data> q_close;
    for (int i = 0; i < N_SUPPORT; ++i) {
        q_tail.push_back({ support[i], 0.0f, (float) (0.02 + 0.06*i) });
        q_close.push_back({ support[i], 0.0f, (float) (0.8*p[support[i]] + 0.2/N_SUPPORT) });
    }
    for (int t : outside) {
        q_tail.push_back({ t, 0.0f, 0.15f });
    }
    q_tail  = normalized(q_tail);
    q_close = normalized(q_close);
    const std::vector<llama_token_data> q_greedy = { { support[1], 0.0f, 1.0f } };

    // n_draft drafts of one q per trial (fixed, or a new one from make_q), the output token at index `at`
    auto scenario = [&](const char * name, int n_draft, int at, const std::function<std::vector<llama_token_data>()> & make_q) {
        common_sampler_ptr smpl(common_sampler_init(model, sampling));
        std::vector<int> counts(n_vocab, 0);
        int n = 0;
        int n_pos_bad = 0;
        for (int trial = 0; trial < N_TRIALS; ++trial) {
            std::vector<std::vector<llama_token_data>> qs;
            llama_tokens draft;
            for (int i = 0; i < n_draft; ++i) {
                qs.push_back(make_q());
                draft.push_back(draw(qs.back()));
            }
            size_t n_pos  = 0;
            bool   pos_ok = true;
            const common_sampler_verify_cb on_pos = [&](size_t i, const llama_token_data_array * cur_p) {
                if (i != n_pos++ || i >= draft.size()) {
                    pos_ok = false;
                    return;
                }
                float p_x = 0.0f;
                for (size_t k = 0; k < cur_p->size; ++k) {
                    if (cur_p->data[k].id == draft[i]) {
                        p_x = cur_p->data[k].p;
                    }
                }
                pos_ok = pos_ok && std::fabs(p_x - p[draft[i]]) < 1e-4;
            };
            const std::vector<int> idxs(draft.size() + 1, -1);
            const auto out = common_sampler_sample_and_accept_n_rejection(smpl.get(), ctx, idxs, draft, qs, false, on_pos);
            // a rejection ends the check at its position, a full acceptance adds one token past the draft
            if (!pos_ok || n_pos != std::min(out.size(), draft.size())) {
                n_pos_bad++;
            }
            if ((int) out.size() > at) {
                counts[out[at]]++;
                n++;
            }
        }
        CHECK(n > N_TRIALS/10);
        CHECK(fits(name, counts, p, n));
        CHECK(n_pos_bad == 0);
    };

    scenario("tail",   1, 0, [&]() { return q_tail; });
    scenario("greedy", 1, 0, [&]() { return q_greedy; });
    scenario("close",  1, 0, [&]() { return q_close; });
    scenario("random", 1, 0, [&]() {
        std::vector<llama_token_data> q;
        for (int i = 0; i < N_SUPPORT; ++i) {
            q.push_back({ support[i], 0.0f, (float) uni(rng) });
        }
        q.push_back({ outside[0], 0.0f, (float) uni(rng) });
        return normalized(q);
    });
    scenario("second", 2, 1, [&]() { return q_close; });

    llama_backend_free();

    if (n_fail == 0) {
        fprintf(stderr, "OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
