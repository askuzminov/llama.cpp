// the gate of --spec-auto (common_speculative_auto, common/fork/speculative-fork.cpp):
// - each gated draft type calibrates its own acceptance by draft probability: the p of the MTP head is its confidence,
//   the p of a DFlash2 lattice pick its share of the top_k of the selector row, the same 0.95 means different things
// - a probability that is not a number (a selector row of -inf only: -inf - -inf) has no calibration column: the gate
//   drops it, the table skips it, the mean acceptance still counts it, and the lattice does not sort a NaN
// the test drives the struct by its public state, as the speculative driver does around keep() and on_accept()

#include "fork/speculative-fork.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what, int line) {
    if (!ok) {
        fprintf(stderr, "line %d: %s\n", line, what);
        n_fail++;
    }
}

#define CHECK(x) check((x), #x, __LINE__)

static const common_speculative_type MTP    = COMMON_SPECULATIVE_TYPE_DRAFT_MTP;
static const common_speculative_type DFLASH = COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;

// one verified draft of sequence 0: its type, the draft probability of each token, how many the target accepted
static void verify(common_speculative_auto & ctl, common_speculative_type type, const std::vector<float> & hp, int n_accepted) {
    auto & s = ctl.seqs[0];
    ctl.open = true;
    s.st     = 1;
    s.type   = type;
    s.n_drf  = (int32_t) hp.size();
    s.hp     = hp;
    ctl.on_accept(0, (uint16_t) n_accepted);
}

int main() {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();

    // per type: the same probability, accepted by one type and rejected by the other
    {
        common_speculative_auto ctl;
        ctl.init(true, 1, 3, nullptr);

        for (int i = 0; i < 300; ++i) {
            verify(ctl, MTP,    { 0.95f }, 1);
            verify(ctl, DFLASH, { 0.95f }, 0);
        }
        printf("p 0.95 at position 0: MTP %.3f, DFlash %.3f\n", ctl.acc_head(MTP, 0, 0.95f), ctl.acc_head(DFLASH, 0, 0.95f));
        CHECK(ctl.acc_head(MTP,    0, 0.95f) > 0.9);
        CHECK(ctl.acc_head(DFLASH, 0, 0.95f) < 0.1);
        CHECK(ctl.acc_mean(MTP)    > 0.9);
        CHECK(ctl.acc_mean(DFLASH) < 0.1);
    }

    // a probability that is not a number
    {
        common_speculative_auto ctl;
        ctl.init(true, 1, 3, nullptr);

        CHECK(ctl.acc_head(MTP, 0, nan) == 0.0);
        CHECK(ctl.acc_head(MTP, 0, inf) == 0.0);

        // no column for it: the table stays as it was, the mean acceptance takes the result
        double before[common_speculative_auto::n_bins];
        for (int b = 0; b < common_speculative_auto::n_bins; ++b) {
            before[b] = ctl.cal_n[DFLASH][0][b];
        }
        verify(ctl, DFLASH, { nan }, 1);
        bool same = true;
        for (int b = 0; b < common_speculative_auto::n_bins; ++b) {
            same = same && ctl.cal_n[DFLASH][0][b] == before[b];
        }
        CHECK(same);
        CHECK(ctl.acc_n[DFLASH] > 0.0);
    }

    // the DFlash2 lattice: a selector row of -inf only
    {
        common_speculative_auto ctl;
        ctl.init(true, 1, 3, nullptr);

        const float ids   [4] = { 11, 12, 13, 14 };
        const float scores[4] = { -inf, -inf, -inf, -inf };

        // without the gate the pick stays in the draft, without a probability
        ctl.mode     = common_speculative_auto::MODE_OFF;
        ctl.drafting = DFLASH;
        ctl.seqs[0]  = {};
        CHECK(common_speculative_fork_lattice_keep(&ctl, 0, ids, scores, 4, 0, false));
        CHECK(ctl.seqs[0].hp.size() == 1 && std::isnan(ctl.seqs[0].hp[0]));

        // the gate drops it
        ctl.mode     = common_speculative_auto::MODE_GATE;
        ctl.thr_keep = 0.01;
        ctl.thr_step = 0.01;
        ctl.seqs[0]  = {};
        CHECK(!common_speculative_fork_lattice_keep(&ctl, 0, ids, scores, 4, 0, false));
        CHECK(ctl.seqs[0].hp.empty());
    }

    // the DFlash2 lattice with scores: the share of the pick, a candidate without a number counts for nothing
    {
        common_speculative_auto ctl;
        ctl.init(true, 1, 3, nullptr);
        ctl.mode     = common_speculative_auto::MODE_OFF;
        ctl.drafting = DFLASH;
        ctl.seqs[0]  = {};

        const float ids   [4] = { 11, 12, 13, 14 };
        const float scores[4] = { 0.0f, 2.0f, -inf, nan };
        CHECK(common_speculative_fork_lattice_keep(&ctl, 0, ids, scores, 4, 1, false));
        const float want = 1.0f / (1.0f + std::exp(-2.0f));
        CHECK(ctl.seqs[0].hp.size() == 1 && std::fabs(ctl.seqs[0].hp[0] - want) < 1e-6f);
    }

    // a sampled walk (--spec-draft-sampling probabilistic): the gate judges the argmax of the row, whatever pick was
    // drawn. rejection sampling returns the target distribution only when the draft takes a position independently of
    // the draw; a gate on the drawn pick would keep the draws of the head and drop those of the tail
    {
        common_speculative_auto ctl;
        ctl.init(true, 1, 3, nullptr);

        // the calibration: a draft p of 0.95 accepted, 0.15 rejected
        for (int i = 0; i < 300; ++i) {
            verify(ctl, DFLASH, { 0.95f }, 1);
            verify(ctl, DFLASH, { 0.15f }, 0);
        }
        ctl.mode     = common_speculative_auto::MODE_GATE;
        ctl.drafting = DFLASH;
        ctl.thr_keep = 0.5;
        ctl.thr_step = 0.0;

        // shares 0.94, 0.05, 0.02: the pick of the tail alone would be dropped
        const float ids   [3] = { 11, 12, 13 };
        const float scores[3] = { 4.0f, 1.0f, 0.0f };
        for (int32_t pick = 0; pick < 3; ++pick) {
            ctl.seqs[0] = {};
            CHECK(common_speculative_fork_lattice_keep(&ctl, 0, ids, scores, 3, pick, true));
            CHECK(ctl.seqs[0].hp.size() == 1 && ctl.seqs[0].hp[0] > 0.9f);
        }

        // the row decides the other way too: a flat row is dropped whatever the pick
        const float flat[3] = { 0.0f, 0.0f, 0.0f };
        for (int32_t pick = 0; pick < 3; ++pick) {
            ctl.seqs[0] = {};
            CHECK(!common_speculative_fork_lattice_keep(&ctl, 0, ids, flat, 3, pick, true));
        }
    }

    // a sampled MTP token: keep() reads the distribution (its top, data[0] of the sorted candidates), never the token the
    // head drew (selected), or a kept draft would lean towards the draws the gate likes and rejection sampling would no
    // longer return the target distribution (04.11)
    {
        common_speculative_auto ctl;
        ctl.init(true, 1, 3, nullptr);
        for (int i = 0; i < 300; ++i) {
            verify(ctl, MTP, { 0.95f }, 1);
            verify(ctl, MTP, { 0.15f }, 0);
        }
        ctl.mode     = common_speculative_auto::MODE_GATE;
        ctl.drafting = MTP;
        ctl.thr_keep = 0.5;
        ctl.thr_step = 0.0;

        for (const float top : { 0.95f, 0.15f }) {
            int decisions = 0;
            for (int32_t drawn = 0; drawn < 3; ++drawn) {
                llama_token_data cand[3] = { { 7, 0.0f, top }, { 8, 0.0f, (1.0f - top)*0.7f }, { 9, 0.0f, (1.0f - top)*0.3f } };
                llama_token_data_array cur_p = { cand, 3, drawn, true };
                ctl.seqs[0] = {};
                bool more = true;
                decisions += ctl.keep(0, &cur_p, more) ? 1 : 0;
                CHECK(ctl.seqs[0].hp.empty() || ctl.seqs[0].hp[0] == top);
            }
            // the same answer for every drawn token: kept for a confident head, dropped for an unsure one
            CHECK(decisions == (top > 0.5f ? 3 : 0));
        }
    }

    // the trace: the overlap of the draft with the target (ov) needs a token distribution of the drafter. the candidates
    // of a DFlash2 lattice carry selector scores, which taken as logits at the target temperature mean nothing: -1
    {
        const char * path = "test-spec-auto-fork.trace.csv";
        {
            common_speculative_auto ctl;
            ctl.init(false, 1, 3, path);
            CHECK(ctl.trace != nullptr);
            ctl.open = true;
            ctl.mode = common_speculative_auto::MODE_OFF;

            std::vector<llama_token_data> tgt = { { 11, 0.0f, 0.5f }, { 12, 0.0f, 0.3f }, { 13, 0.0f, 0.2f } };
            llama_token_data_array tgt_p = { tgt.data(), tgt.size(), 0, true };

            const float ids   [3] = { 11, 12, 13 };
            const float scores[3] = { 2.0f, 1.0f, 0.0f };
            ctl.drafting = DFLASH;
            ctl.seqs[0]  = {};
            ctl.seqs[0].st = 1;
            CHECK(common_speculative_fork_lattice_keep(&ctl, 0, ids, scores, 3, 0, false));
            ctl.on_verify(0, 0, &tgt_p, 11, 1.0f);
            CHECK(ctl.seqs[0].pt.size() == 1 && ctl.seqs[0].pt[0] == 0.5f);
            CHECK(ctl.seqs[0].ov.size() == 1 && ctl.seqs[0].ov[0] == -1.0f);

            std::vector<llama_token_data> head = { { 11, 3.0f, 0.7f }, { 12, 1.0f, 0.2f }, { 13, 0.0f, 0.1f } };
            llama_token_data_array head_p = { head.data(), head.size(), 0, true };
            bool more = true;
            ctl.drafting = MTP;
            ctl.seqs[0]  = {};
            ctl.seqs[0].st = 1;
            CHECK(ctl.keep(0, &head_p, more));
            ctl.on_verify(0, 0, &tgt_p, 11, 1.0f);
            CHECK(ctl.seqs[0].ov.size() == 1 && ctl.seqs[0].ov[0] > 0.0f && ctl.seqs[0].ov[0] <= 1.0f);

            // a sampled walk (--spec-draft-sampling probabilistic) draws from the softmax of the scores at the target
            // temperature: that is the token distribution of the drafter, and the overlap is computed
            ctl.drafting = DFLASH;
            ctl.seqs[0]  = {};
            ctl.seqs[0].st = 1;
            CHECK(common_speculative_fork_lattice_keep(&ctl, 0, ids, scores, 3, 1, true));
            ctl.on_verify(0, 0, &tgt_p, 12, 1.0f);
            CHECK(ctl.seqs[0].ov.size() == 1 && ctl.seqs[0].ov[0] > 0.0f && ctl.seqs[0].ov[0] <= 1.0f);
        }
        std::remove(path);
    }

    // positions 3+ (09.11): each its own calibration row, empty at the start and leaning on the row before it, so a
    // deep position starts at the estimate of the position before it and not at the draft probability; the next
    // position's predicted acceptance (the step decision) leans the same way. LLAMA_SPEC_AUTO_POOL3=1: one row for 3+
    // with the draft probability as its prior and the mean of all positions for the step, as before
    {
        common_speculative_auto ctl;
        ctl.init(true, 1, 7, nullptr);

        // drafts of 7 at p 0.95: in four cycles of five positions 0-2 are accepted and position 3 is not, in the fifth
        // position 0 is not
        for (int i = 0; i < 400; ++i) {
            const int n_acc = i % 5 == 4 ? 0 : 3;
            verify(ctl, MTP, {0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f, 0.95f}, n_acc);
        }
        const double a2 = ctl.acc_head(MTP, 2, 0.95f);
        const double a3 = ctl.acc_head(MTP, 3, 0.95f);
        const double a6 = ctl.acc_head(MTP, 6, 0.95f);
        printf("p 0.95, positions 2 / 3 / 6: %.3f %.3f %.3f, next position 2 / 3 / 6: %.3f %.3f %.3f\n", a2, a3, a6,
               ctl.acc_next(MTP, 2), ctl.acc_next(MTP, 3), ctl.acc_next(MTP, 6));
        CHECK(a2 > 0.6);
        CHECK(a3 < 0.1);          // own observations: never accepted at position 3
        CHECK(a6 <= a3 + 1e-9);   // never drafted past position 3 here: leans on position 3
        CHECK(ctl.acc_next(MTP, 2) == ctl.acc_mean(MTP));
        CHECK(ctl.acc_next(MTP, 3) < 0.1);

        // a fresh table: a deep row starts at the estimate of row 2, not at the draft probability (0.95)
        common_speculative_auto fresh;
        fresh.init(true, 1, 7, nullptr);
        for (int i = 0; i < 400; ++i) {
            verify(fresh, MTP, {0.95f, 0.95f, 0.95f}, i % 2 == 0 ? 3 : 2);   // position 2 accepted half the time
        }
        printf("fresh: p 0.95 at position 2 %.3f, at position 5 %.3f\n", fresh.acc_head(MTP, 2, 0.95f), fresh.acc_head(MTP, 5, 0.95f));
        CHECK(std::fabs(fresh.acc_head(MTP, 5, 0.95f) - fresh.acc_head(MTP, 2, 0.95f)) < 1e-9);

#ifdef _WIN32
        _putenv_s("LLAMA_SPEC_AUTO_POOL3", "1");
#else
        setenv("LLAMA_SPEC_AUTO_POOL3", "1", 1);
#endif
        common_speculative_auto pooled;
        pooled.init(true, 1, 7, nullptr);
        CHECK(pooled.pool3);
        // the shared row 3+ starts at the draft probability, positions 3 and 6 share it
        CHECK(std::fabs(pooled.acc_head(MTP, 3, 0.95f) - 0.95) < 1e-9);
        for (int i = 0; i < 50; ++i) {
            verify(pooled, MTP, {0.95f, 0.95f, 0.95f, 0.95f, 0.95f}, 4);
        }
        CHECK(pooled.acc_head(MTP, 3, 0.95f) == pooled.acc_head(MTP, 6, 0.95f));
        CHECK(pooled.acc_next(MTP, 5) == pooled.acc_mean(MTP));
#ifdef _WIN32
        _putenv_s("LLAMA_SPEC_AUTO_POOL3", "");
#else
        unsetenv("LLAMA_SPEC_AUTO_POOL3");
#endif
    }

    // the draft cap by generating slots (09.11): a verify step within 8 tokens by default - 2 slots 3 drafts, 3 and 4
    // slots 1, 5 and more none; one slot keeps --spec-auto as it is
    if (getenv("LLAMA_SPEC_DRAFT_BY_SLOTS") == nullptr && getenv("LLAMA_SPEC_SLOT_BATCH") == nullptr) {
        CHECK(common_speculative_fork_draft_by_slots(1) == -1);
        CHECK(common_speculative_fork_draft_by_slots(2) == 3);
        CHECK(common_speculative_fork_draft_by_slots(3) == 1);
        CHECK(common_speculative_fork_draft_by_slots(4) == 1);
        CHECK(common_speculative_fork_draft_by_slots(5) == 0);
        CHECK(common_speculative_fork_draft_by_slots(40) == 0);
    }

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }

    printf("OK\n");

    return 0;
}
