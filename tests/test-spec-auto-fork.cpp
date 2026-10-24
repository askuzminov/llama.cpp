// the gate of --spec-auto (common_speculative_auto, common/fork/speculative-fork.cpp):
// - each gated draft type calibrates its own acceptance by draft probability: the p of the MTP head is its confidence,
//   the p of a DFlash2 lattice pick its share of the top_k of the selector row, the same 0.95 means different things
// - a probability that is not a number (a selector row of -inf only: -inf - -inf) has no calibration column: the gate
//   drops it, the table skips it, the mean acceptance still counts it, and the lattice does not sort a NaN
// the test drives the struct by its public state, as the speculative driver does around keep() and on_accept()

#include "fork/speculative-fork.h"

#include <cmath>
#include <cstdio>
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

    if (n_fail > 0) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }

    printf("OK\n");

    return 0;
}
