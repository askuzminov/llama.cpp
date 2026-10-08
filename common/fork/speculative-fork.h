#pragma once

// the fork's speculative decoding additions: the cost-aware draft length (--spec-auto) and the per-cycle trace

#include "llama.h"
#include "common.h"
#include "speculative.h"

#include <cstdint>
#include <cstdio>
#include <vector>

// draft length cap of --spec-auto when --spec-draft-n-max is not given
#define COMMON_SPECULATIVE_AUTO_N_MAX 6

struct common_speculative_impl;

// the type of an implementation (defined in speculative.cpp, where the struct is)
common_speculative_type common_speculative_impl_type(const common_speculative_impl * impl);

// cost-aware draft length (--spec-auto) and the per-cycle trace (LLAMA_SPEC_TRACE=<file>)
//
// one more draft position pays off while the chance that it and all positions before it are accepted
// is above the number of tokens that its cost gives at the current rate: P(0..j accepted) > R * c.
// R and the verify cost per token are fit on the running cycles. the chance comes from the head
// probability of the draft, calibrated against the observed acceptance, or for drafts without
// probabilities (ngram) from the acceptance per position
struct common_speculative_auto {
    using draft_params_vec = std::vector<common_speculative_draft_params>;

    enum mode_t {
        MODE_OFF,   // no cap and no gate
        MODE_PROBE, // rotating cap on model drafts and no gate, keeps the fit and the calibration alive
        MODE_GATE,
    };

    static constexpr int    n_depth = 4;     // calibration rows: draft position 0, 1, 2, 3+
    static constexpr int    n_bins  = 10;    // calibration columns: head probability in steps of 0.1
    static constexpr double f_cal   = 0.99;  // forgetting factor of the acceptance estimates, per update
    static constexpr double f_cycle = 0.995; // forgetting factor of the cost and rate fits, per cycle
    static constexpr int    n_warm  = 64;    // valid probe cycles before the first gated one
    static constexpr int    n_probe = 32;    // after the warm up, one cycle in n_probe is a probe
    static constexpr int    n_hist  = 17;

    struct seq_state {
        int8_t  st    = 0; // 0 idle, 1 waits for accept, 2 done
        int32_t n_drf = 0;
        int32_t n_acc = 0;
        double  pcum  = 1.0;

        common_speculative_type type = COMMON_SPECULATIVE_TYPE_NONE;

        std::vector<float> hp; // head probability of each drafted token, MTP only

        // trace only
        std::vector<std::vector<llama_token_data>> q; // draft candidates per position
        std::vector<float> pt; // target probability of the drafted token
        std::vector<float> pm; // highest target probability
        std::vector<float> ov; // sum of min(target, draft), the acceptance of exact speculative sampling
    };

    bool   enabled = false;
    FILE * trace   = nullptr;

    double cal_acc[n_depth][n_bins] = {};
    double cal_n  [n_depth][n_bins] = {};

    std::vector<double> pos_acc[COMMON_SPECULATIVE_TYPE_COUNT];
    std::vector<double> pos_n  [COMMON_SPECULATIVE_TYPE_COUNT];

    // mean conditional acceptance of the MTP positions, predicts the next position before it is drafted
    double acc_sum = 0.0;
    double acc_n   = 0.0;

    // verify time [us] = t0 + t1 * n_tokens, least squares with forgetting
    double s0 = 0.0, sx = 0.0, sxx = 0.0, sy = 0.0, sxy = 0.0;
    double t0 = 0.0, t1 = 0.0;
    bool   fit_ok = false;

    double t_step = 0.0; // one draft decode step [us]
    double r_tok  = 0.0; // decayed totals of the tokens and the time, for the rate
    double r_us   = 0.0;

    double thr_keep = 0.0; // keep a drafted token while P > thr_keep
    double thr_step = 0.0; // run one more draft step while P * acc_mean > thr_step

    int32_t n_cap_probe = 1;
    int64_t n_cycle  = 0;
    int64_t n_valid  = 0;
    int64_t n_probes = 0;
    mode_t  mode     = MODE_OFF;
    int32_t cap      = -1; // draft cap of a probe cycle

    int64_t hist[n_hist] = {}; // draft lengths of the gated cycles

    // the cycle in progress
    bool    open    = false;
    bool    valid   = false;
    int64_t t_start = 0;
    int64_t t_draft = 0;

    std::vector<seq_state> seqs;

    // per implementation in common_speculative_draft: the n_max to restore after a cap, the sequences skipped by a cap of 0
    std::vector<int32_t> n_max_save;
    std::vector<uint8_t> skipped;

    ~common_speculative_auto();

    void init(bool enabled, uint32_t n_seq, int32_t n_cap, const char * path_trace);

    bool active() const {
        return enabled || trace;
    }

    // tokens per us
    double rate() const {
        return r_us > 0.0 ? r_tok / r_us : 0.0;
    }

    double acc_mean() const {
        return acc_n > 0.0 ? acc_sum / acc_n : 0.5;
    }

    double acc_head(int j, float p) const;
    double acc_pos(common_speculative_type type, int j) const;

    // draft cap for an implementation in this cycle: -1 keeps the caller's cap, 0 skips the implementation
    int32_t impl_cap(common_speculative_type type) const;

    // common_speculative_draft: begin the cycle and return the start time, end it after the drafts
    int64_t draft_begin(const draft_params_vec & dparams);
    void    draft_done(const draft_params_vec & dparams, const std::vector<common_speculative_impl *> & impl_last, int64_t t_start_us);

    // common_speculative_draft, per implementation: apply the cap before its draft and return it (-1 = none),
    // let the skipped sequences draft again after it, restore n_max after its drafts are taken
    int32_t cap_begin(common_speculative_type type, draft_params_vec & dparams);
    void    cap_drafted(int32_t cap_impl, draft_params_vec & dparams);
    void    cap_end(int32_t cap_impl, draft_params_vec & dparams);

    void cancel() {
        open = false;
    }

    void cycle_begin(const draft_params_vec & dparams);

    void draft_end(const draft_params_vec & dparams, const std::vector<common_speculative_impl *> & impl_last, int64_t t_us);

    // MTP: called for each drafted head token. false drops the token and ends the draft,
    // `more` = false keeps the token but ends the draft
    bool keep(llama_seq_id seq_id, const llama_token_data_array * cur_p, bool & more);

    void on_steps(int n_steps, int64_t t_us);

    void on_accept(llama_seq_id seq_id, uint16_t n_accepted);

    void on_verify(llama_seq_id seq_id, size_t i, const llama_token_data_array * cur_p, llama_token id_draft, float temp);

    void fit();

    void cycle_end(int64_t now);

    void print() const;
};
