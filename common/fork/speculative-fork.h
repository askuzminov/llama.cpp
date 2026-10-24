#pragma once

// the fork's speculative decoding additions: the cost-aware draft length (--spec-auto) and the per-cycle trace

#include "llama.h"
#include "common.h"
#include "speculative.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// draft length cap of --spec-auto when --spec-draft-n-max is not given. 3 since 12.10: on the 395 the cap 6 ran 35.87
// t/s against 37.43 with 3 (MTP without ngram-mod, the same prompts), its verify cost 7.7 ms per token against 6.5 and
// its recurrent state 788 MiB against 450; the user's own test on the 3090 also had 3 ahead. ngram-mod drafts are not
// bound by it
#define COMMON_SPECULATIVE_AUTO_N_MAX 3
// the cap of --spec-auto for a DFlash / DSpark draft without --spec-draft-n-max (the draft clamps it to its block size
// - 1). 7 measured 22.10 with the gate (chats to 213K): the 395 36.91 against 36.60 t/s at 3 (the block of 8 drafts in
// 10.8 against 7.2 ms), the 3090 26.42 against 28.02 (the 7 rollback states of the target take 450 MiB more VRAM from
// the expert cache, 49 slots per layer against 53, and the verifies get longer). so 3, as the MTP head
#define COMMON_SPECULATIVE_AUTO_N_MAX_DFLASH 3

struct common_speculative_impl;

// the type of an implementation (defined in speculative.cpp, where the struct is)
common_speculative_type common_speculative_impl_type(const common_speculative_impl * impl);

// the ubatch of an MTP draft context. it holds as many cells as the target, and its dense KQ mask is n_ctx x n_ubatch
// f16 both in the host compute buffer and on the device: 2 GiB each at 262144 x 4096 (the 395 11.10: compute buffers
// of the MTP context 2800 MiB on the device and 2248 MiB pinned on the host). the catch-up over a prompt goes in
// chunks of this size, llama_decode splits the target ubatch. LLAMA_MTP_UBATCH: tokens (default 1024), 0 = the target's
uint32_t common_speculative_fork_mtp_ubatch(uint32_t n_ubatch);

// the ubatch of a DFlash / DSpark draft context. its decoder graph is reserved for n_ubatch tokens and computes the
// logits of all of them, [n_vocab, n_ubatch] f32: 3.8 GiB at 248320 x 4096, while a draft decodes one block (8
// positions per sequence) and the KV injection of the target features goes in chunks of this size anyway. on the 395
// (22.10) the reserve put the LM head of the draft on the CPU, with a 644 MiB copy of the target's output.weight and a
// 4564 MiB host compute buffer. LLAMA_DFLASH_UBATCH: tokens (default 512), 0 = the target's (or --spec-draft-n-ubatch)
uint32_t common_speculative_fork_dflash_ubatch(uint32_t n_ubatch);

// a non-causal draft (DFlash, unless its model says dflash.attention.causal) decodes the blocks of all drafting
// sequences as one batch, which llama_decode takes only as one ubatch: n_seq_max blocks of n_max + 1 tokens (LiliCorr:
// of its block size). its context then gets at least that many tokens per ubatch and per batch, over the cap above and
// over a smaller -ub; the assert of a larger batch was reached from 129 slots at n_max 3, from 33 at a block of 16
void common_speculative_fork_dflash_fit(const llama_model * model_dft, int32_t n_max, llama_context_params & cparams);

// the drafts of several generating slots (23.10, after Rulith Inference's apply_spec_draft_by_slots and
// apply_spec_even_drafts). the hybrid memory splits a verify batch into ubatches of equal tokens per sequence, so
// drafts of different lengths verify in two or three passes of the whole model (Rulith, three conversations at 4K:
// 34.3 -> 45.7 tok/s summed once even); and in this MoE a verified token reads ~10 more experts, which conversations
// decoding together cannot share (Rulith, four: 37.9 -> 47.4 tok/s with the cap).
// even: the MTP drafter keeps a sequence whose confident run ended drafting while another one is still confident and
// cuts all of them to the longest confident run; common_speculative_draft cuts what is left uneven to the shortest.
// LLAMA_SPEC_EVEN=0: per sequence, as upstream
bool    common_speculative_fork_even_drafts();
// the draft cap of a step with n_gen generating slots, -1 for none. LLAMA_SPEC_DRAFT_BY_SLOTS: the caps for 1, 2, ...
// slots, the last one for more (default "3,3,2,2,2,1": what Rulith's measured adaptation picked on real tasks - two
// conversations three drafts, four two, eight one), "0" or "" = no cap. one slot keeps --spec-auto as it is
int32_t common_speculative_fork_draft_by_slots(int n_gen);

// the catch-up of the MTP head over a decoded batch (process of the MTP implementation): in chunks of the draft
// ubatch, so that the rows of the next chunk are rendered while the GPU computes the last one, and a batch longer than
// a verify keeps only its last row of hidden state (drafting goes on from it; accept follows verify batches only).
// on the 395 (14.10, GGML_TIMELINE) a 16384-token prompt batch stood the GPU 772 ms before the catch-up (the batch of
// 16384 rows of 40 KiB rendered in one piece) and 288 ms after it (all rows copied for the verify). false with
// LLAMA_SPEC_CATCHUP_FORK=0: one decode call of the whole batch and every row kept (upstream)
bool common_speculative_fork_catchup();

// ngram-cache: begin() of a prompt clears the context cache of the slot (the default since cca809f46), the prompt is
// indexed again in full from cache_size 0, so the n-grams of a chat history are counted once. upstream (since 83c2d012a)
// keeps the context cache across prompts, the history of a chat counted again with every prompt of it, and merges it
// into the dynamic cache file when the implementation goes. LLAMA_NGRAM_CACHE_KEEP=1: upstream
bool common_speculative_fork_ngram_cache_keep();

// LLAMA_DRAFT_LOAD_FIRST (1 by default since 15.10 night, 0: upstream order): the server loads the weights of the draft
// model (-md) before the target model and its context, and common_speculative_init_result takes them. the 395 (15.10
// night): PP 728.1 against 727.2, TG 37.75 against 37.55 in the chats to 213K; in 10 starts each, the head buffer past
// the carve-out failed to allocate once in the upstream order, never loaded first. on the 395 the head loaded last, after the KV
// and the compute buffers of the target, at ~99.2 GB of the device heap: past the 96 GiB carve-out, and its first
// upload stalled in some starts (15.10: the first start of each of three profile arms, exit 75 after 181 s). loaded
// first, its buffer is placed while the device is nearly empty and what goes past the carve-out is a compute buffer
void          common_speculative_fork_preload_draft(const common_params & params_base);
// the preloaded draft model of this path, once, nullptr when there is none
llama_model * common_speculative_fork_take_draft(const std::string & path);

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
    static constexpr int    n_bins  = 10;    // calibration columns: draft probability in steps of 0.1
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

        std::vector<float> hp; // draft probability of each drafted token: the MTP head, the DFlash selector or sampler

        // trace only
        std::vector<std::vector<llama_token_data>> q; // draft candidates per position
        std::vector<uint8_t> q_dist; // per position: 1 when the logits of q are a token distribution of the drafter
        std::vector<float> pt; // target probability of the drafted token
        std::vector<float> pm; // highest target probability
        std::vector<float> ov; // sum of min(target, draft), the acceptance of exact speculative sampling with the drafter
                               // sampling at the target temperature; -1 where the drafter has no token distribution (the
                               // candidates of a DFlash2 lattice carry selector scores, not logits)
    };

    bool   enabled = false;
    FILE * trace   = nullptr;

    // per draft type: the probabilities of the gated drafts mean different things (the MTP head: its confidence, the
    // DFlash2 lattice: the share of the pick among the top_k of its selector row), so each type has its own table
    double cal_acc[COMMON_SPECULATIVE_TYPE_COUNT][n_depth][n_bins] = {};
    double cal_n  [COMMON_SPECULATIVE_TYPE_COUNT][n_depth][n_bins] = {};

    std::vector<double> pos_acc[COMMON_SPECULATIVE_TYPE_COUNT];
    std::vector<double> pos_n  [COMMON_SPECULATIVE_TYPE_COUNT];

    // mean conditional acceptance of the gated positions per type, predicts the next position before it is drafted
    double acc_sum[COMMON_SPECULATIVE_TYPE_COUNT] = {};
    double acc_n  [COMMON_SPECULATIVE_TYPE_COUNT] = {};

    // the implementation whose draft keep() and pad() see, set by cap_begin
    common_speculative_type drafting = COMMON_SPECULATIVE_TYPE_NONE;

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

    double acc_mean(common_speculative_type type) const {
        return acc_n[type] > 0.0 ? acc_sum[type] / acc_n[type] : 0.5;
    }

    // the calibrated acceptance of a draft of this type at position j with probability p, 0 for a p that is not a
    // number (a DFlash selector row of -inf only)
    double acc_head(common_speculative_type type, int j, float p) const;
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

    // MTP and DFlash: called for each drafted token. false drops the token and ends the draft,
    // `more` = false keeps the token but ends the draft (MTP: no next head step; a DFlash block costs one decode)
    bool keep(llama_seq_id seq_id, const llama_token_data_array * cur_p, bool & more);

    void on_steps(int n_steps, int64_t t_us);

    // MTP with even drafts: a token drafted after the sequence's confident run, to the length of another sequence's
    // run (common_speculative_fork_even_drafts). its probability goes to the calibration with the verify result, the
    // gate does not decide it
    void pad(llama_seq_id seq_id, const llama_token_data_array * cur_p);

    void on_accept(llama_seq_id seq_id, uint16_t n_accepted);

    void on_verify(llama_seq_id seq_id, size_t i, const llama_token_data_array * cur_p, llama_token id_draft, float temp);

    void fit();

    void cycle_end(int64_t now);

    void print() const;
};

// a DFlash2 / LiLiCorr lattice position for the gate of --spec-auto: the candidate ids and their scores given the
// pick at the position before, the softmax over them as the draft probabilities (the pick first). false ends the draft
bool common_speculative_fork_lattice_keep(common_speculative_auto * ctl, llama_seq_id seq_id, const float * ids,
        const float * scores, int32_t top_k, int32_t chosen);
