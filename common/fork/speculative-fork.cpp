#include "speculative-fork.h"

#include "env-fork.h"
#include "common.h"
#include "ggml.h"
#include "log.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define SPC_INF(fmt, ...) LOG_INF("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)
#define SPC_WRN(fmt, ...) LOG_WRN("spec %12.*s: " fmt, 12, __func__, __VA_ARGS__)

static bool common_speculative_type_is_model(common_speculative_type type) {
    return type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE || type == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
           type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP    || type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
           type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
}

// the drafts that pass each token through keep(): the gate and the calibration by draft probability.
// DFlash since 22.10: on the 395 it verified all 3 positions with a position-only cap (0.726 / 0.496 / 0.327 accepted),
// where the MTP head stopped at the doubtful ones (4 tokens verified against ~3.2, the cycle 69.9 against 68.2 ms)
static bool common_speculative_type_is_gated(common_speculative_type type) {
    return type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP || type == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH ||
           type == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
}

//
// common_speculative_auto
//

common_speculative_auto::~common_speculative_auto() {
    if (trace) {
        fclose(trace);
    }
}

void common_speculative_auto::init(bool enabled, uint32_t n_seq, int32_t n_cap, const char * path_trace) {
    this->enabled = enabled;
    n_cap_probe = std::max(1, n_cap);
    seqs.assign(n_seq, {});

    // prior: the draft probability is the acceptance, worth two observations
    for (int t = 0; t < COMMON_SPECULATIVE_TYPE_COUNT; ++t) {
        for (int d = 0; d < n_depth; ++d) {
            for (int b = 0; b < n_bins; ++b) {
                cal_acc[t][d][b] = 2.0 * (b + 0.5) / n_bins;
                cal_n  [t][d][b] = 2.0;
            }
        }
    }

    if (path_trace && path_trace[0]) {
        trace = fopen(path_trace, "w");
        if (!trace) {
            SPC_WRN("cannot open the trace file '%s'\n", path_trace);
        } else {
            fprintf(trace, "cycle,mode,valid,seq,impl,n_draft,n_acc,t_cycle_us,t_draft_us,n_verify,rate_tps,t0_us,t1_us,t_step_us,thr_keep,thr_step,hp,pt,pm,ov\n");
            fflush(trace);
        }
    }
}

// the calibration column of a draft probability, -1 for one that is not a number: (int) of a NaN is undefined
static int common_speculative_auto_bin(float p) {
    if (!std::isfinite(p)) {
        return -1;
    }
    return std::clamp((int) (p * common_speculative_auto::n_bins), 0, common_speculative_auto::n_bins - 1);
}

double common_speculative_auto::acc_head(common_speculative_type type, int j, float p) const {
    const int d = std::min(j, n_depth - 1);
    const int b = common_speculative_auto_bin(p);
    return b < 0 ? 0.0 : cal_acc[type][d][b] / cal_n[type][d][b];
}

// prior 0.9 worth two observations: an ngram draft needs a long match to exist at all.
// positions 3+ share one estimate, as the MTP calibration rows: a matched ngram tends to keep matching.
// with one estimate per position, the far positions get few observations and the draft grows by only 1-2 tokens per cycle
double common_speculative_auto::acc_pos(common_speculative_type type, int j) const {
    const auto & a = pos_acc[type];
    const auto & n = pos_n[type];
    if (a.empty()) {
        return 0.9;
    }
    const int k = std::min(j, (int) a.size() - 1);
    return (a[k] + 1.8) / (n[k] + 2.0);
}

int32_t common_speculative_auto::impl_cap(common_speculative_type type) const {
    if (mode == MODE_PROBE) {
        return common_speculative_type_is_model(type) ? cap : -1;
    }
    if (mode != MODE_GATE || common_speculative_type_is_gated(type)) {
        return -1;
    }
    double p = 1.0;
    int32_t n = 0;
    while (n < 1024) {
        p *= acc_pos(type, n);
        if (p < thr_keep) {
            break;
        }
        n++;
    }
    return n;
}

int32_t common_speculative_auto::cap_begin(common_speculative_type type, draft_params_vec & dparams) {
    drafting = type;

    const int32_t cap_impl = active() ? impl_cap(type) : -1;
    if (cap_impl >= 0) {
        n_max_save.resize(dparams.size());
        skipped.resize(dparams.size(), 0);

        for (size_t seq_id = 0; seq_id < dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];
            n_max_save[seq_id] = dp.n_max;
            if (!dp.drafting) {
                continue;
            }
            if (cap_impl == 0) {
                dp.drafting = false;
                skipped[seq_id] = 1;
            } else if (dp.n_max <= 0 || cap_impl < dp.n_max) {
                dp.n_max = cap_impl;
            }
        }
    }

    return cap_impl;
}

void common_speculative_auto::cap_drafted(int32_t cap_impl, draft_params_vec & dparams) {
    if (cap_impl == 0) {
        for (size_t seq_id = 0; seq_id < dparams.size(); ++seq_id) {
            if (skipped[seq_id]) {
                dparams[seq_id].drafting = true;
                skipped[seq_id] = 0;
            }
        }
    }
}

void common_speculative_auto::cap_end(int32_t cap_impl, draft_params_vec & dparams) {
    if (cap_impl >= 0) {
        for (size_t seq_id = 0; seq_id < dparams.size(); ++seq_id) {
            dparams[seq_id].n_max = n_max_save[seq_id];
        }
    }
}

void common_speculative_auto::cycle_begin(const draft_params_vec & dparams) {
    const int64_t now = ggml_time_us();

    cycle_end(now);

    n_cycle++;

    mode = MODE_OFF;
    cap  = -1;

    if (enabled) {
        if (n_valid < n_warm || n_cycle % n_probe == 0) {
            mode = MODE_PROBE;
            cap  = (int32_t) (n_probes++ % (n_cap_probe + 1));
        } else if (fit_ok && rate() > 0.0) {
            mode     = MODE_GATE;
            thr_keep = rate() * t1;
            thr_step = rate() * (t1 + t_step);
        }
    }

    open    = true;
    valid   = true;
    t_start = now;
    t_draft = 0;

    for (size_t seq_id = 0; seq_id < seqs.size(); ++seq_id) {
        auto & s = seqs[seq_id];

        s.st    = dparams[seq_id].drafting ? 1 : 0;
        s.n_drf = 0;
        s.n_acc = 0;
        s.pcum  = 1.0;
        s.type  = COMMON_SPECULATIVE_TYPE_NONE;
        s.hp.clear();
        s.q.clear();
        s.q_dist.clear();
        s.pt.clear();
        s.pm.clear();
        s.ov.clear();
    }
}

void common_speculative_auto::draft_end(const draft_params_vec & dparams, const std::vector<common_speculative_impl *> & impl_last, int64_t t_us) {
    t_draft = t_us;

    for (size_t seq_id = 0; seq_id < seqs.size(); ++seq_id) {
        auto & s = seqs[seq_id];
        if (s.st == 0) {
            continue;
        }

        s.n_drf = dparams[seq_id].result ? (int32_t) dparams[seq_id].result->size() : 0;

        if (s.n_drf == 0) {
            // no draft: the target decodes one token and does not report an accept
            s.st = 2;
            s.hp.clear();
            continue;
        }

        s.type = impl_last[seq_id] ? common_speculative_impl_type(impl_last[seq_id]) : COMMON_SPECULATIVE_TYPE_NONE;
        if (!common_speculative_type_is_gated(s.type)) {
            s.hp.clear();
            s.q.clear();
            s.q_dist.clear();
        } else {
            // the caller may truncate the draft
            if ((int32_t) s.hp.size() > s.n_drf) {
                s.hp.resize(s.n_drf);
            }
            if ((int32_t) s.q.size() > s.n_drf) {
                s.q.resize(s.n_drf);
            }
            if ((int32_t) s.q_dist.size() > s.n_drf) {
                s.q_dist.resize(s.n_drf);
            }
        }
    }
}

int64_t common_speculative_auto::draft_begin(const draft_params_vec & dparams) {
    const int64_t t_start_us = ggml_time_us();

    if (active()) {
        cycle_begin(dparams);
    }

    return t_start_us;
}

void common_speculative_auto::draft_done(const draft_params_vec & dparams, const std::vector<common_speculative_impl *> & impl_last, int64_t t_start_us) {
    if (active()) {
        draft_end(dparams, impl_last, ggml_time_us() - t_start_us);
    }
}

bool common_speculative_auto::keep(llama_seq_id seq_id, const llama_token_data_array * cur_p, bool & more) {
    auto & s = seqs[seq_id];

    const float p = cur_p->data[0].p;

    if (mode == MODE_GATE) {
        const double pc = s.pcum * acc_head(drafting, (int) s.hp.size(), p);
        if (pc < thr_keep) {
            return false;
        }
        s.pcum = pc;
        more   = pc * acc_mean(drafting) >= thr_step;
    }

    s.hp.push_back(p);

    if (trace) {
        s.q.emplace_back(cur_p->data, cur_p->data + std::min<size_t>(cur_p->size, 10));
        s.q_dist.push_back(1);
    }

    return true;
}

void common_speculative_auto::pad(llama_seq_id seq_id, const llama_token_data_array * cur_p) {
    if (seq_id < 0 || seq_id >= (llama_seq_id) seqs.size()) {
        return;
    }

    auto & s = seqs[seq_id];

    s.hp.push_back(cur_p->data[0].p);

    if (trace) {
        s.q.emplace_back(cur_p->data, cur_p->data + std::min<size_t>(cur_p->size, 10));
        s.q_dist.push_back(1);
    }
}

void common_speculative_auto::on_steps(int n_steps, int64_t t_us) {
    if (n_steps <= 0) {
        return;
    }
    const double v = (double) t_us / n_steps;
    t_step = t_step > 0.0 ? 0.95 * t_step + 0.05 * v : v;
}

void common_speculative_auto::on_accept(llama_seq_id seq_id, uint16_t n_accepted) {
    if (!open || seq_id < 0 || seq_id >= (llama_seq_id) seqs.size()) {
        return;
    }

    auto & s = seqs[seq_id];

    // a second report of the same draft (rollback, then replay) spans two target passes
    if (s.st != 1) {
        valid = false;
        return;
    }

    s.st    = 2;
    s.n_acc = n_accepted;

    const int n_test = std::min<int>(n_accepted + 1, s.n_drf);

    if (common_speculative_type_is_gated(s.type) && (int) s.hp.size() == s.n_drf) {
        const int t = s.type;
        for (int j = 0; j < n_test; ++j) {
            const int d = std::min(j, n_depth - 1);
            const int b = common_speculative_auto_bin(s.hp[j]);
            const double a = j < n_accepted ? 1.0 : 0.0;

            // a probability that is not a number has no column: its acceptance still counts for the mean
            if (b >= 0) {
                cal_acc[t][d][b] = cal_acc[t][d][b] * f_cal + a;
                cal_n  [t][d][b] = cal_n  [t][d][b] * f_cal + 1.0;
            }

            acc_sum[t] = acc_sum[t] * f_cal + a;
            acc_n  [t] = acc_n  [t] * f_cal + 1.0;
        }
    } else if (s.type != COMMON_SPECULATIVE_TYPE_NONE) {
        auto & a = pos_acc[s.type];
        auto & n = pos_n  [s.type];
        const int n_row = std::min(n_test, n_depth);
        if ((int) a.size() < n_row) {
            a.resize(n_row, 0.0);
            n.resize(n_row, 0.0);
        }
        for (int j = 0; j < n_test; ++j) {
            const int d = std::min(j, n_depth - 1);
            a[d] = a[d] * f_cal + (j < n_accepted ? 1.0 : 0.0);
            n[d] = n[d] * f_cal + 1.0;
        }
    }
}

void common_speculative_auto::on_verify(llama_seq_id seq_id, size_t i, const llama_token_data_array * cur_p, llama_token id_draft, float temp) {
    if (!trace || !open || seq_id < 0 || seq_id >= (llama_seq_id) seqs.size()) {
        return;
    }

    auto & s = seqs[seq_id];
    if (s.st != 1 || i != s.pt.size()) {
        return;
    }

    float pt = 0.0f;
    float pm = 0.0f;
    for (size_t k = 0; k < cur_p->size; ++k) {
        pm = std::max(pm, cur_p->data[k].p);
        if (cur_p->data[k].id == id_draft) {
            pt = cur_p->data[k].p;
        }
    }

    // the draft candidates at the target temperature against the target distribution
    float ov = -1.0f;
    if (i < s.q.size() && !s.q[i].empty() && i < s.q_dist.size() && s.q_dist[i]) {
        const auto & q = s.q[i];
        if (temp <= 0.0f) {
            ov = pt;
        } else {
            float lmax = q[0].logit;
            for (const auto & c : q) {
                lmax = std::max(lmax, c.logit);
            }
            std::vector<float> w(q.size());
            float sum = 0.0f;
            for (size_t k = 0; k < q.size(); ++k) {
                w[k] = std::exp((q[k].logit - lmax) / temp);
                sum += w[k];
            }
            ov = 0.0f;
            for (size_t k = 0; k < cur_p->size; ++k) {
                for (size_t m = 0; m < q.size(); ++m) {
                    if (q[m].id == cur_p->data[k].id) {
                        ov += std::min(w[m] / sum, cur_p->data[k].p);
                        break;
                    }
                }
            }
        }
    }

    s.pt.push_back(pt);
    s.pm.push_back(pm);
    s.ov.push_back(ov);
}

void common_speculative_auto::fit() {
    const double det = s0 * sxx - sx * sx;

    // the draft lengths must spread, otherwise the slope is noise
    if (s0 < 8.0 || det < 0.1 * s0 * s0) {
        fit_ok = false;
        return;
    }

    const double b = (s0 * sxy - sx * sy) / det;

    fit_ok = b > 0.0;
    if (fit_ok) {
        t1 = b;
        t0 = (sy - b * sx) / s0;
    }
}

void common_speculative_auto::cycle_end(int64_t now) {
    if (!open) {
        return;
    }
    open = false;

    int32_t n_seq = 0;
    int32_t n_tok = 0;
    int32_t n_ver = 0;

    for (const auto & s : seqs) {
        if (s.st == 0) {
            continue;
        }
        if (s.st == 1) {
            valid = false;
        }
        n_seq++;
        n_tok += 1 + s.n_acc;
        n_ver += 1 + s.n_drf;
    }

    if (n_seq == 0) {
        return;
    }

    const double t_cycle = (double) (now - t_start);
    const double x = n_ver;
    const double y = t_cycle - (double) t_draft;

    // the pass shared its time with other work (a prompt, an idle wait)
    if (valid && fit_ok && y > 3.0 * (t0 + t1 * x)) {
        valid = false;
    }
    if (valid && t_cycle > 60e6) {
        valid = false;
    }

    if (valid) {
        s0  = s0  * f_cycle + 1.0;
        sx  = sx  * f_cycle + x;
        sxx = sxx * f_cycle + x * x;
        sy  = sy  * f_cycle + y;
        sxy = sxy * f_cycle + x * y;
        fit();

        r_tok = r_tok * f_cycle + n_tok;
        r_us  = r_us  * f_cycle + t_cycle;

        n_valid++;

        if (mode == MODE_GATE) {
            for (const auto & s : seqs) {
                if (s.st != 0) {
                    hist[std::min(s.n_drf, n_hist - 1)]++;
                }
            }
        }
    }

    if (!trace) {
        return;
    }

    static const char * mode_str[] = { "off", "probe", "gate" };

    const auto join = [](const std::vector<float> & v) {
        std::string res;
        for (size_t k = 0; k < v.size(); ++k) {
            res += string_format(k == 0 ? "%.4f" : ";%.4f", v[k]);
        }
        return res;
    };

    for (size_t seq_id = 0; seq_id < seqs.size(); ++seq_id) {
        const auto & s = seqs[seq_id];
        if (s.st == 0) {
            continue;
        }
        fprintf(trace, "%" PRId64 ",%s,%d,%zu,%s,%d,%d,%.0f,%" PRId64 ",%d,%.3f,%.1f,%.1f,%.1f,%.4f,%.4f,%s,%s,%s,%s\n",
                n_cycle, mode_str[mode], valid ? 1 : 0, seq_id,
                s.n_drf > 0 ? common_speculative_type_to_str(s.type).c_str() : "none",
                s.n_drf, s.n_acc, t_cycle, t_draft, n_ver,
                rate() * 1e6, t0, t1, t_step, thr_keep, thr_step,
                join(s.hp).c_str(), join(s.pt).c_str(), join(s.pm).c_str(), join(s.ov).c_str());
    }
    // the process can be killed without exit, so do not keep lines in the buffer
    fflush(trace);
}

void common_speculative_auto::print() const {
    SPC_INF("auto: %s, cycles %" PRId64 " (%" PRId64 " valid), rate %.2f t/s, verify %.2f ms + %.3f ms/token%s, draft step %.3f ms, keep > %.3f, step > %.3f\n",
            enabled ? "on" : "trace only", n_cycle, n_valid, rate() * 1e6, t0 / 1e3, t1 / 1e3, fit_ok ? "" : " (no fit)",
            t_step / 1e3, thr_keep, thr_step);

    std::string str;
    for (int k = 0; k < n_hist; ++k) {
        if (hist[k] > 0) {
            str += string_format(" %d%s:%" PRId64, k, k == n_hist - 1 ? "+" : "", hist[k]);
        }
    }
    if (!str.empty()) {
        SPC_INF("auto: gated draft lengths:%s\n", str.c_str());
    }

    for (int t = 0; t < COMMON_SPECULATIVE_TYPE_COUNT; ++t) {
        if (acc_n[t] <= 0.0) {
            continue;
        }
        const std::string name = common_speculative_type_to_str((common_speculative_type) t);
        SPC_INF("auto: %s: mean acc %.3f\n", name.c_str(), acc_mean((common_speculative_type) t));
        for (int d = 0; d < n_depth; ++d) {
            str.clear();
            for (int b = 0; b < n_bins; ++b) {
                str += string_format(" %.2f", cal_acc[t][d][b] / cal_n[t][d][b]);
            }
            SPC_INF("auto: %s: acceptance by draft p (0.0..1.0 in steps of 0.1), pos %d%s:%s\n", name.c_str(), d,
                    d == n_depth - 1 ? "+" : "", str.c_str());
        }
    }

    for (int t = 0; t < COMMON_SPECULATIVE_TYPE_COUNT; ++t) {
        if (pos_n[t].empty()) {
            continue;
        }
        str.clear();
        for (size_t j = 0; j < pos_n[t].size(); ++j) {
            str += string_format(" %.2f", acc_pos((common_speculative_type) t, (int) j));
        }
        SPC_INF("auto: acceptance by position (0, 1, 2, 3+), %s:%s\n", common_speculative_type_to_str((common_speculative_type) t).c_str(), str.c_str());
    }
}

bool common_speculative_fork_lattice_keep(common_speculative_auto * ctl, llama_seq_id seq_id, const float * ids,
        const float * scores, int32_t top_k, int32_t chosen) {
    // a selector row of -inf only gives the pick no score (-inf - -inf is NaN): no probabilities then, keep() gets NaN,
    // which the gate drops and the calibration skips. a candidate without a number has none either
    const bool scored = std::isfinite(scores[chosen]);

    std::vector<llama_token_data> cand((size_t) top_k);
    float sum = 0.0f;
    for (int32_t k = 0; k < top_k; ++k) {
        const float p = scored && std::isfinite(scores[k]) ? std::exp(scores[k] - scores[chosen]) : 0.0f;
        cand[k] = { (llama_token) ids[k], scores[k], p };
        sum += p;
    }
    // with a score the pick adds exp(0) = 1 to the sum
    for (auto & c : cand) {
        c.p = scored ? c.p / sum : NAN;
    }
    // the pick is the argmax, so it leads the order keep() and the trace read. a NaN breaks the order of a sort
    std::swap(cand[0], cand[chosen]);
    if (scored) {
        std::sort(cand.begin() + 1, cand.end(), [](const llama_token_data & a, const llama_token_data & b) { return a.p > b.p; });
    }

    llama_token_data_array cur_p = { cand.data(), cand.size(), 0, true };
    bool more = true;
    const bool keep = ctl->keep(seq_id, &cur_p, more);

    // the scores of the selector are no token distribution of the drafter: no overlap with the target in the trace
    auto & s = ctl->seqs[seq_id];
    if (keep && !s.q_dist.empty()) {
        s.q_dist.back() = 0;
    }
    return keep;
}

uint32_t common_speculative_fork_mtp_ubatch(uint32_t n_ubatch) {
    const int64_t cap = common_fork_env_int("LLAMA_MTP_UBATCH", 1024);
    return cap > 0 ? (uint32_t) std::min<int64_t>(n_ubatch, cap) : n_ubatch;
}

bool common_speculative_fork_even_drafts() {
    static const bool even = [] {
        return common_fork_env_int("LLAMA_SPEC_EVEN", 1) != 0;
    }();
    return even;
}

int32_t common_speculative_fork_draft_by_slots(int n_gen) {
    static const std::vector<int32_t> caps = [] {
        static const char * def = "3,3,2,2,2,1";

        // whole numbers separated by commas; an item that is not one would read as 0, a draft cap of 0
        const auto parse = [](const std::string & text, std::vector<int32_t> & res) {
            size_t pos = 0;
            while (pos < text.size()) {
                const size_t end = text.find(',', pos);
                const std::string item = text.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
                if (item.find_first_not_of(" \t") != std::string::npos) {
                    errno = 0;
                    char * e = nullptr;
                    const long v = std::strtol(item.c_str(), &e, 10);
                    while (*e == ' ' || *e == '\t') {
                        e++;
                    }
                    if (e == item.c_str() || *e != '\0' || errno == ERANGE || v < INT32_MIN || v > INT32_MAX) {
                        return false;
                    }
                    res.push_back((int32_t) v);
                }
                if (end == std::string::npos) {
                    break;
                }
                pos = end + 1;
            }
            return true;
        };

        const char * env = getenv("LLAMA_SPEC_DRAFT_BY_SLOTS");
        std::vector<int32_t> res;
        if (env == nullptr || !parse(env, res)) {
            if (env != nullptr) {
                SPC_WRN("LLAMA_SPEC_DRAFT_BY_SLOTS=\"%s\" is not a list of whole numbers, using %s\n", env, def);
            }
            res.clear();
            parse(def, res);
        }
        if (res.size() == 1 && res[0] == 0) {
            res.clear();
        }
        return res;
    }();
    if (caps.empty() || n_gen <= 1) {
        return -1;
    }
    return caps[std::min<size_t>((size_t) n_gen - 1, caps.size() - 1)];
}

uint32_t common_speculative_fork_dflash_ubatch(uint32_t n_ubatch) {
    const int64_t cap = common_fork_env_int("LLAMA_DFLASH_UBATCH", 512);
    return cap > 0 ? (uint32_t) std::min<int64_t>(n_ubatch, cap) : n_ubatch;
}

void common_speculative_fork_dflash_fit(const llama_model * model_dft, int32_t n_max, llama_context_params & cparams) {
    char buf[32] = {};
    if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0 && std::string(buf) == "true") {
        return; // a causal draft may split its batch
    }

    int32_t per_seq = std::max(1, n_max + 1);
    if (llama_model_meta_val_str(model_dft, "lilicorr.candidate_topk", buf, sizeof(buf)) >= 0 && std::atoi(buf) > 0) {
        int32_t block_size = 16;
        if (llama_model_meta_val_str(model_dft, "dflash.block_size", buf, sizeof(buf)) >= 0) {
            block_size = std::atoi(buf);
        }
        per_seq = std::max(per_seq, block_size);
    }

    const uint32_t need = (uint32_t) per_seq * std::max<uint32_t>(1, cparams.n_seq_max);
    if (cparams.n_ubatch < need || cparams.n_batch < need) {
        SPC_INF("dflash: the blocks of %u sequences (%d tokens each) decode as one ubatch, n_ubatch %u -> %u, n_batch %u -> %u\n",
                cparams.n_seq_max, per_seq, cparams.n_ubatch, std::max(cparams.n_ubatch, need), cparams.n_batch, std::max(cparams.n_batch, need));
        cparams.n_ubatch = std::max(cparams.n_ubatch, need);
        cparams.n_batch  = std::max(cparams.n_batch, need);
    }
}

static llama_model * g_draft_preloaded = nullptr;
static std::string   g_draft_preloaded_path;

void common_speculative_fork_preload_draft(const common_params & params_base) {
    // on by default since 15.10 night (the 395: speed the same, see the header); LLAMA_DRAFT_LOAD_FIRST=0: upstream order
    if (common_fork_env_int("LLAMA_DRAFT_LOAD_FIRST", 1) == 0 || !params_base.speculative.has_dft() || g_draft_preloaded != nullptr) {
        return;
    }
    // the same parameters as common_speculative_init_result gets from the server
    // mparams points into params_dft (devices, overrides), which lives until the load returns
    common_params params_dft = common_base_params_to_speculative(params_base);
    const auto mparams = common_model_params_to_llama(params_dft);
    const std::string & path = params_dft.speculative.draft.mparams.path;
    LOG_INF("%s: loading the draft model '%s' before the target (LLAMA_DRAFT_LOAD_FIRST)\n", __func__, path.c_str());
    g_draft_preloaded = llama_model_load_from_file(path.c_str(), mparams);
    g_draft_preloaded_path = g_draft_preloaded ? path : std::string();
}

llama_model * common_speculative_fork_take_draft(const std::string & path) {
    if (g_draft_preloaded == nullptr || path != g_draft_preloaded_path) {
        return nullptr;
    }
    llama_model * m = g_draft_preloaded;
    g_draft_preloaded = nullptr;
    g_draft_preloaded_path.clear();
    return m;
}

bool common_speculative_fork_catchup() {
    static const bool enabled = [] {
        return common_fork_env_int("LLAMA_SPEC_CATCHUP_FORK", 1) != 0;
    }();
    return enabled;
}

bool common_speculative_fork_ngram_cache_keep() {
    static const bool keep = [] {
        return common_fork_env_int("LLAMA_NGRAM_CACHE_KEEP", 0) != 0;
    }();
    return keep;
}
