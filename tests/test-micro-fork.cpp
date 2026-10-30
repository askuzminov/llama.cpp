// fork: the qwen4exp QSA indexer on one device at the shapes of the model, each way the graph can build it, timed as
// whole graphs (test-backend-ops perf repeats only the last node, so it cannot time a fused head matmul):
//   lid      LIGHTNING_INDEXER: sum_h relu(q_h . k) + mask in one op (the CUDA and Vulkan kernels)
//   headsum  four MUL_MAT + RELU + ADD and the bias add, as the qwen4exp graph builds them on Vulkan (split heads),
//            where they fuse into one MUL_MAT_HEADSUM
//   batched  one MUL_MAT over the four heads, RELU and the sum of the heads, the graph of the other backends
// q [128, 4 x tokens] f32 and the block keys [128, blocks] f16 are the same for every case (fixed seed); every block
// is visible (zero mask or bias), as for a decode step at depth.
//
// usage: test-micro-fork [-d <device>] [-c <regex of the case>] [-t <tokens,...>] [-b <blocks,...>] [--ms <ms>]
// one line per shape and case: microseconds per graph (median), TFLOP/s of the q.k products, GB/s of the bytes a kernel
// has to move at least (keys, q, visibility, output), and the largest difference to the first case relative to its
// largest score

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <regex>
#include <string>
#include <vector>

static constexpr int64_t HEAD = 128; // the indexer head size of qwen4exp
static constexpr int64_t NH   = 4;   // its heads

enum class micro_kind { lid, headsum, batched };

struct micro_case {
    const char * name;
    micro_kind   kind;
};

static const micro_case MICRO_CASES[] = {
    { "lid",     micro_kind::lid     },
    { "headsum", micro_kind::headsum },
    { "batched", micro_kind::batched },
};

static std::vector<int64_t> parse_list(const char * s) {
    std::vector<int64_t> out;
    for (const char * p = s; *p; ) {
        char * end = nullptr;
        const long long v = strtoll(p, &end, 10);
        if (end == p) {
            break;
        }
        out.push_back(v);
        p = *end == ',' ? end + 1 : end;
    }
    return out;
}

// one copy of the case over the shared inputs, in the node order of build_qsa_top_k; returns its output
static ggml_tensor * build_one(ggml_context * ctx, ggml_cgraph * gf, micro_kind kind, ggml_tensor * q, ggml_tensor * k,
                               ggml_tensor * w, ggml_tensor * vis, int64_t tok, int64_t blk) {
    switch (kind) {
        case micro_kind::lid: {
            ggml_tensor * out = ggml_lightning_indexer(ctx,
                    ggml_reshape_4d(ctx, q,   HEAD, NH, tok, 1),
                    ggml_reshape_4d(ctx, k,   HEAD, 1, blk, 1), w,
                    ggml_reshape_4d(ctx, vis, blk, tok, 1, 1));
            ggml_build_forward_expand(gf, out);
            return out;
        }
        case micro_kind::headsum: {
            // each head joins the sum right after it is computed
            ggml_tensor * score = nullptr;
            for (int64_t h = 0; h < NH; ++h) {
                ggml_tensor * qh   = ggml_view_2d(ctx, q, HEAD, tok, NH*q->nb[1], h*q->nb[1]);
                ggml_tensor * head = ggml_relu(ctx, ggml_mul_mat(ctx, k, qh));
                score = score ? ggml_add(ctx, score, head) : head;
                ggml_build_forward_expand(gf, score);
            }
            score = ggml_add(ctx, score, vis);
            ggml_build_forward_expand(gf, score);
            return score;
        }
        case micro_kind::batched: {
            ggml_tensor * score = ggml_relu(ctx, ggml_mul_mat(ctx, k, q));
            score = ggml_reshape_3d(ctx, score, blk, NH, tok);
            ggml_tensor * summed = nullptr;
            for (int64_t h = 0; h < NH; ++h) {
                ggml_tensor * head = ggml_view_2d(ctx, score, blk, tok, score->nb[2], h*score->nb[1]);
                summed = summed ? ggml_add(ctx, summed, head) : ggml_cont(ctx, head);
            }
            summed = ggml_add(ctx, summed, vis);
            ggml_build_forward_expand(gf, summed);
            return summed;
        }
    }
    return nullptr;
}

struct micro_result {
    bool               ok = false;
    std::string        note;
    double             us = 0.0;
    int                reps = 0;
    std::vector<float> out; // the first copy's output
};

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// the case as a graph of reps copies, reps chosen so that one compute takes about 2 ms; times computes for at least
// min_ms and returns the median per copy
static micro_result run_case(ggml_backend_t backend, micro_kind kind, int64_t tok, int64_t blk, double min_ms, int fixed_reps) {
    micro_result res;
    int reps = fixed_reps > 0 ? fixed_reps : 1;
    for (int attempt = 0; attempt < 2; ++attempt) {
        const size_t n_nodes = (size_t) reps*24 + 32;
        ggml_init_params ip = {
            /*.mem_size   =*/ ggml_tensor_overhead()*(2*n_nodes) + ggml_graph_overhead_custom(2*n_nodes, false),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        ggml_context * ctx = ggml_init(ip);

        ggml_tensor * q   = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, HEAD, NH*tok);
        ggml_tensor * k   = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, HEAD, blk);
        ggml_tensor * vis = ggml_new_tensor_2d(ctx, kind == micro_kind::lid ? GGML_TYPE_F16 : GGML_TYPE_F32, blk, tok);
        // inputs and outputs at once: the graph allocator reuses the memory of an input once its last reader ran, and
        // the inputs are set once here for many computes
        for (ggml_tensor * t : { q, k, vis }) {
            ggml_set_input(t);
            ggml_set_output(t);
        }

        ggml_cgraph * gf = ggml_new_graph_custom(ctx, 2*n_nodes, false);

        // the head weights of the qwen4exp graph: a FILL of 1
        ggml_tensor * w = nullptr;
        if (kind == micro_kind::lid) {
            w = ggml_fill(ctx, ggml_new_tensor_4d(ctx, GGML_TYPE_F32, NH, tok, 1, 1), 1.0f);
        }

        ggml_tensor * first = nullptr;
        for (int r = 0; r < reps; ++r) {
            ggml_tensor * out = build_one(ctx, gf, kind, q, k, w, vis, tok, blk);
            ggml_set_output(out);
            first = first ? first : out;
        }

        // a node the device does not run would fall back to nothing here (no scheduler): skip the case
        for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
            ggml_tensor * node = ggml_graph_node(gf, i);
            const bool empty = node->op == GGML_OP_NONE || node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE ||
                               node->op == GGML_OP_PERMUTE || node->op == GGML_OP_TRANSPOSE;
            if (!empty && !ggml_backend_supports_op(backend, node)) {
                res.note = std::string("not supported: ") + ggml_op_desc(node);
                ggml_free(ctx);
                return res;
            }
        }

        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!ggml_gallocr_alloc_graph(galloc, gf)) {
            res.note = "no memory";
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            return res;
        }

        // the same inputs for every case of a shape
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<float> qv(ggml_nelements(q));
        for (float & x : qv) {
            x = dist(rng);
        }
        ggml_backend_tensor_set(q, qv.data(), 0, ggml_nbytes(q));
        std::vector<ggml_fp16_t> kv(ggml_nelements(k));
        for (ggml_fp16_t & x : kv) {
            x = ggml_fp32_to_fp16(dist(rng));
        }
        ggml_backend_tensor_set(k, kv.data(), 0, ggml_nbytes(k));
        std::vector<uint8_t> zeros(ggml_nbytes(vis), 0); // 0 is a zero in f16 and f32
        ggml_backend_tensor_set(vis, zeros.data(), 0, zeros.size());

        // the first computes build the pipelines
        for (int i = 0; i < 2; ++i) {
            if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
                res.note = "compute failed";
                ggml_gallocr_free(galloc);
                ggml_free(ctx);
                return res;
            }
        }

        double t0 = now_s();
        ggml_backend_graph_compute(backend, gf);
        const double t1 = now_s() - t0;
        if (attempt == 0 && t1 < 1.5e-3 && fixed_reps <= 0) {
            // a short graph: more copies, so that the submit and wait of one compute do not count
            reps = (int) std::min<double>(256.0, std::ceil(2e-3 / std::max(t1, 1e-6)));
            ggml_gallocr_free(galloc);
            ggml_free(ctx);
            continue;
        }

        std::vector<double> times;
        double total = 0.0;
        while (total < min_ms*1e-3 || times.size() < 5) {
            t0 = now_s();
            ggml_backend_graph_compute(backend, gf);
            const double dt = now_s() - t0;
            times.push_back(dt);
            total += dt;
            if (times.size() >= 2000) {
                break;
            }
        }
        std::sort(times.begin(), times.end());
        res.us   = times[times.size()/2] / reps * 1e6;
        res.reps = reps;

        res.out.resize(ggml_nelements(first));
        ggml_backend_tensor_get(first, res.out.data(), 0, ggml_nbytes(first));

        res.ok = true;
        ggml_gallocr_free(galloc);
        ggml_free(ctx);
        return res;
    }
    return res;
}

static double rel_diff(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) {
        return NAN;
    }
    double amax = 0.0, dmax = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        amax = std::max(amax, (double) std::fabs(a[i]));
        dmax = std::max(dmax, (double) std::fabs(a[i] - b[i]));
    }
    return amax > 0.0 ? dmax / amax : dmax;
}

int main(int argc, char ** argv) {
    const char *         dev_name = nullptr;
    std::string          filter   = ".";
    std::vector<int64_t> tokens   = { 1, 2, 3, 4, 8, 16, 32, 64, 65, 128, 512, 4096 }; // decode, MTP checks, prompts
    std::vector<int64_t> blocks   = { 1024, 8192, 32768, 65536 }; // 4-cell blocks: a cache of 4K, 32K, 128K, 256K
    double               min_ms   = 200.0;
    int                  reps     = 0; // 0: chosen per shape

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-d" && i + 1 < argc) {
            dev_name = argv[++i];
        } else if (a == "-c" && i + 1 < argc) {
            filter = argv[++i];
        } else if (a == "-t" && i + 1 < argc) {
            tokens = parse_list(argv[++i]);
        } else if (a == "-b" && i + 1 < argc) {
            blocks = parse_list(argv[++i]);
        } else if (a == "--ms" && i + 1 < argc) {
            min_ms = atof(argv[++i]);
        } else if (a == "--reps" && i + 1 < argc) {
            reps = atoi(argv[++i]);
        } else {
            fprintf(stderr, "usage: %s [-d <device>] [-c <regex of the case>] [-t <tokens,...>] [-b <blocks,...>] [--ms <ms>] [--reps <copies>]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    if (dev_name) {
        dev = ggml_backend_dev_by_name(dev_name);
        if (!dev) {
            fprintf(stderr, "no device %s\n", dev_name);
            return 1;
        }
    } else {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        }
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (!backend) {
        fprintf(stderr, "cannot init %s\n", ggml_backend_dev_name(dev));
        return 1;
    }

    const std::regex re(filter);
    printf("device %s (%s), indexer of %lld heads of %lld, keys f16\n", ggml_backend_dev_name(dev),
           ggml_backend_dev_description(dev), (long long) NH, (long long) HEAD);
    printf("%7s %7s  %-8s %11s %8s %8s %5s %10s\n", "blocks", "tokens", "case", "us", "TFLOP/s", "GB/s", "reps", "diff");

    for (int64_t blk : blocks) {
        for (int64_t tok : tokens) {
            std::vector<float> ref;
            for (const micro_case & c : MICRO_CASES) {
                if (!std::regex_search(c.name, re)) {
                    continue;
                }
                micro_result r = run_case(backend, c.kind, tok, blk, min_ms, reps);
                if (!r.ok) {
                    printf("%7lld %7lld  %-8s %s\n", (long long) blk, (long long) tok, c.name, r.note.c_str());
                    fflush(stdout);
                    continue;
                }
                const double flops = 2.0*NH*HEAD*(double) blk*(double) tok;
                const double bytes = 2.0*HEAD*blk + 4.0*HEAD*NH*tok + (c.kind == micro_kind::lid ? 2.0 : 4.0)*blk*tok + 4.0*blk*tok;
                const double diff  = ref.empty() ? 0.0 : rel_diff(ref, r.out);
                if (ref.empty()) {
                    ref = std::move(r.out);
                }
                printf("%7lld %7lld  %-8s %11.2f %8.3f %8.1f %5d %10.2e\n", (long long) blk, (long long) tok, c.name, r.us,
                       flops / (r.us*1e-6) / 1e12, bytes / (r.us*1e-6) / 1e9, r.reps, diff);
                fflush(stdout);
            }
        }
    }

    ggml_backend_free(backend);
    return 0;
}
