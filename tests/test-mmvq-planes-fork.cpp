// the q8_1 activations of the Vulkan int8 mat-vec in one or two planes (GGML_VK_MMVQ_PLANES, see
// ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vecq_planes_fork.glsl): MUL_MAT of 1-8 columns (mul_mat_vecq.comp) and
// MUL_MAT_ID of 1-8 tokens (mul_mat_vec_id_fork.comp for q4_K and q5_1, mul_mat_vecq.comp with MUL_MAT_ID for the
// others) on a Vulkan device against the same weights and activations in double. The NMSE of the device must match
// that of the activations rounded on the CPU as the planes hold them (one plane: an NMSE around 1e-4 on these
// activations, two planes about 2^-14 of it); a float mat-vec would be near 1e-14, and the scale of plane 2 taken from
// d before its rounding to f16 ten times the NMSE of two planes. The test sets GGML_VK_MMVQ_PLANES, GGML_VK_FORCE_MMVQ
// (the int8 mat-vec at every shape and type) and, unless set, GGML_VK_INT_DOT_EMULATED=1 (MoltenVK has the int8 dot
// without acceleration) and GGML_VK_MMV_ID_FORK=2 before the backends load.
//
// usage: test-mmvq-planes-fork [--planes 1|2] [--types <list>] [-dev <Vulkan device>] [--dump <file>]
//   --types: GGML_VK_MMVQ_PLANES_TYPES, the weight types of two planes (comma-separated names); the others then must
//            show one plane
//   --dump: the outputs of the device, all cases in a row as floats (two builds compared byte for byte)
//   --verify <tile>: GGML_VK_MM_F16REG_VERIFY, the dense q8_0 of 2-8 columns (1024 rows and more) on the f16 coopmat tile
//            of mul_mm_f16reg.comp: those cases must run there (its log line: the pipeline that reads the f32
//            activations in place, _b32_vfy, or with GGML_VK_MM_F16REG_VERIFY_B16=1 the one that reads their f16
//            copy, _b16_vfy) and match weights and activations rounded to f16 (the shader decodes q8_0 to f16(d * q),
//            the activations go through the saturating f16 rounding either way); on a device without coopmat with
//            GGML_VK_MM_F16REG=<tile> GGML_VK_MM_F16REG_EMULATE=1; TEST_MMVQ_VERIFY=<tile> does the same
// TEST_MMVQ_PLANES_REQUIRE=1: a skip (no Vulkan device, no int8 mat-vec) fails (01-check on the 395)

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <mutex>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static int    n_fail = 0;
static FILE * g_dump = nullptr;

// the log of the backends (--verify: the line of each dispatch of the verify tile)
static std::mutex  g_log_mutex;
static std::string g_log;

static void log_cb(ggml_log_level level, const char * text, void * /*user_data*/) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_log += text;
    }
    if (level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        fputs(text, stderr);
    }
}

// the planes argument of the f16 coopmat tile of the draft checks: weights and activations rounded to f16
static constexpr int PLANES_F16 = 16;

static float f16_round(float v) {
    return ggml_fp16_to_fp32(ggml_fp32_to_fp16(std::min(65504.0f, std::max(-65504.0f, v))));
}

static void set_env(const char * name, const char * value) {
#ifdef _WIN32
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}

// activations with the spread of a model's: normal values, a few rows of blocks scaled up, rare large ones
static void fill_x(std::vector<float> & x, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> u(0.0f, 1.0f);
    for (size_t i = 0; i < x.size(); ++i) {
        float v = nd(rng);
        if ((i / 32) % 7 == 3) {
            v *= 8.0f;
        }
        if (u(rng) < 0.002f) {
            v *= 30.0f;
        }
        x[i] = v;
    }
}

// the activations as the planes hold them (quantize_q8_1.comp): one plane d * q1 (d in f16), two planes
// d * q1 + d / 128 * q2
static std::vector<float> planes_of(const std::vector<float> & x, int planes) {
    std::vector<float> r(x.size());
    if (planes == PLANES_F16) {
        for (size_t i = 0; i < x.size(); ++i) {
            r[i] = f16_round(x[i]);
        }
        return r;
    }
    for (size_t b = 0; b < x.size(); b += 32) {
        float amax = 0.0f;
        for (size_t i = b; i < b + 32; ++i) {
            amax = std::max(amax, std::fabs(x[i]));
        }
        const float d     = amax / 127.0f;
        const float d_inv = d != 0.0f ? 1.0f / d : 0.0f;
        const float dh    = ggml_fp16_to_fp32(ggml_fp32_to_fp16(d));
        const float r_inv = dh != 0.0f ? 128.0f / dh : 0.0f;
        for (size_t i = b; i < b + 32; ++i) {
            const float q1 = std::round(x[i] * d_inv);
            float v = q1 * dh;
            if (planes == 2) {
                const float q2 = std::min(127.0f, std::max(-127.0f, std::round((x[i] - q1 * dh) * r_inv)));
                v += q2 * (dh / 128.0f);
            }
            r[i] = v;
        }
    }
    return r;
}

struct result {
    bool   ran  = false;
    double nmse = 0.0;   // the device against double math
    double sim  = 0.0;   // the CPU rounding of the activations as the planes hold them against double math
};

// y = W x on the device for a MUL_MAT (n_exp == 0) or a MUL_MAT_ID of n_exp experts, n_used a token; the reference
// from the dequantized weights in double
static result run(ggml_backend_t backend, int planes, ggml_type type, int64_t k, int64_t m, int64_t n, int64_t n_exp,
                  int64_t n_used, std::mt19937 & rng) {
    result r;
    const bool id = n_exp > 0;
    const int64_t ne = id ? n_exp : 1;

    ggml_init_params ip = { 16*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * w   = ggml_new_tensor_3d(ctx, type, k, m, ne);
    ggml_tensor * x   = id ? ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, 1, n)
                           : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
    ggml_tensor * ids = id ? ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_used, n) : nullptr;
    ggml_tensor * y   = id ? ggml_mul_mat_id(ctx, w, x, ids) : ggml_mul_mat(ctx, w, x);
    ggml_cgraph * gf  = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    if (!ggml_backend_supports_op(backend, y)) {
        ggml_free(ctx);
        return r;
    }
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // weights: uniform values quantized, and their dequantized copy
    std::vector<float> wf(k*m*ne);
    {
        std::uniform_real_distribution<float> u(-1.0f, 1.0f);
        for (auto & v : wf) {
            v = u(rng);
        }
        std::vector<uint8_t> wq(ggml_nbytes(w));
        ggml_quantize_chunk(type, wf.data(), wq.data(), 0, m*ne, k, nullptr);
        ggml_get_type_traits(type)->to_float(wq.data(), wf.data(), k*m*ne);
        ggml_backend_tensor_set(w, wq.data(), 0, wq.size());
    }
    // the f16 tile computes with f16(d * q)
    std::vector<float> wp;
    if (planes == PLANES_F16) {
        wp.resize(wf.size());
        for (size_t i = 0; i < wf.size(); ++i) {
            wp[i] = f16_round(wf[i]);
        }
    }
    std::vector<float> xf(k*n);
    fill_x(xf, rng);
    const std::vector<float> xp = planes_of(xf, planes);
    ggml_backend_tensor_set(x, xf.data(), 0, xf.size()*sizeof(float));
    std::vector<int32_t> idv;
    if (id) {
        // distinct experts a token, tokens sharing some
        idv.resize(n_used*n);
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t s = 0; s < n_used; ++s) {
                idv[t*n_used + s] = (int32_t) ((t*3 + s*5) % n_exp);
            }
        }
        ggml_backend_tensor_set(ids, idv.data(), 0, idv.size()*sizeof(int32_t));
    }

    if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "compute failed\n");
        n_fail++;
    } else {
        std::vector<float> yf(ggml_nelements(y));
        ggml_backend_tensor_get(y, yf.data(), 0, yf.size()*sizeof(float));
        if (g_dump != nullptr) {
            fwrite(yf.data(), sizeof(float), yf.size(), g_dump);
        }
        double err = 0.0;
        double ref = 0.0;
        double sim = 0.0;
        const int64_t n_slot = id ? n_used : 1;
        for (int64_t t = 0; t < n; ++t) {
            for (int64_t s = 0; s < n_slot; ++s) {
                const int64_t e = id ? idv[t*n_used + s] : 0;
                for (int64_t i = 0; i < m; ++i) {
                    const float * wr = wf.data() + (e*m + i)*k;
                    const float * wq = wp.empty() ? wr : wp.data() + (e*m + i)*k;
                    const float * xr = xf.data() + t*k;
                    const float * pr = xp.data() + t*k;
                    double acc = 0.0;
                    double acp = 0.0;
                    for (int64_t j = 0; j < k; ++j) {
                        acc += (double) wr[j] * (double) xr[j];
                        acp += (double) wq[j] * (double) pr[j];
                    }
                    const double v = yf[(t*n_slot + s)*m + i];
                    err += (v - acc)*(v - acc);
                    sim += (acp - acc)*(acp - acc);
                    ref += acc*acc;
                }
            }
        }
        r.ran  = true;
        r.nmse = ref > 0.0 ? err/ref : err;
        r.sim  = ref > 0.0 ? sim/ref : sim;
    }

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

int main(int argc, char ** argv) {
    int          planes   = 2;
    const char * dev_name = nullptr;
    std::string  types;
    const char * verify   = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--planes") == 0 && i + 1 < argc) {
            planes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--types") == 0 && i + 1 < argc) {
            types = argv[++i];
        } else if (strcmp(argv[i], "--verify") == 0 && i + 1 < argc) {
            verify = argv[++i];
        } else if ((strcmp(argv[i], "-dev") == 0 || strcmp(argv[i], "--device") == 0) && i + 1 < argc) {
            dev_name = argv[++i];
        } else if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
            g_dump = fopen(argv[++i], "wb");
            if (g_dump == nullptr) {
                fprintf(stderr, "%s: cannot write %s\n", argv[0], argv[i]);
                return 1;
            }
        } else {
            fprintf(stderr, "usage: %s [--planes 1|2] [--types <list>] [--verify <tile>] [-dev <Vulkan device>] [--dump <file>]\n", argv[0]);
            return 1;
        }
    }
    // TEST_MMVQ_VERIFY=<tile> as --verify (01-check runs the unit without arguments)
    if (verify == nullptr) {
        const char * e = getenv("TEST_MMVQ_VERIFY");
        if (e != nullptr && *e != 0 && strcmp(e, "0") != 0) {
            verify = e;
        }
    }
    if (planes != 1 && planes != 2) {
        fprintf(stderr, "%s: --planes takes 1 or 2\n", argv[0]);
        return 1;
    }

    set_env("GGML_VK_MMVQ_PLANES", planes == 2 ? "2" : "1");
    set_env("GGML_VK_MMVQ_PLANES_TYPES", types.c_str());
    // the planes a weight type gets: two with --planes 2 for the types of --types (all without it)
    auto planes_for = [&](ggml_type type) {
        if (planes != 2 || types.empty()) {
            return planes;
        }
        std::string name = ggml_type_name(type);
        for (auto & c : name) {
            c = (char) tolower((unsigned char) c);
        }
        std::string list = "," + types + ",";
        for (auto & c : list) {
            c = (char) tolower((unsigned char) c);
        }
        return list.find("," + name + ",") != std::string::npos ? 2 : 1;
    };
    set_env("GGML_VK_FORCE_MMVQ", "1");
    if (getenv("GGML_VK_INT_DOT_EMULATED") == nullptr) {
        set_env("GGML_VK_INT_DOT_EMULATED", "1");
    }
    if (getenv("GGML_VK_MMV_ID_FORK") == nullptr) {
        set_env("GGML_VK_MMV_ID_FORK", "2");
    }
    if (verify != nullptr) {
        set_env("GGML_VK_MM_F16REG_VERIFY", verify);
        set_env("GGML_VK_MM_F16REG_LOG", "1");
    } else {
        // the tile of the draft checks is the default on AMD RDNA3 since 13.11 night: off here, so that every MUL_MAT
        // case runs on the int8 mat-vec (the cases have 64 rows, under its 1024, but the test should not rely on that)
        set_env("GGML_VK_MM_F16REG_VERIFY", "0");
    }
    ggml_log_set(log_cb, nullptr);

    ggml_backend_load_all();

    ggml_backend_dev_t dev = nullptr;
    if (dev_name != nullptr) {
        dev = ggml_backend_dev_by_name(dev_name);
    } else {
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (strncmp(ggml_backend_dev_name(d), "Vulkan", 6) == 0) {
                dev = d;
                break;
            }
        }
    }
    const char * req     = getenv("TEST_MMVQ_PLANES_REQUIRE");
    const int    skip_rc = req != nullptr && atoi(req) != 0 ? 1 : 0;
    if (dev == nullptr || strncmp(ggml_backend_dev_name(dev), "Vulkan", 6) != 0) {
        printf("%s: SKIPPED, no Vulkan device%s%s\n", argv[0], dev_name ? " " : "", dev_name ? dev_name : "");
        return skip_rc;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
    if (backend == nullptr) {
        fprintf(stderr, "%s: cannot init %s\n", argv[0], ggml_backend_dev_name(dev));
        return 1;
    }

    // the NMSE of the device over that of the CPU rounding: the rounding ties and the float sums of the device apart,
    // the same activations. One plane: upstream's sum of a block (of the codes times d before its rounding to f16) adds
    // a little for the weights with a minimum
    const double lo = 0.9;
    auto         hi = [](int p) { return p == 2 ? 1.1 : 1.2; };

    // the int8 dot of MoltenVK on the AMD GPU of a Mac gets the q8_0 mat-vec wrong (upstream mul_mat_vecq.comp with one
    // plane fails test-backend-ops there with GGML_VK_INT_DOT_EMULATED=1 GGML_VK_FORCE_MMVQ=1; the Intel GPU passes)
#ifdef __APPLE__
    const bool q8_0_broken = strstr(ggml_backend_dev_description(dev), "AMD") != nullptr;
#else
    const bool q8_0_broken = false;
#endif

    std::mt19937 rng(1234);
    int n_ran = 0;
    double worst = 0.0;

    auto check = [&](const char * what, ggml_type type, int64_t k, int64_t m, int64_t n, int64_t n_exp,
                     int64_t n_used) {
        const int    p = planes_for(type);
        const result r = run(backend, p, type, k, m, n, n_exp, n_used, rng);
        if (!r.ran) {
            printf("  %-10s %-6s k %5lld n %lld: not supported\n", what, ggml_type_name(type), (long long) k,
                   (long long) n);
            return;
        }
        const double ratio = r.sim > 0.0 ? r.nmse / r.sim : 0.0;
        const bool   known = q8_0_broken && type == GGML_TYPE_Q8_0;
        const bool   ok    = ratio > lo && ratio < hi(p);
        printf("  %-10s %-6s k %5lld n %lld, %d plane%s: NMSE %.3e, of the rounding %.3e, ratio %.3f%s\n", what,
               ggml_type_name(type), (long long) k, (long long) n, p, p == 2 ? "s" : " ", r.nmse, r.sim, ratio,
               ok ? "" : known ? "  (the int8 dot of MoltenVK on AMD, not counted)" : "  <- outside the range");
        if (known) {
            return;
        }
        n_ran++;
        worst = std::max(worst, r.nmse);
        if (!ok) {
            n_fail++;
        }
    };

    printf("%s: %s (%s), %d plane%s%s%s, NMSE of the device over that of the rounding in (%.2f, %.2f) for two planes, "
           "(%.2f, %.2f) for one\n", argv[0], ggml_backend_dev_name(dev), ggml_backend_dev_description(dev), planes,
           planes == 2 ? "s" : "", types.empty() ? "" : " for ", types.c_str(), lo, hi(2), lo, hi(1));
    for (ggml_type type : { GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
                            GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K }) {
        for (int64_t n : { 1, 2, 3, 5, 8 }) {
            check("MUL_MAT", type, 4096, 64, n, 0, 0);
        }
    }
    for (ggml_type type : { GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0, GGML_TYPE_Q5_K }) {
        const int64_t k = type == GGML_TYPE_Q5_1 ? 640 : 2560;
        for (int64_t n : { 1, 2, 4, 8 }) {
            check("MUL_MAT_ID", type, k, 96, n, 16, 4);
        }
    }

    if (verify != nullptr) {
        // the dense q8_0 of the draft checks on the f16 coopmat tile: each case must log its dispatch on the pipeline of
        // its B path
        const char * b16_env  = getenv("GGML_VK_MM_F16REG_VERIFY_B16");
        const char * want_vfy = b16_env != nullptr && atoi(b16_env) != 0 ? "_b16_vfy" : "_b32_vfy";
        printf("%s: the draft checks on the tile %s (%s), weights and activations in f16: NMSE of the device over that of "
               "the rounding in (%.2f, %.2f)\n", argv[0], verify, want_vfy + 1, lo, hi(2));
        auto check_verify = [&](int64_t k, int64_t m, int64_t n) {
            {
                std::lock_guard<std::mutex> lock(g_log_mutex);
                g_log.clear();
            }
            const result r = run(backend, PLANES_F16, GGML_TYPE_Q8_0, k, m, n, 0, 0, rng);
            bool used = false;
            {
                std::lock_guard<std::mutex> lock(g_log_mutex);
                used = g_log.find(want_vfy) != std::string::npos;
            }
            if (!r.ran) {
                printf("  MUL_MAT    q8_0   m %5lld k %5lld n %lld: not supported\n", (long long) m, (long long) k, (long long) n);
                n_fail++;
                return;
            }
            const double ratio = r.sim > 0.0 ? r.nmse / r.sim : 0.0;
            const bool   ok    = used && ratio > lo && ratio < hi(2);
            printf("  MUL_MAT    q8_0   m %5lld k %5lld n %lld, f16 tile: NMSE %.3e, of the rounding %.3e, ratio %.3f%s\n",
                   (long long) m, (long long) k, (long long) n, r.nmse, r.sim, ratio,
                   !used ? "  <- not on the verify pipeline" : ok ? "" : "  <- outside the range");
            n_ran++;
            worst = std::max(worst, r.nmse);
            if (!ok) {
                n_fail++;
            }
        };
        for (int64_t n : { 2, 3, 5, 8 }) {
            check_verify(2560, 1024, n);
        }
        check_verify(6144, 2560, 8);
        check_verify(320, 4096, 7);
        check_verify(640, 2560, 3);
        check_verify(64, 1040, 4);
    }

    ggml_backend_free(backend);
    if (g_dump != nullptr) {
        fclose(g_dump);
    }

    if (n_ran > 0 && worst < 1e-13) {
        // a float mat-vec everywhere: no int8 dot on this device
        printf("%s: SKIPPED, the device has no int8 mat-vec (NMSE up to %.1e)\n", argv[0], worst);
        return skip_rc;
    }
    printf("%s: %d cases, NMSE up to %.2e, %s\n", argv[0], n_ran, worst, n_fail == 0 ? "OK" : "FAILED");
    return n_fail == 0 && n_ran > 0 ? 0 : 1;
}
