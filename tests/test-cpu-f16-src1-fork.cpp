// the F16 activations that the fork's CPU backend takes against quantized weights (ggml-cpu.cpp supports_op, packed
// through F32 into the vec_dot_type in ggml_compute_forward_mul_mat and _mul_mat_id): MUL_MAT and MUL_MAT_ID with an
// F16 src1 against the same values in F32, on the CPU backend as a model runs it (no use_ref, so the tiled K-quant
// kernels of x86 builds take the batches of 8 rows and more). the tiled MUL_MAT driver converts F32 only: before
// 10.11 an F16 src1 of 8 rows and more aborted there (tiled.cpp GGML_ASSERT) on AVX builds
//
// usage: test-cpu-f16-src1-fork

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

static int n_fail = 0;

// c = a x b, with b in f16 and the same values in f32; the largest difference against the largest value
static double compare(const ggml_tensor * c16, const ggml_tensor * c32) {
    const float * x = (const float *) c16->data;
    const float * y = (const float *) c32->data;
    double diff = 0.0;
    double top  = 0.0;
    for (int64_t i = 0; i < ggml_nelements(c16); ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i])) {
            return INFINITY;
        }
        diff = std::max(diff, (double) std::fabs(x[i] - y[i]));
        top  = std::max(top,  (double) std::fabs(y[i]));
    }
    return top > 0.0 ? diff/top : diff;
}

static void fill(ggml_tensor * t32, ggml_tensor * t16, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    for (int64_t i = 0; i < ggml_nelements(t32); ++i) {
        const ggml_fp16_t h = ggml_fp32_to_fp16(u(rng));
        ((ggml_fp16_t *) t16->data)[i] = h;
        ((float *) t32->data)[i] = ggml_fp16_to_fp32(h);
    }
}

static void quantize(ggml_tensor * q, int64_t k, int64_t rows, std::mt19937 & rng) {
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::vector<float> w(k*rows);
    for (auto & v : w) {
        v = u(rng);
    }
    ggml_quantize_chunk(q->type, w.data(), q->data, 0, rows, k, nullptr);
}

int main() {
    ggml_backend_load_all();
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (cpu == nullptr) {
        fprintf(stderr, "no CPU backend\n");
        return 1;
    }

    const ggml_type types[] = {
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q3_K, GGML_TYPE_Q2_K, GGML_TYPE_IQ4_XS,
        GGML_TYPE_Q8_0, GGML_TYPE_Q4_0,
    };
    const int64_t k    = 512;
    const int64_t m    = 32;
    const int64_t n_as = 4;
    const int64_t used = 2;

    std::mt19937 rng(42);
    for (ggml_type type : types) {
        for (int64_t n : { 1, 4, 16, 64 }) {
            ggml_init_params ip = { 32*1024*1024, nullptr, false };
            ggml_context * ctx = ggml_init(ip);

            // MUL_MAT
            ggml_tensor * a   = ggml_new_tensor_2d(ctx, type, k, m);
            ggml_tensor * b32 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, k, n);
            ggml_tensor * b16 = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, k, n);
            quantize(a, k, m, rng);
            fill(b32, b16, rng);
            ggml_tensor * c32 = ggml_mul_mat(ctx, a, b32);
            ggml_tensor * c16 = ggml_mul_mat(ctx, a, b16);

            // MUL_MAT_ID: n tokens, each to `used` of n_as experts, so that an expert gets about n/2 rows
            ggml_tensor * as   = ggml_new_tensor_3d(ctx, type, k, m, n_as);
            ggml_tensor * e32  = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, used, n);
            ggml_tensor * e16  = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, k, used, n);
            ggml_tensor * ids  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, used, n);
            quantize(as, k, m*n_as, rng);
            fill(e32, e16, rng);
            for (int64_t t = 0; t < n; ++t) {
                const int32_t first = (int32_t) (rng() % n_as);
                ((int32_t *) ids->data)[t*used + 0] = first;
                ((int32_t *) ids->data)[t*used + 1] = (first + 1 + (int32_t) (rng() % (n_as - 1))) % n_as;
            }
            ggml_tensor * d32 = ggml_mul_mat_id(ctx, as, e32, ids);
            ggml_tensor * d16 = ggml_mul_mat_id(ctx, as, e16, ids);

            const bool mm_ok   = ggml_backend_supports_op(cpu, c16);
            const bool mmid_ok = ggml_backend_supports_op(cpu, d16);

            ggml_cgraph * gf = ggml_new_graph(ctx);
            ggml_build_forward_expand(gf, c32);
            ggml_build_forward_expand(gf, d32);
            if (mm_ok) {
                ggml_build_forward_expand(gf, c16);
            }
            if (mmid_ok) {
                ggml_build_forward_expand(gf, d16);
            }
            if (ggml_backend_graph_compute(cpu, gf) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "%s n=%lld: compute failed\n", ggml_type_name(type), (long long) n);
                n_fail++;
            } else {
                // the same q8 activations; the f32 batch may go to the tiled kernel and the f16 one not, so the sums
                // differ in their order only
                const double tol = 1e-5;
                if (mm_ok) {
                    const double r = compare(c16, c32);
                    if (!(r <= tol)) {
                        fprintf(stderr, "MUL_MAT %s n=%lld: f16 src1 %.3g apart from f32\n", ggml_type_name(type), (long long) n, r);
                        n_fail++;
                    }
                }
                if (mmid_ok) {
                    const double r = compare(d16, d32);
                    if (!(r <= tol)) {
                        fprintf(stderr, "MUL_MAT_ID %s n=%lld: f16 src1 %.3g apart from f32\n", ggml_type_name(type), (long long) n, r);
                        n_fail++;
                    }
                }
            }
            ggml_free(ctx);
        }
    }

    ggml_backend_free(cpu);
    printf("test-cpu-f16-src1-fork: %s\n", n_fail == 0 ? "ok" : "FAILED");
    return n_fail == 0 ? 0 : 1;
}
