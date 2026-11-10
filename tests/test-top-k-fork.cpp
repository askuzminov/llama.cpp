// the k + 1 largest of the sampler's fast top-k (common/fork/top-k-fork.h): the blocks it passes over with vector
// compares must leave the heap as one value at a time leaves it, values, ids and order, with logit biases, ties, -inf
// and NaN among the values, at the sizes around a block of 64 and at the vocabulary of qwen4exp (248320)

#include "fork/top-k-fork.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using heap_t = std::vector<std::pair<float, int32_t>>;

// the loop common/sampling.cpp had before the blocks
static void one_at_a_time(const float * logits, const float * bias, int n, int k, heap_t & h) {
    const auto greater = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
        return a.first > b.first;
    };
    h.clear();
    for (int i = 0; i < n; ++i) {
        const float v = bias ? logits[i] + bias[i] : logits[i];
        if ((int) h.size() <= k) {
            h.emplace_back(v, (int32_t) i);
            std::push_heap(h.begin(), h.end(), greater);
        } else if (v > h.front().first) {
            std::pop_heap(h.begin(), h.end(), greater);
            h.back() = { v, (int32_t) i };
            std::push_heap(h.begin(), h.end(), greater);
        }
    }
    std::sort(h.begin(), h.end(), greater);
}

static bool same(const heap_t & a, const heap_t & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (std::memcmp(&a[i].first, &b[i].first, sizeof(float)) != 0 || a[i].second != b[i].second) {
            return false;
        }
    }
    return true;
}

int main() {
    std::mt19937                    rng(42);
    std::normal_distribution<float> normal(0.0f, 2.5f);
    const float                     inf = std::numeric_limits<float>::infinity();
    const float                     nan = std::numeric_limits<float>::quiet_NaN();

    enum kind { NORMAL, PEAKS, ASCENDING, DESCENDING, TIES, NEG_INF, WITH_NAN, COUNT };
    const char * names[COUNT] = { "normal", "peaks", "ascending", "descending", "ties", "-inf", "nan" };

    const int sizes[] = { 42, 63, 64, 65, 105, 106, 127, 128, 129, 1000, 4097, 248320 };
    const int ks[]    = { 1, 2, 40, 64, 256 };

    int n_cases = 0;
    int n_fail  = 0;
    heap_t a;
    heap_t b;
    for (int n : sizes) {
        for (int k : ks) {
            if (n <= k + 1) {
                continue;
            }
            for (int kd = 0; kd < COUNT; ++kd) {
                for (int with_bias = 0; with_bias < 2; ++with_bias) {
                    std::vector<float> logits(n);
                    for (int i = 0; i < n; ++i) {
                        switch (kd) {
                            case NORMAL:     logits[i] = normal(rng); break;
                            case PEAKS:      logits[i] = rng() % 997 == 0 ? 12.0f + normal(rng) : normal(rng); break;
                            case ASCENDING:  logits[i] = (float) i * 0.001f; break;
                            case DESCENDING: logits[i] = (float) (n - i) * 0.001f; break;
                            case TIES:       logits[i] = std::round(normal(rng)); break;
                            case NEG_INF:    logits[i] = (i / 64) % 3 == 0 ? -inf : normal(rng); break;
                            // after the first k + 1 only: a NaN in the heap leaves std::sort without a strict
                            // weak order, and a later NaN never goes in
                            case WITH_NAN:   logits[i] = i > k && rng() % 101 == 0 ? nan : normal(rng); break;
                        }
                    }
                    std::vector<float> bias;
                    if (with_bias) {
                        bias.assign(n, 0.0f);
                        for (int i = 0; i < n; ++i) {
                            const unsigned r = rng() % 211;
                            bias[i] = r == 0 ? -inf : r == 1 ? 5.0f : r == 2 ? -3.5f : 0.0f;
                        }
                    }
                    const float * pb = with_bias ? bias.data() : nullptr;
                    one_at_a_time(logits.data(), pb, n, k, a);
                    common_fork_top_k_heap(logits.data(), pb, n, k, b);
                    n_cases++;
                    if (!same(a, b)) {
                        n_fail++;
                        fprintf(stderr, "differs: n %d, k %d, %s, bias %d\n", n, k, names[kd], with_bias);
                    }
                }
            }
        }
    }

    // the vector compare against the scalar one: values at, just above and just below the threshold
    {
        float x[64];
        float zero[64] = {};
        for (int pos = 0; pos < 64; ++pos) {
            for (int j = 0; j < 64; ++j) {
                x[j] = 1.0f;
            }
            x[pos] = std::nextafter(1.0f, 2.0f);
            n_cases++;
            if (!common_fork_top_k_any_above(x, nullptr, 64, 1.0f) || !common_fork_top_k_any_above(x, zero, 64, 1.0f)) {
                n_fail++;
                fprintf(stderr, "any_above misses the value above at %d\n", pos);
            }
            x[pos] = 1.0f;
            n_cases++;
            if (common_fork_top_k_any_above(x, nullptr, 64, 1.0f) || common_fork_top_k_any_above(x, zero, 64, 1.0f)) {
                n_fail++;
                fprintf(stderr, "any_above takes a value equal to the threshold at %d\n", pos);
            }
            x[pos] = nan;
            n_cases++;
            if (common_fork_top_k_any_above(x, nullptr, 64, 1.0f) || common_fork_top_k_any_above(x, nullptr, 64, nan)) {
                n_fail++;
                fprintf(stderr, "any_above takes a NaN at %d\n", pos);
            }
        }
    }

    printf("test-top-k-fork: %d cases, %d failed\n", n_cases, n_fail);
    return n_fail == 0 ? 0 : 1;
}
