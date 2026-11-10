#pragma once

// the k + 1 largest of n values for the fast top-k of the sampler (LLAMA_SAMPLER_FAST_TOP_K, common/sampling.cpp),
// header-only so that test-top-k-fork checks it against one value at a time

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#if !defined(COMMON_FORK_TOP_K_NO_SIMD) && (defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64))
#    include <emmintrin.h>
#    define COMMON_FORK_TOP_K_SSE2
#elif !defined(COMMON_FORK_TOP_K_NO_SIMD) && defined(__ARM_NEON) && defined(__aarch64__)
#    include <arm_neon.h>
#    define COMMON_FORK_TOP_K_NEON
#endif

// whether one of x[0..n) (+ b[0..n) when b is not null) is above thr; n a multiple of 4. a NaN is not above, as with >
inline bool common_fork_top_k_any_above(const float * x, const float * b, int n, float thr) {
#if defined(COMMON_FORK_TOP_K_SSE2)
    const __m128 t   = _mm_set1_ps(thr);
    __m128       acc = _mm_setzero_ps();
    if (b) {
        for (int j = 0; j < n; j += 4) {
            acc = _mm_or_ps(acc, _mm_cmpgt_ps(_mm_add_ps(_mm_loadu_ps(x + j), _mm_loadu_ps(b + j)), t));
        }
    } else {
        for (int j = 0; j < n; j += 4) {
            acc = _mm_or_ps(acc, _mm_cmpgt_ps(_mm_loadu_ps(x + j), t));
        }
    }
    return _mm_movemask_ps(acc) != 0;
#elif defined(COMMON_FORK_TOP_K_NEON)
    const float32x4_t t   = vdupq_n_f32(thr);
    uint32x4_t        acc = vdupq_n_u32(0);
    if (b) {
        for (int j = 0; j < n; j += 4) {
            acc = vorrq_u32(acc, vcgtq_f32(vaddq_f32(vld1q_f32(x + j), vld1q_f32(b + j)), t));
        }
    } else {
        for (int j = 0; j < n; j += 4) {
            acc = vorrq_u32(acc, vcgtq_f32(vld1q_f32(x + j), t));
        }
    }
    return vmaxvq_u32(acc) != 0;
#else
    int any = 0;
    if (b) {
        for (int j = 0; j < n; ++j) {
            any |= (x[j] + b[j]) > thr;
        }
    } else {
        for (int j = 0; j < n; ++j) {
            any |= x[j] > thr;
        }
    }
    return any != 0;
#endif
}

// h: the k + 1 largest of logits[i] (+ bias[i] when bias is not null), largest first, as a min-heap of one value at a
// time leaves them: the first k + 1 go in, a later value replaces the minimum when it is above it (09.11: one value at
// a time took 0.23 ms for 248320 logits on an i9-9980HK, ~0.2 ms a verified position on the 395 by the timeline). a
// block of 64 values none of which is above the minimum is passed over with vector compares (0.05 ms there): the
// minimum only grows, so none of them would have gone in, and the heap takes the same values in the same order
inline void common_fork_top_k_heap(const float * logits, const float * bias, int n, int k,
                                   std::vector<std::pair<float, int32_t>> & h) {
    const auto greater = [](const std::pair<float, int32_t> & a, const std::pair<float, int32_t> & b) {
        return a.first > b.first;
    };

    h.clear();

    int i = 0;
    for (; i < n && (int) h.size() <= k; ++i) {
        h.emplace_back(bias ? logits[i] + bias[i] : logits[i], (int32_t) i);
        std::push_heap(h.begin(), h.end(), greater);
    }

    const auto offer = [&](int j) {
        const float v = bias ? logits[j] + bias[j] : logits[j];
        if (v > h.front().first) {
            std::pop_heap(h.begin(), h.end(), greater);
            h.back() = { v, (int32_t) j };
            std::push_heap(h.begin(), h.end(), greater);
        }
    };

    constexpr int B = 64;
    for (; i + B <= n; i += B) {
        if (!common_fork_top_k_any_above(logits + i, bias ? bias + i : nullptr, B, h.front().first)) {
            continue;
        }
        for (int j = i; j < i + B; ++j) {
            offer(j);
        }
    }
    for (; i < n; ++i) {
        offer(i);
    }

    std::sort(h.begin(), h.end(), greater);
}
