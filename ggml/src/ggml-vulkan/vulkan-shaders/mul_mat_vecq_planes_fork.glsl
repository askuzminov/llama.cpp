// fork: the q8_1 activations of the int8 mat-vec in two planes (GGML_VK_MMVQ_PLANES=2, quantize_q8_1.comp with
// PLANES 2). A group of 128 values holds two block_q8_1_x4 in a row: plane 1 as one plane holds it (the codes q1 of
// scale d, d in f16), plane 2 the rest x - d * q1 in codes q2 of scale d / 128. So a value is d / 128 * (128 * q1 + q2)
// and the integer dot of a weight word with both planes is 128 * dot(w, q1) + dot(w, q2): the functions of
// mul_mat_vecq_funcs.glsl stay as they are, they get that dot from dotPacked4x8EXT (the macro below), d / 128 as the
// scale and the sum of the 32 values the planes hold from the two (d, sum) pairs (plane 1: the sum in f16, plane 2:
// the rest of it). The error of a value drops from d / 2 to d / 256 (about 2^-15 of the block's largest value).
// Included by mul_mat_vecq.comp after cache_b_qs and cache_b_ds, before mul_mat_vecq_funcs.glsl; mul_mat_vecq.comp
// undefines the two macros after it.

layout (constant_id = 3) const uint MMVQ_PLANES = 1;

// the words of the plane 1 codes (x) and of the plane 2 codes (y) that the functions read as cache_b_qs
i32vec2 cache_b_qs_p[K_PER_ITER / 4];

int32_t mmvq_dot_planes(const int32_t a, const i32vec2 b) {
    const int32_t s = dotPacked4x8EXT(a, b.x);
    return MMVQ_PLANES == 2 ? s * 128 + dotPacked4x8EXT(a, b.y) : s;
}

// after the loads of plane 1 (cache_b_qs, cache_b_ds) from group outer: the codes of plane 2 at the same indices from
// group outer + 1, and the scale and the sum of both
void mmvq_planes_load(const uint outer, const uint inner, const uint b_qs_idx) {
    [[unroll]] for (uint k = 0; k < K_PER_ITER / 4; ++k) {
        cache_b_qs_p[k] = i32vec2(cache_b_qs[k], 0);
    }
    if (MMVQ_PLANES != 2) {
        return;
    }
    const uint o2 = outer + 1;
#if QUANT_R == 2
    cache_b_qs_p[0].y = data_b[o2].qs[inner * 8 + b_qs_idx];
    cache_b_qs_p[1].y = data_b[o2].qs[inner * 8 + b_qs_idx + 4];
#elif K_PER_ITER == 8
    cache_b_qs_p[0].y = data_b[o2].qs[inner * 8 + b_qs_idx * 2];
    cache_b_qs_p[1].y = data_b[o2].qs[inner * 8 + b_qs_idx * 2 + 1];
#elif K_PER_ITER == 16
    [[unroll]] for (uint k = 0; k < 4; ++k) {
        cache_b_qs_p[k].y = data_b[o2].qs[inner * 8 + b_qs_idx * 4 + k];
    }
#elif K_PER_ITER == 32
    [[unroll]] for (uint k = 0; k < 8; ++k) {
        cache_b_qs_p[k].y = data_b[o2].qs[inner * 8 + k];
    }
#else
#error unimplemented
#endif
    cache_b_ds = vec2(cache_b_ds.x * (1.0 / 128.0), cache_b_ds.y + float(data_b[o2].ds[inner].y));
}

#define cache_b_qs cache_b_qs_p
#define dotPacked4x8EXT(a, b) mmvq_dot_planes(a, b)
