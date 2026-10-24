// fork: GATED_DELTA_NET_LAZY, the gated delta net with deferred rollback (ggml_gated_delta_net_lazy, op_params[0] 1)
// and the replay alone (ggml_gated_delta_net_replay, 2), included by gated_delta_net.cu. Info row y of a cell: the
// state row read, the row written, the records replayed, the record set read and the set written. The state is read,
// brought up to date with the records and written back; the net then runs the tokens and writes, per token, its record
// (delta [S_v, H], key [S_v, H_k], gate [H]) instead of a snapshot. Laid out as gated_delta_net_cuda (cols_per_warp
// columns a warp, a reduction segment a column) and with its expressions, so a replay gives the state the plain net
// leaves.

template <int S_v, bool replay_only, int cols_per_warp = gdn_cols_per_warp>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_lazy_cuda(const float *   q,
                          const float *   k,
                          const float *   v,
                          const float *   g,
                          const float *   beta,
                          float *         state_all,
                          float *         rec,
                          const int32_t * info,
                          float *         dst,
                          int64_t         H,
                          int64_t         H_k,
                          int64_t         n_tokens,
                          int64_t         sq1,
                          int64_t         sq2,
                          int64_t         sq3,
                          int64_t         sk1,
                          int64_t         sk2,
                          int64_t         sk3,
                          int64_t         sv1,
                          int64_t         sv2,
                          int64_t         sv3,
                          int64_t         sb1,
                          int64_t         sb2,
                          int64_t         sb3,
                          const uint3     rq3_magic,
                          float           scale,
                          int64_t         rec_floats,
                          int64_t         rec_set) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int lanes_per_col = warp_size / cols_per_warp;
    constexpr int rows_per_lane = S_v / lanes_per_col;
    static_assert(S_v % lanes_per_col == 0, "S_v must be a multiple of lanes_per_col");

    const int lane        = threadIdx.x;
    const int col_in_warp = lane / lanes_per_col;
    const int lane_in_col = lane - col_in_warp * lanes_per_col;
    const int col = (blockIdx.z * blockDim.y + threadIdx.y) * cols_per_warp + col_in_warp;

    const int64_t kh = h_idx % H_k;

    ggml_cuda_pdl_sync();

    const int32_t * inf     = info + 5 * sequence;
    const int64_t   row_in  = inf[0];
    const int64_t   row_out = inf[1];
    const int       n_rep   = inf[2];

    // the replay alone leaves a state without records where it is
    if (replay_only && n_rep == 0 && row_in == row_out) {
        return;
    }

    const int64_t D     = H * S_v * S_v;
    const float * s_in  = state_all + row_in  * D + h_idx * S_v * S_v + col * S_v;
    float *       s_out = state_all + row_out * D + h_idx * S_v * S_v + col * S_v;

    float s_shard[rows_per_lane];
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        s_shard[r] = s_in[r * lanes_per_col + lane_in_col];
    }

    // the records of the tokens the rollback kept: the decay, then the key times the delta of the column
    const float * rd = rec + (int64_t) inf[3] * rec_set;
    for (int j = 0; j < n_rep; j++) {
        const float * rj    = rd + j * rec_floats;
        const float   d     = rj[h_idx * S_v + col];
        const float   g_val = rj[H * S_v + H_k * S_v + h_idx];
        const float * kk    = rj + H * S_v + kh * S_v;
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_shard[r] = g_val * s_shard[r] + kk[r * lanes_per_col + lane_in_col] * d;
        }
    }
    if (n_rep > 0 || row_in != row_out) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            s_out[r * lanes_per_col + lane_in_col] = s_shard[r];
        }
    }

    if constexpr (!replay_only) {
        const uint32_t iq3       = fastdiv(sequence, rq3_magic);
        float *        attn_data = dst + (sequence * n_tokens * H + h_idx) * S_v;
        float *        wr        = rec + (int64_t) inf[4] * rec_set;

        // the key and the gate of a head go to its record once, from the first column of the head
        const bool first_col = blockIdx.z == 0 && threadIdx.y == 0 && col_in_warp == 0;

        for (int t = 0; t < n_tokens; t++) {
            const float * q_t = q + iq3 * sq3 + t * sq2 + kh * sq1;
            const float * k_t = k + iq3 * sk3 + t * sk2 + kh * sk1;
            const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

            const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
            const float   beta_val  = beta[gb_offset];
            const float   g_val     = expf(g[gb_offset]);

            float k_reg[rows_per_lane];
            float q_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * lanes_per_col + lane_in_col;
                k_reg[r] = k_t[i];
                q_reg[r] = q_t[i];
            }

            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<lanes_per_col>(kv_shard);

            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<lanes_per_col>(attn_partial);

            if (lane_in_col == 0) {
                attn_data[col] = attn_col * scale;
            }
            attn_data += S_v * H;

            float * rt = wr + t * rec_floats;
            if (lane_in_col == 0) {
                rt[h_idx * S_v + col] = delta_col;
            }
            if (first_col) {
                if ((int64_t) h_idx < H_k) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        rt[H * S_v + h_idx * S_v + r * lanes_per_col + lane_in_col] = k_reg[r];
                    }
                }
                if (lane_in_col == 0) {
                    rt[H * S_v + H_k * S_v + h_idx] = g_val;
                }
            }
        }
    }
}

template <bool replay_only>
static void launch_gated_delta_net_lazy(
        const float * q_d, const float * k_d, const float * v_d, const float * g_d, const float * b_d,
        float * state_d, float * rec_d, const int32_t * info_d, float * dst_d,
        int64_t S_v, int64_t H, int64_t H_k, int64_t n_tokens, int64_t n_info,
        int64_t sq1, int64_t sq2, int64_t sq3,
        int64_t sk1, int64_t sk2, int64_t sk3,
        int64_t sv1, int64_t sv2, int64_t sv3,
        int64_t sb1, int64_t sb2, int64_t sb3,
        int64_t rq3, float scale, int64_t rec_floats, int64_t rec_set, cudaStream_t stream) {
    // the grid of launch_gated_delta_net
    const int warp_size     = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int nsm           = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
    const int cols_per_warp = gdn_cols_per_warp;
    int num_warps = 4;
    while (num_warps > 1 && H*n_info*(S_v / (cols_per_warp * num_warps)) < nsm) {
        num_warps /= 2;
    }
    dim3 grid_dims(H, n_info, (S_v + cols_per_warp * num_warps - 1) / (cols_per_warp * num_warps));
    dim3 block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 rq3_magic = init_fastdiv_values(rq3);

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);
    switch (S_v) {
        case 16:
            ggml_cuda_kernel_launch(gated_delta_net_lazy_cuda<16, replay_only>, launch_params,
                q_d, k_d, v_d, g_d, b_d, state_d, rec_d, info_d, dst_d, H, H_k, n_tokens,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3, sb1, sb2, sb3, rq3_magic, scale, rec_floats, rec_set);
            break;
        case 32:
            ggml_cuda_kernel_launch(gated_delta_net_lazy_cuda<32, replay_only>, launch_params,
                q_d, k_d, v_d, g_d, b_d, state_d, rec_d, info_d, dst_d, H, H_k, n_tokens,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3, sb1, sb2, sb3, rq3_magic, scale, rec_floats, rec_set);
            break;
        case 64:
            ggml_cuda_kernel_launch(gated_delta_net_lazy_cuda<64, replay_only>, launch_params,
                q_d, k_d, v_d, g_d, b_d, state_d, rec_d, info_d, dst_d, H, H_k, n_tokens,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3, sb1, sb2, sb3, rq3_magic, scale, rec_floats, rec_set);
            break;
        case 128:
            ggml_cuda_kernel_launch(gated_delta_net_lazy_cuda<128, replay_only>, launch_params,
                q_d, k_d, v_d, g_d, b_d, state_d, rec_d, info_d, dst_d, H, H_k, n_tokens,
                sq1, sq2, sq3, sk1, sk2, sk3, sv1, sv2, sv3, sb1, sb2, sb3, rq3_magic, scale, rec_floats, rec_set);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

bool ggml_cuda_gated_delta_net_lazy_supported(const ggml_tensor * op) {
    const int     mode = ggml_get_op_params_i32(op, 0);
    const int64_t S_v  = mode == 2 ? ggml_get_op_params_i32(op, 1) : op->src[2]->ne[0];
    if (S_v != 16 && S_v != 32 && S_v != 64 && S_v != 128) {
        return false;
    }
    if (op->type != GGML_TYPE_F32 || op->src[5]->type != GGML_TYPE_F32 || op->src[6]->type != GGML_TYPE_F32 ||
        op->src[7]->type != GGML_TYPE_I32) {
        return false;
    }
    for (int i = 0; mode == 1 && i < 5; i++) {
        if (op->src[i]->type != GGML_TYPE_F32) {
            return false;
        }
    }
    return true;
}

void ggml_cuda_op_gated_delta_net_lazy(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int mode = ggml_get_op_params_i32(dst, 0);

    const ggml_tensor * src_state = dst->src[5];
    const ggml_tensor * src_rec   = dst->src[6];
    const ggml_tensor * src_info  = dst->src[7];

    const int64_t S_v        = mode == 2 ? ggml_get_op_params_i32(dst, 1) : dst->src[2]->ne[0];
    const int64_t H          = mode == 2 ? ggml_get_op_params_i32(dst, 2) : dst->src[2]->ne[1];
    const int64_t H_k        = mode == 2 ? ggml_get_op_params_i32(dst, 3) : dst->src[1]->ne[1];
    const int64_t n_info     = src_info->ne[1];
    const int64_t rec_floats = src_rec->ne[0];
    const int64_t rec_set    = src_rec->nb[2] / sizeof(float);

    GGML_ASSERT(ggml_is_contiguous(src_state) && ggml_is_contiguous(src_rec) && ggml_is_contiguous(src_info));

    float *         state_d = (float *) src_state->data;
    float *         rec_d   = (float *) src_rec->data;
    const int32_t * info_d  = (const int32_t *) src_info->data;

    cudaStream_t stream = ctx.stream();

    if (mode == 2) {
        launch_gated_delta_net_lazy<true>(nullptr, nullptr, nullptr, nullptr, nullptr, state_d, rec_d, info_d, nullptr,
            S_v, H, H_k, 0, n_info, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1.0f, rec_floats, rec_set, stream);
        return;
    }

    const ggml_tensor * src_q    = dst->src[0];
    const ggml_tensor * src_k    = dst->src[1];
    const ggml_tensor * src_v    = dst->src[2];
    const ggml_tensor * src_g    = dst->src[3];
    const ggml_tensor * src_beta = dst->src[4];

    GGML_ASSERT(ggml_is_contiguous_rows(src_q) && ggml_is_contiguous_rows(src_k) && ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_is_contiguous(src_g) && ggml_is_contiguous(src_beta));
    GGML_ASSERT(src_v->ne[3] % src_q->ne[3] == 0);

    launch_gated_delta_net_lazy<false>(
        (const float *) src_q->data, (const float *) src_k->data, (const float *) src_v->data,
        (const float *) src_g->data, (const float *) src_beta->data, state_d, rec_d, info_d, (float *) dst->data,
        S_v, H, H_k, src_v->ne[2], n_info,
        src_q->nb[1] / sizeof(float), src_q->nb[2] / sizeof(float), src_q->nb[3] / sizeof(float),
        src_k->nb[1] / sizeof(float), src_k->nb[2] / sizeof(float), src_k->nb[3] / sizeof(float),
        src_v->nb[1] / sizeof(float), src_v->nb[2] / sizeof(float), src_v->nb[3] / sizeof(float),
        src_beta->nb[1] / sizeof(float), src_beta->nb[2] / sizeof(float), src_beta->nb[3] / sizeof(float),
        src_v->ne[3] / src_q->ne[3], 1.0f / sqrtf((float) S_v), rec_floats, rec_set, stream);
}
