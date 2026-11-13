#pragma once

// The parts of the Vulkan backend that only the fork has. The upstream files keep the fields this code needs and short
// calls into it, so that an upstream merge meets as little fork code as possible.

#include "../ggml-vulkan-common.h"

#include <functional>

// hc_post_gate followed by the grouped RMSNorm and gamma of the next hc mix (qwen4exp hc_combine -> hc_mix)
static constexpr std::initializer_list<ggml_op> hc_post_norm_pattern { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST, GGML_OP_RMS_NORM, GGML_OP_MUL };

static constexpr std::initializer_list<std::array<int, 3>> hc_post_norm_edges {
    { 1, 0, 0 }, // sigmoid->src[0]  == scale
    { 2, 0, 1 }, // scale->src[0]    == sigmoid
    { 3, 2, 2 }, // hc_post->src[2]  == scale (post)
    { 4, 0, 3 }, // rms_norm->src[0] == hc_post
    { 5, 0, 4 }, // mul->src[0]      == rms_norm
};

// hc_post_norm with the normalized streams then copied to f16 (the f16 hc read of qwen4exp, LLAMA_HC_F16): the shader
// stores them in f16 itself, the f32 ones are never written
static constexpr std::initializer_list<ggml_op> hc_post_norm_f16_pattern { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE, GGML_OP_DSV4_HC_POST, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_CPY };

static constexpr std::initializer_list<std::array<int, 3>> hc_post_norm_f16_edges {
    { 1, 0, 0 }, // sigmoid->src[0]  == scale
    { 2, 0, 1 }, // scale->src[0]    == sigmoid
    { 3, 2, 2 }, // hc_post->src[2]  == scale (post)
    { 4, 0, 3 }, // rms_norm->src[0] == hc_post
    { 5, 0, 4 }, // mul->src[0]      == rms_norm
    { 6, 0, 5 }, // cpy->src[0]      == mul
};

struct vk_op_dsv4_hc_post_norm_push_constants {
    uint32_t n_embd;
    uint32_t n_tokens;

    uint32_t nbx1;
    uint32_t nbr1; uint32_t nbr2;
    uint32_t nbp0; uint32_t nbp1;
    uint32_t nbw1; uint32_t nbw2;
    uint32_t nbd1; uint32_t nbd2;
    uint32_t nbn1; uint32_t nbn2;

    uint32_t x_offset;
    uint32_t r_offset;
    uint32_t p_offset;
    uint32_t w_offset;
    uint32_t d_offset;
    uint32_t n_offset;

    uint32_t gate;
    float    gate_scale_in;
    float    gate_scale_out;
    float    eps;
};

// the gated norm of the qwen4exp GDN output: RMS_NORM, MUL by the weight, SIGMOID of the gate, MUL (build_norm_gated)
static constexpr std::initializer_list<ggml_op> rms_norm_mul_sigmoid_mul_pattern { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_UNARY, GGML_OP_MUL };

static constexpr std::initializer_list<std::array<int, 3>> rms_norm_mul_sigmoid_mul_edges {
    { 1, 0, 0 }, // mul->src[0]  == rms_norm
    { 3, 0, 1 }, // mul2->src[0] == mul
    { 3, 1, 2 }, // mul2->src[1] == sigmoid
};

// the conv of a GDN layer over the concat of its state and x, with and without the SILU (CONCAT_SSM_CONV(_SILU))
static constexpr std::initializer_list<ggml_op> concat_ssm_conv_silu_pattern { GGML_OP_CONCAT, GGML_OP_SSM_CONV, GGML_OP_UNARY };
static constexpr std::initializer_list<ggml_op> concat_ssm_conv_pattern { GGML_OP_CONCAT, GGML_OP_SSM_CONV };

// mul_mat_vec_id_fork.comp (the MUL_MAT_ID mat-vec of the fork), strides in elements
struct vk_mmv_id_fork_push_constants {
    uint32_t ncols;
    uint32_t nrows;
    uint32_t expert_stride;   // blocks of one expert's matrix
    uint32_t ne11;
    uint32_t nei0;
    uint32_t nei1;
    uint32_t nbi1;
    uint32_t fusion_flags;
    uint32_t stride_d;
    uint32_t batch_stride_d;
};

// the l2 norm of the GDN q and k: RMS_NORM, SCALE (build_gdn_l2_norm)
static constexpr std::initializer_list<ggml_op> rms_norm_scale_pattern { GGML_OP_RMS_NORM, GGML_OP_SCALE };

// rms_norm_rows.comp, strides and offsets in elements
struct vk_op_rms_norm_rows_push_constants {
    uint32_t ncols;
    uint32_t ne01;
    uint32_t ne02;
    uint32_t nrows;
    uint32_t a_nb1, a_nb2, a_nb3;
    uint32_t g_nb1, g_nb2, g_nb3;
    uint32_t d_nb1, d_nb2, d_nb3;
    uint32_t w_ne0;
    uint32_t a_off, w_off, g_off, d_off;
    float    eps;
    float    scale;
};

// ssm_conv_concat.comp, strides and offsets in elements
struct vk_op_ssm_conv_concat_push_constants {
    uint32_t s_nb0, s_nb1, s_nb2;
    uint32_t x_nb0, x_nb1, x_nb2;
    uint32_t w_nb1;
    uint32_t d_nb1, d_nb2;
    uint32_t s_off, x_off, w_off, d_off;
    uint32_t nc;
    uint32_t ncs;
    uint32_t nr;
    uint32_t n_t;
    uint32_t n_s;
};

// gated_delta_net_lazy.comp, strides in elements
struct vk_op_gated_delta_net_lazy_push_constants {
    uint32_t H;
    uint32_t H_k;
    uint32_t n_tokens;
    uint32_t sq1, sq2, sq3;
    uint32_t sk1, sk2, sk3;
    uint32_t sv1, sv2, sv3;
    uint32_t sb1, sb2, sb3;
    uint32_t rq3;
    float    scale;
    uint32_t rec_floats;
    uint32_t rec_set;
};

// the large int8 coopmat tiles that GGML_VK_MMQ_INT_TILE, _ID and _GLU set, read by ggml_vk_load_shaders
struct vk_fork_int_tiles {
    // GGML_VK_MMQ_INT_TILE: the large int8 coopmat tile with its own workgroup size
    bool mmq_int_tile_set = false;
    std::array<uint32_t, 3> l_mmq_int_wg_denoms {};
    // GGML_VK_MMQ_INT_TILE_ID: the large int8 coopmat tile of matmul_id alone, the dense tile by default
    bool mmq_int_tile_id_set = false;
    std::vector<uint32_t> l_warptile_mmq_cm1_int_id;
    std::array<uint32_t, 3> l_mmq_int_id_wg_denoms {};
    // GGML_VK_MMQ_INT_TILE_GLU: the large int8 coopmat tile of the fused gate/up matmul_id (FUSED_GLU) alone
    bool mmq_int_tile_glu_set = false;
    std::vector<uint32_t> l_warptile_mmq_cm1_int_glu;
    std::array<uint32_t, 3> l_mmq_int_glu_wg_denoms {};
};

// helpers of ggml-vulkan.cpp that the fork files call
bool ggml_vk_matmul_cm1_int_shmem_support(const vk_device& device, const std::vector<uint32_t>& warptile, bool mul_mat_id, ggml_type src0_type, bool glu);
const std::vector<vk_matmul_pipeline_pair>* ggml_vk_get_mul_mat_mat_pipeline_map(ggml_backend_vk_context * ctx, ggml_type src0_type, ggml_type src1_type, ggml_prec prec, bool mul_mat_id, bool glu);
vk_pipeline ggml_vk_guess_matmul_pipeline_map(ggml_backend_vk_context * ctx, const std::vector<vk_matmul_pipeline_pair>& configs, uint32_t m, uint32_t n, bool aligned, bool mul_mat_id);
uint32_t ggml_vk_guess_matmul_pipeline_align_map(ggml_backend_vk_context * ctx, const std::vector<vk_matmul_pipeline_pair>& configs, uint32_t m, uint32_t n, bool mul_mat_id);
bool ggml_vk_can_fuse_hc_post_gate(const struct ggml_cgraph * cgraph, int node_idx);

//
// ggml-vulkan-fork.cpp: device setup, tile overrides, the shared buffer type, graph timing
//

// ggml_vk_get_device
// GGML_SCHED_LOG_REALLOC: the buffer limits of the device
void ggml_vk_fork_log_limits(const vk_device & device);
// ggml_backend_vk_host_buffer_type_alloc_buffer: the slack added to a pinned host buffer of `size` bytes, 0 where the
// slack alone would take it over the device limit (GGML_VK_HOST_PAD_UPSTREAM=1: always 32, as upstream)
size_t ggml_vk_fork_host_buffer_pad(const vk_device & device, size_t size);
// GGML_VK_MMID_F16REG: sets mmid_f16reg_mode (1 after the coopmat A layout probe, 2 emulated), runs once the compute
// queue exists
void ggml_vk_fork_mmid_f16reg_init(vk_device & device);
bool ggml_vk_fork_coopmat_a_probe(vk_device & device);
// coopmat_c_probe.comp: the f32 accumulator layout mul_mmid_f16reg.comp writes from with EPI 1
bool ggml_vk_fork_coopmat_c_probe(vk_device & device);
// GGML_VK_MM_F16REG=<r|l><rows>x<tokens>/<m>x<n>: the variant of the dense q8_0 matmul on mul_mm_f16reg.comp (sets
// mm_f16reg_mode, _lds, _tile, _name), after the shared memory limit and the A layout probe
void ggml_vk_fork_mm_f16reg_init(vk_device & device);
// GGML_VK_INT_COOPMAT: sets coopmat_int_ops and returns whether int8 coopmat may be enabled; mode gets the env value
// (-1 when unset)
bool ggml_vk_fork_int_coopmat(vk_device & device, int & mode);
// the log line of an enabled int8 coopmat shape
void ggml_vk_fork_log_int_coopmat(const vk_device & device, int mode, uint32_t m, uint32_t n, uint32_t k);
// GGML_VK_F16ACC: f16 accumulators of the coopmat matmuls, off by default on the AMD driver
void ggml_vk_fork_f16acc(vk_device & device);
// GGML_VK_SHMEM_LIMIT, GGML_VK_SHMEM_PROBE: the shared memory limit of the shaders, once the compute queue exists
void ggml_vk_fork_shmem_limit(vk_device & device);

// sparse buffers of the device (25.10), logged once: an elastic KV on Vulkan would reserve the range of the whole cache
// and bind memory to the cells in use, as the CUDA one maps VMM granules (MoltenVK has none)
void ggml_vk_fork_log_sparse(const vk_device & device);

// elastic buffers (26.10): the clear of a buffer with memory bound to some granules only clears those; false for an
// ordinary buffer
bool ggml_vk_fork_elastic_clear(ggml_backend_vk_buffer_context * ctx, uint8_t value);
// GGML_VK_INT_LARGE_TILE: the large int8 coopmat tiles for matmul and matmul_id even where the large f16 tile is off
void ggml_vk_fork_int_large_tile(const vk_device & device, bool & matmul, bool & matmul_id);
// the <name>_Shared buffer type and GGML_VK_MM_CHUNK_MB
void ggml_vk_fork_device_init(vk_device & device, size_t idx);

// ggml_vk_load_shaders
// GGML_VK_MMQ_TILE: the medium tile of the quant matmuls and matmul_id
void ggml_vk_fork_mmq_tile(vk_device & device, const vk_pipeline & requested, std::vector<uint32_t> & m_warptile_mmq,
                           std::vector<uint32_t> & m_warptile_mmqid, std::array<uint32_t, 3> & m_mmq_wg_denoms);
// GGML_VK_MMQ_INT_TILE, _ID, _GLU: the large int8 coopmat tiles
void ggml_vk_fork_int_tiles(vk_device & device, const vk_pipeline & requested, uint32_t cm1_sg, uint32_t itm, uint32_t itn,
                            uint32_t itk, std::vector<uint32_t> & l_warptile_mmq_cm1_int, vk_fork_int_tiles & tiles);

// get_proc_address of the backend reg: ggml_backend_dev_get_extra_bufts (the <name>_Shared buffer type)
void * ggml_backend_vk_reg_get_proc_address_fork(ggml_backend_reg_t reg, const char * name);

// GGML_TIMELINE (ggml_timeline_on): the gpu span of each command buffer of a compute queue, timestamps at its begin
// (ggml_vk_ctx_begin) and its end (ggml_vk_ctx_end), read back at the syncs (ggml_vk_synchronize) and written to the
// timeline at the end of the capture (ggml_backend_profiler_fork)
void ggml_vk_fork_timeline_begin(vk_device & device, vk_context & subctx);
void ggml_vk_fork_timeline_end(vk_context & ctx);
void ggml_vk_fork_timeline_collect(const vk_device & device);

// GGML_VK_GRAPH_TIMING: the start of graph_compute, returns the entry time (0 when off)
int64_t ggml_vk_fork_graph_timing_begin(ggml_backend_vk_context * ctx);
// the end of the last node in ggml_vk_build_graph, before the context ends
void ggml_vk_fork_graph_timing_end(ggml_backend_vk_context * ctx, const vk_context & compute_ctx, bool last_node);
// the end of graph_compute
void ggml_vk_fork_graph_timing_finish(ggml_backend_vk_context * ctx, int64_t t_enter_us);

//
// ggml-vulkan-ops-fork.cpp: pipeline choices and paths of single ops
//

// GET_ROWS: rows shorter than the workgroup take the flat pipeline
bool ggml_vk_get_rows_flat(const ggml_tensor * src0, const ggml_tensor * src1);
vk_pipeline ggml_vk_get_rows_flat_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst);
// CONCAT of a transposed source through tiles, nullptr for the plain concat
vk_pipeline ggml_vk_concat_transpose_pipeline(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1);
// TOP_K: radix-select instead of the tournament (GGML_VK_TOPK_RADIX); tournament_cols: the elements of the widest
// workgroup of the tournament, what it reduces to k in one pass
bool ggml_vk_topk_use_radix(uint32_t k, uint32_t ncols, uint32_t nrows, uint32_t tournament_cols);
// TOP_K radix-select with an output in index order and the ties from the lowest index (GGML_VK_TOPK_RADIX_STABLE):
// 1 with subgroups where the device has them (else upstream), 2 without subgroups, 0 the upstream atomic compaction
int ggml_vk_topk_radix_stable();
// the copy of f32 src to f16 (dst: the layout of the copy, nullptr = contiguous) for the f16reg matmuls: the upstream
// pipeline ggml_vk_get_cpy_pipeline picks, its saturating twin for the f32 -> f16 ones (cpy_f16_sat_fork.comp)
vk_pipeline ggml_vk_fork_cpy_f16_sat(ggml_backend_vk_context * ctx, const ggml_tensor * src, const ggml_tensor * dst);

// MUL_MAT_ID: the mul_mmid_f16reg.comp pipeline for these operands, or nullptr for the upstream choice
// (y_in_place: src1 can be read where it is; b16: set when the pipeline reads f16 activations - src1 when it is f16,
// else an f16 copy that the caller makes in prealloc_y; d16: the fused gate/up writes its result as f16)
vk_pipeline ggml_vk_fork_mmid_f16reg(ggml_backend_vk_context * ctx, const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * glu_up,
                                     bool y_in_place, uint64_t n_per_expert, bool * b16 = nullptr, bool d16 = false);
// MUL_MAT: the matmuls that stay off the int8 path (GGML_VK_INT_COOPMAT=3, graph outputs, K under 512)
bool ggml_vk_mul_mat_int8_off(const ggml_backend_vk_context * ctx, const ggml_tensor * dst, uint64_t ne10);
// the physical memory the system can still hand out without paging, 0 when unknown (fork/ggml-vulkan-host-fork.cpp)
uint64_t ggml_vk_fork_host_avail_bytes();
// the local video memory all processes hold on the adapter with this LUID, by the system's counter (Windows); false when
// unknown (fork/ggml-vulkan-host-fork.cpp)
bool ggml_vk_fork_adapter_dedicated_usage(const uint8_t luid[8], uint64_t * bytes);
// whether `size` more bytes of device local memory type `type` stay within the limit of its heap (less
// GGML_VK_DEVICE_MARGIN_MIB): the heap less what other processes hold where the system counts that, else the
// VK_EXT_memory_budget (ggml_vk_fork_heap_limit); true for other memory and without the extension. ggml_vk_create_buffer
// skips a type that has no room: the AMD driver of the 395 hands out device memory past the carve-out and backs it with
// system memory
bool ggml_vk_fork_heap_has_room(const vk_device & device, uint32_t type, uint64_t size);
// MUL_MAT: whether ggml_vk_mul_mat quantizes its f16 src1 into prealloc_y as q8_1 x4 blocks (in ggml-vulkan.cpp)
bool ggml_vk_fork_mul_mat_quantizes_y_x4(ggml_backend_vk_context * ctx, const ggml_tensor * dst);
// MUL_MAT_ID of up to 8 tokens on mul_mat_vec_id_fork.comp (GGML_VK_MMV_ID_FORK): the mode and pipelines at device
// init; whether the mat-vec path takes dst there (in ggml-vulkan.cpp, ggml_vk_can_fuse asks it for the MUL by the
// routing weights at more than one token); the dispatch, false for the upstream path
void ggml_vk_fork_mmv_id_init(vk_device & device);
// the dense int8 mat-vec of 5-8 columns on RDNA3/4: true keeps upstream's 4 rows per workgroup (rm_int_n in
// ggml_vk_load_shaders), false the rows of 1-4 columns (GGML_VK_MMVQ_N5_ROWS4)
bool ggml_vk_fork_mmvq_n5_rows4(const vk_device & device);
// the q8_1 activations of the int8 mat-vec in two planes (GGML_VK_MMVQ_PLANES=2, see mul_mat_vecq_planes_fork.glsl):
// device->mmvq_planes at device init; the planes of a weight type; the spec constants of a pipeline of mul_mat_vecq.comp
// with the planes added (in ggml_vk_load_shaders); the quantize pipeline of the mat-vec of a weight type (the matmul
// keeps one plane)
void ggml_vk_fork_mmvq_planes_init(vk_device & device);
uint32_t ggml_vk_fork_mmvq_planes_of(const vk_device & device, ggml_type type);
void ggml_vk_fork_mmvq_spec(const vk_device & device, const char * name, std::vector<uint32_t> & spec);
vk_pipeline ggml_vk_fork_mmvq_quantize_pipeline(const ggml_backend_vk_context * ctx, ggml_type type);
// GGML_VK_INT_DOT_EMULATED=1: the int8 dot where the device has it without acceleration (MoltenVK), for tests
bool ggml_vk_fork_int_dot_emulated();
bool ggml_vk_fork_mmv_id_takes(const ggml_backend_vk_context * ctx, const ggml_tensor * dst);
bool ggml_vk_fork_mmv_id(ggml_backend_vk_context * ctx, vk_context & subctx, const struct ggml_cgraph * cgraph, int node_idx);
// MUL_MAT of q8_0 weights on mul_mm_f16reg.comp (GGML_VK_MM_F16REG): dispatches it and returns true, false for the
// upstream path (off, a fused op, another type or shape). hc_pre: the gated DSV4_HC_PRE after it takes the store
// (MUL_MAT_DSV4_HC_PRE); check_only: the answer without a dispatch
// the dense q8_0 MUL_MAT of 2-8 columns on mul_mm_f16reg.comp with the tile of GGML_VK_MM_F16REG_VERIFY: dispatches it
// and returns true, false for the mat-vec path; check_only: the answer without a dispatch
bool ggml_vk_fork_mm_f16reg_verify(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * src0,
                                   const ggml_tensor * src1, const ggml_tensor * dst, bool check_only);
bool ggml_vk_fork_mm_f16reg(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * src0, const ggml_tensor * src1,
                            const ggml_tensor * dst, const ggml_tensor * fused_dst, uint32_t fusion_flags,
                            const ggml_tensor * hc_pre = nullptr, bool check_only = false);
// GATED_DELTA_NET_LAZY: the net with deferred rollback and the replay alone (ggml_gated_delta_net_lazy), they write into
// the state rows and records they get as sources
bool ggml_vk_gated_delta_net_lazy_supported(const ggml_tensor * op);
void ggml_vk_gated_delta_net_lazy(ggml_backend_vk_context * ctx, vk_context & subctx, ggml_tensor * dst);
// set_tensor_2d_async: writes weights in place where the device memory is mapped, returns false for the staged copy
bool ggml_vk_set_tensor_mapped(vk_context & cpy_ctx, const ggml_tensor * tensor, vk_buffer & buf, uint64_t dst_offset, const void * data,
                               size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data);

//
// ggml-vulkan-fusion-fork.cpp: the fusions of the fork
//

// dispatchers of fused nodes
void ggml_vk_dsv4_hc_post_norm(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx);
// RMS_NORM of rows up to 512 columns, plain or with the fused MUL (RMS_NORM_MUL), SCALE or gated norm; false when the
// node takes the upstream path (GGML_VK_RMS_NORM_ROWS=0, longer rows, the partials of a fused ADD, other fusions)
bool ggml_vk_fork_rms_norm_rows(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx);
// CONCAT, SSM_CONV (+ SILU)
void ggml_vk_fork_ssm_conv_concat(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx);
// LIGHTNING_INDEXER of a prompt on the MUL_MAT_HEADSUM matmul, false when it does not fit (ggml-vulkan-fusion-fork.cpp)
bool ggml_vk_fork_lid_headsum(ggml_backend_vk_context * ctx, vk_context & subctx, const ggml_tensor * dst);
void ggml_vk_mul_mat_headsum(ggml_backend_vk_context * ctx, vk_context& subctx, const struct ggml_cgraph * cgraph, int node_idx);

// the fusion chain of graph_compute: each sets num_additional_fused_ops, the ctx flags, op_srcs_fused_elementwise and
// fusion_string and returns true when its fusion applies at node_idx
void ggml_vk_fork_fusion_reset(ggml_backend_vk_context * ctx);
// MUL_MAT_HEADSUM, MUL_MAT_RELU, MUL_MAT_ID_MUL_MAT_ID_GLU
bool ggml_vk_fork_fuse_mul_mat(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx,
                               bool * op_srcs_fused_elementwise, const char *& fusion_string);
// HC_POST_NORM
bool ggml_vk_fork_fuse_hc_post_norm(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx,
                                    bool * op_srcs_fused_elementwise, const char *& fusion_string);
// RMS_NORM_SCALE, RMS_NORM_MUL_SIGMOID_MUL (rms_norm_rows.comp)
bool ggml_vk_fork_fuse_rms_norm_rows(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx,
                                     bool * op_srcs_fused_elementwise, const char *& fusion_string);
// CONCAT_SSM_CONV(_SILU)
bool ggml_vk_fork_fuse_concat_ssm_conv(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx,
                                       bool * op_srcs_fused_elementwise, const char *& fusion_string);
// GATED_DELTA_NET_CACHE
bool ggml_vk_fork_fuse_gdn_cache(ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx,
                                 bool * op_srcs_fused_elementwise, const char *& fusion_string);

// ggml_vk_graph_optimize
// the nodes of MUL_MAT_HEADSUM patterns, which the window does not pull forward one by one
std::vector<uint8_t> ggml_vk_fork_headsum_nodes(ggml_backend_vk_context * ctx, const struct ggml_cgraph * graph);
// the interior nodes of RMS_NORM_MUL_SIGMOID_MUL and CONCAT_SSM_CONV patterns, which the window does not pull forward
std::vector<uint8_t> ggml_vk_fork_protected_nodes(const struct ggml_cgraph * graph);
// schedules the MUL_MAT_HEADSUM pattern that starts at first_unused in a row, false if there is none
bool ggml_vk_fork_keep_headsum(struct ggml_cgraph * graph, struct ggml_backend_graph_optimize_params * params, int & first_unused,
                               std::vector<uint8_t> & used, std::vector<ggml_tensor *> & new_order, std::set<ggml_tensor *> & used_node_set);
// pulls the nodes a fusion needs next to the node at first_unused: the cache cpy of GATED_DELTA_NET, the other matmul
// and the swiglu of a MoE gate/up, the SSM_CONV (+ SILU) of a CONCAT
void ggml_vk_fork_pull_nodes(ggml_backend_vk_context * ctx, const struct ggml_cgraph * graph, int first_unused, int num_to_check,
                             std::vector<uint8_t> & used, std::vector<int> & current_set, const std::function<bool(int, int)> & is_src_of,
                             struct ggml_backend_graph_optimize_params * params);
