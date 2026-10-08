// the fork's members of llm_graph_context: the device cache of host MoE experts in the graph (--moe-cache).
// llama-graph.h declares them

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-cparams.h"
#include "llama-moecache-fork.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>

// mirror of the scheduler: the backend that runs a mul_mat_id node with a weight in a host buffer
static bool moe_cache_runs_on_cpu(ggml_backend_sched_t sched, ggml_backend_t backend_cpu, bool op_offload, const ggml_tensor * op) {
    ggml_backend_buffer_t buf = op->src[0]->buffer;

    const int n_backends = ggml_backend_sched_get_n_backends(sched);

    int id = -1;
    for (int i = 0; i < n_backends; ++i) {
        ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
        if (ggml_backend_supports_buft(b, ggml_backend_buffer_get_type(buf)) && ggml_backend_supports_op(b, op)) {
            id = i;
            break;
        }
    }

    if (id < 0 || ggml_backend_sched_get_backend(sched, id) != backend_cpu) {
        return false;
    }

    // a large batch goes to the device with a copy of the weights
    if (op_offload && id == n_backends - 1 && ggml_backend_buffer_is_host(buf)) {
        for (int i = 0; i < id; ++i) {
            ggml_backend_t b = ggml_backend_sched_get_backend(sched, i);
            if (ggml_backend_supports_op(b, op) && ggml_backend_offload_op(b, op)) {
                return false;
            }
        }
    }

    return true;
}

ggml_tensor * llm_graph_context::build_moe_cache(
         ggml_tensor * mm_up,
         ggml_tensor * mm_gate,
         ggml_tensor * mm_down,
     llm_ffn_op_type   type_op,
                bool   has_gate,
                bool   has_gate_exps,
                 int   il) const {
    const llama_moe_cache_fork::layer * cl = moe_cache_fork->get_layer(il);
    if (cl == nullptr) {
        return nullptr;
    }

    auto is_mm = [](const ggml_tensor * t, const ggml_tensor * w) {
        return t != nullptr && t->op == GGML_OP_MUL_MAT_ID && t->src[0] == w && t->src[3] == nullptr;
    };

    // lora or other weights change the matmuls
    if (!is_mm(mm_up, cl->w_up) || !is_mm(mm_down, cl->w_down) || (mm_gate != nullptr) != (cl->w_gate != nullptr) ||
            (mm_gate != nullptr && !is_mm(mm_gate, cl->w_gate))) {
        return nullptr;
    }

    ggml_tensor * cur              = mm_up->src[1]; // [n_embd, 1, n_tokens]
    ggml_tensor * selected_experts = mm_up->src[2]; // [n_expert_used, n_tokens]

    const int64_t n_used = selected_experts->ne[0];
    const int64_t n_tok  = selected_experts->ne[1];

    if (n_tok > llama_moe_cache_fork::N_TOKENS_MAX || n_used*n_tok > ggml_nelements(cl->ids)) {
        return nullptr;
    }

    ggml_tensor * mms[3] = { mm_up, mm_gate, mm_down };

    // the cache only replaces host matmuls, a batch that the scheduler offloads keeps its path
    for (ggml_tensor * mm : mms) {
        if (mm != nullptr && !moe_cache_runs_on_cpu(sched, backend_cpu, cparams.op_offload, mm)) {
            return nullptr;
        }
    }

    ggml_backend_t backend_dev = nullptr;
    for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
        if (ggml_backend_get_device(ggml_backend_sched_get_backend(sched, i)) == moe_cache_fork->get_dev()) {
            backend_dev = ggml_backend_sched_get_backend(sched, i);
        }
    }
    if (backend_dev == nullptr) {
        return nullptr;
    }

    // slot of each routed expert, the experts that are not cached get the zero slot, so a row of ids can repeat a slot
    ggml_tensor * ids = ggml_cont(ctx0, selected_experts);
    ids = ggml_reshape_1d(ctx0, ids, n_used*n_tok);
    ids = ggml_get_rows(ctx0, cl->table, ids);
    ids = ggml_reshape_2d(ctx0, ids, n_used, n_tok);
    cb(ids, "ffn_moe_cache_ids", il);

    ggml_tensor * up   = nullptr;
    ggml_tensor * gate = nullptr;

    if (cl->merged) {
        ggml_tensor * gate_up = ggml_mul_mat_id(ctx0, cl->s_up, cur, ids);
        ggml_mul_mat_set_hint(gate_up, GGML_HINT_IDS_REPEAT);
        cb(gate_up, "ffn_moe_cache_gate_up", il);

        const int64_t n_ff = gate_up->ne[0] / 2;
        gate = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], 0);
        up   = ggml_view_3d(ctx0, gate_up, n_ff, gate_up->ne[1], gate_up->ne[2], gate_up->nb[1], gate_up->nb[2], n_ff * gate_up->nb[0]);
    } else {
        up = ggml_mul_mat_id(ctx0, cl->s_up, cur, ids);
        ggml_mul_mat_set_hint(up, GGML_HINT_IDS_REPEAT);
        cb(up, "ffn_moe_cache_up", il);

        gate = up;
        if (cl->s_gate) {
            gate = ggml_mul_mat_id(ctx0, cl->s_gate, cur, ids);
            ggml_mul_mat_set_hint(gate, GGML_HINT_IDS_REPEAT);
            cb(gate, "ffn_moe_cache_gate", il);
        }
    }

    ggml_tensor * act = build_moe_ffn_act(gate, up, type_op, has_gate, has_gate_exps, il);

    ggml_tensor * down = ggml_mul_mat_id(ctx0, cl->s_down, act, ids);
    ggml_mul_mat_set_hint(down, GGML_HINT_IDS_REPEAT);
    if (arch == LLM_ARCH_MISTRAL4) {
        ggml_prec_set_src(down, GGML_PREC_F32, 1);
    }
    cb(down, "ffn_moe_cache_down", il);

    // every new node has to run on the cache device
    {
        std::vector<ggml_tensor *> todo = { down };
        std::vector<ggml_tensor *> seen;
        while (!todo.empty()) {
            ggml_tensor * t = todo.back();
            todo.pop_back();
            if (t == nullptr || t == cur || t == selected_experts || t->op == GGML_OP_NONE ||
                    std::find(seen.begin(), seen.end(), t) != seen.end()) {
                continue;
            }
            seen.push_back(t);
            if (!ggml_backend_supports_op(backend_dev, t)) {
                return nullptr;
            }
            for (ggml_tensor * src : t->src) {
                todo.push_back(src);
            }
        }
    }

    for (ggml_tensor * mm : mms) {
        if (mm != nullptr) {
            ggml_mul_mat_id_add_skip(mm, cl->skip);
        }
    }

    for (ggml_tensor * mm : mms) {
        if (mm != nullptr && !ggml_backend_supports_op(backend_cpu, mm)) {
            for (ggml_tensor * m : mms) {
                if (m != nullptr) {
                    ggml_mul_mat_id_add_skip(m, nullptr);
                }
            }
            return nullptr;
        }
    }

    // routing ids for llama_moe_cache_fork::update()
    ggml_tensor * ids_dst = ggml_view_2d(ctx0, cl->ids, n_used, n_tok, n_used*ggml_element_size(cl->ids), 0);
    ggml_tensor * ids_host = ggml_cpy(ctx0, selected_experts, ids_dst);

    // the scheduler overlap is on unless GGML_SCHED_PARALLEL_CPU=0 (ggml_backend_sched_new)
    const char * parallel_cpu = getenv("GGML_SCHED_PARALLEL_CPU");
    if (parallel_cpu == nullptr || atoi(parallel_cpu) > 0) {
        // Copy the CPU inputs before the device starts the cached experts.
        ggml_tensor * cur_host = ggml_dup(ctx0, cur);
        ggml_set_name(cur_host, "ffn_moe_cache_input_host");
        ggml_backend_sched_set_tensor_backend(sched, cur_host, backend_cpu);
        ggml_backend_sched_set_tensor_backend(sched, ids_host, backend_cpu);
        ggml_build_forward_expand(gf, cur_host);
        ggml_build_forward_expand(gf, ids_host);

        mm_up->src[1] = cur_host;
        for (ggml_tensor * mm : mms) {
            if (mm != nullptr) {
                mm->src[2] = ids_host;
            }
        }
        if (mm_gate != nullptr) {
            mm_gate->src[1] = cur_host;
        }
    }

    ggml_build_forward_expand(gf, down);
    ggml_build_forward_expand(gf, ids_host);

    res->moe_cache_il.push_back(il);
    res->moe_cache_n_ids.push_back(n_used*n_tok);

    return down;
}
