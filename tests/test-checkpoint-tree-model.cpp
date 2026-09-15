#include "arg.h"
#include "common.h"
#include "checkpoint-tree.h"
#include "llama.h"

#include <cstdio>
#include <vector>

// Round-trip test for the context-checkpoint tree against a (tiny) real model:
// build a shared BASE + two divergent branches, then restore each branch through the tree and
// verify the recovered sequence state matches what a straight decode produced. This exercises the
// delta encode/apply along tree paths and the branch-selection matching with a real KV cache.
// Requires an attention (non-recurrent) model, e.g. the tiny stories260K used by the server tests.

static std::vector<uint8_t> get_state(llama_context * ctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, 0);
    std::vector<uint8_t> data(size);
    const size_t written = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), seq_id, 0);
    GGML_ASSERT(written == data.size());
    return data;
}

static common_prompt_checkpoint make_cp(llama_context * ctx, llama_seq_id seq_id,
        int64_t n_tokens, llama_pos pos_min, llama_pos pos_max, llama_pos base_pos) {
    common_prompt_checkpoint cp;
    cp.update_pos(n_tokens, pos_min, pos_max);
    // NONE = full state (both base + SWA sub-caches), matching what the server writes for
    // attention/SWA models. PARTIAL_ONLY would mean "SWA-only" for an iSWA cache and so would
    // not round-trip against a full-state snapshot; it is reserved for hybrid (recurrent-only).
    cp.update_tgt(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_NONE, base_pos);
    return cp;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.n_ctx = 2048;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    auto init = common_init_from_params(params);
    llama_model * model = init->model();
    if (!model) {
        return 1;
    }

    llama_context_ptr ctx(llama_init_from_model(model, common_context_params_to_llama(params)));
    if (!ctx) {
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // two continuations sharing a common prefix
    std::vector<llama_token> tokA = common_tokenize(vocab, "The quick brown fox jumps over the lazy dog near the river.", true);
    std::vector<llama_token> tokB = common_tokenize(vocab, "The quick brown fox runs across the wide green meadow today.", true);
    if (tokA.size() < 6 || tokB.size() < 6) {
        return 1;
    }

    size_t P = 0;
    while (P < tokA.size() && P < tokB.size() && tokA[P] == tokB[P]) {
        ++P;
    }
    if (P < 2 || P >= tokA.size() || P >= tokB.size()) {
        fprintf(stderr, "need a shared prefix and divergence in both branches (P = %zu)\n", P);
        return 1;
    }

    const llama_seq_id seq_id = 0;
    common_checkpoint_tree tree;

    // shared prefix -> BASE (root)
    int n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokA, (int) P, n_past, params.n_batch, {}, false)) {
        return 1;
    }
    const llama_pos base_pos = n_past - 1;
    common_prompt_checkpoint base = make_cp(ctx.get(), seq_id, n_past, 0, base_pos, -1);
    const int32_t root = tree.add(-1, std::vector<llama_token>(tokA.begin(), tokA.begin() + P), std::move(base));

    // branch A
    if (!common_prompt_batch_decode(ctx.get(), tokA, (int) (tokA.size() - P), n_past, params.n_batch, {}, false)) {
        return 1;
    }
    const std::vector<uint8_t> expectedA = get_state(ctx.get(), seq_id);
    common_prompt_checkpoint dA = make_cp(ctx.get(), seq_id, n_past, base_pos + 1, n_past - 1, base_pos);
    const int32_t nodeA = tree.add(root, std::vector<llama_token>(tokA.begin() + P, tokA.end()), std::move(dA));

    // restore the base, then decode branch B from the divergence point
    GGML_ASSERT(tree.get(root)->cp.apply(ctx.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
    n_past = (int) P;
    if (!common_prompt_batch_decode(ctx.get(), tokB, (int) (tokB.size() - P), n_past, params.n_batch, {}, false)) {
        return 1;
    }
    const std::vector<uint8_t> expectedB = get_state(ctx.get(), seq_id);
    common_prompt_checkpoint dB = make_cp(ctx.get(), seq_id, n_past, base_pos + 1, n_past - 1, base_pos);
    const int32_t nodeB = tree.add(root, std::vector<llama_token>(tokB.begin() + P, tokB.end()), std::move(dB));

    // branch selection by request tokens
    size_t m = 0;
    if (tree.find_best(tokA, m) != nodeA || m != tokA.size()) {
        fprintf(stderr, "find_best(A) mismatch\n");
        return 1;
    }
    if (tree.find_best(tokB, m) != nodeB || m != tokB.size()) {
        fprintf(stderr, "find_best(B) mismatch\n");
        return 1;
    }

    // restore each branch through the tree (base + delta path) and compare recovered state
    auto restore = [&](int32_t node) -> std::vector<uint8_t> {
        const auto path = tree.path_to(node);
        GGML_ASSERT(!path.empty());
        for (size_t k = 0; k < path.size(); ++k) {
            GGML_ASSERT(tree.get(path[k])->cp.apply(ctx.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_NONE));
        }
        return get_state(ctx.get(), seq_id);
    };

    if (restore(nodeA) != expectedA) {
        fprintf(stderr, "branch A round-trip mismatch\n");
        return 1;
    }
    if (restore(nodeB) != expectedB) {
        fprintf(stderr, "branch B round-trip mismatch\n");
        return 1;
    }
    // reuse A again after B, proving branches are independent and repeatedly restorable
    if (restore(nodeA) != expectedA) {
        fprintf(stderr, "branch A re-restore mismatch\n");
        return 1;
    }

    printf("checkpoint tree model round-trip passed: prefix=%zu tokA=%zu tokB=%zu (base=%zu A=%zu B=%zu bytes)\n",
            P, tokA.size(), tokB.size(),
            tree.get(root)->cp.size(), tree.get(nodeA)->cp.size(), tree.get(nodeB)->cp.size());
    return 0;
}
