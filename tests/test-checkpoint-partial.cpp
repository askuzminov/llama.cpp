// Coverage for the partial (PARTIAL_ONLY) context checkpoints the server creates.
//
// PARTIAL_ONLY means a different thing per model family:
//   - plain attention: the whole state
//   - iSWA:            the SWA cache only (the base cache is rolled back with seq_rm)
//   - hybrid:          the recurrent state only (the attention KV is rolled back with seq_rm)
//   - recurrent:       the whole state
//
// Two properties are checked for whichever family the given model belongs to:
//   A) restoring a partial checkpoint and rolling the rest of the memory back with seq_rm, then
//      replaying the remaining tokens, reproduces the logits of an uninterrupted run;
//   B) if the memory can produce a partial delta, base + delta reproduces the full partial state.

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static const llama_state_seq_flags FLAGS = LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

static std::vector<uint8_t> get_state(llama_context * ctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, FLAGS);
    std::vector<uint8_t> data(size);
    const size_t written = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), seq_id, FLAGS);
    GGML_ASSERT(written == data.size());
    return data;
}

static bool set_state(llama_context * ctx, const std::vector<uint8_t> & data, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, data.data(), data.size(), seq_id, FLAGS) == data.size();
}

static std::vector<float> get_logits(llama_context * ctx, const llama_model * model) {
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    const float * src = llama_get_logits(ctx);
    GGML_ASSERT(src);
    return std::vector<float>(src, src + n_vocab);
}

static int argmax(const std::vector<float> & v) {
    int best = 0;
    for (int i = 1; i < (int) v.size(); ++i) {
        if (v[i] > v[best]) {
            best = i;
        }
    }
    return best;
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

    std::vector<llama_token> tokens;
    if (llama_vocab_type(vocab) == LLAMA_VOCAB_TYPE_NONE) {
        // synthetic test models carry no tokenizer
        const int n_vocab = llama_vocab_n_tokens(vocab);
        for (int i = 0; i < 16; ++i) {
            tokens.push_back((llama_token) ((7*i + 1) % n_vocab));
        }
    } else {
        tokens = common_tokenize(
                vocab, "The quick brown fox jumps over the lazy dog and keeps running across the field.", true);
    }

    if (tokens.size() < 6) {
        fprintf(stderr, "prompt is too short for this test\n");
        return 1;
    }

    const llama_seq_id seq_id = 0;
    const int n_base = (int) tokens.size() / 2;

    llama_memory_t mem = llama_get_memory(ctx.get());
    if (!mem) {
        printf("model has no memory - nothing to checkpoint\n");
        return 0;
    }

    const bool is_hybrid    = llama_model_is_hybrid(model);
    const bool is_recurrent = llama_model_is_recurrent(model);
    const int  n_swa        = llama_model_n_swa(model);

    printf("model family: hybrid = %d, recurrent = %d, n_swa = %d\n", is_hybrid, is_recurrent, n_swa);

    // reference run: decode the whole prompt in two batches
    int n_past = 0;
    if (!common_prompt_batch_decode(ctx.get(), tokens, n_base, n_past, params.n_batch, {}, false)) {
        return 1;
    }

    const llama_pos base_pos = n_past - 1;
    const std::vector<uint8_t> snapshot = get_state(ctx.get(), seq_id);
    if (snapshot.empty()) {
        fprintf(stderr, "partial checkpoint is empty\n");
        return 1;
    }

    if (!common_prompt_batch_decode(
                ctx.get(), tokens, (int) tokens.size() - n_base, n_past, params.n_batch, {}, false)) {
        return 1;
    }

    const std::vector<uint8_t> expected  = get_state (ctx.get(), seq_id);
    const std::vector<float>   logits_ref = get_logits(ctx.get(), model);

    // --- B) partial delta round-trip, when the memory supports it -----------------------------

    const size_t delta_size = llama_state_seq_get_delta_ext(ctx.get(), nullptr, 0, seq_id, FLAGS, base_pos);
    if (delta_size > 0) {
        std::vector<uint8_t> delta(delta_size);
        if (llama_state_seq_get_delta_ext(ctx.get(), delta.data(), delta.size(), seq_id, FLAGS, base_pos) != delta.size()) {
            fprintf(stderr, "failed to save the partial delta state\n");
            return 1;
        }

        if (!set_state(ctx.get(), snapshot, seq_id)) {
            fprintf(stderr, "failed to restore the partial checkpoint\n");
            return 1;
        }
        if (llama_state_seq_apply_delta(ctx.get(), delta.data(), delta.size(), seq_id, FLAGS, base_pos) != 0) {
            fprintf(stderr, "failed to apply the partial delta state\n");
            return 1;
        }

        if (get_state(ctx.get(), seq_id) != expected) {
            fprintf(stderr, "partial delta round-trip mismatch\n");
            return 1;
        }

        printf("partial delta round-trip passed: base = %zu, delta = %zu, full = %zu\n",
                snapshot.size(), delta.size(), expected.size());
    } else {
        printf("partial delta not supported by this memory - the server falls back to a full checkpoint\n");
    }

    // --- A) restore + seq_rm rollback + replay ------------------------------------------------

    // the checkpoint goes back first: it puts the recurrent state (if any) back at base_pos, which
    // is what makes the seq_rm below legal for hybrid and recurrent memory
    if (!set_state(ctx.get(), snapshot, seq_id)) {
        fprintf(stderr, "failed to restore the partial checkpoint\n");
        return 1;
    }
    if (!llama_memory_seq_rm(mem, seq_id, base_pos + 1, -1)) {
        fprintf(stderr, "failed to roll the memory back to pos %d\n", base_pos);
        return 1;
    }

    n_past = n_base;
    if (!common_prompt_batch_decode(
                ctx.get(), tokens, (int) tokens.size() - n_base, n_past, params.n_batch, {}, false)) {
        return 1;
    }

    const std::vector<float> logits_new = get_logits(ctx.get(), model);
    if (logits_new.size() != logits_ref.size()) {
        fprintf(stderr, "logits size mismatch\n");
        return 1;
    }

    float max_diff = 0.0f;
    for (size_t i = 0; i < logits_ref.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(logits_ref[i] - logits_new[i]));
    }

    if (argmax(logits_ref) != argmax(logits_new) || max_diff > 1e-2f) {
        fprintf(stderr, "replay after a partial checkpoint diverged: argmax %d vs %d, max diff = %f\n",
                argmax(logits_ref), argmax(logits_new), max_diff);
        return 1;
    }

    printf("partial checkpoint replay passed: checkpoint = %zu bytes, max logit diff = %f\n",
            snapshot.size(), max_diff);

    return 0;
}
