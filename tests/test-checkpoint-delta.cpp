#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <vector>

static std::vector<uint8_t> get_state(llama_context * ctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, 0);
    std::vector<uint8_t> data(size);
    const size_t written = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), seq_id, 0);
    GGML_ASSERT(written == data.size());
    return data;
}

static bool set_state(llama_context * ctx, const std::vector<uint8_t> & data, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, data.data(), data.size(), seq_id, 0) == data.size();
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
    std::vector<llama_token> tokens = common_tokenize(
            vocab, "The quick brown fox jumps over the lazy dog and keeps running.", true);
    if (tokens.size() < 4) {
        return 1;
    }

    const llama_seq_id seq_id = 0;
    const int n_base = (int) tokens.size() / 2;
    const std::vector<llama_token> tokens_base(tokens.begin(), tokens.begin() + n_base);
    int n_past = 0;

    if (!common_prompt_batch_decode(ctx.get(), tokens_base, n_base, n_past, params.n_batch, {}, false)) {
        return 1;
    }

    const llama_pos base_pos = n_past - 1;
    const std::vector<uint8_t> base = get_state(ctx.get(), seq_id);

    if (!common_prompt_batch_decode(
                ctx.get(), tokens, (int) tokens.size() - n_base, n_past, params.n_batch, {}, false)) {
        return 1;
    }

    const std::vector<uint8_t> expected = get_state(ctx.get(), seq_id);
    const size_t delta_size = llama_state_seq_get_delta_ext(ctx.get(), nullptr, 0, seq_id, 0, base_pos);
    if (delta_size == 0) {
        fprintf(stderr, "failed to measure delta state\n");
        return 1;
    }

    std::vector<uint8_t> delta(delta_size);
    if (llama_state_seq_get_delta_ext(ctx.get(), delta.data(), delta.size(), seq_id, 0, base_pos) != delta.size()) {
        fprintf(stderr, "failed to save delta state\n");
        return 1;
    }

    if (!set_state(ctx.get(), base, seq_id)) {
        fprintf(stderr, "failed to restore base state\n");
        return 1;
    }
    if (llama_state_seq_apply_delta(ctx.get(), delta.data(), delta.size(), seq_id, 0, base_pos) != 0) {
        fprintf(stderr, "failed to apply delta state\n");
        return 1;
    }

    const std::vector<uint8_t> actual = get_state(ctx.get(), seq_id);
    if (actual != expected) {
        fprintf(stderr, "delta round-trip mismatch\n");
        return 1;
    }

    printf("delta round-trip passed: base=%zu delta=%zu full=%zu\n",
            base.size(), delta.size(), expected.size());
    return 0;
}
