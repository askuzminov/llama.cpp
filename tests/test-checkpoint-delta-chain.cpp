#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <vector>

static std::vector<uint8_t> get_state(llama_context * ctx, llama_seq_id seq_id) {
    const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, 0);
    std::vector<uint8_t> data(size);
    const size_t written = llama_state_seq_get_data_ext(ctx, data.data(), data.size(), seq_id, 0);
    GGML_ASSERT(written == data.size());
    return data;
}

static std::vector<uint8_t> get_delta(llama_context * ctx, llama_seq_id seq_id, llama_pos base_pos) {
    const size_t size = llama_state_seq_get_delta_ext(ctx, nullptr, 0, seq_id, 0, base_pos);
    std::vector<uint8_t> data(size);
    const size_t written = llama_state_seq_get_delta_ext(ctx, data.data(), data.size(), seq_id, 0, base_pos);
    GGML_ASSERT(written == data.size());
    return data;
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
            vocab, "One two three four five six seven eight nine ten eleven twelve.", true);
    if (tokens.size() < 6) {
        return 1;
    }

    const llama_seq_id seq_id = 0;
    const int n0 = (int) tokens.size() / 3;
    const int n1 = (int) tokens.size() / 3;
    const int n2 = (int) tokens.size() - n0 - n1;
    const std::vector<llama_token> tokens0(tokens.begin(), tokens.begin() + n0);
    const std::vector<llama_token> tokens1(tokens.begin(), tokens.begin() + n0 + n1);
    int n_past = 0;

    if (!common_prompt_batch_decode(ctx.get(), tokens0, n0, n_past, params.n_batch, {}, false)) {
        return 1;
    }
    const std::vector<uint8_t> base = get_state(ctx.get(), seq_id);
    const llama_pos pos0 = n_past - 1;

    if (!common_prompt_batch_decode(ctx.get(), tokens1, n1, n_past, params.n_batch, {}, false)) {
        return 1;
    }
    const std::vector<uint8_t> delta1 = get_delta(ctx.get(), seq_id, pos0);
    const llama_pos pos1 = n_past - 1;

    if (!common_prompt_batch_decode(ctx.get(), tokens, n2, n_past, params.n_batch, {}, false)) {
        return 1;
    }
    const std::vector<uint8_t> delta2 = get_delta(ctx.get(), seq_id, pos1);
    const std::vector<uint8_t> expected = get_state(ctx.get(), seq_id);

    if (llama_state_seq_set_data_ext(ctx.get(), base.data(), base.size(), seq_id, 0) != base.size()) {
        return 1;
    }
    if (llama_state_seq_apply_delta(ctx.get(), delta1.data(), delta1.size(), seq_id, 0, pos0) != 0) {
        return 1;
    }
    if (llama_state_seq_apply_delta(ctx.get(), delta2.data(), delta2.size(), seq_id, 0, pos1) != 0) {
        return 1;
    }

    if (get_state(ctx.get(), seq_id) != expected) {
        fprintf(stderr, "delta chain round-trip mismatch\n");
        return 1;
    }

    printf("delta chain passed: base=%zu delta1=%zu delta2=%zu full=%zu\n",
            base.size(), delta1.size(), delta2.size(), expected.size());
    return 0;
}
