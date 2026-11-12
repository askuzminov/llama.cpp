// fork: a batch with a token in two sequences on a model with PLE n-gram embeddings (qwen4exp). The input of the
// n-gram embeddings takes the preceding tokens of one sequence and asserts n_seq_id == 1 in both graph copies; since
// review 10.11 llama_decode refuses such a batch (-1) instead of aborting, and the context stays usable.
//
// usage: test-ple-shared-token-fork -m <dir>/qwen4exp-moe.gguf

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>

int main(int argc, char ** argv) {
    std::string path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[i + 1];
        }
    }
    if (path.empty()) {
        fprintf(stderr, "usage: %s -m <qwen4exp model with PLE n-gram embeddings>\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path.c_str());
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 256;
    cparams.n_batch    = 64;
    cparams.n_ubatch   = 64;
    cparams.n_seq_max  = 2;
    cparams.kv_unified = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "cannot create a context\n");
        return 1;
    }

    // a prefix shared by both sequences: one token in sequences 0 and 1
    llama_batch batch = llama_batch_init(4, 0, 2);
    batch.n_tokens     = 1;
    batch.token[0]     = 1;
    batch.pos[0]       = 0;
    batch.n_seq_id[0]  = 2;
    batch.seq_id[0][0] = 0;
    batch.seq_id[0][1] = 1;
    batch.logits[0]    = 1;
    const int ret_shared = llama_decode(ctx, batch);

    // the same token in one sequence goes through
    batch.n_seq_id[0] = 1;
    const int ret_one = llama_decode(ctx, batch);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    printf("shared token: %d (expected -1), one sequence: %d (expected 0)\n", ret_shared, ret_one);
    return ret_shared == -1 && ret_one == 0 ? 0 : 1;
}
