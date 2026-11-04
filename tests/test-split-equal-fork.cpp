// fork: llama_split_equal tells the server whether drafts of different lengths verify in one pass, and the drafts are
// made even only when they do not (common_speculative_fork_even_drafts). Two sequences, one with 4 tokens (a draft of
// 3) and one with 1, decoded at once with a stream per sequence and with one stream: the graphs of that batch must be 2
// where llama_split_equal says true and 1 where it says false. The qwen4exp hybrid must split equally, so the drafts of
// the server stay even there
//
// usage: test-split-equal-fork --models <the dir of the generated models>

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>

static int g_n_graphs = 0;

// one logits tensor per graph: every token of the batch is an output
static bool eval_cb(ggml_tensor * t, bool ask, void * /*user_data*/) {
    if (ask && strcmp(t->name, "result_output") == 0) {
        g_n_graphs++;
    }
    return false;
}

static bool decode(llama_context * ctx, const int (&n)[2], llama_pos pos0) {
    llama_batch batch = llama_batch_init(n[0] + n[1], 0, 1);
    for (llama_seq_id s = 0; s < 2; ++s) {
        for (int i = 0; i < n[s]; ++i) {
            const int k = batch.n_tokens++;
            batch.token   [k]    = 1 + s + 2*i;
            batch.pos     [k]    = pos0 + i;
            batch.n_seq_id[k]    = 1;
            batch.seq_id  [k][0] = s;
            batch.logits  [k]    = true;
        }
    }
    const int ret = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return ret == 0;
}

int main(int argc, char ** argv) {
    std::string dir;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "--models") == 0) {
            dir = argv[++i];
        }
    }
    if (dir.empty()) {
        fprintf(stderr, "usage: %s --models <dir>\n", argv[0]);
        return 1;
    }

    llama_log_set([](ggml_log_level, const char *, void *) {}, nullptr);
    llama_backend_init();

    // llama_split_equal with one stream and with a stream per sequence
    struct expect {
        const char * model;
        bool         unified;
        bool         separate;
    };
    const expect cases[] = {
        { "llama-dense",         false, true },
        { "gemma2-dense",        false, true }, // iSWA
        { "granitehybrid-dense", true,  true },
        { "mamba2-dense",        true,  true },
        { "qwen4exp-moe",        true,  true },
    };

    int n_fail = 0;
    for (const auto & c : cases) {
        const std::string path = dir + "/" + c.model + ".gguf";
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        llama_model * model = llama_model_load_from_file(path.c_str(), mp);
        if (model == nullptr) {
            fprintf(stderr, "cannot load %s\n", path.c_str());
            return 1;
        }
        for (const bool unified : { true, false }) {
            llama_context_params cp = llama_context_default_params();
            cp.n_ctx       = 512;
            cp.n_batch     = 64;
            cp.n_ubatch    = 64;
            cp.n_seq_max   = 2;
            cp.kv_unified  = unified;
            cp.cb_eval     = eval_cb;
            llama_context * ctx = llama_init_from_model(model, cp);
            if (ctx == nullptr) {
                fprintf(stderr, "cannot create a context for %s\n", c.model);
                return 1;
            }

            const bool split = llama_split_equal(ctx);
            const bool want  = unified ? c.unified : c.separate;

            // a prompt of 3 tokens per sequence, then the verify batch of 4 + 1 tokens
            bool ok = decode(ctx, { 3, 3 }, 0);
            g_n_graphs = 0;
            ok = ok && decode(ctx, { 4, 1 }, 3);

            const bool pass = ok && split == want && g_n_graphs == (split ? 2 : 1);
            printf("%-20s %-8s split_equal %d, graphs of the 4 + 1 batch %d%s\n", c.model, unified ? "unified" : "separate",
                    split, g_n_graphs, pass ? "" : (ok ? "  FAIL" : "  FAIL (decode)"));
            n_fail += pass ? 0 : 1;

            llama_free(ctx);
        }
        llama_model_free(model);
    }

    llama_backend_free();

    return n_fail == 0 ? 0 : 1;
}
