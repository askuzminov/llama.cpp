// fork: the graph cache (LLAMA_GRAPH_CACHE, default 8) restores the split and allocation of an earlier graph from a
// scheduler snapshot when only the buffers did not grow since (ggml_backend_sched_snapshot_restore checks n_realloc).
// Review 10.11 (B2.1 #2): between the save and the restore another graph can re-plan the same buffers without growing
// them (other backends of its nodes: the offload of the weights of host layers depends on the batch size), and the
// restored graph then runs on the addresses of its own plan. This test alternates such graphs, batches of 1 and 64
// tokens with half of the layers on the host, and wants the logits of a run with the graph cache bit for bit equal to
// those of a run without graph reuse (LLAMA_GRAPH_REUSE_DISABLE=1): every graph then is split and allocated anew.
//
// usage: test-graph-cache-fork -m <model> [-dev <device> -ngl <layers on the device, fewer than the model has>]

#include "llama.h"
#include "ggml-backend.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> run(llama_model * model, bool reuse, int & n_reuse_hits) {
    if (reuse) {
        unsetenv("LLAMA_GRAPH_REUSE_DISABLE");
    } else {
        setenv("LLAMA_GRAPH_REUSE_DISABLE", "1", 1);
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx      = 1024;
    cparams.n_batch    = 128;
    cparams.n_ubatch   = 128;
    cparams.n_seq_max  = 1;
    cparams.op_offload = true;
    cparams.no_perf    = false;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "cannot create a context\n");
        exit(1);
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    // graphs of 1 and 64 tokens alternate, with repeats: each shape comes back after the other one was planned
    const int sizes[] = { 64, 1, 1, 64, 1, 64, 64, 1, 3, 64, 1, 3, 1, 64 };
    std::vector<float> out;
    int pos = 0;
    for (int n : sizes) {
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;
        for (int i = 0; i < n; ++i) {
            batch.token[i]     = (llama_token) ((pos * 7919 + 13) % (n_vocab - 10) + 5);
            batch.pos[i]       = pos++;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = i == n - 1;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed\n");
            exit(1);
        }
        const float * logits = llama_get_logits_ith(ctx, n - 1);
        out.insert(out.end(), logits, logits + n_vocab);
        llama_batch_free(batch);
    }
    n_reuse_hits = llama_perf_context(ctx).n_reused;
    llama_free(ctx);
    return out;
}

int main(int argc, char ** argv) {
    std::string path;
    std::string dev_name;
    int ngl = -1;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[i + 1];
        } else if (strcmp(argv[i], "-dev") == 0) {
            dev_name = argv[i + 1];
        } else if (strcmp(argv[i], "-ngl") == 0) {
            ngl = atoi(argv[i + 1]);
        }
    }
    if (path.empty()) {
        fprintf(stderr, "usage: %s -m <model> [-dev <device>] [-ngl <layers on the device>]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    ggml_backend_dev_t devs[2] = { nullptr, nullptr };
    if (!dev_name.empty()) {
        devs[0] = ggml_backend_dev_by_name(dev_name.c_str());
        if (devs[0] == nullptr) {
            fprintf(stderr, "no device %s\n", dev_name.c_str());
            return 1;
        }
        mparams.devices = devs;
    }
    mparams.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(path.c_str(), mparams);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path.c_str());
        return 1;
    }
    int hits_on = 0;
    int hits_off = 0;
    const std::vector<float> on  = run(model, true, hits_on);
    const std::vector<float> off = run(model, false, hits_off);

    size_t n_diff = 0;
    double max_diff = 0.0;
    for (size_t i = 0; i < on.size(); ++i) {
        const double d = std::fabs((double) on[i] - (double) off[i]);
        n_diff += on[i] != off[i];
        max_diff = std::max(max_diff, d);
    }
    printf("graph cache: %d graphs reused, without reuse: %d; logits of %zu steps: %zu values differ, max diff %g\n",
           hits_on, hits_off, on.size() / llama_vocab_n_tokens(llama_model_get_vocab(model)), n_diff, max_diff);

    llama_model_free(model);
    llama_backend_free();
    return n_diff == 0 && hits_on > 0 && hits_off == 0 ? 0 : 1;
}
