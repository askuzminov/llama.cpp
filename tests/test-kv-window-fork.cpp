// fork: the window cache of the MTP head (llama_kv_cache::swa_room_on_fork, src/fork/llama-kv-window-fork.cpp). A cache
// of window + one ubatch cells, as llama_mtp_window_kv makes it, holds a conversation longer than the window, then an
// image whose tokens share one temporal position (M-RoPE) and outnumber the spare ubatch, then text after it. With the
// upstream rule (the flag off) the rest of the image finds no cell - the server failed at that image on every retry
// (04.11). With the flag the oldest cells of the sequence give way: the whole image is placed and the text goes on.
// Metadata only: the ubatches are placed and applied, no graph runs.
//
// usage: test-kv-window-fork -m <dir>/qwen4exp-moe.gguf

#include "llama.h"

#include "../src/llama-batch.h"
#include "../src/llama-kv-cache.h"
#include "../src/llama-model.h"
#include "../src/llama-vocab.h"

#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

static int g_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); g_fail++; } } while (0)

static constexpr uint32_t N_SWA    = 64;
static constexpr uint32_t N_UBATCH = 32;
static constexpr uint32_t N_POS    = 4; // M-RoPE: temporal, y, x, other

struct tok {
    llama_pos t, y, x;
};

// one batch of embeddings of sequence 0 into the cache, every ubatch placed and applied as a decode does it
static bool place(llama_kv_cache & kv, uint32_t n_embd, const std::vector<tok> & toks) {
    llama_batch_ext b(toks.size(), n_embd, n_embd, 1, &kv, 0, N_POS);
    std::vector<float> row(n_embd, 0.0f);
    for (const auto & t : toks) {
        const int32_t idx = b.add_token(0);
        if (idx < 0) {
            return false;
        }
        const llama_pos pos[N_POS] = { t.t, t.y, t.x, 0 };
        if (!b.set_token_embd(idx, { row.data(), 1, n_embd }) || !b.set_token_pos(idx, pos) || !b.set_output(idx, false)) {
            return false;
        }
    }

    llama_vocab vocab;
    llama_batch_allocr balloc(N_POS);
    if (!balloc.init(b, vocab, false)) {
        fprintf(stderr, "the batch was rejected\n");
        return false;
    }

    auto mctx = kv.init_batch(balloc, N_UBATCH, false);
    if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
        return false;
    }
    do {
        if (!mctx->apply()) {
            return false;
        }
    } while (mctx->next());
    return true;
}

static std::vector<tok> text(llama_pos p0, llama_pos p1) {
    std::vector<tok> res;
    for (llama_pos p = p0; p < p1; ++p) {
        res.push_back({ p, p, p });
    }
    return res;
}

// an image of w x h tokens at temporal position t
static std::vector<tok> image(llama_pos t, int w, int h) {
    std::vector<tok> res;
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            res.push_back({ t, t + y, t + x });
        }
    }
    return res;
}

int main(int argc, char ** argv) {
    const char * path = nullptr;
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], "-m") == 0) {
            path = argv[++i];
        }
    }
    if (path == nullptr) {
        fprintf(stderr, "usage: %s -m <dir>/qwen4exp-moe.gguf\n", argv[0]);
        return 1;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(path, mp);
    if (model == nullptr) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }

    const llama_hparams & hparams = model->hparams;
    const uint32_t        n_embd  = hparams.n_embd;

    // one attention layer is enough for the cells
    int32_t il_attn = -1;
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.is_recr(il)) {
            il_attn = (int32_t) il;
            break;
        }
    }
    CHECK(il_attn >= 0);

    auto make = [&](bool room) {
        auto kv = std::make_unique<llama_kv_cache>(*model, false, hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true,
                N_SWA + N_UBATCH, 1, N_SWA, LLAMA_SWA_TYPE_STANDARD, nullptr,
                [&](int32_t il) { return il == il_attn; }, nullptr, nullptr, "test");
        kv->swa_reuse_shared_fork = room;
        kv->swa_room_on_fork      = room;
        return kv;
    };

    auto cells_at = [](const llama_kv_cache & kv, llama_pos t) {
        uint32_t n = 0;
        const auto & pos = kv.get_cells(0).seq_pos_get(0);
        for (const auto & [p, idx] : pos) {
            n += p == t;
        }
        return n;
    };

    // the upstream rule: 100 positions of text, then a 8 x 5 image at 100: the first ubatch of the image takes the 32
    // masked cells, the other 8 tokens find one
    {
        auto kv = make(false);
        CHECK(place(*kv, n_embd, text(0, 100)));
        CHECK(!place(*kv, n_embd, image(100, 8, 5)));
    }

    // the fork: the oldest cells of the sequence give way for the image, the text after it goes on token by token
    {
        auto kv = make(true);
        CHECK(place(*kv, n_embd, text(0, 100)));
        CHECK(place(*kv, n_embd, image(100, 8, 5)));
        CHECK(cells_at(*kv, 100) == 40);
        CHECK(kv->seq_pos_max(0) == 100);
        CHECK(kv->get_n_kv_used() <= N_SWA + N_UBATCH);
        // the text before the image keeps its newest positions
        CHECK(cells_at(*kv, 99) == 1);

        // M-RoPE: the text after a 8 x 5 image starts 8 positions on
        bool ok = true;
        for (llama_pos p = 108; p < 108 + 3*(llama_pos) N_SWA && ok; ++p) {
            ok = place(*kv, n_embd, text(p, p + 1));
        }
        CHECK(ok);
        CHECK(kv->seq_pos_max(0) == 108 + 3*(llama_pos) N_SWA - 1);
        // the window has passed the image: the last N_SWA positions are all there
        for (llama_pos p = 108 + 2*(llama_pos) N_SWA; p < 108 + 3*(llama_pos) N_SWA; ++p) {
            CHECK(cells_at(*kv, p) == 1);
        }

        // a second image, larger than the whole spare ubatch twice over
        const llama_pos t2 = 108 + 3*(llama_pos) N_SWA;
        CHECK(place(*kv, n_embd, image(t2, 10, 8)));
        CHECK(cells_at(*kv, t2) == 80);
    }

    llama_model_free(model);
    llama_backend_free();

    printf("test-kv-window-fork: %s\n", g_fail == 0 ? "OK" : "FAILED");
    return g_fail == 0 ? 0 : 1;
}
