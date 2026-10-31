// fork: the token count from which the lightning indexer of 4 heads runs the tile kernel, included by
// lightning-indexer.cu. the tile kernel reads a block of keys once for up to 8 tokens; upstream takes it from 8 tokens and
// runs smaller batches on the vector kernel, which reads the keys again for every token. the MTP checks of qwen4exp are
// 3-4 tokens: on the 3090 (test-micro-fork, 31.10, 65536 blocks) the vector kernel took 70 / 102 / 135 us at 2 / 3 / 4
// tokens and the tile kernel 66 us at 8, per layer. the tile kernel zeroes the tokens past the batch and skips their
// rows. GGML_CUDA_LID_TILE_MIN: the tokens from which the tile kernel runs (2; 8 as upstream)

#include <cstdlib>

static int ggml_cuda_fork_lid_tile_min() {
    static const int n = [] {
        const char * env = getenv("GGML_CUDA_LID_TILE_MIN");
        const int v = env ? atoi(env) : 2;
        return v > 0 ? v : 1;
    }();
    return n;
}
