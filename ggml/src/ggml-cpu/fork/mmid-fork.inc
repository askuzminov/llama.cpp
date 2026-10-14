// fork: GGML_CPU_MMID_FLAT=<rows> (1: 64, off by default). A MUL_MAT_ID of a few tokens per expert (a decode step of the
// experts in RAM) by default splits the rows of each expert into nth pieces, piece i to thread i, expert after expert:
// every thread does the same work, and a thread on a slower core (an E-core, a busy hyper-thread sibling) holds the
// node until it is done. Here the rows of all the selected experts go out in chunks of <rows> from one counter of the
// threadpool, so a fast thread takes more chunks, and a chunk is a longer run of one matrix. Included by ggml-cpu.c
// before ggml_compute_forward_mul_mat_id

// the experts of a node with at most this many rows each take the flat split
#define GGML_CPU_MMID_FLAT_MAX_ROWS 16

static int ggml_cpu_mmid_flat_rows_fork(void) {
    static int rows = -1;
    if (rows < 0) {
        const char * s = getenv("GGML_CPU_MMID_FLAT");
        int v = s ? atoi(s) : 0;
        v = v <= 0 ? 0 : v == 1 ? 64 : v;
        rows = (v + 15) / 16 * 16; // whole blocks of ggml_compute_forward_mul_mat_id_one_chunk
    }
    return rows;
}

// after the barrier of ggml_compute_forward_mul_mat_id, false when the node takes the default split. Thread 0 set the
// chunk counter of the threadpool to nth before that barrier
static bool ggml_cpu_mmid_flat_fork(
        const struct ggml_compute_params * params,
        struct ggml_tensor * dst,
        const int64_t * matrix_row_counts,
        const struct mmid_row_mapping * matrix_rows,
        const void * wdata,
        size_t row_size,
        bool src1_cont) {
    const int rows = ggml_cpu_mmid_flat_rows_fork();
    if (rows <= 0) {
        return false;
    }

    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
    const struct ggml_tensor * ids  = dst->src[2];

    const int64_t n_as = src0->ne[2];
    const int64_t ne01 = src0->ne[1];

    int64_t n_used = 0;
    for (int64_t a = 0; a < n_as; ++a) {
        if (matrix_row_counts[a] > GGML_CPU_MMID_FLAT_MAX_ROWS) {
            return false;
        }
        n_used += matrix_row_counts[a] > 0;
    }

    const int64_t n_chunk = (ne01 + rows - 1) / rows; // chunks per expert
    const int64_t total   = n_used * n_chunk;

    // the counter only grows, so each thread walks the experts forward: a = the expert of item cur, k its rank
    int64_t a = -1;
    int64_t k = -1;
    for (int64_t cur = params->ith; cur < total; cur = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        const int64_t want = cur / n_chunk;
        while (k < want) {
            a++;
            while (a < n_as && matrix_row_counts[a] == 0) {
                a++;
            }
            k++;
        }
        const int64_t ir0_start = (cur % n_chunk) * rows;
        const int64_t ir0_end   = MIN(ir0_start + rows, ne01);
        ggml_compute_forward_mul_mat_id_one_chunk(dst, src0, src1, ids, a, ir0_start, ir0_end, 0, matrix_row_counts[a],
                (const char *) src0->data + a*src0->nb[2], matrix_rows, row_size, src1_cont, wdata);
    }
    return true;
}
