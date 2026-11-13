# the fork's tests, included at the end of tests/CMakeLists.txt: the helpers, fixtures and model paths defined there
# are in scope

llama_build(test-checkpoint-partial.cpp)

# the window KV of the dense MTP head (LLAMA_MTP_WINDOW_KV): the rule is header-only, every platform builds this
llama_build_and_test(test-mtp-window-fork.cpp)
target_include_directories(test-mtp-window-fork PRIVATE ${PROJECT_SOURCE_DIR}/src)

# the counts file of the vocabulary subset of the MTP head (LLAMA_MTP_VOCAB_STATS): the save that adds to the file, the
# files it leaves alone, the default place in the llama.cpp cache; header-only, every platform builds this
llama_build_and_test(test-mtp-vocab-stats-fork.cpp)
target_include_directories(test-mtp-vocab-stats-fork PRIVATE ${PROJECT_SOURCE_DIR}/src)

# the gate of --spec-auto: a calibration per draft type, probabilities that are not a number (common/fork)
llama_build_and_test(test-spec-auto-fork.cpp)

# the fork's knobs of the environment: a value that is not a whole number gives the default, not the 0 of atoi
llama_build_and_test(test-env-fork.cpp)

# the k + 1 largest of the sampler's fast top-k (common/fork/top-k-fork.h): the same heap as one value at a time
llama_build_and_test(test-top-k-fork.cpp)

# the F16 activations the fork's CPU backend takes against quantized weights (MUL_MAT, MUL_MAT_ID) against the same
# values in F32, as a model runs them (the tiled K-quant kernels of x86 builds included); no model
llama_build_and_test(test-cpu-f16-src1-fork.cpp LINK ggml)

# the q8_1 activations of the Vulkan int8 mat-vec in two planes (GGML_VK_MMVQ_PLANES=2) and in one, against double
# math (MUL_MAT of 1-8 columns, MUL_MAT_ID of 1-8 tokens); skipped without a Vulkan device with the int8 dot
llama_build_and_test(test-mmvq-planes-fork.cpp LINK ggml)
set_tests_properties(test-mmvq-planes-fork PROPERTIES SKIP_REGULAR_EXPRESSION "test-mmvq-planes-fork: SKIPPED")
llama_test(test-mmvq-planes-fork NAME test-mmvq-planes-fork-1 ARGS --planes 1)
set_tests_properties(test-mmvq-planes-fork-1 PROPERTIES SKIP_REGULAR_EXPRESSION "test-mmvq-planes-fork: SKIPPED")
# two planes for some weight types only (GGML_VK_MMVQ_PLANES_TYPES), the others one
llama_test(test-mmvq-planes-fork NAME test-mmvq-planes-fork-types ARGS --planes 2 --types q5_1,q8_0)
set_tests_properties(test-mmvq-planes-fork-types PROPERTIES SKIP_REGULAR_EXPRESSION "test-mmvq-planes-fork: SKIPPED")
# the dense q8_0 of the draft checks on the f16 coopmat tile of mul_mm_f16reg.comp (GGML_VK_MM_F16REG_VERIFY), emulated
# here so that devices without coopmat run it too
llama_test(test-mmvq-planes-fork NAME test-mmvq-planes-fork-verify ARGS --planes 1 --verify r16x16/8x1)
set_tests_properties(test-mmvq-planes-fork-verify PROPERTIES SKIP_REGULAR_EXPRESSION "test-mmvq-planes-fork: SKIPPED"
                     ENVIRONMENT "GGML_VK_MM_F16REG=r16x96/8x1;GGML_VK_MM_F16REG_EMULATE=1")

# the qwen4exp indexer timed on one device at the shapes of the model (lightning indexer, the head matmul of the Vulkan
# graph and of the other backends): a benchmark, built but not a ctest; 04-micro.bat runs it on the machines
llama_build(test-micro-fork.cpp LINK ggml)

# the f16 hc read of qwen4exp (LLAMA_HC_F16) and other formats and places of rounding emulated on the CPU, on a deep
# model built from the generated qwen4exp-moe.gguf: a measurement of minutes, built but not a ctest
llama_build(test-hc-round-fork.cpp)

# the elastic buffers of the backends (CUDA VMM, Vulkan sparse buffers) on every GPU device that has them, skipped
# elsewhere; 01-check runs it on the machines (CHECKUNITS)
llama_build_and_test(test-elastic-buffer-fork.cpp)
set_tests_properties(test-elastic-buffer-fork PROPERTIES SKIP_REGULAR_EXPRESSION "test-elastic-buffer-fork: SKIPPED")

# prompt-cache disk spill: block-aligned unbuffered file layout, cold-start adoption, rejection of
# foreign or truncated files. server-context is a static library, so this links on Windows too.
llama_build_and_test(test-prompt-cache-spill.cpp)
target_include_directories(test-prompt-cache-spill PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_link_libraries(test-prompt-cache-spill PRIVATE server-context)
llama_test(test-prompt-cache-spill NAME test-prompt-cache-compat LABEL "model behavior" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-prompt-cache-compat PROPERTIES FIXTURES_REQUIRED test-download-model)

# the checkpoint at the end of a long system prompt (LLAMA_CKPT_SYSTEM_BREAK): where the system block ends and the
# eviction that keeps its checkpoint; no model
llama_build_and_test(test-ckpt-system-fork.cpp)
target_include_directories(test-ckpt-system-fork PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_link_libraries(test-ckpt-system-fork PRIVATE server-context)

# the row cache test calls internal functions of libllama. a Windows DLL exports LLAMA_API only, so there the test is
# built from the sources of the row cache and llama_file, with the two helpers of llama-impl.cpp that they use
# (test-row-cache-impl-fork.cpp). LLAMA_FORK_ROW_CACHE_STANDALONE=ON builds it that way on any platform
if (WIN32 AND BUILD_SHARED_LIBS)
    set(LLAMA_FORK_ROW_CACHE_STANDALONE_DEFAULT ON)
else()
    set(LLAMA_FORK_ROW_CACHE_STANDALONE_DEFAULT OFF)
endif()
option(LLAMA_FORK_ROW_CACHE_STANDALONE "fork: build test-row-cache from the sources it tests" ${LLAMA_FORK_ROW_CACHE_STANDALONE_DEFAULT})

if (LLAMA_FORK_ROW_CACHE_STANDALONE)
    add_executable(test-row-cache
        test-row-cache.cpp
        test-row-cache-impl-fork.cpp
        ${PROJECT_SOURCE_DIR}/src/fork/llama-row-cache-fork.cpp
        ${PROJECT_SOURCE_DIR}/src/llama-mmap.cpp)
    target_include_directories(test-row-cache PRIVATE ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
    target_link_libraries(test-row-cache PRIVATE ggml)
    target_compile_features(test-row-cache PRIVATE cxx_std_17)
    add_test(NAME test-row-cache COMMAND $<TARGET_FILE:test-row-cache>)
    set_property(TEST test-row-cache PROPERTY LABELS main)
elseif (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # internal functions not exported with LLAMA_API (as the upstream block of the same condition)
    llama_build_and_test(test-row-cache.cpp)
    target_include_directories(test-row-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)
endif()

# QSA with several sequences (registered below where the matrix models exist): built everywhere, as it needs only the
# public API. a Windows DLL build has no generated models, so it runs there only by hand, on a qwen4exp-moe.gguf
# generated elsewhere: test-qsa-multi-fork -m <model> --dense-out <file>, then the same with --dense-in <file>
llama_build(test-qsa-multi-fork.cpp)
target_include_directories(test-qsa-multi-fork PRIVATE ${PROJECT_SOURCE_DIR}/src)

# the state of one sequence into a context of another KV stream count (registered below where the matrix models exist)
llama_build(test-state-streams-fork.cpp)

# the device cache of the host experts with --phase-mem (keep, fill, kv) against the release of all slots at every
# phase switch: built everywhere (llama.h, gguf.h), registered where the matrix models exist, skips without a GPU
llama_build(test-moe-cache-fork.cpp)
llama_build(test-kv-elastic-streams-fork.cpp)

# llama_split_equal against the graphs of a batch of drafts of different lengths (registered below where the matrix
# models exist): only llama.h
llama_build(test-split-equal-fork.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # the models of this matrix come from test-llama-archs (the generate-models fixture), which upstream builds under
    # the same condition: on a Windows DLL build there are none

    # the window cache of the MTP head with an image that does not move the window (swa_room_on_fork): the cells of a
    # llama_kv_cache made by the test, no graph; internal classes, so only where they link
    llama_build(test-kv-window-fork.cpp LINK llama)
    llama_test(test-kv-window-fork LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    set_tests_properties(test-kv-window-fork PROPERTIES FIXTURES_REQUIRED generate-models)

    # the flatten of the deferred rollback of the gated delta net against the plain net: the saved state after a short
    # batch and a rollback, byte for byte, and the logits of the next token (rec_flatten, LLAMA_GDN_LAZY)
    llama_build(test-gdn-flatten-fork.cpp)
    llama_test(test-gdn-flatten-fork LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    set_tests_properties(test-gdn-flatten-fork PROPERTIES FIXTURES_REQUIRED generate-models)

    # partial (PARTIAL_ONLY) context checkpoints, one test per memory family:
    # plain attention, iSWA, hybrid (attention + recurrent), pure recurrent and the hybrid of qwen4exp (QSA + GDN)
    foreach(CKPT_MODEL llama-dense gemma2-dense phi3-dense granitehybrid-dense nemotron_h-dense falcon-h1-dense mamba2-dense qwen4exp-moe)
        llama_test(
            test-checkpoint-partial
            NAME test-checkpoint-partial-${CKPT_MODEL}
            LABEL main
            ARGS -m "${MODEL_DIR}/${CKPT_MODEL}.gguf"
        )
        set_tests_properties(test-checkpoint-partial-${CKPT_MODEL} PROPERTIES
            FIXTURES_REQUIRED generate-models
        )
    endforeach()

    # the same with sparse QSA in the replay: top-k 16 instead of the 131072 of the file, which selects every cell
    llama_test(
        test-checkpoint-partial
        NAME test-checkpoint-partial-qwen4exp-moe-qsa
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --tokens 256 --override-kv qwen4exp.attention.indexer.top_k=int:16
    )
    set_tests_properties(test-checkpoint-partial-qwen4exp-moe-qsa PROPERTIES
        FIXTURES_REQUIRED generate-models
    )

    # a token in two sequences on the PLE n-gram embeddings of qwen4exp: llama_decode refuses it, no abort (review 10.11)
    llama_build_and_test(test-ple-shared-token-fork.cpp LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    set_tests_properties(test-ple-shared-token-fork PROPERTIES FIXTURES_REQUIRED generate-models)

    llama_test(test-graph-cache-fork LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    set_tests_properties(test-graph-cache-fork PROPERTIES FIXTURES_REQUIRED generate-models)

    # the graph cache: graphs of 1 and 64 tokens alternate, the logits equal those without graph reuse (review 10.11). on the
# CPU alone; with a device and part of the layers on it, by hand: -dev <device> -ngl <n>
llama_build(test-graph-cache-fork.cpp)

# which memories verify drafts of different lengths in one pass: the even drafts of the server depend on it
    llama_test(test-split-equal-fork LABEL main ARGS --models "${MODEL_DIR}")
    set_tests_properties(test-split-equal-fork PROPERTIES FIXTURES_REQUIRED generate-models)

    # QSA with several sequences in one unified cache (the server's slots): the groupings carried across the rollbacks
    # of drafts, checked by LLAMA_QSA_CHECK, the mask from positions per sequence, and the logits of every sequence
    # against the same sequence alone, on the qwen4exp model of the matrix. the mask from positions must give the logits
    # of the dense mask bit for bit, and LLAMA_QSA_MASK_POS is read once per process: a run with the dense mask writes
    # its logits first. the binary is built on every platform (below), it needs only llama.h
    llama_test(test-state-streams-fork LABEL main ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    set_tests_properties(test-state-streams-fork PROPERTIES FIXTURES_REQUIRED generate-models)

    set(QSA_MULTI_DENSE "${CMAKE_CURRENT_BINARY_DIR}/test-qsa-multi-fork-dense.bin")
    llama_test(test-qsa-multi-fork NAME test-ple-read-error-fork LABEL "model behavior" ARGS --ple-read-error -m "${MODEL_DIR}/qwen4exp-moe.gguf")
    set_tests_properties(test-ple-read-error-fork PROPERTIES FIXTURES_REQUIRED generate-models)
    llama_test(
        test-qsa-multi-fork
        NAME test-qsa-multi-fork-dense
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --dense-out "${QSA_MULTI_DENSE}"
    )
    set_tests_properties(test-qsa-multi-fork-dense PROPERTIES
        FIXTURES_REQUIRED generate-models
        FIXTURES_SETUP qsa-multi-dense
    )
    llama_test(
        test-qsa-multi-fork
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --dense-in "${QSA_MULTI_DENSE}"
    )
    set_tests_properties(test-qsa-multi-fork PROPERTIES
        FIXTURES_REQUIRED "generate-models;qsa-multi-dense"
    )

    # the elastic KV cache with a stream per slot (no --kv-unified): the static cache writes the logits of the scenario,
    # an elastic one gives them bit for bit with reads past a capacity as zeros (Vulkan) and with them mapped (CUDA)
    set(KV_ELASTIC_STREAMS_REF "${CMAKE_CURRENT_BINARY_DIR}/test-kv-elastic-streams-fork-ref.bin")
    llama_test(
        test-kv-elastic-streams-fork
        NAME test-kv-elastic-streams-fork-ref
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --out "${KV_ELASTIC_STREAMS_REF}"
    )
    set_tests_properties(test-kv-elastic-streams-fork-ref PROPERTIES
        FIXTURES_REQUIRED generate-models
        FIXTURES_SETUP kv-elastic-streams-ref
    )
    foreach(KV_ELASTIC_MODE zero map)
        llama_test(
            test-kv-elastic-streams-fork
            NAME test-kv-elastic-streams-fork-${KV_ELASTIC_MODE}
            LABEL main
            ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --ref "${KV_ELASTIC_STREAMS_REF}" --mode ${KV_ELASTIC_MODE}
        )
        set_tests_properties(test-kv-elastic-streams-fork-${KV_ELASTIC_MODE} PROPERTIES
            FIXTURES_REQUIRED "generate-models;kv-elastic-streams-ref"
        )
    endforeach()
    llama_test(
        test-kv-elastic-streams-fork
        NAME test-kv-elastic-streams-fork-reclaim
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --mode reclaim
    )
    set_tests_properties(test-kv-elastic-streams-fork-reclaim PROPERTIES FIXTURES_REQUIRED generate-models)

    # the moe-cache through the phase switches: the reference run writes the model it builds from the qwen4exp of the
    # matrix (32 experts, no per-expert scales) and its logprobs, each mode compares with them in a run of its own
    set(MOE_CACHE_REF "${CMAKE_CURRENT_BINARY_DIR}/test-moe-cache-fork-ref.bin")
    llama_test(
        test-moe-cache-fork
        NAME test-moe-cache-fork-ref
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --out "${MOE_CACHE_REF}"
    )
    # alone: the plan of the phases takes the free memory of the GPU (llama_context::phase_prompt), which another test
    # on the GPU changes, and keep / upload compare bit for bit
    set_tests_properties(test-moe-cache-fork-ref PROPERTIES
        FIXTURES_REQUIRED generate-models
        FIXTURES_SETUP moe-cache-ref
        SKIP_REGULAR_EXPRESSION "test-moe-cache-fork: SKIPPED"
        RUN_SERIAL TRUE
    )
    foreach(MOE_CACHE_MODE keep fill kv upload state)
        llama_test(
            test-moe-cache-fork
            NAME test-moe-cache-fork-${MOE_CACHE_MODE}
            LABEL main
            ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --ref "${MOE_CACHE_REF}" --mode ${MOE_CACHE_MODE}
        )
        set_tests_properties(test-moe-cache-fork-${MOE_CACHE_MODE} PROPERTIES
            FIXTURES_REQUIRED "generate-models;moe-cache-ref"
            SKIP_REGULAR_EXPRESSION "test-moe-cache-fork: SKIPPED"
            RUN_SERIAL TRUE
        )
    endforeach()
endif()

# rejection sampling of speculative drafts against a sharp target, one chi-square over all tokens (the upstream test,
# on the flat distribution of its tiny model, misses a residual from p and swapped p_x / q_x)
llama_build_and_test(test-spec-rejection-fork.cpp ARGS -m "${SPEC_TARGET_DIR}/llama-dense.gguf")
set_tests_properties(test-spec-rejection-fork PROPERTIES FIXTURES_REQUIRED speculative-model)

# a sampling drafter reports q, the distribution its drafted tokens are drawn from: their counts against it (draft-simple,
# the llama of the speculative fixture as target and draft)
llama_build_and_test(test-spec-draft-q-fork.cpp ARGS -m "${SPEC_TARGET_DIR}/llama-dense.gguf")
set_tests_properties(test-spec-draft-q-fork PROPERTIES FIXTURES_REQUIRED speculative-model)

# Test checkpoint delta encoding/decoding
llama_build_and_test(test-checkpoint-delta.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-checkpoint-delta PROPERTIES FIXTURES_REQUIRED test-download-model)

# Test checkpoint delta-only chain (all checkpoints are deltas)
llama_build_and_test(test-checkpoint-delta-chain.cpp LABEL "model" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-checkpoint-delta-chain PROPERTIES FIXTURES_REQUIRED test-download-model)

# the rollback test aborts on any graph reallocation (GGML_SCHED_DEBUG_REALLOC=1). the fork plans the 1-output prompt
# graph last and plans a graph with more outputs again while computing (llama_context::sched_reserve, LLAMA_REPLAN_DISABLE),
# which that check counts as unexpected: run it without the replan, and once more with it but without the check
if (TEST test-recurrent-state-rollback)
    set_tests_properties(test-recurrent-state-rollback PROPERTIES
        ENVIRONMENT "GGML_SCHED_DEBUG_REALLOC=1;LLAMA_REPLAN_DISABLE=1"
    )
    llama_test(
        test-recurrent-state-rollback
        NAME test-recurrent-state-rollback-replan
        LABEL main
        ARGS --models "${MODEL_DIR}"
    )
    set_tests_properties(test-recurrent-state-rollback-replan PROPERTIES
        FIXTURES_REQUIRED generate-models
    )
endif()
