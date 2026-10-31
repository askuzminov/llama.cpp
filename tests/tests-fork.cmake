# the fork's tests, included at the end of tests/CMakeLists.txt: the helpers, fixtures and model paths defined there
# are in scope

llama_build(test-checkpoint-partial.cpp)

# the window KV of the dense MTP head (LLAMA_MTP_WINDOW_KV): the rule is header-only, every platform builds this
llama_build_and_test(test-mtp-window-fork.cpp)
target_include_directories(test-mtp-window-fork PRIVATE ${PROJECT_SOURCE_DIR}/src)

# the gate of --spec-auto: a calibration per draft type, probabilities that are not a number (common/fork)
llama_build_and_test(test-spec-auto-fork.cpp)

# the fork's knobs of the environment: a value that is not a whole number gives the default, not the 0 of atoi
llama_build_and_test(test-env-fork.cpp)

# the qwen4exp indexer timed on one device at the shapes of the model (lightning indexer, the head matmul of the Vulkan
# graph and of the other backends): a benchmark, built but not a ctest; 04-micro.bat runs it on the machines
llama_build(test-micro-fork.cpp LINK ggml)

# the f16 hc read of qwen4exp (LLAMA_HC_F16) and other formats and places of rounding emulated on the CPU, on a deep
# model built from the generated qwen4exp-moe.gguf: a measurement of minutes, built but not a ctest
llama_build(test-hc-round-fork.cpp)

# the elastic buffers of the backends (CUDA VMM, Vulkan sparse buffers) on every GPU device that has them, skipped
# elsewhere; 01-check runs it on the machines (CHECKUNITS)
llama_build_and_test(test-elastic-buffer-fork.cpp)

# prompt-cache disk spill: block-aligned unbuffered file layout, cold-start adoption, rejection of
# foreign or truncated files. server-context is a static library, so this links on Windows too.
llama_build_and_test(test-prompt-cache-spill.cpp)
target_include_directories(test-prompt-cache-spill PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_link_libraries(test-prompt-cache-spill PRIVATE server-context)
llama_test(test-prompt-cache-spill NAME test-prompt-cache-compat LABEL "model behavior" ARGS -m "${MODEL_DEST}")
set_tests_properties(test-prompt-cache-compat PROPERTIES FIXTURES_REQUIRED test-download-model)

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

# the device cache of the host experts with --phase-mem (keep, fill, kv) against the release of all slots at every
# phase switch: built everywhere (llama.h, gguf.h), registered where the matrix models exist, skips without a GPU
llama_build(test-moe-cache-fork.cpp)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # the models of this matrix come from test-llama-archs (the generate-models fixture), which upstream builds under
    # the same condition: on a Windows DLL build there are none
    # partial (PARTIAL_ONLY) context checkpoints, one test per memory family:
    # plain attention, iSWA, hybrid (attention + recurrent) and pure recurrent
    foreach(CKPT_MODEL llama-dense gemma2-dense phi3-dense granitehybrid-dense nemotron_h-dense falcon-h1-dense mamba2-dense)
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

    # QSA with several sequences in one unified cache (the server's slots): the groupings carried across the rollbacks
    # of drafts, checked by LLAMA_QSA_CHECK, the mask from positions per sequence, and the logits of every sequence
    # against the same sequence alone, on the qwen4exp model of the matrix. the mask from positions must give the logits
    # of the dense mask bit for bit, and LLAMA_QSA_MASK_POS is read once per process: a run with the dense mask writes
    # its logits first. the binary is built on every platform (below), it needs only llama.h
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

    # the moe-cache through the phase switches: the reference run writes the model it builds from the qwen4exp of the
    # matrix (32 experts, no per-expert scales) and its logprobs, each mode compares with them in a run of its own
    set(MOE_CACHE_REF "${CMAKE_CURRENT_BINARY_DIR}/test-moe-cache-fork-ref.bin")
    llama_test(
        test-moe-cache-fork
        NAME test-moe-cache-fork-ref
        LABEL main
        ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --out "${MOE_CACHE_REF}"
    )
    set_tests_properties(test-moe-cache-fork-ref PROPERTIES
        FIXTURES_REQUIRED generate-models
        FIXTURES_SETUP moe-cache-ref
    )
    foreach(MOE_CACHE_MODE keep fill kv upload)
        llama_test(
            test-moe-cache-fork
            NAME test-moe-cache-fork-${MOE_CACHE_MODE}
            LABEL main
            ARGS -m "${MODEL_DIR}/qwen4exp-moe.gguf" --ref "${MOE_CACHE_REF}" --mode ${MOE_CACHE_MODE}
        )
        set_tests_properties(test-moe-cache-fork-${MOE_CACHE_MODE} PROPERTIES
            FIXTURES_REQUIRED "generate-models;moe-cache-ref"
        )
    endforeach()
endif()

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
