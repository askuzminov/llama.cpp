# the fork's tests, included at the end of tests/CMakeLists.txt: the helpers, fixtures and model paths defined there
# are in scope

llama_build(test-checkpoint-partial.cpp)

# prompt-cache disk spill: block-aligned unbuffered file layout, cold-start adoption, rejection of
# foreign or truncated files. server-context is a static library, so this links on Windows too.
llama_build_and_test(test-prompt-cache-spill.cpp)
target_include_directories(test-prompt-cache-spill PRIVATE
    ${PROJECT_SOURCE_DIR}/tools/server
    ${PROJECT_SOURCE_DIR}/tools/mtmd)
target_link_libraries(test-prompt-cache-spill PRIVATE server-context)

if (NOT WIN32 OR NOT BUILD_SHARED_LIBS)
    # internal functions not exported with LLAMA_API (as the upstream block of the same condition)
    llama_build_and_test(test-row-cache.cpp)
    target_include_directories(test-row-cache PRIVATE ${PROJECT_SOURCE_DIR}/src)

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
