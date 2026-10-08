#pragma once

// the fork's command line options and its additions to the argument handling in arg.cpp

#include "arg.h"
#include "common.h"

#include "ggml-backend.h"

#include <functional>
#include <map>
#include <string>

// the fork's options are registered at fixed points of common_params_parser_init, so that --help keeps its order
enum common_arg_fork_at {
    COMMON_ARG_FORK_AT_CTX_CHECKPOINTS,  // after --ctx-checkpoints: --cache-ram-reserve
    COMMON_ARG_FORK_AT_CACHE_RAM,        // after --cache-ram: --cache-spill-dir, --cache-disk, --cache-min-tokens
    COMMON_ARG_FORK_AT_N_CPU_MOE,        // after --n-cpu-moe: --moe-cache, --phase-mem
    COMMON_ARG_FORK_AT_SPEC_SYNTH_RATES, // after --spec-synth-rates: --spec-auto
};

void common_fork_add_args(common_arg_fork_at at, const common_params & params, const std::function<void(common_arg)> & add_opt);

// --spec-auto in common_models_handler_apply, after the speculative type is inferred:
// spec_types_given - a type was requested, spec_types_default - no type is set (none requested, none inferred)
void common_fork_spec_auto_apply(common_params & params, bool spec_types_given, bool spec_types_default);

// add the extra buffer types of a device to the list that --override-tensor resolves names in
void common_fork_buft_list_add_extra(ggml_backend_dev_t dev, std::map<std::string, ggml_backend_buffer_type_t> & buft_list);
