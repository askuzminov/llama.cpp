#include "arg-fork.h"

#include "common.h"
#include "speculative-fork.h"

#include <algorithm>
#include <stdexcept>
#include <string>

//
// options
//

static void common_params_add_args_ctx_checkpoints(const common_params & params, const std::function<void(common_arg)> & add_opt) {
    add_opt(common_arg(
        {"-crr", "--cache-ram-reserve"}, "N",
        string_format("keep at least N MiB of host RAM free by evicting context checkpoints under memory pressure\n"
            "(default: %d, -1 = auto (fraction of total RAM), 0 = disabled). Checked on every checkpoint creation,\n"
            "so it also covers memory taken by other processes after startup.", params.cache_ram_reserve_mib),
        [](common_params & params, int value) {
            if (value < -1) {
                throw std::invalid_argument("cache-ram-reserve must be >= -1");
            }
            params.cache_ram_reserve_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_RAM_RESERVE").set_examples({LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}));
}

static void common_params_add_args_cache_ram(const common_params & params, const std::function<void(common_arg)> & add_opt) {
    add_opt(common_arg(
        {"--cache-spill-dir"}, "PATH",
        "[experimental] spill cold prompt-cache states to PATH (on disk) instead of dropping them under memory "
        "pressure; loaded back on a cache hit, which leaves the file in place. While the server is idle, resident "
        "states are copied out as well, so a later eviction frees the RAM with no I/O and a crash does not lose the "
        "cache. The files are kept on shutdown and picked up again on the next start (cold-start reuse) as long as "
        "the model and KV configuration match. The context checkpoints go out with the state, so a prompt that "
        "matches only a prefix can still roll back after a restart. Several models may share one directory: a file "
        "is named after the configuration it belongs to, so a model only ever reads and drops its own. "
        "Best on fast NVMe. (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.cache_spill_dir = value;
        }
    ).set_env("LLAMA_ARG_CACHE_SPILL_DIR").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-disk"}, "N",
        string_format("[experimental] disk budget in MiB for spilled prompt-cache states (see --cache-spill-dir). "
            "This budget is the only thing that deletes them: once it is exceeded the least valuable states go "
            "first, ranked by how often each was reused times how much work it saves. The budget counts one model, "
            "so a directory shared by several of them holds up to N MiB for each (default: %d, 0 = unlimited)", params.cache_disk_mib),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("cache-disk must be >= 0");
            }
            params.cache_disk_mib = value;
        }
    ).set_env("LLAMA_ARG_CACHE_DISK").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-min-tokens"}, "N",
        string_format("smallest piece of work worth a prompt-cache disk write, in tokens (default: %d, 0 = no minimum). "
            "Prompts shorter than N are not cached at all: they are cheap to recompute, while a hybrid model stores the "
            "same fixed recurrent state for any length, so caching one mostly takes room from the long prompts that pay "
            "off. A state whose front is already on disk is also not written again until it grows N tokens past it, "
            "which keeps a chat that gains a few hundred tokens per turn from rewriting gigabytes every turn. A crash "
            "then costs up to N tokens of recompute", params.cache_min_tokens),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("cache-min-tokens must be >= 0");
            }
            params.cache_min_tokens = value;
        }
    ).set_env("LLAMA_ARG_CACHE_MIN_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
}

static void common_params_add_args_n_cpu_moe(const common_params & /*params*/, const std::function<void(common_arg)> & add_opt) {
    add_opt(common_arg(
        {"--moe-cache"}, "N",
        "keep the most recently used MoE experts of the CPU layers in a device cache: N slots per layer, 'auto' to use free device memory, 0 = off (default: 0) [EXPERIMENTAL]",
        [](common_params & params, const std::string & value) {
            if (value == "auto") {
                params.n_moe_cache = -1;
            } else {
                params.n_moe_cache = std::stoi(value);
                if (params.n_moe_cache < 0) {
                    throw std::invalid_argument("invalid value");
                }
            }
        }
    ).set_env("LLAMA_ARG_MOE_CACHE"));
    add_opt(common_arg(
        {"--phase-mem"},
        "with one sequence, size the compute buffers per phase: a prompt batch gets the largest ubatch that fits the free device memory "
        "at its KV depth (up to -ub), and a generation batch gives the free memory to --moe-cache (default: off) [EXPERIMENTAL]",
        [](common_params & params) {
            params.phase_mem = true;
        }
    ).set_env("LLAMA_ARG_PHASE_MEM"));
}

static void common_params_add_args_spec_synth_rates(const common_params & params, const std::function<void(common_arg)> & add_opt) {
    add_opt(common_arg(
        {"--spec-auto"},
        {"--no-spec-auto"},
        string_format("pick the draft length per cycle from the measured verify cost and the calibrated draft acceptance. --spec-draft-n-max is the cap, %d when not given; "
                      "without --spec-type, ngram-mod is added next to the inferred draft type (default: %s)",
                      COMMON_SPECULATIVE_AUTO_N_MAX, params.speculative.auto_n ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.speculative.auto_n = value;
        }
    ).set_spec().set_examples({LLAMA_EXAMPLE_SPECULATIVE, LLAMA_EXAMPLE_SERVER, LLAMA_EXAMPLE_CLI}).set_env("LLAMA_ARG_SPEC_AUTO"));
}

void common_fork_add_args(common_arg_fork_at at, const common_params & params, const std::function<void(common_arg)> & add_opt) {
    switch (at) {
        case COMMON_ARG_FORK_AT_CTX_CHECKPOINTS:  common_params_add_args_ctx_checkpoints (params, add_opt); break;
        case COMMON_ARG_FORK_AT_CACHE_RAM:        common_params_add_args_cache_ram       (params, add_opt); break;
        case COMMON_ARG_FORK_AT_N_CPU_MOE:        common_params_add_args_n_cpu_moe       (params, add_opt); break;
        case COMMON_ARG_FORK_AT_SPEC_SYNTH_RATES: common_params_add_args_spec_synth_rates(params, add_opt); break;
    }
}

//
// common_models_handler_apply
//

void common_fork_spec_auto_apply(common_params & params, bool spec_types_given, bool spec_types_default) {
    // --spec-auto without an explicit --spec-draft-n-max raises the cap. without an explicit --spec-type,
    // ngram-mod joins the inferred model draft: it drafts only on a long match and costs nearly nothing when it finds none
    if (params.speculative.auto_n) {
        auto & spec = params.speculative;

        if (!spec.draft.n_max_set) {
            spec.draft.n_max = std::max(spec.draft.n_max, COMMON_SPECULATIVE_AUTO_N_MAX);
        }

        if (!spec_types_given && !spec_types_default) {
            spec.types.push_back(COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        }
    }
}

//
// parse_tensor_buffer_overrides
//

void common_fork_buft_list_add_extra(ggml_backend_dev_t dev, std::map<std::string, ggml_backend_buffer_type_t> & buft_list) {
    // the extra buffer types of the device too, as Vulkan0_Shared (the host memory a UMA gpu reads in place)
    auto * reg = ggml_backend_dev_backend_reg(dev);
    auto get_extra_bufts = reg ? (ggml_backend_dev_get_extra_bufts_t)
        ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts") : nullptr;
    for (auto * extra = get_extra_bufts ? get_extra_bufts(dev) : nullptr; extra && *extra; ++extra) {
        buft_list[ggml_backend_buft_name(*extra)] = *extra;
    }
}
