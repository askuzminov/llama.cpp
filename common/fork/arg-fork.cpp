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
        "pressure; loaded back on a cache hit, which leaves the file in place. A state is written when it leaves RAM "
        "(over --cache-ram or the host-RAM reserve) and on shutdown, the writes of the RAM limit wait for a moment "
        "when no slot runs; --cache-min-tokens sets the smallest state written and the rewrite step, "
        "--cache-idle-write also copies every changed state out while idle. "
        "The files are kept on shutdown and picked up again on the next start (cold-start reuse) as long as "
        "the model and KV configuration match. The context checkpoints go out with the state, so a prompt that "
        "matches only a prefix can still roll back after a restart. Several models may share one directory: a file "
        "is read only by its matching configuration; the disk budget is shared by all models in PATH. "
        "Best on fast NVMe. (default: disabled)",
        [](common_params & params, const std::string & value) {
            params.cache_spill_dir = value;
        }
    ).set_env("LLAMA_ARG_CACHE_SPILL_DIR").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-disk"}, "N",
        string_format("[experimental] disk budget in MiB for spilled prompt-cache states (see --cache-spill-dir). "
            "The limit covers all models in the directory. Oldest unused files are evicted first across models. "
            "Use the same budget in servers sharing PATH (default: %d, 0 = unlimited)", params.cache_disk_mib),
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
            "A shorter state stays in RAM only and is not written out on eviction or exit: it is cheap to recompute. A "
            "state whose front is already on disk is also not written again until it grows N tokens past it, which "
            "keeps a chat that gains a few hundred tokens per turn from rewriting gigabytes every turn: it leaves RAM "
            "without a write, and coming back costs up to N tokens of recompute. The RAM cache has its own minimum, "
            "--cache-ram-min-tokens", params.cache_min_tokens),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("cache-min-tokens must be >= 0");
            }
            params.cache_min_tokens = value;
        }
    ).set_env("LLAMA_ARG_CACHE_MIN_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-ram-min-tokens"}, "N",
        string_format("prompts shorter than N tokens are not kept in the RAM prompt cache (default: %d, 0 = no minimum): "
            "a hybrid model stores the same fixed recurrent state for any length, so many short states take room from "
            "the long ones. With more conversations than slots, a conversation that is not kept is computed again from "
            "the start when it comes back", params.cache_ram_min_tokens),
        [](common_params & params, int value) {
            if (value < 0) {
                throw std::invalid_argument("cache-ram-min-tokens must be >= 0");
            }
            params.cache_ram_min_tokens = value;
        }
    ).set_env("LLAMA_ARG_CACHE_RAM_MIN_TOKENS").set_examples({LLAMA_EXAMPLE_SERVER}));
    add_opt(common_arg(
        {"--cache-idle-write"},
        {"--no-cache-idle-write"},
        string_format("while all slots are idle, copy every changed prompt-cache state to --cache-spill-dir "
            "(write-behind): a later eviction then frees the RAM with no write and a crash loses at most "
            "--cache-min-tokens tokens of a conversation, at the cost of a disk write per changed state "
            "(default: %s)", params.cache_idle_write ? "enabled" : "disabled"),
        [](common_params & params, bool value) {
            params.cache_idle_write = value;
        }
    ).set_env("LLAMA_ARG_CACHE_IDLE_WRITE").set_examples({LLAMA_EXAMPLE_SERVER}));
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
                      "the draft types are those of --spec-type or the inferred one (default: %s)",
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

void common_fork_add_preset_options(std::vector<common_arg> & args) {
    args.push_back(common_arg(
        {"pool"}, "on|off",
        "in server router mode, share this model with the routers of the local network that share it under the same "
        "name or alias: each takes the requests for it and sends them to the machine that holds the conversation or has "
        "a free slot (tools/server/fork/server-pool-fork.cpp, a dashboard at /pool)",
        [](common_params &, const std::string &) { /* unused */ }
    ).set_env(COMMON_ARG_PRESET_POOL_FORK).set_preset_only());

    // the options of the pool, in the section of the model with pool = on or in [*]; server-pool-fork.cpp reads them by
    // the env name (COMMON_ARG_PRESET_POOL_FORK "_<KEY>") and keeps its defaults for the ones not given
    static const struct {
        const char * key;
        const char * env;
        const char * hint;
        const char * help;
    } pool_options[] = {
        {"pool-name",        COMMON_ARG_PRESET_POOL_FORK "_NAME",        "NAME",
         "this machine in the pool (default: the host name); with the port it is the id of the node, which the machines "
         "must not share"},
        {"pool-peers",       COMMON_ARG_PRESET_POOL_FORK "_PEERS",       "HOST:PORT,...",
         "routers of the pool added without the broadcasts, for networks without them"},
        {"pool-port",        COMMON_ARG_PRESET_POOL_FORK "_PORT",        "N",
         "the UDP port the routers of the pool find each other on (default: 12345)"},
        {"pool-discover",    COMMON_ARG_PRESET_POOL_FORK "_DISCOVER",    "0|1",
         "find the routers of the pool by broadcasts (default: 1); 0 = only pool-peers"},
        {"pool-move-tokens", COMMON_ARG_PRESET_POOL_FORK "_MOVE_TOKENS", "N",
         "a conversation with this many tokens of its own on a busy node waits for it, a shorter one goes to a free node "
         "at once; a state moves from this many tokens on (default: 4096)"},
        {"pool-wait-max",    COMMON_ARG_PRESET_POOL_FORK "_WAIT_MAX",    "SECONDS",
         "the longest wait for the node of a conversation (default: 300)"},
        {"pool-migrate",     COMMON_ARG_PRESET_POOL_FORK "_MIGRATE",     "0|1",
         "a conversation that goes to another node takes its cached state along when that is faster than computing it "
         "again (default: 1)"},
        {"pool-link-mbps",   COMMON_ARG_PRESET_POOL_FORK "_LINK_MBPS",   "N",
         "the speed of a link until it is measured, Mbit/s (default: 500)"},
        {"pool-prefill-tps", COMMON_ARG_PRESET_POOL_FORK "_PREFILL_TPS", "N",
         "the prefill speed of a node until it is measured, tokens/s (default: 700)"},
        {"pool-poll-ms",     COMMON_ARG_PRESET_POOL_FORK "_POLL_MS",     "N",
         "how often each node is polled, ms (default: 1000)"},
        {"pool-probe-s",     COMMON_ARG_PRESET_POOL_FORK "_PROBE_S",     "N",
         "how often a node on a slower link tries the faster ones, seconds (default: 10)"},
        {"pool-same-model",  COMMON_ARG_PRESET_POOL_FORK "_SAME_MODEL",  "0|1",
         "a node of another model than the most of the others takes no request (default: 1)"},
    };
    for (const auto & o : pool_options) {
        args.push_back(common_arg(
            {o.key}, o.hint, o.help,
            [](common_params &, const std::string &) { /* unused */ }
        ).set_env(o.env).set_preset_only());
    }
}

//
// common_models_handler_apply
//

void common_fork_spec_auto_apply(common_params & params, bool spec_types_given, bool spec_types_default) {
    // --spec-auto without an explicit --spec-draft-n-max takes COMMON_SPECULATIVE_AUTO_N_MAX as the cap.
    // until 12.10 it also added ngram-mod next to an inferred model draft. on the 3090 that cost 17 percent on the
    // repeated text (a larger fitted verify cost, more cycles without a draft) and 28.21 against 29.80 t/s over all
    // prompts; on the 395 it gave +16 percent on the repeated text only. ngram-mod now comes only with --spec-type
    (void) spec_types_given;
    (void) spec_types_default;
    if (params.speculative.auto_n) {
        auto & spec = params.speculative;

        if (!spec.draft.n_max_set) {
            // the cap itself, not a max with the default n_max: a larger upstream default would have made it no cap. a
            // DFlash draft has its own (COMMON_SPECULATIVE_AUTO_N_MAX_DFLASH, equal to the MTP one since 22.10, kept
            // apart so that either can be measured alone, as the 7 of DFlash was)
            const bool dflash = std::any_of(spec.types.begin(), spec.types.end(), [](auto t) {
                return t == COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH || t == COMMON_SPECULATIVE_TYPE_DRAFT_DSPARK;
            });
            spec.draft.n_max = dflash ? COMMON_SPECULATIVE_AUTO_N_MAX_DFLASH : COMMON_SPECULATIVE_AUTO_N_MAX;
        }

        // the draft sampled from the head and verified by rejection (26.10) unless --spec-draft-sampling says otherwise:
        // a sampled target accepts 1 - TV(p, q) of such drafts against p(argmax q) of the greedy ones. the 395, chat to
        // 213K at temperature 1.0: 156 draft tokens accepted a turn against 145-148 of three greedy arms, TG +3.2
        // percent; at 0.6 155 against 152, TG the same; the 3090: the chat the same. --spec-draft-sampling greedy:
        // upstream
        if (!spec.draft.probabilistic_set) {
            spec.draft.probabilistic = true;
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
