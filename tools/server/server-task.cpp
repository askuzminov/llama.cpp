#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>

#if defined(_WIN32)
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   ifndef WIN32_LEAN_AND_MEAN
#       define WIN32_LEAN_AND_MEAN
#   endif
#   include <windows.h>
#   include <malloc.h>
#else
#   include <cerrno>
#   include <fcntl.h>
#   include <sys/stat.h>
#   include <unistd.h>
#endif

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_parser_params.format)},
            {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
            {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
            {"generation_prompt",         chat_parser_params.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_parser_params.format)},
        {"reasoning_format",          common_reasoning_format_name(chat_parser_params.reasoning_format)},
        {"reasoning_in_content",      chat_parser_params.reasoning_in_content},
        {"generation_prompt",         chat_parser_params.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(const common_chat_parser_params & chat_parser_params)
    : chat_parser_params(chat_parser_params)
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
    if (chat_parser_params.is_continuation && !chat_parser_params.echo) {
        // initialize chat_msg to avoid emitting a delta containing the assistant prefill
        chat_msg = common_chat_parse("", true, chat_parser_params);
    }
}

common_chat_msg task_result_state::update_chat_msg(
        const std::string & text_added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    generated_text += text_added;
    auto msg_prv_copy = chat_msg;
    //SRV_DBG("Parsing chat message: %s\n", generated_text.c_str());
    auto new_msg = common_chat_parse(
        generated_text,
        is_partial,
        chat_parser_params);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

//
// server_prompt_cache
//
// on-disk layout of a spilled prompt-cache state: header, then the serialized prompt, then the two
// state blobs, then the context checkpoints (an index of fixed records, then their blobs).
// Self-describing so that a spill dir can be picked up again after a restart.
namespace {

struct spill_header {
    char     magic[8];
    uint32_t version;
    uint32_t n_used;
    uint64_t uid;
    uint64_t signature;
    uint64_t n_prompt; // server_tokens::serialize() output, in llama_token units
    uint64_t n_main;
    uint64_t n_drft;
    uint64_t n_ckpt;       // context checkpoints, i.e. spill_ckpt records in the index
    uint64_t n_ckpt_bytes; // their blobs, padding included
};

static_assert(sizeof(spill_header) == 72, "unexpected spill_header layout");

// one context checkpoint in the index; its blobs follow in the same order, each padded
struct spill_ckpt {
    int64_t  n_tokens;
    int32_t  id_task;
    int32_t  pos_min;
    int32_t  pos_max;
    int32_t  base_pos;
    uint64_t n_tgt;
    uint64_t n_dft;
    uint64_t n_spec;
};

static_assert(sizeof(spill_ckpt) == 48, "unexpected spill_ckpt layout");

constexpr char     SPILL_MAGIC[8] = { 'L', 'C', 'P', 'C', 'A', 'C', 'H', 'E' };
constexpr uint32_t SPILL_VERSION  = 4;

// Spill files bypass the OS file cache. A spilled state is written once and read back at most
// once, so caching it buys nothing and costs memory - on a host sized for a large model those
// pages compete with the model itself, and on Windows they are reported as available memory,
// which feeds straight back into the context-checkpoint budget.
//
// Bypassing the cache constrains the layout: file offset, transfer length and buffer address must
// all be multiples of the device block size. 4096 covers both 512e and 4Kn media. Every section is
// padded up to it, so the file is larger than its payload and the logical sizes live in the header.
constexpr uint64_t SPILL_ALIGN   = SERVER_STATE_ALIGN;
constexpr size_t   SPILL_CHUNK   = 4*1024*1024;  // bounce buffer, for sections that are not aligned
constexpr size_t   SPILL_REQUEST = 16*1024*1024; // direct transfer size, same as the model loader

constexpr uint64_t spill_align_up(uint64_t n) {
    return (n + SPILL_ALIGN - 1) & ~(SPILL_ALIGN - 1);
}

// an aligned buffer can be handed to the device as is, with no copy in between
bool spill_is_aligned(const void * p) {
    return ((uintptr_t) p % SPILL_ALIGN) == 0;
}

// the sizes come from a file that may have been truncated or tampered with
constexpr uint64_t SPILL_MAX_BYTES = 1ull << 40; // far above any real state
constexpr uint64_t SPILL_MAX_CKPT  = 4096;       // far above any real checkpoint count

bool spill_header_sane(const spill_header & h) {
    return h.n_prompt < SPILL_MAX_BYTES/sizeof(llama_token) && h.n_main < SPILL_MAX_BYTES &&
           h.n_drft < SPILL_MAX_BYTES && h.n_ckpt < SPILL_MAX_CKPT && h.n_ckpt_bytes < SPILL_MAX_BYTES;
}

bool spill_ckpt_sane(const spill_ckpt & c) {
    return c.n_tgt < SPILL_MAX_BYTES && c.n_dft < SPILL_MAX_BYTES && c.n_spec < SPILL_MAX_BYTES;
}

// UTF-8 -> filesystem path. On Windows a narrow string is taken to be in the ANSI codepage, which
// mangles everything outside it (a Cyrillic user directory, say), so convert explicitly.
std::filesystem::path spill_fs_path(const std::string & utf8) {
#if defined(_WIN32)
    if (utf8.empty()) {
        return {};
    }

    const int n = MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int) utf8.size(), nullptr, 0);
    if (n <= 0) {
        return {};
    }

    std::wstring wide((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.c_str(), (int) utf8.size(), &wide[0], n);

    return std::filesystem::path(wide);
#else
    return std::filesystem::path(utf8);
#endif
}

// filesystem path -> UTF-8, for log lines (path::string() would go through the ANSI codepage)
std::string spill_fs_utf8(const std::filesystem::path & path) {
#if defined(_WIN32)
    const std::wstring wide = path.wstring();
    if (wide.empty()) {
        return {};
    }

    const int n = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
        return {};
    }

    std::string utf8((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), (int) wide.size(), &utf8[0], n, nullptr, nullptr);

    return utf8;
#else
    return path.string();
#endif
}

// file name of a spilled state. The signature keeps the caches of different models apart in a
// shared spill dir: without it their uid counters collide and they overwrite each other's files
std::string spill_name_of(uint64_t signature, uint64_t uid) {
    char buf[64];
    snprintf(buf, sizeof(buf), "state-%016llx-%llu.bin", (unsigned long long) signature, (unsigned long long) uid);
    return buf;
}

enum spill_name_kind {
    SPILL_NAME_OTHER, // another model's file, or not one of ours at all
    SPILL_NAME_MINE,
    SPILL_NAME_OLD,   // ours by shape, from a build that had no signature in the name
};

spill_name_kind spill_name_classify(const std::string & name, uint64_t signature) {
    static const std::string pre = "state-";
    static const std::string ext = ".bin";

    if (name.size() <= pre.size() + ext.size() ||
        name.compare(0, pre.size(), pre) != 0 ||
        name.compare(name.size() - ext.size(), ext.size(), ext) != 0) {
        return SPILL_NAME_OTHER;
    }

    const std::string mid = name.substr(pre.size(), name.size() - pre.size() - ext.size());

    if (mid.find_first_not_of("0123456789") == std::string::npos) {
        return SPILL_NAME_OLD; // "state-<uid>.bin"
    }

    const size_t sep = mid.find('-');
    if (sep == std::string::npos || sep + 1 >= mid.size() ||
        mid.find_first_not_of("0123456789", sep + 1) != std::string::npos) {
        return SPILL_NAME_OTHER;
    }

    char expect[32];
    snprintf(expect, sizeof(expect), "%016llx-", (unsigned long long) signature);

    return mid.compare(0, sep + 1, expect) == 0 ? SPILL_NAME_MINE : SPILL_NAME_OTHER;
}

// section offsets implied by a header
struct spill_layout {
    uint64_t off_prompt;
    uint64_t off_main;
    uint64_t off_drft;
    uint64_t off_index;
    uint64_t off_ckpt;
    uint64_t size;
};

spill_layout spill_layout_of(const spill_header & h) {
    spill_layout l = {};
    l.off_prompt = SPILL_ALIGN;
    l.off_main   = l.off_prompt + spill_align_up(h.n_prompt*sizeof(llama_token));
    l.off_drft   = l.off_main   + spill_align_up(h.n_main);
    l.off_index  = l.off_drft   + spill_align_up(h.n_drft);
    l.off_ckpt   = l.off_index  + spill_align_up(h.n_ckpt*sizeof(spill_ckpt));
    l.size       = l.off_ckpt   + h.n_ckpt_bytes; // already a sum of padded sections
    return l;
}

// One spill file, read or written with the cache bypassed. An aligned buffer goes to the device as
// is; anything else passes through an aligned bounce buffer, so callers may hand over ordinary
// unaligned pointers and sizes. Falls back to cached I/O when the platform or filesystem refuses
// the unbuffered open - the layout stays valid either way.
class spill_file {
public:
    spill_file() = default;

    ~spill_file() {
        close();
    }

    spill_file(const spill_file &)             = delete;
    spill_file & operator=(const spill_file &) = delete;

    bool open_read (const std::filesystem::path & path) { return open_impl(path, false); }
    bool open_write(const std::filesystem::path & path) { return open_impl(path, true);  }

    // a whole state can be gigabytes, so a write is stopped between device transfers when the
    // caller says the machine is needed elsewhere. The half-written file is then thrown away
    void set_cancel(const std::function<bool()> * fn) { cancel = fn; }

    bool was_aborted() const { return aborted; }

    void close() {
#if defined(_WIN32)
        if (fh != INVALID_HANDLE_VALUE) {
            CloseHandle(fh);
            fh = INVALID_HANDLE_VALUE;
        }
        if (buf) {
            _aligned_free(buf);
            buf = nullptr;
        }
#else
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        if (buf) {
            free(buf);
            buf = nullptr;
        }
#endif
        used = 0;
    }

    uint64_t size() const {
#if defined(_WIN32)
        LARGE_INTEGER li;
        return GetFileSizeEx(fh, &li) ? (uint64_t) li.QuadPart : 0;
#else
        struct stat st = {};
        return fstat(fd, &st) == 0 ? (uint64_t) st.st_size : 0;
#endif
    }

    // append `n` bytes, then zero-pad up to the next SPILL_ALIGN boundary
    bool write_section(const void * src, size_t n) {
        const uint8_t * p = (const uint8_t *) src;

        // a large aligned source goes straight to the file; only the tail that is shorter than one
        // block still needs the bounce buffer, to be zero-padded. `used` is a whole number of
        // blocks here, so flushing it keeps the file offset aligned
        if (n >= SPILL_REQUEST && spill_is_aligned(p)) {
            if (used > 0 && !flush()) {
                return false;
            }

            const size_t direct = n & ~(size_t) (SPILL_ALIGN - 1);
            if (!write_raw(p, direct)) {
                return false;
            }

            p += direct;
            n -= direct;
        }

        while (n > 0) {
            const size_t k = std::min(n, SPILL_CHUNK - used);
            memcpy(buf + used, p, k);
            used += k;
            p    += k;
            n    -= k;

            if (used == SPILL_CHUNK && (check_cancel() || !flush())) {
                return false;
            }
        }

        const size_t pad = (size_t) (spill_align_up(used) - used);
        if (pad > 0) {
            memset(buf + used, 0, pad);
            used += pad;
        }

        return used < SPILL_CHUNK || flush();
    }

    // write out whatever is still buffered (always a whole number of blocks)
    bool finish() {
        return used == 0 || flush();
    }

    // read `n` bytes starting at a block-aligned file offset
    bool read_at(uint64_t offset, void * dst, size_t n) {
        if (offset % SPILL_ALIGN != 0) {
            return false;
        }

        uint8_t * p = (uint8_t *) dst;

        // a large aligned destination takes the whole blocks straight from the file; the tail that
        // is shorter than one block still comes through the bounce buffer
        if (n >= SPILL_REQUEST && spill_is_aligned(p)) {
            const size_t direct = n & ~(size_t) (SPILL_ALIGN - 1);
            if (!read_raw_into(offset, p, direct)) {
                return false;
            }

            p      += direct;
            n      -= direct;
            offset += direct;
        }

        while (n > 0) {
            const size_t want = (size_t) std::min<uint64_t>(SPILL_CHUNK, spill_align_up(n));
            const size_t got  = read_raw(offset, want);
            if (got == 0) {
                return false;
            }

            // an unbuffered handle only accepts aligned offsets, so a short read is cut back to a
            // whole number of blocks - unless what came back already covers the rest of the request
            const size_t k = got >= n ? n : got - got % (size_t) SPILL_ALIGN;
            if (k == 0) {
                return false; // less than one block: no way to continue from an aligned offset
            }

            memcpy(p, buf, k);

            p      += k;
            n      -= k;
            offset += k;
        }

        return true;
    }

private:
    bool check_cancel() {
        if (cancel && *cancel && (*cancel)()) {
            aborted = true;
        }

        return aborted;
    }

    // the fallback is a property of the volume, not of one file - say it once
    static void warn_buffered() {
        static bool once = false;
        if (!once) {
            once = true;
            SRV_WRN("%s", " - prompt-cache spill: unbuffered I/O is not available, using the OS cache\n");
        }
    }

    bool open_impl(const std::filesystem::path & path, bool write) {
        close();

        aborted = false;

#if defined(_WIN32)
        buf = (uint8_t *) _aligned_malloc(SPILL_CHUNK, (size_t) SPILL_ALIGN);
        if (!buf) {
            return false;
        }

        const std::wstring name = path.wstring();

        const DWORD access = write ? GENERIC_WRITE : GENERIC_READ;
        const DWORD share  = write ? 0 : FILE_SHARE_READ;
        const DWORD disp   = write ? CREATE_ALWAYS : OPEN_EXISTING;

        fh = CreateFileW(name.c_str(), access, share, nullptr, disp,
                FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (fh == INVALID_HANDLE_VALUE) {
            // the volume may not support unbuffered access
            fh = CreateFileW(name.c_str(), access, share, nullptr, disp, FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
            if (fh != INVALID_HANDLE_VALUE) {
                warn_buffered();
            }
        }
        if (fh == INVALID_HANDLE_VALUE) {
            close();
            return false;
        }
#else
        const std::string name = path.string();
        if (posix_memalign((void **) &buf, (size_t) SPILL_ALIGN, SPILL_CHUNK) != 0) {
            buf = nullptr;
            return false;
        }

        const int flags = write ? (O_WRONLY | O_CREAT | O_TRUNC) : O_RDONLY;

#if defined(O_DIRECT)
        fd = ::open(name.c_str(), flags | O_DIRECT, 0644);
        if (fd < 0) {
            fd = ::open(name.c_str(), flags, 0644); // filesystem without O_DIRECT support
            if (fd >= 0) {
                warn_buffered();
            }
        }
#else
        fd = ::open(name.c_str(), flags, 0644);
#endif
        if (fd < 0) {
            close();
            return false;
        }

#if defined(F_NOCACHE)
        fcntl(fd, F_NOCACHE, 1); // macOS has no O_DIRECT; this is the equivalent
#endif
#endif
        used = 0;
        return true;
    }

    bool flush() {
        const size_t n = used;
        used = 0;

#if defined(_WIN32)
        DWORD written = 0;
        return WriteFile(fh, buf, (DWORD) n, &written, nullptr) && written == n;
#else
        size_t off = 0;
        while (off < n) {
            const ssize_t k = ::write(fd, buf + off, n - off);
            if (k <= 0) {
                if (k < 0 && errno == EINTR) {
                    continue;
                }
                return false;
            }
            off += (size_t) k;
        }
        return true;
#endif
    }

    // write `n` bytes from the caller's buffer at the current offset; `n` is a whole number of
    // blocks and the buffer is aligned, so the bounce buffer is not needed
    bool write_raw(const void * src, size_t n) {
        const uint8_t * p = (const uint8_t *) src;

        while (n > 0) {
            if (check_cancel()) {
                return false;
            }

            const size_t want = std::min<size_t>(n, SPILL_REQUEST);

#if defined(_WIN32)
            DWORD k = 0;
            if (!WriteFile(fh, p, (DWORD) want, &k, nullptr)) {
                return false;
            }
#else
            const ssize_t r = ::write(fd, p, want);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            const size_t k = (size_t) r;
#endif
            // an unbuffered handle can only go on from a block boundary
            if (k == 0 || (k != n && k % SPILL_ALIGN != 0)) {
                return false;
            }

            p += k;
            n -= k;
        }

        return true;
    }

    // read `len` bytes into the caller's buffer; offset, length and buffer are all aligned
    bool read_raw_into(uint64_t offset, void * dst, size_t len) {
        uint8_t * p = (uint8_t *) dst;

#if defined(_WIN32)
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG) offset;
        if (!SetFilePointerEx(fh, li, nullptr, FILE_BEGIN)) {
            return false;
        }
#endif

        while (len > 0) {
            const size_t want = std::min<size_t>(len, SPILL_REQUEST);

#if defined(_WIN32)
            DWORD k = 0;
            if (!ReadFile(fh, p, (DWORD) want, &k, nullptr)) {
                return false;
            }
#else
            const ssize_t r = ::pread(fd, p, want, (off_t) offset);
            if (r < 0) {
                if (errno == EINTR) {
                    continue;
                }
                return false;
            }
            const size_t k = (size_t) r;
#endif
            if (k == 0 || (k != len && k % SPILL_ALIGN != 0)) {
                return false;
            }

            p      += k;
            len    -= k;
            offset += k;
        }

        return true;
    }

    size_t read_raw(uint64_t offset, size_t len) {
#if defined(_WIN32)
        LARGE_INTEGER li;
        li.QuadPart = (LONGLONG) offset;
        if (!SetFilePointerEx(fh, li, nullptr, FILE_BEGIN)) {
            return 0;
        }
        DWORD got = 0;
        if (!ReadFile(fh, buf, (DWORD) len, &got, nullptr)) {
            return 0;
        }
        return (size_t) got;
#else
        const ssize_t got = ::pread(fd, buf, len, (off_t) offset);
        return got > 0 ? (size_t) got : 0;
#endif
    }

#if defined(_WIN32)
    HANDLE fh = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif

    uint8_t * buf  = nullptr; // aligned bounce buffer, SPILL_CHUNK bytes
    size_t    used = 0;       // bytes buffered on the write path

    const std::function<bool()> * cancel = nullptr; // polled between device transfers, may be null
    bool aborted = false;                           // the cancel callback stopped this write
};

} // namespace

server_prompt_cache::server_prompt_cache(size_t limit_size_mib, size_t limit_tokens, size_t reserve_bytes,
                                         const std::string & spill_dir, size_t spill_limit_bytes,
                                         uint64_t signature, bool has_mtmd) {
    this->limit_size   = 1024ull*1024ull*limit_size_mib;
    this->limit_tokens = limit_tokens;
    this->reserve_size = reserve_bytes;
    this->spill_dir    = spill_dir;
    this->spill_limit  = spill_limit_bytes;
    this->signature    = signature;
    this->has_mtmd     = has_mtmd;

    if (!this->spill_dir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(spill_dir_path(), ec);
        if (ec) {
            SRV_WRN(" - could not create prompt-cache spill dir '%s' (%s) - disabling spill\n",
                    this->spill_dir.c_str(), ec.message().c_str());
            this->spill_dir.clear();
        } else {
            const size_t n = restore_spilled();
            if (n > 0) {
                SRV_INF(" - adopted %zu prompt-cache state(s) from '%s' (%.3f MiB on disk)\n",
                        n, this->spill_dir.c_str(), disk_size() / (1024.0 * 1024.0));
            }
        }
    }
}

server_prompt_cache::~server_prompt_cache() {
    // a destructor must not propagate: a failed persist only costs the spilled states
    try {
        persist();
    } catch (const std::exception & e) {
        SRV_ERR(" - failed to persist the prompt cache: %s\n", e.what());
    } catch (...) {
        SRV_ERR("%s", " - failed to persist the prompt cache\n");
    }
}

std::filesystem::path server_prompt_cache::spill_dir_path() const {
    return spill_fs_path(spill_dir);
}

std::filesystem::path server_prompt_cache::spill_path(uint64_t uid) const {
    return spill_dir_path() / spill_name_of(signature, uid);
}

size_t server_prompt_cache::restore_spilled() {
    std::error_code ec;
    std::filesystem::directory_iterator it(spill_dir_path(), ec);
    if (ec) {
        return 0;
    }

    // adopt in a stable order so that the list stays oldest-first by uid
    std::vector<server_prompt_cache_state> found;

    for (const auto & entry : it) {
        if (!entry.is_regular_file(ec) || ec) {
            continue;
        }

        const std::filesystem::path path = entry.path();

        const spill_name_kind kind = spill_name_classify(spill_fs_utf8(path.filename()), signature);
        if (kind == SPILL_NAME_OTHER) {
            continue; // another model spills here too - leave its files alone
        }

        spill_file f;
        if (!f.open_read(path)) {
            continue;
        }

        spill_header h = {};
        if (!f.read_at(0, &h, sizeof(h))) {
            continue; // not one of ours
        }
        if (memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) != 0) {
            continue; // not one of ours - leave it alone
        }
        if (kind == SPILL_NAME_OLD || h.version != SPILL_VERSION) {
            // our file, written by another build: the layout or the name is gone, so is the state
            f.close();
            std::filesystem::remove(path, ec);
            continue;
        }

        // the header disagrees with the name, or the file is truncated - the state cannot be restored
        const spill_layout lay = spill_header_sane(h) ? spill_layout_of(h) : spill_layout{};
        if (h.signature != signature || lay.size == 0 || lay.size != f.size()) {
            f.close();
            std::filesystem::remove(path, ec);
            continue;
        }

        llama_tokens packed(h.n_prompt);
        if (h.n_prompt && !f.read_at(lay.off_prompt, packed.data(), h.n_prompt*sizeof(llama_token))) {
            f.close();
            std::filesystem::remove(path, ec);
            continue;
        }
        f.close();

        server_prompt_cache_state st;
        try {
            st.prompt.tokens = server_tokens::deserialize(packed, has_mtmd);
        } catch (const std::exception & e) {
            // the media chunks cannot be rebuilt: the server runs without an mmproj now, or the
            // chunk format moved on. Either way the state is unusable
            SRV_WRN(" - discarding prompt-cache spill file '%s': %s\n", spill_fs_utf8(path).c_str(), e.what());
            std::filesystem::remove(path, ec);
            continue;
        }
        st.uid           = h.uid;
        st.prompt.n_used = h.n_used;
        st.on_disk       = true;
        st.ckpt_on_disk  = h.n_ckpt;
        st.disk_bytes    = lay.size;

        next_uid = std::max(next_uid, h.uid + 1);

        found.push_back(std::move(st));
    }

    std::sort(found.begin(), found.end(),
            [](const server_prompt_cache_state & a, const server_prompt_cache_state & b) { return a.uid < b.uid; });

    for (auto & st : found) {
        states.push_back(std::move(st));
    }

    enforce_disk_limit();

    return found.size();
}

void server_prompt_cache::persist() {
    if (spill_dir.empty()) {
        // no spill dir: nothing was written, nothing to clean up
        return;
    }

    for (auto & st : states) {
        spill_state(st);
    }

    enforce_disk_limit();

    size_t n_disk = 0;
    for (const auto & st : states) {
        n_disk += st.on_disk ? 1 : 0;
    }

    SRV_INF(" - prompt cache: %zu of %zu states kept on disk, %.3f MiB\n",
            n_disk, states.size(), disk_size() / (1024.0 * 1024.0));
}

bool server_prompt_cache::write_state(server_prompt_cache_state & st, const std::function<bool()> * cancel, size_t disk_free) {
    if (spill_dir.empty() || st.on_disk) {
        return false;
    }
    if (st.data.main.empty() && st.data.drft.empty()) {
        return false; // nothing big to write out
    }

    // the prompt is stored alongside the state so the file can be picked up after a restart. Media
    // chunks come with it, without their pixels: the image is already encoded into the state blob,
    // and matching the prompt again only needs the chunk id and its token count
    std::vector<char> prompt;
    try {
        prompt = st.prompt.tokens.serialize();
    } catch (const std::exception & e) {
        SRV_ERR(" - failed to serialize a prompt-cache state: %s\n", e.what());
        return false;
    }

    GGML_ASSERT(prompt.size() % sizeof(llama_token) == 0);

    spill_header h = {};
    memcpy(h.magic, SPILL_MAGIC, sizeof(h.magic));
    h.version   = SPILL_VERSION;
    h.n_used    = st.prompt.n_used;
    h.signature = signature;
    h.n_prompt  = prompt.size()/sizeof(llama_token);
    h.n_main    = st.data.main.size();
    h.n_drft    = st.data.drft.size();

    // the context checkpoints go out with the state: after a restart they are what lets a prompt
    // that matches only a prefix roll back, instead of being processed from the start again
    std::vector<spill_ckpt> index;
    index.reserve(st.prompt.checkpoints.size());

    for (const auto & cp : st.prompt.checkpoints) {
        index.push_back({
            cp.n_tokens, cp.id_task, cp.pos_min, cp.pos_max, cp.base_pos,
            cp.data_tgt.size(), cp.data_dft.size(), cp.data_spec.size(),
        });

        h.n_ckpt_bytes += spill_align_up(cp.data_tgt.size())
                        + spill_align_up(cp.data_dft.size())
                        + spill_align_up(cp.data_spec.size());
    }

    h.n_ckpt = index.size();

    // a checkpoint only saves recomputation, so when the disk budget cannot take both, the
    // checkpoints go and the state is still written out
    if (h.n_ckpt > 0 && spill_layout_of(h).size > disk_free) {
        h.n_ckpt       = 0;
        h.n_ckpt_bytes = 0;
        index.clear();
    }

    // what does not fit the disk budget is skipped rather than written and deleted again right after
    if (spill_layout_of(h).size > disk_free) {
        return false;
    }

    if (st.uid == 0) {
        st.uid = next_uid++;
    }
    h.uid = st.uid;

    const std::filesystem::path path = spill_path(st.uid);

    spill_file f;
    f.set_cancel(cancel);

    if (!f.open_write(path)) {
        SRV_ERR(" - failed to create prompt-cache spill file '%s'\n", spill_fs_utf8(path).c_str());
        return false;
    }

    // sections are written in the order spill_layout_of() describes; an empty one writes nothing
    // and contributes no padding, so the offsets still line up
    bool ok =
        f.write_section(&h, sizeof(h)) &&
        (h.n_prompt == 0 || f.write_section(prompt.data(), prompt.size())) &&
        (h.n_main   == 0 || f.write_section(st.data.main.data(), h.n_main)) &&
        (h.n_drft   == 0 || f.write_section(st.data.drft.data(), h.n_drft)) &&
        (h.n_ckpt   == 0 || f.write_section(index.data(), index.size()*sizeof(spill_ckpt)));

    if (ok && h.n_ckpt > 0) {
        for (const auto & cp : st.prompt.checkpoints) {
            ok = (cp.data_tgt.empty()  || f.write_section(cp.data_tgt.data(),  cp.data_tgt.size())) &&
                 (cp.data_dft.empty()  || f.write_section(cp.data_dft.data(),  cp.data_dft.size())) &&
                 (cp.data_spec.empty() || f.write_section(cp.data_spec.data(), cp.data_spec.size()));
            if (!ok) {
                break;
            }
        }
    }

    ok = ok && f.finish();

    f.close();

    if (!ok) {
        // a cancelled write is not an error: the state stays in RAM and goes out at the next idle
        if (!f.was_aborted()) {
            SRV_ERR(" - failed to write prompt-cache spill file '%s' (%.3f MiB)\n",
                    spill_fs_utf8(path).c_str(), spill_layout_of(h).size / (1024.0 * 1024.0));
        }

        std::error_code ec;
        std::filesystem::remove(path, ec);
        return false;
    }

    st.disk_bytes   = spill_layout_of(h).size;
    st.ckpt_on_disk = h.n_ckpt;
    st.on_disk      = true;
    return true;
}

void server_prompt_cache::evict_state(server_prompt_cache_state & st) {
    if (!st.on_disk) {
        return; // without a disk copy the blobs are the only copy
    }

    st.data.main.clear();  st.data.main.shrink_to_fit();
    st.data.drft.clear();  st.data.drft.shrink_to_fit();

    // the checkpoints are in the file too, unless the disk budget refused them
    if (st.ckpt_on_disk == st.prompt.checkpoints.size()) {
        st.prompt.checkpoints.clear();
    }
}

bool server_prompt_cache::spill_state(server_prompt_cache_state & st, size_t disk_free) {
    if (!st.on_disk && !write_state(st, nullptr, disk_free)) {
        return false;
    }

    evict_state(st);
    return true;
}

bool server_prompt_cache::unspill_state(server_prompt_cache_state & st) {
    if (!st.on_disk) {
        return true;
    }

    // the file stays. It holds exactly the bytes the state is handing back, so taking it away only
    // buys a full rewrite at the next idle. enforce_disk_limit decides when a file goes
    if (st.data.size() > 0 && st.prompt.checkpoints.size() >= st.ckpt_on_disk) {
        return true; // clean: already in RAM, no read needed
    }

    const std::filesystem::path path = spill_path(st.uid);

    spill_file f;

    // a file that cannot be read back is of no use to anyone: drop it and clear the on-disk flag,
    // so the state is not retried on every load and stops counting against the disk budget.
    // the entry is left with no data at all, which is what the caller must react to.
    const auto drop = [&]() {
        f.close();

        std::error_code ec;
        std::filesystem::remove(path, ec);

        st.data.main.clear(); st.data.main.shrink_to_fit();
        st.data.drft.clear(); st.data.drft.shrink_to_fit();
        st.prompt.checkpoints.clear();
        st.on_disk      = false;
        st.disk_bytes   = 0;
        st.ckpt_on_disk = 0;

        return false;
    };

    if (!f.open_read(path)) {
        SRV_ERR(" - failed to read spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
    }

    spill_header h = {};
    if (!f.read_at(0, &h, sizeof(h)) ||
            memcmp(h.magic, SPILL_MAGIC, sizeof(h.magic)) != 0 ||
            h.version != SPILL_VERSION ||
            h.signature != signature ||
            !spill_header_sane(h) ||
            spill_layout_of(h).size != f.size()) {
        SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
    }

    const spill_layout lay = spill_layout_of(h);

    const auto rd = [&](server_state_buf & v, uint64_t off, uint64_t n) -> bool {
        v.resize(n);
        return n == 0 || f.read_at(off, v.data(), n);
    };
    if (!rd(st.data.main, lay.off_main, h.n_main) || !rd(st.data.drft, lay.off_drft, h.n_drft)) {
        SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
        return drop();
    }

    if (h.n_ckpt > 0) {
        // checkpoints the disk budget refused are only in RAM, so the list is replaced by the file
        // only when the file actually holds one
        st.prompt.checkpoints.clear();

        std::vector<spill_ckpt> index(h.n_ckpt);

        if (!f.read_at(lay.off_index, index.data(), index.size()*sizeof(spill_ckpt))) {
            SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
            return drop();
        }

        // the index must describe exactly the bytes the header reserved, or the walk below would
        // read past the sections it owns
        uint64_t n_bytes = 0;
        for (const auto & e : index) {
            if (!spill_ckpt_sane(e)) {
                n_bytes = h.n_ckpt_bytes + 1;
                break;
            }
            n_bytes += spill_align_up(e.n_tgt) + spill_align_up(e.n_dft) + spill_align_up(e.n_spec);
        }

        if (n_bytes != h.n_ckpt_bytes) {
            SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
            return drop();
        }

        uint64_t off = lay.off_ckpt;

        const auto rd_ckpt = [&](std::vector<uint8_t> & v, uint64_t n) -> bool {
            v.resize(n);
            if (n > 0 && !f.read_at(off, v.data(), n)) {
                return false;
            }
            off += spill_align_up(n);
            return true;
        };

        for (const auto & e : index) {
            common_prompt_checkpoint cp;
            cp.n_tokens = e.n_tokens;
            cp.id_task  = e.id_task;
            cp.pos_min  = e.pos_min;
            cp.pos_max  = e.pos_max;
            cp.base_pos = e.base_pos;

            if (!rd_ckpt(cp.data_tgt, e.n_tgt) || !rd_ckpt(cp.data_dft, e.n_dft) || !rd_ckpt(cp.data_spec, e.n_spec)) {
                SRV_ERR(" - corrupt spilled prompt-cache state '%s'\n", spill_fs_utf8(path).c_str());
                return drop();
            }

            st.prompt.checkpoints.push_back(std::move(cp));
        }
    }

    f.close();

    return true; // the state is clean now: same bytes in RAM and on disk
}

void server_prompt_cache::erase_spill(const server_prompt_cache_state & st) {
    if (st.on_disk && !spill_dir.empty()) {
        std::error_code ec;
        std::filesystem::remove(spill_path(st.uid), ec);
    }
}

void server_prompt_cache::enforce_disk_limit() {
    if (spill_dir.empty() || spill_limit == 0) {
        return;
    }
    size_t used = disk_size();

    // drop the least valuable on-disk state (oldest first on a tie) until within budget. Value is
    // how often the state was reused times how much work it saves, so a rarely used or a short and
    // cheap to redo prompt goes first and a cold start keeps the long ones that pay off
    while (used > spill_limit) {
        auto   sel     = states.end();
        size_t sel_val = 0;

        for (auto it = states.begin(); it != states.end(); ++it) {
            if (!it->on_disk) {
                continue;
            }
            const size_t val = (size_t) (it->prompt.n_used + 1) * it->prompt.tokens.size();

            if (sel == states.end() || val < sel_val) {
                sel     = it;
                sel_val = val;
            }
        }
        if (sel == states.end()) {
            break; // nothing on disk left to drop
        }
        const bool resident = sel->data.size() > 0;

        SRV_WRN(" - prompt-cache disk budget exceeded, dropping %s state (%.3f MiB, %d tokens, used %u time(s))\n",
                resident ? "the disk copy of a resident" : "spilled", sel->disk_bytes / (1024.0 * 1024.0),
                sel->prompt.n_tokens(), sel->prompt.n_used);

        used -= std::min(used, sel->disk_bytes);
        erase_spill(*sel);

        if (resident) {
            // the blobs are still in RAM, so only the copy goes - the entry stays usable
            sel->on_disk    = false;
            sel->disk_bytes = 0;
        } else {
            states.erase(sel);
        }
    }
}

size_t server_prompt_cache::free_ram(size_t need) {
    size_t freed = 0;

    if (!spill_dir.empty()) {
        // spill oldest resident states to disk: this frees their RAM but keeps them restorable, so
        // we never drop a state just to satisfy a RAM deficit. Spilling the whole (bounded) cache
        // is the most we can do for host-RAM pressure; the disk budget alone decides what is
        // dropped for good (enforce_disk_limit)
        size_t disk_free = spill_limit == 0 ? SIZE_MAX : spill_limit - std::min(spill_limit, disk_size());

        // a clean state already has its copy on disk (write_behind put it there while idle), so
        // releasing its RAM costs no I/O at all. Take those first
        for (auto & st : states) {
            if (freed >= need) {
                break;
            }
            if (!st.is_clean()) {
                continue;
            }
            const size_t big = st.size();
            evict_state(st);
            freed += big - st.size();
        }

        for (auto & st : states) {
            if (freed >= need) {
                break;
            }
            if (st.on_disk || st.data.size() == 0) {
                continue;
            }
            const size_t big = st.size();
            if (spill_state(st, disk_free)) {
                freed     += big - st.size();
                disk_free -= std::min(disk_free, st.disk_bytes);
            }
        }

        // checkpoints the disk budget refused, or ones read back by a cache hit, are still in RAM.
        // they only save recomputation, so let them go when spilling did not cover the deficit
        for (auto & st : states) {
            if (freed >= need) {
                break;
            }
            if (!st.on_disk || st.prompt.checkpoints.empty()) {
                continue;
            }
            size_t big = 0;
            for (const auto & ckpt : st.prompt.checkpoints) {
                big += ckpt.size();
            }
            st.prompt.checkpoints.clear();
            freed += big;
        }

        enforce_disk_limit();
        return freed;
    }

    // no spill dir: fall back to dropping oldest states outright (front = oldest)
    while (freed < need && !states.empty()) {
        SRV_WRN(" - dropping oldest prompt-cache entry (%.3f MiB)\n",
                states.front().size() / (1024.0 * 1024.0));
        freed += states.front().size();
        states.pop_front();
    }

    return freed;
}

// how many leading tokens of `st` some other entry already holds on disk. Writing `st` out buys
// only the tokens past that point
static size_t spill_prefix_on_disk(const std::list<server_prompt_cache_state> & states,
        const server_prompt_cache_state & st) {
    size_t res = 0;

    for (const auto & other : states) {
        if (&other == &st || !other.on_disk) {
            continue;
        }

        const size_t len = other.prompt.tokens.get_common_prefix(st.prompt.tokens);
        if (len == other.prompt.tokens.size()) {
            res = std::max(res, len);
        }
    }

    return res;
}

size_t server_prompt_cache::write_behind(const std::function<bool()> & interrupted) {
    if (spill_dir.empty()) {
        return 0;
    }

    size_t disk_free = spill_limit == 0 ? SIZE_MAX : spill_limit - std::min(spill_limit, disk_size());

    size_t n_written = 0;
    size_t n_bytes   = 0;

    const int64_t t_start = ggml_time_us();

    // oldest first, the same order free_ram spills in: those are the states it will evict next, and
    // evicting a state that is already on disk is free
    for (auto it = states.begin(); it != states.end(); ++it) {
        auto & st = *it;

        if (st.on_disk || st.data.size() == 0) {
            continue;
        }

        // an older file already holds the front of this state. A rewrite would move gigabytes to
        // win back only the tokens past that point, so let them pile up to the minimum first. What
        // a crash costs is then that much recompute, not the whole prompt
        const size_t n_have = spill_prefix_on_disk(states, st);
        if (n_have > 0 && st.prompt.tokens.size() - n_have < min_tokens) {
            SRV_TRC(" - %zu of %zu tokens are on disk already, below the %zu token rewrite step, skipping\n",
                    n_have, st.prompt.tokens.size(), min_tokens);
            continue;
        }

        // a request is waiting - the rest goes out at the next idle
        if (interrupted && interrupted()) {
            break;
        }

        if (!write_state(st, &interrupted, disk_free)) {
            if (interrupted && interrupted()) {
                break; // stopped part way through, the file is gone and the state is still in RAM
            }
            continue;
        }

        disk_free -= std::min(disk_free, st.disk_bytes);
        n_bytes   += st.disk_bytes;
        n_written += 1;

        // the new file repeats what the shorter ones in front of it hold, so those go. This is
        // deduplication, not eviction: nothing that is still the only copy of something is touched
        for (auto it2 = states.begin(); it2 != states.end();) {
            if (it2 == it || !it2->on_disk || it2->data.size() > 0 ||
                    it2->prompt.tokens.get_common_prefix(st.prompt.tokens) != it2->prompt.tokens.size()) {
                ++it2;
                continue;
            }

            SRV_TRC(" - dropping the disk copy of a %d token prefix, the %d token state covers it\n",
                    it2->prompt.n_tokens(), st.prompt.n_tokens());

            if (disk_free != SIZE_MAX) {
                disk_free += it2->disk_bytes;
            }

            erase_spill(*it2);
            it2 = states.erase(it2);
        }
    }

    if (n_written > 0) {
        const double t_ms = (ggml_time_us() - t_start) / 1000.0;

        SRV_INF(" - prompt cache: wrote %zu state(s) to disk while idle, %.3f MiB in %.0f ms (%.2f GB/s)\n",
                n_written, n_bytes / (1024.0 * 1024.0), t_ms, n_bytes / (t_ms * 1e6));
    }

    return n_written;
}

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::disk_size() const {
    size_t res = 0;

    for (const auto & state : states) {
        if (state.on_disk) {
            res += state.disk_bytes;
        }
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // a short prompt is cheap to recompute, while the recurrent part of a hybrid state costs the
    // same no matter how long the prompt is - caching one only takes room from the long prompts
    if (min_tokens > 0 && prompt.tokens.size() < min_tokens) {
        SRV_TRC(" - prompt has %zu tokens, below the cache minimum of %zu, skipping\n",
                prompt.tokens.size(), min_tokens);
        return nullptr;
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt. One that is on disk
    // only gives up its RAM: its file already holds the front of the new state, so keeping it lets
    // write_behind skip the rewrite, and a branch off that prefix is still restorable
    for (auto it = states.begin(); it != states.end();) {
        const size_t len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == it->prompt.tokens.size()) {
            if (it->on_disk) {
                SRV_TRC(" - obsolete cached prompt with length %zu stays on disk\n", len);

                evict_state(*it);
                ++it;
                continue;
            }

            SRV_TRC(" - removing obsolete cached prompt with length %zu\n", len);

            erase_spill(*it);
            it = states.erase(it);
        } else {
            ++it;
        }
    }

    const size_t size_cur = size();

    if (limit_size > 0 && size_cur + state_size_new > limit_size) {
        // make room before allocating the new vectors to avoid breaching the RAM limit; with a
        // spill dir this moves cold states to disk instead of dropping them
        free_ram(size_cur + state_size_new - limit_size);
    }

    // make room so the new state keeps host RAM above the reserve
    evict_for_reserve(state_size_new);

    server_state_buf state_data_tgt;
    server_state_buf state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens          =*/ prompt.tokens.clone(),
            /*.checkpoints     =*/ prompt.checkpoints,
            /*.n_used          =*/ prompt.n_used,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

    return &states.back();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        // pull the state back from disk if it was spilled
        if (it_best->on_disk && !unspill_state(*it_best)) {
            SRV_ERR("%s", "failed to load spilled prompt-cache state\n");
            erase_spill(*it_best);
            states.erase(it_best);
            return false;
        }

        // an entry with no state blob would restore nothing while the slot believes its tokens are
        // in the KV cache - drop it instead
        if (it_best->data.main.empty()) {
            SRV_ERR("%s", "prompt-cache entry holds no state, dropping it\n");
            erase_spill(*it_best);
            states.erase(it_best);
            return false;
        }

        {
            auto & data = it_best->data.main;

            const size_t size = data.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);

                // the target context is half-written and the entry cannot be trusted either
                erase_spill(*it_best);
                states.erase(it_best);

                return false;
            }

            data.clear();
            data.shrink_to_fit();
        }

        {
            auto & data = it_best->data.drft;

            if (!data.empty()) {
                GGML_ASSERT(ctx_dft);

                const size_t size = data.size();
                const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.data(), size, id_slot, 0);
                if (n != size) {
                    SRV_WRN("failed to restore state with size %zu\n", size);

                    // the main blob was consumed above, so what is left could never be restored again
                    erase_spill(*it_best);
                    states.erase(it_best);

                    return false;
                }

                data.clear();
                data.shrink_to_fit();
            }
        }

        prompt = std::move(it_best->prompt);
        prompt.n_used++;

        if (it_best->on_disk) {
            // the blobs went into the slot but the file is untouched and costs nothing to keep, so
            // the entry stays as a disk-only record: the budget still accounts for it, a branch off
            // this prefix can still be restored, and the slot does not have to write it all again
            it_best->prompt        = server_prompt();
            it_best->prompt.tokens = prompt.tokens.clone();
            it_best->prompt.n_used = prompt.n_used;
        } else {
            states.erase(it_best);
        }
    }

    return true;
}

void server_prompt_cache::evict_for_reserve(size_t extra) {
    if (reserve_size == 0) {
        return;
    }

    const size_t avail = common_host_mem_available();
    if (avail == 0) {
        return; // could not determine available RAM -> don't act on it
    }

    const size_t want = reserve_size + extra;
    if (avail >= want) {
        return;
    }

    // free the deficit only (freed heap may not return to the OS immediately, so re-querying
    // available RAM here would over-evict). free_ram spills cold states to disk when a spill dir
    // is configured, and only drops them outright as a last resort.
    const size_t need = want - avail;
    SRV_WRN(" - host memory pressure, relieving %.3f MiB from the prompt cache (available %.3f MiB < reserve %.3f MiB)\n",
            need / (1024.0 * 1024.0), avail / (1024.0 * 1024.0), reserve_size / (1024.0 * 1024.0));
    free_ram(need);
}

void server_prompt_cache::update() {
    if (limit_size > 0 && size() > limit_size) {
        // spill cold states to disk (or drop them without a spill dir) to fit the RAM limit
        free_ram(size() - limit_size);
    }

    // keep the on-disk spill within its budget
    enforce_disk_limit();

    // keep host RAM above the reserve (adapts to memory taken by other processes over time)
    evict_for_reserve(0);

    // size() walks the whole list, so it is taken once and kept up to date
    size_t size_cur = size();

    // a state that lives only on disk holds no RAM. With a disk budget set that budget bounds it,
    // so the token limit only has to bound what is resident; without one this limit is still the
    // only thing that stops the directory from growing, so it keeps counting
    const bool disk_bounded = !spill_dir.empty() && spill_limit > 0;

    const auto off_disk = [&](const server_prompt_cache_state & state) {
        return state.on_disk && state.data.size() == 0;
    };

    // average size per token - a state that is only on disk contributes its file, otherwise the
    // estimate collapses and the token limit stops bounding the cache. A clean state sits in both
    // places and must be counted once
    size_t n_tokens_cur = 0;
    size_t size_off     = 0;

    for (const auto & state : states) {
        if (disk_bounded && off_disk(state)) {
            continue;
        }

        n_tokens_cur += state.prompt.n_tokens();
        size_off     += off_disk(state) ? state.disk_bytes : 0;
    }

    const float size_per_token = std::max<float>(1.0f, float(size_cur + size_off) / (std::max<size_t>(1, n_tokens_cur)));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (n_tokens_cur > limit_tokens_cur) {
            auto it = states.begin();

            while (disk_bounded && it != states.end() && off_disk(*it)) {
                ++it; // the disk budget owns this one
            }
            if (it == states.end()) {
                break;
            }

            const size_t size_front = it->size();

            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, size_front / (1024.0 * 1024.0));

            n_tokens_cur -= std::min<size_t>(n_tokens_cur, (size_t) it->prompt.n_tokens());
            size_cur     -= std::min(size_cur, size_front);

            erase_spill(*it);
            states.erase(it);
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size_cur / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}
