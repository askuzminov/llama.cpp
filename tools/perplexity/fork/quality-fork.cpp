#include "arg.h"
#include "common.h"
#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static json read_json(const std::string & path) {
    std::ifstream file(fs::u8path(path));
    require(file.good(), "cannot open plan");
    return json::parse(file);
}

static void write_json(const std::string & path, const json & value) {
    std::ofstream file(fs::u8path(path));
    file << value.dump(2) << '\n';
    require(file.good(), "cannot write report");
}

template<typename T> static void write_values(std::ostream & out, const std::vector<T> & values) {
    out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(T));
    require(out.good(), "cannot write reference");
}

template<typename T> static std::vector<T> read_values(std::istream & in, size_t count) {
    std::vector<T> values(count);
    in.read(reinterpret_cast<char *>(values.data()), values.size() * sizeof(T));
    require(in.good(), "truncated binary file");
    return values;
}

static double log_sum_exp(const float * logits, int32_t n) {
    double maximum = -INFINITY;
    for (int32_t i = 0; i < n; ++i) {
        require(std::isfinite(logits[i]), "non-finite logit");
        maximum = std::max(maximum, double(logits[i]));
    }
    double sum = 0;
    for (int32_t i = 0; i < n; ++i) {
        sum += std::exp(double(logits[i]) - maximum);
    }
    return maximum + std::log(sum);
}

static int run(common_params & params, const json & plan) {
    const auto action = plan.at("action").get<std::string>();
    if (action == "info" || action == "tokens") {
        auto mp = common_model_params_to_llama(params);
        mp.vocab_only = true;
        mp.n_gpu_layers = 0;
        llama_model_ptr model(llama_model_load_from_file(params.model.path.c_str(), mp));
        require(model != nullptr, "cannot load model metadata");
        // A vocab-only load skips hparams. Read the native length from the GGUF metadata instead.
        char architecture[256], length[64];
        require(llama_model_meta_val_str(model.get(), "general.architecture", architecture, sizeof(architecture)) > 0, "missing model architecture");
        const auto key = std::string(architecture) + ".context_length";
        require(llama_model_meta_val_str(model.get(), key.c_str(), length, sizeof(length)) > 0, "missing native context length");
        const auto native_context = std::stoll(length);
        require(native_context >= 2 && native_context < INT32_MAX, "invalid native context length");
        json report = {{"native_context", native_context},
                       {"vocab", llama_vocab_n_tokens(llama_model_get_vocab(model.get()))}};
        if (action == "tokens") {
            const auto corpus = fs::u8path(plan.at("corpus").get<std::string>());
            require(fs::file_size(corpus) <= (1ULL << 30), "corpus exceeds 1 GiB");
            std::ifstream file(corpus, std::ios::binary);
            require(file.good(), "cannot open corpus");
            std::string text((std::istreambuf_iterator<char>(file)), {});
            auto tokens = common_tokenize(llama_model_get_vocab(model.get()), text, true, false);
            const auto count = plan.at("count").get<int64_t>();
            require(count >= 2 && count <= INT32_MAX, "invalid token count");
            require(tokens.size() >= size_t(count), "corpus is too short; provide a longer text (no repetition is added)");
            report["corpus_tokens"] = tokens.size();
            tokens.resize(size_t(count));
            std::ofstream out(fs::u8path(plan.at("tokens").get<std::string>()), std::ios::binary);
            write_values(out, tokens);
        }
        write_json(plan.at("report"), report);
        return 0;
    }
    require(action == "capture" || action == "compare", "unknown action");
    const int64_t context = plan.at("context").get<int64_t>();
    require(context >= 2 && context < INT32_MAX, "invalid context");
    const auto positions = plan.at("positions").get<std::vector<uint32_t>>();
    require(!positions.empty() && positions.size() <= size_t(context), "invalid probe count");
    require(positions.back() == context - 1 && std::is_sorted(positions.begin(), positions.end()) &&
            std::adjacent_find(positions.begin(), positions.end()) == positions.end(), "probes must be sorted, unique and include the final position");
    const auto token_path = fs::u8path(plan.at("tokens").get<std::string>());
    require(fs::file_size(token_path) == (context + 1) * sizeof(llama_token), "wrong token file length");
    std::ifstream token_file(token_path, std::ios::binary);
    const auto tokens = read_values<llama_token>(token_file, size_t(context + 1));
    params.n_ctx = int32_t(context);
    params.fit_params = false;
    params.n_parallel = 1;
    params.kv_unified = true;
    params.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
    params.sampling.backend_sampling = false;
    const int rows = plan.value("max_outputs", 32);
    require(rows > 0 && rows <= 128, "max_outputs must be in 1..128");
    // positions in a decode range go in batches of at most decode_batch tokens, so their logits come from the decode
    // path of the model (a few tokens per ubatch: incomplete QSA tail blocks, the small-batch kernels) and not from a
    // prompt ubatch. 0: everything in batches of n_batch
    const int decode_batch = plan.value("decode_batch", 0);
    require(decode_batch >= 0 && decode_batch <= 64, "decode_batch must be in 0..64");
    std::vector<std::pair<int64_t, int64_t>> decode_ranges;
    for (const auto & r : plan.value("decode_ranges", json::array())) {
        const auto a = r.at(0).get<int64_t>(), b = r.at(1).get<int64_t>();
        require(0 <= a && a < b && b <= context, "decode_ranges must be [start, end) inside the context");
        require(decode_ranges.empty() || decode_ranges.back().second <= a, "decode_ranges must be sorted and disjoint");
        decode_ranges.emplace_back(a, b);
    }
    require(decode_ranges.empty() || decode_batch > 0, "decode_ranges need decode_batch");
    const auto in_decode = [&](int64_t pos) {
        for (const auto & r : decode_ranges) {
            if (pos >= r.first && pos < r.second) {
                return true;
            }
        }
        return false;
    };
    params.n_batch = std::min(params.n_batch, params.n_ctx);
    params.n_outputs_max = rows;
    params.n_outputs_max_per_seq = rows;
    auto init = common_init_from_params(params);
    auto * model = init->model();
    auto * ctx = init->context();
    require(model && ctx, "cannot load model/context");
    require(llama_n_ctx(ctx) >= context, "allocated context is too small");
    const int32_t vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
    require(vocab > 0, "invalid vocabulary");
    for (auto token : tokens) {
        require(token >= 0 && token < vocab, "token outside vocabulary");
    }
    const auto reference = fs::u8path(plan.at("reference").get<std::string>());
    const auto temporary = fs::u8path(plan.at("reference").get<std::string>() + ".tmp");
    std::ofstream out;
    std::ifstream in;
    const std::vector<uint32_t> header = {1, uint32_t(context), uint32_t(vocab), uint32_t(positions.size())};
    const uint64_t header_bytes = 8 + header.size() * 4 + tokens.size() * 4 + positions.size() * 4;
    require(positions.size() <= (UINT64_MAX - header_bytes) / (uint64_t(vocab) * sizeof(float)), "reference size overflow");
    const uint64_t expected_bytes = header_bytes + uint64_t(positions.size()) * vocab * sizeof(float);
    if (action == "capture") {
        out.open(temporary, std::ios::binary | std::ios::trunc);
        out.write("LQFREF01", 8);
        write_values(out, header);
        write_values(out, tokens);
        write_values(out, positions);
    } else {
        require(fs::file_size(reference) == expected_bytes, "wrong reference size");
        in.open(reference, std::ios::binary);
        char magic[8];
        in.read(magic, sizeof(magic));
        require(in.good() && std::memcmp(magic, "LQFREF01", 8) == 0, "wrong reference format");
        require(read_values<uint32_t>(in, header.size()) == header, "reference context/vocabulary/probes differ");
        require(read_values<llama_token>(in, tokens.size()) == tokens, "reference input tokens differ");
        require(read_values<uint32_t>(in, positions.size()) == positions, "reference positions differ");
    }
    json samples = json::array();
    common_batch batch(ctx);
    size_t next = 0;
    int32_t processed = 0;
    const auto start = std::chrono::steady_clock::now();
    while (processed < context) {
        batch.clear();
        std::vector<std::pair<int32_t, uint32_t>> outputs;
        const bool decode = in_decode(processed);
        const int32_t limit = decode ? decode_batch : params.n_batch;
        while (processed < context && batch.size() < limit) {
            // a batch is all decode or all prompt
            if (batch.size() > 0 && in_decode(processed) != decode) {
                break;
            }
            const bool score = next < positions.size() && positions[next] == uint32_t(processed);
            if (score && outputs.size() == size_t(rows)) {
                break;
            }
            const int32_t index = batch.add(tokens[processed], processed, 0, score);
            if (score) {
                outputs.emplace_back(index, positions[next++]);
            }
            ++processed;
        }
        require(llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decode failed");
        for (const auto & output : outputs) {
            const float * logits = llama_get_logits_ith(ctx, output.first);
            require(logits != nullptr, "missing logits");
            const double z = log_sum_exp(logits, vocab);
            const auto label = tokens[output.second + 1];
            json sample = {{"position", output.second}, {"nll", z - logits[label]}, {"decode", decode}};
            if (action == "capture") {
                out.write(reinterpret_cast<const char *>(logits), size_t(vocab) * sizeof(float));
                require(out.good(), "cannot write logits");
            } else {
                auto ref = read_values<float>(in, size_t(vocab));
                const double rz = log_sum_exp(ref.data(), vocab);
                double kl = 0, tv = 0, max_diff = 0;
                for (int32_t i = 0; i < vocab; ++i) {
                    const double lp = ref[i] - rz, lq = logits[i] - z;
                    kl += std::exp(lp) * (lp - lq);
                    tv += std::abs(std::exp(lp) - std::exp(lq));
                    max_diff = std::max(max_diff, std::abs(double(ref[i]) - logits[i]));
                }
                sample["reference_nll"] = rz - ref[label];
                sample["delta_nll"] = sample["nll"].get<double>() - (rz - ref[label]);
                sample["kl"] = std::max(0.0, kl);
                sample["tv"] = tv / 2;
                sample["max_logit_difference"] = max_diff;
                sample["top1_equal"] = std::max_element(logits, logits + vocab) - logits == std::max_element(ref.begin(), ref.end()) - ref.begin();
            }
            samples.push_back(std::move(sample));
        }
        if (!decode || processed == context) {
            std::cerr << "processed " << processed << '/' << context << ", probes " << samples.size() << '/' << positions.size() << '\n';
        }
    }
    if (action == "capture") {
        out.close();
        require(out.good() && fs::file_size(temporary) == expected_bytes, "incomplete reference");
        fs::rename(temporary, reference);
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    write_json(plan.at("report"), {{"context", context}, {"processed", processed}, {"vocab", vocab},
               {"native_context", llama_model_n_ctx_train(model)}, {"max_outputs", rows}, {"decode_batch", decode_batch},
               {"reference_bytes", expected_bytes}, {"seconds", elapsed}, {"samples", samples}});
    return 0;
}

int main(int argc, char ** argv) {
    try {
        common_init();
        std::string path;
        std::vector<char *> args;
        for (int i = 0; i < argc; ++i) {
            if (std::strcmp(argv[i], "--probe-plan") == 0) {
                require(i + 1 < argc && path.empty(), "--probe-plan requires one JSON file");
                path = argv[++i];
            } else {
                args.push_back(argv[i]);
            }
        }
        require(!path.empty(), "usage: llama-quality-fork --probe-plan PLAN.json [model options]");
        common_params params;
        params.escape = false;
        const int n_args = int(args.size());
        args.push_back(nullptr);
        if (!common_params_parse(n_args, args.data(), params, LLAMA_EXAMPLE_PERPLEXITY)) {
            return 1;
        }
        llama_backend_init();
        llama_numa_init(params.numa);
        const int result = run(params, read_json(path));
        llama_backend_free();
        return result;
    } catch (const std::exception & error) {
        std::cerr << "quality: " << error.what() << '\n';
        return 1;
    }
}
