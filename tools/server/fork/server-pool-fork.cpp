// fork: --pool, llama-server as a router over remote llama-server instances of one model. every node serves the same
// model with the same context, other flags may differ; the agents use the address of the pool:
// - a request goes to a node with a free slot (fewer requests in flight than its slots); a new conversation to the node
//   with the most free slots by weight
// - a conversation keeps to the node that holds its prompt cache: the pool hashes the prompt message by message (first
//   the tools and the system prompt) and remembers which node served each prefix, and a request goes to the node with
//   the longest prefix of it. there the cache is reused, anywhere else the whole context is computed again (100K
//   tokens are about 2.4 minutes of prefill on the 395). X-Conversation-Id or X-Session-Id, when a client sends one,
//   names the conversation instead
// - when that node has no free slot, a conversation of at least move_tokens waits for it as long as a recompute
//   elsewhere would take (its tokens at the prefill speed of that node, measured from the timings of the answers, the
//   ini prefill_tps until then), up to wait_max seconds; a shorter one goes to a free node at once
// - with no free slot anywhere the requests wait in the pool, the oldest first; a client that goes away leaves
// - a conversation that goes to another node takes its cached state along when both nodes can (the fork with the prompt
//   cache, the same model and configuration) and the transfer is shorter than computing it again: the pool asks the
//   node that holds it for the state, reads it in chunks and writes them to the other node, from RAM or the file of a
//   state on disk to RAM, with nothing written to disk. --api-key on the nodes guards it (api_key in the pool ini)
// - a node can have several urls in the order of preference (a direct cable, then the WiFi): the pool uses the first
//   that answers and tries the ones before it every probe_s seconds
// - /pool is a dashboard of the nodes (slots, speed, memory), the queue, the conversations and the decisions
// the nodes run llama-server; the one of the fork also gives the memory of its devices, its model and the signature of
// its prompt cache (GET /pool/node), and moves conversations (POST /pool/export, /pool/import/...)

#include "server-pool-fork.h"

#include "server-common.h"
#include "server-context.h"
#include "server-models.h" // server_http_proxy

#include "common.h"
#include "fork/common-fork.h" // common_host_mem_available
#include "log.h"

#include "ggml-backend.h"

#include <cpp-httplib/httplib.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "server-pool-ui-fork.inc" // POOL_UI_PARTS: the page of the dashboard

//
// helpers
//

static int64_t pool_wall_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

static std::string pool_trim(const std::string & s) {
    const size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) {
        return "";
    }
    const size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string pool_lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
    return s;
}

static uint64_t pool_hash(uint64_t h, const std::string & s) {
    for (unsigned char c : s) {
        h = (h ^ c) * 0x100000001b3ull; // FNV-1a
    }
    return h ^ ((uint64_t) s.size() * 0x9E3779B97F4A7C15ull);
}

static constexpr uint64_t POOL_HASH_SEED = 0xcbf29ce484222325ull;
static constexpr uint64_t POOL_ID_SEED   = 0x84222325cbf29ce4ull;

static std::string pool_header(const std::map<std::string, std::string> & headers, const char * name_lower) {
    for (const auto & [k, v] : headers) {
        if (pool_lower(k) == name_lower) {
            return v;
        }
    }
    return "";
}

static server_http_res_ptr pool_error(int status, const std::string & message) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->data = safe_json_to_str({{"error", {{"code", status}, {"message", message}, {"type", "pool_error"}}}});
    return res;
}

// [section] then key = value lines; ; or # starts a comment
using pool_ini = std::vector<std::pair<std::string, std::map<std::string, std::string>>>;

static pool_ini pool_read_ini(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open " + path);
    }
    pool_ini ini;
    std::string line;
    int n_line = 0;
    while (std::getline(f, line)) {
        n_line++;
        std::string t = pool_trim(line);
        if (t.empty() || t[0] == ';' || t[0] == '#') {
            continue;
        }
        if (t[0] == '[') {
            if (t.back() != ']') {
                throw std::runtime_error(path + ":" + std::to_string(n_line) + ": a section ends with ]");
            }
            ini.push_back({pool_trim(t.substr(1, t.size() - 2)), {}});
            continue;
        }
        const size_t eq = t.find('=');
        if (eq == std::string::npos || ini.empty()) {
            throw std::runtime_error(path + ":" + std::to_string(n_line) + ": expected key = value inside a section");
        }
        std::string val = pool_trim(t.substr(eq + 1));
        for (const char * c : {" ;", " #", "\t;", "\t#"}) {
            const size_t at = val.find(c);
            if (at != std::string::npos) {
                val = pool_trim(val.substr(0, at));
            }
        }
        ini.back().second[pool_lower(pool_trim(t.substr(0, eq)))] = val;
    }
    return ini;
}

// http://host:port, https://host:port, [ipv6]:port; the path is ignored
static void pool_parse_url(const std::string & url, std::string & scheme, std::string & host, int & port) {
    const size_t p = url.find("://");
    scheme = p == std::string::npos ? "http" : pool_lower(url.substr(0, p));
    std::string rest = p == std::string::npos ? url : url.substr(p + 3);
    rest = rest.substr(0, rest.find('/'));
    port = scheme == "https" ? 443 : 80;
    if (!rest.empty() && rest[0] == '[') {
        const size_t rb = rest.find(']');
        if (rb == std::string::npos) {
            throw std::runtime_error("bad url " + url);
        }
        host = rest.substr(1, rb - 1);
        rest = rest.substr(rb + 1);
        if (!rest.empty() && rest[0] == ':') {
            port = std::stoi(rest.substr(1));
        }
    } else {
        const size_t colon = rest.rfind(':');
        host = colon == std::string::npos ? rest : rest.substr(0, colon);
        if (colon != std::string::npos) {
            port = std::stoi(rest.substr(colon + 1));
        }
    }
    if (host.empty() || port <= 0 || (scheme != "http" && scheme != "https")) {
        throw std::runtime_error("bad url " + url);
    }
}

//
// the prompt as a chain of hashes: [0] the tools and the system prompt, [i] everything through the i-th message
//

struct pool_key {
    std::vector<uint64_t> chain;
    std::vector<int64_t>  chars; // characters through chain[i]
    std::string           id;    // X-Conversation-Id or X-Session-Id, "" = none
};

static std::string pool_canon_content(const json & c) {
    return c.is_string() ? c.get<std::string>() : c.dump();
}

// what of a message reaches the prompt: the role, the content, the tool calls and their ids
static std::string pool_canon_message(const json & m) {
    if (!m.is_object()) {
        return m.dump();
    }
    std::string s = json_value(m, "role", std::string());
    s += '\x1f';
    if (m.contains("content") && !m.at("content").is_null()) {
        s += pool_canon_content(m.at("content"));
    }
    for (const char * k : {"tool_calls", "tool_call_id", "name"}) {
        if (m.contains(k) && !m.at(k).is_null()) {
            s += '\x1f';
            s += m.at(k).dump();
        }
    }
    return s;
}

static pool_key pool_key_of(const std::string & path, const json & body, const std::map<std::string, std::string> & headers) {
    pool_key k;
    k.id = pool_header(headers, "x-conversation-id");
    if (k.id.empty()) {
        k.id = pool_header(headers, "x-session-id");
    }
    if (!body.is_object()) {
        return k;
    }
    uint64_t h = POOL_HASH_SEED;
    auto push = [&](const std::string & s) {
        h = pool_hash(h, s);
        k.chain.push_back(h);
        k.chars.push_back((k.chars.empty() ? 0 : k.chars.back()) + (int64_t) s.size());
    };
    auto dump_of = [&](const char * key) {
        return body.contains(key) ? body.at(key).dump() : std::string();
    };
    auto push_messages = [&](const char * key) {
        if (body.contains(key) && body.at(key).is_array()) {
            for (const auto & m : body.at(key)) {
                push(pool_canon_message(m));
            }
        }
    };
    if (path.find("chat/completions") != std::string::npos) {
        push("chat\x1f" + dump_of("tools"));
        push_messages("messages");
    } else if (path.find("/messages") != std::string::npos) {
        push("anthropic\x1f" + dump_of("tools") + "\x1f" + dump_of("system"));
        push_messages("messages");
    } else if (path.find("responses") != std::string::npos) {
        push("responses\x1f" + dump_of("tools") + "\x1f" + dump_of("instructions"));
        if (body.contains("input") && body.at("input").is_array()) {
            push_messages("input");
        } else if (body.contains("input")) {
            push(pool_canon_content(body.at("input")));
        }
    } else if (path.find("completion") != std::string::npos) {
        push("completion");
        if (body.contains("prompt")) {
            const std::string p = pool_canon_content(body.at("prompt"));
            for (size_t i = 0; i < p.size(); i += 4096) {
                push(p.substr(i, 4096));
            }
        }
    }
    return k;
}

//
// the pool
//

struct pool_config {
    std::string name;            // the name the dashboard shows
    int     poll_ms     = 1000;  // how often each node is asked for its health and slots
    int64_t move_tokens = 2048;  // a conversation this long waits for its node, a shorter one moves at once
    int     wait_max_s  = 300;   // the longest wait for the node of a conversation
    double  prefill_tps = 700.0; // the prefill speed a wait is weighed against (tokens/s)
    double  chars_tok   = 3.5;   // characters per token, for the estimates
    int     ttl_s       = 6*3600; // how long the pool remembers which node holds a prefix
    std::string api_key;         // sent to the nodes that have none of their own
    std::string model;           // the name of the model in /v1/models, the same from any node; "" = as a node names it
    bool    migrate    = true;   // a conversation moves with its cached state when the nodes can
    double  link_mbps  = 500.0;  // the speed of a link not measured yet and not given after its url (Mbit/s)
    int     probe_s    = 10;     // how often a node on a fallback link tries the links before it (s)
    bool    same_model = true;   // a node of another model than the most of the others takes no request
};

// a chunk of the move of a conversation, and of the probe of a link
static constexpr uint64_t POOL_CHUNK = 16ull << 20;

// one way to a node: a direct cable, the WiFi
struct pool_link {
    std::string url;
    std::string scheme;
    std::string host;
    int         port     = 0;
    double      mbps_cfg = 0.0;  // the speed given after the url (Mbit/s), 0 = none
    // under impl::mu. measured (bytes/s, 0 = not yet): a probe when the link starts to serve, then the moves
    double      bps_read  = 0.0; // from the node to the pool
    double      bps_write = 0.0; // from the pool to the node
    bool        probed    = false; // the probe was tried since the link last answered
    std::string error;           // why it failed the last time it was tried, "" = it answered
};

static std::string pool_base(const pool_link & l) {
    return l.scheme + "://" + (l.host.find(':') != std::string::npos ? "[" + l.host + "]" : l.host) + ":" + std::to_string(l.port);
}

// what a request is for a move: the kind POST /pool/export reads, "" = no move (the prompt is not a conversation)
static const char * pool_kind_of(const std::string & path) {
    if (path.find("chat/completions") != std::string::npos) {
        return "chat";
    }
    if (path.find("/messages") != std::string::npos) {
        return "anthropic";
    }
    if (path.find("responses") != std::string::npos) {
        return "responses";
    }
    if (path.find("completion") != std::string::npos && path.find("infill") == std::string::npos) {
        return "completion";
    }
    return "";
}

// the message of an error answer of a node, or its status
static std::string pool_error_text(const httplib::Result & res) {
    if (!res) {
        return httplib::to_string(res.error());
    }
    const json j = json::parse_no_throw(res->body);
    if (!j.is_discarded() && j.is_object() && j.contains("error") && j.at("error").is_object()) {
        return "HTTP " + std::to_string(res->status) + ": " + json_value(j.at("error"), "message", std::string());
    }
    return "HTTP " + std::to_string(res->status);
}

// a model of the list of a router node (GET /models) under `name`: the name of its preset or an alias of it, as the
// router takes it in a request
static bool pool_model_named(const json & m, const std::string & name) {
    if (json_value(m, "id", std::string()) == name) {
        return true;
    }
    if (m.contains("aliases") && m.at("aliases").is_array()) {
        for (const auto & a : m.at("aliases")) {
            if (a.is_string() && a.get<std::string>() == name) {
                return true;
            }
        }
    }
    return false;
}

struct pool_node {
    // config, fixed after the start
    std::string name;
    std::vector<pool_link> links; // in the order of preference: a direct cable first, the WiFi after it
    int         slots_cfg   = 0;   // 0 = what the node reports
    double      weight      = 1.0;
    double      prefill_cfg = 0.0; // its prefill speed (tokens/s), 0 = the one of the pool
    std::string api_key;
    std::string model;           // a node in router mode (serve.bat): the model on it, set in every request

    // state, under impl::mu (the bps and error of the links too)
    int         link    = 0;     // the link in use
    int64_t     t_probe = 0;     // the last try of the links before it, while on a fallback
    std::string status = "down"; // up, down, loading, unloaded
    bool        draining = false;
    bool        mismatch = false; // another model than the most of the nodes (same_model): it takes no request
    std::string signature;       // the portable signature of its prompt cache, "" = unknown (GET /pool/node)
    bool        migrate = false; // it moves conversations: the fork with a prompt cache
    uint64_t    cache_ram = 0;   // the RAM limit of its prompt cache, the largest state it takes; 0 = none
    json        model_info = json::object(); // its model (GET /pool/node): parameters, size, type, vocabulary
    std::string error;
    int64_t     t_ok      = 0;   // the last good poll (ms, ggml_time_ms)
    int         slots     = 0;   // capacity
    int         inflight  = 0;   // requests proxied and not finished
    int         busy      = 0;   // slots processing, at the last poll
    int         external  = 0;   // of them, requests that did not come through the pool
    uint64_t    n_requests = 0;
    double      gen_tps    = 0.0;
    double      prompt_tps = 0.0;
    double      prefill_tps = 0.0; // measured: the timings of the prompts of move_tokens and more it read, 0 = none yet
    // measured: the part of a prefix of other conversations it reused for a prompt (cache_n of the answers), -1 = none
    // yet. a transformer takes all of it, a hybrid model (qwen4exp: gated delta net) or SWA only up to a checkpoint
    // within it, which a node often does not have
    double      share = -1.0;
    json        props   = json::object();
    json        devices = json::array();
    int64_t     t_props = 0;

    struct slot_t {
        int     id          = 0;
        int64_t id_task     = -1;
        bool    processing  = false;
        int     n_ctx       = 0;
        int64_t n_prompt    = 0;
        int64_t n_processed = 0;
        int64_t n_cached    = 0;
        int64_t n_decoded   = 0;
        double  gen_tps     = 0.0;
        double  prompt_tps  = 0.0;
        int64_t t           = 0;
    };
    std::vector<slot_t> slot_state;
};

struct pool_req {
    uint64_t             id = 0;
    std::string          conv;
    bool                 conv_new      = false;
    std::string          path;
    int64_t              t_arrive      = 0;
    std::vector<int64_t> cached;             // per node: the tokens of this prompt it holds, estimated
    std::vector<int64_t> shared;             // per node: of them, those from other conversations (before its share)
    std::vector<int64_t> shared_max;         // per node: to the end of the message after them, where this prompt differs
    int                  depth         = 0;  // the most messages of the prompt a node holds
    int64_t              prompt_tokens = 0;  // the whole prompt, estimated
    int                  node          = -1; // where it went
    int64_t              t_start       = 0;
    std::string          stage;              // "moving": its state comes from another node

    int64_t cached_max() const {
        return cached.empty() ? 0 : *std::max_element(cached.begin(), cached.end());
    }
};

struct server_pool_fork::impl {
    const common_params & params;
    pool_config cfg;
    std::vector<pool_node> nodes; // fixed size after the start

    std::mutex mu;
    std::condition_variable cv;
    std::deque<pool_req *>       queue;   // waiting, the oldest first (the requests live on the HTTP threads)
    std::map<uint64_t, pool_req> running; // proxied and not finished
    uint64_t next_id = 1;

    // a prefix hash -> the nodes that served a prompt with it, and when
    std::unordered_map<uint64_t, std::vector<std::pair<int, int64_t>>> affinity;

    // the last prefix hash of a request -> its conversation and when: the next request of the conversation starts with
    // it (a history only grows), while the system prompt and the first messages can be the same in many
    std::unordered_map<uint64_t, std::pair<std::string, int64_t>> tails;

    struct conv_t {
        int      node = -1;
        int      depth = 0;
        int64_t  est_tokens = 0;
        int64_t  prompt_tokens = 0;
        int64_t  t_last = 0;
        uint64_t n = 0;
        std::vector<int64_t> held; // per node: the tokens of the conversation it took last
    };
    std::unordered_map<std::string, conv_t> convs;

    double state_bpt = 0.0; // bytes of a moved state per token (an average of the moves), 0 = no move yet

    struct event_t {
        int64_t     t;
        std::string kind;
        std::string conv;
        std::string node;
        std::string detail;
    };
    std::deque<event_t> events;
    int64_t t_trim = 0;

    std::atomic<bool>        stopping{false};
    std::vector<std::thread> pollers;
    std::mutex               mu_stop;
    std::condition_variable  cv_stop;

    explicit impl(const common_params & params) : params(params) {}

    void load(const std::string & path) {
        for (const auto & sec : pool_read_ini(path)) {
            const std::string & section = sec.first;
            const auto & kv = sec.second;
            auto get = [&](const char * key, const std::string & def) {
                auto it = kv.find(key);
                return it == kv.end() ? def : it->second;
            };
            if (section == "pool") {
                cfg.name        = get("name", cfg.name);
                cfg.poll_ms     = std::max(100, std::stoi(get("poll_ms", std::to_string(cfg.poll_ms))));
                cfg.move_tokens = std::stoll(get("move_tokens", std::to_string(cfg.move_tokens)));
                cfg.wait_max_s  = std::stoi(get("wait_max", std::to_string(cfg.wait_max_s)));
                cfg.prefill_tps = std::max(1.0, std::stod(get("prefill_tps", std::to_string(cfg.prefill_tps))));
                cfg.chars_tok   = std::max(0.5, std::stod(get("chars_per_token", std::to_string(cfg.chars_tok))));
                cfg.ttl_s       = std::stoi(get("affinity_ttl", std::to_string(cfg.ttl_s)));
                cfg.api_key     = get("api_key", cfg.api_key);
                cfg.model       = get("model", cfg.model);
                cfg.migrate     = std::stoi(get("migrate", cfg.migrate ? "1" : "0")) != 0;
                cfg.link_mbps   = std::max(1.0, std::stod(get("link_mbps", std::to_string(cfg.link_mbps))));
                cfg.probe_s     = std::max(1, std::stoi(get("probe_s", std::to_string(cfg.probe_s))));
                cfg.same_model  = std::stoi(get("same_model", cfg.same_model ? "1" : "0")) != 0;
            } else if (section.rfind("node.", 0) == 0 || section.rfind("node ", 0) == 0) {
                pool_node n;
                n.name = pool_trim(section.substr(5));
                // url = <url> [Mbit/s], <url> [Mbit/s], ...: the first that answers serves
                const std::string urls = get("url", "");
                size_t at = 0;
                while (at < urls.size()) {
                    size_t comma = urls.find(',', at);
                    const std::string item = pool_trim(urls.substr(at, comma == std::string::npos ? std::string::npos : comma - at));
                    at = comma == std::string::npos ? urls.size() : comma + 1;
                    if (item.empty()) {
                        continue;
                    }
                    pool_link l;
                    const size_t sp = item.find_first_of(" \t");
                    l.url = item.substr(0, sp);
                    if (sp != std::string::npos) {
                        l.mbps_cfg = std::stod(pool_trim(item.substr(sp)));
                    }
                    pool_parse_url(l.url, l.scheme, l.host, l.port);
                    n.links.push_back(l);
                }
                if (n.name.empty() || n.links.empty()) {
                    throw std::runtime_error(path + ": [" + section + "] needs a name and url = http://host:port");
                }
                n.slots_cfg   = std::stoi(get("slots", "0"));
                n.weight      = std::max(0.01, std::stod(get("weight", "1")));
                n.prefill_cfg = std::max(0.0, std::stod(get("prefill_tps", "0")));
                n.api_key     = get("api_key", "");
                n.model       = get("model", "");
                nodes.push_back(std::move(n));
            } else {
                throw std::runtime_error(path + ": unknown section [" + section + "], expected [pool] or [node.<name>]");
            }
        }
        if (nodes.empty()) {
            throw std::runtime_error(path + ": no [node.<name>] section");
        }
        for (auto & n : nodes) {
            if (n.api_key.empty()) {
                n.api_key = cfg.api_key;
            }
        }
    }

    //
    // under mu
    //

    static bool usable(const pool_node & n) {
        return n.status == "up" && !n.draining && !n.mismatch;
    }

    // the prefill speed of a node: measured (note_timings), else its prefill_tps, else the one of [pool]
    double prefill_of(int n) const {
        if (nodes[n].prefill_tps > 0) {
            return nodes[n].prefill_tps;
        }
        return nodes[n].prefill_cfg > 0 ? nodes[n].prefill_cfg : cfg.prefill_tps;
    }

    // the speed of the link a node is on, from the node or to it (bytes/s): measured, or given after its url, or the
    // default
    double link_bps(const pool_node & n, bool write) const {
        const pool_link & l = n.links[n.link];
        const double m = write ? l.bps_write : l.bps_read;
        return m > 0 ? m : (l.mbps_cfg > 0 ? l.mbps_cfg : cfg.link_mbps) * 1e6 / 8;
    }

    // a conversation can move from `from` to `to` with its state: both nodes of the fork with the prompt cache, of the
    // same model and configuration (the portable signature of the cache), `from` answers
    bool can_migrate(int from, int to) const {
        const pool_node & a = nodes[from];
        const pool_node & b = nodes[to];
        return cfg.migrate && from != to && a.status == "up" && a.migrate && b.migrate && !a.signature.empty() &&
               a.signature == b.signature;
    }

    // a move of `bytes` over a link of `bps` bytes/s: the requests, the transfer, and the copies on the two nodes at
    // 2 GB/s (the export reads the file of a state on disk or copies one in RAM, the target restores it into a slot)
    static double move_s(double bytes, double bps) {
        return 0.2 + bytes / bps + bytes / 2e9;
    }

    // how long a move of a state of `tokens` takes through the pool: the slower of the two links, with the bytes per
    // token of the moves so far. 0 before the first move: unknown, a conversation then moves rather than waits
    double transfer_s(int from, int to, int64_t tokens) const {
        if (state_bpt <= 0) {
            return 0.0;
        }
        return move_s(tokens * state_bpt, std::min(link_bps(nodes[from], false), link_bps(nodes[to], true)));
    }

    static int free_slots(const pool_node & n) {
        return usable(n) ? n.slots - n.inflight - n.external : 0;
    }

    // the free node with the most of the prompt cached, then the most free slots by weight, then the fewest requests
    int best_free(const std::vector<int64_t> & cached) const {
        int best = -1;
        for (int i = 0; i < (int) nodes.size(); ++i) {
            const int f = free_slots(nodes[i]);
            if (f <= 0) {
                continue;
            }
            if (best < 0) {
                best = i;
                continue;
            }
            const int64_t ci = cached.empty() ? 0 : cached[i];
            const int64_t cb = cached.empty() ? 0 : cached[best];
            if (ci != cb) {
                best = ci > cb ? i : best;
                continue;
            }
            const double si = nodes[i].weight * f / std::max(1, nodes[i].slots);
            const double sb = nodes[best].weight * free_slots(nodes[best]) / std::max(1, nodes[best].slots);
            if (si > sb + 1e-9 || (si > sb - 1e-9 && nodes[i].n_requests < nodes[best].n_requests)) {
                best = i;
            }
        }
        return best;
    }

    // the usable node with the most of the prompt cached, -1 = none holds any of it
    int holder(const pool_req & r) const {
        int best = -1;
        for (int i = 0; i < (int) nodes.size() && i < (int) r.cached.size(); ++i) {
            if (usable(nodes[i]) && r.cached[i] > 0 && (best < 0 || r.cached[i] > r.cached[best])) {
                best = i;
            }
        }
        return best;
    }

    // where a request goes now, -1 = it waits. a busy node with more of the prompt cached than the best free one is
    // worth waiting for when the difference is at least move_tokens, as long as computing it elsewhere would take.
    // only for the history of a conversation: a new one holds just what it shares with others (the system prompt, the
    // first messages of an agent), which a free node computes once and then has for all of them; waiting for the
    // node that served the first conversation would put every new one on it
    int choose(const pool_req & r, int64_t now) const {
        const int alt = best_free(r.cached);
        const int hold = holder(r);
        if (alt < 0 || hold < 0 || r.conv_new) {
            return alt;
        }
        const int64_t gain = r.cached[hold] - r.cached[alt];
        if (gain >= cfg.move_tokens) {
            // waiting is worth at most what going elsewhere costs: computing the difference, or moving the state
            double limit_s = std::min<double>(cfg.wait_max_s, gain / prefill_of(alt));
            if (can_migrate(hold, alt)) {
                limit_s = std::min(limit_s, transfer_s(hold, alt, r.cached[hold]));
            }
            if ((now - r.t_arrive) / 1000.0 < limit_s) {
                return -1;
            }
        }
        return alt;
    }

    // what each node holds of the prompt, and the conversation the request continues
    void lookup(const pool_key & k, pool_req & r) const {
        r.prompt_tokens = k.chars.empty() ? 0 : (int64_t) (k.chars.back() / cfg.chars_tok);
        r.cached.assign(nodes.size(), 0);
        r.shared_max.assign(nodes.size(), 0);
        std::vector<int> depth(nodes.size(), 0);
        if (!k.id.empty()) {
            auto it = affinity.find(pool_hash(POOL_ID_SEED, k.id));
            if (it != affinity.end()) {
                for (const auto & e : it->second) {
                    if (nodes[e.first].status != "down") {
                        depth[e.first] = (int) k.chain.size() - 1;
                        r.cached[e.first] = r.prompt_tokens;
                    }
                }
            }
        }
        // [0] holds only the tools, the first message (often the system prompt) is [1]
        for (int i = (int) k.chain.size() - 1; i >= 1; --i) {
            auto it = affinity.find(k.chain[i]);
            if (it == affinity.end()) {
                continue;
            }
            for (const auto & e : it->second) {
                if (nodes[e.first].status != "down" && depth[e.first] == 0) {
                    depth[e.first] = i;
                    r.cached[e.first] = (int64_t) (k.chars[i] / cfg.chars_tok);
                    r.shared_max[e.first] = (int64_t) (k.chars[std::min(i + 1, (int) k.chars.size() - 1)] / cfg.chars_tok);
                }
            }
        }
        r.depth = depth.empty() ? 0 : *std::max_element(depth.begin(), depth.end());

        r.conv.clear();
        if (!k.id.empty()) {
            r.conv = k.id.substr(0, 40);
            r.conv_new = convs.find(r.conv) == convs.end();
            count_shared(r);
            return;
        }
        for (int i = (int) k.chain.size() - 1; i >= 0 && r.conv.empty(); --i) {
            auto it = tails.find(k.chain[i]);
            if (it != tails.end()) {
                r.conv = it->second.first;
            }
        }
        if (r.conv.empty() && !k.chain.empty()) {
            char buf[32];
            snprintf(buf, sizeof(buf), "c%08x", (unsigned) (k.chain.back() >> 32));
            r.conv = buf;
            r.conv_new = true;
        }
        count_shared(r);
    }

    // what a node served of the conversation counts in full, what it holds from other conversations (the system prompt
    // and the tools of an agent, the first messages) at the share it reused of such a prefix so far, 0 until measured
    void count_shared(pool_req & r) const {
        auto it = r.conv_new ? convs.end() : convs.find(r.conv);
        r.shared.assign(r.cached.size(), 0);
        for (int i = 0; i < (int) r.cached.size(); ++i) {
            const bool served = it != convs.end() && i < (int) it->second.held.size() && it->second.held[i] > 0;
            if (!served) {
                r.shared[i] = r.cached[i];
                r.cached[i] = (int64_t) (r.cached[i] * std::max(0.0, nodes[i].share));
            }
        }
    }

    void remember(const pool_key & k, const pool_req & r, int node, int64_t now) {
        auto put = [&](uint64_t h) {
            auto & holders = affinity[h];
            for (auto & e : holders) {
                if (e.first == node) {
                    e.second = now;
                    return;
                }
            }
            if (holders.size() >= 8) {
                auto oldest = std::min_element(holders.begin(), holders.end(),
                    [](const auto & a, const auto & b) { return a.second < b.second; });
                *oldest = {node, now};
            } else {
                holders.push_back({node, now});
            }
        };
        for (uint64_t h : k.chain) {
            put(h);
        }
        if (!k.id.empty()) {
            put(pool_hash(POOL_ID_SEED, k.id));
        } else if (!k.chain.empty() && !r.conv.empty()) {
            // the conversation moves on: the tail it continued from is no longer its end
            for (int i = (int) k.chain.size() - 2; i >= 0; --i) {
                auto it = tails.find(k.chain[i]);
                if (it != tails.end() && it->second.first == r.conv) {
                    tails.erase(it);
                    break;
                }
            }
            tails[k.chain.back()] = {r.conv, now};
        }
    }

    void event(const char * kind, const std::string & conv, int node, const std::string & detail) {
        events.push_back({pool_wall_ms(), kind, conv, node >= 0 ? nodes[node].name : "", detail});
        while (events.size() > 300) {
            events.pop_front();
        }
    }

    void trim(int64_t now) {
        if (now - t_trim < 60000) {
            return;
        }
        t_trim = now;
        const int64_t t_old = now - (int64_t) cfg.ttl_s * 1000;
        for (auto it = affinity.begin(); it != affinity.end(); ) {
            auto & holders = it->second;
            holders.erase(std::remove_if(holders.begin(), holders.end(),
                [&](const auto & e) { return e.second < t_old; }), holders.end());
            it = holders.empty() ? affinity.erase(it) : std::next(it);
        }
        for (auto it = tails.begin(); it != tails.end(); ) {
            it = it->second.second < t_old ? tails.erase(it) : std::next(it);
        }
        const int64_t w_old = pool_wall_ms() - (int64_t) cfg.ttl_s * 1000;
        for (auto it = convs.begin(); it != convs.end(); ) {
            it = it->second.t_last < w_old ? convs.erase(it) : std::next(it);
        }
    }

    // a node for a request: waits for a free slot, -1 when the client went away, -2 when no node is up
    int acquire(pool_req & r, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mu);
        queue.push_back(&r);
        int64_t t_none = 0;
        while (true) {
            const int64_t now = ggml_time_ms();
            if (should_stop()) {
                queue.erase(std::find(queue.begin(), queue.end(), &r));
                event("gone", r.conv, -1, "the client left while it waited");
                return -1;
            }
            const int n = choose(r, now);
            if (n >= 0) {
                // an older request that would take the same node goes first
                bool older = false;
                for (pool_req * q : queue) {
                    if (q == &r) {
                        break;
                    }
                    if (choose(*q, now) == n) {
                        older = true;
                        break;
                    }
                }
                if (!older) {
                    queue.erase(std::find(queue.begin(), queue.end(), &r));
                    nodes[n].inflight++;
                    nodes[n].n_requests++;
                    r.node = n;
                    r.t_start = now;
                    running[r.id] = r;
                    return n;
                }
            }
            bool any = false;
            for (const auto & node : nodes) {
                any = any || usable(node);
            }
            if (!any) {
                t_none = t_none == 0 ? now : t_none;
                if (now - t_none > 30000) {
                    queue.erase(std::find(queue.begin(), queue.end(), &r));
                    event("fail", r.conv, -1, "no node is up");
                    return -2;
                }
            } else {
                t_none = 0;
            }
            cv.wait_for(lk, std::chrono::milliseconds(200));
        }
    }

    void release(uint64_t id, int status) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = running.find(id);
        if (it == running.end()) {
            return;
        }
        const pool_req & r = it->second;
        nodes[r.node].inflight--;
        char buf[96];
        snprintf(buf, sizeof(buf), "HTTP %d, %.1f s", status, (ggml_time_ms() - r.t_start) / 1000.0);
        event(status >= 400 ? "error" : "done", r.conv, r.node, buf);
        running.erase(it);
        cv.notify_all();
    }

    // the last "timings" object of an answer (the final event of a stream, or the end of a JSON body; it holds no
    // nested object):
    // - the prompt it read, when that was at least move_tokens - the size of the prefills the speed is weighed for, where
    //   a short prompt shows the overhead of a request more than the speed
    // - with `shared` (the fraction of the prompt the node held from other conversations, estimated from the
    //   characters; at least move_tokens of it) the share of it the node took from its cache (cache_n). a node that took
    //   more than reaches to the end of the next message (`shared_max`), where the prompt differs from the others, held
    //   this conversation itself (one the pool forgot, as after its restart): no sample
    void note_timings(int n, const std::string & tail, double shared, double shared_max) {
        const size_t at  = tail.rfind("\"timings\":");
        const size_t beg = at == std::string::npos ? std::string::npos : tail.find('{', at);
        const size_t end = beg == std::string::npos ? std::string::npos : tail.find('}', beg);
        if (end == std::string::npos) {
            return;
        }
        const json t = json::parse_no_throw(tail.substr(beg, end - beg + 1));
        if (!t.is_object()) {
            return;
        }
        const double n_prompt = json_value(t, "prompt_n", 0.0);
        const double n_cache  = json_value(t, "cache_n", 0.0);
        const double t_ms     = json_value(t, "prompt_ms", 0.0);
        auto avg = [](double old, double x, double none) { return old > none ? 0.5*old + 0.5*x : x; };
        std::lock_guard<std::mutex> lk(mu);
        if (n_prompt >= cfg.move_tokens && t_ms > 0) {
            nodes[n].prefill_tps = avg(nodes[n].prefill_tps, n_prompt*1000.0/t_ms, 0.0);
        }
        const double total = n_prompt + n_cache;
        if (shared > 0 && total > 0 && n_cache <= shared_max * total) {
            nodes[n].share = avg(nodes[n].share, std::min(1.0, n_cache / (shared * total)), -1.0);
        }
    }

    // the least busy node that is up, for the requests that take no slot (tokenize, models, props)
    int any_node() const {
        int best = -1;
        for (int i = 0; i < (int) nodes.size(); ++i) {
            if (nodes[i].status == "up" && (best < 0 || nodes[i].inflight < nodes[best].inflight)) {
                best = i;
            }
        }
        return best;
    }

    std::map<std::string, std::string> headers_for(const pool_node & node, const std::map<std::string, std::string> & in) const {
        std::map<std::string, std::string> out = in;
        if (!node.api_key.empty()) {
            for (auto it = out.begin(); it != out.end(); ) {
                it = pool_lower(it->first) == "authorization" ? out.erase(it) : std::next(it);
            }
            out["Authorization"] = "Bearer " + node.api_key;
        }
        return out;
    }

    std::unique_ptr<server_http_proxy> proxy_to(int n, const server_http_req & req, const std::string & method) {
        const pool_node & node = nodes[n];
        pool_link l;
        {
            std::lock_guard<std::mutex> lk(mu);
            l = node.links[node.link];
        }
        std::string path  = req.path;
        std::string query = req.query_string;
        std::string body  = req.body;
        if (!node.model.empty()) {
            // a node in router mode picks the instance by the model: in the body of a POST, in ?model= of a GET
            if (method == "POST") {
                json j = json::parse_no_throw(body);
                if (!j.is_discarded() && j.is_object()) {
                    j["model"] = node.model;
                    body = j.dump();
                }
            } else {
                std::string kept;
                size_t at = 0;
                while (at <= query.size() && !query.empty()) {
                    const size_t amp = query.find('&', at);
                    const std::string part = query.substr(at, amp == std::string::npos ? std::string::npos : amp - at);
                    if (!part.empty() && part.rfind("model=", 0) != 0) {
                        kept += (kept.empty() ? "" : "&") + part;
                    }
                    if (amp == std::string::npos) {
                        break;
                    }
                    at = amp + 1;
                }
                query = kept + (kept.empty() ? "" : "&") + "model=" + node.model;
            }
        }
        if (!query.empty()) {
            path += '?' + query;
        }
        return std::make_unique<server_http_proxy>(method, l.scheme, l.host, l.port, path,
            headers_for(node, req.headers), body, req.files, req.should_stop,
            params.timeout_read, params.timeout_write);
    }

    // the conversation of `r` moves from `from` to `to` with its state: `from` gives the cached state of the prompt
    // (POST /pool/export), the pool reads it in chunks and writes them to `to` (a thread each way, so that both links run
    // at once), `to` takes it into its prompt cache in RAM (POST /pool/import/done). false with `why` when it did not
    // move: `to` then computes the conversation from the start, as without a move
    bool migrate(const pool_req & r, int from, int to, const char * kind, const json & body,
                 const std::function<bool()> & should_stop, std::string & why, std::string & detail) {
        const pool_node & nf = nodes[from]; // the config fields only, the state is under mu
        const pool_node & nt = nodes[to];
        int lf_i;
        int lt_i;
        pool_link lf;
        pool_link lt;
        double bps;
        double prefill_to;
        uint64_t cache_ram_to;
        {
            std::lock_guard<std::mutex> lk(mu);
            lf_i = nf.link;
            lt_i = nt.link;
            lf   = nf.links[lf_i];
            lt   = nt.links[lt_i];
            bps  = std::min(link_bps(nf, false), link_bps(nt, true));
            prefill_to   = prefill_of(to);
            cache_ram_to = nt.cache_ram;
        }
        auto hdrs_of = [](const pool_node & n) {
            httplib::Headers h;
            if (!n.api_key.empty()) {
                h.emplace("Authorization", "Bearer " + n.api_key);
            }
            return h;
        };
        // a router node takes the model in ?model= and in the body of a POST; autoload=0: a move never loads a model
        auto path_of = [](const pool_node & n, const std::string & path, std::string q) {
            if (!n.model.empty()) {
                q += (q.empty() ? "" : "&") + std::string("model=") + n.model + "&autoload=0";
            }
            return q.empty() ? path : path + "?" + q;
        };
        auto body_of = [](const pool_node & n, json j) {
            if (!n.model.empty()) {
                j["model"] = n.model;
            }
            return j.dump();
        };
        const httplib::Headers hf = hdrs_of(nf);
        const httplib::Headers ht = hdrs_of(nt);
        httplib::Client cf(pool_base(lf));
        httplib::Client ct(pool_base(lt));
        cf.set_connection_timeout(3, 0);
        ct.set_connection_timeout(3, 0);
        // the export copies a state only in RAM, behind the batch the node is computing
        cf.set_read_timeout(params.timeout_read, 0);
        ct.set_read_timeout(60, 0);
        cf.set_write_timeout(60, 0);
        ct.set_write_timeout(60, 0);

        const int64_t t0 = ggml_time_ms();
        auto rx = cf.Post(path_of(nf, "/pool/export", ""), hf,
            body_of(nf, {{"kind", kind}, {"body", body}, {"min_match", cfg.move_tokens}}),
            "application/json");
        if (!rx || rx->status != 200) {
            why = "no state: " + pool_error_text(rx);
            return false;
        }
        cf.set_read_timeout(60, 0);
        const json ex = json::parse_no_throw(rx->body);
        const std::string id  = ex.is_object() ? json_value(ex, "id", std::string()) : std::string();
        const std::string sig = ex.is_object() ? json_value(ex, "signature", std::string()) : std::string();
        const uint64_t size   = ex.is_object() ? json_value(ex, "size", (uint64_t) 0) : 0;
        const int64_t n_tok   = ex.is_object() ? json_value(ex, "n_tokens", (int64_t) 0) : 0;
        const int64_t n_reuse = ex.is_object() ? json_value(ex, "n_reuse", (int64_t) 0) : 0;
        const int64_t n_ckpt  = ex.is_object() ? json_value(ex, "n_ckpt", (int64_t) 0) : 0;
        const int64_t n_ckpt_all = ex.is_object() ? json_value(ex, "n_ckpt_all", (int64_t) 0) : 0;
        const bool from_disk  = ex.is_object() && json_value(ex, "from_disk", false);
        auto drop_export = [&]() {
            cf.Post(path_of(nf, "/pool/export/done", ""), hf, body_of(nf, {{"id", id}}), "application/json");
        };
        if (id.empty() || size == 0) {
            why = "a bad answer to the export";
            drop_export();
            return false;
        }

        // the other node keeps the state in RAM, in its prompt cache
        if (cache_ram_to > 0 && size > cache_ram_to) {
            why = string_format("the state of %.0f MB is larger than the prompt cache of %s (%.0f MB)", size / 1e6,
                                nt.name.c_str(), cache_ram_to / 1e6);
            drop_export();
            return false;
        }
        // worth it only when the transfer is shorter than computing what it brings
        const double t_xfer = move_s((double) size, bps);
        const double t_comp = std::max<int64_t>(0, n_reuse - r.cached[to]) / prefill_to;
        if (t_xfer >= t_comp) {
            why = string_format("the move of %.0f MB would take ~%.0f s, computing ~%lld tok ~%.0f s", size / 1e6, t_xfer,
                                (long long) n_reuse, t_comp);
            drop_export();
            {
                std::lock_guard<std::mutex> lk(mu);
                state_bpt = state_bpt > 0 ? 0.5*state_bpt + 0.5*size / std::max<int64_t>(1, n_tok) : (double) size / std::max<int64_t>(1, n_tok);
            }
            return false;
        }

        // the reader keeps up to two chunks ahead of the writer
        std::mutex              m;
        std::condition_variable cq;
        std::deque<std::string> q;
        bool        read_done = false;
        bool        failed    = false;
        std::string err;
        double      t_read    = 0.0;
        std::thread reader([&]() {
            for (uint64_t off = 0; off < size; ) {
                {
                    std::unique_lock<std::mutex> lk(m);
                    cq.wait(lk, [&] { return q.size() < 2 || failed; });
                    if (failed) {
                        break;
                    }
                }
                const int64_t ts = ggml_time_us();
                auto rd = cf.Get(path_of(nf, "/pool/export/data", "id=" + id + "&offset=" + std::to_string(off) +
                                                                  "&size=" + std::to_string(POOL_CHUNK)), hf);
                std::lock_guard<std::mutex> lk(m);
                if (!rd || rd->status != 200 || rd->body.empty()) {
                    failed = true;
                    err = "read from " + nf.name + ": " + (rd && rd->body.empty() ? "no data" : pool_error_text(rd));
                    break;
                }
                t_read += (ggml_time_us() - ts) / 1e6;
                off    += rd->body.size();
                q.push_back(std::move(rd->body));
                cq.notify_all();
            }
            std::lock_guard<std::mutex> lk(m);
            read_done = true;
            cq.notify_all();
        });

        uint64_t off = 0;
        double t_write = 0.0;
        while (true) {
            std::string chunk;
            {
                std::unique_lock<std::mutex> lk(m);
                cq.wait_for(lk, std::chrono::milliseconds(200), [&] { return !q.empty() || read_done || failed; });
                if (!failed && should_stop()) {
                    failed = true;
                    err = "the client left";
                }
                if (failed || (q.empty() && read_done)) {
                    cq.notify_all();
                    break;
                }
                if (q.empty()) {
                    continue;
                }
                chunk = std::move(q.front());
                q.pop_front();
                cq.notify_all();
            }
            const int64_t ts = ggml_time_us();
            auto rw = ct.Post(path_of(nt, "/pool/import/data", "id=" + id + "&offset=" + std::to_string(off)), ht, chunk,
                              "application/octet-stream");
            if (!rw || rw->status != 200) {
                std::lock_guard<std::mutex> lk(m);
                failed = true;
                err = "write to " + nt.name + ": " + pool_error_text(rw);
                cq.notify_all();
                break;
            }
            t_write += (ggml_time_us() - ts) / 1e6;
            off     += chunk.size();
        }
        reader.join();

        auto abort_import = [&]() {
            ct.Post(path_of(nt, "/pool/import/done", ""), ht, body_of(nt, {{"id", id}, {"abort", true}}), "application/json");
        };
        if (failed || off != size) {
            why = failed ? err : "the state came short";
            abort_import();
            drop_export();
            return false;
        }
        auto ri = ct.Post(path_of(nt, "/pool/import/done", ""), ht, body_of(nt, {{"id", id}, {"signature", sig}}), "application/json");
        drop_export();
        if (!ri || ri->status != 200) {
            why = "import: " + pool_error_text(ri);
            abort_import();
            return false;
        }

        const double dt = std::max<int64_t>(1, ggml_time_ms() - t0) / 1000.0;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto avg = [](double old, double x) { return old > 0 ? 0.5*old + 0.5*x : x; };
            if (t_read > 0) {
                nodes[from].links[lf_i].bps_read = avg(nodes[from].links[lf_i].bps_read, size / t_read);
            }
            if (t_write > 0) {
                nodes[to].links[lt_i].bps_write = avg(nodes[to].links[lt_i].bps_write, size / t_write);
            }
            state_bpt = avg(state_bpt, (double) size / std::max<int64_t>(1, n_tok));
        }
        detail = string_format("~%lld tok from %s (%s): %.0f MB in %.1f s (%.0f MB/s), %lld of %lld checkpoints", (long long) n_tok,
                               nf.name.c_str(), from_disk ? "its file" : "RAM", size / 1e6, dt, size / 1e6 / dt,
                               (long long) n_ckpt, (long long) n_ckpt_all);
        return true;
    }

    // a request that takes a slot: the node by the conversation and the free slots, waiting in the queue if need be
    server_http_res_ptr forward_slot(const server_http_req & req, const std::string & method) {
        const json body = json::parse_no_throw(req.body);
        const pool_key key = pool_key_of(req.path, body.is_discarded() ? json() : body, req.headers);

        pool_req r;
        r.path = req.path;
        r.t_arrive = ggml_time_ms();
        {
            std::lock_guard<std::mutex> lk(mu);
            r.id = next_id++;
            lookup(key, r);
        }

        const int n = acquire(r, req.should_stop);
        if (n == -1) {
            return pool_error(499, "the client closed the request");
        }
        if (n == -2) {
            return pool_error(503, "no node of the pool is up");
        }

        const char * kind = pool_kind_of(req.path);
        int move_from = -1;
        std::string move_why;
        {
            std::lock_guard<std::mutex> lk(mu);
            const int64_t now = ggml_time_ms();
            remember(key, r, n, now);
            trim(now);
            const double  waited = (now - r.t_arrive) / 1000.0;
            const int64_t best   = r.cached_max();
            const int64_t got    = r.cached[n];
            const std::string w  = waited >= 0.5 ? string_format(", waited %.1f s", waited) : "";
            if (r.conv_new) {
                event("new", r.conv, n, r.shared[n] > 0
                    ? string_format("~%lld of ~%lld tok shared with others cached here%s", (long long) r.shared[n], (long long) r.prompt_tokens, w.c_str())
                    : string_format("~%lld tok%s", (long long) r.prompt_tokens, w.c_str()));
            } else if (best == 0) {
                event("cold", r.conv, n, string_format("no node holds it, ~%lld tok%s", (long long) r.prompt_tokens, w.c_str()));
            } else if (got >= best) {
                event("keep", r.conv, n, string_format("%d msgs, ~%lld of ~%lld tok cached%s", r.depth, (long long) got,
                    (long long) r.prompt_tokens, w.c_str()));
            } else {
                int from = 0;
                for (int i = 0; i < (int) nodes.size(); ++i) {
                    from = r.cached[i] > r.cached[from] ? i : from;
                }
                move_why = string_format("~%lld tok cached on %s (%s), ~%lld here, ~%lld tok to compute%s",
                    (long long) best, nodes[from].name.c_str(),
                    usable(nodes[from]) ? "busy" : (nodes[from].draining ? "draining" : nodes[from].status.c_str()),
                    (long long) got, (long long) (r.prompt_tokens - got), w.c_str());
                if (*kind && best - got >= cfg.move_tokens && can_migrate(from, n)) {
                    move_from = from;
                    running[r.id].stage = "moving";
                } else {
                    event("move", r.conv, n, move_why);
                }
            }
            if (!r.conv.empty()) {
                auto & c = convs[r.conv];
                c.node = n;
                c.depth = (int) key.chain.size() - 1;
                c.est_tokens = got;
                c.prompt_tokens = r.prompt_tokens;
                c.t_last = pool_wall_ms();
                c.n++;
                c.held.resize(nodes.size(), 0);
                c.held[n] = r.prompt_tokens;
            }
        }

        bool moved = false;
        if (move_from >= 0) {
            std::string why;
            std::string detail;
            moved = migrate(r, move_from, n, kind, body, req.should_stop, why, detail);
            std::lock_guard<std::mutex> lk(mu);
            auto it = running.find(r.id);
            if (it != running.end()) {
                it->second.stage.clear();
            }
            if (moved) {
                event("migrate", r.conv, n, detail);
            } else {
                event("move", r.conv, n, move_why + "; no state moved: " + why);
            }
        }
        if (req.should_stop()) {
            release(r.id, 499);
            return pool_error(499, "the client closed the request");
        }

        try {
            auto proxy = proxy_to(n, req, method);
            server_http_proxy * p = proxy.get();
            const uint64_t id = r.id;
            // the end of the answer, for its timings: the prefill speed, and the share of what the node held of the prompt
            // from other conversations that it reused (not when the state of the conversation came along)
            const bool   sample     = !moved && r.shared[n] >= cfg.move_tokens && r.prompt_tokens > 0;
            const double shared     = sample ? (double) r.shared[n] / r.prompt_tokens : 0.0;
            const double shared_max = sample ? (double) r.shared_max[n] / r.prompt_tokens : 0.0;
            constexpr size_t TAIL = 16*1024;
            auto tail = std::make_shared<std::string>();
            auto next = proxy->next;
            proxy->next = [next, tail](std::string & out) -> bool {
                const bool more = next(out);
                tail->append(out);
                if (tail->size() > TAIL) {
                    tail->erase(0, tail->size() - TAIL);
                }
                return more;
            };
            proxy->cleanup = [this, id, p, n, tail, shared, shared_max]() {
                note_timings(n, *tail, shared, shared_max);
                release(id, p->status);
            };
            return std::unique_ptr<server_http_res>(std::move(proxy));
        } catch (...) {
            release(r.id, 500);
            throw;
        }
    }

    // a request that takes no slot goes to any node that is up
    server_http_res_ptr forward_any(const server_http_req & req, const std::string & method) {
        int n;
        {
            std::lock_guard<std::mutex> lk(mu);
            n = any_node();
        }
        if (n < 0) {
            return pool_error(503, "no node of the pool is up");
        }
        return std::unique_ptr<server_http_res>(proxy_to(n, req, method));
    }

    //
    // polling
    //

    static void update_slots(pool_node & n, const json & slots, int64_t now) {
        std::vector<pool_node::slot_t> next;
        int busy = 0;
        double gen = 0.0;
        double pp  = 0.0;
        for (const auto & s : slots) {
            if (!s.is_object()) {
                continue;
            }
            pool_node::slot_t st;
            st.id          = json_value(s, "id", 0);
            st.id_task     = json_value(s, "id_task", (int64_t) -1);
            st.processing  = json_value(s, "is_processing", false);
            st.n_ctx       = json_value(s, "n_ctx", 0);
            st.n_prompt    = json_value(s, "n_prompt_tokens", (int64_t) 0);
            st.n_processed = json_value(s, "n_prompt_tokens_processed", (int64_t) 0);
            st.n_cached    = json_value(s, "n_prompt_tokens_cache", (int64_t) 0);
            if (s.contains("next_token") && s.at("next_token").is_array() && !s.at("next_token").empty()) {
                st.n_decoded = json_value(s.at("next_token")[0], "n_decoded", (int64_t) 0);
            }
            st.t = now;
            for (const auto & p : n.slot_state) {
                if (p.id != st.id) {
                    continue;
                }
                if (st.processing && p.id_task == st.id_task && now > p.t) {
                    const double dt = (now - p.t) / 1000.0;
                    const double g = std::max<int64_t>(0, st.n_decoded - p.n_decoded) / dt;
                    const double q = std::max<int64_t>(0, st.n_processed - p.n_processed) / dt;
                    st.gen_tps    = 0.5*g + 0.5*p.gen_tps;
                    st.prompt_tps = 0.5*q + 0.5*p.prompt_tps;
                }
                break;
            }
            busy += st.processing ? 1 : 0;
            gen  += st.gen_tps;
            pp   += st.prompt_tps;
            next.push_back(st);
        }
        n.slot_state = std::move(next);
        n.busy       = busy;
        n.gen_tps    = gen;
        n.prompt_tps = pp;
    }

    // the state of a model on a node in router mode: up when its instance is loaded
    static std::string model_state(httplib::Client & cli, const httplib::Headers & hdrs, const std::string & model, std::string & error) {
        auto rm = cli.Get("/models", hdrs);
        if (!rm || rm->status != 200) {
            error = rm ? "models: HTTP " + std::to_string(rm->status) : httplib::to_string(rm.error());
            return "down";
        }
        const json j = json::parse_no_throw(rm->body);
        const json * list = nullptr;
        if (!j.is_discarded() && j.is_object() && j.contains("data") && j.at("data").is_array()) {
            list = &j.at("data");
        } else if (!j.is_discarded() && j.is_array()) {
            list = &j;
        }
        if (list) {
            for (const auto & m : *list) {
                if (!m.is_object() || !pool_model_named(m, model)) {
                    continue;
                }
                const std::string v = m.contains("status") && m.at("status").is_object()
                    ? json_value(m.at("status"), "value", std::string()) : std::string();
                if (v == "loaded" || v == "sleeping") {
                    return "up";
                }
                if (v == "loading" || v == "downloading") {
                    return "loading";
                }
                error = "model " + model + " is " + (v.empty() ? "unknown" : v);
                return "unloaded";
            }
        }
        error = "no model " + model + " on the node";
        return "down";
    }

    // a node of another model than the most of the others takes no request (same_model). the first in the config
    // decides a tie
    void update_mismatch() {
        std::map<std::string, int> count;
        for (const auto & n : nodes) {
            if (!n.model_info.empty()) {
                count[n.model_info.dump()]++;
            }
        }
        std::string ref;
        int n_ref = 0;
        for (const auto & n : nodes) {
            if (!n.model_info.empty() && count[n.model_info.dump()] > n_ref) {
                ref   = n.model_info.dump();
                n_ref = count[ref];
            }
        }
        for (int i = 0; i < (int) nodes.size(); ++i) {
            pool_node & n = nodes[i];
            const bool mismatch = cfg.same_model && !n.model_info.empty() && n.model_info.dump() != ref;
            if (mismatch != n.mismatch) {
                event(mismatch ? "mismatch" : "match", "", i, mismatch
                    ? "another model than the other nodes (" + n.model_info.dump() + "): it takes no request, same_model = 0 lets it"
                    : "the same model as the other nodes again");
                n.mismatch = mismatch;
                cv.notify_all();
            }
        }
    }

    struct poll_result {
        bool        reached = false; // an HTTP answer came: the link works, whatever the state of the node
        std::string status  = "down";
        std::string error;
        json        props;
        json        slots;
        json        info;
    };

    // one poll of a node over one link
    poll_result poll_link(const pool_node & node, const pool_link & l, const httplib::Headers & hdrs, bool want_props) const {
        poll_result pr;
        try {
            httplib::Client cli(pool_base(l));
            cli.set_connection_timeout(1, 0);
            cli.set_read_timeout(3, 0);
            cli.set_write_timeout(3, 0);
            auto res = cli.Get("/health", hdrs);
            if (!res) {
                pr.error = httplib::to_string(res.error());
                return pr;
            }
            pr.reached = true;
            if (res->status == 503) {
                pr.status = "loading";
            } else if (res->status != 200) {
                pr.error = "health: HTTP " + std::to_string(res->status);
            } else {
                pr.status = node.model.empty() ? "up" : model_state(cli, hdrs, node.model, pr.error);
            }
            if (pr.status == "up") {
                // a router node passes these to the instance of the model; autoload=0: a poll never loads it
                const std::string q = node.model.empty() ? "" : "?model=" + node.model + "&autoload=0";
                if (want_props) {
                    auto rp = cli.Get("/props" + q, hdrs);
                    if (rp && rp->status == 200) {
                        pr.props = json::parse_no_throw(rp->body);
                    }
                }
                auto rs = cli.Get("/slots" + q, hdrs);
                if (rs && rs->status == 200) {
                    pr.slots = json::parse_no_throw(rs->body);
                }
                auto ri = cli.Get("/pool/node" + q, hdrs);
                if (ri && ri->status == 200) {
                    pr.info = json::parse_no_throw(ri->body);
                }
            }
        } catch (const std::exception & e) {
            pr.error = e.what();
        }
        return pr;
    }

    // the speed of a link both ways (bytes/s, 0 = no answer): one chunk of a move read from the node, one written to it
    static void probe_link(const pool_node & node, const pool_link & l, const httplib::Headers & hdrs, double & rd, double & wr) {
        httplib::Client cli(pool_base(l));
        cli.set_keep_alive(true);
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(60, 0);
        cli.set_write_timeout(60, 0);
        // a router node passes it to the instance of the model; autoload=0: a probe never loads it
        const std::string q = node.model.empty() ? "" : "model=" + node.model + "&autoload=0";
        if (!cli.Get("/health", hdrs)) { // the connection, before the clock starts
            return;
        }
        int64_t t0 = ggml_time_us();
        auto r = cli.Get("/pool/probe?size=" + std::to_string(POOL_CHUNK) + (q.empty() ? "" : "&" + q), hdrs);
        if (r && r->status == 200 && r->body.size() == POOL_CHUNK) {
            rd = POOL_CHUNK / std::max(1e-6, (ggml_time_us() - t0) / 1e6);
        }
        t0 = ggml_time_us();
        auto w = cli.Post("/pool/probe" + (q.empty() ? "" : "?" + q), hdrs, std::string(POOL_CHUNK, '\0'), "application/octet-stream");
        if (w && w->status == 200 && json_value(json::parse_no_throw(w->body), "size", (uint64_t) 0) == POOL_CHUNK) {
            wr = POOL_CHUNK / std::max(1e-6, (ggml_time_us() - t0) / 1e6);
        }
    }

    void poll_node(int i) {
        pool_node & node = nodes[i];
        httplib::Headers hdrs;
        if (!node.api_key.empty()) {
            hdrs.emplace("Authorization", "Bearer " + node.api_key);
        }

        while (!stopping.load()) {
            const int64_t t0 = ggml_time_ms();
            int  active;
            bool want_props;
            bool probe;
            {
                std::lock_guard<std::mutex> lk(mu);
                active     = node.link;
                want_props = node.status != "up" || t0 - node.t_props > 30000;
                probe      = active > 0 && t0 - node.t_probe >= (int64_t) cfg.probe_s * 1000;
                if (probe) {
                    node.t_probe = t0;
                }
            }
            // the links in the order they are tried, the first that answers serves: on a fallback the ones before it
            // now and then, the one in use, then the others in the order of the config
            std::vector<int> order;
            for (int k = 0; probe && k < active; ++k) {
                order.push_back(k);
            }
            order.push_back(active);
            for (int k = 0; k < (int) node.links.size(); ++k) {
                if (k != active && !(probe && k < active)) {
                    order.push_back(k);
                }
            }
            poll_result pr;
            int used = -1;
            std::vector<std::pair<int, std::string>> failed;
            for (int k : order) {
                pr = poll_link(node, node.links[k], hdrs, want_props || k != active);
                if (pr.reached) {
                    used = k;
                    break;
                }
                failed.push_back({k, pr.error});
            }

            {
                std::lock_guard<std::mutex> lk(mu);
                const int64_t now = ggml_time_ms();
                for (const auto & [k, e] : failed) {
                    node.links[k].error  = e;
                    node.links[k].probed = false;
                }
                if (used >= 0) {
                    node.links[used].error.clear();
                    if (used != node.link) {
                        const pool_link & was = node.links[node.link];
                        const std::string why = used < node.link ? "back to a preferred link" : pool_base(was) + ": " + was.error;
                        event("link", "", i, pool_base(node.links[used]) + " (" + why + ")");
                        SRV_INF("pool: node %s now on %s (%s)\n", node.name.c_str(), pool_base(node.links[used]).c_str(), why.c_str());
                        node.link = used;
                    }
                }
                const std::string status = used >= 0 ? pr.status : "down";
                const std::string error  = used >= 0 ? pr.error : node.links[active].error;
                if (status != node.status) {
                    event(status.c_str(), "", i, status == "up" ? pool_base(node.links[node.link]) : (error.empty() ? status : error));
                    SRV_INF("pool: node %s is %s%s%s\n", node.name.c_str(), status.c_str(), error.empty() ? "" : ": ", error.c_str());
                }
                node.status = status;
                node.error  = error;
                if (status == "up") {
                    node.t_ok = now;
                }
                if (!pr.props.is_discarded() && pr.props.is_object()) {
                    node.props   = pr.props;
                    node.t_props = now;
                }
                if (!pr.slots.is_discarded() && pr.slots.is_array()) {
                    update_slots(node, pr.slots, now);
                } else if (status != "up") {
                    node.slot_state.clear();
                    node.busy = 0;
                    node.gen_tps = node.prompt_tps = 0.0;
                }
                if (!pr.info.is_discarded() && pr.info.is_object() && pr.info.contains("devices")) {
                    node.devices    = pr.info.at("devices");
                    node.signature  = json_value(pr.info, "cache_signature", std::string());
                    node.migrate    = json_value(pr.info, "migrate", false);
                    node.cache_ram  = json_value(pr.info, "cache_ram", (uint64_t) 0);
                    node.model_info = pr.info.contains("model") && pr.info.at("model").is_object() ? pr.info.at("model") : json::object();
                    update_mismatch();
                }
                int reported = json_value(node.props, "total_slots", 0);
                if (reported <= 0) {
                    reported = (int) node.slot_state.size();
                }
                node.slots    = node.slots_cfg > 0 ? node.slots_cfg : reported;
                node.external = std::max(0, node.busy - node.inflight);
                cv.notify_all();
            }

            // a link of a node that moves conversations is measured when it starts to serve, until then a move is
            // weighed at the speed of the config. the moves measure it from then on
            if (used >= 0) {
                bool need;
                {
                    std::lock_guard<std::mutex> lk(mu);
                    pool_link & l = node.links[used];
                    need = cfg.migrate && node.migrate && node.status == "up" && !l.probed;
                    l.probed = l.probed || need;
                }
                if (need) {
                    double rd = 0.0;
                    double wr = 0.0;
                    probe_link(node, node.links[used], hdrs, rd, wr);
                    std::lock_guard<std::mutex> lk(mu);
                    pool_link & l = node.links[used];
                    l.bps_read  = l.bps_read  > 0 ? l.bps_read  : rd;
                    l.bps_write = l.bps_write > 0 ? l.bps_write : wr;
                    SRV_INF("pool: link %s of node %s: %.0f MB/s from it, %.0f MB/s to it\n", pool_base(l).c_str(),
                            node.name.c_str(), rd / 1e6, wr / 1e6);
                }
            }

            std::unique_lock<std::mutex> lk(mu_stop);
            cv_stop.wait_for(lk, std::chrono::milliseconds(std::max<int64_t>(50, cfg.poll_ms - (ggml_time_ms() - t0))),
                [&] { return stopping.load(); });
        }
    }

    // [pool] model: the pool names the model, the same whichever node answers; of the list of a router node only the
    // model of the node
    server_http_res_ptr models_alias() {
        int n;
        pool_link l;
        {
            std::lock_guard<std::mutex> lk(mu);
            n = any_node();
            if (n >= 0) {
                l = nodes[n].links[nodes[n].link];
            }
        }
        if (n < 0) {
            return pool_error(503, "no node of the pool is up");
        }
        const pool_node & node = nodes[n];
        httplib::Client cli(pool_base(l));
        cli.set_connection_timeout(2, 0);
        cli.set_read_timeout(10, 0);
        httplib::Headers h;
        if (!node.api_key.empty()) {
            h.emplace("Authorization", "Bearer " + node.api_key);
        }
        auto res = cli.Get("/v1/models", h);
        json j = res && res->status == 200 ? json::parse_no_throw(res->body) : json();
        if (j.is_discarded() || !j.is_object()) {
            return pool_error(502, "node " + node.name + ": " + pool_error_text(res));
        }
        for (const char * key : {"data", "models"}) {
            if (!j.contains(key) || !j.at(key).is_array()) {
                continue;
            }
            json kept = json::array();
            for (json m : j.at(key)) {
                if (!m.is_object()) {
                    continue;
                }
                const std::string id = json_value(m, "id", json_value(m, "model", json_value(m, "name", std::string())));
                if (!node.model.empty() && id != node.model && !pool_model_named(m, node.model)) {
                    continue;
                }
                for (const char * f : {"id", "model", "name"}) {
                    if (m.contains(f)) {
                        m[f] = cfg.model;
                    }
                }
                if (m.contains("aliases")) {
                    m["aliases"] = json::array({cfg.model});
                }
                kept.push_back(m);
                break;
            }
            j[key] = kept;
        }
        auto out = std::make_unique<server_http_res>();
        out->data = safe_json_to_str(j);
        return out;
    }

    //
    // the dashboard
    //

    json status_json() {
        std::lock_guard<std::mutex> lk(mu);
        const int64_t now  = ggml_time_ms();
        const int64_t wall = pool_wall_ms();

        json jnodes = json::array();
        int up = 0, slots_total = 0, slots_busy = 0, inflight = 0;
        double gen = 0.0, pp = 0.0;
        for (int i = 0; i < (int) nodes.size(); ++i) {
            const pool_node & n = nodes[i];
            json jslots = json::array();
            for (const auto & s : n.slot_state) {
                jslots.push_back({
                    {"id", s.id}, {"state", !s.processing ? "idle" : (s.n_decoded == 0 ? "prompt" : "gen")},
                    {"n_ctx", s.n_ctx}, {"n_prompt", s.n_prompt}, {"n_processed", s.n_processed}, {"n_cached", s.n_cached},
                    {"n_decoded", s.n_decoded}, {"gen_tps", s.gen_tps}, {"prompt_tps", s.prompt_tps},
                });
            }
            json jrun = json::array();
            for (const auto & [id, r] : running) {
                if (r.node == i) {
                    jrun.push_back({{"id", id}, {"conv", r.conv}, {"age_ms", now - r.t_start}, {"path", r.path}, {"stage", r.stage}});
                }
            }
            int queued_for = 0;
            for (const pool_req * q : queue) {
                queued_for += holder(*q) == i ? 1 : 0;
            }
            std::string model = json_value(n.props, "model_alias", std::string());
            if (model.empty()) {
                model = json_value(n.props, "model_path", std::string());
            }
            if (model.find_last_of("/\\") != std::string::npos) {
                model = model.substr(model.find_last_of("/\\") + 1);
            }
            int n_ctx = 0;
            if (n.props.contains("default_generation_settings") && n.props.at("default_generation_settings").is_object()) {
                n_ctx = json_value(n.props.at("default_generation_settings"), "n_ctx", 0);
            }
            json jlinks = json::array();
            for (int k = 0; k < (int) n.links.size(); ++k) {
                const pool_link & l = n.links[k];
                jlinks.push_back({
                    {"url", pool_base(l)}, {"active", k == n.link}, {"error", l.error},
                    {"mbps_read", l.bps_read * 8 / 1e6}, {"mbps_write", l.bps_write * 8 / 1e6}, {"mbps_cfg", l.mbps_cfg},
                });
            }
            jnodes.push_back({
                {"name", n.name}, {"url", pool_base(n.links[n.link])}, {"fallback", n.link > 0}, {"links", jlinks},
                {"status", n.status}, {"draining", n.draining}, {"mismatch", n.mismatch}, {"error", n.error},
                {"model", model}, {"model_info", n.model_info}, {"migrate", n.migrate}, {"signature", n.signature},
                {"n_ctx", n_ctx}, {"slots", n.slots}, {"inflight", n.inflight}, {"busy", n.busy},
                {"external", n.external}, {"weight", n.weight}, {"requests", n.n_requests},
                {"gen_tps", n.gen_tps}, {"prompt_tps", n.prompt_tps}, {"queued_for", queued_for},
                {"prefill_tps", n.prefill_tps}, {"prefill_cfg", n.prefill_cfg > 0 ? n.prefill_cfg : cfg.prefill_tps},
                {"share", n.share},
                {"seen_ms", n.t_ok > 0 ? now - n.t_ok : -1}, {"devices", n.devices}, {"slot_list", jslots}, {"running", jrun},
            });
            if (n.status == "up") {
                up++;
            }
            if (usable(n)) {
                slots_total += n.slots;
                slots_busy  += std::min(n.slots, n.inflight + n.external);
            }
            inflight += n.inflight;
            gen += n.gen_tps;
            pp  += n.prompt_tps;
        }

        json jqueue = json::array();
        for (const pool_req * q : queue) {
            const int64_t best = q->cached_max();
            std::string want;
            for (int i = 0; i < (int) nodes.size() && best > 0; ++i) {
                if (q->cached[i] == best) {
                    want += (want.empty() ? "" : ", ") + nodes[i].name;
                }
            }
            jqueue.push_back({
                {"id", q->id}, {"conv", q->conv}, {"want", want}, {"age_ms", now - q->t_arrive}, {"est_tokens", best},
                {"prompt_tokens", q->prompt_tokens}, {"depth", q->depth}, {"path", q->path},
            });
        }

        std::vector<std::pair<std::string, conv_t>> cs(convs.begin(), convs.end());
        std::sort(cs.begin(), cs.end(), [](const auto & a, const auto & b) { return a.second.t_last > b.second.t_last; });
        json jconvs = json::array();
        for (size_t i = 0; i < cs.size() && i < 60; ++i) {
            const auto & [id, c] = cs[i];
            bool active = false;
            for (const auto & [rid, r] : running) {
                active = active || r.conv == id;
            }
            jconvs.push_back({
                {"conv", id}, {"node", c.node >= 0 ? nodes[c.node].name : ""}, {"messages", c.depth},
                {"cached_tokens", c.est_tokens}, {"prompt_tokens", c.prompt_tokens}, {"age_ms", wall - c.t_last},
                {"requests", c.n}, {"active", active},
            });
        }

        json jevents = json::array();
        for (auto it = events.rbegin(); it != events.rend() && jevents.size() < 120; ++it) {
            jevents.push_back({{"t", it->t}, {"kind", it->kind}, {"conv", it->conv}, {"node", it->node}, {"detail", it->detail}});
        }

        return {
            {"name", cfg.name}, {"time", wall},
            {"pool", {
                {"nodes_up", up}, {"nodes", (int) nodes.size()}, {"slots", slots_total}, {"slots_busy", slots_busy},
                {"inflight", inflight}, {"queued", (int) queue.size()}, {"gen_tps", gen}, {"prompt_tps", pp},
                {"conversations", (int) convs.size()}, {"prefixes", (int) affinity.size()},
            }},
            {"config", {
                {"move_tokens", cfg.move_tokens}, {"wait_max", cfg.wait_max_s}, {"prefill_tps", cfg.prefill_tps},
                {"poll_ms", cfg.poll_ms}, {"model", cfg.model}, {"migrate", cfg.migrate}, {"link_mbps", cfg.link_mbps},
                {"same_model", cfg.same_model},
            }},
            {"state_kib_per_token", state_bpt / 1024},
            {"nodes", jnodes}, {"queue", jqueue}, {"conversations", jconvs}, {"events", jevents},
        };
    }
};

//
// server_pool_fork
//

server_pool_fork::server_pool_fork(const common_params & params) : pimpl(std::make_unique<impl>(params)) {
    pimpl->load(params.pool_file);
    for (const auto & n : pimpl->nodes) {
        std::string urls;
        for (const auto & l : n.links) {
            urls += (urls.empty() ? "" : ", then ") + pool_base(l);
        }
        SRV_INF("pool: node %s at %s%s\n", n.name.c_str(), urls.c_str(),
                n.slots_cfg > 0 ? string_format(", %d slots", n.slots_cfg).c_str() : "");
    }
}

server_pool_fork::~server_pool_fork() {
    stop();
}

void server_pool_fork::start() {
    for (int i = 0; i < (int) pimpl->nodes.size(); ++i) {
        pimpl->pollers.emplace_back([this, i]() { pimpl->poll_node(i); });
    }
}

void server_pool_fork::stop() {
    if (pimpl->stopping.exchange(true)) {
        return;
    }
    {
        std::lock_guard<std::mutex> lk(pimpl->mu_stop);
    }
    pimpl->cv_stop.notify_all();
    pimpl->cv.notify_all();
    for (auto & t : pimpl->pollers) {
        if (t.joinable()) {
            t.join();
        }
    }
}

void server_pool_fork::attach(server_http_context & ctx_http, server_routes & routes) {
    impl * p = pimpl.get();

    auto slot_post = [p](const server_http_req & req) { return p->forward_slot(req, "POST"); };
    auto any_post  = [p](const server_http_req & req) { return p->forward_any(req, "POST"); };
    auto any_get   = [p](const server_http_req & req) { return p->forward_any(req, "GET"); };
    auto node_only = [](const server_http_req &) {
        return pool_error(400, "a pool does not route this request: send it to a node directly");
    };

    routes.get_health = [p](const server_http_req &) {
        int up = 0;
        {
            std::lock_guard<std::mutex> lk(p->mu);
            for (const auto & n : p->nodes) {
                up += n.status == "up" ? 1 : 0;
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->status = up > 0 ? 200 : 503;
        res->data = safe_json_to_str({{"status", up > 0 ? "ok" : "no node is up"}, {"nodes_up", up}});
        return res;
    };
    routes.get_slots = [p](const server_http_req &) {
        json out = json::array();
        {
            std::lock_guard<std::mutex> lk(p->mu);
            for (const auto & n : p->nodes) {
                for (const auto & s : n.slot_state) {
                    out.push_back({{"node", n.name}, {"id", s.id}, {"is_processing", s.processing}, {"n_ctx", s.n_ctx},
                                   {"n_prompt_tokens", s.n_prompt}, {"n_decoded", s.n_decoded}});
                }
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str(out);
        return res;
    };

    routes.get_props                   = any_get;
    routes.get_models                  = any_get;
    if (!p->cfg.model.empty()) {
        routes.get_models = [p](const server_http_req &) { return p->models_alias(); };
    }
    routes.get_lora_adapters           = any_get;
    routes.get_metrics                 = node_only;
    routes.post_props                  = node_only;
    routes.post_slots                  = node_only;
    routes.post_control                = node_only;
    routes.post_lora_adapters          = node_only;

    routes.post_completions            = slot_post;
    routes.post_completions_oai        = slot_post;
    routes.post_chat_completions       = slot_post;
    routes.post_responses_oai          = slot_post;
    routes.post_anthropic_messages     = slot_post;
    routes.post_infill                 = slot_post;
    routes.post_embeddings             = slot_post;
    routes.post_embeddings_oai         = slot_post;
    routes.post_rerank                 = slot_post;
    routes.post_transcriptions_oai     = slot_post;
    routes.post_tts                    = slot_post;
    routes.post_systemone              = slot_post;

    routes.post_chat_completions_tok   = any_post;
    routes.post_responses_tok_oai      = any_post;
    routes.post_anthropic_count_tokens = any_post;
    routes.post_tokenize               = any_post;
    routes.post_detokenize             = any_post;
    routes.post_apply_template         = any_post;

    routes.ws_realtime = [](const server_http_req &, server_http_context::ws_conn & conn) {
        conn.close();
    };

    ctx_http.get("/pool", [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "text/html; charset=utf-8";
        for (const char * part : POOL_UI_PARTS) {
            res->data += part;
        }
        return res;
    });
    ctx_http.get("/pool/status", [p](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        try {
            res->data = safe_json_to_str(p->status_json());
        } catch (const std::exception & e) {
            return pool_error(500, e.what());
        }
        return res;
    });
    // {"node": "<name>", "drain": true}: no new request goes to the node, the running ones finish
    ctx_http.post("/pool/drain", [p](const server_http_req & req) {
        const json body = json::parse_no_throw(req.body);
        if (body.is_discarded() || !body.is_object() || !body.contains("node")) {
            return pool_error(400, "expected {\"node\": \"<name>\", \"drain\": true|false}");
        }
        const std::string name = json_value(body, "node", std::string());
        const bool drain = json_value(body, "drain", true);
        std::lock_guard<std::mutex> lk(p->mu);
        for (int i = 0; i < (int) p->nodes.size(); ++i) {
            if (p->nodes[i].name == name) {
                p->nodes[i].draining = drain;
                p->event(drain ? "drain" : "undrain", "", i, "");
                p->cv.notify_all();
                auto res = std::make_unique<server_http_res>();
                res->data = safe_json_to_str({{"node", name}, {"draining", drain}});
                return res;
            }
        }
        return pool_error(404, "no node " + name);
    });
}

//
// the node side
//

json server_pool_node_devices_fork() {
    json devices = json::array();
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        size_t free  = 0;
        size_t total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        const char * type = "other";
        switch (ggml_backend_dev_type(dev)) {
            case GGML_BACKEND_DEVICE_TYPE_CPU:
                // the CPU device reports the RAM it may use as free: the host knows what is available
                type  = "cpu";
                total = common_host_mem_total() > 0 ? common_host_mem_total() : total;
                free  = common_host_mem_available() > 0 ? common_host_mem_available() : free;
                break;
            case GGML_BACKEND_DEVICE_TYPE_GPU:   type = "gpu";   break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU:  type = "igpu";  break;
            default:                             type = "other"; break;
        }
        devices.push_back({
            {"name", ggml_backend_dev_name(dev)}, {"description", ggml_backend_dev_description(dev)},
            {"type", type}, {"free", free}, {"total", total},
        });
    }
    return devices;
}

std::string server_pool_xfer_id_fork() {
    static std::mutex mu;
    static std::mt19937_64 rng(((uint64_t) std::random_device{}() << 32) ^ (uint64_t) std::random_device{}() ^
                               (uint64_t) ggml_time_us());
    std::lock_guard<std::mutex> lk(mu);
    char buf[24];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) rng());
    return buf;
}

std::filesystem::path server_pool_xfer_path_fork(const std::string & spill_dir, const char * kind, const std::string & id) {
    if (spill_dir.empty() || id.empty() || id.size() > 32 || id.find_first_not_of("0123456789abcdef") != std::string::npos) {
        return {};
    }
    return server_fs_path_fork(spill_dir) / (std::string("pool-") + kind + "-" + id + ".bin");
}

// id -> when it was used last, the export or the import
static std::mutex pool_xfer_mu;
static std::map<std::string, std::pair<int64_t, std::shared_ptr<const server_pool_plan_fork>>> pool_plans;
static std::map<std::string, std::pair<int64_t, std::shared_ptr<server_pool_inbox_fork>>>      pool_inbox;

static void pool_plan_drop(const server_pool_plan_fork & plan) {
    if (plan.file_link) {
        std::error_code ec;
        std::filesystem::remove(plan.file, ec);
    }
}

void server_pool_xfer_plan_fork(const std::string & id, server_pool_plan_fork plan) {
    std::lock_guard<std::mutex> lk(pool_xfer_mu);
    pool_plans[id] = {ggml_time_ms(), std::make_shared<const server_pool_plan_fork>(std::move(plan))};
}

void server_pool_xfer_forget_fork(const std::string & id) {
    std::shared_ptr<const server_pool_plan_fork> plan;
    {
        std::lock_guard<std::mutex> lk(pool_xfer_mu);
        auto it = pool_plans.find(id);
        if (it == pool_plans.end()) {
            return;
        }
        plan = it->second.second;
        pool_plans.erase(it);
    }
    pool_plan_drop(*plan);
}

bool server_pool_xfer_read_fork(const std::string & id, uint64_t offset, uint64_t size, std::string & out, std::string & error) {
    std::shared_ptr<const server_pool_plan_fork> plan;
    {
        std::lock_guard<std::mutex> lk(pool_xfer_mu);
        auto it = pool_plans.find(id);
        if (it != pool_plans.end()) {
            it->second.first = ggml_time_ms();
            plan = it->second.second;
        }
    }
    if (!plan) {
        error = "no such transfer";
        return false;
    }
    if (offset > plan->size) {
        error = "the offset is past the end of the state";
        return false;
    }
    std::ifstream f;
    if (!plan->file.empty()) {
        f.open(plan->file, std::ios::binary);
        if (!f) {
            error = "cannot open the state file";
            return false;
        }
    }
    out.resize((size_t) std::min(size, plan->size - offset));
    uint64_t at   = 0; // where the piece starts in the state
    uint64_t done = 0;
    for (const auto & pc : plan->pieces) {
        if (done == out.size()) {
            break;
        }
        const uint64_t pos = offset + done;
        if (pos < at + pc.len) {
            const uint64_t skip = pos - at;
            const uint64_t n    = std::min<uint64_t>(pc.len - skip, out.size() - done);
            if (!pc.data.empty()) {
                memcpy(&out[done], pc.data.data() + skip, n);
            } else {
                f.seekg((std::streamoff) (pc.off + skip));
                if (!f.read(&out[done], (std::streamsize) n)) {
                    error = "cannot read the state";
                    return false;
                }
            }
            done += n;
        }
        at += pc.len;
    }
    return true;
}

bool server_pool_inbox_write_fork(const std::string & id, uint64_t offset, const std::string & data, uint64_t max_bytes,
                                  std::string & error) {
    if (id.empty() || id.size() > 32 || id.find_first_not_of("0123456789abcdef") != std::string::npos) {
        error = "no such transfer";
        return false;
    }
    // the chunks of one import come one after another, the lock keeps them so
    std::lock_guard<std::mutex> lk(pool_xfer_mu);
    auto it = pool_inbox.find(id);
    if (offset == 0) {
        it = pool_inbox.insert_or_assign(id, std::make_pair(int64_t(0), std::make_shared<server_pool_inbox_fork>())).first;
    } else if (it == pool_inbox.end()) {
        error = "no such transfer";
        return false;
    }
    it->second.first = ggml_time_ms();
    if (!it->second.second->write(offset, data.data(), data.size(), max_bytes, error)) {
        pool_inbox.erase(it);
        return false;
    }
    return true;
}

std::shared_ptr<server_pool_inbox_fork> server_pool_inbox_take_fork(const std::string & id) {
    std::lock_guard<std::mutex> lk(pool_xfer_mu);
    auto it = pool_inbox.find(id);
    if (it == pool_inbox.end()) {
        return nullptr;
    }
    auto in = std::move(it->second.second);
    pool_inbox.erase(it);
    return in;
}

void server_pool_xfer_clean_fork(const std::string & spill_dir, int idle_s) {
    std::vector<std::shared_ptr<const server_pool_plan_fork>> stale;
    {
        std::lock_guard<std::mutex> lk(pool_xfer_mu);
        const int64_t now = ggml_time_ms();
        for (auto it = pool_plans.begin(); it != pool_plans.end(); ) {
            if (now - it->second.first > (int64_t) idle_s * 1000) {
                stale.push_back(it->second.second);
                it = pool_plans.erase(it);
            } else {
                ++it;
            }
        }
        for (auto it = pool_inbox.begin(); it != pool_inbox.end(); ) {
            it = now - it->second.first > (int64_t) idle_s * 1000 ? pool_inbox.erase(it) : std::next(it);
        }
    }
    for (const auto & plan : stale) {
        pool_plan_drop(*plan);
    }
    if (spill_dir.empty()) {
        return;
    }
    // the links of a server that stopped midway
    std::error_code ec;
    const auto now = std::filesystem::file_time_type::clock::now();
    for (std::filesystem::directory_iterator it(server_fs_path_fork(spill_dir), ec), end; !ec && it != end; it.increment(ec)) {
        const std::string name = it->path().filename().u8string(); // string() throws on Windows for a name outside the codepage
        if (name.rfind("pool-export-", 0) != 0 && name.rfind("pool-import-", 0) != 0) {
            continue;
        }
        const std::string id = name.substr(12, name.size() - 12 - 4);
        {
            std::lock_guard<std::mutex> lk(pool_xfer_mu);
            if (pool_plans.count(id) > 0) {
                continue;
            }
        }
        std::error_code ec_file;
        const auto t = it->last_write_time(ec_file);
        if (!ec_file && now - t > std::chrono::seconds(idle_s)) {
            std::filesystem::remove(it->path(), ec_file);
        }
    }
}

server_http_context::handler_t server_pool_router_post_fork(server_models_routes & routes) {
    return [&routes](const server_http_req & req) -> server_http_res_ptr {
        // the name of the preset for an alias, as the other routes of the router pass it: proxy_request counts the
        // requests of a model under the name it gets, and an alias there would become a model of its own that the
        // alias then finds. no autoload: the move of a conversation never loads a model
        const std::string name = req.get_param("model");
        const auto meta = routes.models.get_meta(name);
        return routes.models.proxy_request(req, "POST", meta.has_value() ? meta->name : name, false);
    };
}
