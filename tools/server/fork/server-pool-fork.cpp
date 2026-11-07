// fork: a pool, the routers of a local network that share a model. a model with pool = on in models.ini (serve.bat) is
// shared with the routers that share it under the same name or alias: each of them takes the requests for it and sends
// each to one of the machines of the pool (a node, this one too):
// - the routers find each other with UDP broadcasts on every local network (port 12345). a router seen on two networks
//   (a direct cable and the WiFi) gets a link on each, the faster one measured serves, the other is the fallback. with
//   --api-key (the same on every router) a link counts once the other side answered a challenge with a hash of the key,
//   and only then the key goes to it
// - a request goes to a node with a free slot (fewer requests in flight than its slots); a new conversation to the node
//   with the most free slots by its measured speed
// - a conversation keeps to the node that holds its prompt cache: the pool hashes the prompt message by message (first
//   the tools and the system prompt) and remembers which node served each prefix, and a request goes to the node with
//   the longest prefix of it. there the cache is reused, anywhere else the whole context is computed again (100K
//   tokens are about 2.4 minutes of prefill on the 395). X-Conversation-Id or X-Session-Id, when a client sends one,
//   names the conversation instead
// - when that node has no free slot, a conversation of at least move_tokens waits for it as long as a recompute
//   elsewhere would take (its tokens at the prefill speed of that node, measured from the timings of the answers),
//   up to wait_max seconds; a shorter one goes to a free node at once
// - with no free slot anywhere the requests wait in the pool, the oldest first; a client that goes away leaves
// - a conversation that goes to another node takes its cached state along when both nodes can (the same model and
//   configuration) and the transfer is shorter than computing it again: the router that took the request asks the node
//   that holds it for the state (POST /pool/export), and the other node reads it from there itself (POST /pool/pull),
//   from RAM or the file of a state on disk to RAM, with nothing written to disk
// - every router of the pool takes requests, the agents may use any of them; on a node the requests that came through
//   another router count as busy slots
// - /pool is a dashboard of the nodes (slots, speed, memory), the queue, the conversations and the decisions
// the options, all optional, are preset-only keys next to pool = on (or in [*]): pool-name (this node, default the host
// name), pool-peers (host:port,... of routers to add without discovery), pool-port (UDP, 12345), pool-discover (0: no
// broadcasts), pool-move-tokens, pool-wait-max, pool-migrate, pool-link-mbps, pool-prefill-tps, pool-poll-ms,
// pool-probe-s, pool-same-model (common/fork/arg-fork.cpp)

#include "server-pool-fork.h"

#include "server-common.h"
#include "server-context.h"
#include "server-models.h" // server_http_proxy, server_models_routes

#include "arg.h"
#include "common.h"
#include "fork/arg-fork.h"    // COMMON_ARG_PRESET_POOL_FORK
#include "fork/common-fork.h" // common_host_mem_available
#include "log.h"

#include "ggml-backend.h"

#include <cpp-httplib/httplib.h>
#include "hash/hash.h"

#if defined(_WIN32)
#include <iphlpapi.h>
#ifndef SIO_UDP_CONNRESET
#define SIO_UDP_CONNRESET _WSAIOW(IOC_VENDOR, 12) // in mswsock.h of some SDKs only
#endif
#if defined(_MSC_VER)
#pragma comment(lib, "iphlpapi.lib")
#endif
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <set>
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

// v[i], 0 past its end: the vectors of a request have a value per node it saw, a node may join later
static int64_t pool_at(const std::vector<int64_t> & v, int i) {
    return i >= 0 && i < (int) v.size() ? v[i] : 0;
}

// a random hex string of 16 digits
static std::string pool_random_hex() {
    static std::mutex mu;
    static std::mt19937_64 rng(((uint64_t) std::random_device{}() << 32) ^ (uint64_t) std::random_device{}() ^
                               (uint64_t) ggml_time_us());
    std::lock_guard<std::mutex> lk(mu);
    char buf[24];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) rng());
    return buf;
}

// this process, the same for every pool of the router: a router of another machine with the same id has another
static const std::string & pool_instance() {
    static const std::string s = pool_random_hex();
    return s;
}

// host:port or http://host:port, [ipv6]:port; the path is ignored
static void pool_parse_host_port(const std::string & s, std::string & host, int & port) {
    const size_t p = s.find("://");
    if (p != std::string::npos && pool_lower(s.substr(0, p)) != "http") {
        throw std::runtime_error("a peer of the pool takes http only: " + s);
    }
    std::string rest = p == std::string::npos ? s : s.substr(p + 3);
    rest = rest.substr(0, rest.find('/'));
    port = 0;
    if (!rest.empty() && rest[0] == '[') {
        const size_t rb = rest.find(']');
        if (rb == std::string::npos) {
            throw std::runtime_error("bad address " + s);
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
    if (host.empty() || port <= 0) {
        throw std::runtime_error("bad address " + s + ", expected host:port");
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
// the discovery: UDP datagrams of JSON on the local networks
//

// the IPv4 broadcast addresses of the interfaces that are up, and the limited broadcast (network byte order)
static std::vector<uint32_t> pool_broadcast_addrs() {
    std::vector<uint32_t> out;
#if defined(_WIN32)
    ULONG size = 16*1024;
    std::vector<unsigned char> buf;
    ULONG rc = ERROR_BUFFER_OVERFLOW;
    for (int i = 0; i < 4 && rc == ERROR_BUFFER_OVERFLOW; ++i) {
        buf.resize(size);
        rc = GetAdaptersAddresses(AF_INET, GAA_FLAG_SKIP_ANYCAST | GAA_FLAG_SKIP_MULTICAST | GAA_FLAG_SKIP_DNS_SERVER,
                                  nullptr, (IP_ADAPTER_ADDRESSES *) buf.data(), &size);
    }
    if (rc == NO_ERROR) {
        for (auto * a = (IP_ADAPTER_ADDRESSES *) buf.data(); a != nullptr; a = a->Next) {
            if (a->OperStatus != IfOperStatusUp || a->IfType == IF_TYPE_SOFTWARE_LOOPBACK) {
                continue;
            }
            for (auto * u = a->FirstUnicastAddress; u != nullptr; u = u->Next) {
                if (u->Address.lpSockaddr == nullptr || u->Address.lpSockaddr->sa_family != AF_INET) {
                    continue;
                }
                const uint32_t ip = ((sockaddr_in *) u->Address.lpSockaddr)->sin_addr.s_addr;
                const int prefix = u->OnLinkPrefixLength;
                const uint32_t mask = prefix <= 0 ? 0u : prefix >= 32 ? 0xffffffffu : htonl(0xffffffffu << (32 - prefix));
                out.push_back(ip | ~mask);
            }
        }
    }
#else
    ifaddrs * ifs = nullptr;
    if (getifaddrs(&ifs) == 0) {
        for (ifaddrs * i = ifs; i != nullptr; i = i->ifa_next) {
            if (i->ifa_addr == nullptr || i->ifa_addr->sa_family != AF_INET || !(i->ifa_flags & IFF_UP) ||
                (i->ifa_flags & IFF_LOOPBACK) || !(i->ifa_flags & IFF_BROADCAST) || i->ifa_broadaddr == nullptr) {
                continue;
            }
            out.push_back(((sockaddr_in *) i->ifa_broadaddr)->sin_addr.s_addr);
        }
        freeifaddrs(ifs);
    }
#endif
    out.push_back(htonl(INADDR_BROADCAST));
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

static void pool_sock_close(socket_t & s) {
    if (s != INVALID_SOCKET) {
#if defined(_WIN32)
        closesocket(s);
#else
        close(s);
#endif
        s = INVALID_SOCKET;
    }
}

// a UDP socket that may broadcast, on `port` (0 = any). shared: other routers of this machine bind the same port (the
// tests run several), every one of them gets the broadcasts
static socket_t pool_udp_socket(int port, bool shared) {
    socket_t s = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (s == INVALID_SOCKET) {
        return s;
    }
    int one = 1;
    setsockopt(s, SOL_SOCKET, SO_BROADCAST, (const char *) &one, (socklen_t) sizeof(one));
#if defined(_WIN32)
    // a datagram to a port nobody listens on would fail the next recvfrom with WSAECONNRESET
    BOOL report = FALSE;
    DWORD got = 0;
    WSAIoctl(s, SIO_UDP_CONNRESET, &report, (DWORD) sizeof(report), nullptr, 0, &got, nullptr, nullptr);
#endif
    if (shared) {
        setsockopt(s, SOL_SOCKET, SO_REUSEADDR, (const char *) &one, (socklen_t) sizeof(one));
#ifdef SO_REUSEPORT
        setsockopt(s, SOL_SOCKET, SO_REUSEPORT, (const char *) &one, (socklen_t) sizeof(one));
#endif
    }
    sockaddr_in a = {};
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = htonl(INADDR_ANY);
    a.sin_port        = htons((uint16_t) port);
    if (bind(s, (const sockaddr *) &a, (socklen_t) sizeof(a)) != 0) {
        pool_sock_close(s);
    }
    return s;
}

static void pool_udp_send(socket_t s, uint32_t addr, int port, const std::string & msg) {
    sockaddr_in a = {};
    a.sin_family      = AF_INET;
    a.sin_addr.s_addr = addr;
    a.sin_port        = htons((uint16_t) port);
    sendto(s, msg.data(), (int) msg.size(), 0, (const sockaddr *) &a, (socklen_t) sizeof(a));
}

//
// the pool
//

struct pool_config {
    std::string name;            // this router in the pool, the others show it so; "" = the host name
    int     port        = 12345; // the UDP port of the discovery
    bool    discover    = true;  // the broadcasts; false = only the peers below
    std::vector<std::string> peers; // host:port of routers of the pool, added without discovery
    int     poll_ms     = 1000;  // how often each node is asked for its health and slots
    int64_t move_tokens = 4096;  // a conversation this long waits for its node, a shorter one moves at once
    int     wait_max_s  = 300;   // the longest wait for the node of a conversation
    double  prefill_tps = 700.0; // the prefill speed a wait is weighed against until a node has measured its own
    double  chars_tok   = 3.5;   // characters per token, for the estimates
    int     ttl_s       = 6*3600; // how long the pool remembers which node holds a prefix
    bool    migrate     = true;  // a conversation moves with its cached state when the nodes can
    bool    drain_move  = true;  // a drained node hands its active conversations over at once (evacuate)
    double  link_mbps   = 500.0; // the speed of a link not measured yet (Mbit/s)
    int     probe_s     = 10;    // how often a node on a slower link tries the faster ones again (s)
    bool    same_model  = true;  // a node of another model than the most of the others takes no request
    int     rebalance_s = 15;    // a conversation leaves a crowded node when the move costs at most this (s), 0 = never
    int     quiet_wait_s = 60;   // the longest wait of a long prompt for a node where nothing runs (s), 0 = no wait
    int64_t short_tokens = 512;  // a request for at most this many answer tokens goes ahead in the queue, 0 = none does
    int     evict_idle_s = 300;  // a load the pool asks for unloads on that router only a model idle this long (s), 0 = any idle
};

// the options of the pool next to pool = on in models.ini (pool-<key>, common/fork/arg-fork.cpp); a bad value stops
// the start of the router with the name of the key
static void pool_read_options(pool_config & cfg, const common_preset & preset) {
    auto get = [&](const char * suffix, const char * key, const std::function<void(const std::string &)> & set) {
        std::string v;
        if (!preset.get_option(std::string(COMMON_ARG_PRESET_POOL_FORK) + suffix, v)) {
            return;
        }
        try {
            set(pool_trim(v));
        } catch (const std::exception &) {
            throw std::runtime_error(std::string(key) + " = " + v + ": not a number");
        }
    };
    get("_NAME",        "pool-name",        [&](const std::string & v) { cfg.name        = v; });
    get("_PORT",        "pool-port",        [&](const std::string & v) { cfg.port        = std::stoi(v); });
    get("_DISCOVER",    "pool-discover",    [&](const std::string & v) { cfg.discover    = common_arg_utils::is_truthy(v); });
    get("_POLL_MS",     "pool-poll-ms",     [&](const std::string & v) { cfg.poll_ms     = std::max(100, std::stoi(v)); });
    get("_MOVE_TOKENS", "pool-move-tokens", [&](const std::string & v) { cfg.move_tokens = std::stoll(v); });
    get("_WAIT_MAX",    "pool-wait-max",    [&](const std::string & v) { cfg.wait_max_s  = std::stoi(v); });
    get("_PREFILL_TPS", "pool-prefill-tps", [&](const std::string & v) { cfg.prefill_tps = std::max(1.0, std::stod(v)); });
    get("_MIGRATE",     "pool-migrate",     [&](const std::string & v) { cfg.migrate     = common_arg_utils::is_truthy(v); });
    get("_DRAIN_MOVE",  "pool-drain-move",  [&](const std::string & v) { cfg.drain_move  = common_arg_utils::is_truthy(v); });
    get("_LINK_MBPS",   "pool-link-mbps",   [&](const std::string & v) { cfg.link_mbps   = std::max(1.0, std::stod(v)); });
    get("_PROBE_S",     "pool-probe-s",     [&](const std::string & v) { cfg.probe_s     = std::max(1, std::stoi(v)); });
    get("_SAME_MODEL",  "pool-same-model",  [&](const std::string & v) { cfg.same_model  = common_arg_utils::is_truthy(v); });
    get("_REBALANCE_S", "pool-rebalance-s", [&](const std::string & v) { cfg.rebalance_s = std::max(0, std::stoi(v)); });
    get("_QUIET_WAIT_S", "pool-quiet-wait-s", [&](const std::string & v) { cfg.quiet_wait_s = std::max(0, std::stoi(v)); });
    get("_SHORT_TOKENS", "pool-short-tokens", [&](const std::string & v) { cfg.short_tokens = std::max<int64_t>(0, std::stoll(v)); });
    get("_EVICT_IDLE_S", "pool-evict-idle-s", [&](const std::string & v) { cfg.evict_idle_s = std::max(0, std::stoi(v)); });
    get("_PEERS",       "pool-peers",       [&](const std::string & v) {
        size_t p = 0;
        while (p < v.size()) {
            const size_t comma = v.find(',', p);
            const std::string peer = pool_trim(v.substr(p, comma == std::string::npos ? std::string::npos : comma - p));
            p = comma == std::string::npos ? v.size() : comma + 1;
            if (!peer.empty()) {
                cfg.peers.push_back(peer);
            }
        }
    });
}

// a chunk of the move of a conversation, and of the probe of a link
static constexpr uint64_t POOL_CHUNK = 16ull << 20;
// the end of an answer the pool keeps for its timings (the last event of a stream, the end of a JSON body). at file
// scope: MSVC does not let a lambda use a constexpr local it does not capture (C3493)
static constexpr size_t POOL_TAIL = 16*1024;
// the most nodes a pool takes: the nodes live in a vector of this size, so that a node joins without moving the others
static constexpr int POOL_MAX_NODES = 64;
// the header of a request the pool sends to a node: the router there serves it itself
static constexpr const char * POOL_HOP_HEADER = "X-Llama-Pool";
// the header of the requests the pool makes itself: polls, probes, moves; logged at the debug level
static constexpr const char * POOL_QUIET_HEADER = "X-Llama-Pool-Poll";

// one way to a node: a direct cable, the WiFi
struct pool_link {
    std::string host;
    int         port      = 0;
    // under impl::mu. measured (bytes/s, 0 = not yet): a probe when the link first answers, then the moves
    double      bps_read  = 0.0; // from the node to this router
    double      bps_write = 0.0; // from this router to the node
    bool        probed    = false; // the probe was tried since the link last answered
    std::string error;           // why it failed the last time it was tried, "" = it answered
};

static std::string pool_base(const pool_link & l) {
    return "http://" + (l.host.find(':') != std::string::npos ? "[" + l.host + "]" : l.host) + ":" + std::to_string(l.port);
}

// the speed a link is ranked by: the slower way, 0 = not measured
static double pool_link_rank(const pool_link & l) {
    return l.probed ? std::min(l.bps_read, l.bps_write) : 0.0;
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

// a model of the list of a router (GET /models) under `name`: the name of its preset or an alias of it, as the router
// takes it in a request
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
    // fixed once the node joined, read without the lock
    std::string id;              // the router: <host>:<port>
    std::string name;
    std::string model;           // the name the router there takes the model under, set in every request to it
    bool        self = false;    // this router

    // state, under impl::mu (the links too: another one may join)
    std::vector<pool_link> links; // the first found first; the one in use is `link`
    int         link    = 0;
    int64_t     t_probe = 0;     // the last try of the faster links, while on a slower one
    std::string status = "down"; // up, down, loading, unloaded
    bool        draining = false;
    bool        mismatch = false; // another model than the most of the nodes (same_model): it takes no request
    bool        failed   = false; // the load of the model failed there (the exit code in error): no load is asked of it
    int         n_running  = 0;   // the other models its router runs, loaded or loading
    int         max_models = 0;   // models_max of its router (GET /props), 0 = no limit
    bool        want_load  = false; // a load of the model to ask of its router at the next poll
    bool        load_asked = false; // a load asked of its router and not refused, until a poll finds it on its way
    int64_t     t_load     = 0;   // the last load asked of it
    std::string signature;       // the portable signature of its prompt cache, "" = unknown (GET /pool/node)
    bool        migrate = false; // it moves conversations: the fork with a prompt cache
    uint64_t    cache_ram = 0;   // the RAM limit of its prompt cache, the largest state it takes; 0 = none
    json        model_info = json::object(); // its model (GET /pool/node): parameters, size, type, vocabulary
    // the room of its unified KV cache (GET /pool/node "room", server_room_report_fork): the cells its slots hold with
    // their answer reserve, the size, the admission margin and answer reserve, the tasks waiting in its queue; held -1:
    // no room logic there (no --kv-unified, an older node)
    int64_t     room_held     = -1;
    int64_t     room_slots    = 0;   // the slots counted in room_held
    int64_t     room_ctx      = 0;
    int64_t     room_margin   = 0;
    int64_t     room_answer   = 0;
    int64_t     room_deferred = 0;
    json        stats = json::object(); // what its model instance computed since it started (GET /pool/node "stats")
    json        gpus  = json::array();  // the temperature, power and fan of its GPUs (GET /pool/node "gpus")
    json        peers = json::array(); // its links to the other nodes (GET /pool/router), for the speed of a move
    std::string error;
    int64_t     t_ok      = 0;   // the last good poll (ms, ggml_time_ms)
    int         slots     = 0;   // capacity
    int         inflight  = 0;   // requests proxied by this router and not finished
    int         busy      = 0;   // slots processing, at the last poll
    int         external  = 0;   // of them, requests that did not come through this router
    uint64_t    n_requests = 0;
    double      gen_tps    = 0.0;
    double      prompt_tps = 0.0;
    double      prefill_tps = 0.0; // measured: the timings of the prompts of move_tokens and more it read, 0 = none yet
    // measured: the timings of the answers of 16 tokens and more that ran alone on the node (07.11: under load the slots
    // of a node share its GPU and a prompt next to an answer pauses it, a 395 with 2 slots measured half the speed of a
    // 3090 with 1 and its weight was half), 0 = none yet; the speed of all answers apart, for the dashboard
    double      decode_tps  = 0.0;
    double      decode_all  = 0.0;
    uint64_t    starts      = 0;   // the requests this router started there, to tell an answer that ran alone
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
        // the prompt counter of a slot moves once per batch (16384 tokens take ~18 s on the 395, the polls come every
        // second): its last move (time, count, tokens) and the speed measured over it
        int64_t t_pp        = 0;
        int64_t n_pp        = 0;
        int64_t dn_pp       = 0;
        double  pp_rate     = 0.0;
    };
    std::vector<slot_t> slot_state;
};

// what the timings of an answer say (prompt_n, prompt_ms, predicted_n, predicted_ms), ok = false without them
struct pool_timings {
    bool   ok       = false;
    double n_prompt = 0.0;
    double t_prompt = 0.0; // ms
    double n_gen    = 0.0;
    double t_gen    = 0.0; // ms
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
    int64_t              max_answer    = 0;  // the answer limit of the request (max_tokens and the like), 0 = none
    bool                 short_answer  = false; // max_answer within pool-short-tokens: it goes ahead in the queue
    int                  node          = -1; // where it went
    int64_t              t_start       = 0;
    std::string          stage;              // "moving": its state comes from another node
    bool                 alone         = false; // nothing else ran on its node when it started
    uint64_t             starts_at     = 0;  // the starts of its node with it: another start meanwhile = not alone

    int64_t cached_max() const {
        return cached.empty() ? 0 : *std::max_element(cached.begin(), cached.end());
    }
};

struct server_pool_fork::impl {
    const common_params & params;
    server_models_routes & models_routes;
    pool_config cfg;
    std::string model;           // the model of this router: the name of its preset
    std::set<std::string> names; // its name and aliases: a request for any of them goes through the pool
    std::string id;              // this router in the pool: <name or host>:<port>
    std::string instance = pool_instance(); // this process: another router with the same id is another machine
    std::string self_host;       // how this router reaches itself
    int         http_port = 0;
    std::string api_key;         // --api-key of this router: the other routers of the pool have it too

    // [0] this router, the others as they are found; a node never leaves (a router that stops is down)
    std::vector<pool_node> nodes = std::vector<pool_node>(POOL_MAX_NODES);
    std::atomic<int>       n_nodes{0};

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
        int64_t  t_moved = 0;      // when it went to another node last (wall ms)
        std::vector<int64_t> held; // per node: the tokens of the conversation it took last
        // the last request, for the export when its node is drained (evacuate): its kind, body and key; none for a
        // conversation shorter than pool-move-tokens
        std::string kind;
        json        body;
        pool_key    key;
        int64_t     t_evac_fail = 0; // when its state last failed to leave a drained node (wall ms)
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
    std::atomic<bool>        started{false};
    std::mutex               mu_stop;
    std::condition_variable  cv_stop;
    std::mutex               mu_threads;
    std::vector<std::thread> pollers; // under mu_threads

    impl(const common_params & params, server_models_routes & models_routes) : params(params), models_routes(models_routes) {}

    void init(const std::string & name_of_model) {
        model = name_of_model;
        names.insert(model);
        if (auto meta = models_routes.models.get_meta(model)) {
            names.insert(meta->aliases.begin(), meta->aliases.end());
            pool_read_options(cfg, meta->preset);
        }
        if (!params.api_keys.empty()) {
            api_key = params.api_keys.front();
        }
        http_port = params.port;
        char host[256] = {};
        if (gethostname(host, (int) sizeof(host) - 1) != 0 || host[0] == '\0') {
            snprintf(host, sizeof(host), "%s", "localhost");
        }
        // the id of the router in the pool: pool-name, else the host name; with the port, so that the routers of one
        // machine (the tests) differ
        id = (cfg.name.empty() ? std::string(host) : cfg.name) + ":" + std::to_string(http_port);
        if (cfg.name.empty()) {
            cfg.name = host;
        }
        // the address this router listens on, unless it is every address
        self_host = "127.0.0.1";
        if (!params.hostnames.empty()) {
            const std::string & h = params.hostnames.front();
            if (!h.empty() && h != "0.0.0.0" && h != "::" && h != "localhost") {
                self_host = h;
            }
        }
        pool_node & self = nodes[0];
        self.id    = id;
        self.name  = cfg.name;
        self.model = model;
        self.self  = true;
        pool_link l;
        l.host = self_host;
        l.port = http_port;
        self.links.push_back(l);
        n_nodes = 1;
    }

    // the headers of the requests the pool makes itself
    httplib::Headers auth_headers() const {
        httplib::Headers h = {{POOL_QUIET_HEADER, "1"}};
        if (!api_key.empty()) {
            h.emplace("Authorization", "Bearer " + api_key);
        }
        return h;
    }

    //
    // the nodes
    //

    // the node of router `node_id`, joined if new; -1 when the pool is full. under mu
    int node_of(const std::string & node_id, const std::string & name, const std::vector<std::string> & peer_names) {
        const int n = n_nodes.load();
        for (int i = 0; i < n; ++i) {
            if (nodes[i].id == node_id) {
                return i;
            }
        }
        if (n >= POOL_MAX_NODES) {
            return -1;
        }
        pool_node & node = nodes[n];
        node.id   = node_id;
        node.name = name.empty() ? node_id : name;
        for (int i = 0; i < n; ++i) {
            if (nodes[i].name == node.name) {
                node.name = node_id; // two routers of one name (one host): the port tells them apart
                break;
            }
        }
        // the name it takes the model under: this router's own, else an alias both have
        node.model = model;
        if (std::find(peer_names.begin(), peer_names.end(), model) == peer_names.end()) {
            for (const auto & nm : names) {
                if (std::find(peer_names.begin(), peer_names.end(), nm) != peer_names.end()) {
                    node.model = nm;
                    break;
                }
            }
        }
        n_nodes = n + 1;
        return n;
    }

    // a router of the pool reached at host:port: its node, and the link if new
    void found(const std::string & node_id, const std::string & name, const std::string & host, int port,
               const std::vector<std::string> & peer_names, const char * how) {
        int  n;
        bool joined = false;
        bool linked = false;
        {
            std::lock_guard<std::mutex> lk(mu);
            const int before = n_nodes.load();
            n = node_of(node_id, name, peer_names);
            if (n < 0) {
                return;
            }
            joined = n >= before;
            pool_node & node = nodes[n];
            bool have = false;
            for (const auto & l : node.links) {
                have = have || (l.host == host && l.port == port);
            }
            if (!have) {
                pool_link l;
                l.host = host;
                l.port = port;
                node.links.push_back(l);
                linked = true;
                event(joined ? "found" : "link+", "", n, string_format("%s at %s:%d (%s)", node.id.c_str(), host.c_str(), port, how));
                SRV_INF("pool: %s %s at %s:%d (%s)\n", joined ? "found" : "another link to", node.name.c_str(), host.c_str(), port, how);
            }
            cv.notify_all();
        }
        if (joined && linked) {
            start_poller(n);
        }
    }

    void start_poller(int n) {
        std::lock_guard<std::mutex> lk(mu_threads);
        if (stopping.load()) {
            return;
        }
        pollers.emplace_back([this, n]() { poll_node(n); });
    }

    //
    // under mu
    //

    static bool usable(const pool_node & n) {
        return n.status == "up" && !n.draining && !n.mismatch;
    }

    // the prefill speed of a node: measured (note_timings), else the one of the pool
    double prefill_of(int n) const {
        return nodes[n].prefill_tps > 0 ? nodes[n].prefill_tps : cfg.prefill_tps;
    }

    // the share of the new conversations a node takes: its speed for one conversation alone (decode_tps) against the
    // fastest node, 1 until measured
    double weight_of(int n) const {
        double best = 0.0;
        for (int i = 0; i < n_nodes.load(); ++i) {
            best = std::max(best, nodes[i].decode_tps);
        }
        return best > 0 && nodes[n].decode_tps > 0 ? nodes[n].decode_tps / best : 1.0;
    }

    // the speed (bytes/s) a state moves at from `from` to `to`, which reads it from there itself: the link of `to` to
    // `from` as `to` measured it (this router's own, or the one the node reports in GET /pool/router), else the default
    double pair_bps(int from, int to) const {
        const pool_node & t = nodes[to];
        if (t.self) {
            const pool_node & f = nodes[from];
            const double m = f.links.empty() ? 0.0 : f.links[f.link].bps_read;
            return m > 0 ? m : cfg.link_mbps * 1e6 / 8;
        }
        for (const auto & p : t.peers) {
            if (!p.is_object() || json_value(p, "id", std::string()) != nodes[from].id || !p.contains("links")) {
                continue;
            }
            for (const auto & l : p.at("links")) {
                if (l.is_object() && json_value(l, "active", false) && json_value(l, "mbps_in", 0.0) > 0) {
                    return json_value(l, "mbps_in", 0.0) * 1e6 / 8;
                }
            }
        }
        return cfg.link_mbps * 1e6 / 8;
    }

    // a conversation can move from `from` to `to` with its state: both nodes of the fork with the prompt cache, of the
    // same model and configuration (the portable signature of the cache), `from` answers
    bool can_migrate(int from, int to) const {
        const pool_node & a = nodes[from];
        const pool_node & b = nodes[to];
        return cfg.migrate && from != to && a.status == "up" && a.migrate && b.migrate && !a.signature.empty() &&
               a.signature == b.signature;
    }

    // a move of `bytes` at `bps` bytes/s: the requests, the transfer, and the copies on the two nodes at 2 GB/s (the
    // export reads the file of a state on disk or copies one in RAM, the target restores it into a slot)
    static double move_s(double bytes, double bps) {
        return 0.2 + bytes / bps + bytes / 2e9;
    }

    // how long a move of a state of `tokens` takes, with the bytes per token of the moves so far. 0 before the first
    // move: unknown, a conversation then moves rather than waits
    double transfer_s(int from, int to, int64_t tokens) const {
        if (state_bpt <= 0) {
            return 0.0;
        }
        return move_s(tokens * state_bpt, pair_bps(from, to));
    }

    static int free_slots(const pool_node & n) {
        return usable(n) ? n.slots - n.inflight - n.external : 0;
    }

    // the free node for a request. the history of a conversation: the node with the most of it cached, then the most free
    // slots by weight, then the fewest requests. a new conversation (balance): the most free slots by weight first, the
    // cache only between nodes as free - what it can have cached is the prefix it shares with others (the system prompt
    // and the tools of an agent), a prefill a node pays once, while the node it goes to serves the whole conversation:
    // with the cache first every new conversation of an agent went to the node that served the first one while it had a
    // free slot (03.11: one node full, one with a single request, one idle)
    // quiet: a request with a long prompt to read first takes a node where nothing runs, if there is one: a prompt
    // pauses the generating slots of its node until it is read (LLAMA_PREFILL_EXCLUSIVE)
    int best_free(const pool_req & r, bool balance = false, bool quiet = false) const {
        const std::vector<int64_t> & cached = r.cached;
        int best = -1;
        for (int i = 0; i < n_nodes.load(); ++i) {
            const int f = free_slots(nodes[i]);
            if (f <= 0 || !room_fits(i, r)) {
                continue;
            }
            if (best < 0) {
                best = i;
                continue;
            }
            if (quiet && (streams(nodes[i]) == 0) != (streams(nodes[best]) == 0)) {
                best = streams(nodes[i]) == 0 ? i : best;
                continue;
            }
            const int64_t ci = pool_at(cached, i);
            const int64_t cb = pool_at(cached, best);
            const double  si = weight_of(i) * f / std::max(1, nodes[i].slots);
            const double  sb = weight_of(best) * free_slots(nodes[best]) / std::max(1, nodes[best].slots);
            if (balance && (si > sb + 1e-9 || si < sb - 1e-9)) {
                best = si > sb ? i : best;
                continue;
            }
            if (ci != cb) {
                best = ci > cb ? i : best;
                continue;
            }
            if (si > sb + 1e-9 || (si > sb - 1e-9 && nodes[i].n_requests < nodes[best].n_requests)) {
                best = i;
            }
        }
        return best;
    }

    // whether the prompt of `r` fits into the unified KV cache of node i next to what its slots hold, as the node admits
    // a task (room_for_task_fork): one that does not fit waits inside the node until a slot there finishes, where the
    // pool can no longer send it to a node that frees first (03.11: 4 agents of 34-140K tok in the 262K of a 395, turns
    // of 200-887 s there against 6-90 s on the 3090 for the same conversations). the node reports at the start of a
    // step, seconds apart during a long prefill: the requests of this router it has not counted yet (the last ones
    // sent, as many as it runs there over the slots in the report) count by their estimate. a node with tasks waiting
    // in its queue has no room. without room logic there (held -1) every prompt fits. under mu
    bool room_fits(int i, const pool_req & r) const {
        const pool_node & n = nodes[i];
        if (n.room_held < 0 || n.room_ctx <= 0) {
            return true;
        }
        if (n.room_deferred > 0) {
            return false;
        }
        if (r.prompt_tokens + n.room_margin > n.room_ctx) {
            return true; // longer than the cache: the node lets it go, to the error of its own path
        }
        // the answer reserve of the node, less when the request allows less
        auto answer = [&](const pool_req & q) {
            return q.max_answer > 0 ? std::min(n.room_answer, q.max_answer) : n.room_answer;
        };
        std::vector<const pool_req *> mine;
        for (const auto & e : running) {
            if (e.second.node == i) {
                mine.push_back(&e.second);
            }
        }
        std::sort(mine.begin(), mine.end(), [](const pool_req * a, const pool_req * b) { return a->t_start > b->t_start; });
        int64_t held = n.room_held;
        for (int64_t k = 0; k < (int64_t) mine.size() - n.room_slots; ++k) {
            held += mine[k]->prompt_tokens + answer(*mine[k]);
        }
        return held == 0 || held + r.prompt_tokens + answer(r) + n.room_margin <= n.room_ctx;
    }

    // whether the other requests in the queue fill the free slots of node n with no history to compute there: new
    // conversations, and those whose node with the most cached is n (07.11: with a queue on every node a conversation
    // that waited as long as its prompt takes went to the first node to free a slot and was computed there from the start,
    // in the slot of a request that node had cached)
    bool others_fill(const pool_req & r, int n) const {
        int want = 0;
        for (const pool_req * q : queue) {
            if (q != &r && (q->conv_new || holder(*q) == n)) {
                ++want;
            }
        }
        return want > 0 && want >= free_slots(nodes[n]);
    }

    // the usable node with the most of the prompt cached, -1 = none holds any of it
    int holder(const pool_req & r) const {
        int best = -1;
        for (int i = 0; i < n_nodes.load() && i < (int) r.cached.size(); ++i) {
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
        // a prompt this long to read even where most of it is cached pauses the generation of its node for a while
        const bool long_prompt = r.prompt_tokens - r.cached_max() >= cfg.move_tokens;
        const int alt = best_free(r, r.conv_new, r.conv_new && long_prompt);
        const int hold = holder(r);
        if (alt < 0 || hold < 0 || r.conv_new) {
            return quiet_wait(r, alt, now);
        }
        const int light = best_free(r, true, true);
        if (rebalance(r, hold, light)) {
            return quiet_wait(r, light, now);
        }
        // its node is busy (best_free takes the node with the most cached when it is free). while the other requests in
        // the queue fill the free slots of alt with nothing to compute again there, a move only reorders the work: the
        // slot it takes on alt would serve one of them, and the pool pays the move on top - the difference computed, the
        // state read or sent, one more copy of it in a cache (07.11, user: both nodes full with queues of 4-6, the
        // conversations went back and forth for nothing). it waits for its node then, up to pool-wait-max; it leaves only
        // for a slot that would stand idle otherwise (and a crowded node or a long prompt for a quiet one, above)
        if (alt != hold && others_fill(r, alt)) {
            return (now - r.t_arrive) / 1000.0 < cfg.wait_max_s ? -1 : quiet_wait(r, alt, now);
        }
        const int64_t gain = pool_at(r.cached, hold) - pool_at(r.cached, alt);
        if (gain >= cfg.move_tokens) {
            // waiting is worth at most what going elsewhere costs: computing the difference, or moving the state
            double limit_s = std::min<double>(cfg.wait_max_s, gain / prefill_of(alt));
            if (can_migrate(hold, alt)) {
                limit_s = std::min(limit_s, transfer_s(hold, alt, pool_at(r.cached, hold)));
            }
            if ((now - r.t_arrive) / 1000.0 < limit_s) {
                return -1;
            }
        }
        return quiet_wait(r, alt, now);
    }

    // whether `q` (queued before `r` when `older`) takes a node before `r`: a request for a short answer (pool-short-tokens:
    // the permission check of an agent, which waits for it before it runs its tool; 03.11: such requests of Qwen Code
    // waited 14-33 s in the queue behind turns of minutes) goes ahead of one for a long answer, unless that one waited
    // pool-wait-max already; else the older one first
    bool goes_first(const pool_req & q, bool older, const pool_req & r, int64_t now) const {
        auto aged = [&](const pool_req & x) {
            return cfg.wait_max_s > 0 && (now - x.t_arrive) / 1000.0 >= cfg.wait_max_s;
        };
        if (q.short_answer != r.short_answer) {
            const pool_req & s = q.short_answer ? q : r; // the short one
            const pool_req & l = q.short_answer ? r : q; // the long one
            if (!aged(l)) {
                return &s == &q;
            }
        }
        return older;
    }

    // a long prompt (pool-move-tokens and more to read) for a node where requests run waits for that node or another to go
    // quiet: read there now, it would pause their answers (LLAMA_PREFILL_EXCLUSIVE) - and an answer ends. it waits as
    // long as the pause would cost (its read time times the requests that run there), at most quiet_wait_s seconds
    int quiet_wait(const pool_req & r, int n, int64_t now) const {
        if (n < 0 || cfg.quiet_wait_s <= 0 || r.short_answer) {
            return n; // a short answer does not wait: its client waits for it
        }
        const int64_t todo = r.prompt_tokens - pool_at(r.cached, n);
        const int     busy = streams(nodes[n]);
        if (todo < cfg.move_tokens || busy == 0) {
            return n;
        }
        const double limit_s = std::min<double>(cfg.quiet_wait_s, todo / prefill_of(n) * busy);
        return (now - r.t_arrive) / 1000.0 < limit_s ? -1 : n;
    }

    // the requests in work on a node, of this router and of the others
    static int streams(const pool_node & n) {
        return n.inflight + n.external;
    }

    // a conversation leaves its node for a free one with clearly fewer requests in work: every stream a node runs slows
    // the others there (the 395: one stream 39 t/s, four 17 t/s each), while other nodes stand. it moves when its node,
    // with it, would run 1.5 streams more than the other by the speed of the nodes, and the move costs at most
    // rebalance_s seconds (the transfer of its state, or computing it there); once in 5 minutes at most, so that the pool
    // settles. under mu
    // a conversation with a long new part to read (pool-move-tokens and more) also leaves a node where other requests
    // run for one where nothing runs: there its prompt pauses nobody
    bool rebalance(const pool_req & r, int hold, int light) const {
        if (cfg.rebalance_s <= 0 || light < 0 || light == hold) {
            return false;
        }
        const double load_hold  = (streams(nodes[hold])  + 1) / std::max(0.05, weight_of(hold));
        const double load_light = (streams(nodes[light]) + 1) / std::max(0.05, weight_of(light));
        const bool crowded = load_hold >= load_light + 1.5;
        const bool long_prompt = r.prompt_tokens - pool_at(r.cached, hold) >= cfg.move_tokens &&
                                 streams(nodes[hold]) > 0 && streams(nodes[light]) == 0;
        if (!crowded && !long_prompt) {
            return false;
        }
        auto it = convs.find(r.conv);
        if (it != convs.end() && pool_wall_ms() - it->second.t_moved < 300000) {
            return false;
        }
        double cost = std::max<int64_t>(0, pool_at(r.cached, hold) - pool_at(r.cached, light)) / prefill_of(light);
        if (can_migrate(hold, light)) {
            cost = std::min(cost, transfer_s(hold, light, pool_at(r.cached, hold)));
        }
        return cost <= cfg.rebalance_s;
    }

    // what each node holds of the prompt, and the conversation the request continues
    void lookup(const pool_key & k, pool_req & r) const {
        const int n = n_nodes.load();
        r.prompt_tokens = k.chars.empty() ? 0 : (int64_t) (k.chars.back() / cfg.chars_tok);
        r.cached.assign(n, 0);
        r.shared_max.assign(n, 0);
        std::vector<int> depth(n, 0);
        if (!k.id.empty()) {
            auto it = affinity.find(pool_hash(POOL_ID_SEED, k.id));
            if (it != affinity.end()) {
                for (const auto & e : it->second) {
                    if (e.first < n && nodes[e.first].status != "down") {
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
                if (e.first < n && nodes[e.first].status != "down" && depth[e.first] == 0) {
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

    // the requests for the model wait or come: the routers that have it unloaded load it, so that every node takes a
    // part - those with room for it (fewer other models running than their models_max), each once a minute at most; and
    // when no node has it or loads it and none has room, one more that unloads another model for it, this router
    // first, as a router without a pool would - but only a model idle for evict_idle_s there (the router decides, also
    // for the pools of other entries): two pools of models that fit there one at a time do not swap them every minute.
    // a failed load is asked again after 5 minutes: on the 395 a load runs out of device memory now and then, and the
    // next one goes through. under mu
    void wake(int64_t now) {
        auto loadable = [&](const pool_node & n) {
            return n.status == "unloaded" && !n.draining && !n.mismatch && !n.want_load &&
                   now - n.t_load > (n.failed ? 300000 : 60000);
        };
        bool have = false;
        for (int i = 0; i < n_nodes.load(); ++i) {
            const pool_node & n = nodes[i];
            have = have || usable(n) || n.status == "loading" || n.want_load || n.load_asked;
        }
        bool room = false;
        for (int i = 0; i < n_nodes.load(); ++i) {
            pool_node & n = nodes[i];
            if (loadable(n) && (n.max_models <= 0 || n.n_running < n.max_models)) {
                n.want_load = true;
                room = true;
            }
        }
        if (!have && !room) {
            for (int i = 0; i < n_nodes.load(); ++i) {
                if (loadable(nodes[i])) {
                    nodes[i].want_load = true;
                    break;
                }
            }
        }
    }

    // a node for a request: waits for a free slot, -1 when the client went away, -2 when no node is up. the model
    // loads on the routers that do not have it (wake); while one loads the requests wait for it
    int acquire(pool_req & r, const std::function<bool()> & should_stop) {
        std::unique_lock<std::mutex> lk(mu);
        queue.push_back(&r);
        int64_t t_none = 0;
        while (true) {
            const int64_t now = ggml_time_ms();
            if (should_stop() || stopping.load()) {
                queue.erase(std::find(queue.begin(), queue.end(), &r));
                event("gone", r.conv, -1, stopping.load() ? "the router stops" : "the client left while it waited");
                return -1;
            }
            wake(now);
            int n = choose(r, now);
            if (n >= 0) {
                // another request that would take the same node goes first: an older one, or one for a short answer
                bool older = true;
                for (pool_req * q : queue) {
                    if (q == &r) {
                        older = false;
                        continue;
                    }
                    if (goes_first(*q, older, r, now) && choose(*q, now) == n) {
                        n = -1;
                        break;
                    }
                }
            }
            if (n >= 0) {
                queue.erase(std::find(queue.begin(), queue.end(), &r));
                r.alone = streams(nodes[n]) == 0;
                nodes[n].inflight++;
                nodes[n].n_requests++;
                r.starts_at = ++nodes[n].starts;
                r.node = n;
                r.t_start = now;
                running[r.id] = r;
                return n;
            }
            bool any = false;
            bool loading = false;
            for (int i = 0; i < n_nodes.load(); ++i) {
                any = any || usable(nodes[i]);
                loading = loading || nodes[i].status == "loading" || nodes[i].want_load || nodes[i].load_asked;
            }
            if (!any && !loading) {
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

    // the end of a request. a done event with the timings of the answer: the prompt, the answer and the rest of the time
    // (a state moved before it, the queue of the node: room in a unified cache, slots, the network), where a slow
    // node shows what takes its time
    void release(uint64_t req_id, int status, const pool_timings & t = {}) {
        std::lock_guard<std::mutex> lk(mu);
        auto it = running.find(req_id);
        if (it == running.end()) {
            return;
        }
        const pool_req & r = it->second;
        nodes[r.node].inflight--;
        const double total = (ggml_time_ms() - r.t_start) / 1000.0;
        std::string text = string_format("HTTP %d, %.1f s", status, total);
        if (t.ok) {
            const double rest = total - (t.t_prompt + t.t_gen) / 1000.0;
            text += string_format(": prompt %.0f tok %.1f s, answer %.0f tok %.1f s", t.n_prompt, t.t_prompt / 1000.0,
                                  t.n_gen, t.t_gen / 1000.0);
            if (t.t_gen > 0 && t.n_gen > 0) {
                text += string_format(" (%.1f t/s)", t.n_gen * 1000.0 / t.t_gen);
            }
            if (rest >= 0.5) {
                text += string_format(", the rest %.1f s", rest);
            }
        }
        event(status >= 400 ? "error" : "done", r.conv, r.node, text);
        running.erase(it);
        cv.notify_all();
    }

    // the last "timings" object of an answer (the final event of a stream, or the end of a JSON body; it holds no
    // nested object):
    // - the prompt it read, when that was at least move_tokens - the size of the prefills the speed is weighed for, where
    //   a short prompt shows the overhead of a request more than the speed
    // - the answer, when it was 16 tokens or more: the decode speed the new conversations are shared out by
    // - with `shared` (the fraction of the prompt the node held from other conversations, estimated from the
    //   characters; at least move_tokens of it) the share of it the node took from its cache (cache_n). a node that took
    //   more than reaches to the end of the next message (`shared_max`), where the prompt differs from the others, held
    //   this conversation itself (one the pool forgot, as after its restart): no sample
    pool_timings note_timings(int n, const std::string & tail, double shared, double shared_max, uint64_t req_id) {
        pool_timings res;
        const size_t at  = tail.rfind("\"timings\":");
        const size_t beg = at == std::string::npos ? std::string::npos : tail.find('{', at);
        const size_t end = beg == std::string::npos ? std::string::npos : tail.find('}', beg);
        if (end == std::string::npos) {
            return res;
        }
        const json t = json::parse_no_throw(tail.substr(beg, end - beg + 1));
        if (!t.is_object()) {
            return res;
        }
        const double n_prompt = json_value(t, "prompt_n", 0.0);
        const double n_cache  = json_value(t, "cache_n", 0.0);
        const double t_ms     = json_value(t, "prompt_ms", 0.0);
        const double n_gen    = json_value(t, "predicted_n", 0.0);
        const double t_gen    = json_value(t, "predicted_ms", 0.0);
        res = {true, n_prompt, t_ms, n_gen, t_gen};
        auto avg = [](double old, double x, double none) { return old > none ? 0.5*old + 0.5*x : x; };
        std::lock_guard<std::mutex> lk(mu);
        if (n_prompt >= cfg.move_tokens && t_ms > 0) {
            nodes[n].prefill_tps = avg(nodes[n].prefill_tps, n_prompt*1000.0/t_ms, 0.0);
        }
        if (n_gen >= 16 && t_gen > 0) {
            // the weight of a node: one stream on it, nothing next to it from start to end (other routers included)
            auto it = running.find(req_id);
            const bool alone = it != running.end() && it->second.alone && it->second.starts_at == nodes[n].starts &&
                               nodes[n].external == 0;
            if (alone) {
                nodes[n].decode_tps = avg(nodes[n].decode_tps, n_gen*1000.0/t_gen, 0.0);
            }
            nodes[n].decode_all = avg(nodes[n].decode_all, n_gen*1000.0/t_gen, 0.0);
        }
        const double total = n_prompt + n_cache;
        if (shared > 0 && total > 0 && n_cache <= shared_max * total) {
            nodes[n].share = avg(nodes[n].share, std::min(1.0, n_cache / (shared * total)), -1.0);
        }
        return res;
    }

    // the link a node is on, a copy
    pool_link link_of(int n) {
        std::lock_guard<std::mutex> lk(mu);
        const pool_node & node = nodes[n];
        return node.links[node.link];
    }

    std::map<std::string, std::string> headers_for(const std::map<std::string, std::string> & in) const {
        std::map<std::string, std::string> out = in;
        for (auto it = out.begin(); it != out.end(); ) {
            const std::string k = pool_lower(it->first);
            it = (k == "authorization" && !api_key.empty()) || k == pool_lower(POOL_HOP_HEADER) ? out.erase(it) : std::next(it);
        }
        if (!api_key.empty()) {
            out["Authorization"] = "Bearer " + api_key;
        }
        out[POOL_HOP_HEADER] = id;
        return out;
    }

    // the request to node `n`: the router there picks the instance by the model in the body
    std::unique_ptr<server_http_proxy> proxy_to(int n, const server_http_req & req, const std::string & method, json body) {
        const pool_link l = link_of(n);
        std::string path = req.path;
        if (!req.query_string.empty()) {
            path += '?' + req.query_string;
        }
        body["model"] = nodes[n].model;
        return std::make_unique<server_http_proxy>(method, "http", l.host, l.port, path,
            headers_for(req.headers), body.dump(), req.files, req.should_stop,
            params.timeout_read, params.timeout_write);
    }

    // the conversation of `r` moves from `from` to `to` with its state: `from` gives the cached state of the prompt
    // (POST /pool/export), `to` reads it from `from` itself and takes it into its prompt cache in RAM (POST /pool/pull).
    // false with `why` when it did not move: `to` then computes the conversation from the start, as without a move
    // what node `to` holds of the prompt in fact, in its tokens (the longest prefix in its cache and its idle slots, an
    // upper bound of what a slot restores there), and the tokens of the prompt there; -1 when the node cannot tell. the
    // pool's own figure is where the conversation went before, and the node may have let that copy go since (07.11)
    int64_t probe_hold(int to, const char * kind, const json & body, int64_t & n_tokens) {
        const pool_node & nt = nodes[to]; // the fixed fields only, the state is under mu
        pool_link lt;
        {
            std::lock_guard<std::mutex> lk(mu);
            lt = nt.links[nt.link];
        }
        httplib::Client ct(pool_base(lt));
        ct.set_connection_timeout(3, 0);
        ct.set_read_timeout(30, 0);
        json req_pr = {{"kind", kind}, {"body", body}, {"probe", true}, {"model", nt.model}};
        req_pr["body"]["model"] = nt.model;
        auto rx = ct.Post("/pool/export?model=" + nt.model + "&autoload=0", auth_headers(), req_pr.dump(), "application/json");
        const json ex = rx && rx->status == 200 ? json::parse_no_throw(rx->body) : json();
        if (!ex.is_object() || !ex.contains("n_match")) {
            return -1;
        }
        n_tokens = json_value(ex, "n_tokens", (int64_t) 0);
        return json_value(ex, "n_match", (int64_t) -1);
    }

    // to_held: what `to` holds of the prompt in its tokens as it answered a probe, -1 = the estimate of the pool
    bool migrate(const pool_req & r, int from, int to, const char * kind, const json & body,
                 const std::function<bool()> & should_stop, std::string & why, std::string & detail, int64_t to_held = -1) {
        const pool_node & nf = nodes[from]; // the fixed fields only, the state is under mu
        const pool_node & nt = nodes[to];
        pool_link lf;
        pool_link lt;
        uint64_t cache_ram_to;
        {
            std::lock_guard<std::mutex> lk(mu);
            lf = nf.links[nf.link];
            lt = nt.links[nt.link];
            cache_ram_to = nt.cache_ram;
        }
        const httplib::Headers hdrs = auth_headers();
        httplib::Client cf(pool_base(lf));
        httplib::Client ct(pool_base(lt));
        cf.set_connection_timeout(3, 0);
        ct.set_connection_timeout(3, 0);
        // the export copies a state only in RAM, behind the batch the node is computing; the pull moves it
        cf.set_read_timeout(params.timeout_read, 0);
        ct.set_read_timeout(params.timeout_read, 0);
        cf.set_write_timeout(60, 0);
        ct.set_write_timeout(60, 0);
        // a router takes the model in ?model= and in the body of a POST; autoload=0: a move never loads a model
        const std::string q_from = "?model=" + nf.model + "&autoload=0";

        const int64_t t0 = ggml_time_ms();
        json req_ex = {{"kind", kind}, {"body", body}, {"min_match", cfg.move_tokens}, {"model", nf.model}};
        req_ex["body"]["model"] = nf.model;
        auto rx = cf.Post("/pool/export" + q_from, hdrs, req_ex.dump(), "application/json");
        if (!rx || rx->status != 200) {
            why = "no state: " + pool_error_text(rx);
            return false;
        }
        const json ex = json::parse_no_throw(rx->body);
        const std::string xid = ex.is_object() ? json_value(ex, "id", std::string()) : std::string();
        const std::string sig = ex.is_object() ? json_value(ex, "signature", std::string()) : std::string();
        const uint64_t size   = ex.is_object() ? json_value(ex, "size", (uint64_t) 0) : 0;
        const int64_t n_tok   = ex.is_object() ? json_value(ex, "n_tokens", (int64_t) 0) : 0;
        const int64_t n_reuse = ex.is_object() ? json_value(ex, "n_reuse", (int64_t) 0) : 0;
        const int64_t n_ckpt  = ex.is_object() ? json_value(ex, "n_ckpt", (int64_t) 0) : 0;
        const int64_t n_ckpt_all = ex.is_object() ? json_value(ex, "n_ckpt_all", (int64_t) 0) : 0;
        const bool from_disk  = ex.is_object() && json_value(ex, "from_disk", false);
        auto drop_export = [&]() {
            cf.set_read_timeout(10, 0);
            cf.Post("/pool/export/done" + q_from, hdrs, json{{"id", xid}, {"model", nf.model}}.dump(), "application/json");
        };
        if (xid.empty() || size == 0) {
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
        // the state goes whenever it brings something: a transfer takes the network and the disks, computing what it
        // brings takes the GPU every queued request waits for (07.11, user: "a transfer is always cheaper"; before, the
        // pool compared the estimated times and computed when that came out shorter)
        const int64_t gain = std::max<int64_t>(0, n_reuse - (to_held >= 0 ? to_held : pool_at(r.cached, to)));
        if (gain <= 0) {
            why = string_format("the state reaches ~%lld tok (a checkpoint), the node holds ~%lld", (long long) n_reuse,
                                (long long) (to_held >= 0 ? to_held : pool_at(r.cached, to)));
            drop_export();
            std::lock_guard<std::mutex> lk(mu);
            state_bpt = state_bpt > 0 ? 0.5*state_bpt + 0.5*size / std::max<int64_t>(1, n_tok) : (double) size / std::max<int64_t>(1, n_tok);
            return false;
        }

        // the pull, on the other node; a client that leaves stops it
        std::atomic<bool> pulled{false};
        std::thread watch([&]() {
            while (!pulled.load()) {
                if (should_stop()) {
                    ct.stop();
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        });
        const json req_pull = {{"from", nf.id}, {"id", xid}, {"size", size}, {"signature", sig},
                               {"source_model", nf.model}, {"model", nt.model}};
        auto rp = ct.Post("/pool/pull", hdrs, req_pull.dump(), "application/json");
        pulled = true;
        watch.join();
        drop_export();
        if (!rp || rp->status != 200) {
            why = should_stop() ? "the client left" : "pull: " + pool_error_text(rp);
            return false;
        }
        const json res = json::parse_no_throw(rp->body);
        const double dt = std::max<int64_t>(1, ggml_time_ms() - t0) / 1000.0;
        {
            std::lock_guard<std::mutex> lk(mu);
            state_bpt = state_bpt > 0 ? 0.5*state_bpt + 0.5*size / std::max<int64_t>(1, n_tok) : (double) size / std::max<int64_t>(1, n_tok);
        }
        const double t_read = res.is_object() ? json_value(res, "read_s", 0.0) : 0.0;
        detail = string_format("~%lld tok from %s (%s): %.0f MB in %.1f s (read %.0f MB/s), %lld of %lld checkpoints",
                               (long long) n_tok, nf.name.c_str(), from_disk ? "its file" : "RAM", size / 1e6, dt,
                               t_read > 0 ? size / 1e6 / t_read : 0.0, (long long) n_ckpt, (long long) n_ckpt_all);
        return true;
    }

    // POST /pool/pull, on the node a conversation moves to: the state of export `id` from node `from`, read from there
    // in chunks (GET /pool/export/data) and written to the instance of this router (POST /pool/import/data, a thread
    // each way, so that the network and the import run at once), which takes it into its prompt cache in RAM
    server_http_res_ptr pull(const server_http_req & req) {
        const json body = json::parse_no_throw(req.body);
        if (body.is_discarded() || !body.is_object()) {
            return pool_error(400, "expected {\"from\": <node>, \"id\": <export>, \"size\": <bytes>, ...}");
        }
        const std::string from_id = json_value(body, "from", std::string());
        const std::string xid     = json_value(body, "id", std::string());
        const std::string sig     = json_value(body, "signature", std::string());
        const std::string src_model = json_value(body, "source_model", std::string());
        const std::string dst_model = json_value(body, "model", model);
        const uint64_t size       = json_value(body, "size", (uint64_t) 0);
        int src = -1;
        int src_link = 0;
        pool_link lf;
        {
            std::lock_guard<std::mutex> lk(mu);
            for (int i = 1; i < n_nodes.load(); ++i) {
                if (nodes[i].id == from_id && !nodes[i].links.empty()) {
                    src      = i;
                    src_link = nodes[i].link;
                    lf       = nodes[i].links[src_link];
                }
            }
        }
        if (src < 0 || size == 0 || xid.empty()) {
            // 400, not 404: the HTTP server puts its own body into every 404
            return pool_error(400, "no node " + from_id + " in the pool of " + id);
        }
        const httplib::Headers hdrs = auth_headers();
        httplib::Client cf(pool_base(lf));
        httplib::Client ct(self_host, http_port);
        cf.set_connection_timeout(3, 0);
        ct.set_connection_timeout(3, 0);
        cf.set_read_timeout(60, 0);
        ct.set_read_timeout(60, 0);
        cf.set_write_timeout(60, 0);
        ct.set_write_timeout(60, 0);
        const std::string q_src = "&model=" + src_model + "&autoload=0";
        const std::string q_dst = "model=" + dst_model + "&autoload=0";

        // room in the RAM of the instance first (07.11): the state takes its whole size in RAM as its chunks come, and the
        // RAM of a 395 is what the GPU leaves - the cache there moves older states to disk for it now, or says it cannot,
        // before gigabytes cross the network
        {
            auto rr = ct.Post("/pool/import/done?" + q_dst, hdrs, json{{"room", size}, {"model", dst_model}}.dump(), "application/json");
            // 404: an instance of an older build takes the request for a transfer it does not know, and makes room after
            if (!rr || (rr->status != 200 && rr->status != 404)) {
                return pool_error(507, "no room for the state: " + pool_error_text(rr));
            }
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
                auto rd = cf.Get("/pool/export/data?id=" + xid + "&offset=" + std::to_string(off) + "&size=" +
                                 std::to_string(POOL_CHUNK) + q_src, hdrs);
                std::lock_guard<std::mutex> lk(m);
                if (!rd || rd->status != 200 || rd->body.empty()) {
                    failed = true;
                    err = "read from " + from_id + ": " + (rd && rd->body.empty() ? "no data" : pool_error_text(rd));
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

        const int64_t t0 = ggml_time_ms();
        uint64_t off = 0;
        double t_write = 0.0;
        while (true) {
            std::string chunk;
            {
                std::unique_lock<std::mutex> lk(m);
                cq.wait_for(lk, std::chrono::milliseconds(200), [&] { return !q.empty() || read_done || failed; });
                if (!failed && req.should_stop && req.should_stop()) {
                    failed = true;
                    err = "the router that asked for it left";
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
            auto rw = ct.Post("/pool/import/data?" + q_dst + "&id=" + xid + "&offset=" + std::to_string(off), hdrs, chunk,
                              "application/octet-stream");
            if (!rw || rw->status != 200) {
                std::lock_guard<std::mutex> lk(m);
                failed = true;
                err = "import: " + pool_error_text(rw);
                cq.notify_all();
                break;
            }
            t_write += (ggml_time_us() - ts) / 1e6;
            off     += chunk.size();
        }
        reader.join();

        auto abort_import = [&]() {
            ct.Post("/pool/import/done?" + q_dst, hdrs, json{{"id", xid}, {"abort", true}, {"model", dst_model}}.dump(), "application/json");
        };
        if (failed || off != size) {
            abort_import();
            return pool_error(502, failed ? err : "the state came short");
        }
        auto ri = ct.Post("/pool/import/done?" + q_dst, hdrs, json{{"id", xid}, {"signature", sig}, {"model", dst_model}}.dump(),
                          "application/json");
        if (!ri || ri->status != 200) {
            abort_import();
            return pool_error(502, "import: " + pool_error_text(ri));
        }
        {
            std::lock_guard<std::mutex> lk(mu);
            pool_link & l = nodes[src].links[src_link];
            if (t_read > 0) {
                l.bps_read = l.bps_read > 0 ? 0.5*l.bps_read + 0.5*size / t_read : size / t_read;
            }
        }
        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str({{"size", size}, {"seconds", (ggml_time_ms() - t0) / 1000.0}, {"read_s", t_read},
                                      {"write_s", t_write}});
        return res;
    }

    // a request for the model: the node by the conversation and the free slots, waiting in the queue if need be
    server_http_res_ptr forward_slot(const server_http_req & req, const std::string & method, const json & body) {
        const pool_key key = pool_key_of(req.path, body, req.headers);

        pool_req r;
        r.path = req.path;
        r.t_arrive = ggml_time_ms();
        for (const char * k : {"max_tokens", "max_completion_tokens", "max_output_tokens", "n_predict"}) {
            const int64_t v = body.is_object() && body.contains(k) && body.at(k).is_number_integer() ? body.at(k).get<int64_t>() : 0;
            if (v > 0) {
                r.max_answer = v;
                break;
            }
        }
        r.short_answer = r.max_answer > 0 && r.max_answer <= cfg.short_tokens;
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
        int move_from  = -1;
        int probe_from = -1;
        int64_t best   = 0;
        std::string move_why;
        {
            std::lock_guard<std::mutex> lk(mu);
            const int64_t now = ggml_time_ms();
            remember(key, r, n, now);
            trim(now);
            const double  waited = (now - r.t_arrive) / 1000.0;
            best                 = r.cached_max();
            const int64_t got    = pool_at(r.cached, n);
            const std::string w  = (waited >= 0.5 ? string_format(", waited %.1f s", waited) : std::string()) +
                                   (r.short_answer ? string_format(", short answer (max %lld tok)", (long long) r.max_answer) : std::string());
            if (r.conv_new) {
                event("new", r.conv, n, pool_at(r.shared, n) > 0
                    ? string_format("~%lld of ~%lld tok shared with others cached here%s", (long long) pool_at(r.shared, n),
                                    (long long) r.prompt_tokens, w.c_str())
                    : string_format("~%lld tok%s", (long long) r.prompt_tokens, w.c_str()));
            } else if (best == 0) {
                event("cold", r.conv, n, string_format("no node holds it, ~%lld tok%s", (long long) r.prompt_tokens, w.c_str()));
            } else if (got >= best) {
                event("keep", r.conv, n, string_format("%d msgs, ~%lld of ~%lld tok cached%s", r.depth, (long long) got,
                    (long long) r.prompt_tokens, w.c_str()));
            } else {
                int from = 0;
                for (int i = 0; i < (int) r.cached.size(); ++i) {
                    from = r.cached[i] > r.cached[from] ? i : from;
                }
                const std::string state = usable(nodes[from])
                    ? (free_slots(nodes[from]) == 0 ? std::string("busy")
                       : !room_fits(from, r) ? string_format("no KV room, %d in work", streams(nodes[from]))
                       : string_format("crowded, %d in work", streams(nodes[from])))
                    : (nodes[from].draining ? "draining" : nodes[from].status);
                move_why = string_format("~%lld tok cached on %s (%s), ~%lld here, ~%lld tok to compute%s",
                    (long long) best, nodes[from].name.c_str(), state.c_str(),
                    (long long) got, (long long) (r.prompt_tokens - got), w.c_str());
                // where it goes, by name (07.11): "migrate" - with its state (the state goes whenever it brings
                // pool-move-tokens or more and can move, migrate()); "switch" - onto an older copy on that node, which
                // computes the difference; "recompute" - that node computes the prompt, the reason after "no state"
                if (*kind && best - got >= cfg.move_tokens && can_migrate(from, n)) {
                    move_from = from;
                    running[r.id].stage = "moving";
                } else if (*kind && got > 0 && can_migrate(from, n)) {
                    // the node is held to have an older copy, the difference below pool-move-tokens: it says what it has,
                    // a copy it let go would have the whole prompt computed there
                    probe_from = from;
                    running[r.id].stage = "moving";
                } else {
                    const std::string no_state = !*kind ? "; no state: the request is not a chat or a completion"
                                               : !can_migrate(from, n) ? "; no state: none moves between these nodes "
                                                 "(pool-migrate, another configuration, no prompt cache)" : "";
                    event(got > 0 && best - got < cfg.move_tokens ? "switch" : "recompute", r.conv, n, move_why + no_state);
                }
            }
            if (!r.conv.empty()) {
                auto & c = convs[r.conv];
                if (c.node >= 0 && c.node != n) {
                    c.t_moved = pool_wall_ms();
                }
                c.node = n;
                c.depth = (int) key.chain.size() - 1;
                c.est_tokens = got;
                c.prompt_tokens = r.prompt_tokens;
                c.t_last = pool_wall_ms();
                c.n++;
                c.held.resize(n_nodes.load(), 0);
                c.held[n] = r.prompt_tokens;
                if (*kind && r.prompt_tokens >= cfg.move_tokens) {
                    c.kind = kind;
                    c.body = body;
                    c.key  = key;
                } else {
                    c.kind.clear();
                    c.body = json();
                }
            }
        }

        // a move onto an older copy: the copy as the node has it. the node counts its tokens, the pool characters (its
        // thresholds too), so the copy goes into the units of the pool by its share of the prompt. the state comes along
        // when the copy is short of what the holder has by pool-move-tokens or more
        int64_t to_held = -1;
        if (probe_from >= 0) {
            int64_t n_tok = 0;
            const int64_t held = probe_hold(n, kind, body, n_tok);
            std::lock_guard<std::mutex> lk(mu);
            const int64_t held_est = held >= 0 && n_tok > 0 && r.prompt_tokens > 0
                ? (int64_t) ((double) held * r.prompt_tokens / n_tok) : -1;
            if (held_est >= 0 && best - held_est >= cfg.move_tokens) {
                move_from = probe_from;
                to_held   = held;
                move_why += string_format("; %lld of %lld tok there in fact", (long long) held, (long long) n_tok);
            } else {
                auto it = running.find(r.id);
                if (it != running.end()) {
                    it->second.stage.clear();
                }
                event("switch", r.conv, n, move_why + (held >= 0
                    ? string_format("; %lld of %lld tok there in fact", (long long) held, (long long) n_tok)
                    : std::string("; the node did not say what it holds")));
            }
        }

        bool moved = false;
        if (move_from >= 0) {
            std::string why;
            std::string detail;
            moved = migrate(r, move_from, n, kind, body, req.should_stop, why, detail, to_held);
            std::lock_guard<std::mutex> lk(mu);
            auto it = running.find(r.id);
            if (it != running.end()) {
                it->second.stage.clear();
            }
            if (moved) {
                event("migrate", r.conv, n, detail);
            } else {
                event("recompute", r.conv, n, move_why + "; no state: " + why);
            }
        }
        if (req.should_stop()) {
            release(r.id, 499);
            return pool_error(499, "the client closed the request");
        }

        try {
            auto proxy = proxy_to(n, req, method, body);
            server_http_proxy * p = proxy.get();
            const uint64_t req_id = r.id;
            // the end of the answer, for its timings: the prefill and decode speeds, and the share of what the node held
            // of the prompt from other conversations that it reused (not when the state of the conversation came along)
            const bool   sample     = !moved && pool_at(r.shared, n) >= cfg.move_tokens && r.prompt_tokens > 0;
            const double shared     = sample ? (double) pool_at(r.shared, n) / r.prompt_tokens : 0.0;
            const double shared_max = sample ? (double) pool_at(r.shared_max, n) / r.prompt_tokens : 0.0;
            auto tail = std::make_shared<std::string>();
            auto next = proxy->next;
            proxy->next = [next, tail](std::string & out) -> bool {
                const bool more = next(out);
                tail->append(out);
                if (tail->size() > POOL_TAIL) {
                    tail->erase(0, tail->size() - POOL_TAIL);
                }
                return more;
            };
            proxy->cleanup = [this, req_id, p, n, tail, shared, shared_max]() {
                release(req_id, p->status, note_timings(n, *tail, shared, shared_max, req_id));
            };
            return std::unique_ptr<server_http_res>(std::move(proxy));
        } catch (...) {
            release(r.id, 500);
            throw;
        }
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
                    st.gen_tps = 0.5*g + 0.5*p.gen_tps;

                    // the prompt: the tokens of the last move of the counter over the time since the move before it,
                    // held until the next move. a poll between two moves would read 0, the poll of a move a whole batch
                    // in one interval. past the time the last move took, the speed falls as that move over the time
                    // since (a stalled prompt goes to 0); once the slot generates, its prompt is read
                    st.t_pp    = p.t_pp;
                    st.n_pp    = p.n_pp;
                    st.dn_pp   = p.dn_pp;
                    st.pp_rate = p.pp_rate;
                    if (st.n_processed != p.n_pp && p.t_pp > 0) {
                        st.dn_pp   = std::max<int64_t>(0, st.n_processed - p.n_pp);
                        st.pp_rate = now > p.t_pp ? st.dn_pp*1000.0/(now - p.t_pp) : p.pp_rate;
                        st.t_pp    = now;
                        st.n_pp    = st.n_processed;
                    }
                    const double since = (now - st.t_pp) / 1000.0;
                    st.prompt_tps = st.n_decoded > 0 ? 0.0 :
                                    since > 0.0 && st.pp_rate > 0.0 ? std::min(st.pp_rate, st.dn_pp/since) : st.pp_rate;
                }
                break;
            }
            // a new task, or the first poll of one: its prompt counter starts here
            if (st.t_pp == 0) {
                st.t_pp = now;
                st.n_pp = st.n_processed;
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

    // the state of the model on a router: up when its instance is loaded; failed when its load failed, n_running the
    // other models the router runs
    static std::string model_state(httplib::Client & cli, const httplib::Headers & hdrs, const std::string & name, std::string & error,
                                   bool & failed, int & n_running) {
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
        std::string state = "down";
        error = "no model " + name + " on the node";
        failed = false;
        n_running = 0;
        for (size_t k = 0; list && k < list->size(); ++k) {
            const json & m = (*list)[k];
            if (!m.is_object()) {
                continue;
            }
            const json st = m.contains("status") && m.at("status").is_object() ? m.at("status") : json::object();
            const std::string v = json_value(st, "value", std::string());
            const bool running = v == "loaded" || v == "sleeping" || v == "loading" || v == "downloading";
            if (!pool_model_named(m, name)) {
                n_running += running ? 1 : 0;
                continue;
            }
            error.clear();
            if (v == "loaded" || v == "sleeping") {
                state = "up";
            } else if (v == "loading" || v == "downloading") {
                state = "loading";
            } else if (json_value(st, "failed", false)) {
                failed = true;
                error  = string_format("the model %s failed to load, exit code %d: see the log of that router", name.c_str(),
                                       json_value(st, "exit_code", 0));
                state  = "unloaded";
            } else {
                error = "model " + name + " is " + (v.empty() ? "unknown" : v) + ", not loaded on that router";
                state = "unloaded";
            }
        }
        return state;
    }

    // a node of another model than the most of the others takes no request (same_model). the first node decides a tie
    void update_mismatch() {
        std::map<std::string, int> count;
        for (int i = 0; i < n_nodes.load(); ++i) {
            if (!nodes[i].model_info.empty()) {
                count[nodes[i].model_info.dump()]++;
            }
        }
        std::string ref;
        int n_ref = 0;
        for (int i = 0; i < n_nodes.load(); ++i) {
            if (!nodes[i].model_info.empty() && count[nodes[i].model_info.dump()] > n_ref) {
                ref   = nodes[i].model_info.dump();
                n_ref = count[ref];
            }
        }
        for (int i = 0; i < n_nodes.load(); ++i) {
            pool_node & n = nodes[i];
            const bool mismatch = cfg.same_model && !n.model_info.empty() && n.model_info.dump() != ref;
            if (mismatch != n.mismatch) {
                event(mismatch ? "mismatch" : "match", "", i, mismatch
                    ? "another model than the other nodes (" + n.model_info.dump() + "): it takes no request, pool-same-model = 0 lets it"
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
        json        router;
        bool        failed     = false;
        int         n_running  = 0;
        int         max_models = -1; // -1 = not read this time
    };

    // one poll of a node over one link
    poll_result poll_link(const pool_node & node, const pool_link & l, const httplib::Headers & hdrs, bool want_props,
                          bool want_router) const {
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
            if (res->status != 200) {
                pr.error = "health: HTTP " + std::to_string(res->status);
                return pr;
            }
            if (want_router) {
                auto rr = cli.Get("/pool/router?model=" + node.model, hdrs);
                if (rr && rr->status == 200) {
                    pr.router = json::parse_no_throw(rr->body);
                }
            }
            pr.status = model_state(cli, hdrs, node.model, pr.error, pr.failed, pr.n_running);
            if (want_props) {
                // the props of the router itself: how many models it runs at once
                auto rp = cli.Get("/props", hdrs);
                const json j = rp && rp->status == 200 ? json::parse_no_throw(rp->body) : json();
                if (j.is_object() && j.contains("max_instances")) {
                    pr.max_models = json_value(j, "max_instances", 0);
                }
            }
            if (pr.status == "up") {
                // the router passes these to the instance of the model; autoload=0: a poll never loads it
                const std::string q = "?model=" + node.model + "&autoload=0";
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

    // the speed of a link both ways (bytes/s, 0 = no answer): one chunk of a move read from the router, one written to it
    static void probe_link(const pool_link & l, const httplib::Headers & hdrs, double & rd, double & wr) {
        httplib::Client cli(pool_base(l));
        cli.set_keep_alive(true);
        cli.set_connection_timeout(1, 0);
        // 16 MiB: 0.2 s over the WiFi; a stop of the router waits for a probe that runs
        cli.set_read_timeout(15, 0);
        cli.set_write_timeout(15, 0);
        if (!cli.Get("/health", hdrs)) { // the connection, before the clock starts
            return;
        }
        int64_t t0 = ggml_time_us();
        auto r = cli.Get("/pool/probe?size=" + std::to_string(POOL_CHUNK), hdrs);
        if (r && r->status == 200 && r->body.size() == POOL_CHUNK) {
            rd = POOL_CHUNK / std::max(1e-6, (ggml_time_us() - t0) / 1e6);
        }
        t0 = ggml_time_us();
        auto w = cli.Post("/pool/probe", hdrs, std::string(POOL_CHUNK, '\0'), "application/octet-stream");
        if (w && w->status == 200 && json_value(json::parse_no_throw(w->body), "size", (uint64_t) 0) == POOL_CHUNK) {
            wr = POOL_CHUNK / std::max(1e-6, (ggml_time_us() - t0) / 1e6);
        }
    }

    // the links of a node in the order a poll tries them: the one in use, then the faster measured ones (the ones not
    // measured last); every probe_s seconds the faster ones first, so that a node goes back to a faster link that
    // answers again. under mu
    std::vector<int> link_order(const pool_node & node, bool retry) const {
        std::vector<int> rest;
        for (int k = 0; k < (int) node.links.size(); ++k) {
            if (k != node.link) {
                rest.push_back(k);
            }
        }
        std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) {
            return pool_link_rank(node.links[a]) > pool_link_rank(node.links[b]);
        });
        std::vector<int> order;
        const double cur = pool_link_rank(node.links[node.link]);
        if (retry) {
            for (int k : rest) {
                if (pool_link_rank(node.links[k]) > cur) {
                    order.push_back(k);
                }
            }
        }
        order.push_back(node.link);
        for (int k : rest) {
            if (std::find(order.begin(), order.end(), k) == order.end()) {
                order.push_back(k);
            }
        }
        return order;
    }

    void poll_node(int i) {
        pool_node & node = nodes[i];
        const httplib::Headers hdrs = auth_headers();
        int64_t t_router = 0;

        while (!stopping.load()) {
            const int64_t t0 = ggml_time_ms();
            std::vector<int> order;
            std::vector<pool_link> links;
            int  active;
            bool want_props;
            bool want_router;
            bool retry;
            {
                std::lock_guard<std::mutex> lk(mu);
                want_props  = node.status != "up" || t0 - node.t_props > 30000;
                want_router = !node.self && t0 - t_router >= 5000;
                retry = t0 - node.t_probe >= (int64_t) cfg.probe_s * 1000;
                if (retry) {
                    node.t_probe = t0;
                }
                order  = link_order(node, retry);
                links  = node.links;
                active = node.link;
            }
            poll_result pr;
            int used = -1;
            std::vector<std::pair<int, std::string>> failed;
            for (int k : order) {
                pr = poll_link(node, links[k], hdrs, want_props || k != active, want_router);
                if (pr.reached) {
                    used = k;
                    break;
                }
                failed.push_back({k, pr.error});
            }
            if (used >= 0 && want_router) {
                t_router = t0;
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
                        const std::string why = was.error.empty() ? "a faster link" : pool_base(was) + ": " + was.error;
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
                // the asked load is on its way or over, or the router never took it up
                if (status != "unloaded" || (used >= 0 && pr.failed) || now - node.t_load > 120000) {
                    node.load_asked = false;
                }
                node.error  = error;
                if (used >= 0) {
                    node.failed    = pr.failed;
                    node.n_running = pr.n_running;
                    if (pr.max_models >= 0) {
                        node.max_models = pr.max_models;
                    }
                }
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
                    const json room = pr.info.contains("room") && pr.info.at("room").is_object() ? pr.info.at("room") : json::object();
                    node.room_held     = json_value(room, "held", (int64_t) -1);
                    node.room_slots    = json_value(room, "slots", (int64_t) 0);
                    node.room_ctx      = json_value(room, "n_ctx", (int64_t) 0);
                    node.room_margin   = json_value(room, "margin", (int64_t) 0);
                    node.room_answer   = json_value(room, "answer", (int64_t) 0);
                    node.room_deferred = json_value(room, "deferred", (int64_t) 0);
                    node.stats = pr.info.contains("stats") && pr.info.at("stats").is_object() ? pr.info.at("stats") : json::object();
                    node.gpus  = pr.info.contains("gpus") && pr.info.at("gpus").is_array() ? pr.info.at("gpus") : json::array();
                    update_mismatch();
                }
                if (!pr.router.is_discarded() && pr.router.is_object() && pr.router.contains("peers")) {
                    node.peers = pr.router.at("peers");
                }
                int reported = json_value(node.props, "total_slots", 0);
                if (reported <= 0) {
                    reported = (int) node.slot_state.size();
                }
                node.slots    = reported;
                node.external = std::max(0, node.busy - node.inflight);
                cv.notify_all();
            }

            // a load the requests for the model asked for (wake): the router there loads it as on a request of its own
            bool load;
            {
                std::lock_guard<std::mutex> lk(mu);
                load = node.want_load && used >= 0;
                node.want_load = false;
                if (load) {
                    node.t_load = ggml_time_ms();
                    // the router answers when the model it unloads for this one is gone, which can take long: until
                    // then the node is not unloaded for wake, or it would ask the next node too and both would unload
                    node.load_asked = true;
                }
            }
            if (load) {
                httplib::Client cli(pool_base(links[used]));
                cli.set_connection_timeout(1, 0);
                cli.set_read_timeout(10, 0);
                const json body = {{"model", node.model}, {"evict_idle_s", cfg.evict_idle_s}};
                auto rl = cli.Post("/models/load", hdrs, body.dump(), "application/json");
                std::lock_guard<std::mutex> lk(mu);
                const bool ok = rl && rl->status == 200;
                // a refusal ends the wait; no answer in time (the router still waits for another model to unload) does not
                node.load_asked = !rl || rl->status == 200;
                event("load", "", i, ok ? "asked to load " + node.model : "cannot load " + node.model + ": " + pool_error_text(rl));
                SRV_INF("pool: %s %s on %s%s%s\n", ok ? "loading" : "cannot load", node.model.c_str(), node.name.c_str(),
                        ok ? "" : ": ", ok ? "" : pool_error_text(rl).c_str());
            }

            // a link is measured when it first answers (one that did not answer, every probe_s seconds), the moves
            // measure it from then on; a node goes to a link that measured faster than the one in use (a quarter more)
            if (used >= 0) {
                for (int k = 0; k < (int) links.size() && !stopping.load(); ++k) {
                    bool need;
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        pool_link & l = node.links[k];
                        need = !l.probed && (k == used || l.error.empty() || retry);
                        l.probed = l.probed || need;
                    }
                    if (!need) {
                        continue;
                    }
                    double rd = 0.0;
                    double wr = 0.0;
                    probe_link(links[k], hdrs, rd, wr);
                    std::lock_guard<std::mutex> lk(mu);
                    pool_link & l = node.links[k];
                    if (rd <= 0 && wr <= 0) {
                        l.probed = false;
                        l.error  = l.error.empty() ? "no answer to the probe" : l.error;
                        continue;
                    }
                    l.error.clear();
                    l.bps_read  = l.bps_read  > 0 ? l.bps_read  : rd;
                    l.bps_write = l.bps_write > 0 ? l.bps_write : wr;
                    SRV_INF("pool: link %s of node %s: %.0f MB/s from it, %.0f MB/s to it\n", pool_base(l).c_str(),
                            node.name.c_str(), rd / 1e6, wr / 1e6);
                    if (k != node.link && pool_link_rank(l) > 1.25 * pool_link_rank(node.links[node.link])) {
                        event("link", "", i, string_format("%s (a faster link: %.0f MB/s)", pool_base(l).c_str(), pool_link_rank(l) / 1e6));
                        SRV_INF("pool: node %s now on %s (a faster link)\n", node.name.c_str(), pool_base(l).c_str());
                        node.link = k;
                    }
                }
            }

            std::unique_lock<std::mutex> lk(mu_stop);
            cv_stop.wait_for(lk, std::chrono::milliseconds(std::max<int64_t>(50, cfg.poll_ms - (ggml_time_ms() - t0))),
                [&] { return stopping.load(); });
        }
    }

    void start() {
        if (!started.exchange(true)) {
            start_poller(0);
            std::lock_guard<std::mutex> lk(mu_threads);
            if (!stopping.load()) {
                pollers.emplace_back([this]() { evacuate(); });
            }
        }
    }

    // a conversation of a drained node whose state can go to another node now, the one used last first: an active one
    // (a request in the last 30 minutes), its last request known, none of its requests runs or waits (a running one
    // leaves when it ends, a waiting one moves at its placement), no try failed in the last 10 minutes; the target is the
    // node with the fewest active conversations per slot by its speed that its state can move to. false when there is
    // none. under mu
    bool evac_pick(std::string & conv, int & from, int & to) const {
        if (!cfg.drain_move) {
            return false;
        }
        const int64_t now = pool_wall_ms();
        std::vector<std::pair<int64_t, const std::string *>> order;
        for (const auto & e : convs) {
            order.push_back({e.second.t_last, &e.first});
        }
        std::sort(order.begin(), order.end(), [](const auto & a, const auto & b) { return a.first > b.first; });
        for (const auto & o : order) {
            const auto & e = *convs.find(*o.second);
            const conv_t & c = e.second;
            if (c.node < 0 || c.node >= n_nodes.load() || !nodes[c.node].draining || nodes[c.node].status != "up" ||
                    c.kind.empty() || now - c.t_last > 1800 * 1000 || now - c.t_evac_fail < 600000) {
                continue;
            }
            const bool busy = std::any_of(running.begin(), running.end(), [&](const auto & q) { return q.second.conv == e.first; }) ||
                              std::any_of(queue.begin(), queue.end(), [&](const pool_req * q) { return q->conv == e.first; });
            if (busy) {
                continue;
            }
            // the target: the fewest active conversations per slot by the speed of the node, those handed over before
            // counted (with every node busy the free slots tell nothing, and a score by them sent every conversation of
            // the drained node to one node)
            int best = -1;
            double best_load = 0.0;
            for (int i = 0; i < n_nodes.load(); ++i) {
                if (i == c.node || !usable(nodes[i]) || !can_migrate(c.node, i)) {
                    continue;
                }
                int held = 0;
                for (const auto & h : convs) {
                    held += h.second.node == i && now - h.second.t_last <= 1800 * 1000 ? 1 : 0;
                }
                const double load = (held + 1) / (std::max(1, nodes[i].slots) * std::max(0.05, weight_of(i)));
                if (best < 0 || load < best_load) {
                    best      = i;
                    best_load = load;
                }
            }
            if (best >= 0) {
                conv = e.first;
                from = c.node;
                to   = best;
                return true;
            }
        }
        return false;
    }

    // a drained node hands its conversations over (07.11, user: "a drain should pass the state to another node, at
    // least of the active conversations"): their states go to other nodes one at a time, before their next requests
    // come, which then stay where the state is (keep) - the node can go away without them being computed again
    void evacuate() {
        while (!stopping.load()) {
            std::string conv;
            int from = -1;
            int to   = -1;
            std::string kind;
            json body;
            pool_req r;
            {
                std::lock_guard<std::mutex> lk(mu);
                if (evac_pick(conv, from, to)) {
                    const conv_t & c = convs[conv];
                    kind     = c.kind;
                    body     = c.body;
                    r.conv   = conv;
                    r.cached.assign(n_nodes.load(), 0);
                }
            }
            if (from < 0) {
                std::unique_lock<std::mutex> lk(mu_stop);
                cv_stop.wait_for(lk, std::chrono::milliseconds(500), [&] { return stopping.load(); });
                continue;
            }
            std::string why;
            std::string detail;
            const bool moved = migrate(r, from, to, kind.c_str(), body, [this]() { return stopping.load(); }, why, detail, 0);
            std::lock_guard<std::mutex> lk(mu);
            auto it = convs.find(conv);
            if (it == convs.end()) {
                continue;
            }
            conv_t & c = it->second;
            if (moved) {
                // the next request finds the state on `to`: the history it continues is held there now, as by the node
                // that served it (count_shared counts a node in `held` in full)
                remember(c.key, r, to, ggml_time_ms());
                c.node    = to;
                c.t_moved = pool_wall_ms();
                c.held.resize(n_nodes.load(), 0);
                c.held[to] = c.prompt_tokens;
                event("migrate", conv, to, "drain of " + nodes[from].name + ": " + detail);
            } else {
                c.t_evac_fail = pool_wall_ms();
                event("drain", conv, from, "the state stays on the drained node: " + why);
            }
        }
    }

    void stop() {
        if (stopping.exchange(true)) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_stop);
        }
        cv_stop.notify_all();
        cv.notify_all();
        std::vector<std::thread> threads;
        {
            std::lock_guard<std::mutex> lk(mu_threads);
            threads.swap(pollers);
        }
        for (auto & t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    //
    // the dashboard, and what the other routers read
    //

    json router_json() {
        std::lock_guard<std::mutex> lk(mu);
        json peers = json::array();
        for (int i = 0; i < n_nodes.load(); ++i) {
            const pool_node & n = nodes[i];
            if (n.self) {
                continue;
            }
            json links = json::array();
            for (int k = 0; k < (int) n.links.size(); ++k) {
                const pool_link & l = n.links[k];
                links.push_back({{"url", pool_base(l)}, {"active", k == n.link}, {"error", l.error},
                                 {"mbps_in", l.bps_read * 8 / 1e6}, {"mbps_out", l.bps_write * 8 / 1e6}});
            }
            peers.push_back({{"id", n.id}, {"name", n.name}, {"status", n.status}, {"links", links}});
        }
        return {{"id", id}, {"name", cfg.name}, {"port", http_port}, {"model", model},
                {"names", std::vector<std::string>(names.begin(), names.end())}, {"peers", peers}};
    }

    json status_json() {
        std::lock_guard<std::mutex> lk(mu);
        const int64_t now  = ggml_time_ms();
        const int64_t wall = pool_wall_ms();

        json jnodes = json::array();
        int up = 0, slots_total = 0, slots_busy = 0, inflight = 0;
        double gen = 0.0, pp = 0.0;
        for (int i = 0; i < n_nodes.load(); ++i) {
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
            for (const auto & [rid, r] : running) {
                if (r.node == i) {
                    jrun.push_back({{"id", rid}, {"conv", r.conv}, {"age_ms", now - r.t_start}, {"path", r.path}, {"stage", r.stage}});
                }
            }
            int queued_for = 0;
            for (const pool_req * q : queue) {
                queued_for += holder(*q) == i ? 1 : 0;
            }
            std::string model_name = json_value(n.props, "model_alias", std::string());
            if (model_name.empty()) {
                model_name = json_value(n.props, "model_path", std::string());
            }
            if (model_name.find_last_of("/\\") != std::string::npos) {
                model_name = model_name.substr(model_name.find_last_of("/\\") + 1);
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
                    {"mbps_read", l.bps_read * 8 / 1e6}, {"mbps_write", l.bps_write * 8 / 1e6}, {"mbps_cfg", 0},
                });
            }
            // on a fallback: a link that answered measured faster than the one in use
            bool fallback = false;
            for (const auto & l : n.links) {
                fallback = fallback || (l.error.empty() && pool_link_rank(l) > pool_link_rank(n.links[n.link]));
            }
            jnodes.push_back({
                {"name", n.name}, {"id", n.id}, {"self", n.self}, {"url", pool_base(n.links[n.link])},
                {"fallback", fallback}, {"links", jlinks},
                {"status", n.status}, {"draining", n.draining}, {"mismatch", n.mismatch}, {"error", n.error},
                {"model", model_name}, {"model_info", n.model_info}, {"migrate", n.migrate}, {"signature", n.signature},
                {"n_ctx", n_ctx}, {"slots", n.slots}, {"inflight", n.inflight}, {"busy", n.busy},
                {"external", n.external}, {"weight", weight_of(i)}, {"requests", n.n_requests},
                {"gen_tps", n.gen_tps}, {"prompt_tps", n.prompt_tps}, {"queued_for", queued_for},
                {"prefill_tps", n.prefill_tps}, {"prefill_cfg", cfg.prefill_tps}, {"decode_tps", n.decode_tps}, {"decode_all", n.decode_all},
                {"share", n.share},
                {"room", n.room_held >= 0 ? json{{"held", n.room_held}, {"n_ctx", n.room_ctx}, {"deferred", n.room_deferred}} : json()},
                {"stats", n.stats}, {"gpus", n.gpus},
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
            for (int i = 0; i < (int) q->cached.size() && best > 0; ++i) {
                if (q->cached[i] == best) {
                    want += (want.empty() ? "" : ", ") + nodes[i].name;
                }
            }
            jqueue.push_back({
                {"id", q->id}, {"conv", q->conv}, {"want", want}, {"age_ms", now - q->t_arrive}, {"est_tokens", best},
                {"prompt_tokens", q->prompt_tokens}, {"depth", q->depth}, {"path", q->path}, {"short", q->short_answer},
            });
        }

        std::vector<std::pair<std::string, conv_t>> cs(convs.begin(), convs.end());
        std::sort(cs.begin(), cs.end(), [](const auto & a, const auto & b) { return a.second.t_last > b.second.t_last; });
        json jconvs = json::array();
        for (size_t i = 0; i < cs.size() && i < 60; ++i) {
            const auto & [cid, c] = cs[i];
            bool active = false;
            for (const auto & [rid, r] : running) {
                active = active || r.conv == cid;
            }
            jconvs.push_back({
                {"conv", cid}, {"node", c.node >= 0 ? nodes[c.node].name : ""}, {"messages", c.depth},
                {"cached_tokens", c.est_tokens}, {"prompt_tokens", c.prompt_tokens}, {"age_ms", wall - c.t_last},
                {"requests", c.n}, {"active", active},
            });
        }

        json jevents = json::array();
        for (auto it = events.rbegin(); it != events.rend() && jevents.size() < 120; ++it) {
            jevents.push_back({{"t", it->t}, {"kind", it->kind}, {"conv", it->conv}, {"node", it->node}, {"detail", it->detail}});
        }

        return {
            {"name", cfg.name}, {"id", id}, {"time", wall},
            {"pool", {
                {"nodes_up", up}, {"nodes", n_nodes.load()}, {"slots", slots_total}, {"slots_busy", slots_busy},
                {"inflight", inflight}, {"queued", (int) queue.size()}, {"gen_tps", gen}, {"prompt_tps", pp},
                {"conversations", (int) convs.size()}, {"prefixes", (int) affinity.size()},
            }},
            {"config", {
                {"model", model}, {"names", std::vector<std::string>(names.begin(), names.end())},
                {"move_tokens", cfg.move_tokens}, {"wait_max", cfg.wait_max_s}, {"prefill_tps", cfg.prefill_tps},
                {"poll_ms", cfg.poll_ms}, {"migrate", cfg.migrate}, {"drain_move", cfg.drain_move}, {"link_mbps", cfg.link_mbps},
                {"same_model", cfg.same_model}, {"port", cfg.port}, {"discover", cfg.discover}, {"peers", cfg.peers},
                {"rebalance_s", cfg.rebalance_s}, {"quiet_wait_s", cfg.quiet_wait_s}, {"short_tokens", cfg.short_tokens},
                {"evict_idle_s", cfg.evict_idle_s},
            }},
            {"state_kib_per_token", state_bpt / 1024},
            {"nodes", jnodes}, {"queue", jqueue}, {"conversations", jconvs}, {"events", jevents},
        };
    }
};

//
// the discovery: one for all the pools of the router. broadcasts come to sock_bcast (the port of the pools), the answers
// to sock_reply (a port of its own, as the broadcasts go out from it); a hello names the models of every pool, and a
// router that answered the challenge joins each pool it shares a model with
//

struct server_pool_fork::discovery_t {
    std::vector<impl *> pools;
    std::string id;              // the id, name, key and port of the first pool: the same router in every pool
    std::string name;
    std::string api_key;
    std::string instance = pool_instance();
    int         http_port = 0;
    int         port      = 12345;
    bool        discover  = false; // any pool with pool-discover
    std::vector<std::string> peers; // pool-peers of every pool
    std::set<std::string>    names; // the names of the models of every pool
    httplib::Headers         hdrs;

    socket_t sock_bcast = INVALID_SOCKET;
    socket_t sock_reply = INVALID_SOCKET;
    std::map<std::string, int64_t> challenges; // the challenges of this router -> when sent, discovery thread only
    int64_t t_conflict = 0;                    // the last warning about a router with the same id, discovery thread only

    std::atomic<bool>       stopping{false};
    std::mutex              mu_stop;
    std::condition_variable cv_stop;
    std::thread             th;

    explicit discovery_t(const std::vector<std::unique_ptr<impl>> & all) {
        for (const auto & p : all) {
            pools.push_back(p.get());
            discover = discover || p->cfg.discover;
            for (const auto & peer : p->cfg.peers) {
                if (std::find(peers.begin(), peers.end(), peer) == peers.end()) {
                    peers.push_back(peer);
                }
            }
            names.insert(p->names.begin(), p->names.end());
        }
        const impl * first = pools.front();
        id        = first->id;
        name      = first->cfg.name;
        api_key   = first->api_key;
        http_port = first->http_port;
        port      = first->cfg.port;
        hdrs      = first->auth_headers();
    }

    std::string proof_of(const std::string & challenge, const std::string & prover) const {
        if (api_key.empty()) {
            return "";
        }
        const std::string s = api_key + "|" + challenge + "|" + prover;
        return hash_sha256_hex(s.data(), s.size());
    }

    json hello_of(const char * type) const {
        return {{"llama_pool", 1}, {"t", type}, {"id", id}, {"i", instance}, {"name", name}, {"port", http_port},
                {"names", std::vector<std::string>(names.begin(), names.end())}};
    }

    std::string challenge() {
        const std::string c = pool_random_hex();
        const int64_t now = ggml_time_ms();
        for (auto it = challenges.begin(); it != challenges.end(); ) {
            it = now - it->second > 60000 ? challenges.erase(it) : std::next(it);
        }
        challenges[c] = now;
        return c;
    }

    // a router of the pools reached at host:port joins every pool it shares a model with
    void found(const std::string & peer, const std::string & peer_name, const std::string & host, int peer_port,
               const std::vector<std::string> & peer_names, const char * how) {
        for (impl * p : pools) {
            bool shared = false;
            for (const auto & nm : peer_names) {
                shared = shared || p->names.count(nm) > 0;
            }
            if (shared) {
                p->found(peer, peer_name, host, peer_port, peer_names, how);
            }
        }
    }

    // a datagram from addr:port. hello (a broadcast): answer with a proof of the key for its challenge and a challenge
    // of our own; proof: the other router answered our challenge, the link counts, and it gets a proof back if it asks
    void on_datagram(const std::string & data, uint32_t addr, int from_port) {
        const json m = json::parse_no_throw(data);
        if (m.is_discarded() || !m.is_object() || json_value(m, "llama_pool", 0) != 1) {
            return;
        }
        const std::string peer = json_value(m, "id", std::string());
        const int peer_port    = json_value(m, "port", 0);
        char host[INET_ADDRSTRLEN] = {};
        in_addr ia = {};
        ia.s_addr = addr;
        inet_ntop(AF_INET, &ia, host, sizeof(host));
        if (peer == id && json_value(m, "i", instance) != instance) {
            // a router of another machine with the same host name and port: they cannot tell each other apart
            const int64_t now = ggml_time_ms();
            if (now - t_conflict > 60000) {
                t_conflict = now;
                SRV_WRN("pool: the router at %s has the same id %s: give one of them a name (pool-name in models.ini)\n", host, id.c_str());
                for (impl * p : pools) {
                    std::lock_guard<std::mutex> lk(p->mu);
                    p->event("conflict", "", 0, std::string("the router at ") + host + " has the same id " + id + ": pool-name in models.ini on one of them");
                }
            }
            return;
        }
        if (peer.empty() || peer == id || peer_port <= 0 || !m.contains("names") || !m.at("names").is_array()) {
            return;
        }
        std::vector<std::string> peer_names;
        bool shared = false;
        for (const auto & nm : m.at("names")) {
            if (nm.is_string()) {
                peer_names.push_back(nm.get<std::string>());
                shared = shared || names.count(peer_names.back()) > 0;
            }
        }
        if (!shared) {
            return; // other models, other pools
        }
        const std::string type = json_value(m, "t", std::string());
        const std::string c    = json_value(m, "c", std::string());
        if (type == "hello" || (type == "proof" && !c.empty())) {
            json ans = hello_of("proof");
            ans["a"] = c;
            ans["p"] = proof_of(c, id);
            if (type == "hello") {
                ans["c"] = challenge();
            }
            pool_udp_send(sock_reply, addr, from_port, ans.dump());
        }
        if (type != "proof") {
            return;
        }
        const std::string a = json_value(m, "a", std::string());
        const std::string p = json_value(m, "p", std::string());
        if (challenges.count(a) == 0 || p != proof_of(a, peer)) {
            SRV_DBG("pool: %s did not answer the challenge (another key)\n", peer.c_str());
            return;
        }
        found(peer, json_value(m, "name", std::string()), host, peer_port, peer_names, "broadcast");
    }

    // a router given in pool-peers: GET /pool/router with the key, which the config trusts it with
    void try_peer(const std::string & peer) {
        std::string host;
        int peer_port = 0;
        try {
            pool_parse_host_port(peer, host, peer_port);
        } catch (const std::exception & e) {
            SRV_WRN("pool: %s\n", e.what());
            return;
        }
        httplib::Client cli(host, peer_port);
        cli.set_connection_timeout(1, 0);
        cli.set_read_timeout(3, 0);
        auto r = cli.Get("/pool/router", hdrs);
        const json j = r && r->status == 200 ? json::parse_no_throw(r->body) : json();
        const char * key = j.is_object() && j.contains("all_names") ? "all_names" : "names";
        if (!j.is_object() || !j.contains(key) || !j.at(key).is_array()) {
            return;
        }
        std::vector<std::string> peer_names;
        for (const auto & nm : j.at(key)) {
            if (nm.is_string()) {
                peer_names.push_back(nm.get<std::string>());
            }
        }
        const std::string peer_id = json_value(j, "id", std::string());
        if (!peer_id.empty() && peer_id != id) {
            found(peer_id, json_value(j, "name", std::string()), host, peer_port, peer_names, "peers");
        }
    }

    void loop() {
        int64_t t_hello = 0;
        int64_t t_peers = 0;
        const int64_t t_start = ggml_time_ms();
        std::vector<char> buf(4096);
        while (!stopping.load()) {
            const int64_t now = ggml_time_ms();
            // a hello every 2 s for the first 30 s, then every 15 s: a router that starts is found within a round trip
            const int64_t every = now - t_start < 30000 ? 2000 : 15000;
            if (discover && sock_reply != INVALID_SOCKET && now - t_hello >= every) {
                t_hello = now;
                json h = hello_of("hello");
                h["c"] = challenge();
                const std::string msg = h.dump();
                for (uint32_t a : pool_broadcast_addrs()) {
                    pool_udp_send(sock_reply, a, port, msg);
                }
            }
            if (!peers.empty() && now - t_peers >= every) {
                t_peers = now;
                for (const auto & peer : peers) {
                    try_peer(peer);
                }
            }
            if (sock_reply == INVALID_SOCKET && sock_bcast == INVALID_SOCKET) {
                std::unique_lock<std::mutex> lk(mu_stop);
                cv_stop.wait_for(lk, std::chrono::milliseconds(250), [&] { return stopping.load(); });
                continue;
            }
            fd_set rd;
            FD_ZERO(&rd);
            socket_t hi = 0;
            for (socket_t s : {sock_bcast, sock_reply}) {
                if (s != INVALID_SOCKET) {
                    FD_SET(s, &rd);
                    hi = std::max(hi, s);
                }
            }
            timeval tv = {0, 250000};
            if (select((int) hi + 1, &rd, nullptr, nullptr, &tv) <= 0) {
                continue;
            }
            for (socket_t s : {sock_bcast, sock_reply}) {
                if (s == INVALID_SOCKET || !FD_ISSET(s, &rd)) {
                    continue;
                }
                sockaddr_in from = {};
                socklen_t len = sizeof(from);
                const int n = (int) recvfrom(s, buf.data(), (int) buf.size(), 0, (sockaddr *) &from, &len);
                if (n > 0) {
                    on_datagram(std::string(buf.data(), n), from.sin_addr.s_addr, ntohs(from.sin_port));
                }
            }
        }
    }

    void start() {
        sock_reply = pool_udp_socket(0, false);
        if (discover) {
            sock_bcast = pool_udp_socket(port, true);
            if (sock_bcast == INVALID_SOCKET) {
                // the answers to its own broadcasts still come: it finds the others, they find it only by their answers
                SRV_WRN("pool: cannot listen on UDP port %d, the other routers find this one only when it asks\n", port);
            }
        }
        if (sock_reply == INVALID_SOCKET) {
            SRV_WRN("%s", "pool: no UDP socket, the routers of the pool are found only from pool-peers\n");
        }
        th = std::thread([this]() { loop(); });
    }

    void stop() {
        if (stopping.exchange(true)) {
            return;
        }
        {
            std::lock_guard<std::mutex> lk(mu_stop);
        }
        cv_stop.notify_all();
        if (th.joinable()) {
            th.join();
        }
        pool_sock_close(sock_bcast);
        pool_sock_close(sock_reply);
    }
};

//
// server_pool_fork
//

std::unique_ptr<server_pool_fork> server_pool_fork::create(const common_params & params, server_models_routes & models_routes) {
    std::vector<std::string> pooled;
    for (const auto & meta : models_routes.models.get_all_meta()) {
        std::string val;
        if (meta.preset.get_option(COMMON_ARG_PRESET_POOL_FORK, val) && common_arg_utils::is_truthy(val)) {
            pooled.push_back(meta.name);
        }
    }
    if (pooled.empty()) {
        return nullptr;
    }
    return std::make_unique<server_pool_fork>(params, models_routes, pooled);
}

server_pool_fork::server_pool_fork(const common_params & params, server_models_routes & models_routes, const std::vector<std::string> & models) {
    for (const auto & model : models) {
        auto p = std::make_unique<impl>(params, models_routes);
        p->init(model);
        std::string list;
        for (const auto & nm : p->names) {
            list += (list.empty() ? "" : ", ") + nm;
        }
        SRV_INF("pool: %s shares %s (%s) as %s%s\n", p->id.c_str(), model.c_str(), list.c_str(), p->cfg.name.c_str(),
                p->api_key.empty() ? "" : ", its links need the key");
        pools.push_back(std::move(p));
    }
    // one router in every pool: the id and the name of the first (pool-name may differ between the models)
    for (size_t k = 1; k < pools.size(); ++k) {
        impl & p = *pools[k];
        if (p.id != pools[0]->id) {
            SRV_WRN("pool: %s takes the name of the pool of %s (%s), the router has one id in every pool\n", p.model.c_str(),
                    pools[0]->model.c_str(), pools[0]->id.c_str());
        }
        p.id = pools[0]->id;
        p.cfg.name = pools[0]->cfg.name;
        p.nodes[0].id   = p.id;
        p.nodes[0].name = p.cfg.name;
    }
    disc = std::make_unique<discovery_t>(pools);
}

server_pool_fork::~server_pool_fork() {
    stop();
}

void server_pool_fork::start() {
    for (auto & p : pools) {
        p->start();
    }
    disc->start();
}

void server_pool_fork::stop() {
    if (disc) {
        disc->stop(); // it adds nodes to the pools
    }
    for (auto & p : pools) {
        p->stop();
    }
}

void server_pool_fork::attach(server_http_context & ctx_http, server_routes & routes) {
    // the pool of a model by any of its names; nullptr = none
    auto pool_of = [this](const std::string & name) -> impl * {
        for (auto & p : pools) {
            if (p->names.count(name) > 0) {
                return p.get();
            }
        }
        return nullptr;
    };
    // the pool a request of the dashboard or of another router names in ?model=, the first one without it
    auto pool_by_param = [this, pool_of](const server_http_req & req) -> impl * {
        const std::string name = req.get_param("model");
        return name.empty() ? pools.front().get() : pool_of(name);
    };

    // a request for a model of a pool goes through that pool, unless another router of the pool sent it here; any other
    // request stays with this router
    auto wrap = [pool_of](server_http_context::handler_t own) -> server_http_context::handler_t {
        return [pool_of, own](const server_http_req & req) -> server_http_res_ptr {
            if (pool_header(req.headers, "x-llama-pool").empty()) {
                const json body = json::parse_no_throw(req.body);
                impl * p = body.is_object() ? pool_of(json_value(body, "model", std::string())) : nullptr;
                if (p != nullptr) {
                    return p->forward_slot(req, "POST", body);
                }
            }
            return own(req);
        };
    };
    routes.post_completions            = wrap(routes.post_completions);
    routes.post_completions_oai        = wrap(routes.post_completions_oai);
    routes.post_chat_completions       = wrap(routes.post_chat_completions);
    routes.post_responses_oai          = wrap(routes.post_responses_oai);
    routes.post_anthropic_messages     = wrap(routes.post_anthropic_messages);
    routes.post_infill                 = wrap(routes.post_infill);
    routes.post_embeddings             = wrap(routes.post_embeddings);
    routes.post_embeddings_oai         = wrap(routes.post_embeddings_oai);
    routes.post_rerank                 = wrap(routes.post_rerank);
    routes.post_transcriptions_oai     = wrap(routes.post_transcriptions_oai);
    routes.post_tts                    = wrap(routes.post_tts);
    routes.post_systemone              = wrap(routes.post_systemone);

    ctx_http.get("/pool", [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->content_type = "text/html; charset=utf-8";
        for (const char * part : POOL_UI_PARTS) {
            res->data += part;
        }
        return res;
    });
    // ?model=: the pool of that model, the first one without it; "pools" lists the models of all of them
    ctx_http.get("/pool/status", [this, pool_by_param](const server_http_req & req) {
        impl * p = pool_by_param(req);
        if (p == nullptr) {
            return pool_error(400, "no pool of the model " + req.get_param("model"));
        }
        auto res = std::make_unique<server_http_res>();
        try {
            json st = p->status_json();
            st["pools"] = json::array();
            for (const auto & q : pools) {
                st["pools"].push_back(q->model);
            }
            res->data = safe_json_to_str(st);
        } catch (const std::exception & e) {
            return pool_error(500, e.what());
        }
        return res;
    });
    // {"node": "<name>", "drain": true}: this router sends no new request to the node (in every pool), the running ones
    // finish
    ctx_http.post("/pool/drain", [this](const server_http_req & req) {
        const json body = json::parse_no_throw(req.body);
        if (body.is_discarded() || !body.is_object() || !body.contains("node")) {
            return pool_error(400, "expected {\"node\": \"<name>\", \"drain\": true|false}");
        }
        const std::string name = json_value(body, "node", std::string());
        const bool drain = json_value(body, "drain", true);
        bool found = false;
        for (auto & p : pools) {
            std::lock_guard<std::mutex> lk(p->mu);
            for (int i = 0; i < p->n_nodes.load(); ++i) {
                if (p->nodes[i].name == name) {
                    p->nodes[i].draining = drain;
                    p->event(drain ? "drain" : "undrain", "", i, "");
                    p->cv.notify_all();
                    found = true;
                }
            }
        }
        if (!found) {
            return pool_error(400, "no node " + name);
        }
        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str({{"node", name}, {"draining", drain}});
        return res;
    });
    // what the other routers read (?model=): this router, the model under every name, and its links to the other nodes
    ctx_http.get("/pool/router", [this, pool_by_param](const server_http_req & req) {
        impl * p = pool_by_param(req);
        if (p == nullptr) {
            return pool_error(400, "no pool of the model " + req.get_param("model"));
        }
        json j = p->router_json();
        j["all_names"] = std::vector<std::string>(disc->names.begin(), disc->names.end()); // the models of every pool
        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str(j);
        return res;
    });
    // the pull goes to the pool of the model it names (its name on this router)
    ctx_http.post("/pool/pull", [pool_of](const server_http_req & req) {
        const json body = json::parse_no_throw(req.body);
        impl * p = body.is_object() ? pool_of(json_value(body, "model", std::string())) : nullptr;
        if (p == nullptr) {
            return pool_error(400, "no pool of the model in the request");
        }
        return p->pull(req);
    });
    // ?size=<bytes>: the probe of a link by another router of the pool, that many bytes from here (one chunk of a move)
    ctx_http.get("/pool/probe", [](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        uint64_t size = 0;
        try {
            size = std::min<uint64_t>(std::stoull(req.get_param("size", "16777216")), 64ull << 20);
        } catch (const std::exception &) {
            return pool_error(400, "bad size");
        }
        res->data.assign(size, '\0');
        res->content_type = "application/octet-stream";
        return res;
    });
    // the probe of a link the other way: the bytes in the body go nowhere
    ctx_http.post("/pool/probe", [](const server_http_req & req) {
        auto res = std::make_unique<server_http_res>();
        res->data = safe_json_to_str({{"size", req.body.size()}});
        return res;
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
    return pool_random_hex();
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

bool server_pool_quiet_fork(const server_http_req & req) {
    return !pool_header(req.headers, "x-llama-pool-poll").empty();
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
