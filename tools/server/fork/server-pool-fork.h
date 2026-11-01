#pragma once

// fork: --pool, llama-server as a router over remote instances of one model (server-pool-fork.cpp)

#include "server-common.h" // json
#include "server-http.h"
#include "server-task.h"   // server_pool_plan_fork

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

struct common_params;
struct server_routes;
struct server_models_routes;

struct server_pool_fork {
    // reads the nodes and the options from params.pool_file, throws on a bad file
    explicit server_pool_fork(const common_params & params);
    ~server_pool_fork();

    // the API routes go to the nodes; /pool (the dashboard), /pool/status and /pool/drain are added
    void attach(server_http_context & ctx_http, server_routes & routes);

    // the polling of the nodes: health, slots, device memory
    void start();
    void stop();

    struct impl;
    std::unique_ptr<impl> pimpl;
};

// the node side; server_routes in server-context.cpp has the handlers (*_pool_*_fork)

// GET /pool/node: the devices with their memory, for the dashboard of a pool
json server_pool_node_devices_fork();

// a conversation moves to another node with its cached state, read and written in chunks by the HTTP threads, with
// nothing written to disk: the export sends the file of a state on disk (a hard link pool-export-<id>.bin next to it in
// the --cache-spill-dir) or a copy of a state in RAM, the import parses the chunks into RAM
std::string           server_pool_xfer_id_fork();
// the link of an export: empty when `id` is not one of 1 to 32 lowercase hex digits or the node has no spill dir
std::filesystem::path server_pool_xfer_path_fork(const std::string & spill_dir, const char * kind, const std::string & id);
// the pieces an export sends (server_prompt_cache::pool_export), kept until POST /pool/export/done
void server_pool_xfer_plan_fork(const std::string & id, server_pool_plan_fork plan);
void server_pool_xfer_forget_fork(const std::string & id);
// up to `size` bytes of export `id` from `offset`, fewer at the end
bool server_pool_xfer_read_fork(const std::string & id, uint64_t offset, uint64_t size, std::string & out, std::string & error);
// the chunks of import `id` come in order, offset 0 starts it. max_bytes: the largest state the node takes, 0 = any
bool server_pool_inbox_write_fork(const std::string & id, uint64_t offset, const std::string & data, uint64_t max_bytes,
                                  std::string & error);
// the import `id`, no longer kept here; nullptr when there is none
std::shared_ptr<server_pool_inbox_fork> server_pool_inbox_take_fork(const std::string & id);
// what a pool that stopped midway left: exports and imports untouched for `idle_s`, and the links older than that
void server_pool_xfer_clean_fork(const std::string & spill_dir, int idle_s);

// a router (serve.bat) as a node: POST /pool/import/data goes to the instance of ?model=, the chunks are not JSON
server_http_context::handler_t server_pool_router_post_fork(server_models_routes & routes);
