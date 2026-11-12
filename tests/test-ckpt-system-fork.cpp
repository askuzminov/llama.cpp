// fork: the checkpoint at the end of a long system prompt (LLAMA_CKPT_SYSTEM_BREAK, tools/server/fork/server-context-fork.h)
// - server_ckpt_system_end_fork: where the system block ends by the message spans of a prompt (a template with a
//   delimiter of the system role, or the block before the first message), -1 below the minimum
// - server_ckpt_evict_front_chain / server_ckpt_evict_over_budget with `keep`: the chain of that checkpoint goes last
// - server_ckpt_next_base_pos: no delta hangs off it
// no model: the checkpoints are lists of sizes, the minimum is set in the environment before the first call

#include "fork/server-context-fork.h"

#include <cstdio>
#include <memory>
#include <cstdlib>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

// the lists hold shared immutable checkpoints (upstream e152332a1)
static std::shared_ptr<const common_prompt_checkpoint> ckpt(int64_t n_tokens, size_t bytes, bool delta = false) {
    auto cp = std::make_shared<common_prompt_checkpoint>();
    cp->n_tokens = n_tokens;
    cp->pos_max  = (llama_pos) n_tokens - 1;
    cp->base_pos = delta ? 0 : -1;
    cp->data_tgt.assign(bytes, 1);
    return cp;
}

static common_chat_msg_spans spans_of(std::initializer_list<std::pair<common_chat_role, size_t>> list) {
    common_chat_msg_spans s;
    for (const auto & [role, pos] : list) {
        s.add(role, pos, 1);
    }
    return s;
}

static std::vector<int64_t> tokens_of(const server_ckpt_list & l) {
    std::vector<int64_t> out;
    for (const auto & cp : l) {
        out.push_back(cp->n_tokens);
    }
    return out;
}

int main() {
#ifdef _WIN32
    _putenv_s("LLAMA_CKPT_SYSTEM_BREAK", "64");
#else
    setenv("LLAMA_CKPT_SYSTEM_BREAK", "64", 1);
#endif
    CHECK(server_ckpt_system_break_fork() == 64);

    // the end of the system block
    CHECK(server_ckpt_system_end_fork(spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, 100}})) == 100);
    CHECK(server_ckpt_system_end_fork(spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}, {COMMON_CHAT_ROLE_USER, 30}})) == -1);
    CHECK(server_ckpt_system_end_fork(spans_of({{COMMON_CHAT_ROLE_SYSTEM, 0}})) == -1);
    CHECK(server_ckpt_system_end_fork(spans_of({{COMMON_CHAT_ROLE_USER, 150}, {COMMON_CHAT_ROLE_ASSISTANT, 200}})) == 150);
    CHECK(server_ckpt_system_end_fork(spans_of({{COMMON_CHAT_ROLE_USER, 1}, {COMMON_CHAT_ROLE_ASSISTANT, 200}})) == -1);
    CHECK(server_ckpt_system_end_fork(common_chat_msg_spans()) == -1);

    // the front chain is the one to keep: the next chain goes
    {
        server_ckpt_list l = {ckpt(150, 10), ckpt(300, 10), ckpt(450, 10)};
        CHECK(server_ckpt_evict_front_chain(l, 150) == 10);
        CHECK(tokens_of(l) == std::vector<int64_t>({150, 450}));
        CHECK(server_ckpt_evict_front_chain(l, 150) == 10);
        CHECK(tokens_of(l) == std::vector<int64_t>({150}));
        CHECK(server_ckpt_evict_front_chain(l, 150) == 10); // nothing else left: it goes too
        CHECK(l.empty());
    }
    // without keep, or with another position, the front one goes as before
    {
        server_ckpt_list l = {ckpt(150, 10), ckpt(300, 10)};
        CHECK(server_ckpt_evict_front_chain(l) == 10);
        CHECK(tokens_of(l) == std::vector<int64_t>({300}));
        server_ckpt_list m = {ckpt(150, 10), ckpt(300, 10)};
        CHECK(server_ckpt_evict_front_chain(m, 151) == 10);
        CHECK(tokens_of(m) == std::vector<int64_t>({300}));
    }
    // the chain after the kept one goes with its deltas
    {
        server_ckpt_list l = {ckpt(150, 10), ckpt(300, 10), ckpt(320, 2, true), ckpt(450, 10)};
        CHECK(server_ckpt_evict_front_chain(l, 150) == 12);
        CHECK(tokens_of(l) == std::vector<int64_t>({150, 450}));
    }
    // over the budget: the kept chain stays and so does the newest one
    {
        server_ckpt_list l = {ckpt(150, 10), ckpt(300, 10), ckpt(450, 10)};
        CHECK(server_ckpt_evict_over_budget(l, 15, 150) == 10);
        CHECK(tokens_of(l) == std::vector<int64_t>({150, 450}));
        server_ckpt_list m = {ckpt(150, 10), ckpt(300, 10), ckpt(450, 10)};
        CHECK(server_ckpt_evict_over_budget(m, 15) == 20); // as before: down to the newest chain
        CHECK(tokens_of(m) == std::vector<int64_t>({450}));
    }
    // no delta hangs off the kept checkpoint (the model is not read on that path)
    {
        server_ckpt_list l = {ckpt(150, 10)};
        CHECK(server_ckpt_next_base_pos(l, nullptr, 150) == -1);
    }

    if (n_fail == 0) {
        fprintf(stderr, "OK\n");
    }
    return n_fail == 0 ? 0 : 1;
}
