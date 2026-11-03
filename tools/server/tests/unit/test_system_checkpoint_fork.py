import os
import pytest
from utils import *

# fork: a checkpoint at the end of a long system prompt (LLAMA_CKPT_SYSTEM_BREAK, tools/server/fork/server-context-fork.h).
# tinygemma3 uses SWA: it rolls a sequence back only to a checkpoint, so another conversation with the same system
# prompt takes that prompt from the state of the slot only when a checkpoint sits at its end. a prompt in one batch
# (n_batch 512) leaves none there by itself. the template of gemma 3 puts the system prompt into the first user turn,
# this one gives it a turn of its own (<start_of_turn> is a special token, so the message delimiters are found)

server: ServerProcess

TEMPLATE = (
    "{{ bos_token }}{%- for message in messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\n' }}{%- endif -%}"
)

SYSTEM = {"role": "system", "content": "You are a careful assistant for a small library. Answer briefly and politely, "
                                       "name the shelf of every book you mention. " * 6}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()
    server.no_mmproj = True
    server.n_slots = 1
    server.n_ctx = 4096
    server.n_batch = 512
    server.jinja = True
    server.chat_template_file = os.path.join(TMP_DIR, "test_system_checkpoint.jinja")
    with open(server.chat_template_file, "w") as f:
        f.write(TEMPLATE)
    yield
    server.stop()


def chat(messages: list) -> dict:
    res = server.make_request("POST", "/chat/completions", data={
        "messages": messages, "max_tokens": 8, "temperature": 0,
    })
    assert res.status_code == 200, res.body
    return res.body


def system_tokens() -> int:
    # the tokens of the prompt through the system message: where the prompt of a conversation with it and a user message
    # stops sharing with another one
    a = server.make_request("POST", "/apply-template", data={"messages": [SYSTEM, {"role": "user", "content": "x"}]})
    b = server.make_request("POST", "/apply-template", data={"messages": [SYSTEM, {"role": "user", "content": "y"}]})
    pa, pb = a.body["prompt"], b.body["prompt"]
    n = 0
    while n < min(len(pa), len(pb)) and pa[n] == pb[n]:
        n += 1
    res = server.make_request("POST", "/tokenize", data={"content": pa[:n], "add_special": True, "parse_special": True})
    return len(res.body["tokens"])


@pytest.mark.parametrize("min_tokens", [64, 0])
def test_system_prompt_from_another_conversation(min_tokens: int):
    # with the break the second conversation takes the system prompt from the first one; without it nothing
    server.env_fork = {"LLAMA_CKPT_SYSTEM_BREAK": str(min_tokens)}
    server.start()
    n_sys = system_tokens()
    assert n_sys > 64
    chat([SYSTEM, {"role": "user", "content": "question one about the books"}])
    t = chat([SYSTEM, {"role": "user", "content": "question two about the opening hours"}])["timings"]
    if min_tokens:
        assert n_sys - 8 <= t["cache_n"] <= n_sys + 8, (t, n_sys)
    else:
        assert t["cache_n"] == 0, (t, n_sys)


def test_system_checkpoint_is_evicted_last():
    # two checkpoints a slot at most and no minimum step between them: the turns of a conversation evict its older
    # checkpoints, but not the one at the end of the system prompt, so that another conversation still takes the prompt
    server.env_fork = {"LLAMA_CKPT_SYSTEM_BREAK": "64"}
    server.n_ctx_checkpoints = 2
    server.checkpoint_min_step = 0
    server.start()
    n_sys = system_tokens()
    msgs = [SYSTEM, {"role": "user", "content": "question one about the books"}]
    for i in range(4):
        msgs.append({"role": "assistant", "content": chat(msgs)["choices"][0]["message"]["content"]})
        msgs.append({"role": "user", "content": f"and what about the maps, part {i}? " * 4})
    t = chat([SYSTEM, {"role": "user", "content": "question two about the opening hours"}])["timings"]
    assert n_sys - 8 <= t["cache_n"] <= n_sys + 8, (t, n_sys)
