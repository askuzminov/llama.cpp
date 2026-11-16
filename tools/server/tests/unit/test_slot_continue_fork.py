import os
import pytest
from utils import *

# fork: which request continues what a slot holds (server_slot_continues_fork in tools/server/fork/server-context-fork.h).
# A request that rewrites the tail of the last prompt of a conversation stays in its slot; a new conversation with the
# same system prompt goes to another one. No prompt cache here: a request that leaves its slot starts from nothing, so
# the cached tokens tell where it went. The template gives the system prompt a turn of its own (see
# test_system_checkpoint_fork.py), so the server knows where the system block ends.

server: ServerProcess

TEMPLATE = (
    "{{ bos_token }}{%- for message in messages -%}"
    "{%- set role = 'model' if message['role'] == 'assistant' else message['role'] -%}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<start_of_turn>model\n' }}{%- endif -%}"
)

SYSTEM  = {"role": "system", "content": "You are a careful assistant for a small library. Answer briefly and politely, "
                                        "name the shelf of every book you mention. " * 6}
HISTORY = {"role": "user", "content": "Here is the list of the books that came in this week, with their shelves. " * 100}
ANSWER  = {"role": "assistant", "content": "Noted."}


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinygemma3()
    server.no_mmproj = True
    server.n_slots = 2
    server.n_ctx = 8192
    server.n_batch = 512
    server.cache_ram = 0
    server.debug = True  # --verbose: the answers name their slot
    server.jinja = True
    server.chat_template_file = os.path.join(TMP_DIR, "test_slot_continue.jinja")
    with open(server.chat_template_file, "w") as f:
        f.write(TEMPLATE)
    yield
    server.stop()


def chat(messages: list) -> dict:
    res = server.make_request("POST", "/chat/completions", data={
        "messages": messages, "max_tokens": 8, "temperature": 0,
    })
    assert res.status_code == 200, res.body
    # --verbose: the slot of the request comes with the answer
    return {"id_slot": res.body["__verbose"]["id_slot"], **res.body["timings"]}


def test_rewritten_tail_stays_in_its_slot():
    global server
    server.start()

    first = chat([SYSTEM, HISTORY, ANSWER, {"role": "user", "content": "Which shelf holds the atlas? " * 6}])
    assert first["cache_n"] == 0

    # the client sends its last message again in other words: the request shares all but that message with the slot
    t = chat([SYSTEM, HISTORY, ANSWER, {"role": "user", "content": "Where do I find the old maps? " * 6}])
    assert t["id_slot"] == first["id_slot"]
    assert t["cache_n"] > first["prompt_n"] // 2, t


def test_new_conversation_with_the_same_system_prompt_takes_another_slot():
    global server
    server.start()

    first = chat([SYSTEM, {"role": "user", "content": "Which shelf holds the atlas? " * 6}])

    # it shares the system prompt and nothing after it: the first conversation keeps its slot
    t = chat([SYSTEM, {"role": "user", "content": "Do you open on Sundays? " * 6}])
    assert t["id_slot"] != first["id_slot"]
    t = chat([SYSTEM, {"role": "user", "content": "Which shelf holds the atlas? " * 6}, {"role": "assistant", "content": "Shelf four."},
              {"role": "user", "content": "And the dictionaries?"}])
    assert t["id_slot"] == first["id_slot"]


def test_new_conversation_where_the_system_block_is_not_known():
    global server
    # the template of gemma puts the system prompt into the first user message, so the server does not know where the
    # system block ends: a new conversation that shares it brings less after it than it would drop of the first one
    server.chat_template_file = None
    server.start()

    first = chat([SYSTEM, {"role": "user", "content": "Which shelf holds the atlas? " * 6}, {"role": "assistant", "content": "Shelf four."},
                  {"role": "user", "content": "And the dictionaries? " * 6}])
    t = chat([SYSTEM, {"role": "user", "content": "Do you open on Sundays?"}])
    assert t["id_slot"] != first["id_slot"]
