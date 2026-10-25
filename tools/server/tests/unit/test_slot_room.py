# fork: the room of a unified KV cache (LLAMA_SLOT_ROOM, server_slot_room_fork in tools/server/server-context.cpp).
# Every slot may use the whole cache, but not all of them together: an idle slot gives way into the prompt cache
# instead of being purged, a load from the prompt cache makes room for its state first, a task without room waits in
# the queue, and a running slot pauses (its state to RAM) while the others need the cells. Upstream (LLAMA_SLOT_ROOM=0)
# purges idle slots without saving them and fails the running slots of a full cache: every test here fails with it.
# --no-cache-idle-slots keeps the idle slots in the cache, as agents run the server.
# The room is made before each batch (room_fork), so no decode may run out of cells: that path retries with idle slots
# evicted inside the decode, and the tests require it never ran.

import os
import tempfile

import pytest
from utils import *

server = ServerPreset.tinyllama2()

# three stories that share no prefix, of 33, 51 and 57 tokens a sentence
STORY_A = "Once upon a time there was a little girl named Lily who loved to play in the garden with her red ball. "
STORY_B = "The old man walked slowly to the market every morning to buy fresh bread and warm milk for his gray cat. "
STORY_C = "In a small village near the sea lived a fisherman who sang quiet songs to the waves late every night. "


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path) as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 512
    server.n_slots = 3
    server.kv_unified = True
    server.no_cache_idle_slots = True
    server.cache_ram = 100
    server.temperature = 0.0
    server.n_predict = 16
    server.server_slots = True
    fd, server.log_path = tempfile.mkstemp(suffix='.log')
    os.close(fd)
    yield


def n_tokens(text: str) -> int:
    res = server.make_request("POST", "/tokenize", data={"content": text})
    assert res.status_code == 200
    return len(res.body["tokens"])


NO_ROOM = "failed to find free space in the KV cache"


def complete(prompt: str, n_predict: int, id_slot: int | None = None):
    data = {"prompt": prompt, "n_predict": n_predict, "cache_prompt": True, "ignore_eos": True}
    if id_slot is not None:
        data["id_slot"] = id_slot
    return server.make_request("POST", "/completion", data=data)


def test_idle_slot_gives_way_and_comes_back():
    global server
    server.start()
    log = LogReader(server.log_path)

    prompts = [STORY_A * 5, STORY_B * 3, STORY_C * 3]
    sizes = [n_tokens(p) + server.n_predict for p in prompts]
    # the premise: each conversation fits the cache, the three of them do not
    assert max(sizes) < server.n_ctx < sum(sizes)

    for slot, prompt in enumerate(prompts):
        res = complete(prompt, server.n_predict, id_slot=slot)
        assert res.status_code == 200
        assert res.body["tokens_predicted"] == server.n_predict

    # the third took the room of the idle slot used least recently, which went into the prompt cache first
    text = log.drain()
    assert "purging slot 0 with" in text
    assert "the least recently used idle one, saved to the prompt cache" in text

    # the first conversation comes back: its state is loaded from the prompt cache, the second slot giving way for it.
    # the load makes room for the state only, the room of its new tail and of its answer comes before each batch, the
    # third slot giving way then
    res = complete(STORY_A * 5 + "Lily smiled. " + STORY_B, 120)
    assert res.status_code == 200
    assert res.body["tokens_predicted"] == 120
    text += log.drain()
    assert "purging slot 1 with" in text
    assert "purging slot 2 with" in text
    assert res.body["timings"]["cache_n"] >= n_tokens(STORY_A * 5) - 1
    assert NO_ROOM not in text


def test_task_waits_for_room():
    global server
    server.start()
    log = LogReader(server.log_path)

    n_predict = 100
    prompts = [STORY_A, STORY_B, STORY_C * 3]
    sizes = [n_tokens(p) + n_predict for p in prompts]
    # the premise: any two run together, the three of them do not fit
    assert sum(sizes) - min(sizes) < server.n_ctx < sum(sizes)

    results = parallel_function_calls([(complete, (p, n_predict)) for p in prompts])

    # the last one waited in the queue until a slot was done, nothing failed
    for res in results:
        assert res.status_code == 200
        assert res.body["tokens_predicted"] == n_predict
    text = log.drain()
    assert "waits for room" in text
    assert NO_ROOM not in text


def test_running_slot_pauses_for_room(monkeypatch):
    global server
    # no reserve for the answers: both tasks start, then their answers outgrow the cache together
    monkeypatch.setenv("LLAMA_SLOT_ANSWER_ROOM", "0")
    server.n_slots = 2
    server.start()
    log = LogReader(server.log_path)

    n_predict = 200
    prompts = [STORY_A * 2, STORY_B * 2]
    sizes = [n_tokens(p) + n_predict for p in prompts]
    # the premise: each fits the cache alone, the two answers together do not
    assert max(sizes) < server.n_ctx < sum(sizes)

    results = parallel_function_calls([(complete, (p, n_predict)) for p in prompts])
    text = log.drain()

    # the task that came last paused while the other one needed the cells, and went on where it stopped
    for res in results:
        assert res.status_code == 200
        assert res.body["tokens_predicted"] == n_predict
    assert "paused for room in the unified KV cache" in text
    assert "resumed with room in the unified KV cache" in text
    assert NO_ROOM not in text

    # the answers are those of each prompt alone: the paused state came back as it was
    for prompt, res in zip(prompts, results):
        alone = complete(prompt, n_predict)
        assert alone.status_code == 200
        assert alone.body["content"] == res.body["content"]


def test_no_pause_below_ram_reserve(monkeypatch):
    global server
    # the case above, with a host RAM reserve no host has: a paused state would take host RAM below it, so no slot
    # pauses (room_ram_fork), and the decode that finds no room takes the upstream path
    monkeypatch.setenv("LLAMA_SLOT_ANSWER_ROOM", "0")
    server.n_slots = 2
    server.cache_ram_reserve = 1 << 30
    server.start()
    log = LogReader(server.log_path)

    n_predict = 200
    prompts = [STORY_A * 2, STORY_B * 2]
    parallel_function_calls([(complete, (p, n_predict)) for p in prompts])
    text = log.drain()

    assert "no pause for room" in text
    assert "paused for room in the unified KV cache" not in text
