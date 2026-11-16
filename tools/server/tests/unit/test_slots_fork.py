import os
import re
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from utils import *

# fork: several slots (server-context.cpp).
# - LLAMA_PREFILL_ORDER: the prompts go into the batch in the order of the tokens they have left. In the order of the
#   slots a short query waited until the long prompt of a slot before it was done.
# - evict_before_fork: the slot of a request nothing continued gives way before the slot of a conversation.
# - LLAMA_PREFILL_GEN_WAIT: with the prompts first (LLAMA_PREFILL_EXCLUSIVE) the generating slots wait for a prompt no
#   longer than this, then they have their turn.

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.n_ctx = 4096      # a slot of this model holds 2048 tokens at most
    server.n_slots = 2
    server.n_batch = server.n_ubatch = 16
    server.server_continuous_batching = True
    server.cache_ram = 0
    server.n_ctx_checkpoints = 0


def test_short_prompt_goes_before_a_long_one():
    global server
    server.start()

    n_long = 1900
    seen = {"processed": 0, "total": 0}
    started = threading.Event()

    def read_long():
        last = None
        for chunk in server.make_stream_request("POST", "/completion", data={
            "prompt": [100] * n_long, "n_predict": 1, "id_slot": 0,
            "stream": True, "return_progress": True, "temperature": 0,
        }):
            progress = chunk.get("prompt_progress")
            if progress:
                seen["processed"] = progress["processed"]
                seen["total"]     = progress["total"]
                if progress["processed"] > 0:
                    started.set()
            last = chunk
        return last

    with ThreadPoolExecutor(max_workers=1) as pool:
        answer = pool.submit(read_long)
        assert started.wait(timeout=60)
        assert seen["processed"] < n_long // 2, "the long prompt is too short for this machine"

        res = server.make_request("POST", "/completion", data={
            "prompt": [200] * 64, "n_predict": 1, "id_slot": 1, "temperature": 0,
        })
        assert res.status_code == 200
        # the short prompt is done while the long one is still read
        processed = seen["processed"]
        assert processed < seen["total"], f"the short prompt waited for the long one: {processed} of {seen['total']}"
        assert answer.result(timeout=120)["stop"]


def test_slot_of_a_single_request_gives_way_first():
    global server
    server.n_slots = 3
    server.n_ctx = 6144
    server.n_batch = server.n_ubatch = 512
    server.cache_ram = 64
    server.debug = True
    server.start()

    def complete(prompt, n_predict=4):
        res = server.make_request("POST", "/completion", data={
            "prompt": prompt, "n_predict": n_predict, "temperature": 0, "cache_prompt": True, "return_tokens": True,
        })
        assert res.status_code == 200, res.body
        return res.body

    def slots():
        res = server.make_request("GET", "/slots")
        assert res.status_code == 200
        return res.body

    # a conversation of two turns in one slot, then two single requests in the others
    talk = [100] * 600
    body = complete(talk)
    id_talk = body["id_slot"]
    talk = talk + body["tokens"] + [101] * 50
    body = complete(talk)
    assert body["id_slot"] == id_talk
    talk = talk + body["tokens"] + [102] * 50
    ids = {complete([300] * 200)["id_slot"], complete([400] * 200)["id_slot"]}
    assert len(ids) == 2 and id_talk not in ids

    # every slot is taken, and the conversation was used least recently: one more request takes the slot of a single
    # request, the conversation keeps its own
    assert complete([500] * 200)["id_slot"] in ids
    body = complete(talk)
    assert body["id_slot"] == id_talk
    assert body["timings"]["cache_n"] >= 650, body["timings"]


@pytest.mark.parametrize("wait_ms,progress", [(20, True), (0, False)])
def test_generation_takes_turns_with_a_long_prompt(monkeypatch, wait_ms, progress):
    global server
    monkeypatch.setenv("LLAMA_PREFILL_EXCLUSIVE", "1")
    monkeypatch.setenv("LLAMA_PREFILL_GEN_WAIT", str(wait_ms))
    monkeypatch.setenv("LLAMA_PREFILL_GEN_RUN", "20")
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()
    started = threading.Event()

    def generate():
        last = None
        for chunk in server.make_stream_request("POST", "/completion", data={
            "prompt": [100] * 16, "n_predict": 1500, "id_slot": 0,
            "stream": True, "ignore_eos": True, "temperature": 0,
        }):
            if chunk.get("content"):
                started.set()
            last = chunk
        return last

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            answer = pool.submit(generate)
            assert started.wait(timeout=60)
            res = server.make_request("POST", "/completion", data={
                "prompt": [200] * 1900, "n_predict": 2, "id_slot": 1, "temperature": 0,
            })
            assert res.status_code == 200
            assert answer.result(timeout=120)["stop"]
        with open(server.log_path, errors="replace") as f:
            text = f.read()
    finally:
        server.stop()
        os.remove(server.log_path)

    # the decode steps of the generating slot between the start and the end of the prompt of the other one
    begin = re.search(r"slot .*id\s+1 \|.*new prompt", text)
    end = re.search(r"slot .*id\s+1 \|.*slot decode token", text)
    assert begin and end and begin.start() < end.start()
    n_decode = len(re.findall(r"slot .*id\s+0 \|.*slot decode token", text[begin.end():end.start()]))
    if progress:
        assert n_decode > 10, f"only {n_decode} decode steps during the prompt of the other slot"
    else:
        assert n_decode <= 1, f"{n_decode} decode steps during the prompt of the other slot"
