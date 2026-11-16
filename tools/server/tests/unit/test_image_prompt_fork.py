import base64
import os
import random
import struct
import tempfile
import zlib

import pytest
from utils import *

# fork: images in a prompt (tools/server/server-context.cpp, fork/server-context-fork.h).
# - server_ckpt_tail_fork: a checkpoint near the end of a prompt goes before the image that stands on its point, and one
#   more where the request before this one left the last prompt
# - server_embd_cache_fork: an image that comes again is not encoded again
# - server_mtmd_mixed_fork: the images go into the batch next to the text
# The client of the tests ends every request with the same two images and puts its new text before them, as an agent
# did on the 395 (14.11). tinygemma3 uses SWA, so a prompt longer than its window goes back only to a checkpoint.

server: ServerProcess

MARKER = "<__media__>"
STORY  = "The quick brown fox jumps over the lazy dog near the old mill. "


class LogReader:
    def __init__(self, path):
        self.path = path
        self.pos = 0

    def drain(self):
        with open(self.path, errors="replace") as f:
            f.seek(self.pos)
            content = f.read()
            self.pos = f.tell()
        return content


def png(seed: int) -> str:
    # a 32x32 image of noise, no file and no download
    rng = random.Random(seed)
    raw = b"".join(b"\x00" + bytes(rng.randrange(256) for _ in range(3 * 32)) for _ in range(32))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(data).decode()


def run_turns(monkeypatch, env: dict, n_turns: int = 4):
    global server
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", MARKER)
    for name in ("LLAMA_CKPT_TAIL", "LLAMA_MTMD_MIXED", "LLAMA_MTMD_EMBD_CACHE"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.n_ctx = 8192
    server.n_batch = 2048
    server.n_ubatch = 512
    server.n_predict = 8
    server.debug = True
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()
    log = LogReader(server.log_path)

    images = [png(1), png(2)]
    history = STORY * 100
    turns = []
    try:
        for i in range(n_turns):
            res = server.make_request("POST", "/completion", data={
                "prompt": {"prompt_string": history + MARKER + " and " + MARKER + " What is on them?", "multimodal_data": images},
                "temperature": 0.0, "top_k": 1, "n_predict": 8, "cache_prompt": True,
            })
            assert res.status_code == 200, res.body
            turns.append({"content": res.body["content"], **res.body["timings"]})
            history += res.body["content"] + " Then the fox met a hare, number %d. " % i
        text = log.drain()
    finally:
        server.stop()
        os.remove(server.log_path)
    return turns, text


def test_tail_of_images_is_read_once_a_turn(monkeypatch):
    turns, text = run_turns(monkeypatch, {})
    # two images of 256 tokens, the answer and a sentence: the turns after the first go back to where the images stood
    for t in turns[2:]:
        assert t["prompt_n"] < 256 * 2 + 80, t
    # the images were encoded for the first turn only
    assert text.count("its embeddings come from the cache") >= 2 * (len(turns) - 1)


def test_images_next_to_the_text_give_the_same_answers(monkeypatch):
    apart, _ = run_turns(monkeypatch, {"LLAMA_MTMD_MIXED": "0"})
    mixed, text = run_turns(monkeypatch, {"LLAMA_MTMD_MIXED": "1"})
    assert "next to the text" in text
    assert [t["content"] for t in mixed] == [t["content"] for t in apart]
    assert [t["prompt_n"] for t in mixed] == [t["prompt_n"] for t in apart]
