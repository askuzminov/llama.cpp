# fork: the delta context checkpoints (common_prompt_checkpoint::base_pos, server_ckpt_next_base_pos) through
# /slots save and restore. The appendix of the slot file keeps upstream's layout, which has no base_pos: the loader
# takes it from the header of the delta state (llama_state_seq_delta_base_pos). Before 10.11 a restored delta was
# read as a whole state, failed to load, and the prompt was computed again from the start.

import pytest
from utils import *

server = ServerPreset.tinygemma3()

# 20 sentences and an ending: the checkpoints at the end of the prompt (4 + n_ubatch and 4 tokens before it) are a base
# and a delta on it, and the two endings differ in their last word, after the delta
BASE = "The quick brown fox jumps over the lazy dog. " * 20
PROMPT_A = BASE + "The first ending of this story is a happy one."
PROMPT_B = BASE + "The first ending of this story is a happy end."


@pytest.fixture(autouse=True)
def create_server(tmp_path):
    global server
    server = ServerPreset.tinygemma3()  # SWA: its checkpoints can be deltas
    server.slot_save_path = str(tmp_path)
    server.temperature = 0.0
    server.cache_ram = 0
    server.n_ubatch = 32


def completion(prompt):
    res = server.make_request("POST", "/completion", data={
        "prompt": prompt,
        "id_slot": 1,
        "cache_prompt": True,
        "n_predict": 4,
    })
    assert res.status_code == 200, res.body
    return res.body


def test_slot_restore_loads_a_delta_checkpoint():
    server.start()

    # live: B goes on from the delta checkpoint of A
    n_full = completion(PROMPT_A)["timings"]["prompt_n"]
    live = completion(PROMPT_B)
    n_live = live["timings"]["prompt_n"]
    assert n_live < 16 < n_full, (n_live, n_full)

    # the same after a save of A, another prompt in the slot and a restore
    res = server.make_request("POST", "/slots/1?action=erase")
    assert res.status_code == 200
    completion(PROMPT_A)
    res = server.make_request("POST", "/slots/1?action=save", data={"filename": "delta.bin"})
    assert res.status_code == 200
    completion("Unrelated text with no common prefix occupies the slot now.")
    res = server.make_request("POST", "/slots/1?action=restore", data={"filename": "delta.bin"})
    assert res.status_code == 200
    restored = completion(PROMPT_B)
    assert restored["timings"]["prompt_n"] == n_live
    assert restored["content"] == live["content"]
