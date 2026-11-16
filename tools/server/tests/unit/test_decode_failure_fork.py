import base64
import os
import random
import re
import shutil
import struct
import subprocess
import tempfile
import zlib

import pytest
from utils import *

# fork: a decode that fails (server-context.cpp). LLAMA_DEBUG_UBATCH_FAIL=N fails the Nth ubatch of the target context
# before its graph is built, as when the graph finds no memory, -N after its graph ran; it counts the ubatches of at
# least LLAMA_DEBUG_UBATCH_FAIL_SEQS sequences (2 by default). What the memory holds after a failure is checked on the
# cells, in tests/test-rs-rollback-fork.cpp and tests/test-dsv4-failed-fork.cpp.
# - keep_decoded_fork: the slots of the batch keep the tokens the memory still holds of them, the ubatches before the
#   one that failed. Upstream clears them (LLAMA_SLOT_KEEP_DECODED=0), and the next request reads its whole prompt again.
# - drop_lost_slots_fork: after a graph that ran the memory gives up every sequence whose state the graph can have
#   written: the recurrent state of a slot that was not in the batch too, when its cell lies between the cells of the
#   batch (llama_memory_recurrent::ubatch_restore). The server must forget the prompt of such a slot. With its tokens
#   kept the next request of the slot finds no cells, which aborts the server. A request for two completions puts two
#   slots into one batch, so no threads are needed. tinyLFM3 has a recurrent state next to its attention.
# - a batch the draft context fails on: the target holds it and the draft context does not. Upstream ends the tasks and
#   keeps the tokens of the slots, and a draft model then refuses every next batch of the slot (its positions must be
#   consecutive). The slot goes back to what both contexts hold. A draft model is a context of the same kind as the
#   target, so the knob fails its Nth ubatch too, a request later.
# - an image decoded on its own in several batches, one of which fails: the memory holds a part of the image, the slot
#   lists none of it. A plain attention cache takes the part back and the slot keeps the text before the image, with a
#   draft model too, whose context gets every batch of the image after the target. A memory with a recurrent state
#   cannot take a part of an image back, and after a graph that ran it holds nothing of the slot: the slot is cleared,
#   or its next request finds no cells for the tokens it lists, which aborts the server (tinyLFM3 has a vision
#   projector).
# - an image that goes into the batch of the text around it (server_mtmd_mixed_fork) and fails to encode
#   (LLAMA_SERVER_DEBUG_ENCODE_FAIL=N, the Nth encode): the text of that step before the image is counted in the tokens
#   of the slot and leaves the batch undecoded. The slot goes back to the tokens of the batches before.
# - a model with a recurrent state next to its attention and no window of attention, the memory of the 395 model, which
#   no preset has: the qwen4exp of the test matrix, generated here by test-llama-archs as ctest does (hybrid_model).
#   Its vocabulary is 128 ids, so the prompts are token ids. A slot of the batch goes on from the tokens it keeps
#   without a checkpoint; after a graph that ran the memory holds nothing of it and the prompt is read again; a slot
#   that was not in the batch keeps its state when the graph did not run, and reads its prompt again when it ran; with
#   a draft model the slot is cleared, as the recurrent state does not go back to where the draft is and a draft model
#   cannot start in the middle of a conversation.

PROMPT = "The quick brown fox jumps over the lazy dog near the old mill. "

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org-dev/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def run(monkeypatch, knob: str | None):
    if knob:
        monkeypatch.setenv("LLAMA_DEBUG_UBATCH_FAIL", knob)
    else:
        monkeypatch.delenv("LLAMA_DEBUG_UBATCH_FAIL", raising=False)

    server = ServerPreset.tinylfm3omni()
    server.n_slots = 3
    server.n_ctx = 4096
    server.kv_unified = True
    server.no_cache_idle_slots = True # the idle slot stays in the memory
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()

    def post(slot: int, prompt: str, n_cmpl: int = 1):
        return server.make_request("POST", "/completion", data={
            "prompt": prompt, "id_slot": slot, "n_cmpl": n_cmpl,
            "temperature": 0.0, "top_k": 1, "n_predict": 16, "cache_prompt": True,
        })

    try:
        # slot 0 takes the first cell of the recurrent state, slot 1 the second
        assert post(0, PROMPT * 4).status_code == 200
        first = post(1, PROMPT * 5 + "Then the hare said:")
        assert first.status_code == 200, first.body
        # two completions: slot 0 and the free slot 2 in one batch, the cell of slot 1 between theirs
        two = post(0, PROMPT * 6, n_cmpl=2)
        second = post(1, PROMPT * 5 + "Then the hare said:" + first.body["content"] + " And the fox:")
        assert second.status_code == 200, second.body
        with open(server.log_path, errors="replace") as f:
            text = f.read()
    finally:
        server.stop()
        os.remove(server.log_path)
    return two.status_code, [first.body["content"], second.body["content"]], text


def test_slot_outside_a_failed_batch_reads_its_prompt_again(monkeypatch):
    status, ref, _ = run(monkeypatch, None)
    assert status == 200

    status, out, text = run(monkeypatch, "-1")
    assert status == 500
    # the cell of slot 1 lay in the range of the failed ubatch: without this line the test does not reach the path
    assert "the failed decode took the state of this slot" in text
    assert out == ref


def run_prompt(monkeypatch, env: dict):
    for name in ("LLAMA_DEBUG_UBATCH_FAIL", "LLAMA_DEBUG_UBATCH_FAIL_SEQS", "LLAMA_SLOT_KEEP_DECODED"):
        if name in env:
            monkeypatch.setenv(name, env[name])
        else:
            monkeypatch.delenv(name, raising=False)

    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_ctx = 2048
    server.n_batch = 64   # four ubatches a decode
    server.n_ubatch = 16
    server.start()

    res = []
    try:
        for _ in range(3):
            r = server.make_request("POST", "/completion", data={
                "prompt": PROMPT * 30, "id_slot": 0,
                "temperature": 0.0, "top_k": 1, "n_predict": 16, "cache_prompt": True,
            })
            res.append((r.status_code, r.body.get("content"), r.body.get("timings", {}).get("prompt_n")))
    finally:
        server.stop()
    return res


# 10: the prompt is past its first decode, whatever the server decoded before it. a plain attention cache keeps the
# ubatches that ran before the failed one in both cases, its cells do not change once they are written
@pytest.mark.parametrize("n_fail", ["10", "-10"])
def test_prompt_goes_on_after_a_failed_ubatch(monkeypatch, n_fail: str):
    ref = run_prompt(monkeypatch, {})
    assert [status for status, _, _ in ref] == [200, 200, 200]
    n_prompt = ref[0][2]
    assert n_prompt > 300, "the prompt is too short for the ubatch that fails"

    fail = {"LLAMA_DEBUG_UBATCH_FAIL": n_fail, "LLAMA_DEBUG_UBATCH_FAIL_SEQS": "1"}

    out = run_prompt(monkeypatch, fail)
    assert [status for status, _, _ in out] == [500, 200, 200]
    # the request again reads what the failed one left, and answers as a server that never failed
    assert 0 < out[1][2] <= n_prompt - 64
    assert out[1][1] == ref[0][1]
    assert out[2][1] == ref[1][1]

    out = run_prompt(monkeypatch, {**fail, "LLAMA_SLOT_KEEP_DECODED": "0"})
    assert [status for status, _, _ in out] == [500, 200, 200]
    assert out[1][2] == n_prompt
    assert out[1][1] == ref[0][1]


def run_draft(monkeypatch, n_fail: str | None):
    if n_fail:
        monkeypatch.setenv("LLAMA_DEBUG_UBATCH_FAIL", n_fail)
        monkeypatch.setenv("LLAMA_DEBUG_UBATCH_FAIL_SEQS", "1")
    else:
        monkeypatch.delenv("LLAMA_DEBUG_UBATCH_FAIL", raising=False)

    server = ServerPreset.stories15m_moe()
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 2
    server.spec_draft_n_max = 6
    server.fa = "off"
    server.n_slots = 1
    server.n_batch = 64
    server.n_ubatch = 16
    server.start()

    res = []
    try:
        for _ in range(4):
            r = server.make_request("POST", "/completion", data={
                "prompt": PROMPT * 10, "id_slot": 0,
                "temperature": 0.0, "top_k": 1, "n_predict": 24, "cache_prompt": True,
            })
            res.append((r.status_code, r.body.get("content"), r.body.get("timings", {}).get("prompt_n")))
    finally:
        server.stop()
    return res


def test_target_then_draft_context_fail(monkeypatch):
    ref = run_draft(monkeypatch, None)
    assert [status for status, _, _ in ref] == [200, 200, 200, 200]

    # the target fails in its second decode of the prompt, the draft context in its own a request later
    out = run_draft(monkeypatch, "6")
    assert [status for status, _, _ in out] == [500, 500, 200, 200]
    assert 0 < out[2][2] < ref[0][2]
    assert out[2][1] == ref[0][1]
    assert out[3][1] == ref[1][1]


def png(seed: int) -> str:
    # a 32x32 image of noise, no file and no download
    rng = random.Random(seed)
    raw = b"".join(b"\x00" + bytes(rng.randrange(256) for _ in range(3 * 32)) for _ in range(32))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)

    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0)) +
            chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    return base64.b64encode(data).decode()


def run_image(monkeypatch, n_fail: str | None, draft: bool = False, n_requests: int = 3):
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")
    if n_fail:
        monkeypatch.setenv("LLAMA_DEBUG_UBATCH_FAIL", n_fail)
        monkeypatch.setenv("LLAMA_DEBUG_UBATCH_FAIL_SEQS", "1")
    else:
        monkeypatch.delenv("LLAMA_DEBUG_UBATCH_FAIL", raising=False)

    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.n_ctx = 4096
    server.n_batch = 64   # the 256 tokens of an image take four decodes
    server.n_ubatch = 64
    if draft:
        # the model drafts for itself: its draft context takes the batches of the image too
        server.model_draft_hf_repo = server.model_hf_repo
        server.spec_type = "draft-simple"
        server.spec_draft_n_min = 2
        server.spec_draft_n_max = 6
    server.start()

    res = []
    try:
        for _ in range(n_requests):
            r = server.make_request("POST", "/completion", data={
                "prompt": {"prompt_string": PROMPT * 20 + "<__media__>" + " What is on it?", "multimodal_data": [png(1)]},
                "temperature": 0.0, "top_k": 1, "n_predict": 8, "cache_prompt": True,
            })
            res.append((r.status_code, r.body.get("content"), r.body.get("timings", {}).get("prompt_n")))
    finally:
        server.stop()
    return res


def test_text_before_a_failed_image_is_kept(monkeypatch):
    ref = run_image(monkeypatch, None)
    assert [status for status, _, _ in ref] == [200, 200, 200]
    n_prompt = ref[0][2]
    assert n_prompt > 256 + 4 * 64, "the text before the image is too short for the ubatch that fails"

    # the text takes five decodes, the image the four after them, and the server decodes once or twice before the
    # prompt: the 10th ubatch is the third or the last of the image, a part of the image is in the memory then
    out = run_image(monkeypatch, "10")
    assert [status for status, _, _ in out] == [500, 200, 200]
    assert 256 < out[1][2] < n_prompt
    assert out[1][1] == ref[0][1]
    assert out[2][1] == ref[1][1]


def test_text_before_a_failed_image_is_kept_with_a_draft_model(monkeypatch):
    ref = run_image(monkeypatch, None, draft=True, n_requests=4)
    assert [status for status, _, _ in ref] == [200, 200, 200, 200]
    n_prompt = ref[0][2]

    # the target fails in a batch of the image past its first one, then, a request later, the draft context does in a
    # batch of the image the target holds. both times the slot goes back to the text before the image in both contexts
    # (the draft context holds batches of the image the slot does not list), and the third request reads the image
    out = run_image(monkeypatch, "10", draft=True, n_requests=4)
    assert [status for status, _, _ in out] == [500, 500, 200, 200]
    assert 256 < out[2][2] < n_prompt
    assert out[2][1] == ref[0][1]
    assert out[3][1] == ref[1][1]


def run_image_hybrid(monkeypatch, n_fail: str | None):
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")
    if n_fail:
        monkeypatch.setenv("LLAMA_DEBUG_UBATCH_FAIL", n_fail)
        monkeypatch.setenv("LLAMA_DEBUG_UBATCH_FAIL_SEQS", "1")
    else:
        monkeypatch.delenv("LLAMA_DEBUG_UBATCH_FAIL", raising=False)

    server = ServerPreset.tinylfm3omni()
    server.n_slots = 1
    server.n_ctx = 4096
    server.n_batch = 16   # the text before the image is one decode, the 64 tokens of the image take four
    server.n_ubatch = 16
    server.start()

    res = []
    try:
        for _ in range(3):
            r = server.make_request("POST", "/completion", data={
                "prompt": {"prompt_string": PROMPT + "<__media__>" + " What is on it?", "multimodal_data": [png(1)]},
                "temperature": 0.0, "top_k": 1, "n_predict": 8, "cache_prompt": True,
            })
            res.append((r.status_code, r.body.get("content"), r.body.get("timings", {}).get("prompt_n")))
    finally:
        server.stop()
    return res


# 5: a batch of the image, whether the server decoded once or twice before the prompt. before its graph the memory
# holds a part of the image, after it nothing of the slot
@pytest.mark.parametrize("n_fail", ["5", "-5"])
def test_hybrid_slot_is_cleared_after_a_failed_image(monkeypatch, n_fail: str):
    ref = run_image_hybrid(monkeypatch, None)
    assert [status for status, _, _ in ref] == [200, 200, 200]
    assert ref[0][2] > 16 + 3 * 16, "the image is too short for the ubatch that fails"

    out = run_image_hybrid(monkeypatch, n_fail)
    assert [status for status, _, _ in out] == [500, 200, 200]
    assert out[1][2] == ref[0][2]
    assert out[1][1] == ref[0][1]
    assert out[2][1] == ref[1][1]


def run_image_mixed(monkeypatch, n_fail: str | None):
    monkeypatch.setenv("LLAMA_MEDIA_MARKER", "<__media__>")
    monkeypatch.setenv("LLAMA_MTMD_MIXED", "1") # the image goes into the batch of the text around it
    monkeypatch.delenv("LLAMA_DEBUG_UBATCH_FAIL", raising=False)
    if n_fail:
        monkeypatch.setenv("LLAMA_SERVER_DEBUG_ENCODE_FAIL", n_fail)
    else:
        monkeypatch.delenv("LLAMA_SERVER_DEBUG_ENCODE_FAIL", raising=False)

    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.n_ctx = 4096
    server.n_batch = 512   # the 256 tokens of the image fit a batch next to text
    server.n_ubatch = 512
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()

    res = []
    try:
        for _ in range(3):
            r = server.make_request("POST", "/completion", data={
                # the text after the image is longer than a batch: the prompt does not stop before the image for a
                # checkpoint near its end (server_ckpt_tail_fork)
                "prompt": {"prompt_string": PROMPT * 45 + "<__media__>" + PROMPT * 60, "multimodal_data": [png(1)]},
                "temperature": 0.0, "top_k": 1, "n_predict": 8, "cache_prompt": True,
            })
            res.append((r.status_code, r.body.get("content"), r.body.get("timings", {}).get("prompt_n")))
        with open(server.log_path, errors="replace") as f:
            text = f.read()
    finally:
        server.stop()
        os.remove(server.log_path)
    return res, text


def test_text_before_an_image_that_fails_to_encode_is_kept(monkeypatch):
    ref, _ = run_image_mixed(monkeypatch, None)
    assert [status for status, _, _ in ref] == [200, 200, 200]

    out, text = run_image_mixed(monkeypatch, "1")
    assert [status for status, _, _ in out] == [500, 200, 200]
    # the batch of the image began with text, which the slot counted and no decode saw: the slot gives it up
    kept = re.search(r"the slot keeps (\d+) of its (\d+) tokens", text)
    assert kept is not None and 0 < int(kept.group(1)) < int(kept.group(2))
    # how much the request again reads is not checked: tinygemma3 has a window of attention, and with no checkpoint yet
    # it reads the whole prompt whatever the slot keeps. on a hybrid model the request goes on from the tokens of the
    # slot, and with the text counted it skips that text (positions may leave gaps where images are M-RoPE)
    assert out[1][1] == ref[0][1]
    assert out[2][1] == ref[1][1]


@pytest.fixture(scope="module")
def hybrid_model(tmp_path_factory):
    server_bin = os.environ.get("LLAMA_SERVER_BIN_PATH", "../../../build/bin/llama-server")
    generator = os.path.join(os.path.dirname(server_bin), "test-llama-archs")
    if not os.path.exists(generator):
        pytest.skip("test-llama-archs is not built: no model with a recurrent state and no window of attention")
    out = tmp_path_factory.mktemp("hybrid")
    subprocess.run([generator, "-a", "^qwen4exp$", "-o", str(out)], check=True, capture_output=True)
    path = out / "qwen4exp-moe.gguf"
    assert path.exists()
    return str(path)


def ids(seed: int, n: int) -> list:
    rng = random.Random(seed)
    return [3 + rng.randrange(100) for _ in range(n)]


def hybrid_server(model: str) -> ServerProcess:
    server = ServerProcess()
    server.model_hf_repo = None
    server.model_hf_file = None
    server.model_file = model
    server.override_kv = "qwen4exp.context_length=int:4096" # the generated model has 256
    server.n_ctx = 2048
    server.n_batch = 32  # four ubatches a decode
    server.n_ubatch = 8
    server.temperature = 0.0
    return server


def set_knobs(monkeypatch, env: dict):
    for name in ("LLAMA_DEBUG_UBATCH_FAIL", "LLAMA_DEBUG_UBATCH_FAIL_SEQS", "LLAMA_SLOT_KEEP_DECODED", "LLAMA_SERVER_DEBUG_ENCODE_FAIL"):
        if name in env:
            monkeypatch.setenv(name, env[name])
        else:
            monkeypatch.delenv(name, raising=False)


def run_hybrid_prompt(monkeypatch, model: str, env: dict, draft: bool = False, n_requests: int = 3):
    set_knobs(monkeypatch, env)

    server = hybrid_server(model)
    server.n_slots = 1
    if draft:
        server.model_draft = model
        server.spec_type = "draft-simple"
        server.spec_draft_n_min = 2
        server.spec_draft_n_max = 6
    server.start()

    res = []
    try:
        for _ in range(n_requests):
            r = server.make_request("POST", "/completion", data={
                "prompt": ids(1, 200), "id_slot": 0,
                "temperature": 0.0, "top_k": 1, "n_predict": 16, "cache_prompt": True, "return_tokens": True,
            })
            res.append((r.status_code, r.body.get("tokens"), r.body.get("timings", {}).get("prompt_n")))
    finally:
        server.stop()
    return res


# 12: a ubatch of the prompt that is not the first of its decode, whether the server decoded once or twice before
@pytest.mark.parametrize("n_fail", ["12", "-12"])
def test_hybrid_prompt_goes_on_after_a_failed_ubatch(monkeypatch, hybrid_model, n_fail: str):
    ref = run_hybrid_prompt(monkeypatch, hybrid_model, {})
    assert [status for status, _, _ in ref] == [200, 200, 200]
    n_prompt = ref[0][2]
    assert n_prompt == 200

    fail = {"LLAMA_DEBUG_UBATCH_FAIL": n_fail, "LLAMA_DEBUG_UBATCH_FAIL_SEQS": "1"}

    out = run_hybrid_prompt(monkeypatch, hybrid_model, fail)
    assert [status for status, _, _ in out] == [500, 200, 200]
    if n_fail.startswith("-"):
        # the graph ran: the memory gave the sequence up whole
        assert out[1][2] == n_prompt
    else:
        # the ubatches before the failed one stay, those of its own decode too, and no checkpoint is needed to go on
        assert 0 < out[1][2] < n_prompt
        assert (n_prompt - out[1][2]) % 32 != 0
    assert out[1][1] == ref[0][1]
    assert out[2][1] == ref[1][1]

    out = run_hybrid_prompt(monkeypatch, hybrid_model, {**fail, "LLAMA_SLOT_KEEP_DECODED": "0"})
    assert [status for status, _, _ in out] == [500, 200, 200]
    assert out[1][2] == n_prompt
    assert out[1][1] == ref[0][1]


def test_hybrid_slot_is_cleared_with_a_draft_model(monkeypatch, hybrid_model):
    ref = run_hybrid_prompt(monkeypatch, hybrid_model, {}, draft=True, n_requests=4)
    assert [status for status, _, _ in ref] == [200, 200, 200, 200]
    n_prompt = ref[0][2]

    # the target fails inside a decode, the draft context in its own a request later: either way the target holds
    # tokens the draft context has not seen, more of them than the recurrent state goes back
    out = run_hybrid_prompt(monkeypatch, hybrid_model, {"LLAMA_DEBUG_UBATCH_FAIL": "12", "LLAMA_DEBUG_UBATCH_FAIL_SEQS": "1"}, draft=True, n_requests=4)
    assert [status for status, _, _ in out] == [500, 500, 200, 200]
    assert out[2][2] == n_prompt
    assert out[2][1] == ref[0][1]
    assert out[3][1] == ref[1][1]


def run_hybrid_bystander(monkeypatch, model: str, knob: str | None):
    set_knobs(monkeypatch, {"LLAMA_DEBUG_UBATCH_FAIL": knob} if knob else {})

    server = hybrid_server(model)
    server.n_slots = 3
    server.kv_unified = True
    server.no_cache_idle_slots = True # the idle slot stays in the memory
    server.slot_save_path = tempfile.mkdtemp()
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    server.start()

    def post(slot: int, prompt: list, n_cmpl: int = 1):
        return server.make_request("POST", "/completion", data={
            "prompt": prompt, "id_slot": slot, "n_cmpl": n_cmpl,
            "temperature": 0.0, "top_k": 1, "n_predict": 16, "cache_prompt": True, "return_tokens": True,
        })

    try:
        # slot 0 takes the first cell of the recurrent state, slot 1 the second
        assert post(0, ids(1, 40)).status_code == 200
        first = post(1, ids(2, 48))
        assert first.status_code == 200, first.body
        # two completions: slot 0 and the free slot 2 in one batch, the cell of slot 1 between theirs
        two = post(0, ids(1, 40) + ids(3, 24), n_cmpl=2)
        # what the memory holds of slot 1 now, as a file: the answers of the generated model depend too little on
        # its recurrent state to show a wrong one
        state = None
        saved = server.make_request("POST", "/slots/1?action=save", data={"filename": "slot1.bin"})
        if saved.status_code == 200 and saved.body.get("n_saved", 0) > 0:
            with open(os.path.join(server.slot_save_path, "slot1.bin"), "rb") as f:
                state = f.read()
        second = post(1, ids(2, 48) + first.body["tokens"] + ids(4, 2))
        assert second.status_code == 200, second.body
        with open(server.log_path, errors="replace") as f:
            text = f.read()
    finally:
        server.stop()
        os.remove(server.log_path)
        shutil.rmtree(server.slot_save_path, ignore_errors=True)
    return two.status_code, [first.body["tokens"], second.body["tokens"]], text, state


def test_hybrid_slot_outside_a_failed_batch(monkeypatch, hybrid_model):
    status, ref, _, ref_state = run_hybrid_bystander(monkeypatch, hybrid_model, None)
    assert status == 200
    assert ref_state is not None

    # the graph did not run: the cells go back. the slot holds the state it had, byte for byte, and reads nothing again
    status, out, text, state = run_hybrid_bystander(monkeypatch, hybrid_model, "1")
    assert status == 500
    assert "the failed decode took the state of this slot" not in text
    assert state == ref_state
    assert out == ref

    # the graph ran: the memory gave the state of the slot up, its prompt is read again
    status, out, text, state = run_hybrid_bystander(monkeypatch, hybrid_model, "-1")
    assert status == 500
    assert "the failed decode took the state of this slot" in text
    assert state is None
    assert out == ref
