import pytest
from utils import *

# fork: a rollback of rejected draft tokens that the memory refuses (rollback_recover_fork in server-context.cpp).
# LLAMA_SERVER_DEBUG_ROLLBACK_FAIL=N leaves out every Nth rollback, as a recurrent memory does when another ubatch of the
# decode took the snapshots of the sequence. The slot must go back and decode again what it holds, so the answers are
# those of a server whose rollbacks all work. Until 16.11 the slot was cleared and the rest of the answer came from an
# empty context.

MODEL_DRAFT_FILE_URL = "https://huggingface.co/ggml-org-dev/tiny-llamas/resolve/main/stories15M-q4_0.gguf"


def run_turns(monkeypatch, every: int):
    if every > 0:
        monkeypatch.setenv("LLAMA_SERVER_DEBUG_ROLLBACK_FAIL", str(every))
    else:
        monkeypatch.delenv("LLAMA_SERVER_DEBUG_ROLLBACK_FAIL", raising=False)

    server = ServerPreset.stories15m_moe()
    server.model_draft = download_file(MODEL_DRAFT_FILE_URL)
    server.spec_type = "draft-simple"
    server.spec_draft_n_min = 4
    server.spec_draft_n_max = 8
    server.fa = "off"
    server.n_slots = 1
    server.start()

    answers = []
    n_draft = 0
    n_accepted = 0
    prompt = "Once upon a time there was a little girl who liked to"
    try:
        for _ in range(3):
            res = server.make_request("POST", "/completion", data={
                "prompt": prompt,
                "temperature": 0.0,
                "top_k": 1,
                "n_predict": 48,
                "cache_prompt": True,
            })
            assert res.status_code == 200, res.body
            answers.append(res.body["content"])
            n_draft    += res.body["timings"]["draft_n"]
            n_accepted += res.body["timings"]["draft_n_accepted"]
            # the next turn goes on from the state the slot was left with
            prompt += res.body["content"] + " Then she"
    finally:
        server.stop()
    return answers, n_draft, n_accepted


def test_refused_rollback_keeps_the_answers(monkeypatch):
    ref, n_draft, n_accepted = run_turns(monkeypatch, 0)
    assert n_accepted < n_draft, "no draft token was rejected, nothing to roll back"

    out, _, _ = run_turns(monkeypatch, 1)
    assert out == ref
