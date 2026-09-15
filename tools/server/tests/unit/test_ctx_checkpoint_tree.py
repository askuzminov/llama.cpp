import pytest
from utils import *

# End-to-end tests for the experimental branch-tree context checkpoints (--ctx-checkpoints-tree).
#
# Note on model choice: context checkpoints are only *created* for models that cannot roll back
# their KV cache directly - i.e. SWA / recurrent / hybrid models (see do_checkpoint in the server).
# For a plain attention model the server rolls back with a partial seq-rm, so no checkpoints (and no
# tree) are ever built. Therefore:
#   - the no-regression test uses a plain attention model (tinyllama2): the tree must be inert and
#     behave exactly like the linear path;
#   - the engagement test uses a SWA model (tinygemma3): the tree is actually exercised.


PREFIX = (
    "Once upon a time there was a little girl who lived in a small house near the woods and she "
    "loved to play outside every single day with her friends and her small brown dog by the river"
)


def _completion_prompt_n(server: ServerProcess, suffix: str) -> int:
    res = server.make_request("POST", "/completion", data={
        "prompt": PREFIX + suffix,
        "n_predict": 4,
        "cache_prompt": True,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    return res.body["timings"]["prompt_n"]


def _branch_scenario_completion(server: ServerProcess):
    # A -> B (shares the long prefix, diverges at the tail) -> A again
    a1 = _completion_prompt_n(server, " and the cats")
    b  = _completion_prompt_n(server, " and the dogs")
    a2 = _completion_prompt_n(server, " and the cats")
    return a1, b, a2


def test_ctx_checkpoint_tree_no_regression():
    # Plain attention model: no checkpoints are created, so the tree must be a no-op and prompt
    # reuse must be identical to the linear path.
    # Single slot (n_slots=1) on purpose: with multiple slots the server's global prompt cache can
    # restore a matching prefix into a fresh slot and mask a broken per-slot reconcile. Pinning to
    # one slot forces all reuse through reconcile_from_tree, so a tree that fails to defer to the
    # linear path (and instead resets to n_past=0) is actually caught here.
    off = ServerPreset.tinyllama2()
    off.temperature = 0.0
    off.n_slots = 1

    on = ServerPreset.tinyllama2()
    on.temperature = 0.0
    on.n_slots = 1
    on.ctx_checkpoints_tree = True
    on.checkpoint_min_step = 1
    on.n_ctx_checkpoints = 16

    try:
        off.start()
        r_off = _branch_scenario_completion(off)
    finally:
        off.stop()

    try:
        on.start()
        r_on = _branch_scenario_completion(on)
    finally:
        on.stop()

    # the tree path must never process more prompt tokens than the linear path ...
    assert r_on[2] <= r_off[2]
    # ... and on a non-checkpointing model it must be exactly identical
    assert r_on == r_off


def _chat_timings(server: ServerProcess, system: str, user: str):
    res = server.make_request("POST", "/v1/chat/completions", data={
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "n_predict": 4,
        "temperature": 0.0,
    })
    assert res.status_code == 200
    t = res.body["timings"]
    return t["prompt_n"], t["cache_n"]


SYSTEM = (
    "You are a careful and helpful assistant. You always think step by step, keep answers short and "
    "factual, never repeat the user question before answering, and consider the full conversation "
    "context before responding to the user in any situation whatsoever on any given day."
)


def _branch_scenario_chat(server: ServerProcess):
    # shared system prefix; user turns A and B diverge; then return to A
    _chat_timings(server, SYSTEM, "Tell me about cats and their behavior in detail")
    _chat_timings(server, SYSTEM, "Tell me about dogs and their behavior in detail")
    a2_prompt_n, a2_cache_n = _chat_timings(server, SYSTEM, "Tell me about cats and their behavior in detail")
    return a2_prompt_n, a2_cache_n


def test_ctx_checkpoint_tree_swa_engages():
    # SWA model: checkpoints are created, so the tree is actually exercised. Returning to branch A
    # after branch B should reuse at least as much as the linear path (and, when the tree keeps the
    # A branch alive, strictly more - i.e. fewer prompt tokens re-evaluated).
    off = ServerPreset.tinygemma3()
    off.temperature = 0.0
    off.n_ctx = 1024
    off.jinja = True
    off.checkpoint_min_step = 1
    off.n_ctx_checkpoints = 16

    on = ServerPreset.tinygemma3()
    on.temperature = 0.0
    on.n_ctx = 1024
    on.jinja = True
    on.checkpoint_min_step = 1
    on.n_ctx_checkpoints = 16
    on.ctx_checkpoints_tree = True

    try:
        off.start()
        off_prompt_n, off_cache_n = _branch_scenario_chat(off)
    finally:
        off.stop()

    try:
        on.start()
        on_prompt_n, on_cache_n = _branch_scenario_chat(on)
    finally:
        on.stop()

    # returning to branch A must not be worse with the tree than without it
    assert on_prompt_n <= off_prompt_n
    assert on_cache_n  >= off_cache_n
    # correctness sanity: some reuse happened on the branch return
    assert on_cache_n > 0
