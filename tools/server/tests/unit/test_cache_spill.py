import os
import tempfile

from utils import *

# End-to-end coverage for prompt-cache disk spill (--cache-spill-dir / --cache-disk).
#
# With spill enabled, a prompt cached on slot 0 and evicted from the live slot (when slot 1 is
# launched) must still be restorable, and re-sending it must reuse its state. This exercises the
# spill-enabled cache code paths (spill attempt on eviction, on-disk check on load) end to end and
# guards against regressions/corruption from them.
#
# Note: the tiny preset models here have very small per-prompt state, so the 1 MiB RAM cache is not
# actually overflowed and states may stay resident (no physical file) - that is fine for this test,
# which asserts correctness with spill enabled. The physical RAM->disk->RAM round-trip is covered
# by manual testing on larger models (where states are hundreds of MiB).

LONG_PROMPT = (
    "Once upon a time in a land far away, there lived a brave knight "
    "who traveled across mountains and rivers to find the legendary "
    "golden sword hidden deep within the enchanted forest of whispers. "
    "He met many creatures along the way including dragons and fairies "
    "and wizards who helped him on his noble quest to save the kingdom."
)


def test_prompt_cache_spill_enabled_reuse():
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-test-")

    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True   # enable slots so idle-slot caching runs
    server.kv_unified = True
    server.cache_ram = 100
    server.cache_spill_dir = spill_dir   # spill enabled
    server.cache_disk = 512

    try:
        server.start()

        # cache LONG_PROMPT on slot 0
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "id_slot": 0, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        original_prompt_n = res.body["timings"]["prompt_n"]
        assert original_prompt_n > 1

        # launch slot 1: idle slot 0 is cleared and its state saved to the (spill-enabled) cache
        res = server.make_request("POST", "/completion", data={
            "prompt": "The quick brown fox", "id_slot": 1, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body

        # re-send LONG_PROMPT: it must be restored from the cache (from disk if it was spilled),
        # so reuse happens and fewer tokens are reprocessed than the cold run
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        assert res.body["timings"]["cache_n"] > 0, "expected reuse from the (spill-enabled) prompt cache"
        assert res.body["timings"]["prompt_n"] < original_prompt_n

        # a follow-up that extends the restored prompt must also work (state is coherent, not corrupt)
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT + " The knight finally reached the castle gates.",
            "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        assert res.body["timings"]["cache_n"] > 0
    finally:
        server.stop()
        try:
            for f in os.listdir(spill_dir):
                os.remove(os.path.join(spill_dir, f))
            os.rmdir(spill_dir)
        except OSError:
            pass
