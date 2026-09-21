import os
import tempfile
import time

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


def test_prompt_cache_spill_written_on_shutdown():
    # A server that is stopped right after its last request must still write that prompt out: the
    # slot holds the only copy until a later task pushes it into the cache, so a clean shutdown has
    # to hand the idle slots over before it drops the contexts. Restarting on the same spill dir
    # then has to reuse the prompt without reprocessing it.
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-stop-")

    def make_server():
        server = ServerPreset.tinyllama2()
        server.n_slots = 1
        server.n_predict = 4
        server.temperature = 0.0
        server.server_slots = True
        server.cache_ram = 100
        server.cache_spill_dir = spill_dir
        server.cache_disk = 512
        return server

    server = make_server()

    try:
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        cold_prompt_n = res.body["timings"]["prompt_n"]
        assert cold_prompt_n > 1

        # nothing else is sent: the prompt lives only in the slot
        server.stop()

        files = os.listdir(spill_dir)
        assert len(files) == 1, f"expected one spill file after a clean stop, got {files}"
        assert os.path.getsize(os.path.join(spill_dir, files[0])) > 0

        # cold start on the same directory must restore it
        server = make_server()
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        assert res.body["timings"]["cache_n"] > 0, "expected reuse of the spilled prompt after a restart"
        assert res.body["timings"]["prompt_n"] < cold_prompt_n
    finally:
        server.stop()
        try:
            for f in os.listdir(spill_dir):
                os.remove(os.path.join(spill_dir, f))
            os.rmdir(spill_dir)
        except OSError:
            pass


def test_prompt_cache_write_behind_while_idle():
    # Write-behind: the cache must reach the disk while the server is idle, not only at shutdown.
    # The blobs stay in RAM as well, so a later eviction needs no I/O and a crash loses nothing.
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-idle-")

    server = ServerPreset.tinyllama2()
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.cache_spill_dir = spill_dir
    server.cache_disk = 512

    try:
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body

        # the write runs on the idle path right after the response is sent
        files = []
        for _ in range(50):
            files = [f for f in os.listdir(spill_dir) if os.path.getsize(os.path.join(spill_dir, f)) > 0]
            if files:
                break
            time.sleep(0.1)

        assert len(files) == 1, f"expected the state on disk while the server is still running, got {files}"

        # a second idle round must not write it again
        res = server.make_request("GET", "/health")
        assert res.status_code == 200, res.body
        time.sleep(0.5)
        assert os.listdir(spill_dir) == files, "a clean state was written a second time"

        # the state is still usable: extending the prompt must reuse it
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


def test_prompt_cache_spill_with_mmproj():
    # An mmproj marks every prompt of the slot as multimodal, also a text-only one. The spill must
    # still take such a prompt: only a prompt that really holds a media chunk stays in RAM.
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-mtmd-")

    server = ServerPreset.tinygemma3()
    server.n_slots = 1
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.cache_ram = 100
    server.cache_spill_dir = spill_dir
    server.cache_disk = 512

    try:
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body

        files = []
        for _ in range(50):
            files = [f for f in os.listdir(spill_dir) if os.path.getsize(os.path.join(spill_dir, f)) > 0]
            if files:
                break
            time.sleep(0.1)

        assert len(files) == 1, f"expected the text-only state on disk, got {files}"

        # the server is still alive and the state is still usable
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


def test_prompt_cache_hit_keeps_the_file():
    # A cache hit hands the state to the slot, but the file holds the very same bytes, so taking it
    # away would only buy a full rewrite at the next idle. The follow-up state grows by a few tokens
    # only, which is below the rewrite step, so the listing must come out of all this unchanged.
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-keep-")

    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    server.cache_ram = 100
    server.cache_spill_dir = spill_dir
    server.cache_disk = 512
    server.cache_min_tokens = 32   # under LONG_PROMPT, over the short prompt and over the follow-up

    try:
        server.start()

        # cache LONG_PROMPT on slot 0, then launch slot 1 so slot 0 is saved to the cache
        for id_slot, prompt in ((0, LONG_PROMPT), (1, "The quick brown fox")):
            res = server.make_request("POST", "/completion", data={
                "prompt": prompt, "id_slot": id_slot, "cache_prompt": True, "temperature": 0.0,
            })
            assert res.status_code == 200, res.body

        # write-behind puts it on disk while the server is idle
        before = []
        for _ in range(50):
            before = sorted(f for f in os.listdir(spill_dir) if os.path.getsize(os.path.join(spill_dir, f)) > 0)
            if before:
                break
            time.sleep(0.1)

        assert len(before) == 1, f"expected the long prompt on disk and the short one skipped, got {before}"

        # re-send it on slot 1: restored from the cache, and the file has to survive that
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "id_slot": 1, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        assert res.body["timings"]["cache_n"] > 0, "expected reuse from the prompt cache"

        time.sleep(0.5)
        assert sorted(os.listdir(spill_dir)) == before, "a cache hit must not rewrite the spill dir"

        # and the state is still coherent after all that
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


def test_prompt_cache_min_tokens_skips_short_prompts():
    # --cache-min-tokens keeps short prompts out of the cache. With write-behind on, anything that
    # does enter the cache reaches the disk while idle, so an empty spill dir means nothing was kept.
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-min-")

    server = ServerPreset.tinyllama2()
    server.n_slots = 2
    server.n_predict = 4
    server.temperature = 0.0
    server.server_slots = True
    server.kv_unified = True
    server.cache_ram = 100
    server.cache_spill_dir = spill_dir
    server.cache_disk = 512
    server.cache_min_tokens = 4096   # far above anything this test sends

    try:
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "id_slot": 0, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body

        # launch slot 1: idle slot 0 would normally be saved to the cache here
        res = server.make_request("POST", "/completion", data={
            "prompt": "The quick brown fox", "id_slot": 1, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body

        time.sleep(0.5)
        assert os.listdir(spill_dir) == [], "short prompts must not enter the cache"
    finally:
        server.stop()
        try:
            for f in os.listdir(spill_dir):
                os.remove(os.path.join(spill_dir, f))
            os.rmdir(spill_dir)
        except OSError:
            pass
