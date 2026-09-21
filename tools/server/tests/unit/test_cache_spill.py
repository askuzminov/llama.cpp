import base64
import os
import requests
import struct
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

IMG_URL = "https://huggingface.co/ggml-org/tinygemma3-GGUF/resolve/main/test/11_truck.png"


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
    # still take such a prompt. See test_prompt_cache_spill_with_image for one that holds media.
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


def test_prompt_cache_spill_with_image():
    # A prompt that really holds an image must reach the disk too. The image is already encoded into
    # the state blob, so the file only carries the chunk id and its token count. A restart on the
    # same directory has to match the same image again and reuse the state.
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-img-")

    img_res = requests.get(IMG_URL)
    img_res.raise_for_status()
    img = base64.b64encode(img_res.content).decode("utf-8")

    # the marker is random per process unless it is pinned, and both servers must use the same one
    marker_env = os.environ.get("LLAMA_MEDIA_MARKER")
    os.environ["LLAMA_MEDIA_MARKER"] = "<__media__>"

    prompt = {
        "prompt_string": "What is this: <__media__>\nAnswer with one word and then explain yourself.",
        "multimodal_data": [ img ],
    }

    def make_server():
        server = ServerPreset.tinygemma3()
        server.n_slots = 1
        server.n_predict = 4
        server.temperature = 0.0
        server.server_slots = True
        server.cache_ram = 100
        server.cache_spill_dir = spill_dir
        server.cache_disk = 512
        # gemma3 is a SWA model: without the full KV a restored state starts past position 0 and the
        # slot has to reprocess the prompt, which would hide what this test is after
        server.swa_full = True
        return server

    server = make_server()

    try:
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": prompt, "cache_prompt": True, "temperature": 0.0, "top_k": 1,
        })
        assert res.status_code == 200, res.body
        cold_prompt_n = res.body["timings"]["prompt_n"]
        assert cold_prompt_n > 1

        files = []
        for _ in range(50):
            files = [f for f in os.listdir(spill_dir) if os.path.getsize(os.path.join(spill_dir, f)) > 0]
            if files:
                break
            time.sleep(0.1)

        assert len(files) == 1, f"expected the image prompt on disk, got {files}"

        # the pixels must not be in there: the image is already encoded into the state blob, so the
        # prompt section carries the token list plus a chunk of a few dozen bytes
        with open(os.path.join(spill_dir, files[0]), "rb") as f:
            magic, _, _, _, _, n_prompt, _, _ = struct.unpack("<8sIIQQQQQ", f.read(56))
        assert magic == b"LCPCACHE"
        assert n_prompt*4 < cold_prompt_n*4 + 1024, f"the prompt section holds {n_prompt*4} bytes"

        server.stop()

        # cold start on the same directory: the image hashes to the same chunk id, so the restored
        # placeholder matches and the prompt is not encoded again
        server = make_server()
        server.start()

        res = server.make_request("POST", "/completion", data={
            "prompt": prompt, "cache_prompt": True, "temperature": 0.0, "top_k": 1,
        })
        assert res.status_code == 200, res.body
        assert res.body["timings"]["cache_n"] > 0, "expected reuse of the spilled image prompt after a restart"
        assert res.body["timings"]["prompt_n"] < cold_prompt_n
    finally:
        server.stop()
        if marker_env is None:
            os.environ.pop("LLAMA_MEDIA_MARKER", None)
        else:
            os.environ["LLAMA_MEDIA_MARKER"] = marker_env
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


def test_prompt_cache_spill_dir_shared_by_two_models():
    # Two models may point at one spill dir. The file name carries the signature of the
    # configuration, so a model only ever reads and drops its own files, and it still finds them
    # after the other model has run there.
    spill_dir = tempfile.mkdtemp(prefix="llama-cache-spill-share-")

    def setup(server):
        server.n_slots = 1
        server.n_predict = 4
        server.temperature = 0.0
        server.server_slots = True
        server.cache_ram = 100
        server.cache_spill_dir = spill_dir
        server.cache_disk = 512
        return server

    def run(server):
        server.start()
        res = server.make_request("POST", "/completion", data={
            "prompt": LONG_PROMPT, "cache_prompt": True, "temperature": 0.0,
        })
        assert res.status_code == 200, res.body
        server.stop()
        return res.body["timings"]

    server = setup(ServerPreset.tinyllama2())

    try:
        cold = run(server)
        assert cold["prompt_n"] > 1

        mine = os.listdir(spill_dir)
        assert len(mine) == 1, f"expected one spill file after a clean stop, got {mine}"

        # the other model spills into the same directory
        server = setup(ServerPreset.stories15m_moe())
        run(server)

        both = sorted(os.listdir(spill_dir))
        assert len(both) == 2, f"each model must keep its own file, got {both}"
        assert mine[0] in both, "the second model dropped the first one's file"

        # back to the first model: its state survived and is still reusable
        server = setup(ServerPreset.tinyllama2())
        warm = run(server)
        assert warm["cache_n"] > 0, "expected reuse after the other model ran on the same dir"
        assert warm["prompt_n"] < cold["prompt_n"]
    finally:
        server.stop()
        try:
            for f in os.listdir(spill_dir):
                os.remove(os.path.join(spill_dir, f))
            os.rmdir(spill_dir)
        except OSError:
            pass
