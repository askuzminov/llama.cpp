import tempfile
import threading
import pytest
import requests
from utils import *

# fork: --pool, llama-server as a router over remote instances of one model (tools/server/fork/server-pool-fork.cpp).
# two instances of the tiny model with one slot each, and a pool in front of them

pool: ServerProcess
nodes: list[ServerProcess]


@pytest.fixture(autouse=True)
def create_pool():
    global pool, nodes
    nodes = []
    for i in range(2):
        n = ServerPreset.tinyllama2()
        n.server_port = n.server_port + 1 + i
        n.n_slots = 1
        nodes.append(n)
    pool = ServerProcess()
    pool.model_hf_repo = None
    pool.model_hf_file = None
    pool.model_alias = None
    yield
    pool.stop()
    for n in nodes:
        n.stop()


def start_pool(options: str = "", urls: dict | None = None, slots: int | None = None, node_options: dict | None = None) -> dict:
    for n in nodes:
        n.start()
    ini = tempfile.NamedTemporaryFile("w", suffix=".ini", delete=False)
    ini.write("[pool]\npoll_ms = 200\n" + options + "\n")
    for i, n in enumerate(nodes):
        url = (urls or {}).get(i, f"http://{n.server_host}:{n.server_port}")
        ini.write(f"[node.n{i}]\nurl = {url}\n" + (node_options or {}).get(i, "") + "\n")
    ini.close()
    pool.pool_file = ini.name
    pool.start()
    return wait_nodes(slots)


def wait_nodes(slots: int | None = None) -> dict:
    deadline = time.time() + 20
    while time.time() < deadline:
        st = pool.make_request("GET", "/pool/status").body
        if st["pool"]["nodes_up"] == len(nodes) and st["pool"]["slots"] == (len(nodes) if slots is None else slots):
            return st
        time.sleep(0.1)
    raise AssertionError(f"the pool did not see its nodes: {[(n['name'], n['status'], n['error']) for n in st['nodes']]}")


def chat(messages: list, n_predict: int = 8, timings: dict | None = None) -> str:
    res = pool.make_request("POST", "/v1/chat/completions", data={
        "messages": messages, "max_tokens": n_predict, "temperature": 0,
    })
    assert res.status_code == 200, res.body
    if timings is not None:
        timings.update(res.body["timings"])
    return res.body["choices"][0]["message"]["content"]


def events(kind: str) -> list:
    return [e for e in pool.make_request("GET", "/pool/status").body["events"] if e["kind"] == kind]


def drain(node: str, on: bool = True):
    res = pool.make_request("POST", "/pool/drain", data={"node": node, "drain": on})
    assert res.status_code == 200, res.body


def share_of(node: str) -> float:
    return next(n for n in pool.make_request("GET", "/pool/status").body["nodes"] if n["name"] == node)["share"]


def wait_links(want: list) -> list:
    # the pool probes a link in its poller, after the node answered
    deadline = time.time() + 10
    while True:
        links = [n["links"][0] for n in pool.make_request("GET", "/pool/status").body["nodes"]]
        if [l["mbps_read"] > 0 and l["mbps_write"] > 0 for l in links] == want:
            return links
        assert time.time() < deadline, links
        time.sleep(0.1)


def wait_share(node: str) -> float:
    # the pool reads the timings when the answer is done, a moment after the client has it
    deadline = time.time() + 5
    while share_of(node) < 0:
        assert time.time() < deadline, f"no share measured on {node}"
        time.sleep(0.05)
    return share_of(node)


LIBRARY = {"role": "system", "content": "You are a careful assistant for a small library. " * 8}


def test_pool_sees_its_nodes():
    st = start_pool()
    assert [n["status"] for n in st["nodes"]] == ["up", "up"]
    assert [n["slots"] for n in st["nodes"]] == [1, 1]
    res = pool.make_request("GET", "/health")
    assert res.status_code == 200
    res = pool.make_request("GET", "/pool")
    assert res.status_code == 200
    assert "llama pool" in res.body


def test_pool_keeps_a_conversation_on_its_node():
    start_pool()
    msgs = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "a first question about cats"}]
    for _ in range(3):
        msgs.append({"role": "assistant", "content": chat(msgs)})
        msgs.append({"role": "user", "content": "and then?"})
    new = events("new")
    keep = events("keep")
    assert len(new) == 1
    assert len(keep) == 2
    assert all(e["node"] == new[0]["node"] for e in keep)
    assert all(e["conv"] == new[0]["conv"] for e in keep)


def test_pool_spreads_new_conversations():
    start_pool()
    ths = [threading.Thread(target=chat, args=([{"role": "user", "content": f"conversation {i}: tell a long story"}], 64))
           for i in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert {e["node"] for e in events("new")} == {"n0", "n1"}


def test_pool_queues_when_every_slot_is_busy():
    start_pool()
    ths = [threading.Thread(target=chat, args=([{"role": "user", "content": f"conversation {i}: tell a long story"}], 64))
           for i in range(4)]
    for t in ths:
        t.start()
    saw_queue = False
    while any(t.is_alive() for t in ths):
        st = pool.make_request("GET", "/pool/status").body
        saw_queue = saw_queue or st["pool"]["queued"] > 0
        assert st["pool"]["inflight"] <= 2
        time.sleep(0.02)
    for t in ths:
        t.join()
    assert saw_queue
    assert len(events("done")) == 4


def test_pool_drain():
    start_pool()
    res = pool.make_request("POST", "/pool/drain", data={"node": "n0", "drain": True})
    assert res.status_code == 200
    for i in range(3):
        chat([{"role": "user", "content": f"drained {i}"}])
    assert {e["node"] for e in events("new")} == {"n1"}


def test_pool_node_down():
    start_pool()
    nodes[0].stop()
    deadline = time.time() + 10
    while pool.make_request("GET", "/pool/status").body["pool"]["nodes_up"] != 1:
        assert time.time() < deadline, "the pool did not notice the stopped node"
        time.sleep(0.1)
    for i in range(2):
        chat([{"role": "user", "content": f"after the stop {i}"}])
    assert {e["node"] for e in events("new")} == {"n1"}


def test_pool_link_fallback():
    # the first url of n1 leads nowhere: the pool takes the second, and goes back to the first once it answers
    dead = nodes[1].server_port + 50
    late = ServerPreset.tinyllama2()
    late.server_port = dead
    late.n_slots = 1
    try:
        st = start_pool("probe_s = 1",
                        urls={1: f"http://127.0.0.1:{dead}, http://{nodes[1].server_host}:{nodes[1].server_port}"})
        n1 = st["nodes"][1]
        assert n1["fallback"] and n1["links"][1]["active"] and n1["links"][0]["error"]
        assert any(e["node"] == "n1" for e in events("link"))
        chat([{"role": "user", "content": "over the fallback"}])
        late.start()
        deadline = time.time() + 10
        while not pool.make_request("GET", "/pool/status").body["nodes"][1]["links"][0]["active"]:
            assert time.time() < deadline, "the pool did not go back to the first link"
            time.sleep(0.1)
        assert "back to a preferred link" in events("link")[0]["detail"]
    finally:
        late.stop()


@pytest.mark.parametrize("spill,min_tokens", [(False, None), (True, None), (True, 100000)])
def test_pool_moves_the_state_of_a_conversation(spill: bool, min_tokens: int | None):
    # both nodes let the pool move states: a conversation of a drained node goes to the other one with its state, which
    # computes only the new messages. the state leaves from RAM, or from its file when it is on disk. --cache-min-tokens
    # above the conversation keeps it off the disk, not from moving
    for n in nodes:
        n.cache_spill_dir = tempfile.mkdtemp() if spill else None
        n.cache_min_tokens = min_tokens
    start_pool("move_tokens = 16\nwait_max = 0\nprefill_tps = 1")
    st = pool.make_request("GET", "/pool/status").body
    assert all(n["migrate"] for n in st["nodes"])
    assert st["nodes"][0]["signature"] == st["nodes"][1]["signature"] != ""
    msgs = [{"role": "user", "content": "once upon a time there was a little girl who " * 12}]
    t1 = {}
    msgs.append({"role": "assistant", "content": chat(msgs, timings=t1)})
    msgs.append({"role": "user", "content": "and then?"})
    first = events("new")[0]["node"]
    pool.make_request("POST", "/pool/drain", data={"node": first, "drain": True})
    t2 = {}
    chat(msgs, timings=t2)
    moved = events("migrate")
    assert len(moved) == 1 and moved[0]["node"] != first, events("move")
    assert spill or "(RAM)" in moved[0]["detail"], moved[0]["detail"]
    # the state holds the first prompt and its answer
    assert t2["cache_n"] >= t1["prompt_n"] + t1["predicted_n"] - 1, (t1, t2)


def test_pool_moves_no_state_with_migrate_0():
    # migrate = 0 switches the moves off: the conversation of a drained node goes to the other one without its state
    # and is computed there from the start
    start_pool("move_tokens = 16\nwait_max = 0\nprefill_tps = 1\nmigrate = 0")
    msgs = [{"role": "user", "content": "once upon a time there was a little girl who " * 12}]
    msgs.append({"role": "assistant", "content": chat(msgs)})
    msgs.append({"role": "user", "content": "and then?"})
    first = events("new")[0]["node"]
    drain(first)
    t = {}
    chat(msgs, timings=t)
    move = events("move")
    assert events("migrate") == [] and len(move) == 1 and move[0]["node"] != first, (move, events("migrate"))
    assert "no state moved" not in move[0]["detail"] and t["cache_n"] == 0, (move, t)


def test_pool_node_checks_what_comes_in():
    # the import takes the chunks of a transfer in order, and only a state of its own model and configuration
    start_pool()
    url = f"http://{nodes[1].server_host}:{nodes[1].server_port}"
    res = requests.post(f"{url}/pool/import/data?id=00ff&offset=4096", data=b"x" * 16)
    assert res.status_code == 400 and "no such transfer" in res.text, res.text
    res = requests.post(f"{url}/pool/import/data?id=00ff&offset=0", data=b"x" * 4096)
    assert res.status_code == 400 and "not a prompt-cache state" in res.text, res.text
    # a real state, sent out of order, then whole with another signature
    msgs = [{"role": "user", "content": "a story about a lighthouse keeper " * 8}]
    msgs.append({"role": "assistant", "content": chat(msgs)})
    first = events("new")[0]["node"]
    src = nodes[int(first[1:])]
    dst = f"http://{nodes[1 - int(first[1:])].server_host}:{nodes[1 - int(first[1:])].server_port}"
    body = {"messages": msgs + [{"role": "user", "content": "and then?"}], "max_tokens": 4}
    ex = src.make_request("POST", "/pool/export", data={"kind": "chat", "body": body}).body
    assert ex["n_reuse"] > 0 and not ex["from_disk"], ex
    data = requests.get(f"http://{src.server_host}:{src.server_port}/pool/export/data?id={ex['id']}&offset=0&size={1 << 26}").content
    assert len(data) == ex["size"]
    res = requests.post(f"{dst}/pool/import/data?id={ex['id']}&offset=0", data=data[:8192])
    assert res.status_code == 200, res.text
    res = requests.post(f"{dst}/pool/import/data?id={ex['id']}&offset=4096", data=data[4096:8192])
    assert res.status_code == 400 and "out of order" in res.text, res.text
    requests.post(f"{dst}/pool/import/data?id={ex['id']}&offset=0", data=data)
    res = requests.post(f"{dst}/pool/import/done", json={"id": ex["id"], "signature": "0123456789abcdef"})
    assert res.status_code == 400 and "signatures differ" in res.text, res.text
    src.make_request("POST", "/pool/export/done", data={"id": ex["id"]})


def test_pool_node_sends_the_checkpoint_the_request_needs():
    # tinygemma3 (SWA) goes back to a checkpoint when a request shares only a prefix with the state: the export sends
    # the newest checkpoint within that prefix, with the base its deltas hang off, and none when the request goes on
    # from the end of the state. the other node then takes what the export promised (n_reuse)
    for i in range(2):
        n = ServerPreset.tinygemma3()
        n.server_port = nodes[0].server_port + 1 + i
        n.no_mmproj = True
        n.n_slots = 1
        n.n_batch = 32               # the prompt has to span several batches to be checkpointed
        n.n_ubatch = 32
        n.n_ctx_checkpoints = -1
        n.checkpoint_min_step = 0
        n.cache_ram_reserve = 0
        nodes[i] = n
        n.start()
    src, dst = nodes
    story = ("Once upon a time in a land far away, there lived a brave knight who traveled across mountains and rivers "
             "to find the legendary golden sword hidden deep within the enchanted forest of whispers. ")
    prompt = story
    for _ in range(3):
        res = src.make_request("POST", "/completion", data={"prompt": prompt, "n_predict": 4, "temperature": 0, "return_tokens": True})
        assert res.status_code == 200, res.body
        prompt = prompt + "The knight rode on past the river and the old stone bridge. "

    def move(body: dict) -> dict:
        ex = src.make_request("POST", "/pool/export", data={"kind": "completion", "body": body}).body
        assert "id" in ex, ex
        data = requests.get(f"http://{src.server_host}:{src.server_port}/pool/export/data?id={ex['id']}&offset=0&size={1 << 26}").content
        assert len(data) == ex["size"]
        res = requests.post(f"http://{dst.server_host}:{dst.server_port}/pool/import/data?id={ex['id']}&offset=0", data=data)
        assert res.status_code == 200, res.text
        res = requests.post(f"http://{dst.server_host}:{dst.server_port}/pool/import/done", json={"id": ex["id"], "signature": ex["signature"]})
        assert res.status_code == 200, res.text
        src.make_request("POST", "/pool/export/done", data={"id": ex["id"]})
        return ex

    # the prompt changed past its first sentences: back to a checkpoint
    body = {"prompt": story + "Then a storm came over the hills and the knight sought shelter in a cave.", "n_predict": 4, "temperature": 0}
    ex = move(body)
    assert 0 < ex["n_reuse"] < ex["n_match"] and 1 <= ex["n_ckpt"] < ex["n_ckpt_all"], ex
    res = dst.make_request("POST", "/completion", data=body)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["cache_n"] == ex["n_reuse"], (ex, res.body["timings"])
    # a longer prefix: the checkpoint is a delta, its base and the deltas before it go along
    body = {"prompt": prompt.rsplit("The knight", 1)[0] + "A wolf howled.", "n_predict": 4, "temperature": 0}
    ex = move(body)
    assert ex["n_ckpt"] >= 2, ex
    res = dst.make_request("POST", "/completion", data=body)
    assert res.status_code == 200, res.body
    assert res.body["timings"]["cache_n"] == ex["n_reuse"], (ex, res.body["timings"])


@pytest.mark.parametrize("stream", [False, True])
def test_pool_measures_the_prefill_speed(stream: bool):
    # a recompute is weighed at the prefill speed of the node: the pool reads it from the timings at the end of the
    # answers (in the last event of a stream too), from the prompts of move_tokens and more; the ini value until then
    st = start_pool("move_tokens = 64\nprefill_tps = 123")
    assert all(n["prefill_tps"] == 0 and n["prefill_cfg"] == 123 for n in st["nodes"]), st["nodes"]
    t = {}
    chat([{"role": "user", "content": "hi"}], timings=t)
    assert t["prompt_n"] < 64, t  # fewer prompt tokens than move_tokens: no sample
    time.sleep(0.3)
    assert all(n["prefill_tps"] == 0 for n in pool.make_request("GET", "/pool/status").body["nodes"])
    msgs = [{"role": "user", "content": "a story about a lighthouse keeper and the storm at sea " * 12}]
    if stream:
        for _ in pool.make_stream_request("POST", "/v1/chat/completions", data={
            "messages": msgs, "max_tokens": 4, "temperature": 0, "stream": True,
        }):
            pass
    else:
        chat(msgs)
    node = events("new")[0]["node"]  # the newest first
    deadline = time.time() + 5
    while True:
        n = next(x for x in pool.make_request("GET", "/pool/status").body["nodes"] if x["name"] == node)
        if n["prefill_tps"] > 0:
            break
        assert time.time() < deadline, n
        time.sleep(0.1)


def test_pool_counts_a_shared_prefix_at_its_measured_share():
    # what a node holds of a conversation from other ones (the system prompt of an agent) counts at the share of such a
    # prefix the node reused, measured from cache_n of its answers. tinyllama2 is a transformer and reuses all of it:
    # a conversation that leaves its node for one that holds its system prompt has less than move_tokens left to
    # compute there, so it goes without its state
    st = start_pool("move_tokens = 64")
    assert [n["share"] for n in st["nodes"]] == [-1, -1]
    drain("n0")
    chat([LIBRARY, {"role": "user", "content": "question one about the books"}])
    chat([LIBRARY, {"role": "user", "content": "question two about the opening hours"}])
    assert wait_share("n1") > 0.9  # the share of the prompt is estimated from the characters
    assert share_of("n0") == -1
    drain("n0", False)
    drain("n1")
    msgs = [LIBRARY, {"role": "user", "content": "a question"}]
    msgs.append({"role": "assistant", "content": chat(msgs)})
    msgs.append({"role": "user", "content": "and then?"})
    msgs.append({"role": "assistant", "content": chat(msgs)})
    msgs.append({"role": "user", "content": "go on"})
    drain("n1", False)
    drain("n0")
    chat(msgs)
    move = events("move")
    assert events("migrate") == [] and len(move) == 1, (move, events("migrate"))
    assert move[0]["node"] == "n1" and "no state moved" not in move[0]["detail"], move


def test_pool_measures_no_reuse_without_a_checkpoint():
    # tinygemma3 (SWA) takes a prefix only up to a checkpoint within it, which a short prompt does not leave there: it
    # reuses nothing of another conversation, the share is 0, and a conversation that leaves its node for it has all of
    # it to compute there, so the pool tries to move its state. a conversation the node holds itself, which the pool
    # forgot with its restart, reuses more than the prefix it shares with the others: no sample
    for i in range(2):
        n = ServerPreset.tinygemma3()
        n.server_port = nodes[0].server_port + 1 + i
        n.no_mmproj = True
        n.n_slots = 1
        nodes[i] = n
    start_pool("move_tokens = 16")
    drain("n1")
    msgs = [LIBRARY, {"role": "user", "content": "question one about the books"}]
    msgs.append({"role": "assistant", "content": chat(msgs)})
    msgs.append({"role": "user", "content": "and what about the old maps in the reading room? " * 3})
    msgs.append({"role": "assistant", "content": chat(msgs)})
    t = {}
    chat([LIBRARY, {"role": "user", "content": "question two about the opening hours"}], timings=t)
    assert t["cache_n"] == 0, t
    assert wait_share("n0") == 0.0
    drain("n1", False)
    drain("n0")
    conv = [LIBRARY, {"role": "user", "content": "a question about the archive"}]
    conv.append({"role": "assistant", "content": chat(conv)})
    conv.append({"role": "user", "content": "and then?"})
    drain("n0", False)
    drain("n1")
    chat(conv)
    move = events("move")
    assert len(move) == 1 and move[0]["node"] == "n0" and "no state moved" in move[0]["detail"], move
    pool.stop()
    pool.start()
    wait_nodes()
    drain("n1")
    chat([LIBRARY, {"role": "user", "content": "question three about the newspapers"}])
    msgs.append({"role": "user", "content": "thanks"})
    chat(msgs, timings=t)
    assert t["cache_n"] > t["prompt_n"], t  # the node held the conversation
    time.sleep(0.5)
    assert share_of("n0") == -1


def test_pool_node_needs_its_key():
    # the move reads and plants the state of a conversation; --api-key on the nodes guards it as every other route: a
    # request without the key gets nothing, the pool sends the key of its ini and moves the conversation
    for n in nodes:
        n.api_key = "node-key"
    start_pool("api_key = node-key\nmove_tokens = 16\nwait_max = 0\nprefill_tps = 1")
    res = requests.post(f"http://{nodes[0].server_host}:{nodes[0].server_port}/pool/export",
                        json={"kind": "chat", "body": {"messages": [{"role": "user", "content": "hi"}]}})
    assert res.status_code == 401, res.text
    msgs = [{"role": "user", "content": "once upon a time there was a little girl who " * 12}]
    msgs.append({"role": "assistant", "content": chat(msgs)})
    msgs.append({"role": "user", "content": "and then?"})
    first = events("new")[0]["node"]
    pool.make_request("POST", "/pool/drain", data={"node": first, "drain": True})
    chat(msgs)
    moved = events("migrate")
    assert len(moved) == 1 and moved[0]["node"] != first, events("move")


def test_pool_probes_its_links():
    # a link of a node that moves conversations is measured when it starts to serve: one chunk of a move from the node
    # and one to it (GET and POST /pool/probe). a node without a prompt cache moves nothing and is not probed
    nodes[1].cache_ram = 0
    start_pool()
    links = wait_links([True, False])
    assert links[1]["mbps_read"] == 0 and links[1]["mbps_write"] == 0, links
    url = f"http://{nodes[0].server_host}:{nodes[0].server_port}"
    res = requests.get(f"{url}/pool/probe?size=1000")
    assert res.status_code == 200 and len(res.content) == 1000
    res = requests.post(f"{url}/pool/probe", data=b"x" * 1000)
    assert res.status_code == 200 and res.json()["size"] == 1000, res.text
    res = requests.get(f"http://{nodes[1].server_host}:{nodes[1].server_port}/pool/probe?size=1000")
    assert res.status_code == 501, res.text


def test_pool_node_without_a_prompt_cache_moves_nothing():
    nodes[0].cache_ram = 0
    st = start_pool()
    assert [n["migrate"] for n in st["nodes"]] == [False, True]
    res = nodes[0].make_request("POST", "/pool/export", data={"kind": "chat", "body": {"messages": [{"role": "user", "content": "hi"}]}})
    assert res.status_code == 501, res.body


def test_pool_names_the_model():
    start_pool("model = pool-model")
    res = pool.make_request("GET", "/v1/models")
    assert res.status_code == 200
    assert [m["id"] for m in res.body["data"]] == ["pool-model"]


def test_pool_router_node_by_alias():
    # a node in router mode (serve.bat) takes the name of its preset or an alias of it: under the alias the pool sees
    # the model of the node loaded, names it in /v1/models and sends it requests
    preset = os.path.join(TMP_DIR, "test_pool_router.ini")
    with open(preset, "w") as f:
        f.write("[tiny-preset]\nhf-repo = ggml-org/test-model-stories260K\nalias = tiny-alias\nload-on-startup = true\n")
    port = nodes[0].server_port
    nodes[0] = ServerPreset.router()
    nodes[0].server_port = port
    nodes[0].models_preset = preset
    st = start_pool("model = pool-model", node_options={0: "model = tiny-alias"})
    assert [n["mismatch"] for n in st["nodes"]] == [False, False]
    wait_links([True, True])  # the probe of the link goes to the instance too
    res = pool.make_request("GET", "/v1/models")
    assert res.status_code == 200
    assert [m["id"] for m in res.body["data"]] == ["pool-model"]
    pool.make_request("POST", "/pool/drain", data={"node": "n1", "drain": True})
    chat([{"role": "user", "content": "which node"}])
    assert events("new")[0]["node"] == "n0"


@pytest.mark.parametrize("same_model", [True, False])
def test_pool_leaves_out_another_model(same_model: bool):
    # a node of another model than the most of the others (parameters, size, type, vocabulary from /pool/node) takes no
    # request; same_model = 0 lets it take them
    nodes[1] = ServerPreset.stories15m_moe()
    nodes[1].server_port = nodes[0].server_port + 1
    nodes[1].n_slots = 1
    st = start_pool("" if same_model else "same_model = 0", slots=1 if same_model else 2)
    assert [n["mismatch"] for n in st["nodes"]] == [False, same_model]
    assert any(e["node"] == "n1" for e in events("mismatch")) == same_model
    for i in range(2):
        chat([{"role": "user", "content": f"which model {i}"}])
    assert {e["node"] for e in events("new")} == ({"n0"} if same_model else {"n0", "n1"})
