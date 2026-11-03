import os
import re
import socket
import tempfile
import threading
import pytest
import requests
from utils import *

# fork: a pool, the routers of a local network that share a model (pool = on in models.ini,
# tools/server/fork/server-pool-fork.cpp). two routers of the tiny model with one slot each, the requests go to the
# first. each finds the other from pool-peers in its preset; two tests find them by the broadcasts. the routers run
# without LLAMA_SERVER_DEBUG_FAKE_TIMING: with it a router sleeps 2 s in every request it proxies, the polls among them

routers: list[ServerProcess]
KEY: str | None = None

TINY = "hf-repo = ggml-org/test-model-stories260K\nload-on-startup = true\npool = on\n"


def preset(name: str, section: str = "tiny", body: str = TINY + "alias = tiny-alias\n") -> str:
    path = os.path.join(TMP_DIR, f"test_pool_{name}.ini")
    with open(path, "w") as f:
        f.write(f"[{section}]\n{body}")
    return path


@pytest.fixture(autouse=True)
def create_routers():
    global routers, KEY
    KEY = None
    routers = []
    for i in range(2):
        r = ServerPreset.router()
        r.server_port = r.server_port + 1 + i
        r.n_slots = 1
        r.n_ctx = 512
        r.n_batch = 32
        r.models_preset = preset(f"r{i}")
        r.env_fork = {"LLAMA_SERVER_DEBUG_FAKE_TIMING": None}
        routers.append(r)
    yield
    for r in routers:
        r.stop()


def headers() -> dict | None:
    return {"Authorization": f"Bearer {KEY}"} if KEY else None


def status(i: int = 0) -> dict:
    res = routers[i].make_request("GET", "/pool/status", headers=headers())
    assert res.status_code == 200, res.body
    return res.body


def with_pool(r: ServerProcess, options: str):
    # the options of the pool go into the preset of the router, in [*] for every model of it: "move_tokens=16
    # wait_max=0" is pool-move-tokens = 16 and pool-wait-max = 0
    lines = ""
    for item in options.split():
        k, v = item.split("=", 1)
        lines += f"pool-{k.replace('_', '-')} = {v}\n"
    with open(r.models_preset) as f:
        text = f.read()
    text = "[*]\n" + lines + text[4:] if text.startswith("[*]\n") else "[*]\n" + lines + "\n" + text
    with open(r.models_preset, "w") as f:
        f.write(text)


def start_pool(options: str = "", slots: int | None = None, broadcast_port: int | None = None) -> dict:
    ports = [r.server_port for r in routers]
    for i, r in enumerate(routers):
        if broadcast_port is None:
            find = "discover=0 peers=" + ",".join(f"127.0.0.1:{p}" for j, p in enumerate(ports) if j != i)
        else:
            find = f"port={broadcast_port}"
        with_pool(r, f"name=n{i} {find} poll_ms=200 {options}")
        r.start()
    return wait_nodes(slots)


def wait_nodes(slots: int | None = None) -> dict:
    # every router sees every node up
    deadline = time.time() + 30
    while time.time() < deadline:
        sts = [status(i) for i in range(len(routers))]
        want = len(routers) if slots is None else slots
        if all(st["pool"]["nodes_up"] == len(routers) and st["pool"]["slots"] == want for st in sts):
            return sts[0]
        time.sleep(0.1)
    raise AssertionError(f"the routers did not see the pool: {[[(n['name'], n['status'], n['error']) for n in st['nodes']] for st in sts]}")


def chat(messages: list, n_predict: int = 8, timings: dict | None = None, model: str = "tiny", via: int = 0) -> str:
    res = routers[via].make_request("POST", "/v1/chat/completions", headers=headers(), data={
        "model": model, "messages": messages, "max_tokens": n_predict, "temperature": 0,
    })
    assert res.status_code == 200, res.body
    if timings is not None:
        timings.update(res.body["timings"])
    return res.body["choices"][0]["message"]["content"]


def events(kind: str, via: int = 0) -> list:
    return [e for e in status(via)["events"] if e["kind"] == kind]


def drain(node: str, on: bool = True):
    res = routers[0].make_request("POST", "/pool/drain", headers=headers(), data={"node": node, "drain": on})
    assert res.status_code == 200, res.body


def node_of(name: str, via: int = 0) -> dict:
    return next(n for n in status(via)["nodes"] if n["name"] == name)


def wait_share(node: str) -> float:
    # the pool reads the timings when the answer is done, a moment after the client has it
    deadline = time.time() + 5
    while node_of(node)["share"] < 0:
        assert time.time() < deadline, f"no share measured on {node}"
        time.sleep(0.05)
    return node_of(node)["share"]


def lan_ip() -> str | None:
    # an address of this machine on a network (to skip the broadcast tests without one): a UDP socket only picks the
    # route, nothing is sent
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        return None if ip.startswith("127.") else ip
    except OSError:
        return None
    finally:
        s.close()


LIBRARY = {"role": "system", "content": "You are a careful assistant for a small library. " * 8}


def test_pool_sees_its_nodes():
    st = start_pool()
    assert [(n["name"], n["status"], n["self"]) for n in st["nodes"]] == [("n0", "up", True), ("n1", "up", False)]
    assert [n["slots"] for n in st["nodes"]] == [1, 1]
    assert [n["name"] for n in status(1)["nodes"]] == ["n1", "n0"]
    assert st["config"]["names"] == ["tiny", "tiny-alias"]
    assert any(e["node"] == "n1" for e in events("found"))
    res = routers[0].make_request("GET", "/pool")
    assert res.status_code == 200
    assert "llama pool" in res.body


def test_pool_stops_on_a_bad_option():
    # an option of the pool that is not a number stops the start of the router, the log names the key
    with_pool(routers[0], "move_tokens=many")
    routers[0].log_path = os.path.join(TMP_DIR, "test_pool_bad_option.log")
    with pytest.raises(RuntimeError):
        routers[0].start()
    with open(routers[0].log_path) as f:
        assert "pool-move-tokens = many: not a number" in f.read()


def test_pool_loads_the_model_on_demand():
    # the pool polls without loading a model and sends requests only to the nodes that have it loaded: the requests
    # for the model make every router of the pool with room for it load it, they wait in the pool meanwhile
    for i, r in enumerate(routers):
        r.models_preset = preset(f"r{i}", body="hf-repo = ggml-org/test-model-stories260K\npool = on\nalias = tiny-alias\n")
    ports = [r.server_port for r in routers]
    for i, r in enumerate(routers):
        with_pool(r, f"name=n{i} discover=0 peers=127.0.0.1:{ports[1 - i]} poll_ms=200")
        r.start()
    deadline = time.time() + 20
    while [n["status"] for n in status()["nodes"]] != ["unloaded", "unloaded"]:
        assert time.time() < deadline, status()["nodes"]
        time.sleep(0.1)
    time.sleep(1)
    assert events("load") == []  # no request, no load
    chat([{"role": "user", "content": "the first request loads the model"}])
    deadline = time.time() + 20
    while status()["pool"]["nodes_up"] != 2:
        assert time.time() < deadline, status()["nodes"]
        time.sleep(0.1)
    assert sorted(e["node"] for e in events("load")) == ["n0", "n1"], events("load")


def test_pool_shows_a_failed_load():
    # a model that fails to load leaves its node unloaded, the card of the node says why; the load is not asked again
    routers[1].models_preset = preset("r1", body="model = /no/such/model.gguf\npool = on\nalias = tiny-alias\n")
    ports = [r.server_port for r in routers]  # the second node never comes up: no wait_nodes
    for i, r in enumerate(routers):
        with_pool(r, f"name=n{i} discover=0 peers=127.0.0.1:{ports[1 - i]} poll_ms=200")
        r.start()
    deadline = time.time() + 20
    while not any(n["name"] == "n1" for n in status()["nodes"]) or status()["pool"]["nodes_up"] != 1:
        assert time.time() < deadline, status()["nodes"]
        time.sleep(0.1)
    chat([{"role": "user", "content": "on the node that works"}])
    deadline = time.time() + 20
    while "failed to load" not in node_of("n1")["error"]:
        assert time.time() < deadline, node_of("n1")
        time.sleep(0.1)
    assert node_of("n1")["status"] == "unloaded"
    chat([{"role": "user", "content": "again on the node that works"}])
    assert [e["node"] for e in events("new")] == ["n0", "n0"]
    assert [e["node"] for e in events("load")] == ["n1"], events("load")


def test_pool_shares_two_models():
    # two models with pool = on: a pool for each, the requests for either go through its own; one discovery finds the
    # routers for both, the dashboard shows one pool by ?model=
    two = (TINY + "alias = tiny-alias\n" + "\n[moe]\nhf-repo = ggml-org/stories15M_MOE\nhf-file = stories15M_MOE-F16.gguf\n"
           "load-on-startup = true\npool = on\n")
    for i, r in enumerate(routers):
        r.models_preset = preset(f"r{i}", body=two)
    start_pool(slots=2)
    deadline = time.time() + 20
    while status(0)["pool"]["nodes_up"] != 2 or routers[0].make_request("GET", "/pool/status?model=moe").body["pool"]["nodes_up"] != 2:
        assert time.time() < deadline
        time.sleep(0.1)
    st = routers[0].make_request("GET", "/pool/status?model=moe").body
    assert sorted(st["pools"]) == ["moe", "tiny"] and st["config"]["model"] == "moe"
    assert [n["name"] for n in st["nodes"]] == ["n0", "n1"]
    chat([{"role": "user", "content": "a tiny question"}], model="tiny")
    chat([{"role": "user", "content": "a moe question"}], model="moe")
    assert len(events("new")) == 1
    assert len([e for e in st["events"] if e["kind"] == "conflict"]) == 0
    moe_new = [e for e in routers[0].make_request("GET", "/pool/status?model=moe").body["events"] if e["kind"] == "new"]
    assert len(moe_new) == 1
    res = routers[0].make_request("GET", "/pool/status?model=nope")
    assert res.status_code == 400


def test_pool_takes_the_model_under_every_name():
    # a request for the model under its name or an alias goes through the pool, a request for any other model stays
    # with the router
    start_pool()
    chat([{"role": "user", "content": "by the name"}], model="tiny")
    chat([{"role": "user", "content": "by the alias"}], model="tiny-alias")
    assert len(events("new")) == 2
    res = routers[0].make_request("POST", "/v1/chat/completions", data={
        "model": "another", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 2,
    })
    assert res.status_code == 400 and "not found" in res.body["error"]["message"], res.body
    assert len(events("new")) == 2


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
    # a done event tells the prompt and the answer of the request from its timings
    done = events("done")
    assert len(done) == 3 and all(re.match(r"HTTP 200, [\d.]+ s: prompt \d+ tok [\d.]+ s, answer 8 tok [\d.]+ s \([\d.]+ t/s\)",
                                           e["detail"]) for e in done), done
    # the node reports what its model instance computed (its metrics) and the sensors of its GPUs (none here)
    deadline = time.time() + 5
    while (node_of(new[0]["node"])["stats"] or {}).get("gen_tokens", 0) < 24:
        assert time.time() < deadline, node_of(new[0]["node"])["stats"]
        time.sleep(0.05)
    st = node_of(new[0]["node"])["stats"]
    assert st["prompt_tokens"] > 0 and st["uptime_s"] > 0 and st["gen_s"] > 0, st
    assert isinstance(node_of(new[0]["node"])["gpus"], list)


def test_pool_spreads_new_conversations():
    start_pool()
    ths = [threading.Thread(target=chat, args=([{"role": "user", "content": f"conversation {i}: tell a long story"}], 64))
           for i in range(2)]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    assert {e["node"] for e in events("new")} == {"n0", "n1"}


def test_pool_spreads_new_conversations_before_the_cache():
    # a new conversation goes to the node with the most free slots by weight; the shared prefix another node has cached
    # (the system prompt of an agent) decides only between nodes as free. n0: 2 slots, the system prompt cached, one
    # slot busy; n1: 1 slot, free - the new conversation goes to n1 (with the cache first it went to n0)
    routers[0].n_slots = 2
    routers[0].n_ctx = 2048
    start_pool("move_tokens=64", slots=3)
    drain("n1")

    def on_slot_0(messages: list) -> str:  # both in the slot that holds the system prompt, so that n0 measures its share
        res = routers[0].make_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": messages, "max_tokens": 8, "temperature": 0, "id_slot": 0,
        })
        assert res.status_code == 200, res.body
        return res.body["choices"][0]["message"]["content"]

    a = [LIBRARY, {"role": "user", "content": "question one about the books"}]
    a.append({"role": "assistant", "content": on_slot_0(a)})
    on_slot_0([LIBRARY, {"role": "user", "content": "question two about the opening hours"}])
    assert wait_share("n0") > 0.9  # n0 holds the system prompt and takes all of it
    drain("n1", False)
    a.append({"role": "user", "content": "tell a very long story about it"})

    def long_answer():  # goes on on n0, the node of its conversation, and keeps one of its slots busy
        res = routers[0].make_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": a, "max_tokens": 700, "temperature": 0, "ignore_eos": True,
        })
        assert res.status_code == 200, res.body

    t = threading.Thread(target=long_answer)
    t.start()
    deadline = time.time() + 10
    while node_of("n0")["inflight"] < 1:
        assert time.time() < deadline
        time.sleep(0.005)
    chat([LIBRARY, {"role": "user", "content": "question three about the newspapers"}])
    t.join()
    assert events("new")[0]["node"] == "n1", events("new")


@pytest.mark.parametrize("rebalance", [True, False])
def test_pool_rebalances_a_crowded_node(rebalance: bool):
    # a conversation leaves its node when that node runs clearly more requests than a free one and the move is cheap:
    # n0 (3 slots) runs 2 long answers, n1 stands - the next turn of a conversation of n0 goes to n1 with its state.
    # pool-rebalance-s = 0 keeps it on n0
    routers[0].n_slots = 3
    routers[0].n_ctx = 3072
    for r in routers:
        r.kv_unified = True  # one KV stream on both: the same cache signature, the state can move
    start_pool("move_tokens=16 prefill_tps=1" + ("" if rebalance else " rebalance_s=0"), slots=4)
    drain("n1")
    conv = [{"role": "user", "content": "once upon a time there was a little girl who " * 6}]
    conv.append({"role": "assistant", "content": chat(conv)})
    conv.append({"role": "user", "content": "and then?"})

    def long_answer(i: int):
        res = routers[0].make_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": [{"role": "user", "content": f"a long story number {i}"}], "max_tokens": 1000,
            "temperature": 0, "ignore_eos": True,
        })
        assert res.status_code == 200, res.body

    ths = [threading.Thread(target=long_answer, args=(i,)) for i in range(2)]
    for t in ths:
        t.start()
    deadline = time.time() + 10
    while node_of("n0")["inflight"] < 2:
        assert time.time() < deadline
        time.sleep(0.005)
    drain("n1", False)
    chat(conv)
    for t in ths:
        t.join()
    if rebalance:
        moved = events("migrate")
        assert len(moved) == 1 and moved[0]["node"] == "n1", (moved, events("move"), events("keep"))
    else:
        assert events("migrate") == [] and events("keep")[0]["node"] == "n0", (events("keep"), events("move"))


@pytest.mark.parametrize("long_message", [True, False])
def test_pool_moves_a_long_prompt_to_a_quiet_node(long_message: bool):
    # a turn with a long new part to read (pool-move-tokens and more) leaves a node where another request runs for one
    # where nothing runs: there its prompt pauses nobody (LLAMA_PREFILL_EXCLUSIVE). n0 (2 slots) runs one long answer,
    # which is not crowded (2 streams with it against 1); a long next turn goes to n1 with its state, a short one stays
    routers[0].n_slots = 2
    routers[0].n_ctx = 4096
    for r in routers:
        r.kv_unified = True  # one KV stream on both: the same cache signature, the state can move
    start_pool("move_tokens=16 prefill_tps=1", slots=3)
    drain("n1")
    conv = [{"role": "user", "content": "once upon a time there was a little girl who " * 6}]
    conv.append({"role": "assistant", "content": chat(conv)})
    conv.append({"role": "user", "content": "and then she went into the dark forest to find " * 6 if long_message else "and then?"})

    def long_answer():
        res = routers[0].make_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": [{"role": "user", "content": "a long story"}], "max_tokens": 1500,
            "temperature": 0, "ignore_eos": True,
        })
        assert res.status_code == 200, res.body

    t = threading.Thread(target=long_answer)
    t.start()
    deadline = time.time() + 10
    while node_of("n0")["inflight"] < 1:
        assert time.time() < deadline
        time.sleep(0.005)
    drain("n1", False)
    chat(conv)
    t.join()
    if long_message:
        moved = events("migrate")
        assert len(moved) == 1 and moved[0]["node"] == "n1", (moved, events("move"), events("keep"))
    else:
        assert events("migrate") == [] and events("keep")[0]["node"] == "n0", (events("keep"), events("move"))


@pytest.mark.parametrize("quiet_wait,max_tokens", [(30, 1000), (0, 1000), (30, 8)])
def test_pool_waits_with_a_long_prompt_for_a_quiet_node(quiet_wait: int, max_tokens: int):
    # a long prompt for a node where a request runs waits until that request ends (at most pool-quiet-wait-s), instead
    # of pausing its answer: n0 (2 slots, n1 out) runs one long answer, a new conversation with a long prompt comes. it
    # asks for a long answer: one for a short answer (pool-short-tokens) does not wait
    routers[0].n_slots = 2
    routers[0].n_ctx = 4096
    start_pool(f"move_tokens=16 prefill_tps=1 quiet_wait_s={quiet_wait}", slots=3)
    drain("n1")
    done = {}

    def long_answer():
        res = routers[0].make_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": [{"role": "user", "content": "a long story"}], "max_tokens": 1500,
            "temperature": 0, "ignore_eos": True,
        })
        assert res.status_code == 200, res.body
        done["long"] = time.time()

    t = threading.Thread(target=long_answer)
    t.start()
    deadline = time.time() + 10
    while node_of("n0")["inflight"] < 1:
        assert time.time() < deadline
        time.sleep(0.005)
    res = routers[0].make_request("POST", "/v1/chat/completions", data={
        "model": "tiny", "messages": [{"role": "user", "content": "once upon a time there was a little girl who " * 6}],
        "max_tokens": max_tokens, "temperature": 0,
    })
    assert res.status_code == 200, res.body
    done["new"] = time.time()
    t.join()
    if quiet_wait and max_tokens > 512:
        assert done["new"] > done["long"], done
    else:
        assert done["new"] < done["long"], done



@pytest.mark.parametrize("repeat", [60, 3])
def test_pool_sends_no_prompt_that_has_no_room_in_the_kv(repeat: int):
    # a node with a unified KV cache takes a task only when its prompt fits next to what the running slots hold with
    # their answer reserve, else the task waits inside the node. the node reports that room (GET /pool/node) and the pool
    # sends such a prompt to a node with room instead. n0: 2 slots in a unified cache of 4096 cells, an answer reserve
    # of 1500 (LLAMA_SLOT_ANSWER_ROOM); a long answer there (a prompt of ~1760 tokens) holds ~3260 cells, so the next
    # turn of a conversation of n0 (~920 tokens, its answer of 8 tokens reserves 8, as on the node) has a free slot there
    # but no room: it goes to n1. a short one (~60 tokens) fits and stays
    routers[0].n_slots = 2
    routers[0].kv_unified = True
    routers[0].env_fork = {"LLAMA_SERVER_DEBUG_FAKE_TIMING": None, "LLAMA_SLOT_ANSWER_ROOM": "1500"}
    for r in routers:
        r.n_ctx = 4096
        r.n_batch = 512
    start_pool("move_tokens=16 wait_max=0", slots=3)
    drain("n1")
    conv = [{"role": "user", "content": "once upon a time there was a little girl who " * repeat}]
    conv.append({"role": "assistant", "content": chat(conv)})
    conv.append({"role": "user", "content": "and then?"})

    def long_answer():
        res = routers[0].make_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": [{"role": "user", "content": "a long story " * 250}], "max_tokens": 2000,
            "temperature": 0, "ignore_eos": True,
        })
        assert res.status_code == 200, res.body

    t = threading.Thread(target=long_answer)
    t.start()
    deadline = time.time() + 10
    while (node_of("n0")["room"] or {}).get("held", 0) < 3000:
        assert time.time() < deadline, node_of("n0")["room"]
        time.sleep(0.01)
    drain("n1", False)
    chat(conv)
    t.join()
    went = [e for e in events("move") + events("migrate") + events("keep") if e["conv"] == events("new")[-1]["conv"]]
    assert len(went) == 1 and went[0]["node"] == ("n1" if repeat == 60 else "n0"), (went, node_of("n0")["room"])
    assert node_of("n1")["room"] is None  # one slot, no unified cache: no room logic



@pytest.mark.parametrize("short_tokens", [100, 0])
def test_pool_puts_a_short_answer_first(short_tokens: int):
    # a request for a short answer (max_tokens up to pool-short-tokens: the permission check an agent waits for) goes
    # ahead of an older one for a long answer: n0 alone (n1 out, one slot) runs a long answer, a request for 300 tokens
    # queues, then one for 8; the second runs first. with pool-short-tokens = 0 the older one goes first
    routers[0].n_ctx = 2048
    start_pool(f"short_tokens={short_tokens}")
    drain("n1")
    done = {}

    def ask(name: str, content: str, max_tokens: int):
        res = routers[0].make_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "ignore_eos": True,
        })
        assert res.status_code == 200, res.body
        done[name] = time.time()

    def queued(n: int):
        deadline = time.time() + 10
        while status()["pool"]["queued"] < n:
            assert time.time() < deadline, status()["pool"]
            time.sleep(0.005)

    a = threading.Thread(target=ask, args=("a", "a long story", 1500))
    a.start()
    deadline = time.time() + 10
    while node_of("n0")["inflight"] < 1:
        assert time.time() < deadline
        time.sleep(0.005)
    b = threading.Thread(target=ask, args=("b", "another long story", 300))
    b.start()
    queued(1)
    c = threading.Thread(target=ask, args=("c", "is this allowed?", 8))
    c.start()
    queued(2)
    for t in (a, b, c):
        t.join()
    assert done["a"] < min(done["b"], done["c"]), done
    if short_tokens:
        assert done["c"] < done["b"], done
    else:
        assert done["b"] < done["c"], done

def test_pool_queues_when_every_slot_is_busy():
    start_pool()
    ths = [threading.Thread(target=chat, args=([{"role": "user", "content": f"conversation {i}: tell a long story"}], 256))
           for i in range(4)]
    for t in ths:
        t.start()
    saw_queue = False
    while any(t.is_alive() for t in ths):
        st = status()
        saw_queue = saw_queue or st["pool"]["queued"] > 0
        assert st["pool"]["inflight"] <= 2
        time.sleep(0.02)
    for t in ths:
        t.join()
    assert saw_queue
    assert len(events("done")) == 4


def test_pool_any_router_takes_requests():
    # the second router takes requests for the model as well; on a node the requests of the other router count as busy
    start_pool()
    chat([{"role": "user", "content": "through the second router"}], via=1)
    assert len(events("new", via=1)) == 1 and events("new", via=0) == []


def test_pool_drain():
    start_pool()
    drain("n0")
    for i in range(3):
        chat([{"role": "user", "content": f"drained {i}"}])
    assert {e["node"] for e in events("new")} == {"n1"}


def test_pool_node_down():
    start_pool()
    routers[1].stop()
    deadline = time.time() + 10
    while status()["pool"]["nodes_up"] != 1:
        assert time.time() < deadline, "the pool did not notice the stopped router"
        time.sleep(0.1)
    for i in range(2):
        chat([{"role": "user", "content": f"after the stop {i}"}])
    assert {e["node"] for e in events("new")} == {"n0"}


def test_pool_measures_two_links_of_a_node():
    # a router reached at two addresses (a cable and the WiFi; here 127.0.0.1 and localhost) has a link at each: both
    # are measured, one serves
    p1 = routers[1].server_port
    with_pool(routers[0], f"name=n0 discover=0 peers=127.0.0.1:{p1},localhost:{p1} poll_ms=200")
    with_pool(routers[1], f"name=n1 discover=0 peers=127.0.0.1:{routers[0].server_port} poll_ms=200")
    for r in routers:
        r.start()
    deadline = time.time() + 15
    while True:
        links = node_of("n1")["links"] if any(n["name"] == "n1" for n in status()["nodes"]) else []
        if len(links) == 2 and all(l["mbps_read"] > 0 and l["mbps_write"] > 0 for l in links):
            break
        assert time.time() < deadline, links
        time.sleep(0.1)
    assert sorted(l["url"] for l in links) == sorted([f"http://127.0.0.1:{p1}", f"http://localhost:{p1}"])
    assert sum(l["active"] for l in links) == 1
    assert any("another link" in e["detail"] or e["kind"] == "link+" for e in status()["events"])


def test_pool_finds_its_routers_by_broadcast():
    if lan_ip() is None:
        pytest.skip("no network to broadcast on")
    for r in routers:
        r.server_host = "0.0.0.0"
    st = start_pool(broadcast_port=routers[0].server_port + 900)
    n1 = next(n for n in st["nodes"] if n["name"] == "n1")
    assert "(broadcast)" in next(e for e in events("found") if e["node"] == "n1")["detail"]
    assert n1["status"] == "up"
    chat([{"role": "user", "content": "found by a broadcast"}])


def test_pool_broadcast_needs_the_same_key():
    # with --api-key a router takes another one only when it answered the challenge of its broadcast with the hash of
    # the same key: another key, no node
    global KEY
    if lan_ip() is None:
        pytest.skip("no network to broadcast on")
    routers.append(ServerPreset.router())
    routers[2].server_port = routers[0].server_port + 3
    routers[2].n_slots = 1
    routers[2].n_ctx = 512
    routers[2].models_preset = preset("r2")
    routers[2].env_fork = {"LLAMA_SERVER_DEBUG_FAKE_TIMING": None}
    for i, r in enumerate(routers):
        r.server_host = "0.0.0.0"
        r.api_key = "other-key" if i == 2 else "pool-key"
        with_pool(r, f"name=n{i} port={routers[0].server_port + 901} poll_ms=200")
        r.start()
    KEY = "pool-key"
    deadline = time.time() + 20
    while status()["pool"]["nodes_up"] < 2:
        assert time.time() < deadline, status()["nodes"]
        time.sleep(0.1)
    time.sleep(5)  # a few rounds of broadcasts more
    assert [n["name"] for n in status()["nodes"]] == ["n0", "n1"]
    KEY = "other-key"
    assert [n["name"] for n in status(2)["nodes"]] == ["n2"]


@pytest.mark.parametrize("spill,min_tokens", [(False, None), (True, None), (True, 100000)])
def test_pool_moves_the_state_of_a_conversation(spill: bool, min_tokens: int | None):
    # a conversation of a drained node goes to the other one with its state, which that node reads from the drained one
    # itself, and computes only the new messages. the state leaves from RAM, or from its file when it is on disk.
    # --cache-min-tokens above the conversation keeps it off the disk, not from moving
    for r in routers:
        r.cache_spill_dir = tempfile.mkdtemp() if spill else None
        r.cache_min_tokens = min_tokens
    st = start_pool("move_tokens=16 wait_max=0 prefill_tps=1")
    assert all(n["migrate"] for n in st["nodes"])
    assert st["nodes"][0]["signature"] == st["nodes"][1]["signature"] != ""
    msgs = [{"role": "user", "content": "once upon a time there was a little girl who " * 12}]
    t1 = {}
    msgs.append({"role": "assistant", "content": chat(msgs, timings=t1)})
    msgs.append({"role": "user", "content": "and then?"})
    first = events("new")[0]["node"]
    drain(first)
    t2 = {}
    chat(msgs, timings=t2)
    moved = events("migrate")
    assert len(moved) == 1 and moved[0]["node"] != first, events("move")
    assert spill or "(RAM)" in moved[0]["detail"], moved[0]["detail"]
    # the state holds the first prompt and its answer
    assert t2["cache_n"] >= t1["prompt_n"] + t1["predicted_n"] - 1, (t1, t2)


def test_pool_exports_no_state_the_estimate_rejects():
    # the export copies the whole state on its node, so the pool asks for none when the moves so far say computing the
    # difference is shorter: the first move of n0 exports and drops the state (prefill_tps so high that computing is
    # always shorter), which gives the bytes per token; the second does not export
    start_pool("move_tokens=16 wait_max=0 prefill_tps=100000000")
    drain("n1")
    convs = []
    for i in range(2):
        msgs = [{"role": "user", "content": f"story {i}: once upon a time there was a little girl who " * 12}]
        msgs.append({"role": "assistant", "content": chat(msgs)})
        msgs.append({"role": "user", "content": "and then?"})
        convs.append(msgs)
    drain("n1", False)
    drain("n0")
    for msgs in convs:
        chat(msgs)
    move = events("move")
    move.reverse()
    assert len(move) == 2 and all(e["node"] == "n1" for e in move), move
    assert "no state moved: the move of" in move[0]["detail"] and "(estimate)" not in move[0]["detail"], move
    assert "no state moved: the move of" in move[1]["detail"] and "(estimate)" in move[1]["detail"], move


@pytest.mark.parametrize("first", [0, 1])
def test_pool_moves_between_stream_counts(first: int):
    # the state of a conversation moves between nodes of other KV stream counts: n0 runs 2 slots in 2 streams (no
    # --kv-unified), n1 one slot in one. the signatures are the same, and a state restores into the stream of its slot
    routers[0].n_slots = 2
    routers[0].n_ctx = 1024
    st = start_pool("move_tokens=16 wait_max=0 prefill_tps=1", slots=3)
    assert st["nodes"][0]["signature"] == st["nodes"][1]["signature"] != ""
    other = f"n{1 - first}"
    drain(other)  # the conversation starts on n<first>
    msgs = [{"role": "user", "content": "once upon a time there was a little girl who " * 12}]
    t1 = {}
    msgs.append({"role": "assistant", "content": chat(msgs, timings=t1)})
    msgs.append({"role": "user", "content": "and then?"})
    drain(other, False)
    drain(f"n{first}")
    t2 = {}
    chat(msgs, timings=t2)
    moved = events("migrate")
    assert len(moved) == 1 and moved[0]["node"] == other, (events("move"), moved)
    assert t2["cache_n"] >= t1["prompt_n"] + t1["predicted_n"] - 1, (t1, t2)


def test_pool_moves_no_state_with_migrate_0():
    # migrate=0 switches the moves off: the conversation of a drained node goes to the other one without its state and
    # is computed there from the start
    start_pool("move_tokens=16 wait_max=0 prefill_tps=1 migrate=0")
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


def test_pool_pull_needs_a_known_node():
    # POST /pool/pull reads a state only from a node of the pool of the model it names
    start_pool()
    res = routers[1].make_request("POST", "/pool/pull", data={"from": "nowhere:1", "id": "00ff", "size": 10, "model": "tiny"})
    assert res.status_code == 400 and "no node nowhere:1" in res.body["error"]["message"], res.body
    res = routers[1].make_request("POST", "/pool/pull", data={"from": "nowhere:1", "id": "00ff", "size": 10})
    assert res.status_code == 400 and "no pool of the model" in res.body["error"]["message"], res.body
    res = routers[0].make_request("POST", "/pool/drain", data={"node": "nowhere", "drain": True})
    assert res.status_code == 400 and "no node nowhere" in res.body["error"]["message"], res.body


@pytest.mark.parametrize("stream", [False, True])
def test_pool_measures_the_prefill_speed(stream: bool):
    # a recompute is weighed at the prefill speed of the node: the pool reads it from the timings at the end of the
    # answers (in the last event of a stream too), from the prompts of move_tokens and more; the default until then
    st = start_pool("move_tokens=64 prefill_tps=123")
    assert all(n["prefill_tps"] == 0 and n["prefill_cfg"] == 123 for n in st["nodes"]), st["nodes"]
    t = {}
    chat([{"role": "user", "content": "hi"}], timings=t)
    assert t["prompt_n"] < 64, t  # fewer prompt tokens than move_tokens: no sample
    time.sleep(0.3)
    assert all(n["prefill_tps"] == 0 for n in status()["nodes"])
    msgs = [{"role": "user", "content": "a story about a lighthouse keeper and the storm at sea " * 12}]
    if stream:
        for _ in routers[0].make_stream_request("POST", "/v1/chat/completions", data={
            "model": "tiny", "messages": msgs, "max_tokens": 4, "temperature": 0, "stream": True,
        }):
            pass
    else:
        chat(msgs)
    node = events("new")[0]["node"]  # the newest first
    deadline = time.time() + 5
    while node_of(node)["prefill_tps"] <= 0:
        assert time.time() < deadline, node_of(node)
        time.sleep(0.1)


def test_pool_measures_the_decode_speed():
    # the new conversations are shared out by the decode speed of the nodes, measured from the answers of 16 tokens
    # and more
    start_pool()
    chat([{"role": "user", "content": "a short one"}], n_predict=4)
    time.sleep(0.3)
    assert all(n["decode_tps"] == 0 for n in status()["nodes"])
    chat([{"role": "user", "content": "tell a long story"}], n_predict=32)
    node = events("new")[0]["node"]
    deadline = time.time() + 5
    while node_of(node)["decode_tps"] <= 0:
        assert time.time() < deadline, node_of(node)
        time.sleep(0.1)


def test_pool_counts_a_shared_prefix_at_its_measured_share():
    # what a node holds of a conversation from other ones (the system prompt of an agent) counts at the share of such a
    # prefix the node reused, measured from cache_n of its answers. tinyllama2 is a transformer and reuses all of it:
    # a conversation that leaves its node for one that holds its system prompt has less than move_tokens left to
    # compute there, so it goes without its state
    st = start_pool("move_tokens=64")
    assert [n["share"] for n in st["nodes"]] == [-1, -1]
    drain("n0")
    chat([LIBRARY, {"role": "user", "content": "question one about the books"}])
    chat([LIBRARY, {"role": "user", "content": "question two about the opening hours"}])
    assert wait_share("n1") > 0.9  # the share of the prompt is estimated from the characters
    assert node_of("n0")["share"] == -1
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
    # forgot with the restart of its router, reuses more than the prefix it shares with the others: no sample
    for i, r in enumerate(routers):
        r.models_preset = preset(f"g{i}", body="hf-repo = ggml-org/tinygemma3-GGUF:Q8_0\nload-on-startup = true\npool = on\n")
        r.no_mmproj = True
        r.n_batch = 512  # a prompt in one batch leaves no checkpoint
    start_pool("move_tokens=16")
    drain("n0")
    msgs = [LIBRARY, {"role": "user", "content": "question one about the books"}]
    msgs.append({"role": "assistant", "content": chat(msgs)})
    msgs.append({"role": "user", "content": "and what about the old maps in the reading room? " * 3})
    msgs.append({"role": "assistant", "content": chat(msgs)})
    t = {}
    chat([LIBRARY, {"role": "user", "content": "question two about the opening hours"}], timings=t)
    assert t["cache_n"] == 0, t
    assert wait_share("n1") == 0.0
    drain("n0", False)
    drain("n1")
    conv = [LIBRARY, {"role": "user", "content": "a question about the archive"}]
    conv.append({"role": "assistant", "content": chat(conv)})
    conv.append({"role": "user", "content": "and then?"})
    drain("n1", False)
    drain("n0")
    chat(conv)
    move = events("move")
    assert len(move) == 1 and move[0]["node"] == "n1" and "no state moved" in move[0]["detail"], move
    # the first router restarts and forgets; the conversation of the drained n0 stays on n1, which holds it
    routers[0].stop()
    routers[0].start()
    wait_nodes()
    drain("n0")
    chat([LIBRARY, {"role": "user", "content": "question three about the newspapers"}])
    msgs.append({"role": "user", "content": "thanks"})
    chat(msgs, timings=t)
    assert t["cache_n"] > t["prompt_n"], t  # the node held the conversation
    time.sleep(0.5)
    assert node_of("n1")["share"] == -1


def test_pool_router_needs_its_key():
    # with --api-key on the routers (the same on all of them) a request without the key gets nothing, the routers
    # send it to each other and move the conversation
    global KEY
    for r in routers:
        r.api_key = "pool-key"
    KEY = "pool-key"
    start_pool("move_tokens=16 wait_max=0 prefill_tps=1")
    res = requests.post(f"http://{routers[1].server_host}:{routers[1].server_port}/pool/pull", json={"from": "x"})
    assert res.status_code == 401, res.text
    res = requests.get(f"http://{routers[0].server_host}:{routers[0].server_port}/pool/status")
    assert res.status_code == 401, res.text
    res = requests.get(f"http://{routers[0].server_host}:{routers[0].server_port}/pool")
    assert res.status_code == 200 and "llama pool" in res.text  # the page, its data needs the key
    msgs = [{"role": "user", "content": "once upon a time there was a little girl who " * 12}]
    msgs.append({"role": "assistant", "content": chat(msgs)})
    msgs.append({"role": "user", "content": "and then?"})
    first = events("new")[0]["node"]
    drain(first)
    chat(msgs)
    moved = events("migrate")
    assert len(moved) == 1 and moved[0]["node"] != first, events("move")


def test_pool_probes_its_links():
    # a link is measured when it first answers: one chunk of a move from the router and one to it (GET and POST
    # /pool/probe, on the router itself)
    start_pool()
    deadline = time.time() + 10
    while not all(l["mbps_read"] > 0 and l["mbps_write"] > 0 for n in status()["nodes"] for l in n["links"]):
        assert time.time() < deadline, [n["links"] for n in status()["nodes"]]
        time.sleep(0.1)
    url = f"http://{routers[0].server_host}:{routers[0].server_port}"
    res = requests.get(f"{url}/pool/probe?size=1000")
    assert res.status_code == 200 and len(res.content) == 1000
    res = requests.post(f"{url}/pool/probe", data=b"x" * 1000)
    assert res.status_code == 200 and res.json()["size"] == 1000, res.text
    # what the other routers read for the speed of a move: the links of this router to the other nodes
    deadline = time.time() + 10
    while True:
        peers = routers[1].make_request("GET", "/pool/router").body["peers"]
        if [p["name"] for p in peers] == ["n0"] and peers[0]["links"][0]["mbps_in"] > 0:
            break
        assert time.time() < deadline, peers
        time.sleep(0.1)


def test_pool_router_node_by_alias():
    # the model may have another name on another router (the 3090: [tiny-other] with alias = tiny-alias): the routers
    # share it under the alias they both have, and send the requests under it
    routers[1].models_preset = preset("r1", section="tiny-other")
    st = start_pool()
    assert [n["mismatch"] for n in st["nodes"]] == [False, False]
    drain("n0")
    chat([{"role": "user", "content": "which node"}])
    assert events("new")[0]["node"] == "n1"


@pytest.mark.parametrize("same_model", [True, False])
def test_pool_leaves_out_another_model(same_model: bool):
    # a node of another model than the most of the others (parameters, size, type, vocabulary from /pool/node) takes no
    # request; same_model=0 lets it take them
    routers[1].models_preset = preset("r1", body="hf-repo = ggml-org/stories15M_MOE\nhf-file = stories15M_MOE-F16.gguf\n"
                                                 "load-on-startup = true\npool = on\nalias = tiny-alias\n")
    st = start_pool("" if same_model else "same_model=0", slots=1 if same_model else 2)
    assert [n["mismatch"] for n in st["nodes"]] == [False, same_model]
    assert any(e["node"] == "n1" for e in events("mismatch")) == same_model
    for i in range(2):
        chat([{"role": "user", "content": f"which model {i}"}])
    assert {e["node"] for e in events("new")} == ({"n0"} if same_model else {"n0", "n1"})


#
# the node side: the instance moves the state of a conversation, plain instances without a pool
#

def plain_nodes(preset_fn=ServerPreset.tinyllama2, **opts) -> list:
    nodes = []
    for i in range(2):
        n = preset_fn()
        n.server_port = routers[0].server_port + 10 + i
        n.n_slots = 1
        for k, v in opts.items():
            setattr(n, k, v)
        nodes.append(n)
        routers.append(n)  # stopped with the routers
        n.start()
    return nodes


def test_pool_node_checks_what_comes_in():
    # the import takes the chunks of a transfer in order, and only a state of its own model and configuration
    src, dst = plain_nodes()
    url = f"http://{dst.server_host}:{dst.server_port}"
    res = requests.post(f"{url}/pool/import/data?id=00ff&offset=4096", data=b"x" * 16)
    assert res.status_code == 400 and "no such transfer" in res.text, res.text
    res = requests.post(f"{url}/pool/import/data?id=00ff&offset=0", data=b"x" * 4096)
    assert res.status_code == 400 and "not a prompt-cache state" in res.text, res.text
    # a real state, sent out of order, then whole with another signature
    msgs = [{"role": "user", "content": "a story about a lighthouse keeper " * 8}]
    res = src.make_request("POST", "/v1/chat/completions", data={"messages": msgs, "max_tokens": 8, "temperature": 0})
    assert res.status_code == 200, res.body
    msgs.append({"role": "assistant", "content": res.body["choices"][0]["message"]["content"]})
    body = {"messages": msgs + [{"role": "user", "content": "and then?"}], "max_tokens": 4}
    ex = src.make_request("POST", "/pool/export", data={"kind": "chat", "body": body}).body
    assert ex["n_reuse"] > 0 and not ex["from_disk"], ex
    data = requests.get(f"http://{src.server_host}:{src.server_port}/pool/export/data?id={ex['id']}&offset=0&size={1 << 26}").content
    assert len(data) == ex["size"]
    res = requests.post(f"{url}/pool/import/data?id={ex['id']}&offset=0", data=data[:8192])
    assert res.status_code == 200, res.text
    res = requests.post(f"{url}/pool/import/data?id={ex['id']}&offset=4096", data=data[4096:8192])
    assert res.status_code == 400 and "out of order" in res.text, res.text
    requests.post(f"{url}/pool/import/data?id={ex['id']}&offset=0", data=data)
    res = requests.post(f"{url}/pool/import/done", json={"id": ex["id"], "signature": "0123456789abcdef"})
    assert res.status_code == 400 and "signatures differ" in res.text, res.text
    src.make_request("POST", "/pool/export/done", data={"id": ex["id"]})


def test_pool_node_sends_the_checkpoint_the_request_needs():
    # tinygemma3 (SWA) goes back to a checkpoint when a request shares only a prefix with the state: the export sends
    # the newest checkpoint within that prefix, with the base its deltas hang off, and none when the request goes on
    # from the end of the state. the other node then takes what the export promised (n_reuse)
    src, dst = plain_nodes(ServerPreset.tinygemma3, no_mmproj=True,
                           n_batch=32, n_ubatch=32,  # the prompt has to span several batches to be checkpointed
                           n_ctx_checkpoints=-1, checkpoint_min_step=0, cache_ram_reserve=0)
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


def test_pool_node_without_a_prompt_cache_moves_nothing():
    src, _ = plain_nodes(cache_ram=0)
    res = src.make_request("POST", "/pool/export", data={"kind": "chat", "body": {"messages": [{"role": "user", "content": "hi"}]}})
    assert res.status_code == 501, res.body
    res = requests.get(f"http://{src.server_host}:{src.server_port}/pool/probe?size=1000")
    assert res.status_code == 501, res.text
