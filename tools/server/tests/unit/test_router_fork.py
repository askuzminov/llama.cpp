import socket
import tempfile
import pytest
from utils import *

# fork: router tests of the fork. The helpers of test_router use its module-level `server`,
# so the fixture below points that one at the server of this module too
import test_router
from test_router import MODEL_A, _Bg, _load_model_and_wait, _tokenize, _wait_for_model_status

server: ServerProcess

@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.router()
    test_router.server = server


# router shutdown tests: the child must not outlive the router

# LLAMA_SERVER_MEM_WAIT holds the child for 60 s before its load, it waits for memory that never gets free
MEM_WAIT_60S = json.dumps({"ms": 60000, "need": {"host": 1 << 50}})


def _child_pids(pid: int) -> list[int]:
    out = subprocess.run(["ps", "-A", "-o", "pid=", "-o", "ppid="], capture_output=True, text=True, check=True).stdout
    return [int(p) for p, pp in (line.split() for line in out.splitlines()) if int(pp) == pid]


def _wait_pids_gone(pids: list[int], timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        # an exited child can stay a zombie for a short time
        stats = [subprocess.run(["ps", "-o", "stat=", "-p", str(p)], capture_output=True, text=True).stdout.strip() for p in pids]
        if all(s == "" or s.startswith("Z") for s in stats):
            return
        time.sleep(0.05)
    raise AssertionError(f"child processes {pids} still alive {timeout} s after the router stopped")


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signals and ps")
@pytest.mark.parametrize("state", ["loading", "loaded"])
def test_router_killed_child_exits(monkeypatch, state: str):
    """a child exits when the router dies, also before it reported ready"""
    global server
    if state == "loading":
        monkeypatch.setenv("LLAMA_SERVER_MEM_WAIT", MEM_WAIT_60S)
    server.start()

    res = server.make_request("POST", "/models/load", data={"model": MODEL_A})
    assert res.status_code == 200
    _wait_for_model_status(MODEL_A, {state})
    children = _child_pids(server.process.pid)
    assert children, "router has no child process"

    server.process.kill()
    server.process.wait()
    _wait_pids_gone(children, timeout=10)


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signals and ps")
def test_router_stop_does_not_wait_for_load(monkeypatch):
    """a graceful router stop does not wait for a load that a pending request waits for"""
    global server
    monkeypatch.setenv("LLAMA_SERVER_MEM_WAIT", MEM_WAIT_60S)
    server.start()

    pending = _Bg(lambda: _tokenize(MODEL_A)).start()
    _wait_for_model_status(MODEL_A, {"loading"})
    children = _child_pids(server.process.pid)
    assert children, "router has no child process"

    server.process.terminate()
    server.process.wait(timeout=10)
    _wait_pids_gone(children, timeout=10)
    pending.join(30)  # it fails, the router went away


@pytest.mark.skipif(os.name == "nt", reason="uses POSIX signals and ps")
def test_router_killed_after_exit_command_child_exits():
    """a child still shutting down when the router dies exits at its stop timeout, as if the router
    force-killed it. Windows kills the router 5 s after its console window is closed"""
    global server
    stop_timeout = 3
    preset_path = os.path.join(TMP_DIR, "test_stop_timeout.ini")
    with open(preset_path, "w") as f:
        f.write(
            "[model-slow-exit]\n"
            "hf-repo = ggml-org/test-model-stories260K\n"
            f"stop-timeout = {stop_timeout}\n"
        )
    server.models_preset = preset_path
    fd, server.log_path = tempfile.mkstemp(suffix=".log")
    os.close(fd)
    sock = None

    def log() -> str:
        with open(server.log_path) as f:
            return f.read()

    try:
        server.start()
        _load_model_and_wait("model-slow-exit", timeout=120)
        children = _child_pids(server.process.pid)
        assert children, "router has no child process"
        port = int(re.findall(r"name=model-slow-exit on port (\d+)", log())[-1])

        # a request whose body never comes keeps the child in its HTTP shutdown until the read timeout,
        # as a slow model free or prompt-cache write would
        sock = socket.create_connection(("127.0.0.1", port))
        sock.sendall(b"POST /completion HTTP/1.1\r\nHost: localhost\r\nContent-Length: 100\r\n\r\n")
        time.sleep(0.5)  # let the child take the request

        server.process.terminate()
        deadline = time.time() + 10
        while "exit command received" not in log():
            assert time.time() < deadline, "the child did not get the exit command"
            time.sleep(0.01)
        t_exit = time.time()
        server.process.kill()
        server.process.wait()

        _wait_pids_gone(children, timeout=stop_timeout + 5)
        waited = time.time() - t_exit
        # the child keeps the time a live router would give it
        assert waited > stop_timeout - 1, f"the child exited {waited:.2f} s after the exit command"
    finally:
        if sock is not None:
            sock.close()
        os.remove(preset_path)
        os.remove(server.log_path)
