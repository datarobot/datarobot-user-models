#
#  Copyright 2026 DataRobot, Inc. and its affiliates.
#
#  All rights reserved.
#  This is proprietary source code of DataRobot, Inc. and its affiliates.
#  Released under the terms of DataRobot Tool and Utility Agreement.
#
import socket
import subprocess
import sys
import textwrap
import time

import pytest
from gunicorn.util import load_class

from datarobot_drum.drum.gunicorn.workers import DrumGeventWorker

WORKER_CLASS = "datarobot_drum.drum.gunicorn.workers.DrumGeventWorker"
FIRST_BYTE_TIMEOUT = 1


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_until_listening(port, deadline=15):
    start = time.monotonic()
    while time.monotonic() - start < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"gunicorn did not start listening on {port}")


@pytest.fixture
def gunicorn_server(tmp_path):
    (tmp_path / "hello_app.py").write_text(textwrap.dedent("""
            def app(environ, start_response):
                start_response("200 OK", [("Content-Type", "text/plain")])
                return [b"ok"]
            """))
    (tmp_path / "conf.py").write_text(textwrap.dedent(f"""
            def post_fork(server, worker):
                worker.first_byte_timeout = {FIRST_BYTE_TIMEOUT}
            """))
    port = free_port()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "gunicorn",
            "hello_app:app",
            "--config",
            str(tmp_path / "conf.py"),
            "--bind",
            f"127.0.0.1:{port}",
            "--workers",
            "1",
            "--worker-class",
            WORKER_CLASS,
            "--worker-connections",
            "1",
            "--keep-alive",
            "0",
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_until_listening(port)
        # The readiness probe above took the only slot; let it be released.
        time.sleep(0.5)
        yield port
    finally:
        proc.terminate()
        proc.wait(timeout=10)


REQUEST = b"GET / HTTP/1.1\r\nHost: test\r\n\r\n"


def connect(port):
    return socket.create_connection(("127.0.0.1", port), timeout=FIRST_BYTE_TIMEOUT + 5)


def test_worker_class_path_resolves_to_drum_gevent_worker():
    assert load_class(WORKER_CLASS) is DrumGeventWorker


def test_silent_connection_does_not_block_next_request(gunicorn_server):
    with connect(gunicorn_server), connect(gunicorn_server) as client:
        client.sendall(REQUEST)
        assert client.recv(4096).startswith(b"HTTP/1.1 200 OK")


def test_silent_connection_is_closed_after_first_byte_timeout(gunicorn_server):
    with connect(gunicorn_server) as silent:
        assert silent.recv(1) == b""


def test_request_sent_before_first_byte_timeout_is_served(gunicorn_server):
    with connect(gunicorn_server) as client:
        time.sleep(FIRST_BYTE_TIMEOUT / 2)
        client.sendall(REQUEST)
        assert client.recv(4096).startswith(b"HTTP/1.1 200 OK")
