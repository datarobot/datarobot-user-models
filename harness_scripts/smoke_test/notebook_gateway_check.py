"""Kernel gateway check that notebook_smoke_test.sh runs inside a notebook container.

Runs with the kernel venv python, which ships tornado, and talks to the kernel
gateway on 127.0.0.1:8888:
  - waits for GET /api and for the kernel that the gateway prespawns on start
  - runs KERNEL_CHECK_CODE in the prespawned kernel and in a newly created kernel
  - deletes the new kernel

Prints its own sub-checks; the last "CHECK:" line before a failure is the one
that failed.

Env vars:
  STARTUP_TIMEOUT    seconds to wait for the gateway, kernels and replies
  SKIP_KERNEL_EXEC   set to 1 to only check that the prespawned kernel exists
  KERNEL_CHECK_CODE  notebook_kernel_check.py; its run_checks() must print
                     SMOKE_OK when every check passed
  REQUIRED_MODULES   space separated modules that must import in the kernels
"""

import asyncio
import json
import os
import sys
import time
import uuid

from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.websocket import websocket_connect

BASE = "127.0.0.1:8888"
TIMEOUT = int(os.environ["STARTUP_TIMEOUT"])
SKIP_KERNEL_EXEC = os.environ.get("SKIP_KERNEL_EXEC") == "1"
REQUIRED_MODULES = os.environ["REQUIRED_MODULES"].split()
# Kernels don't see this process's env vars, so the module list goes into the code.
CODE = f"{os.environ['KERNEL_CHECK_CODE']}\nrun_checks({REQUIRED_MODULES!r})\n"
http = AsyncHTTPClient()


def log(msg):
    print(f"[smoke]   {msg}", flush=True)


def fail(msg):
    log(f"  FAIL: {msg}")
    sys.exit(1)


async def api(path, method="GET", body=None):
    req = HTTPRequest(f"http://{BASE}{path}", method=method, body=body, request_timeout=TIMEOUT)
    resp = await http.fetch(req)
    return json.loads(resp.body) if resp.body else None


async def wait_for(what, check):
    log(f"waiting up to {TIMEOUT}s for {what}")
    start = time.monotonic()
    last_error = None
    while True:
        try:
            result = await check()
            if result:
                log(f"  {what}: ready after {time.monotonic() - start:.0f}s")
                return result
            last_error = f"empty response: {result!r}"
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
        if time.monotonic() - start > TIMEOUT:
            fail(f"timed out after {TIMEOUT}s waiting for {what}; last error: {last_error}")
        await asyncio.sleep(3)


async def execute(kernel_id):
    log(f"CHECK: run smoke code in kernel {kernel_id}")
    try:
        ws = await websocket_connect(
            HTTPRequest(f"ws://{BASE}/api/kernels/{kernel_id}/channels", request_timeout=TIMEOUT)
        )
    except Exception as e:
        fail(f"cannot open websocket to kernel {kernel_id}: {type(e).__name__}: {e}")
    msg_id = uuid.uuid4().hex
    await ws.write_message(
        json.dumps(
            {
                "header": {
                    "msg_id": msg_id,
                    "username": "smoke",
                    "session": uuid.uuid4().hex,
                    "msg_type": "execute_request",
                    "version": "5.3",
                },
                "parent_header": {},
                "metadata": {},
                "channel": "shell",
                "content": {"code": CODE, "silent": False, "store_history": False},
            }
        )
    )
    output = ""
    while True:
        try:
            raw = await asyncio.wait_for(ws.read_message(), TIMEOUT)
        except asyncio.TimeoutError:
            fail(f"no reply from kernel {kernel_id} in {TIMEOUT}s; output so far:\n{output}")
        if raw is None:
            fail(f"websocket closed by kernel {kernel_id}; output so far:\n{output}")
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        kind = msg["msg_type"]
        if kind == "stream":
            output += msg["content"]["text"]
        elif kind == "error":
            c = msg["content"]
            for line in output.splitlines():
                log(f"  | {line}")
            fail(f"kernel {kernel_id} raised {c['ename']}: {c['evalue']}")
        elif kind == "execute_reply":
            break
    ws.close()
    for line in output.splitlines():
        log(f"  | {line}")
    if "SMOKE_OK" not in output:
        fail(f"kernel {kernel_id} did not print SMOKE_OK")
    log(f"  OK: smoke code ran in kernel {kernel_id}")


async def main():
    log("CHECK: GET /api")
    version = await wait_for("GET /api", lambda: api("/api"))
    log(f"  OK: gateway is up, version {version.get('version')}")

    log("CHECK: prespawned kernel is listed in GET /api/kernels")
    kernels = await wait_for("the prespawned kernel", lambda: api("/api/kernels"))
    log(f"  OK: kernels: {[(k['id'], k.get('execution_state')) for k in kernels]}")
    if SKIP_KERNEL_EXEC:
        log("SKIP: running code in kernels (SKIP_KERNEL_EXEC=1)")
        return

    await execute(kernels[0]["id"])

    log("CHECK: POST /api/kernels creates a new kernel")
    try:
        kernel = await api("/api/kernels", method="POST", body=json.dumps({"name": "python3"}))
    except Exception as e:
        fail(f"cannot create a kernel: {type(e).__name__}: {e}")
    log(f"  OK: created kernel {kernel['id']}")
    await execute(kernel["id"])

    log(f"CHECK: DELETE /api/kernels/{kernel['id']}")
    try:
        await api(f"/api/kernels/{kernel['id']}", method="DELETE")
    except Exception as e:
        fail(f"cannot delete kernel {kernel['id']}: {type(e).__name__}: {e}")
    log("  OK: kernel deleted")


asyncio.run(main())
