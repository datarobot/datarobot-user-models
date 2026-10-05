#!/usr/bin/env bash
# Smoke test for a notebook environment image.
#
# Starts the image the way the Notebooks platform does (start_server.sh as the
# image user, with SSH keys mounted under /var/run/notebooks/ssh) and checks
# that every service comes up and a kernel can run code:
#   - the kernel gateway answers on /api
#   - code runs in the prespawned kernel and in a newly created kernel
#   - drgithelper runs, sshd and the monitoring agent are up
#   - the container log has no Go panics or Python tracebacks
#
# Only needs docker on the host: all HTTP calls run inside the container, and
# the SSH keys live in a docker volume, so it also works with docker-in-docker.
#
# Usage: smoke_test.sh <env_dir> [image]
#   env_dir  environment folder containing env_info.json,
#            e.g. public_dropin_notebook_environments/python313_notebook
#   image    image to test; defaults to
#            docker.io/datarobotdev/<imageRepository>:<environmentVersionId>
#
# Optional env vars:
#   STARTUP_TIMEOUT  seconds to wait for the gateway and kernels (default 300)
#   SKIP_KERNEL_EXEC set to 1 to only check that the prespawned kernel exists,
#                    without running code; ipykernel never answers under amd64
#                    emulation (Rosetta) on Apple Silicon, so use it there

set -euo pipefail

ENV_DIR="${1:?Usage: smoke_test.sh <env_dir> [image]}"
IMAGE="${2:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC:-0}"

KERNEL_DIR=/etc/system/kernel
KERNEL_PYTHON="${KERNEL_DIR}/.venv/bin/python3"

# Imports that cover the stack users rely on; dask hashes with MD5, so it also
# catches a base image that blocks non-FIPS digests.
SMOKE_CODE='
import ssl, numpy, pandas, sklearn, xgboost, shap, datarobot
from dask.base import tokenize
tokenize(pandas.DataFrame({"a": [1]}))
ssl.create_default_context()
print("SMOKE_OK", ssl.OPENSSL_VERSION)
'

log() { echo "[smoke] $*"; }
fail() { echo "[smoke] FAIL: $*" >&2; exit 1; }

if [ -z "${IMAGE}" ]; then
  env_info="${ENV_DIR}/env_info.json"
  # sed instead of python/jq so the host needs nothing but docker
  repo=$(sed -n 's/.*"imageRepository": *"\([^"]*\)".*/\1/p' "${env_info}")
  tag=$(sed -n 's/.*"environmentVersionId": *"\([^"]*\)".*/\1/p' "${env_info}")
  [ -n "${repo}" ] && [ -n "${tag}" ] || fail "cannot read imageRepository/environmentVersionId from ${env_info}"
  IMAGE="docker.io/datarobotdev/${repo}:${tag}"
fi

NAME="notebook-smoke-$$-${RANDOM}"
SSH_VOLUME="${NAME}-ssh"

cleanup() {
  local retval=$?
  if [ "${retval}" -ne 0 ] && docker inspect "${NAME}" >/dev/null 2>&1; then
    echo "----- container log (last 200 lines) -----"
    docker logs --tail 200 "${NAME}" 2>&1 || true
    echo "------------------------------------------"
  fi
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  docker volume rm -f "${SSH_VOLUME}" >/dev/null 2>&1 || true
  exit "${retval}"
}
trap cleanup EXIT

log "Testing image ${IMAGE}"

log "Generating SSH host key and authorized_keys"
docker volume create "${SSH_VOLUME}" >/dev/null
docker run --rm --user root --volume "${SSH_VOLUME}:/ssh" --entrypoint /bin/sh "${IMAGE}" -c '
  mkdir -p /ssh/keys /ssh/authorized_keys &&
  ssh-keygen -q -t ecdsa -b 256 -N "" -f /ssh/keys/ssh_host_key &&
  cp /ssh/keys/ssh_host_key.pub /ssh/authorized_keys/notebooks &&
  chmod -R a+rX /ssh
'

log "Starting container via start_server.sh"
docker run --detach --name "${NAME}" \
  --volume "${SSH_VOLUME}:/var/run/notebooks/ssh:ro" \
  --entrypoint /bin/bash "${IMAGE}" "${KERNEL_DIR}/start_server.sh" >/dev/null

log "Waiting for the kernel gateway and running code in kernels"
# The client runs inside the container with the kernel venv, which ships tornado.
docker exec -i \
  -e STARTUP_TIMEOUT="${STARTUP_TIMEOUT}" \
  -e SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC}" \
  -e SMOKE_CODE="${SMOKE_CODE}" \
  "${NAME}" "${KERNEL_PYTHON}" - <<'EOF' || fail "kernel gateway check failed"
import asyncio, json, os, time, uuid
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.websocket import websocket_connect

BASE = "127.0.0.1:8888"
TIMEOUT = int(os.environ["STARTUP_TIMEOUT"])
CODE = os.environ["SMOKE_CODE"]
http = AsyncHTTPClient()


async def api(path, method="GET", body=None):
    req = HTTPRequest(f"http://{BASE}{path}", method=method, body=body, request_timeout=TIMEOUT)
    resp = await http.fetch(req)
    return json.loads(resp.body) if resp.body else None


async def wait_for(what, check):
    deadline = time.monotonic() + TIMEOUT
    while True:
        try:
            result = await check()
            if result:
                return result
        except Exception:
            pass
        if time.monotonic() > deadline:
            raise SystemExit(f"timed out after {TIMEOUT}s waiting for {what}")
        await asyncio.sleep(3)


async def execute(kernel_id):
    ws = await websocket_connect(
        HTTPRequest(f"ws://{BASE}/api/kernels/{kernel_id}/channels", request_timeout=TIMEOUT)
    )
    msg_id = uuid.uuid4().hex
    await ws.write_message(json.dumps({
        "header": {"msg_id": msg_id, "username": "smoke", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell",
        "content": {"code": CODE, "silent": False, "store_history": False},
    }))
    output = ""
    while True:
        raw = await asyncio.wait_for(ws.read_message(), TIMEOUT)
        if raw is None:
            raise SystemExit(f"websocket closed by kernel {kernel_id}")
        msg = json.loads(raw)
        if msg.get("parent_header", {}).get("msg_id") != msg_id:
            continue
        kind = msg["msg_type"]
        if kind == "stream":
            output += msg["content"]["text"]
        elif kind == "error":
            c = msg["content"]
            raise SystemExit(f"kernel {kernel_id} raised {c['ename']}: {c['evalue']}")
        elif kind == "execute_reply":
            break
    ws.close()
    if "SMOKE_OK" not in output:
        raise SystemExit(f"kernel {kernel_id} did not print SMOKE_OK; output: {output!r}")
    print(f"[smoke]   kernel {kernel_id}: {output.strip()}")


async def main():
    version = await wait_for("GET /api", lambda: api("/api"))
    print(f"[smoke]   gateway is up: {version}")

    kernels = await wait_for("the prespawned kernel", lambda: api("/api/kernels"))
    if os.environ["SKIP_KERNEL_EXEC"] == "1":
        print(f"[smoke]   prespawned kernel {kernels[0]['id']} exists; skipping code execution")
        return

    await execute(kernels[0]["id"])
    kernel = await api("/api/kernels", method="POST", body=json.dumps({"name": "python3"}))
    await execute(kernel["id"])
    await api(f"/api/kernels/{kernel['id']}", method="DELETE")


asyncio.run(main())
EOF

log "Checking drgithelper"
docker exec -e HOME=/home/notebooks "${NAME}" "${KERNEL_DIR}/drgithelper" --version \
  || fail "drgithelper does not run"

log "Checking that sshd and the monitoring agent are running"
processes=$(docker exec "${NAME}" ps -eo args)
grep -q "sshd -D" <<<"${processes}" || fail "sshd is not running"
grep -q "uvicorn agent:app" <<<"${processes}" || fail "monitoring agent (uvicorn) is not running"

log "Checking the container log for panics and tracebacks"
if docker logs "${NAME}" 2>&1 | grep -E "^panic:|Traceback \(most recent call last\)"; then
  fail "found panics or tracebacks in the container log"
fi

log "PASS"
