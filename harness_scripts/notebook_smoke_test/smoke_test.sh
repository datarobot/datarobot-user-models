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

# Code run in every kernel. Each check prints its own line, so the log shows
# which import or call failed. Required modules ship in every notebook image;
# optional ones are checked only when the image has them, since base images
# ship fewer packages. hashlib.md5 and dask tokenize (MD5) catch a base image
# that blocks non-FIPS digests.
SMOKE_CODE='
import hashlib, importlib, importlib.util, ssl, sys
print(f"python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}")
required = ["numpy", "pandas"]
optional = ["sklearn", "xgboost", "shap", "datarobot"]
for name in required + optional:
    if name in optional and importlib.util.find_spec(name) is None:
        print(f"import {name} ... SKIP: not installed")
        continue
    print(f"import {name} ...", end=" ")
    module = importlib.import_module(name)
    print("OK", getattr(module, "__version__", ""))
print("hashlib.md5 ...", end=" ")
hashlib.md5(b"smoke")
print("OK")
if importlib.util.find_spec("dask") is None:
    print("dask tokenize (MD5) ... SKIP: not installed")
else:
    print("dask tokenize (MD5) ...", end=" ")
    import pandas
    from dask.base import tokenize
    tokenize(pandas.DataFrame({"a": [1]}))
    print("OK")
print("ssl.create_default_context() ...", end=" ")
ssl.create_default_context()
print("OK")
print("SMOKE_OK")
'

CURRENT_CHECK="setup"
PASSED_CHECKS=()

log() { echo "[smoke] $*"; }
# check <description>: starts a check; the next check or the end of the script
# marks it as passed.
check() {
  finish_check
  CURRENT_CHECK="$*"
  echo
  log "CHECK: ${CURRENT_CHECK}"
}
finish_check() {
  if [ "${CURRENT_CHECK}" != "setup" ]; then
    log "  OK: ${CURRENT_CHECK}"
    PASSED_CHECKS+=("${CURRENT_CHECK}")
  fi
  CURRENT_CHECK="setup"
}
fail() { echo "[smoke]   FAIL: $*" >&2; exit 1; }

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

print_summary() {
  echo
  log "===== Summary for ${ENV_DIR} ====="
  local c
  for c in "${PASSED_CHECKS[@]+"${PASSED_CHECKS[@]}"}"; do
    log "  [PASS] ${c}"
  done
}

cleanup() {
  local retval=$?
  if [ "${retval}" -ne 0 ]; then
    print_summary
    log "  [FAIL] ${CURRENT_CHECK}"
    log "RESULT: FAILED at check: ${CURRENT_CHECK}"
    if docker inspect "${NAME}" >/dev/null 2>&1; then
      local state
      state=$(docker inspect --format '{{.State.Status}} (exit code {{.State.ExitCode}})' "${NAME}")
      log "container state: ${state}"
      echo
      if [ "$(docker inspect --format '{{.State.Running}}' "${NAME}")" = "true" ]; then
        echo "----- container processes -----"
        docker exec "${NAME}" ps -eo pid,user,args 2>&1 || true
      fi
      echo "----- container log (last 200 lines) -----"
      docker logs --tail 200 "${NAME}" 2>&1 || true
      echo "------------------------------------------"
    fi
  fi
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  docker volume rm -f "${SSH_VOLUME}" >/dev/null 2>&1 || true
  exit "${retval}"
}
trap cleanup EXIT

log "Environment:     ${ENV_DIR}"
log "Image:           ${IMAGE}"
log "Startup timeout: ${STARTUP_TIMEOUT}s"
log "Kernel exec:     $([ "${SKIP_KERNEL_EXEC}" = "1" ] && echo "skipped (SKIP_KERNEL_EXEC=1)" || echo "enabled")"

check "image ${IMAGE} is available locally or can be pulled"
docker image inspect "${IMAGE}" >/dev/null 2>&1 || docker pull "${IMAGE}" \
  || fail "cannot pull ${IMAGE}"
log "  image id: $(docker image inspect --format '{{.Id}} (created {{.Created}}, {{.Architecture}})' "${IMAGE}")"

check "generate SSH host key and authorized_keys in a docker volume"
docker volume create "${SSH_VOLUME}" >/dev/null
docker run --rm --user root --volume "${SSH_VOLUME}:/ssh" --entrypoint /bin/sh "${IMAGE}" -c '
  mkdir -p /ssh/keys /ssh/authorized_keys &&
  ssh-keygen -q -t ecdsa -b 256 -N "" -f /ssh/keys/ssh_host_key &&
  cp /ssh/keys/ssh_host_key.pub /ssh/authorized_keys/notebooks &&
  chmod -R a+rX /ssh
' || fail "cannot generate SSH keys with ssh-keygen from the image"

check "container starts via ${KERNEL_DIR}/start_server.sh"
docker run --detach --name "${NAME}" \
  --volume "${SSH_VOLUME}:/var/run/notebooks/ssh:ro" \
  --entrypoint /bin/bash "${IMAGE}" "${KERNEL_DIR}/start_server.sh" >/dev/null \
  || fail "docker run failed"
log "  container: ${NAME}"

check "kernel gateway answers and kernels run code"
# The client runs inside the container with the kernel venv, which ships tornado.
# It prints its own sub-checks; the last "CHECK:" line before a failure is the
# one that failed.
docker exec -i \
  -e STARTUP_TIMEOUT="${STARTUP_TIMEOUT}" \
  -e SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC}" \
  -e SMOKE_CODE="${SMOKE_CODE}" \
  "${NAME}" "${KERNEL_PYTHON}" - <<'EOF' || fail "kernel gateway check failed, see the last sub-check above"
import asyncio, json, os, sys, time, uuid
from tornado.httpclient import AsyncHTTPClient, HTTPRequest
from tornado.websocket import websocket_connect

BASE = "127.0.0.1:8888"
TIMEOUT = int(os.environ["STARTUP_TIMEOUT"])
CODE = os.environ["SMOKE_CODE"]
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
    await ws.write_message(json.dumps({
        "header": {"msg_id": msg_id, "username": "smoke", "session": uuid.uuid4().hex,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {}, "channel": "shell",
        "content": {"code": CODE, "silent": False, "store_history": False},
    }))
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
    if os.environ["SKIP_KERNEL_EXEC"] == "1":
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
EOF

check "drgithelper --version runs (no Go panic, libcrypto found)"
docker exec -e HOME=/home/notebooks "${NAME}" "${KERNEL_DIR}/drgithelper" --version \
  || fail "drgithelper does not run"

processes=$(docker exec "${NAME}" ps -eo args)
check "sshd is running"
grep "sshd -D" <<<"${processes}" | sed 's/^/[smoke]   /' || fail "no 'sshd -D' process"

check "monitoring agent (uvicorn agent:app) is running"
grep "uvicorn agent:app" <<<"${processes}" | sed 's/^/[smoke]   /' || fail "no 'uvicorn agent:app' process"

check "container log has no Go panics or Python tracebacks"
if docker logs "${NAME}" 2>&1 | grep -n -A 5 -E "^panic:|Traceback \(most recent call last\)"; then
  fail "found panics or tracebacks in the container log (shown above)"
fi

finish_check
print_summary
log "RESULT: PASSED"
