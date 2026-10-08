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
# Usage: notebook_smoke_test.sh <env_dir> [image]
#   env_dir  environment folder containing env_info.json,
#            e.g. public_dropin_notebook_environments/python313_notebook
#   image    image to test; defaults to
#            docker.io/datarobotdev/<imageRepository>:<environmentVersionId>
#
# Optional env vars:
#   COMMIT_SHA       commit the image was built for; only shown in the log header
#   STARTUP_TIMEOUT  seconds to wait for the gateway and kernels (default 300)
#   SKIP_KERNEL_EXEC set to 1 to only check that the prespawned kernel exists,
#                    without running code; ipykernel never answers under amd64
#                    emulation (Rosetta) on Apple Silicon, so use it there

set -euo pipefail

ENV_DIR="${1:?Usage: notebook_smoke_test.sh <env_dir> [image]}"
IMAGE="${2:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC:-0}"
COMMIT_SHA="${COMMIT_SHA:-}"

KERNEL_DIR=/etc/system/kernel
KERNEL_PYTHON="${KERNEL_DIR}/.venv/bin/python3"

# Files kept next to this script:
#   notebook_gateway_check.py     runs inside the container and talks to the kernel gateway
#   notebook_kernel_check.py      code that the gateway check runs in every kernel
#   notebook_required_modules.txt modules that must import in the kernels of each env;
#                                 envs not listed there get DEFAULT_REQUIRED_MODULES
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_CHECK="${SCRIPT_DIR}/notebook_gateway_check.py"
KERNEL_CHECK="${SCRIPT_DIR}/notebook_kernel_check.py"
REQUIRED_MODULES_FILE="${SCRIPT_DIR}/notebook_required_modules.txt"
DEFAULT_REQUIRED_MODULES="numpy pandas"

CURRENT_CHECK="setup"
PASSED_CHECKS=()
WARNINGS=()

log() { echo "[notebook-smoke] $*"; }
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
fail() { echo "[notebook-smoke]   FAIL: $*" >&2; exit 1; }

# Reads a top-level string field from env_info.json; sed instead of python/jq so
# the host needs nothing but docker
env_info_field() { sed -n "s/^ *\"$1\": *\"\([^\"]*\)\".*/\1/p" "${ENV_INFO}" | head -1; }

ENV_INFO="${ENV_DIR}/env_info.json"
ENV_NAME=""
ENV_ID=""
ENV_VERSION_ID=""
if [ -f "${ENV_INFO}" ]; then
  ENV_NAME=$(env_info_field name)
  ENV_ID=$(env_info_field id)
  ENV_VERSION_ID=$(env_info_field environmentVersionId)
fi
if [ -z "${IMAGE}" ]; then
  [ -f "${ENV_INFO}" ] || fail "${ENV_INFO} not found"
  repo=$(env_info_field imageRepository)
  [ -n "${repo}" ] && [ -n "${ENV_VERSION_ID}" ] \
    || fail "cannot read imageRepository/environmentVersionId from ${ENV_INFO}"
  IMAGE="docker.io/datarobotdev/${repo}:${ENV_VERSION_ID}"
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
  for c in "${WARNINGS[@]+"${WARNINGS[@]}"}"; do
    log "  [WARN] ${c}"
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

log "Environment:      ${ENV_DIR}"
log "Env name:         ${ENV_NAME:-n/a}"
log "Image ID:         ${ENV_ID:-n/a}"
log "Image version ID: ${ENV_VERSION_ID:-n/a}"
log "Commit:           ${COMMIT_SHA:-not set}"
log "Image name:       ${IMAGE}"
log "Startup timeout:  ${STARTUP_TIMEOUT}s"
log "Kernel exec:      $([ "${SKIP_KERNEL_EXEC}" = "1" ] && echo "skipped (SKIP_KERNEL_EXEC=1)" || echo "enabled")"

# Env path relative to the repo root, as listed in the required modules file
ENV_PATH="${ENV_DIR#./}"
ENV_PATH="${ENV_PATH%/}"
check "required modules for ${ENV_PATH}"
REQUIRED_MODULES=$(awk -v env="${ENV_PATH}" '$1 == env { $1 = ""; sub(/^ +/, ""); print; exit }' "${REQUIRED_MODULES_FILE}")
if [ -z "${REQUIRED_MODULES}" ]; then
  REQUIRED_MODULES="${DEFAULT_REQUIRED_MODULES}"
  warning="${ENV_PATH} is not listed in $(basename "${REQUIRED_MODULES_FILE}"), checking only the default modules"
  WARNINGS+=("${warning}")
  log "  WARNING: ${warning}"
fi
log "  modules: ${REQUIRED_MODULES}"

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
# The gateway check runs inside the container with the kernel venv, which ships
# tornado; both Python files are streamed in, so nothing is mounted. It prints
# its own sub-checks; the last "CHECK:" line before a failure is the one that failed.
docker exec -i \
  -e STARTUP_TIMEOUT="${STARTUP_TIMEOUT}" \
  -e SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC}" \
  -e KERNEL_CHECK_CODE="$(cat "${KERNEL_CHECK}")" \
  -e REQUIRED_MODULES="${REQUIRED_MODULES}" \
  "${NAME}" "${KERNEL_PYTHON}" - <"${GATEWAY_CHECK}" \
  || fail "kernel gateway check failed, see the last sub-check above"

check "drgithelper --version runs (no Go panic, libcrypto found)"
docker exec -e HOME=/home/notebooks "${NAME}" "${KERNEL_DIR}/drgithelper" --version \
  || fail "drgithelper does not run"

processes=$(docker exec "${NAME}" ps -eo args)
check "sshd is running"
grep "sshd -D" <<<"${processes}" | sed 's/^/[notebook-smoke]   /' || fail "no 'sshd -D' process"

check "monitoring agent (uvicorn agent:app) is running"
grep "uvicorn agent:app" <<<"${processes}" | sed 's/^/[notebook-smoke]   /' || fail "no 'uvicorn agent:app' process"

check "container log has no Go panics or Python tracebacks"
if docker logs "${NAME}" 2>&1 | grep -n -A 5 -E "^panic:|Traceback \(most recent call last\)"; then
  fail "found panics or tracebacks in the container log (shown above)"
fi

finish_check
print_summary
log "RESULT: PASSED"
