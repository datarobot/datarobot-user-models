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
# Those checks run twice, against two containers:
#   1. default    the image's own user, no extra mounts. What every install gets.
#   2. hardened   a uid the image was not built with, plus writable volumes over the
#                 paths start_server.sh has to write to. What a cluster with a
#                 restrictive PodSecurity/Gatekeeper policy gets, and what
#                 nbx-operator produces when notebookSession.writableVolumes is
#                 enabled. The image layers and $HOME are read-only there, so a
#                 session that only ever ran as the build-time uid comes up subtly
#                 broken while the container still looks healthy (FLEET-8918).
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
#   SESSION_UID      uid for the hardened pass (default 1500). Any value the image
#                    was not built with will do; it must not be in /etc/passwd,
#                    which is the situation a cluster-assigned uid is in.
#   SKIP_HARDENED    set to 1 to run the default pass only

set -euo pipefail

ENV_DIR="${1:?Usage: notebook_smoke_test.sh <env_dir> [image]}"
IMAGE="${2:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC:-0}"
COMMIT_SHA="${COMMIT_SHA:-}"
SESSION_UID="${SESSION_UID:-1500}"
SKIP_HARDENED="${SKIP_HARDENED:-0}"

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

RUN_ID="notebook-smoke-$$-${RANDOM}"
SSH_VOLUME="${RUN_ID}-ssh"
# NAME is the container of the pass currently running, which is the one cleanup
# dumps diagnostics for; CONTAINERS is every container to remove on the way out.
NAME=""
CONTAINERS=()

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
    if [ -n "${NAME}" ] && docker inspect "${NAME}" >/dev/null 2>&1; then
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
  local c
  for c in "${CONTAINERS[@]+"${CONTAINERS[@]}"}"; do
    docker rm -f "${c}" >/dev/null 2>&1 || true
  done
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

# run_session <pass label> [extra docker run args...]
# Starts the image the way the platform does and runs every per-session check
# against it. Check names are prefixed with the pass so the summary says which
# of the two containers a failure came from.
run_session() {
  local pass="${1}"
  shift
  local processes

  NAME="${RUN_ID}-$(tr -cd 'a-z0-9' <<<"${pass}" | cut -c1-12)"
  CONTAINERS+=("${NAME}")

  echo
  log "================ PASS: ${pass} ================"

  check "[${pass}] container starts via ${KERNEL_DIR}/start_server.sh"
  docker run --detach --name "${NAME}" \
    --volume "${SSH_VOLUME}:/var/run/notebooks/ssh:ro" \
    ${@+"${@}"} \
    --entrypoint /bin/bash "${IMAGE}" "${KERNEL_DIR}/start_server.sh" >/dev/null \
    || fail "docker run failed"
  log "  container: ${NAME}"
  log "  running as: $(docker exec "${NAME}" id 2>/dev/null || echo 'could not read id')"

  check "[${pass}] kernel gateway answers and kernels run code"
  # The gateway check runs inside the container with the kernel venv, which ships
  # tornado; both Python files are streamed in, so nothing is mounted. It prints
  # its own sub-checks; the last "CHECK:" line before a failure is the one that failed.
  #
  # This is also what proves the kernel can still import its packages in the hardened
  # pass, where the kernel venv is read-only and setup-venv.sh has put a second venv
  # in front of it on PYTHONPATH.
  docker exec -i \
    -e STARTUP_TIMEOUT="${STARTUP_TIMEOUT}" \
    -e SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC}" \
    -e KERNEL_CHECK_CODE="$(cat "${KERNEL_CHECK}")" \
    -e REQUIRED_MODULES="${REQUIRED_MODULES}" \
    "${NAME}" "${KERNEL_PYTHON}" - <"${GATEWAY_CHECK}" \
    || fail "kernel gateway check failed, see the last sub-check above"

  check "[${pass}] drgithelper --version runs (no Go panic, libcrypto found)"
  docker exec -e HOME=/home/notebooks "${NAME}" "${KERNEL_DIR}/drgithelper" --version \
    || fail "drgithelper does not run"

  processes=$(docker exec "${NAME}" ps -eo args)
  check "[${pass}] sshd is running"
  # sshd only reaches this state if start_server.sh managed to put a host key in
  # /etc/ssh/keys. With a volume mounted over that path a plain mkdir fails and the
  # copy never runs, and sshd then dies while the container still looks healthy.
  grep "sshd -D" <<<"${processes}" | sed 's/^/[notebook-smoke]   /' || fail "no 'sshd -D' process"

  check "[${pass}] monitoring agent (uvicorn agent:app) is running"
  grep "uvicorn agent:app" <<<"${processes}" | sed 's/^/[notebook-smoke]   /' || fail "no 'uvicorn agent:app' process"

  check "[${pass}] container log has no Go panics or Python tracebacks"
  if docker logs "${NAME}" 2>&1 | grep -n -A 5 -E "^panic:|Traceback \(most recent call last\)"; then
    fail "found panics or tracebacks in the container log (shown above)"
  fi

  finish_check
  # Freed as soon as the pass is green so two sessions never run at once; the step
  # has a memory limit and a notebook kernel is not small. A pass that failed exits
  # before this, leaving its container for cleanup to dump.
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  NAME=""
}

run_session "default"

if [ "${SKIP_HARDENED}" = "1" ]; then
  warning="hardened pass skipped (SKIP_HARDENED=1), foreign-uid regressions will not be caught"
  WARNINGS+=("${warning}")
  log "WARNING: ${warning}"
else
  # The hardened pass reproduces what nbx-operator builds when
  # notebookSession.writableVolumes is enabled: a uid the image was not built with,
  # and an emptyDir over each path start_server.sh writes to. tmpfs is the local
  # stand-in for an emptyDir, mode 1777 for the ownership kubelet gives one.
  #
  # Group 0 rather than the image's own group: that is what a cluster-assigned uid
  # gets, and it keeps the pass honest about files the image group-owns.
  run_session "hardened uid ${SESSION_UID}" \
    --user "${SESSION_UID}:0" \
    --tmpfs /home/notebooks/.nbx-rw:rw,mode=1777 \
    --tmpfs /etc/authorized_keys:rw,mode=1777 \
    --tmpfs /etc/ssh/keys:rw,mode=1777
fi

print_summary
log "RESULT: PASSED"
