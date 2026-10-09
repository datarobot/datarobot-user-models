#!/usr/bin/env bash
# Smoke test for a notebook environment image.
#
# Starts the image the way the Notebooks platform does (start_server.sh as the
# image user, with SSH keys mounted under /var/run/notebooks/ssh) and checks
# that every service comes up and a kernel can run code:
#   - the kernel gateway answers on /api
#   - code runs in the prespawned kernel and in a newly created kernel
#   - in the kernels, the dataframe extension renders a DataFrame and %pip installs a package
#   - drgithelper runs, sshd and the monitoring agent are up
#   - a login shell gets the session env from the profile env file
#   - an ssh login on port 8022 works and gets the session env
#   - the git credential cache daemon starts
#   - the container log has no Go panics or Python tracebacks
#   - "Permission denied" in the kernel output, drgithelper or the container log is a warning
#
# Those checks run twice: once as the image's own user with no extra mounts, which is what
# every install gets, and once at a uid the image was not built with plus writable volumes
# over the paths start_server.sh writes to, which is what nbx-operator produces when
# notebookSession.writableVolumes is enabled and where an image built for one uid comes up
# broken while the container still looks healthy (FLEET-8918).
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
#   SESSION_UID      uid for the hardened pass (default 1500), any value the image was not
#                    built with and absent from /etc/passwd, as a cluster-assigned uid is
#   SKIP_HARDENED    set to 1 to run the default pass only
#   SSH_CLIENT_IMAGE image for the ssh login check (default alpine:3.22); it shares the
#                    container's network and gets openssh-client from apk

set -euo pipefail

ENV_DIR="${1:?Usage: notebook_smoke_test.sh <env_dir> [image]}"
IMAGE="${2:-}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"
SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC:-0}"
COMMIT_SHA="${COMMIT_SHA:-}"
SESSION_UID="${SESSION_UID:-1500}"
SKIP_HARDENED="${SKIP_HARDENED:-0}"
SSH_CLIENT_IMAGE="${SSH_CLIENT_IMAGE:-alpine:3.22}"

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
# NAME is the pass currently running, the one cleanup dumps diagnostics for, and CONTAINERS
# is every container to remove on the way out.
NAME=""
CONTAINERS=()
# Output of each pass, scanned for "Permission denied" at the end of the pass
OUTPUT_DIR=$(mktemp -d)
# start_server.sh persists the container env into the profile env file, so login shells and
# ssh sessions must see this variable
SESSION_MARKER="nbx-smoke-${RANDOM}"

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
  rm -rf "${OUTPUT_DIR}"
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
# Starts the image the way the platform does and runs every per-session check against it,
# prefixing check names with the pass so the summary says which container failed.
run_session() {
  local pass="${1}"
  shift
  local processes output marker ssh_output

  NAME="${RUN_ID}-$(tr -cd 'a-z0-9' <<<"${pass}" | cut -c1-12)"
  CONTAINERS+=("${NAME}")
  output="${OUTPUT_DIR}/${NAME}.log"

  echo
  log "================ PASS: ${pass} ================"

  check "[${pass}] container starts via ${KERNEL_DIR}/start_server.sh"
  docker run --detach --name "${NAME}" \
    --volume "${SSH_VOLUME}:/var/run/notebooks/ssh:ro" \
    --env NBX_SMOKE_MARKER="${SESSION_MARKER}" \
    ${@+"${@}"} \
    --entrypoint /bin/bash "${IMAGE}" "${KERNEL_DIR}/start_server.sh" >/dev/null \
    || fail "docker run failed"
  log "  container: ${NAME}"
  log "  running as: $(docker exec "${NAME}" id 2>/dev/null || echo 'could not read id')"

  check "[${pass}] kernel gateway answers and kernels run code"
  # Runs inside the container with the kernel venv, which ships tornado, printing sub-checks
  # whose last "CHECK:" line before a failure is the one that failed, and in the hardened pass
  # it is also what proves the kernel still imports its packages with that venv read-only.
  docker exec -i \
    -e STARTUP_TIMEOUT="${STARTUP_TIMEOUT}" \
    -e SKIP_KERNEL_EXEC="${SKIP_KERNEL_EXEC}" \
    -e KERNEL_CHECK_CODE="$(cat "${KERNEL_CHECK}")" \
    -e REQUIRED_MODULES="${REQUIRED_MODULES}" \
    "${NAME}" "${KERNEL_PYTHON}" - <"${GATEWAY_CHECK}" 2>&1 | tee -a "${output}" \
    || fail "kernel gateway check failed, see the last sub-check above"

  check "[${pass}] drgithelper --version runs (no Go panic, libcrypto found)"
  docker exec -e HOME=/home/notebooks "${NAME}" "${KERNEL_DIR}/drgithelper" --version 2>&1 \
    | tee -a "${output}" || fail "drgithelper does not run"

  processes=$(docker exec "${NAME}" ps -eo args)
  check "[${pass}] sshd is running"
  # sshd only reaches this state if start_server.sh got a host key into /etc/ssh/keys, which
  # a plain mkdir over a mounted volume would have silently prevented.
  grep "sshd -D" <<<"${processes}" | sed 's/^/[notebook-smoke]   /' || fail "no 'sshd -D' process"

  check "[${pass}] monitoring agent (uvicorn agent:app) is running"
  grep "uvicorn agent:app" <<<"${processes}" | sed 's/^/[notebook-smoke]   /' || fail "no 'uvicorn agent:app' process"

  check "[${pass}] login shell gets the session env from the profile env file"
  # env -i drops the env docker exec passes, so only /etc/profile.d can set the marker
  marker=$(docker exec "${NAME}" env -i HOME=/home/notebooks bash -lc 'echo "${NBX_SMOKE_MARKER:-}"' 2>&1 \
    | tee -a "${output}" | tail -1) || true
  [ "${marker}" = "${SESSION_MARKER}" ] \
    || fail "login shell has NBX_SMOKE_MARKER='${marker}', expected '${SESSION_MARKER}'"

  check "[${pass}] ssh login on port 8022 works and gets the session env"
  # The image has no ssh client, so it runs in a container on this container's network. The
  # key is the generated host key, whose public half is the authorized key for "notebooks".
  ssh_output=$(docker run --rm --network "container:${NAME}" --volume "${SSH_VOLUME}:/ssh:ro" \
    --env REMOTE_CMD='bash -lc '\''id; echo "${NBX_SMOKE_MARKER:-}"'\''' \
    "${SSH_CLIENT_IMAGE}" sh -c '
      apk add --quiet --no-progress openssh-client >/dev/null &&
      cp /ssh/keys/ssh_host_key /tmp/key && chmod 600 /tmp/key &&
      ssh -i /tmp/key -p 8022 -o BatchMode=yes -o ConnectTimeout=20 -o LogLevel=ERROR \
        -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        notebooks@127.0.0.1 "${REMOTE_CMD}"' 2>&1) || true
  tee -a "${output}" <<<"${ssh_output}" | sed 's/^/[notebook-smoke]   /'
  [ "$(tail -1 <<<"${ssh_output}")" = "${SESSION_MARKER}" ] \
    || fail "ssh session did not print NBX_SMOKE_MARKER='${SESSION_MARKER}' (output above)"

  check "[${pass}] git credential cache daemon starts"
  # credential.helper starts with "cache", and its socket lives under XDG_CACHE_HOME or ~/.cache
  docker exec "${NAME}" env -i HOME=/home/notebooks bash -lc '
    cd /tmp &&
    printf "protocol=https\nhost=smoke.invalid\nusername=u\npassword=p\n\n" | git credential-cache --timeout 30 store &&
    printf "protocol=https\nhost=smoke.invalid\n\n" | git credential-cache get | grep -q "^password=p$" &&
    git credential-cache exit
  ' 2>&1 | tee -a "${output}" | sed 's/^/[notebook-smoke]   /' || fail "git credential-cache cannot store and get a credential"

  check "[${pass}] container log has no Go panics or Python tracebacks"
  docker logs "${NAME}" >>"${output}" 2>&1
  if docker logs "${NAME}" 2>&1 | grep -n -A 5 -E "^panic:|Traceback \(most recent call last\)"; then
    fail "found panics or tracebacks in the container log (shown above)"
  fi

  # Not a failure: a session can work around an unwritable path, but each one is worth a look
  grep -iE "permission denied|errno 13" "${output}" | sed 's/^ *//' | sort -u >"${output}.denied" || true
  while IFS= read -r line; do
    WARNINGS+=("[${pass}] ${line}")
    log "  WARNING: ${line}"
  done <"${output}.denied"

  finish_check
  # Freed once the pass is green so two kernels never run at once against the step's memory
  # limit, while a failed pass exits before this and leaves its container for cleanup to dump.
  docker rm -f "${NAME}" >/dev/null 2>&1 || true
  NAME=""
}

run_session "default"

if [ "${SKIP_HARDENED}" = "1" ]; then
  warning="hardened pass skipped (SKIP_HARDENED=1), foreign-uid regressions will not be caught"
  WARNINGS+=("${warning}")
  log "WARNING: ${warning}"
else
  # Reproduces what nbx-operator builds when notebookSession.writableVolumes is enabled: a
  # foreign uid in group 0 as a cluster assigns, and a tmpfs at mode 1777 standing in for the
  # emptyDir kubelet mounts over each path start_server.sh writes to. exec, because an emptyDir
  # allows it and docker's tmpfs doesn't by default, which would hide the user venv's pip.
  run_session "hardened uid ${SESSION_UID}" \
    --user "${SESSION_UID}:0" \
    --tmpfs /home/notebooks/.nbx-rw:rw,exec,mode=1777 \
    --tmpfs /etc/authorized_keys:rw,mode=1777 \
    --tmpfs /etc/ssh/keys:rw,mode=1777
fi

print_summary
log "RESULT: PASSED"
