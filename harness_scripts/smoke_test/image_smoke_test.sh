#!/usr/bin/env bash
# Smoke test that the image of an environment has been published.
#
# Reads the image from env_info.json, waits for its tag to show up in the
# registry (images are built asynchronously after the PR is updated), then
# pulls it and prints its details.
#
# Only needs docker on the host. Log in to the registry beforehand if the
# image is private.
#
# Usage: image_smoke_test.sh <env_dir> [image]
#   env_dir  environment folder containing env_info.json,
#            e.g. public_dropin_notebook_environments/python313_notebook
#   image    image to check; defaults to
#            docker.io/datarobotdev/<imageRepository>:<environmentVersionId>
#
# Optional env vars:
#   MAXWAIT   seconds to wait for the tag to be published (default 1800)
#   INTERVAL  seconds between registry checks (default 30)

set -euo pipefail

ENV_DIR="${1:?Usage: image_smoke_test.sh <env_dir> [image]}"
IMAGE="${2:-}"
MAXWAIT="${MAXWAIT:-1800}"
INTERVAL="${INTERVAL:-30}"

CURRENT_CHECK="setup"
PASSED_CHECKS=()

log() { echo "[image-smoke] $*"; }
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
fail() { echo "[image-smoke]   FAIL: $*" >&2; exit 1; }

print_summary() {
  echo
  log "===== Summary for ${ENV_DIR} ====="
  local c
  for c in "${PASSED_CHECKS[@]+"${PASSED_CHECKS[@]}"}"; do
    log "  [PASS] ${c}"
  done
}

on_exit() {
  local retval=$?
  if [ "${retval}" -ne 0 ]; then
    print_summary
    log "  [FAIL] ${CURRENT_CHECK}"
    log "RESULT: FAILED at check: ${CURRENT_CHECK}"
  fi
  exit "${retval}"
}
trap on_exit EXIT

log "Environment: ${ENV_DIR}"

if [ -z "${IMAGE}" ]; then
  env_info="${ENV_DIR}/env_info.json"
  check "read the image from ${env_info}"
  [ -f "${env_info}" ] || fail "${env_info} not found"
  # sed instead of python/jq so the host needs nothing but docker
  repo=$(sed -n 's/.*"imageRepository": *"\([^"]*\)".*/\1/p' "${env_info}")
  tag=$(sed -n 's/.*"environmentVersionId": *"\([^"]*\)".*/\1/p' "${env_info}")
  [ -n "${repo}" ] || fail "no imageRepository in ${env_info}"
  [ -n "${tag}" ] || fail "no environmentVersionId in ${env_info}"
  IMAGE="docker.io/datarobotdev/${repo}:${tag}"
fi
log "  image: ${IMAGE}"

check "image tag is published (waiting up to ${MAXWAIT}s)"
start=${SECONDS}
until manifest_error=$(docker manifest inspect "${IMAGE}" 2>&1 >/dev/null); do
  waited=$((SECONDS - start))
  if [ "${waited}" -ge "${MAXWAIT}" ]; then
    fail "tag not found after ${waited}s; last error: ${manifest_error}"
  fi
  log "  not published yet after ${waited}s, checking again in ${INTERVAL}s"
  sleep "${INTERVAL}"
done
log "  published, found after $((SECONDS - start))s"

check "image can be pulled"
docker pull "${IMAGE}" || fail "docker pull ${IMAGE} failed"

check "image metadata"
log "  $(docker image inspect --format 'id {{.Id}}, {{.Os}}/{{.Architecture}}, created {{.Created}}, size {{.Size}} bytes' "${IMAGE}")" \
  || fail "docker image inspect ${IMAGE} failed"

finish_check
print_summary
log "RESULT: PASSED"
