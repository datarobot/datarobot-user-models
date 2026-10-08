#!/usr/bin/env bash
# Smoke test that the image of an environment has been built and published.
#
# Images are built asynchronously after the PR is updated, and the image tag
# (environmentVersionId) stays the same across PR commits, so an existing tag
# may still hold the image built for an earlier commit. With COMMIT_SHA set,
# the script first waits for the image build's GitHub status on that commit
# ("ExecEnv Build: <env_dir>") to be success, and fails as soon as it fails.
# Then it reads the image from env_info.json, checks that its tag is published,
# pulls it and prints its details.
#
# Needs docker on the host, plus curl and jq with COMMIT_SHA. Log in to the
# registry beforehand if the image is private.
#
# Usage: image_smoke_test.sh <env_dir> [image]
#   env_dir  environment folder containing env_info.json,
#            e.g. public_dropin_notebook_environments/python313_notebook
#   image    image to check; defaults to
#            docker.io/datarobotdev/<imageRepository>:<environmentVersionId>
#
# Optional env vars:
#   COMMIT_SHA            commit to wait for the image build on; without it the
#                         build status is not checked, only the tag
#   GH_TOKEN              GitHub token to read commit statuses; required with COMMIT_SHA
#   GITHUB_REPO           repo of COMMIT_SHA (default datarobot/datarobot-user-models)
#   BUILD_STATUS_CONTEXT  build status name (default "ExecEnv Build: <env_dir>")
#   MAXWAIT               seconds to wait for the build and for the tag (default 1800)
#   INTERVAL              seconds between checks (default 30)

set -euo pipefail

ENV_DIR="${1:?Usage: image_smoke_test.sh <env_dir> [image]}"
IMAGE="${2:-}"
COMMIT_SHA="${COMMIT_SHA:-}"
GITHUB_REPO="${GITHUB_REPO:-datarobot/datarobot-user-models}"
MAXWAIT="${MAXWAIT:-1800}"
INTERVAL="${INTERVAL:-30}"

# Env path relative to the repo root, as used in the build status name
ENV_PATH="${ENV_DIR#./}"
ENV_PATH="${ENV_PATH%/}"
BUILD_STATUS_CONTEXT="${BUILD_STATUS_CONTEXT:-ExecEnv Build: ${ENV_PATH}}"

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
log "Commit:      ${COMMIT_SHA:-not set}"

if [ -n "${COMMIT_SHA}" ]; then
  check "'${BUILD_STATUS_CONTEXT}' status is success on ${COMMIT_SHA} (waiting up to ${MAXWAIT}s)"
  [ -n "${GH_TOKEN:-}" ] || fail "GH_TOKEN is required with COMMIT_SHA"
  command -v jq >/dev/null || fail "jq is required with COMMIT_SHA"
  start=${SECONDS}
  last_error=""
  while true; do
    waited=$((SECONDS - start))
    state=""
    build_url=""
    # The combined status holds the latest status of every context on the commit
    if response=$(curl --silent --show-error --fail -L \
      -H "Accept: application/vnd.github+json" \
      -H "Authorization: Token ${GH_TOKEN}" \
      "https://api.github.com/repos/${GITHUB_REPO}/commits/${COMMIT_SHA}/status?per_page=100" 2>&1); then
      state=$(jq -r --arg ctx "${BUILD_STATUS_CONTEXT}" \
        'first(.statuses[] | select(.context == $ctx) | .state) // ""' <<<"${response}")
      build_url=$(jq -r --arg ctx "${BUILD_STATUS_CONTEXT}" \
        'first(.statuses[] | select(.context == $ctx) | .target_url) // ""' <<<"${response}")
    else
      last_error="${response}"
      log "  GitHub API error, will retry: ${last_error}"
    fi
    case "${state}" in
      success)
        log "  build succeeded after ${waited}s: ${build_url}"
        break
        ;;
      failure | error)
        fail "image build ${state}: ${build_url}"
        ;;
    esac
    if [ "${waited}" -ge "${MAXWAIT}" ]; then
      if [ -n "${state}" ]; then
        fail "build still ${state} after ${waited}s: ${build_url}"
      fi
      fail "no '${BUILD_STATUS_CONTEXT}' status on ${COMMIT_SHA} after ${waited}s; the image was not built for this commit${last_error:+; last API error: ${last_error}}"
    fi
    log "  build ${state:-not reported yet} after ${waited}s, checking again in ${INTERVAL}s"
    sleep "${INTERVAL}"
  done
else
  log "SKIP: image build status (COMMIT_SHA not set); the tag may hold an image built for an earlier commit"
fi

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
