#!/usr/bin/env bash
# Checks the FIPS startup self-test baked into an image (/etc/datarobot-fips/fips_selftest.sh, run first by the
# image's start scripts): with DATAROBOT_FIPS_ENFORCE=true it prints its pass lines and exits 0, with an unrecognised
# value or the FIPS provider disabled it exits 78 with a [datarobot-fips] FATAL line.
# Images without the self-test are skipped. Needs docker; the image must already be pulled or pullable.
#
# Usage: fips_selftest_check.sh <image>

set -euo pipefail

IMAGE="${1:?Usage: fips_selftest_check.sh <image>}"

log() { echo "[fips-selftest] $*"; }

if ! docker run --rm --entrypoint sh "${IMAGE}" -c 'test -f /etc/datarobot-fips/fips_selftest.sh'; then
  log "SKIP: no /etc/datarobot-fips/fips_selftest.sh in ${IMAGE}"
  exit 0
fi

# run_selftest <env args...>: runs the self-test in the image with the given docker env args; sets rc and output.
run_selftest() {
  set +e
  output=$(docker run --rm --entrypoint sh "$@" "${IMAGE}" /etc/datarobot-fips/fips_selftest.sh 2>&1)
  rc=$?
  set -e
  echo "${output}"
}

log "enforced, healthy provider: expect the pass lines and exit 0"
run_selftest -e DATAROBOT_FIPS_ENFORCE=true
if [ "${rc}" -ne 0 ]; then
  log "FAIL: exit code ${rc}, expected 0"
  exit 1
fi
for line in '[datarobot-fips] FIPS self-test passed (openssl-fips-test)'; do
  if ! grep -qF "${line}" <<<"${output}"; then
    log "FAIL: missing line: ${line}"
    exit 1
  fi
done

log "unrecognised value: expect FATAL and exit 78"
run_selftest -e DATAROBOT_FIPS_ENFORCE=maybe
if [ "${rc}" -ne 78 ] || ! grep -q '^\[datarobot-fips\] FATAL' <<<"${output}"; then
  log "FAIL: exit code ${rc}, expected 78 with a [datarobot-fips] FATAL line"
  exit 1
fi

log "enforced, FIPS provider disabled (OPENSSL_CONF=/dev/null): expect FATAL and exit 78"
run_selftest -e DATAROBOT_FIPS_ENFORCE=true -e OPENSSL_CONF=/dev/null
if [ "${rc}" -ne 78 ] || ! grep -q '^\[datarobot-fips\] FATAL' <<<"${output}"; then
  log "FAIL: exit code ${rc}, expected 78 with a [datarobot-fips] FATAL line"
  exit 1
fi

log "OK: ${IMAGE} runs the FIPS self-test and halts with exit 78 when it fails"
