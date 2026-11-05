#!/usr/bin/env bash
# Checks that the FIPS guard baked into an image (/etc/datarobot-fips, loaded through sitecustomize) halts
# python when the FIPS provider enters its error state: exit code 78 and a [datarobot-fips] FATAL line.
# Images without the guard are skipped. Needs docker; the image must already be pulled or pullable.
#
# Usage: fips_guard_check.sh <image>

set -euo pipefail

IMAGE="${1:?Usage: fips_guard_check.sh <image>}"

log() { echo "[fips-guard] $*"; }

if ! docker run --rm --entrypoint sh "${IMAGE}" -c 'test -f /etc/datarobot-fips/sitecustomize.py'; then
  log "SKIP: no /etc/datarobot-fips/sitecustomize.py in ${IMAGE}"
  exit 0
fi

# Breaks the provider the way the guide's inject.so does, from inside the process: re-runs the provider
# self-tests with every known-answer test failing, then waits for the guard's watchdog to exit the process.
read -r -d '' BREAK_PROVIDER <<'PY' || true
import ctypes, os, ssl, sys, threading, time
assert "fips_guard" in sys.modules, "sitecustomize did not load fips_guard"
assert any(t.name == "fips-guard" for t in threading.enumerate()), "no fips-guard watchdog thread"
lib = ctypes.CDLL(f"libcrypto.so.{ssl.OPENSSL_VERSION_INFO[0]}", mode=os.RTLD_NOLOAD)
lib.OSSL_PROVIDER_load.restype = ctypes.c_void_p
lib.OSSL_PROVIDER_load.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.OSSL_PROVIDER_self_test.argtypes = [ctypes.c_void_p]
cb = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(lambda p, a: 0)
prov = lib.OSSL_PROVIDER_load(None, b"fips")
assert prov, "no FIPS provider loaded"
lib.OSSL_SELF_TEST_set_callback(None, cb)
lib.OSSL_PROVIDER_self_test(prov)
time.sleep(20)
print("guard did not halt the process", flush=True)
sys.exit(1)
PY

set +e
output=$(docker run --rm --entrypoint python "${IMAGE}" -c "${BREAK_PROVIDER}" 2>&1)
rc=$?
set -e
echo "${output}"
if [ "${rc}" -ne 78 ]; then
  log "FAIL: exit code ${rc}, expected 78"
  exit 1
fi
if ! grep -q '^\[datarobot-fips\] FATAL' <<<"${output}"; then
  log "FAIL: no [datarobot-fips] FATAL line"
  exit 1
fi
log "OK: ${IMAGE} halted with exit 78 and the FATAL line"
