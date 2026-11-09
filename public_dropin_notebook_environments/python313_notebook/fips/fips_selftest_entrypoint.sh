#!/bin/sh
# FIPS self-test entrypoint wrapper.
#
# Runs before the container's real entrypoint. Verifies that FIPS-approved
# crypto is actually in effect and refuses to start the real entrypoint
# otherwise. POSIX sh on purpose.
#
# Environment:
#   DATAROBOT_FIPS_ENFORCE   "true" enables the check; anything else makes this
#                            wrapper a transparent exec of the original
#                            entrypoint.
#   DATAROBOT_FIPS_FALLBACK  What to do when no known self-test tool exists in
#                            the image: "fail" (default) refuses to start;
#                            "warn" logs and continues.

set -u

log() {
    echo "[datarobot-fips] $*" >&2
}

run_selftest() {
    # Chainguard FIPS images ship openssl-fips-test; BoringSSL-based runtimes
    # ship bssl-test_fips. The first tool found decides the outcome.
    for tool in openssl-fips-test bssl-test_fips; do
        if command -v "$tool" >/dev/null 2>&1; then
            log "running FIPS self-test: $tool"
            if "$tool" 1>&2; then
                log "FIPS self-test passed ($tool)"
                return 0
            fi
            log "FATAL: FIPS self-test failed ($tool). Non-approved" \
                "cryptography may be reachable in this image; refusing to" \
                "start the workload."
            return 1
        fi
    done
    return 2
}

if [ "${DATAROBOT_FIPS_ENFORCE:-false}" = "true" ]; then
    run_selftest
    case $? in
        0) ;;
        1) exit 78 ;; # EX_CONFIG: the image's crypto configuration is invalid
        2)
            if [ "${DATAROBOT_FIPS_FALLBACK:-fail}" = "warn" ]; then
                log "WARNING: no FIPS self-test tool found in this image;" \
                    "continuing because DATAROBOT_FIPS_FALLBACK=warn."
            else
                log "FATAL: no FIPS self-test tool (openssl-fips-test," \
                    "bssl-test_fips) found in this image, so FIPS compliance" \
                    "cannot be verified. Base the image on a FIPS-enabled" \
                    "image, or set DATAROBOT_FIPS_FALLBACK=warn to bypass."
                exit 78
            fi
            ;;
    esac
fi

if [ "$#" -eq 0 ]; then
    log "FATAL: no original entrypoint was passed to the FIPS wrapper."
    exit 78
fi

exec "$@"
