#!/bin/sh
# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.
#
# FIPS 140-3 startup self-test (CFX-8623). Run first by start_server.sh and start_server_drum.sh, so the container
# refuses to start unless the OpenSSL FIPS provider passes its power-on self-test.
#
# It runs openssl-fips-test, which the Chainguard FIPS base image ships. The fips_guard watchdog next to this file
# keeps checking the running Python processes; this script is the check at container start that the watchdog does not
# make, and it also halts when the provider is disabled, which the watchdog treats as "not in FIPS mode".
#
# Environment:
#   DATAROBOT_FIPS_ENFORCE  "true", "1", "yes" or "on" (any case) enables the check. Unset, empty, "false", "0", "no"
#                           or "off" skips it and logs that it did. Any other value is a misconfiguration and halts,
#                           so a typo can never switch FIPS off.
#
# Pass lines go to stdout. A failure prints "[datarobot-fips] FATAL: <reason>" to stderr and exits with code 78
# (decision 28 of the FIPS 140-3 programme). There is deliberately no "warn and continue" fallback.
#
# Only openssl-fips-test is accepted: bssl-test_fips tests BoringSSL, a different module than the OpenSSL provider
# this image uses, so it must not stand in for it.

PREFIX='[datarobot-fips]'
SELFTEST_TOOL='openssl-fips-test'
# Upper bound on the self-test, so a hung tool (bad provider config, starved entropy) still ends in a FATAL line and
# exit 78 instead of leaving the container stuck in startup. Not measured on this image.
TIMEOUT_SECONDS=30
EX_CONFIG=78

log() {
    printf '%s %s\n' "${PREFIX}" "$*"
}

fatal() {
    printf '%s FATAL: %s\n' "${PREFIX}" "$*" >&2
    exit "${EX_CONFIG}"
}

raw="${DATAROBOT_FIPS_ENFORCE:-}"
value=$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
case "${value}" in
    true | 1 | yes | on) ;;
    '' | false | 0 | no | off)
        log 'FIPS self-test skipped: DATAROBOT_FIPS_ENFORCE is not enabled'
        exit 0
        ;;
    *)
        fatal "DATAROBOT_FIPS_ENFORCE=\"${raw}\" is not a recognised value (use \"true\" or \"false\"); refusing to start"
        ;;
esac

tool_path=$(command -v "${SELFTEST_TOOL}") ||
    fatal "required FIPS self-test tool ${SELFTEST_TOOL} not found in this image, so the OpenSSL FIPS provider's power-on self-test cannot run and FIPS compliance cannot be verified. Base the image on a FIPS-enabled OpenSSL image; refusing to start"

command -v timeout >/dev/null 2>&1 ||
    fatal "timeout not found in this image, so the FIPS self-test cannot be bounded in time; refusing to start"

log "running FIPS self-test: ${SELFTEST_TOOL}"
# The tool's report goes to stderr so stdout keeps only our pass lines.
timeout "${TIMEOUT_SECONDS}" "${tool_path}" >&2
status=$?
case "${status}" in
    0) ;;
    124) fatal "FIPS self-test timed out after ${TIMEOUT_SECONDS}s (${SELFTEST_TOOL}); refusing to start" ;;
    126 | 127) fatal "could not run FIPS self-test (${SELFTEST_TOOL}, exit ${status}); refusing to start" ;;
    *) fatal "FIPS self-test failed (${SELFTEST_TOOL}, exit ${status}). Non-approved cryptography may be reachable in this image; refusing to start" ;;
esac

log "FIPS self-test passed (${SELFTEST_TOOL})"
