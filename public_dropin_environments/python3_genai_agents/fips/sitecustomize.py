"""Imported at every Python start in this image (sitecustomize on PYTHONPATH): halts the process when the FIPS
crypto module fails. See fips_guard.py next to this file.

DATAROBOT_FIPS_ENFORCE, the switch of the shell self-test wrappers: "false" turns the guard off even in FIPS mode,
"true" also halts when OpenSSL is not in FIPS mode (decision 22)."""

import os
import sys

import fips_guard

enforce = os.environ.get("DATAROBOT_FIPS_ENFORCE")

if enforce != "false" and fips_guard.fips_mode():
    # warn: nothing here does gRPC TLS. loggers=(): a root handler this early makes the app's logging.basicConfig()
    # a no-op. log_startup: children inherit the variable, so the entry point logs once and kernels, pip, uv stay
    # quiet; checks and the FATAL line still run everywhere.
    fips_guard.start(
        bundled_grpc="warn",
        loggers=(),
        log_startup=os.environ.get("DATAROBOT_FIPS_GUARD_LOGGED") != "1",
    )
    os.environ["DATAROBOT_FIPS_GUARD_LOGGED"] = "1"

    # Go binaries (pulumi, gh, task) have their own crypto. fips140=on runs Go's self-tests and puts crypto/tls in
    # FIPS settings; non-approved algorithms called directly still work (only fips140=only blocks them, best effort).
    # An existing fips140= value, including off, is replaced.
    godebug = os.environ.get("GODEBUG", "").split(",")
    godebug = [kv for kv in godebug if kv and not kv.startswith("fips140=")]
    os.environ["GODEBUG"] = ",".join(godebug + ["fips140=on"])

    def _after_fork() -> None:
        # gunicorn replaces a worker that exits 78; 3 stops the whole pod.
        fips_guard.restart_watchdog(
            exit_code=3 if "gunicorn" in sys.modules else fips_guard.EXIT_CODE,
        )

    os.register_at_fork(after_in_child=_after_fork)
elif enforce == "true":
    fips_guard.halt("DATAROBOT_FIPS_ENFORCE=true but OpenSSL is not in FIPS mode")
