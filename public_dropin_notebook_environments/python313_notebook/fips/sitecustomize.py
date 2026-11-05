"""Runs at every Python start in this image (sitecustomize on PYTHONPATH) and halts the process when the FIPS crypto
module fails. One mechanism for the kernel gateway, every kernel, the agent server and anything a user runs. The
check only runs in FIPS mode; see fips_guard.py next to this file."""

import os
import sys

import fips_guard

if fips_guard.fips_mode():
    # Nothing here does gRPC TLS, so stock grpcio in user code is a warning, not a halt. No log handlers: a handler
    # on the root logger this early makes the app's later logging.basicConfig() a no-op and silences its logs.
    fips_guard.start(bundled_grpc="warn", loggers=())

    def _after_fork() -> None:
        # gunicorn (the agent server) replaces a worker that exits 78; 3 makes the arbiter stop the whole pod.
        fips_guard.restart_watchdog(
            exit_code=3 if "gunicorn" in sys.modules else fips_guard.EXIT_CODE,
        )

    os.register_at_fork(after_in_child=_after_fork)
