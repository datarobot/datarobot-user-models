"""FIPS 140-3 startup self-test (CFX-8623).

Run first by start_server.sh and start_server_drum.sh, so the container refuses to start unless the OpenSSL FIPS
provider passes its power-on self-test:

    python3 -S /etc/datarobot-fips/fips_selftest.py

It runs openssl-fips-test, which the Chainguard FIPS base image ships, and checks that Python itself is in FIPS mode.
The fips_guard watchdog next to this file keeps checking the running Python processes; this is the check at container
start that the watchdog does not make, and it also halts when the provider is disabled, which the watchdog treats as
"not in FIPS mode" and ignores.

Run it with -S. That skips the site module and with it sitecustomize, so the one-shot self-test does not start a
watchdog of its own or print the guard's startup lines a second time; fips_guard is imported from this directory,
which Python puts on sys.path for a script.

Environment:
    DATAROBOT_FIPS_ENFORCE  "true", "1", "yes" or "on" (any case) enables the check. Unset, empty, "false", "0", "no"
                            or "off" skips it and logs that it did. Any other value is a misconfiguration and halts,
                            so a typo can never switch FIPS off.

Pass lines go to stdout. A failure prints "[datarobot-fips] FATAL: <reason>" to stderr and exits with code 78
(decision 28 of the FIPS 140-3 programme). There is deliberately no "warn and continue" fallback.

Only openssl-fips-test is accepted: bssl-test_fips tests BoringSSL, a different module than the OpenSSL provider this
image uses, so it must not stand in for it.
"""

import os
import shutil
import subprocess
import sys
from typing import NoReturn

import fips_guard

ENV_ENFORCE = "DATAROBOT_FIPS_ENFORCE"
ENFORCE_VALUES = frozenset({"true", "1", "yes", "on"})
SKIP_VALUES = frozenset({"", "false", "0", "no", "off"})
EXIT_CODE = fips_guard.EXIT_CODE  # 78, EX_CONFIG: the same code the watchdog halts with
PREFIX = "[datarobot-fips]"
SELFTEST_TOOL = "openssl-fips-test"
# Upper bound on the self-test, so a hung tool (bad provider config, starved entropy) still ends in a FATAL line and
# exit 78 instead of leaving the container stuck in startup. Not measured on this image.
SELFTEST_TIMEOUT_SECONDS = 30


def log(message: str) -> None:
    print(f"{PREFIX} {message}", flush=True)


def fatal(reason: str) -> NoReturn:
    sys.stdout.flush()
    sys.stderr.write(f"{PREFIX} FATAL: {reason}\n")
    sys.stderr.flush()
    sys.exit(EXIT_CODE)


def enforcement_requested() -> bool:
    """True to run the self-test, False to skip it; halts on a value that is neither, so a typo can't disable FIPS."""
    raw = os.environ.get(ENV_ENFORCE, "")
    value = raw.strip().lower()
    if value in ENFORCE_VALUES:
        return True
    if value in SKIP_VALUES:
        return False
    fatal(f'{ENV_ENFORCE}="{raw}" is not a recognised value (use "true" or "false"); refusing to start')


def run_selftest_tool() -> None:
    """Run openssl-fips-test and halt unless it exits 0 in time."""
    tool_path = shutil.which(SELFTEST_TOOL)
    if tool_path is None:
        fatal(
            f"required FIPS self-test tool {SELFTEST_TOOL} not found in this image, so the OpenSSL FIPS provider's "
            "power-on self-test cannot run and FIPS compliance cannot be verified. Base the image on a FIPS-enabled "
            "OpenSSL image; refusing to start",
        )
    log(f"running FIPS self-test: {SELFTEST_TOOL}")
    sys.stdout.flush()
    try:
        # The tool's report goes to stderr (fd 2) so stdout keeps only our pass lines.
        result = subprocess.run(  # noqa: S603 - fixed tool, resolved from PATH above, no arguments
            [tool_path],
            stdin=subprocess.DEVNULL,
            stdout=2,
            stderr=2,
            timeout=SELFTEST_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired:
        fatal(f"FIPS self-test timed out after {SELFTEST_TIMEOUT_SECONDS}s ({SELFTEST_TOOL}); refusing to start")
    except OSError as exc:
        fatal(f"could not run FIPS self-test ({SELFTEST_TOOL}): {exc}; refusing to start")
    if result.returncode != 0:
        fatal(
            f"FIPS self-test failed ({SELFTEST_TOOL}, exit {result.returncode}). Non-approved cryptography may be "
            "reachable in this image; refusing to start",
        )
    log(f"FIPS self-test passed ({SELFTEST_TOOL})")


def check_interpreter() -> None:
    """Halt unless Python's OpenSSL runs in FIPS mode (the default properties include fips=yes)."""
    if not fips_guard.fips_mode():
        fatal(
            "Python's OpenSSL is not in FIPS mode, so hashlib and ssl are not restricted to the FIPS provider; "
            "refusing to start",
        )
    log("FIPS mode active in Python (hashlib and ssl use the FIPS provider)")


def run_startup_selftest() -> None:
    if not enforcement_requested():
        log(f"FIPS self-test skipped: {ENV_ENFORCE} is not enabled")
        return
    run_selftest_tool()
    check_interpreter()


if __name__ == "__main__":
    run_startup_selftest()
