# Copyright 2026 DataRobot, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Container-side runner for :class:`DataRobotWorkloadSandbox`.

This script runs *inside* the sandbox container, NOT in the caller's
process. It reads base64-encoded code and inputs from environment
variables, exec's the code with ``inputs`` bound, captures any value
assigned to ``_return``, and emits a ``__DR_SANDBOX_RESULT__:<json>``
marker as the final stdout line so the caller can extract the return
value.
"""

import base64
import json
import os
import signal
import sys
import traceback
from typing import Any

RESULT_MARKER = "__DR_SANDBOX_RESULT__:"

# ===========================================================================
#                  *** EXECUTION CAPPED AT 1 HOUR ***
#
# This sandbox enforces a HARD 1-HOUR (3600s) wall-clock on user code via
# SIGALRM. Any code still running after 60 minutes is interrupted, exits
# with code 124, and reports `__DR_SANDBOX_RESULT__:null`. If you need to
# run something longer, override with the DR_SANDBOX_TIMEOUT_SECS env var
# at container launch — but the caller / workload-api may also impose its
# own (lower) wall-clock limit that this runner cannot extend.
#
# This in-process timeout is defense-in-depth for accidental hangs only —
# it is NOT a security boundary. Malicious code can reset the SIGALRM
# handler; the real enforcement is the container lifecycle / workload-api.
# ===========================================================================
DEFAULT_TIMEOUT_SECS = 3600  # 1 hour


class _SandboxTimeout(Exception):
    """Raised when DR_SANDBOX_TIMEOUT_SECS elapses during exec()."""


def _on_alarm(_signum: int, _frame: Any) -> None:
    raise _SandboxTimeout("sandbox exceeded timeout")


def _decode_env(name: str, default: str) -> Any:
    raw = os.environ.get(name)
    if not raw:
        return json.loads(default)
    return json.loads(base64.b64decode(raw).decode("utf-8"))


def main() -> int:
    """Execute the user code from environment variables. Returns exit code."""
    code_b64 = os.environ.get("DR_SANDBOX_CODE_B64", "")
    if not code_b64:
        print("DR_SANDBOX_CODE_B64 not set", file=sys.stderr)
        return 1
    code = base64.b64decode(code_b64).decode("utf-8")
    inputs = _decode_env("DR_SANDBOX_INPUTS_B64", "{}")

    try:
        timeout_secs = int(os.environ.get("DR_SANDBOX_TIMEOUT_SECS", DEFAULT_TIMEOUT_SECS))
    except ValueError:
        timeout_secs = DEFAULT_TIMEOUT_SECS

    namespace: dict[str, Any] = {"inputs": inputs, "_return": None}
    exit_code = 0
    # Defense-in-depth wall-clock for accidental hangs (infinite loops, runaway
    # ops). Not a security boundary — malicious code can reset the handler;
    # caller-side / workload-api timeout is the real enforcement.
    if timeout_secs > 0:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(timeout_secs)
    # Catch BaseException so sys.exit() / KeyboardInterrupt from user code
    # don't bypass the marker emission below — without this, the caller
    # would see a "successful" run with no parseable result.
    try:
        exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
    except _SandboxTimeout:
        print(f"sandbox exceeded timeout of {timeout_secs}s", file=sys.stderr)
        exit_code = 124
    except SystemExit as exc:
        code_val = exc.code
        if code_val is None:
            exit_code = 0
        elif isinstance(code_val, int):
            exit_code = code_val
        else:
            print(str(code_val), file=sys.stderr)
            exit_code = 1
    except BaseException:  # noqa: BLE001 — must catch KeyboardInterrupt too
        traceback.print_exc()
        exit_code = 1
    finally:
        if timeout_secs > 0:
            signal.alarm(0)

    return_value = namespace.get("_return")
    try:
        encoded = json.dumps(return_value, default=str)
    except (TypeError, ValueError):
        encoded = json.dumps(repr(return_value))
    print(f"{RESULT_MARKER}{encoded}")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
