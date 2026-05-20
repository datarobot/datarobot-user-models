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
import sys
import traceback
from typing import Any

RESULT_MARKER = "__DR_SANDBOX_RESULT__:"


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

    namespace: dict[str, Any] = {"inputs": inputs, "_return": None}
    try:
        exec(compile(code, "<sandbox>", "exec"), namespace)  # noqa: S102
    except Exception:
        traceback.print_exc()
        return 1

    return_value = namespace.get("_return")
    try:
        encoded = json.dumps(return_value, default=str)
    except (TypeError, ValueError):
        encoded = json.dumps(repr(return_value))
    print(f"{RESULT_MARKER}{encoded}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
