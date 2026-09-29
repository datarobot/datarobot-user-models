#!/usr/bin/env python
"""Contract check for the [DataRobot] Python 3.x Pipelines environment.

Run inside the built image: ``python /opt/pipelines-env/smoke_test.py``.
Exit 0 means the image satisfies what pipelines-api layers onto it.
"""

from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys


def main() -> int:
    root = os.environ["INSTALLROOT"]
    failures: list[str] = []

    for module in ("covalent", "covalent_cloud", "cloudpickle", "datarobot", "boto3", "yaml", "uv"):
        try:
            importlib.import_module(module)
        except Exception as exc:  # noqa: BLE001 - report every failure, not the first
            failures.append(f"import {module}: {exc!r}")

    if not os.path.isfile(os.path.join(root, "constraints.txt")):
        failures.append("constraints.txt missing")

    clone = os.path.join(root, "bin", "cloudpickle")
    for tag in ("v2.2.1", "v3.1.2"):
        proc = subprocess.run(["git", "-C", clone, "checkout", "--force", tag], capture_output=True, text=True)
        if proc.returncode != 0:
            failures.append(f"git checkout {tag}: {proc.stderr.strip()[:200]}")

    try:
        import cloudpickle

        def add(x: int, y: int) -> int:
            return x + y

        if pickle.loads(cloudpickle.dumps(add))(2, 3) != 5:
            failures.append("cloudpickle round-trip returned a wrong result")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"cloudpickle round-trip: {exc!r}")

    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print(f"ok: python {sys.version.split()[0]}, cloudpickle {cloudpickle.__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
