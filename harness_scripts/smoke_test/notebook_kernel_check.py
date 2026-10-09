"""Code that notebook_gateway_check.py runs in every kernel of a notebook image.

Defines run_checks(); notebook_gateway_check.py appends a call to it with the
env's required modules from notebook_required_modules.txt. Each check prints its
own line, so the log shows which import or call failed. hashlib.md5 and dask
tokenize (MD5) catch a base image that blocks non-FIPS digests.

The dataframe formatter and pip install checks only run inside a kernel, where
they are what proves a session at a uid the image was not built with still
renders dataframes and can install a package from a cell (FLEET-8918).

To run it directly with an image's python (the call must start a new,
unindented line):

python3 -c "$(cat notebook_kernel_check.py)
run_checks(['numpy', 'pandas'])"
"""

import hashlib
import importlib
import os
import ssl
import subprocess
import sys
import tempfile
import zipfile

DATAFRAME_MIME = "application/vnd.dataframe+json"
SMOKE_PKG = "nbx_smoke_pkg"


def _get_ipython():
    """The running kernel's InteractiveShell, or None when run outside one."""
    try:
        from IPython import get_ipython
    except ImportError:
        return None
    return get_ipython()


def check_dataframe_formatter(ipython):
    """The dataframe_formatter extension is loaded and renders a DataFrame.

    ipython_config.py loads the extension from IPYTHONDIR/extensions, which only
    imports while that directory is on PYTHONPATH; IPython warns and carries on
    when it is not, so a kernel with no dataframe rendering still looks healthy.
    """
    import pandas

    formatters = ipython.display_formatter.formatters
    assert DATAFRAME_MIME in formatters, (
        f"{DATAFRAME_MIME} is not registered, so dataframes render as plain text; "
        f"registered: {sorted(formatters)}"
    )
    data, _ = ipython.display_formatter.format(
        pandas.DataFrame({"a": [1, 2]}), include=[DATAFRAME_MIME]
    )
    assert DATAFRAME_MIME in data, f"{DATAFRAME_MIME} produced no output for a DataFrame"


def check_pip_install():
    """pip install works from a cell, into whichever venv this kernel runs on.

    Installs a wheel built here rather than one from an index, so the check needs
    no network and fails only on the venv being read-only.
    """
    with tempfile.TemporaryDirectory() as tmp:
        wheel = os.path.join(tmp, f"{SMOKE_PKG}-0.0.1-py3-none-any.whl")
        dist_info = f"{SMOKE_PKG}-0.0.1.dist-info"
        with zipfile.ZipFile(wheel, "w") as z:
            z.writestr(f"{SMOKE_PKG}.py", "VALUE = 'nbx-smoke'\n")
            z.writestr(
                f"{dist_info}/METADATA",
                f"Metadata-Version: 2.1\nName: {SMOKE_PKG}\nVersion: 0.0.1\n",
            )
            z.writestr(
                f"{dist_info}/WHEEL",
                "Wheel-Version: 1.0\nGenerator: smoke\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            )
            z.writestr(f"{dist_info}/RECORD", "")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             "--disable-pip-version-check", "--no-input", wheel],
            capture_output=True,
            text=True,
        )
    assert result.returncode == 0, (
        f"pip install failed with {result.returncode}; this is what a user gets when they "
        f"install from a cell:\n{result.stdout}\n{result.stderr}"
    )
    importlib.invalidate_caches()
    module = importlib.import_module(SMOKE_PKG)
    assert module.VALUE == "nbx-smoke", f"installed package imported wrong: {module.VALUE!r}"
    return os.path.dirname(module.__file__)


def run_checks(required_modules):
    print(f"python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}")

    for name in required_modules:
        print(f"import {name} ...", end=" ")
        module = importlib.import_module(name)
        print("OK", getattr(module, "__version__", ""))

    print("hashlib.md5 ...", end=" ")
    hashlib.md5(b"smoke")
    print("OK")

    if "dask" in required_modules:
        print("dask tokenize (MD5) ...", end=" ")
        import pandas
        from dask.base import tokenize

        tokenize(pandas.DataFrame({"a": [1]}))
        print("OK")

    print("ssl.create_default_context() ...", end=" ")
    ssl.create_default_context()
    print("OK")

    ipython = _get_ipython()
    if ipython is None:
        print("dataframe formatter and pip install ... SKIP (not running in a kernel)")
    else:
        print("dataframe_formatter renders a DataFrame ...", end=" ")
        check_dataframe_formatter(ipython)
        print("OK")

        print("pip install from a cell ...", end=" ")
        print("OK, into", check_pip_install())

    print("SMOKE_OK")
