"""Code that notebook_gateway_check.py runs in every kernel of a notebook image.

Defines run_checks(); notebook_gateway_check.py appends a call to it with the
env's required modules from notebook_required_modules.txt. Each check prints its
own line, so the log shows which import or call failed. hashlib.md5 and dask
tokenize (MD5) catch a base image that blocks non-FIPS digests. In a kernel it
also checks that the dataframe extension renders a DataFrame and that %pip
installs a package the kernel can import.

To run it directly with an image's python (the call must start a new,
unindented line):

python3 -c "$(cat notebook_kernel_check.py)
run_checks(['numpy', 'pandas'])"
"""

import base64
import builtins
import hashlib
import importlib
import os
import ssl
import sys
import tempfile
import uuid
import zipfile

# Mimetype that dataframe_formatter from the image's ipython_config.py adds to DataFrames.
DATAFRAME_MIMETYPE = "application/vnd.dataframe+json"


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

    ipython = getattr(builtins, "get_ipython", lambda: None)()
    if ipython is None:
        print("not in an IPython kernel, skipping the dataframe extension and %pip checks")
    else:
        print("dataframe extension renders a DataFrame ...", end=" ")
        import pandas

        mimetypes = ipython.display_formatter.format(pandas.DataFrame({"a": [1]}))[0]
        if DATAFRAME_MIMETYPE not in mimetypes:
            raise RuntimeError(f"no {DATAFRAME_MIMETYPE} in {sorted(mimetypes)}")
        print("OK")

        # A new package name per call, so the second kernel can't import what the first installed.
        print("%pip install from a cell ...")
        name = f"nbx_smoke_pkg_{uuid.uuid4().hex[:8]}"
        wheel = _build_wheel(name)
        ipython.run_line_magic(
            "pip", f"install --no-index --no-deps --quiet --disable-pip-version-check {wheel}"
        )
        importlib.invalidate_caches()
        module = importlib.import_module(name)
        print("%pip install from a cell ... OK", module.__file__)

    print("SMOKE_OK")


def _build_wheel(name):
    """Writes a minimal pure-Python wheel, so pip installs it without network access."""
    dist_info = f"{name}-0.0.1.dist-info"
    files = {
        f"{name}/__init__.py": b"",
        f"{dist_info}/METADATA": f"Metadata-Version: 2.1\nName: {name}\nVersion: 0.0.1\n".encode(),
        f"{dist_info}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    record = ""
    for path, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        record += f"{path},sha256={digest},{len(data)}\n"
    files[f"{dist_info}/RECORD"] = f"{record}{dist_info}/RECORD,,\n".encode()

    wheel = os.path.join(tempfile.mkdtemp(), f"{name}-0.0.1-py3-none-any.whl")
    with zipfile.ZipFile(wheel, "w") as archive:
        for path, data in files.items():
            archive.writestr(path, data)
    return wheel
