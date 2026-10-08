"""Code that notebook_gateway_check.py runs in every kernel of a notebook image.

Defines run_checks(); notebook_gateway_check.py appends a call to it with the
env's required modules from notebook_required_modules.txt. Each check prints its
own line, so the log shows which import or call failed. hashlib.md5 and dask
tokenize (MD5) catch a base image that blocks non-FIPS digests.

To run it directly with an image's python (the call must start a new,
unindented line):

python3 -c "$(cat notebook_kernel_check.py)
run_checks(['numpy', 'pandas'])"
"""

import hashlib
import importlib
import ssl
import sys


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

    print("SMOKE_OK")
