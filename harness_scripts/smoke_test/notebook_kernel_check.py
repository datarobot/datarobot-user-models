"""Code that notebook_gateway_check.py runs in every kernel of a notebook image.

Each check prints its own line, so the log shows which import or call failed.
Required modules ship in every notebook image; optional ones are checked only
when the image has them, since base images ship fewer packages. hashlib.md5 and
dask tokenize (MD5) catch a base image that blocks non-FIPS digests.

Prints SMOKE_OK at the end when every check passed.
"""

import hashlib
import importlib
import importlib.util
import ssl
import sys

REQUIRED_MODULES = ["numpy", "pandas"]
OPTIONAL_MODULES = ["sklearn", "xgboost", "shap", "datarobot"]

print(f"python {sys.version.split()[0]}, {ssl.OPENSSL_VERSION}")

for name in REQUIRED_MODULES + OPTIONAL_MODULES:
    if name in OPTIONAL_MODULES and importlib.util.find_spec(name) is None:
        print(f"import {name} ... SKIP: not installed")
        continue
    print(f"import {name} ...", end=" ")
    module = importlib.import_module(name)
    print("OK", getattr(module, "__version__", ""))

print("hashlib.md5 ...", end=" ")
hashlib.md5(b"smoke")
print("OK")

if importlib.util.find_spec("dask") is None:
    print("dask tokenize (MD5) ... SKIP: not installed")
else:
    print("dask tokenize (MD5) ...", end=" ")
    import pandas
    from dask.base import tokenize

    tokenize(pandas.DataFrame({"a": [1]}))
    print("OK")

print("ssl.create_default_context() ...", end=" ")
ssl.create_default_context()
print("OK")

print("SMOKE_OK")
