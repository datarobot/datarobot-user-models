"""
The FIPS crypto guard (fips/sitecustomize.py + fips/fips_guard.py) is copied into every drop-in
environment that ships it, because each environment is its own Docker build context. The copies must
stay byte-identical: a fix made in one copy and not the others leaves those environments on the old
behavior. This test fails when any copy differs, and when an environment ships the guard without
putting it first on PYTHONPATH, where it would never load.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_FILES = ("sitecustomize.py", "fips_guard.py")
GUARD_DIR = "/etc/datarobot-fips"
FIPS_DIRS = sorted(REPO_ROOT.glob("public_dropin_*/*/fips"))


def _env_name(fips_dir):
    return str(fips_dir.parent.relative_to(REPO_ROOT))


def test_guard_is_shipped_somewhere():
    assert len(FIPS_DIRS) >= 2, f"expected several environments with fips/, found {FIPS_DIRS}"


@pytest.mark.parametrize("file_name", GUARD_FILES)
def test_guard_copies_are_identical(file_name):
    contents = {}
    for fips_dir in FIPS_DIRS:
        path = fips_dir / file_name
        assert path.is_file(), f"{_env_name(fips_dir)} ships fips/ without {file_name}"
        contents.setdefault(path.read_bytes(), []).append(_env_name(fips_dir))
    assert len(contents) == 1, (
        f"fips/{file_name} differs between environments; every copy must be identical. "
        f"Groups of identical copies: {sorted(contents.values())}"
    )


@pytest.mark.parametrize("fips_dir", FIPS_DIRS, ids=_env_name)
def test_guard_is_first_on_pythonpath(fips_dir):
    dockerfile = (fips_dir.parent / "Dockerfile").read_text()
    paths = re.findall(r'PYTHONPATH="?([^"\s\\]+)', dockerfile)
    assert paths, f"{_env_name(fips_dir)}/Dockerfile sets no PYTHONPATH, so fips/ never loads"
    first = paths[-1].split(":")[0]
    assert first == GUARD_DIR, (
        f"{_env_name(fips_dir)}/Dockerfile: the last PYTHONPATH starts with {first!r}, "
        f"expected {GUARD_DIR!r} so sitecustomize.py loads first"
    )
