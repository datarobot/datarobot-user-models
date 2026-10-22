import importlib.util
from pathlib import Path

import pytest

DRUM_VERSION_SCRIPT = (
    Path(__file__).parents[3] / "harness_scripts" / "drum_release" / "drum_version.py"
)
_spec = importlib.util.spec_from_file_location("drum_version", DRUM_VERSION_SCRIPT)
drum_version = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(drum_version)


@pytest.mark.parametrize(
    "published, expected",
    [
        ([], "1.18.1.dev0"),
        (["1.18.0", "1.18.1"], "1.18.1.dev0"),
        (["1.18.1.dev0", "1.18.1.dev1"], "1.18.1.dev2"),
        (["1.18.1.dev0", "1.18.1.dev4"], "1.18.1.dev5"),
    ],
)
def test_next_dev_version_is_one_above_highest_published_dev(published, expected):
    assert drum_version.next_dev_version("1.18.1", published) == expected


def test_next_dev_version_ignores_devs_of_other_versions():
    published = ["1.18.10.dev7", "1.18.0.dev3", "11.18.1.dev9", "1.18.1.dev1"]
    assert drum_version.next_dev_version("1.18.1", published) == "1.18.1.dev2"


@pytest.mark.parametrize(
    "minimum_dev_number, expected", [(0, "1.18.1.dev2"), (2, "1.18.1.dev2"), (5, "1.18.1.dev5")]
)
def test_next_dev_version_never_goes_below_minimum(minimum_dev_number, expected):
    published = ["1.18.1.dev0", "1.18.1.dev1"]
    assert drum_version.next_dev_version("1.18.1", published, minimum_dev_number) == expected


def test_parse_version_reads_version_line():
    source = '"""Docstring."""\nversion = "1.18.1"\n__version__ = version\n'
    assert drum_version.parse_version(source) == "1.18.1"


def test_parse_version_raises_without_version_line():
    with pytest.raises(ValueError):
        drum_version.parse_version('__version__ = "1.18.1"\n')
