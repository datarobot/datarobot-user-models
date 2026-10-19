#!/usr/bin/env python3
"""Read, publish-check and rewrite the DRUM version used by the release pipelines."""

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PACKAGE = "datarobot-drum"
DESCRIPTION_FILE = Path("custom_model_runner/datarobot_drum/drum/description.py")
VERSION_PATTERN = re.compile(r'^version = "(?P<version>[^"]+)"$', re.MULTILINE)
SIMPLE_API_URL = f"https://pypi.org/simple/{PACKAGE}/"
SIMPLE_API_JSON = "application/vnd.pypi.simple.v1+json"


def parse_version(description_source):
    match = VERSION_PATTERN.search(description_source)
    if not match:
        raise ValueError(f"No version found in {DESCRIPTION_FILE}")
    return match["version"]


def published_versions():
    request = urllib.request.Request(SIMPLE_API_URL, headers={"Accept": SIMPLE_API_JSON})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)["versions"]


def next_dev_version(base_version, versions, minimum_dev_number=0):
    dev_pattern = re.compile(rf"{re.escape(base_version)}\.dev(\d+)")
    dev_numbers = [int(m[1]) for v in versions if (m := dev_pattern.fullmatch(v))]
    next_dev_number = max(dev_numbers, default=-1) + 1
    return f"{base_version}.dev{max(next_dev_number, minimum_dev_number)}"


def set_version(new_version):
    source = DESCRIPTION_FILE.read_text()
    DESCRIPTION_FILE.write_text(VERSION_PATTERN.sub(f'version = "{new_version}"', source, count=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("current", help="Print the version in description.py")
    is_published = commands.add_parser(
        "is-published", help="Print true if VERSION is on PyPI, else false"
    )
    is_published.add_argument("version")
    next_dev = commands.add_parser("next-dev", help="Print the next free BASE.devN on PyPI")
    next_dev.add_argument("base_version")
    next_dev.add_argument("--min", type=int, default=0, dest="minimum_dev_number")
    set_cmd = commands.add_parser("set", help="Write VERSION into description.py")
    set_cmd.add_argument("version")
    args = parser.parse_args()

    if args.command == "current":
        print(parse_version(DESCRIPTION_FILE.read_text()))
    elif args.command == "is-published":
        print(str(args.version in published_versions()).lower())
    elif args.command == "next-dev":
        print(next_dev_version(args.base_version, published_versions(), args.minimum_dev_number))
    elif args.command == "set":
        set_version(args.version)


if __name__ == "__main__":
    try:
        main()
    except (urllib.error.URLError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
