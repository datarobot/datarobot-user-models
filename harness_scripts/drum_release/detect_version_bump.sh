#!/bin/sh
# Source from the repository root: . harness_scripts/drum_release/detect_version_bump.sh
# Sets VERSION_BUMPED (true|false) and DRUM_VERSION for Harness output variables.
# POSIX sh so it runs on a plain python alpine image.
# A version counts as bumped when it is not on PyPI yet, so a bump that was merged
# earlier but never released is offered again on the next change to description.py.

set -eu

DRUM_VERSION_PY="harness_scripts/drum_release/drum_version.py"

DRUM_VERSION="$(python3 "${DRUM_VERSION_PY}" current)"
VERSION_BUMPED=false

publish_check=0
python3 "${DRUM_VERSION_PY}" is-published "${DRUM_VERSION}" || publish_check=$?

case "${publish_check}" in
  0)
    echo "DRUM ${DRUM_VERSION} is already on PyPI; nothing to release."
    ;;
  1)
    VERSION_BUMPED=true
    echo "DRUM ${DRUM_VERSION} is not on PyPI yet; release required."
    ;;
  *)
    echo "ERROR: could not query PyPI" >&2
    exit 1
    ;;
esac

export VERSION_BUMPED DRUM_VERSION
