#!/bin/sh
# Source from the repository root: . harness_scripts/drum_release/detect_version_bump.sh
# POSIX sh so it runs on a plain python alpine image.
# A version counts as bumped when it is not on PyPI yet, so a bump that was merged
# earlier but never released is offered again on the next change to description.py.

set -eu

DRUM_VERSION_PY="harness_scripts/drum_release/drum_version.py"

DRUM_VERSION="$(python3 "${DRUM_VERSION_PY}" current)"
if ! is_published="$(python3 "${DRUM_VERSION_PY}" is-published "${DRUM_VERSION}")"; then
  echo "ERROR: could not query PyPI" >&2
  exit 1
fi

if [ "${is_published}" = "true" ]; then
  VERSION_BUMPED=false
  echo "DRUM ${DRUM_VERSION} is already on PyPI; nothing to release."
else
  VERSION_BUMPED=true
  echo "DRUM ${DRUM_VERSION} is not on PyPI yet; release required."
fi

export VERSION_BUMPED DRUM_VERSION
