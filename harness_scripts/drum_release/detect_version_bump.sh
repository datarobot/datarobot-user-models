#!/usr/bin/env bash
# Usage: detect_version_bump.sh [BEFORE_SHA]
# Sets VERSION_BUMPED (true|false) and DRUM_VERSION for Harness output variables.
# BEFORE_SHA is the push's previous master head; when absent (manual run) only PyPI is checked.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRUM_VERSION_PY="${SCRIPT_DIR}/drum_version.py"
DESCRIPTION_FILE="custom_model_runner/datarobot_drum/drum/description.py"
NULL_SHA="0000000000000000000000000000000000000000"

before_sha="${1:-}"
DRUM_VERSION="$(python3 "${DRUM_VERSION_PY}" current)"
VERSION_BUMPED=false

previous_version=""
if [ -n "${before_sha}" ] && [ "${before_sha}" != "${NULL_SHA}" ]; then
  git fetch --quiet --depth=1 origin "${before_sha}"
  previous_version="$(git show "${before_sha}:${DESCRIPTION_FILE}" | python3 "${DRUM_VERSION_PY}" parse)"
fi

publish_check=0
python3 "${DRUM_VERSION_PY}" is-published "${DRUM_VERSION}" || publish_check=$?

if [ "${publish_check}" -eq 2 ]; then
  echo "ERROR: could not query PyPI" >&2
  exit 1
elif [ "${previous_version}" = "${DRUM_VERSION}" ]; then
  echo "DRUM version unchanged (${DRUM_VERSION}); nothing to release."
elif [ "${publish_check}" -eq 0 ]; then
  echo "DRUM ${DRUM_VERSION} is already on PyPI; nothing to release."
else
  VERSION_BUMPED=true
  echo "DRUM version bumped ${previous_version:-<unknown>} -> ${DRUM_VERSION}; release required."
fi

export VERSION_BUMPED DRUM_VERSION
