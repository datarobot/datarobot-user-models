#!/usr/bin/env bash
# Usage: release_drum.sh dev|main
# Builds the DRUM wheel and uploads it to PyPI. Requires PYPI_USERNAME and PYPI_PASSWORD.
# For dev, the version becomes <version>.devN with N the next free number on PyPI.

set -euo pipefail
shopt -s inherit_errexit

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRUM_VERSION_PY="${SCRIPT_DIR}/drum_version.py"
MAX_DEV_UPLOAD_ATTEMPTS=3

release_type="${1:?release type (dev|main) is required}"
: "${PYPI_USERNAME:?}" "${PYPI_PASSWORD:?}"

build_wheel() {
  make -C custom_model_runner >&2
  realpath "$(find custom_model_runner/dist -name 'datarobot_drum*.whl')"
}

upload_wheel() {
  local wheel="${1}"
  local log
  log="$(mktemp)"
  if twine upload --non-interactive -u "${PYPI_USERNAME}" -p "${PYPI_PASSWORD}" "${wheel}" 2>&1 | tee "${log}"; then
    return 0
  fi
  # PyPI never accepts a filename twice, even after the release is deleted, so a
  # computed dev number can still collide; the caller retries with the next one.
  grep -q "File already exists" "${log}" && return 3
  return 1
}

pip3 install --only-binary=:all: twine

case "${release_type}" in
  main)
    version="$(python3 "${DRUM_VERSION_PY}" current)"
    echo "Releasing DRUM ${version}"
    wheel="$(build_wheel)"
    upload_wheel "${wheel}"
    ;;
  dev)
    base_version="$(python3 "${DRUM_VERSION_PY}" current)"
    minimum_dev_number=0
    for attempt in $(seq 1 "${MAX_DEV_UPLOAD_ATTEMPTS}"); do
      version="$(python3 "${DRUM_VERSION_PY}" next-dev "${base_version}" --min "${minimum_dev_number}")"
      python3 "${DRUM_VERSION_PY}" set "${version}"
      echo "Releasing DRUM ${version} (attempt ${attempt}/${MAX_DEV_UPLOAD_ATTEMPTS})"
      wheel="$(build_wheel)"
      upload_status=0
      upload_wheel "${wheel}" || upload_status=$?
      if [ "${upload_status}" -eq 0 ]; then
        break
      elif [ "${upload_status}" -ne 3 ] || [ "${attempt}" -eq "${MAX_DEV_UPLOAD_ATTEMPTS}" ]; then
        exit 1
      fi
      minimum_dev_number=$(( ${version##*.dev} + 1 ))
    done
    ;;
  *)
    echo "ERROR: unknown release type '${release_type}', expected dev or main" >&2
    exit 1
    ;;
esac

echo "Released DRUM ${version}: https://pypi.org/project/datarobot-drum/${version}/"
