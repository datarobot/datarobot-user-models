#!/usr/bin/env bash
set -euo pipefail
export IFS=$'\n'

infile="${1:?}"
vimdiff "${1}" "/tmp/datarobot-user-models/public_dropin_environments/${1}"
git add "${1}"
