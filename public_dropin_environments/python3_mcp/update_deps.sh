#!/usr/bin/env bash
# Copyright 2026 DataRobot, Inc. and its affiliates.
#
# All rights reserved.
# This is proprietary source code of DataRobot, Inc. and its affiliates.
#
# Released under the terms of DataRobot Tool and Utility Agreement.

# =============================================================================
# Refresh this environment's dependency lock and version id.
#
# datarobot-genai releases several times a week; the range pin in pyproject.toml
# means a patch-level (z) bump needs no edits here -- running this script is the
# entire refresh:
#   1. uv lock --upgrade   -- newest versions the range + cve-sync constraints allow
#   2. requirements.txt    -- regenerated (the Execution Environment UI package list)
#   3. environmentVersionId bump -- the version id is the image tag; an unbumped
#                                   id is never rebuilt or installed
#
# A minor (y) bump in datarobot-genai is a BREAKING change: it needs a deliberate
# range edit in pyproject.toml here AND in af-component-datarobot-mcp's template,
# moved in lockstep. See README.md "Keeping dependencies fresh".
#
# Requires uv >= 0.10.0 (asserted by required-version in pyproject.toml).
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

uv lock --upgrade

# Top-level pins only, matching the sibling genai-agents envs; the UI package
# list points readers at uv.lock for the complete picture. The sed strips the
# `; sys_platform == ...` markers uv emits for the [tool.uv] environments list.
uv export --no-hashes --no-annotate --no-header --no-emit-project \
  | grep -E '^(datarobot|datarobot-genai)==' | sed 's/ *;.*//' > requirements.txt

# Bump environmentVersionId (pseudo ObjectId: unix-time + random, the same scheme
# as ../../.harness/scripts/bump_env_info.sh) and keep the tags[] copies in sync.
old_id="$(python3 -c "import json; print(json.load(open('env_info.json'))['environmentVersionId'])")"
new_id="$(python3 -c "import secrets, time; print(format(int(time.time()), '08x') + secrets.token_hex(8))")"
sed -i.bak "s/${old_id}/${new_id}/g" env_info.json && rm -f env_info.json.bak
echo "environmentVersionId: ${old_id} -> ${new_id}"
