#!/usr/bin/env bash
# List stable (non-pre-release) release tags for a GitHub repo, newest first.
# Used to find real datarobot-oss/cli releases to bump CLI_VERSION to, skipping
# noisy "-dev.N" pre-release tags that shouldn't be shipped in notebook images.
#
# Usage: list_stable_releases.sh <owner/repo> [limit]
# Example: list_stable_releases.sh datarobot-oss/cli 15
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <owner/repo> [limit]" >&2
  exit 1
fi

REPO="$1"
LIMIT="${2:-30}"

gh release list --repo "$REPO" --limit "$LIMIT" --exclude-pre-releases --exclude-drafts | awk -F'\t' '{print $3}'
