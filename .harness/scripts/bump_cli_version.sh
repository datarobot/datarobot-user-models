#!/usr/bin/env bash
# Check whether a newer stable datarobot-oss/cli release exists than the one
# currently pinned via `ARG CLI_VERSION=` in the python313 notebook Dockerfile,
# and if so open a PR that bumps the pin to it. Intended to run on a daily
# schedule (see .harness/cli_version_bump/) so routine "pick up the latest
# release" bumps happen without a human kicking them off. CVE-driven bumps,
# which need dependency checks and local Docker builds, stay manual.
#
# This script never merges anything -- it only opens a PR for a human to
# review. It exits 0 (nothing to do) when already up to date, when a PR for
# the exact version transition is already open, or when it can't reach
# GitHub to check (it'll retry on the next scheduled run). It exits 1 only
# when the pin can't be found exactly once, which means the file's shape
# changed and a human needs to look at it rather than have this script guess.
#
# env_info.json is deliberately not touched here: the existing
# update_env_version automation refreshes it, as it does for manual bumps.
#
# Usage: .harness/scripts/bump_cli_version.sh
# Env:   DRY_RUN=1  report what would happen, but create no branch/commit/PR
# Requires: git, gh (authenticated with repo + PR read/write scope), and
# .harness/scripts/list_stable_releases.sh
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

DEFAULT_BRANCH="master"
DOCKERFILE="public_dropin_notebook_environments/python313_notebook/Dockerfile"
SCRIPTS_DIR=".harness/scripts"

VERSION_RE='v[0-9]+\.[0-9]+\.[0-9]+'
# Only the defaulted ARG matches; the bare `ARG CLI_VERSION` re-declaration
# later in the Dockerfile carries no value and must stay untouched.
PIN_LINE_RE='^ARG CLI_VERSION='"${VERSION_RE}"'$'

# Make sure origin/master is current before reading the pin from it or
# branching off it -- never compare against, or build on, a stale local view.
git fetch origin "$DEFAULT_BRANCH" --quiet

DOCKERFILE_CONTENT="$(git show "origin/${DEFAULT_BRANCH}:${DOCKERFILE}")"

# Require exactly one matching pin line; zero or several means the file's
# shape changed in a way this script doesn't understand, so refuse to guess.
PIN_COUNT="$(grep -cE "$PIN_LINE_RE" <<<"$DOCKERFILE_CONTENT" || true)"
if [ "$PIN_COUNT" -ne 1 ]; then
  echo "ERROR: expected exactly 1 line matching '${PIN_LINE_RE}' in ${DOCKERFILE} on origin/${DEFAULT_BRANCH}, found ${PIN_COUNT}. The file's shape has changed -- update this script before it can safely bump it." >&2
  exit 1
fi

CURRENT_VERSION="$(grep -E "$PIN_LINE_RE" <<<"$DOCKERFILE_CONTENT" | grep -oE "$VERSION_RE")"
echo "Currently pinned CLI_VERSION (origin/${DEFAULT_BRANCH}): ${CURRENT_VERSION}"

# list_stable_releases.sh has no retry/fallback of its own -- a transient
# GitHub outage or an expired token makes it exit non-zero exactly the same
# way as "no releases found" would. Treat any failure here as soft: warn
# and let the next scheduled run try again.
if ! LATEST_VERSION="$("${SCRIPTS_DIR}/list_stable_releases.sh" datarobot-oss/cli 1)"; then
  echo "WARNING: could not reach GitHub / determine the latest datarobot-oss/cli release. Will retry on the next scheduled run." >&2
  exit 0
fi

LATEST_VERSION="$(printf '%s' "$LATEST_VERSION" | head -n1 | tr -d '[:space:]')"

if [ -z "$LATEST_VERSION" ]; then
  echo "WARNING: no stable datarobot-oss/cli releases were found. Will retry on the next scheduled run." >&2
  exit 0
fi

# Guard against an unexpected tag shape (e.g. no leading "v") before it is
# interpolated into a sed replacement and a branch name.
if ! [[ "$LATEST_VERSION" =~ ^${VERSION_RE}$ ]]; then
  echo "WARNING: latest release tag '${LATEST_VERSION}' is not in vX.Y.Z form; skipping. A human should check the release." >&2
  exit 0
fi

echo "Latest stable datarobot-oss/cli release: ${LATEST_VERSION}"

if [ "$LATEST_VERSION" = "$CURRENT_VERSION" ]; then
  echo "CLI_VERSION is already up to date (${CURRENT_VERSION}). Nothing to do."
  exit 0
fi

PR_TITLE="[-] Update CLI_VERSION from ${CURRENT_VERSION} to ${LATEST_VERSION}"

# De-dup: don't open a second PR proposing the exact same version transition
# if one is already open (e.g. yesterday's run is still awaiting review).
EXISTING_PR_NUMBER="$(gh pr list --state open --search "\"CLI_VERSION from ${CURRENT_VERSION} to ${LATEST_VERSION}\" in:title" --json number --jq '.[0].number // empty' 2>/dev/null || true)"
if [ -n "$EXISTING_PR_NUMBER" ]; then
  echo "A PR already proposes this exact bump (#${EXISTING_PR_NUMBER}: ${PR_TITLE}). Nothing to do."
  exit 0
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "DRY_RUN=1: would open PR '${PR_TITLE}' on branch auto/bump-cli-${LATEST_VERSION#v}. Stopping."
  exit 0
fi

BRANCH="auto/bump-cli-${LATEST_VERSION#v}"
git checkout -b "$BRANCH" "origin/${DEFAULT_BRANCH}"

# Regex-escape the current version's dots so it's matched literally, not as
# "any character", inside the sed pattern below.
CURRENT_ESCAPED="$(printf '%s' "$CURRENT_VERSION" | sed -E 's/\./\\./g')"

# Edit precisely: match the exact line shape rather than doing a blind
# find-and-replace that could touch an unrelated occurrence of the version.
sed -E "s/^ARG CLI_VERSION=${CURRENT_ESCAPED}\$/ARG CLI_VERSION=${LATEST_VERSION}/" "$DOCKERFILE" > "${DOCKERFILE}.bump_tmp"
mv "${DOCKERFILE}.bump_tmp" "$DOCKERFILE"

# Confirm the pin really changed before committing anything -- a silently
# non-matching sed would otherwise produce an empty or partial commit.
if ! grep -qxF "ARG CLI_VERSION=${LATEST_VERSION}" "$DOCKERFILE" || grep -qF "$CURRENT_VERSION" "$DOCKERFILE"; then
  echo "ERROR: ${DOCKERFILE} was not updated cleanly to ${LATEST_VERSION}. Reverting and aborting." >&2
  git checkout -- "$DOCKERFILE"
  exit 1
fi

git add "$DOCKERFILE"
git commit -m "$PR_TITLE"
git push -u origin "$BRANCH"

PR_BODY_FILE="$(mktemp)"
trap 'rm -f "$PR_BODY_FILE"' EXIT

cat > "$PR_BODY_FILE" <<EOF
Automated bump of the pinned DataRobot CLI ([datarobot-oss/cli](https://github.com/datarobot-oss/cli)) version, picked up from the latest stable release (\`${CURRENT_VERSION}\` -> \`${LATEST_VERSION}\`). See the [release notes](https://github.com/datarobot-oss/cli/releases/tag/${LATEST_VERSION}).

## Files changed

- \`${DOCKERFILE}\` -- \`ARG CLI_VERSION\`

\`env_info.json\` is not changed here; the existing environment version update automation handles it after merge, as for manual bumps.

## Verification

This PR was **not** build-verified locally. The Dockerfile build fails if the downloaded binary does not report this version, and the existing PR checks build this environment.

This automation never auto-merges -- a human reviews and merges this PR like any other.
EOF

if ! gh pr create --base "$DEFAULT_BRANCH" --title "$PR_TITLE" --body-file "$PR_BODY_FILE"; then
  echo "ERROR: gh pr create failed; the branch ${BRANCH} was pushed, open the PR manually." >&2
  exit 1
fi
