#!/usr/bin/env bash
# Bump the pinned pulumi-datarobot version and open a PR for a human to review.
#
# Driven by the provider's release, not by polling: pulumi-datarobot's
# release.yml dispatches a `pulumi-datarobot-released` event once its SDK is
# published on PyPI, and .github/workflows/bump-pulumi-datarobot.yaml runs this
# with the released version. Called without a version (a manual
# workflow_dispatch, or a local run) it resolves the newest release itself.
#
# The pin lives in one place:
#
#   public_dropin_notebook_environments/python313_notebook/requirements.txt
#
# The version handed over by the dispatch is verified, not trusted: it must be
# installable from PyPI and cut as a non-draft, non-prerelease GitHub release
# before anything is edited. The dispatch fires as soon as the provider's SDK
# publish step finishes and PyPI's index lags its own publish API, so that
# check retries for a short while. A pin bumped to a version pip cannot install
# would break every image build.
#
# This script never merges anything -- it only opens a PR. It exits 0 (nothing
# to do) when the pin is already current, when an open PR already proposes that
# version, when the branch exists but its PR was closed, and when the release
# artifacts aren't published yet. It exits 1 only when the pin isn't the shape
# this script knows how to edit, when the edit didn't take, or when the version
# argument isn't a bare X.Y.Z -- cases a human needs to look at rather than
# something this script should guess its way through.
#
# Usage:
#   scripts/bump_pulumi_datarobot.sh --check [<version>]
#   scripts/bump_pulumi_datarobot.sh --open-pr <current> <target>
#
# Requires: git, gh (authenticated, repo + PR write), jq, curl.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

PIN_FILE="public_dropin_notebook_environments/python313_notebook/requirements.txt"
PROVIDER_REPO="datarobot-community/pulumi-datarobot"
PYPI_PACKAGE="pulumi-datarobot"

# Everything committed by --open-pr. More than the pin file where bumping the
# pin obliges us to touch something else as well.
STAGE_PATHS=("$PIN_FILE"
  "public_dropin_notebook_environments/python313_notebook/env_info.json")

# How long to keep re-checking that a dispatched version is really installable.
# Six attempts at this spacing covers PyPI's index lag without stalling the job.
VERIFY_RETRY_DELAY="${VERIFY_RETRY_DELAY:-15}"

BARE_RE='[0-9]+\.[0-9]+\.[0-9]+'
PIN_LINE_RE='^pulumi-datarobot=='"${BARE_RE}"'$'

# Emit a key=value pair to the step outputs when running under Actions, and to
# stdout always, so a local run shows the same decision the workflow saw.
emit() {
  local key="$1" value="$2"
  echo "${key}=${value}"
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    echo "${key}=${value}" >>"$GITHUB_OUTPUT"
  fi
}

# The currently pinned version, aborting loudly if the pin isn't exactly one
# line of the expected shape -- that means the file changed in a way this
# script doesn't understand, so it refuses to guess.
read_pin() {
  local count versions distinct

  if [ ! -f "$PIN_FILE" ]; then
    echo "ERROR: ${PIN_FILE} does not exist. Has it moved? Update this script." >&2
    exit 1
  fi

  count="$(grep -cE "$PIN_LINE_RE" "$PIN_FILE" || true)"
  if [ "$count" -ne 1 ]; then
    echo "ERROR: expected exactly 1 'pulumi-datarobot==X.Y.Z' line in ${PIN_FILE}, found ${count}. Update this script before it can safely bump it." >&2
    exit 1
  fi

  versions="$(grep -oE "$PIN_LINE_RE" "$PIN_FILE" | grep -oE "$BARE_RE" | sort -u)"
  distinct="$(printf '%s\n' "$versions" | grep -c . || true)"
  if [ "$distinct" -ne 1 ]; then
    echo "ERROR: could not determine a single pinned version in ${PIN_FILE}." >&2
    exit 1
  fi

  printf '%s' "$versions"
}

# True when $1 is strictly newer than $2. Guards against "bumping" backwards if
# the pin is ahead of the latest published release for some reason.
version_gt() {
  [ "$1" != "$2" ] && [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -n1)" = "$1" ]
}

# True when $1 is installable from PyPI (a non-yanked release with files) and
# cut as a non-draft, non-prerelease GitHub release. Retried, because the
# release dispatch can reach us before PyPI's index reflects its own publish.
version_available() {
  local version="$1" attempt pypi_json

  for attempt in 1 2 3 4 5 6; do
    if [ "$attempt" -gt 1 ]; then
      echo "  not visible yet, retrying in ${VERIFY_RETRY_DELAY}s (attempt ${attempt}/6)..." >&2
      sleep "$VERIFY_RETRY_DELAY"
    fi

    if ! pypi_json="$(curl -fsS --retry 3 --retry-delay 2 --max-time 30 \
        "https://pypi.org/pypi/${PYPI_PACKAGE}/${version}/json" 2>/dev/null)"; then
      continue
    fi

    # A version whose files are all yanked is published but not installable.
    if ! jq -e '.urls | length > 0 and any(.[]; .yanked == false)' >/dev/null 2>&1 <<<"$pypi_json"; then
      continue
    fi

    if ! gh release view "v${version}" --repo "$PROVIDER_REPO" \
        --json isDraft,isPrerelease \
        --jq 'select(.isDraft == false and .isPrerelease == false)' >/dev/null 2>&1; then
      continue
    fi

    return 0
  done

  return 1
}

# Newest version published on PyPI as a non-yanked release with files, and also
# cut as a non-draft, non-prerelease GitHub release. Used only when no version
# was handed to us. Returns non-zero if either source can't be read.
resolve_target_version() {
  local pypi_json pypi_versions gh_tags candidate

  if ! pypi_json="$(curl -fsS --retry 3 --retry-delay 2 --max-time 30 \
      "https://pypi.org/pypi/${PYPI_PACKAGE}/json")"; then
    return 1
  fi

  if ! pypi_versions="$(jq -r '
        .releases
        | to_entries[]
        | select(.value | length > 0)
        | select(any(.value[]; .yanked == false))
        | .key
      ' <<<"$pypi_json")"; then
    return 1
  fi

  if ! gh_tags="$(gh release list --repo "$PROVIDER_REPO" \
      --exclude-pre-releases --exclude-drafts --limit 50 \
      --json tagName --jq '.[].tagName')"; then
    return 1
  fi

  while read -r tag; do
    [ -n "$tag" ] || continue
    candidate="${tag#v}"
    if grep -qxF "$candidate" <<<"$pypi_versions"; then
      printf '%s' "$candidate"
      return 0
    fi
  done < <(printf '%s\n' "$gh_tags" | sort -Vr)

  return 1
}

do_check() {
  local requested="${1:-}"
  local current target

  current="$(read_pin)"
  echo "Currently pinned pulumi-datarobot: ${current}"

  if [ -n "$requested" ]; then
    if ! [[ "$requested" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
      echo "ERROR: requested version '${requested}' is not a bare X.Y.Z version." >&2
      exit 1
    fi

    echo "Requested version (from the release dispatch): ${requested}"
    echo "Confirming it is installable from PyPI and cut as a GitHub release..."
    if ! version_available "$requested"; then
      echo "WARNING: pulumi-datarobot ${requested} is not yet available on both PyPI and GitHub. Not opening a PR." >&2
      echo "Dispatch it again, or re-run this workflow once it is published." >&2
      emit should_bump false
      return 0
    fi
    echo "Confirmed ${requested} on both PyPI and GitHub."
    target="$requested"
  else
    # Soft failure: a transient PyPI/GitHub outage or an expired token looks the
    # same as "no releases found", so warn rather than failing the job.
    if ! target="$(resolve_target_version)" || [ -z "$target" ]; then
      echo "WARNING: could not determine a pulumi-datarobot release published on both PyPI and GitHub. Nothing to do." >&2
      emit should_bump false
      return 0
    fi
    echo "Latest release on both PyPI and GitHub: ${target}"
  fi

  if [ "$target" = "$current" ]; then
    echo "pulumi-datarobot is already pinned at ${current}. Nothing to do."
    emit should_bump false
    return 0
  fi

  if ! version_gt "$target" "$current"; then
    echo "The pinned version (${current}) is newer than ${target}. Leaving it alone."
    emit should_bump false
    return 0
  fi

  local branch="auto/bump-pulumi-datarobot-${current}-to-${target}"
  local pr_title="chore: bump pulumi-datarobot from ${current} to ${target}"

  # De-dup on branch names, not on a title/body search. `gh pr list --search` is
  # a fuzzy full-text query: searching for the version strings also matches any
  # unrelated open PR that merely mentions them, which would silently suppress
  # real bumps.
  #
  # Two branch shapes count as "already proposed": this script's own, and
  # Dependabot's (dependabot/pip/.../pulumi-datarobot-<version>), so the daily
  # Dependabot backstop and this release-triggered path never both open a PR
  # for the same version. Compared literally with endswith rather than a regex,
  # because the version contains dots.
  local existing
  existing="$(gh pr list --state open --json number,headRefName 2>/dev/null \
    | jq -r --arg branch "$branch" --arg suffix "pulumi-datarobot-${target}" \
        '[.[] | select(.headRefName == $branch or (.headRefName | endswith($suffix)))] | .[0].number // empty' \
    2>/dev/null || true)"
  if [ -n "$existing" ]; then
    echo "An open PR already proposes pulumi-datarobot ${target} (#${existing}). Nothing to do."
    emit should_bump false
    return 0
  fi

  # No open PR, but the branch still exists on the remote: someone closed that
  # PR deliberately. Recreating it would reopen a decision a human already made,
  # and force-pushing over their branch is worse. Skip loudly instead.
  if git ls-remote --exit-code --heads origin "$branch" >/dev/null 2>&1; then
    echo "Branch ${branch} already exists on origin but has no open PR -- its PR was probably closed on purpose. Skipping." >&2
    echo "Delete the remote branch if this bump should be proposed again." >&2
    emit should_bump false
    return 0
  fi

  apply_bump "$current" "$target"

  emit should_bump true
  emit current "$current"
  emit target "$target"
  emit branch "$branch"
  emit pr_title "$pr_title"
}

# Edit precisely: match the exact pin line rather than doing a blind
# find-and-replace that could touch an unrelated occurrence of the same version.
apply_bump() {
  local current="$1" target="$2" escaped
  escaped="$(printf '%s' "$current" | sed -E 's/\./\\./g')"

  sed -E "s/^pulumi-datarobot==${escaped}\$/pulumi-datarobot==${target}/" \
    "$PIN_FILE" >"${PIN_FILE}.bump_tmp"
  mv "${PIN_FILE}.bump_tmp" "$PIN_FILE"

  # A partial bump (the sed pattern silently failing to match) is worse than no
  # bump at all, so confirm the new pin is in place before anything is staged.
  if ! grep -qxF "pulumi-datarobot==${target}" "$PIN_FILE"; then
    echo "ERROR: ${PIN_FILE} does not contain pulumi-datarobot==${target} after editing." >&2
    echo "Reverting and aborting." >&2
    git checkout -- "$PIN_FILE"
    exit 1
  fi

  echo "Applied bump ${current} -> ${target} to ${PIN_FILE}."
}

do_open_pr() {
  local current="$1" target="$2"
  local branch="auto/bump-pulumi-datarobot-${current}-to-${target}"
  local pr_title="chore: bump pulumi-datarobot from ${current} to ${target}"

  if git diff --quiet -- "${STAGE_PATHS[@]}"; then
    echo "ERROR: no pending changes in ${STAGE_PATHS[*]}. Run --check first." >&2
    exit 1
  fi

  git checkout -b "$branch"
  git add -- "${STAGE_PATHS[@]}"
  git commit -m "$pr_title"
  git push -u origin "$branch"

  local body_file
  body_file="$(mktemp)"
  # shellcheck disable=SC2064  # expand body_file now, not at trap time
  trap "rm -f '$body_file'" EXIT

  cat >"$body_file" <<BODY_EOF
Automated bump of the pinned [pulumi-datarobot](https://github.com/${PROVIDER_REPO}) provider (\`${current}\` -> \`${target}\`), opened in response to that provider publishing a release.

## Files changed

- \`${PIN_FILE}\` -- the pulumi-datarobot pin for the Python 3.13 notebook environment
- \`public_dropin_notebook_environments/python313_notebook/env_info.json\` -- a fresh \`environmentVersionId\`, so the platform treats this as a new environment version

The \`environmentVersionId\` was regenerated with \`tools/env_version_update.py\`. Harness has \`update_env_version\` input sets for the \`public_dropin_environments\` envs but none for the notebook environments, so it is not done for us here.

## Verification

\`${target}\` was confirmed installable from PyPI and cut as a non-draft, non-prerelease GitHub release before this PR was opened. Beyond that, this PR is **not** build-verified -- validation is whatever this repo's PR pipelines do with it.

Worth a human eye on the provider's release notes: \`pulumi-datarobot\` is a \`0.x\` provider, so a minor bump can carry breaking resource changes even though this looks routine.

This automation never auto-merges.
BODY_EOF

  gh pr create --base "$(gh repo view --json defaultBranchRef --jq '.defaultBranchRef.name')" \
    --title "$pr_title" --body-file "$body_file"
}

case "${1:-}" in
  --check)
    if [ "$#" -gt 2 ]; then
      echo "Usage: $0 --check [<version>]" >&2
      exit 1
    fi
    do_check "${2:-}"
    ;;
  --open-pr)
    if [ "$#" -ne 3 ]; then
      echo "Usage: $0 --open-pr <current> <target>" >&2
      exit 1
    fi
    do_open_pr "$2" "$3"
    ;;
  *)
    echo "Usage: $0 --check [<version>] | --open-pr <current> <target>" >&2
    exit 1
    ;;
esac
