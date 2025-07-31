#!/usr/bin/env bash
# A script to bump environmentVersionId (and all occurrences of its current
# value) in env_info.json files.
# Note to keep this process as flexible as possible, almost the entire process
# is executed from within this script to reduce reliance on Harness. With this,
# bumps can easily be executed from dev systems or other CI solutions.
#
# NOTE that this script requires that the GH_TOKEN variable be set for a PR to
# be automatically created.

set -euo pipefail
export IFS=$'\n'

gen_objid() {
  printf '%08lx%06lx%04lx%06lx\n' "$(date +%s)" "$((RANDOM))" "$((RANDOM))" "$((RANDOM))"
}

bump_env() {
  local -a env="${1}"
  printf 'Bumping environment version id for %s\n' "${env}" >&2
  if [ ! -f "${env}/env_info.json" ]; then
    printf 'ERROR: Could not find %s/env_info.json\n' "${env}" >&2
    return 1
  fi
  verid="$(jq -r .environmentVersionId < ${env}/env_info.json)"
  newver="$(gen_objid)"
  sed -i "s/${verid}/${newver}/g" "${env}/env_info.json"
}

github.pr() {
  # Create a branch
  local branch=$(mktemp -u svc/bump-envinfo.XXXXX)
  git checkout -b "${branch}"
  git commit -m "[-] (Auto) Bump env_info versions"
  git push origin "${branch}"
  gh pr create --fill -l "Ready for Review"
}

main() {
  local -a envs=("${@}")

  # Allow user to override environments via stdin
  if [ ${#envs[@]} -eq 0 ]; then
    printf 'At least one environment directory is required\n' >&2
    return 1
  fi
  for env in ${envs[@]}; do
    bump_env "${env}" || continue
    git add "${env}/env_info.json"
  done
  github.pr
}

main ${@}
