#!/usr/bin/env bash
# Verify that the Harness entity YAMLs committed under .harness/ are actually
# registered in Harness.
#
# WHY THIS EXISTS
# ---------------
# Committing a trigger or input-set YAML under .harness/ does NOT create the
# entity in Harness. Pipelines and input sets are "remote" entities that must be
# imported from git once, by hand, and triggers are stored inline in Harness and
# are never read from git at all. A PR can therefore add a complete, correct set
# of Harness YAML files, be reviewed, merge -- and change nothing about what
# actually runs. That happened to public_dropin_environments/
# dr_mcp_execute_sandbox_minimal: its trigger and input-set files landed in
# May 2026 (#2137) and were never registered, so no webhook has ever fired for
# that environment and its image has only ever been published by hand.
#
# This script makes that failure mode visible instead of silent. It is strictly
# read-only -- it only issues GETs against the Harness API.
#
# USAGE
#   export HARNESS_PAT=<a Harness personal or service account token>
#   .harness/scripts/check_harness_entities.sh
#
# Exit status:
#   0  every committed trigger / input set is registered in Harness
#   1  at least one committed entity is missing from Harness
#
# Triggers that exist but are disabled are reported as warnings, not failures:
# the per-environment env_image_publish trigger family was deliberately disabled
# in April 2026, so "disabled" is often the intended state. See .harness/README.md.

set -euo pipefail

HARNESS_ACCOUNT_ID="${HARNESS_ACCOUNT_ID:-oP3BKzKwSDe_4hCFYw_UWA}"
HARNESS_ORG="${HARNESS_ORG:-Custom_Models}"
HARNESS_PROJECT="${HARNESS_PROJECT:-datarobotusermodels}"
HARNESS_API="${HARNESS_API:-https://app.harness.io/pipeline/api}"

: "${HARNESS_PAT:?HARNESS_PAT must be set to a Harness API token}"

for tool in curl jq; do
  command -v "${tool}" >/dev/null 2>&1 || {
    printf 'ERROR: %s is required but not installed\n' "${tool}" >&2
    exit 2
  }
done

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

common_query="accountIdentifier=${HARNESS_ACCOUNT_ID}&orgIdentifier=${HARNESS_ORG}&projectIdentifier=${HARNESS_PROJECT}"
cache_dir="$(mktemp -d)"
trap 'rm -rf "${cache_dir}"' EXIT

failures=0
warnings=0

# Print the value of a top-level-ish YAML scalar. Deliberately regex based so
# this script has no dependency on a YAML parser being installed.
yaml_scalar() {
  local file="${1}" key="${2}"
  sed -n "s/^[[:space:]]*${key}:[[:space:]]*\(.*\)$/\1/p" "${file}" | head -n1 | tr -d "\"'"
}

# The pipeline an input set belongs to is nested under "pipeline:", so the first
# bare "identifier:" in the file is the input set's own id, not the pipeline's.
input_set_pipeline() {
  local file="${1}"
  awk '/^[[:space:]]*pipeline:[[:space:]]*$/{found=1; next} found && /identifier:/{sub(/.*identifier:[[:space:]]*/, ""); gsub(/"/, ""); print; exit}' "${file}"
}

# Fetch (once per pipeline) the live list of an entity kind and cache it.
fetch_live() {
  local kind="${1}" pipeline="${2}" cache url
  cache="${cache_dir}/${kind}.${pipeline}.json"
  if [ ! -f "${cache}" ]; then
    case "${kind}" in
      triggers) url="${HARNESS_API}/triggers?${common_query}&targetIdentifier=${pipeline}&size=200" ;;
      inputSets) url="${HARNESS_API}/inputSets?${common_query}&pipelineIdentifier=${pipeline}&size=200" ;;
      *)
        printf 'ERROR: unknown entity kind %s\n' "${kind}" >&2
        exit 2
        ;;
    esac
    curl -sS -H "x-api-key: ${HARNESS_PAT}" "${url}" >"${cache}"
    if [ "$(jq -r '.status' <"${cache}")" != "SUCCESS" ]; then
      printf 'ERROR: Harness API call failed for %s of %s: %s\n' \
        "${kind}" "${pipeline}" "$(jq -r '.message // "unknown error"' <"${cache}")" >&2
      exit 2
    fi
  fi
  printf '%s' "${cache}"
}

check_trigger() {
  local file="${1}" identifier pipeline cache enabled
  identifier="$(yaml_scalar "${file}" identifier)"
  pipeline="$(yaml_scalar "${file}" pipelineIdentifier)"
  if [ -z "${identifier}" ] || [ -z "${pipeline}" ]; then
    printf 'WARN     %s: could not determine identifier/pipelineIdentifier, skipping\n' "${file}"
    warnings=$((warnings + 1))
    return
  fi
  cache="$(fetch_live triggers "${pipeline}")"
  enabled="$(jq -r --arg id "${identifier}" \
    '.data.content[]? | select(.identifier == $id) | .enabled' <"${cache}")"
  if [ -z "${enabled}" ]; then
    printf 'MISSING  trigger %s (%s) is committed but does not exist in Harness\n' \
      "${identifier}" "${file}"
    failures=$((failures + 1))
  elif [ "${enabled}" != "true" ]; then
    printf 'DISABLED trigger %s exists in Harness but is disabled\n' "${identifier}"
    warnings=$((warnings + 1))
  else
    printf 'OK       trigger %s is registered and enabled\n' "${identifier}"
  fi
}

check_input_set() {
  local file="${1}" identifier pipeline cache found
  identifier="$(yaml_scalar "${file}" identifier)"
  pipeline="$(input_set_pipeline "${file}")"
  if [ -z "${identifier}" ] || [ -z "${pipeline}" ]; then
    printf 'WARN     %s: could not determine identifier/pipeline, skipping\n' "${file}"
    warnings=$((warnings + 1))
    return
  fi
  cache="$(fetch_live inputSets "${pipeline}")"
  found="$(jq -r --arg id "${identifier}" \
    '.data.content[]? | select(.identifier == $id) | .identifier' <"${cache}")"
  if [ -z "${found}" ]; then
    printf 'MISSING  input set %s (%s) is committed but does not exist in Harness\n' \
      "${identifier}" "${file}"
    failures=$((failures + 1))
  else
    printf 'OK       input set %s is registered\n' "${identifier}"
  fi
}

main() {
  local file kind
  while IFS= read -r file; do
    kind="$(awk '/^[a-zA-Z]/{sub(/:.*$/, ""); print; exit}' "${file}")"
    case "${kind}" in
      trigger) check_trigger "${file}" ;;
      inputSet) check_input_set "${file}" ;;
      *) ;; # pipelines and everything else are out of scope
    esac
  done < <(find .harness -name '*.yaml' -type f | sort)

  printf '\n%d missing, %d warning(s)\n' "${failures}" "${warnings}"
  if [ "${failures}" -gt 0 ]; then
    printf 'Committed Harness YAML is not registered in Harness. See .harness/README.md.\n' >&2
    return 1
  fi
  return 0
}

main
