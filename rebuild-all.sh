#!/usr/bin/env bash
set -euo pipefail
export IFS=$'\n'

# Do this becuase Harness doesn't fully clone and doesn't leave behind credentials for us to pull ourselves
# git remote rm origin
# git remote add origin https://svc-harness-git2:<+secrets.getValue("account.githubpatsvcharnessgit2")>@github.com/datarobot/datarobot-custom-templates
# git fetch origin

# Provided by genai-ci-tools
# Use HEAD~1, as this should be triggered from m2m
# source /usr/local/bin/git.changed-files.sh HEAD~1

# Define the output variable for the stage matrix loop
export rebuildenvs='{"environments":'
rebuildenvs+="$(printf '%s\n' ${FILES_A[@]} ${FILES_M[@]} | sed -n 's~execution_environments/\([^/]*\)/.*~"\1"~p' | sort | uniq | jq -c --slurp)"
rebuildenvs+='}'




# Define the output variable for the stage matrix loop
export rebuildenvs='{"environments":'
rebuildenvs+="$(printf '%s\n' ${FILES_A[@]} ${FILES_M[@]} | sed -n 's~\(.*\)/env_info.json$~"\1"~p' | sort | uniq | jq -c --slurp)"
rebuildenvs+='}'

printf '%s\n' "${rebuildenvs}"
