#!/bin/bash

# Usage: ./script.sh [root_folder] [filter_folders]
ROOT="${1:-.}"
FILTER_RAW="${2:-}"

# Normalize root (remove trailing slash)
ROOT="${ROOT%/}"

# Convert comma-separated filter to array
IFS=',' read -ra FILTER_DIRS <<< "$FILTER_RAW"

# Base ref to compare against
BASE_REF="master"

echo "📁 Using root folder: '$ROOT'"
if [[ -n "$FILTER_RAW" ]]; then
  echo "🔎 Filtering to folders containing: ${FILTER_DIRS[*]}"
fi
echo ""

echo "🔍 Finding directories that contain 'env_info.json'..."

# 1. Find all directories containing env_info.json, relative to root
mapfile -t ENV_DIRS < <(find "$ROOT" -type f -name "env_info.json" -exec dirname {} \; | sed "s|^\./||;s|^$ROOT/||" | sort -u)

echo "✅ Directories with 'env_info.json':"
for dir in "${ENV_DIRS[@]}"; do
  echo "  $dir"
done

echo ""
echo "🔄 Getting list of changed files compared to '$BASE_REF'..."

# 2. Get the list of changed files, relative to root
mapfile -t CHANGED_FILES < <(git diff --name-only "$BASE_REF" | sed 's|^\./||')

echo "✅ Changed files:"
for file in "${CHANGED_FILES[@]}"; do
  echo "  $file"
done

echo ""
echo "📂 Finding env_info.json directories that have changes under them..."

# 3. Match only correct paths (dir == file or dir/)
MATCHED_DIRS=()
for dir in "${ENV_DIRS[@]}"; do
  for file in "${CHANGED_FILES[@]}"; do
    if [[ "$file" == "$dir" || "$file" == "$dir/"* ]]; then
      MATCHED_DIRS+=("$dir")
      break
    fi
  done
done

# Remove duplicates
UNIQUE_MATCHED_DIRS=($(printf "%s\n" "${MATCHED_DIRS[@]}" | sort -u))

# 4. Optional filtering by folder name
if [[ -n "$FILTER_RAW" ]]; then
  FILTERED_DIRS=()
  for dir in "${UNIQUE_MATCHED_DIRS[@]}"; do
    for filter in "${FILTER_DIRS[@]}"; do
      if [[ "$dir" == *"$filter"* ]]; then
        FILTERED_DIRS+=("$dir")
        break
      fi
    done
  done
  UNIQUE_MATCHED_DIRS=("${FILTERED_DIRS[@]}")
fi

# 5. Save final result as a newline-separated list
changed_environments=$(IFS=$'\n'; echo "${UNIQUE_MATCHED_DIRS[*]}")

echo ""
echo "✅ Final matched directories:"
echo "$changed_environments"

export changed_environments