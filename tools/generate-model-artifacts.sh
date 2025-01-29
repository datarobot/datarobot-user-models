#!/usr/bin/env bash

set -e  # Exit immediately on error

script_name=$(basename ${BASH_SOURCE[0]})
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

USAGE="Usage: ${script_name} -t|--ml-type <[sklearn,keras,xgboost]> -r|--requirements-file <FILE> [-h|--help]"
[ $# -eq 0 ] && { echo $USAGE; exit 0; }

while [ "$1" != "" ]; do
    case $1 in
        -t | --ml-type)             shift; ml_type=$1; [ -z "$ml_type" ] && { echo "Missing ml-type!"; exit -1; } ;;
        -r | --requirements-file)   shift; requirements_file=$1; [ -z "$requirements_file" ] && { echo "Missing requirements file!"; exit -1; } ;;
        --help)                     echo $USAGE; exit 0 ;;
    esac
    shift
done

[[ "$ml_type" =~ ^(sklearn|keras|xgboost)$ ]] || { echo "$ml_type is not an accepted value."; echo $USAGE; exit 1; }
[[ ! -f "$requirements_file" ]] && { echo "Requirements file '${requirements_file}' not found!"; exit 1; }

. "${script_dir}/create-and-source-venv.sh"

workspace="$(realpath "${script_dir}/..")"
drop_in_model_artifacts_dir="${workspace}/tests/fixtures/drop_in_model_artifacts"
case $ml_type in
  sklearn)
    notebook_file="${drop_in_model_artifacts_dir}/SKLearn.ipynb"
    ;;
  keras)
    notebook_file="${drop_in_model_artifacts_dir}/Keras.ipynb"
    ;;
  xgboost)
    notebook_file="${drop_in_model_artifacts_dir}/XGBoost.ipynb"
    ;;
esac

pip install ipykernel nbconvert -r "$requirements_file"

pushd "$workspace"

# Execute the notebook and convert the output to markdown
jupyter nbconvert --to markdown --execute "$notebook_file"

# Process changed files and copy them to their destinations
git status --porcelain "$drop_in_model_artifacts_dir" | awk '{print $2}' | while read -r file_path; do
  file_name=$(basename "$file_path")
  find "$workspace" -path "$drop_in_model_artifacts_dir" -prune -o -type f -name "$file_name" -print | while read -r to_file;
  do
    cp "$file_path" "$to_file"
  done
done

echo "Changed files:"
git  status -sb -uall

popd  # $workspace